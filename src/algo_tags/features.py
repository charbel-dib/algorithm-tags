"""Feature extraction: six blocks turned into one sparse matrix.

| block        | reads                                   | how                          | weight |
|--------------|-----------------------------------------|------------------------------|--------|
| ``words``    | statement                               | TF-IDF, words 1-2            | 1.0    |
| ``chars``    | statement                               | TF-IDF, characters 3-5       | 0.5    |
| ``code``     | Python solution, boilerplate removed    | TF-IDF, words 1-2            | 0.5    |
| ``ast``      | syntax tree of the solution             | TF-IDF on structural tokens  | 0.3    |
| ``difficulty`` | Codeforces rating                     | 8 values, standardised       | 0.25   |
| ``samples``  | sample tests and output specification   | 11 clues, standardised       | 0.25   |

Every block is fitted on the training problems only (vocabulary, IDF, mean and variance), so
nothing from the evaluated problems leaks into the features. TF-IDF rows are L2-normalised; the
weight then sets how much a block counts against the others.
"""

import ast
import re
import warnings
from functools import lru_cache

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

WORDS = dict(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
FLOAT, INT = re.compile(r"\d+\.\d+"), re.compile(r"-?\d+")
IO_LINE = re.compile(r"^\s*(import sys|input\s*=|from sys import|#)")
MODULI = {998244353, 1000000007, 1000000009}


def clean_code(code: str) -> str:
    """Drop fast-I/O boilerplate and comments, which say nothing about the algorithm."""
    return "\n".join(line for line in code.split("\n") if not IO_LINE.match(line))


def _parse(code: str):
    """Parse Python 3 code, or Python 2 code after a light rewrite; None if both fail."""
    py2 = re.sub(r"^(\s*)print\s+(?!\()(.*)$", r"\1print(\2)", code, flags=re.M)
    py2 = py2.replace("raw_input", "input").replace("xrange", "range")
    for source in (code, py2):  # Python 3 first, then a light Python 2 rewrite
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return ast.parse(source)
        except (SyntaxError, ValueError):
            pass
    return None


@lru_cache(maxsize=None)
def ast_tokens(code: str) -> str:
    """Structural tokens of a solution, independent of variable names.

    Example: ``from math import gcd; print(gcd(a, b) % 998244353)`` gives tokens such as
    ``I_math I_gcd C_gcd OP_Mod K_MOD``. Unparsable code (1.3 % of the corpus) gives ``PARSE_ERROR``.
    Results are cached because the same solution is parsed in every cross-validation fold.
    """
    tree = _parse(code)
    if tree is None:
        return "PARSE_ERROR"
    tokens = []
    for node in ast.walk(tree):
        tokens.append("N_" + type(node).__name__)
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name:
                tokens.append("C_" + name)
        elif isinstance(node, ast.Attribute):
            tokens.append("A_" + node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names] + [getattr(node, "module", None) or ""]
            tokens += ["I_" + n.replace(".", "_") for n in names if n]
        elif isinstance(node, ast.BinOp):
            tokens.append("OP_" + type(node.op).__name__)
        elif isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if node.value in MODULI:
                tokens.append("K_MOD")
            elif isinstance(node.value, float):
                tokens.append("K_FLOAT")
    return " ".join(tokens)


def difficulty_features(problems) -> np.ndarray:
    """Rating / 1000, a missing flag and six cumulative buckets (>= 1000, 1300, ..., 2500).

    A missing rating is replaced by 1700, the median, and flagged.
    """
    rows = []
    for p in problems:
        d = p["difficulty"]
        x = 1700.0 if d is None else d
        rows.append([x / 1000, d is None] + [x >= b for b in (1000, 1300, 1600, 1900, 2200, 2500)])
    return np.array(rows, dtype=float)


def sample_features(problems) -> np.ndarray:
    """Eleven clues from the sample tests and the output specification (see ``DENSE_NAMES``).

    They target tags whose statement words can be ambiguous: real numbers in the answer
    (geometry, probabilities), lines of two integers (edges of a graph or a tree), player names
    (games), an answer modulo a prime (combinatorics, probabilities).
    """
    rows = []
    for p in problems:
        lines_in = [line for line in p["sample_in"].split("\n") if line.strip()]
        lines_out = [line for line in p["sample_out"].split("\n") if line.strip()]
        out, spec = p["sample_out"], p["output_spec"]
        rows.append([
            bool(FLOAT.search(out)),  # real-valued answers: geometry, probabilities
            len(FLOAT.findall(out)) / (1 + len(INT.findall(out))),
            np.mean([bool(re.fullmatch(r"\s*-?\d+\s+-?\d+\s*", line)) for line in lines_in]) if lines_in else 0,
            np.mean([bool(re.search(r"[a-zA-Z]{3,}", line)) for line in lines_in]) if lines_in else 0,
            np.log1p(len(lines_in)),
            np.log1p(len(lines_out)),
            bool(re.search(r"\b(Alice|Bob|First|Second|Win|Lose)\b", out, re.I)),  # games
            bool(re.search(r"\b(YES|NO)\b", out, re.I)),
            bool(re.search(r"998244353|10\^9\s*\+\s*7|10\^\{9\}\s*\+\s*7|1000000007|modulo", spec)),
            bool(re.search(r"relative error", spec, re.I)),
            max((len(x.lstrip("-")) for x in INT.findall(p["sample_in"])[:2000]), default=0),
        ])
    return np.array(rows, dtype=float)


TEXT_BLOCKS = {
    # name: (input, vectorizer options, weight)
    "words": (lambda p: p["statement"], dict(max_features=30000, **WORDS), 1.0),
    "chars": (lambda p: p["statement"], dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                             sublinear_tf=True, max_features=60000), 0.5),
    "code": (lambda p: clean_code(p["code"]), dict(max_features=20000, **WORDS), 0.5),
    "ast": (lambda p: ast_tokens(p["code"]), dict(max_features=20000, token_pattern=r"\S+", lowercase=False,
                                                  ngram_range=(1, 2), min_df=2, sublinear_tf=True), 0.3),
}
DENSE_BLOCKS = {"difficulty": (difficulty_features, 0.25), "samples": (sample_features, 0.25)}
DENSE_NAMES = {
    "difficulty": ["rating", "rating_missing"] + [f"rating>={b}" for b in (1000, 1300, 1600, 1900, 2200, 2500)],
    "samples": ["real_answer", "real_share", "edge_lines", "word_lines", "input_lines", "output_lines",
                "player_names", "yes_no", "modulo", "relative_error", "max_digits"],
}


class Features:
    """Concatenation of weighted blocks: TF-IDF blocks are L2-normalised, dense blocks standardised."""

    def __init__(self, blocks):
        """Args: blocks: names taken from ``TEXT_BLOCKS`` and ``DENSE_BLOCKS``."""
        unknown = set(blocks) - set(TEXT_BLOCKS) - set(DENSE_BLOCKS)
        if unknown:
            raise ValueError(f"Unknown feature blocks: {sorted(unknown)}")
        self.blocks = tuple(blocks)

    def fit(self, problems):
        """Learn each block (vocabulary and IDF, or mean and variance) on training problems."""
        self.fitted = {}
        for name in self.blocks:
            if name in TEXT_BLOCKS:
                get, options, _ = TEXT_BLOCKS[name]
                self.fitted[name] = TfidfVectorizer(**options).fit([get(p) for p in problems])
            else:
                self.fitted[name] = StandardScaler().fit(DENSE_BLOCKS[name][0](problems))
        return self

    def names(self) -> list[str]:
        """Readable name of every column, prefixed by its block (e.g. "code:gcd", "difficulty:2")."""
        names = []
        for name in self.blocks:
            if name in TEXT_BLOCKS:
                names += [f"{name}:{term}" for term in self.fitted[name].get_feature_names_out()]
            else:
                names += [f"{name}:{label}" for label in DENSE_NAMES[name]]
        return names

    def transform(self, problems):
        """Sparse matrix (n_problems, n_features) built with the fitted blocks."""
        parts = []
        for name in self.blocks:
            if name in TEXT_BLOCKS:
                get, _, weight = TEXT_BLOCKS[name]
                parts.append(self.fitted[name].transform([get(p) for p in problems]) * weight)
            else:
                function, weight = DENSE_BLOCKS[name]
                values = np.clip(self.fitted[name].transform(function(problems)), -5, 5)
                parts.append(sparse.csr_matrix(values * weight))
        return sparse.hstack(parts, format="csr")
