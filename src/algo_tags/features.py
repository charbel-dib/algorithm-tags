"""TF-IDF et séquences de mots, appris uniquement sur les textes d'entraînement."""

import hashlib
import re
from collections import Counter

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import normalize

from .data import TARGET_TAGS

WORD_TOKEN = re.compile(r"(?u)\b\w+\b")
PROTECTED = {
    "no",
    "not",
    "nor",
    "never",
    "without",
    "only",
    "at",
    "most",
    "least",
    "exactly",
    "all",
    "any",
    "each",
    "every",
    "first",
    "last",
    "one",
    "two",
    "zero",
    "same",
    "different",
    "more",
    "less",
    "before",
    "after",
    "between",
    "within",
    "both",
    "either",
    "neither",
}
PATTERNS = {
    "math": [r"\bequations?\b", r"\barithmetic\b", r"\bformulas?\b"],
    "graphs": [r"\bgraphs?\b", r"\bvert(?:ex|ices)\b", r"\bedges?\b"],
    "strings": [r"\bstrings?\b", r"\bsubstrings?\b", r"\bpalindromes?\b"],
    "number theory": [r"\bprimes?\b", r"\bdivis(?:or|ors|ible)\b", r"\bgcd\b", r"\blcm\b"],
    "trees": [r"\btrees?\b", r"\bsubtrees?\b", r"\bancestors?\b"],
    "geometry": [r"\bpolygons?\b", r"\btriangles?\b", r"\bcircles?\b", r"\bcoordinates\b"],
    "games": [r"\bgames?\b", r"\bwinning strategy\b", r"\boptimal play\b", r"\btake\s+turns\b"],
    "probabilities": [
        r"\bprobabilit(?:y|ies)\b",
        r"\bexpected value\b",
        r"\bexpectation\b",
        r"\buniformly at random\b",
    ],
}


def rule_features(texts):
    # Deux colonnes par tag, comme dans l'expérience d'origine. Les colonnes
    # négatives restent nulles : aucune règle d'exclusion n'a été définie.
    result = np.zeros((len(texts), 16), dtype=np.float32)
    for j, tag in enumerate(TARGET_TAGS):
        patterns = [re.compile(pattern, re.I) for pattern in PATTERNS[tag]]
        result[:, 2 * j] = [sum(bool(p.search(text)) for p in patterns) for text in texts]
    return result


class TfidfBlocks:
    def __init__(self, config):
        self.config = config
        self.blocks = []

    def fit(self, rows):
        self.blocks = []
        texts = [row["statement"] for row in rows]
        modes = ("word", "char_wb") if self.config.get("word_char") else ("word",)
        stopwords = sorted(ENGLISH_STOP_WORDS - PROTECTED) if self.config.get("stopwords") else None
        for mode in modes:
            options = dict(
                analyzer=mode,
                min_df=self.config.get("min_df", 2),
                max_features=self.config.get("max_features", 50000),
                sublinear_tf=self.config.get("sublinear_tf", False),
                dtype=np.float32,
            )
            if mode == "word":
                options.update(token_pattern=WORD_TOKEN.pattern, ngram_range=(1, 2), stop_words=stopwords)
            else:
                options.update(ngram_range=(3, 5))
            self.blocks.append(("statement", TfidfVectorizer(**options).fit(texts)))
        return self

    def transform(self, rows):
        parts = [encoder.transform([row[field] for row in rows]) for field, encoder in self.blocks]
        matrix = sparse.hstack(parts, format="csr")
        if self.config.get("normalize_blocks", False):
            matrix = normalize(matrix)
        if self.config.get("manual_rules", False):
            # Même combinaison que dans l'expérience exploratoire : chaque bloc
            # est normalisé, puis leur concaténation est normalisée à son tour.
            rules = normalize(rule_features([row["statement"] for row in rows]))
            matrix = normalize(sparse.hstack([matrix, sparse.csr_matrix(rules)], format="csr"))
        return matrix


def stable_hash(value):
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little")


class Sequences:
    def __init__(self, config):
        self.config = config
        self.field = "statement"
        self.max_tokens = config.get("max_tokens", 512)
        self.truncation = "head"

    def tokenize(self, text):
        return WORD_TOKEN.findall(text.lower())

    def kept(self, words):
        return words[: self.max_tokens]

    def fit(self, rows):
        counts = Counter()
        for row in rows:
            counts.update(set(self.tokenize(row[self.field])))
        words = sorted(
            (w for w in counts if counts[w] >= self.config.get("min_df", 2)),
            key=lambda word: (-counts[word], word),
        )
        self.vocabulary_ = {
            word: i + 2 for i, word in enumerate(words[: self.config.get("vocab_size", 20000) - 2])
        }
        return self

    def transform(self, rows):
        values = np.zeros((len(rows), self.max_tokens), dtype=np.int64)
        for i, row in enumerate(rows):
            ids = [self.vocabulary_.get(word, 1) for word in self.kept(self.tokenize(row[self.field]))] or [1]
            values[i, : len(ids)] = ids
        return values

    def initialize(self, rows, width, kind, seed):
        from gensim.models import FastText, Word2Vec

        sentences = [self.tokenize(row[self.field]) for row in rows]
        cls = FastText if kind == "fasttext" else Word2Vec
        model = cls(
            sentences=sentences,
            vector_size=width,
            window=5,
            min_count=2,
            workers=1,
            sg=1,
            seed=seed,
            epochs=20,
            hashfxn=stable_hash,
        )
        matrix = (
            np.random.default_rng(seed).normal(0, 0.02, (len(self.vocabulary_) + 2, width)).astype("float32")
        )
        matrix[0] = 0
        for word, i in self.vocabulary_.items():
            if word in model.wv:
                matrix[i] = model.wv[word]
        return matrix
