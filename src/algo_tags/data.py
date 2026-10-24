"""Reading the problems: raw xCodeEval JSON files, the prepared JSONL file and the split.

A *problem* is a plain dict with the model inputs (``statement``, ``code``, ``sample_in``,
``sample_out``, ``output_spec``, ``difficulty``), its ``src_uid`` and, when the tags are known,
``targets``: eight 0/1 values in the order of ``TAGS``.
"""

import csv
import json
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

from .log import get_logger

log = get_logger("data")

TAGS = ("math", "graphs", "strings", "number theory", "trees", "geometry", "games", "probabilities")
SECTIONS = (
    ("DESCRIPTION", "prob_desc_description"),
    ("INPUT", "prob_desc_input_spec"),
    ("OUTPUT", "prob_desc_output_spec"),
    ("NOTES", "prob_desc_notes"),
)
FIELDS = ("src_uid", "statement", "output_spec", "code", "sample_in", "sample_out", "difficulty")


def clean(text) -> str:
    """Unicode NFC, LF line endings and trimmed ends. Symbols and inner spacing are kept."""
    if text is None:
        return ""
    return unicodedata.normalize("NFC", str(text)).replace("\r\n", "\n").replace("\r", "\n").strip()


def _samples(raw: dict, key: str) -> str:
    """The sample tests of a problem, stored by xCodeEval as a JSON-encoded list of strings."""
    try:
        return clean("\n".join(json.loads(raw.get(key) or "[]")))
    except (TypeError, ValueError):
        return ""


def to_problem(raw: dict) -> dict:
    """Turn one raw xCodeEval record into a problem.

    The statement keeps its four sections, each introduced by a marker such as ``[INPUT]``.
    Tags are read only when present, so the same function serves training and prediction.
    Tags outside the eight targets are ignored: a problem tagged only ``greedy`` gets eight zeros.

    Raises:
        ValueError: if the description is missing or empty.
    """
    if not clean(raw.get("prob_desc_description")):
        raise ValueError(f"problem {raw.get('src_uid', '?')}: prob_desc_description is required")
    difficulty = raw.get("difficulty")
    problem = {
        "src_uid": raw.get("src_uid", ""),
        "statement": "\n\n".join(f"[{name}]\n{clean(raw.get(key))}" for name, key in SECTIONS if clean(raw.get(key))),
        "output_spec": clean(raw.get("prob_desc_output_spec")),
        "code": (raw.get("source_code") or "").replace("\r\n", "\n"),
        "sample_in": _samples(raw, "prob_desc_sample_inputs"),
        "sample_out": _samples(raw, "prob_desc_sample_outputs"),
        "difficulty": float(difficulty) if isinstance(difficulty, (int, float)) and difficulty > 0 else None,
    }
    if "tags" in raw:
        problem["targets"] = [int(tag in set(raw["tags"] or [])) for tag in TAGS]
    return problem


def _read_records(path: Path) -> list[dict]:
    """All JSON objects of a file (``.json`` or ``.jsonl``) or of a folder searched recursively."""
    if path.is_dir():
        files = sorted(path.rglob("*.json")) + sorted(path.rglob("*.jsonl"))
    elif path.exists():
        files = [path]
    else:
        raise FileNotFoundError(f"{path} does not exist")
    if not files:
        raise ValueError(f"no .json or .jsonl file found in {path}")
    records = []
    for file in files:
        text = file.read_text(encoding="utf-8-sig")
        if file.suffix == ".jsonl":
            records += [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            records.append(json.loads(text))
    return records


def load_problems(path) -> list[dict]:
    """Load problems from raw JSON files, a folder of them, or the prepared JSONL file.

    Records that already contain a ``statement`` (prepared data) are kept as they are; the
    others are converted with :func:`to_problem`. The result is sorted by ``src_uid``.

    Raises:
        ValueError: on a duplicated ``src_uid`` or an invalid record.
    """
    path = Path(path)
    problems = [r if "statement" in r else to_problem(r) for r in _read_records(path)]
    ids = [p["src_uid"] for p in problems if p["src_uid"]]
    if len(ids) != len(set(ids)):
        duplicated = [uid for uid, n in Counter(ids).items() if n > 1][:3]
        raise ValueError(f"duplicated src_uid in {path}: {duplicated}")
    log.info("loaded %d problems from %s", len(problems), path)
    return sorted(problems, key=lambda p: p["src_uid"])


def preprocess(raw_path, output) -> dict:
    """Validate the raw JSON files and write one prepared problem per line (JSONL).

    A summary (counts, tag supports, missing fields) is written next to the output as
    ``<name>_summary.json`` and returned.
    """
    problems = load_problems(raw_path)
    if any("targets" not in p for p in problems):
        raise ValueError("every raw problem needs a 'tags' field to build the training data")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as f:
        for p in problems:
            f.write(json.dumps({k: p[k] for k in FIELDS + ("targets",)}, ensure_ascii=False) + "\n")
    y = targets(problems)
    summary = {
        "problems": len(problems),
        "without_target": int((~y.any(1)).sum()),
        "with_several_targets": int((y.sum(1) > 1).sum()),
        "positives": {tag: int(y[:, j].sum()) for j, tag in enumerate(TAGS)},
        "missing_difficulty": sum(p["difficulty"] is None for p in problems),
        "empty_code": sum(not p["code"].strip() for p in problems),
    }
    output.with_name(output.stem + "_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.info("prepared %d problems -> %s (%d without any target)", len(problems), output, summary["without_target"])
    return summary


def targets(problems: list[dict]) -> np.ndarray:
    """Boolean matrix (n_problems, 8) of the known tags."""
    return np.array([p["targets"] for p in problems], dtype=bool)


def read_split(path) -> dict[str, str]:
    """Map ``src_uid`` to ``train``, ``validation`` or ``test`` from a two-column CSV file."""
    with open(path, encoding="utf-8", newline="") as f:
        return {row["src_uid"]: row["split"] for row in csv.DictReader(f)}


def select(problems: list[dict], split: dict[str, str], parts) -> list[dict]:
    """Keep the problems whose split is in ``parts`` (for example ``["train", "validation"]``).

    Raises:
        ValueError: if the data and the split file do not list the same problems.
    """
    if {p["src_uid"] for p in problems} != set(split):
        raise ValueError("the data and the split file do not contain the same problems")
    return [p for p in problems if split[p["src_uid"]] in parts]
