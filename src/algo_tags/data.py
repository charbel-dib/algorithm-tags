"""Read xCodeEval problems (raw JSON files) and the train / validation / test split."""

import csv
import json
import unicodedata
from pathlib import Path

import numpy as np

TAGS = ("math", "graphs", "strings", "number theory", "trees", "geometry", "games", "probabilities")
SECTIONS = (
    ("DESCRIPTION", "prob_desc_description"),
    ("INPUT", "prob_desc_input_spec"),
    ("OUTPUT", "prob_desc_output_spec"),
    ("NOTES", "prob_desc_notes"),
)


def clean(text) -> str:
    """Unicode NFC, LF line endings, trimmed. Symbols and inner spacing are kept."""
    if text is None:
        return ""
    return unicodedata.normalize("NFC", str(text)).replace("\r\n", "\n").replace("\r", "\n").strip()


def _samples(raw: dict, key: str) -> str:
    try:
        return clean("\n".join(json.loads(raw.get(key) or "[]")))
    except (TypeError, ValueError):
        return ""


def to_problem(raw: dict) -> dict:
    """Model inputs of one problem. Tags are only read when present (training/evaluation)."""
    if not clean(raw.get("prob_desc_description")):
        raise ValueError("prob_desc_description is required")
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


def load_problems(path) -> list[dict]:
    """Load one JSON file or a folder of JSON files (searched recursively), sorted by src_uid."""
    path = Path(path)
    files = sorted(path.rglob("*.json")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"No JSON file found in {path}")
    problems = [to_problem(json.loads(f.read_text(encoding="utf-8-sig"))) for f in files]
    ids = [p["src_uid"] for p in problems if p["src_uid"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate src_uid in the data")
    return sorted(problems, key=lambda p: p["src_uid"])


def targets(problems: list[dict]) -> np.ndarray:
    return np.array([p["targets"] for p in problems], dtype=bool)


def read_split(path) -> dict[str, str]:
    with open(path, encoding="utf-8", newline="") as f:
        return {row["src_uid"]: row["split"] for row in csv.DictReader(f)}


def select(problems: list[dict], split: dict[str, str], parts) -> list[dict]:
    """Problems whose split is in `parts` (e.g. ["train"] or ["train", "validation"])."""
    if {p["src_uid"] for p in problems} != set(split):
        raise ValueError("The data and the split file do not contain the same problems")
    return [p for p in problems if split[p["src_uid"]] in parts]
