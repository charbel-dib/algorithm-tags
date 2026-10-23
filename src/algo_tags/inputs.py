"""Lecture des JSON bruts ou préparés pour la prédiction et l'évaluation."""

import csv
import json
from pathlib import Path

import numpy as np

from .data import SECTIONS, TARGET_TAGS, TEXT_FIELDS, encode_tags, normalize_text


def input_text(record: dict, field: str) -> str:
    """Lit un JSON brut ou préparé, sans utiliser les tags, identifiants ou code."""
    if not isinstance(record, dict):
        raise ValueError("Un objet JSON est attendu")
    if "inputs" in record:
        if not isinstance(record["inputs"], dict):
            raise ValueError("inputs doit être un objet JSON")
        return normalize_text(record["inputs"].get(field), field=field, required=True)
    if field == "description":
        return normalize_text(record.get(TEXT_FIELDS[field]), field=field, required=True)
    texts = {
        name: normalize_text(record.get(source), field=source, required=name == "description")
        for name, source in TEXT_FIELDS.items()
    }
    return "\n\n".join(f"[{section}]\n{texts[name]}" for section, name in SECTIONS if texts[name])


def read_assignments(path):
    assignments = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["src_uid", "split"]:
            raise ValueError("Colonnes attendues : src_uid,split")
        for row in reader:
            uid, part = row["src_uid"], row["split"]
            if not uid or uid in assignments or part not in ("train", "validation", "test"):
                raise ValueError("Identifiant répété ou affectation de split invalide")
            assignments[uid] = part
    return assignments


def load_records(path, *, labels=False, assignments=None, partition=None):
    """Accepte un JSON, un dossier de JSON ou un JSONL préparé."""
    if (assignments is None) != (partition is None):
        raise ValueError("Le split et la partition doivent être fournis ensemble")
    path = Path(path)
    if path.is_dir():
        files = sorted(path.rglob("*.json"))
    else:
        files = [path]
    rows, seen = [], set()
    for file in files:
        content = file.read_text(encoding="utf-8-sig")
        records = (
            [json.loads(line) for line in content.splitlines() if line.strip()]
            if file.suffix == ".jsonl"
            else [json.loads(content)]
        )
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"{file.name} : objet JSON attendu")
            uid = record.get("src_uid")
            if not isinstance(uid, str) or not uid:
                if labels or assignments is not None:
                    raise ValueError("src_uid requis pour les données supervisées")
                uid = file.stem
            if uid in seen:
                raise ValueError("Identifiant répété : " + uid)
            seen.add(uid)
            if assignments is not None:
                if uid not in assignments:
                    raise ValueError("Identifiant absent du split : " + uid)
                if assignments[uid] != partition:
                    continue
            row = {"src_uid": uid, "inputs": {"statement": input_text(record, "statement")}}
            if labels:
                targets = record.get("targets") if "targets" in record else encode_tags(record.get("tags"))
                if (
                    not isinstance(targets, list)
                    or len(targets) != len(TARGET_TAGS)
                    or any(type(v) is not int or v not in (0, 1) for v in targets)
                ):
                    raise ValueError("targets doit contenir huit entiers binaires")
                row["targets"] = targets
            rows.append(row)
    if assignments is not None and seen != set(assignments):
        raise ValueError("Les identifiants du corpus et du split ne correspondent pas")
    if not rows:
        raise ValueError("Aucun exercice à traiter")
    return sorted(rows, key=lambda row: row["src_uid"])


def arrays(rows):
    return (
        np.array([row["src_uid"] for row in rows]),
        np.array([row["targets"] for row in rows], dtype=np.int8),
    )
