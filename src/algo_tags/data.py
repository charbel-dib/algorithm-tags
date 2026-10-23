"""Préparation déterministe des JSON xCodeEval pour les huit tags retenus."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path

from . import __version__

TARGET_TAGS = (
    "math",
    "graphs",
    "strings",
    "number theory",
    "trees",
    "geometry",
    "games",
    "probabilities",
)

TEXT_FIELDS = {
    "description": "prob_desc_description",
    "input_spec": "prob_desc_input_spec",
    "output_spec": "prob_desc_output_spec",
    "notes": "prob_desc_notes",
}

SECTIONS = (
    ("DESCRIPTION", "description"),
    ("INPUT", "input_spec"),
    ("OUTPUT", "output_spec"),
    ("NOTES", "notes"),
)


def normalize_text(value: object, *, field: str, required: bool = False) -> str:
    """Normalise un champ d'énoncé sans retirer les symboles ni les espaces internes."""
    if value is None and not required:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field} : chaîne attendue")
    text = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    text = text.strip()
    if required and not text:
        raise ValueError(f"{field} : texte non vide attendu")
    return text


def encode_tags(tags: object) -> list[int]:
    """Projette une liste de tags sur TARGET_TAGS ; une liste vide donne huit zéros."""
    if not isinstance(tags, list) or any(not isinstance(tag, str) or not tag.strip() for tag in tags):
        raise ValueError("tags : liste de chaînes non vides attendue ([] est accepté)")
    present = set(tags)
    return [int(tag in present) for tag in TARGET_TAGS]


def prepare_inputs(record: dict) -> dict[str, str]:
    """Prépare les entrées sans consulter les labels ni les métadonnées."""
    texts = {
        name: normalize_text(record.get(field), field=field, required=(name == "description"))
        for name, field in TEXT_FIELDS.items()
    }
    code = record.get("source_code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("source_code : chaîne non vide attendue")
    # Ne pas utiliser strip() : indentation et lignes de bord font partie du code.
    texts["source_code"] = code.replace("\r\n", "\n").replace("\r", "\n")
    texts["statement"] = "\n\n".join(
        f"[{section}]\n{texts[name]}" for section, name in SECTIONS if texts[name]
    )
    return texts


def prepare_record(record: dict, *, source_file: str, source_sha256: str) -> dict:
    """Valide un exemple supervisé et sépare entrées, cibles et métadonnées."""
    for field in ("src_uid", "code_uid"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"{field} : identifiant textuel non vide attendu")
    for field in ("file_name", "lang", "lang_cluster"):
        if record.get(field) is not None and not isinstance(record[field], str):
            raise ValueError(f"{field} : chaîne ou null attendu")
    difficulty = record.get("difficulty")
    if difficulty is not None and (
        isinstance(difficulty, bool)
        or not isinstance(difficulty, (int, float))
        or not math.isfinite(difficulty)
    ):
        raise ValueError("difficulty : nombre fini ou null attendu")
    return {
        "src_uid": record["src_uid"],
        "inputs": prepare_inputs(record),
        "targets": encode_tags(record.get("tags")),
        "metadata": {
            "source_file": source_file,
            "source_sha256": source_sha256,
            "code_uid": record["code_uid"],
            "file_name": record.get("file_name"),
            "lang": record.get("lang"),
            "lang_cluster": record.get("lang_cluster"),
            "difficulty": difficulty,
        },
    }


def _reject_constant(value: str) -> None:
    raise ValueError(f"constante JSON non standard : {value}")


def _write_atomic(path: Path, content: bytes) -> None:
    """Remplace un fichier complet, après fermeture du temporaire (Windows compris)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def preprocess(input_dir: str | Path, output_dir: str | Path) -> dict:
    """Prépare tous les exemples, puis écrit le JSONL et son résumé de contrôle."""
    input_dir = Path(input_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Dossier de données introuvable : {input_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("Le dossier de sortie doit être situé hors du dossier des JSON bruts")
    paths = sorted(
        (p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".json"),
        key=lambda p: p.relative_to(input_dir).as_posix(),
    )
    if not paths:
        raise ValueError(f"Aucun fichier JSON dans {input_dir}")

    prepared, manifest = [], []
    seen_problems, seen_codes = {}, {}
    statuses, positive_counts, cardinality = Counter(), Counter(), Counter()
    optional_empty, difficulty_states = Counter(), Counter()
    repeated_tags = 0

    # Le corpus est petit : tout valider avant de remplacer une sortie existante.
    for path in paths:
        source_file = path.relative_to(input_dir).as_posix()
        try:
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            record = json.loads(content.decode("utf-8-sig"), parse_constant=_reject_constant)
            if not isinstance(record, dict):
                raise ValueError("objet JSON attendu")
            row = prepare_record(record, source_file=source_file, source_sha256=digest)
            for field, seen in (("src_uid", seen_problems), ("code_uid", seen_codes)):
                uid = record[field]
                if uid in seen:
                    raise ValueError(f"{field} répété, déjà présent dans {seen[uid]}")
                seen[uid] = source_file
        except (OSError, UnicodeError, ValueError) as error:
            raise ValueError(f"{source_file} : {error}") from error

        prepared.append(row)
        manifest.append(f"{source_file}\t{digest}")
        tags = record["tags"]
        count = sum(row["targets"])
        status = "empty_original" if not tags else "with_target" if count else "outside_targets"
        statuses[status] += 1
        repeated_tags += len(tags) != len(set(tags))
        cardinality[count] += 1
        positive_counts.update(tag for tag, flag in zip(TARGET_TAGS, row["targets"]) if flag)
        optional_empty.update(
            name for name in ("input_spec", "output_spec", "notes") if not row["inputs"][name]
        )
        difficulty = row["metadata"]["difficulty"]
        difficulty_states[
            "missing" if difficulty is None else "nonpositive" if difficulty <= 0 else "positive"
        ] += 1

    dataset = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in prepared).encode(
        "utf-8"
    )
    summary = {
        "schema_version": 1,
        "package_version": __version__,
        "target_order": list(TARGET_TAGS),
        "files_found": len(paths),
        "records_written": len(prepared),
        "unique_src_uids": len(seen_problems),
        "unique_code_uids": len(seen_codes),
        "annotation_counts": {
            key: statuses[key] for key in ("empty_original", "outside_targets", "with_target")
        },
        "records_without_target": cardinality[0],
        "records_with_target": len(prepared) - cardinality[0],
        "records_with_repeated_tags": repeated_tags,
        "positive_counts": {tag: positive_counts[tag] for tag in TARGET_TAGS},
        "target_cardinality": {str(k): cardinality[k] for k in range(len(TARGET_TAGS) + 1)},
        "optional_empty_fields": {key: optional_empty[key] for key in ("input_spec", "output_spec", "notes")},
        "difficulty_counts": {key: difficulty_states[key] for key in ("missing", "nonpositive", "positive")},
        "corpus_sha256": hashlib.sha256("\n".join(manifest).encode("utf-8")).hexdigest(),
        "dataset_sha256": hashlib.sha256(dataset).hexdigest(),
        "rules": {
            "empty_tags": "eight_zeros",
            "outside_target_tags": "eight_zeros",
            "text": "NFC, LF, strip at field boundaries; internal content preserved",
            "source_code": "LF only; indentation, comments and boundary whitespace preserved",
            "statement_sections": [section for section, _ in SECTIONS],
            "metadata": "kept separately, not model inputs; difficulty unchanged",
            "validation": "stop on malformed records or repeated identifiers; no silent removal",
        },
    }
    report = (json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(output_dir / "dataset.jsonl", dataset)
    _write_atomic(output_dir / "preprocessing_summary.json", report)
    return summary
