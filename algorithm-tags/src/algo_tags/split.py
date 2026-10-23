"""Découpage multilabel 60/20/20, commun à toutes les expériences."""

import csv
import hashlib
import io
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from .data import TARGET_TAGS, _write_atomic


def load_targets(dataset_path: Path) -> tuple[list[str], np.ndarray, str]:
    """Lit uniquement les identifiants et cibles du JSONL préparé."""
    content = dataset_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    summary_path = dataset_path.parent / "preprocessing_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("target_order") != list(TARGET_TAGS):
        raise ValueError("L'ordre des cibles diffère de TARGET_TAGS")
    if summary.get("dataset_sha256") != digest:
        raise ValueError("Le JSONL ne correspond pas au résumé de prétraitement")

    labels = {}
    for line_number, line in enumerate(content.decode("utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("objet JSON attendu")
            uid, targets = row.get("src_uid"), row.get("targets")
            if not isinstance(uid, str) or not uid.strip():
                raise ValueError("src_uid textuel non vide attendu")
            if uid in labels:
                raise ValueError(f"src_uid répété : {uid}")
            if not isinstance(targets, list) or len(targets) != len(TARGET_TAGS):
                raise ValueError("targets doit contenir huit valeurs")
            if any(type(value) is not int or value not in (0, 1) for value in targets):
                raise ValueError("targets doit contenir uniquement des entiers 0 ou 1")
            labels[uid] = targets
        except ValueError as error:
            raise ValueError(f"{dataset_path.name}, ligne {line_number} : {error}") from error

    if len(labels) != summary.get("records_written"):
        raise ValueError("Le nombre d'exemples diffère du résumé de prétraitement")
    if len(labels) < 5:
        raise ValueError("Au moins cinq exemples sont nécessaires au découpage 60/20/20")
    # Même ordre de départ, même si le JSONL a été réordonné puis son hash actualisé.
    ids = sorted(labels)
    return ids, np.array([labels[uid] for uid in ids], dtype=np.int64), digest


def create_split(dataset_path: str | Path, output_dir: str | Path, seed: int = 42) -> dict:
    """Écrit les affectations par problème et les effectifs de chaque ensemble."""
    dataset_path = Path(dataset_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    csv_path = output_dir / "splits.csv"
    report_path = output_dir / "split_summary.json"
    if csv_path.exists() or report_path.exists():
        raise FileExistsError("Un split existe déjà ici. Réutilise-le ou choisis un nouveau dossier.")
    if not 0 <= seed < 2**32:
        raise ValueError("seed doit être compris entre 0 et 2**32 - 1")
    ids, targets, digest = load_targets(dataset_path)

    # La neuvième colonne aide à répartir les vecteurs nuls ; elle n'est pas une cible.
    no_target = targets.sum(axis=1) == 0
    strata = np.column_stack([targets, no_target]).astype(bool)
    indices = np.arange(len(ids))

    test_splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    development, test = next(test_splitter.split(indices, strata))
    validation_splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_local, validation_local = next(validation_splitter.split(development, strata[development]))
    parts = {"train": development[train_local], "validation": development[validation_local], "test": test}

    assigned = np.concatenate(list(parts.values()))
    if any(len(part) == 0 for part in parts.values()) or not np.array_equal(np.sort(assigned), indices):
        raise ValueError("Découpage invalide : chaque exemple doit appartenir à un seul ensemble non vide")

    counts, warnings = {}, []
    assignments = {}
    for name, part in parts.items():
        positives = targets[part].sum(axis=0)
        counts[name] = {
            "records": len(part),
            "fraction": len(part) / len(ids),
            "without_target": int(no_target[part].sum()),
            "positive_counts": {tag: int(n) for tag, n in zip(TARGET_TAGS, positives)},
        }
        for tag, n in zip(TARGET_TAGS, positives):
            if n == 0 or n == len(part):
                warnings.append(
                    f"{name} / {tag} : {n} positifs sur {len(part)} ; une classe binaire est absente"
                )
        assignments.update((ids[i], name) for i in part)

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["src_uid", "split"])
    writer.writerows((uid, assignments[uid]) for uid in ids)
    csv_bytes = buffer.getvalue().encode("utf-8")
    summary = {
        "method": "iterative_multilabel_with_no_target_indicator",
        "seed": seed,
        "requested_fractions": {"train": 0.6, "validation": 0.2, "test": 0.2},
        "records": len(ids),
        "target_order": list(TARGET_TAGS),
        "dataset_sha256": digest,
        "split_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "versions": {name: version(name) for name in ("iterative-stratification", "numpy", "scikit-learn")},
        "checks": {"unique_src_uids": True, "disjoint_and_complete": True},
        "splits": counts,
        "warnings": warnings,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(csv_path, csv_bytes)
    _write_atomic(report_path, (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return summary
