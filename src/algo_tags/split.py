"""Stratified multilabel 60 / 20 / 20 split (the one stored in data/splits/splits.csv)."""

import csv

import numpy as np

from .data import targets


def make_split(problems, seed=42) -> dict[str, str]:
    """Iterative stratification on the 8 tags plus a "no tag" column, so empty problems are balanced too."""
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    ids = [p["src_uid"] for p in problems]  # problems are sorted by src_uid
    y = targets(problems)
    strata = np.column_stack([y, ~y.any(1)])
    index = np.arange(len(ids))
    dev, test = next(MultilabelStratifiedShuffleSplit(1, test_size=0.2, random_state=seed).split(index, strata))
    train, val = next(MultilabelStratifiedShuffleSplit(1, test_size=0.25, random_state=seed).split(dev, strata[dev]))
    parts = {"train": dev[train], "validation": dev[val], "test": test}
    return {ids[i]: name for name, part in parts.items() for i in part}


def write_split(split: dict[str, str], path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["src_uid", "split"])
        writer.writerows(sorted(split.items()))
