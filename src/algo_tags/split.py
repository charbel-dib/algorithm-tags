"""Stratified multilabel split into train / validation / test (60 / 20 / 20 by default).

The reference split, built with the defaults, is stored in ``data/splits/splits.csv``.

Iterative stratification (Sechidis et al., 2011) keeps the share of every tag close in the three
parts. A ninth column "no tag" is added so that problems without any target are balanced too.
"""

import csv
from pathlib import Path

import numpy as np

from .data import targets
from .log import get_logger

log = get_logger("split")


def make_split(problems, seed=42, test_size=0.2, validation_size=0.2) -> dict[str, str]:
    """Map each ``src_uid`` to ``train``, ``validation`` or ``test`` (requires ``iterative-stratification``).

    ``test_size`` and ``validation_size`` are fractions of all the problems; train gets the rest.
    The test part is drawn first, then the validation part from what remains. The problems must be
    sorted by ``src_uid`` (as returned by ``load_problems``) for the split to be reproducible.

    Raises:
        ValueError: if a size is not strictly between 0 and 1, or if both leave nothing for train.
    """
    for name, size in (("test_size", test_size), ("validation_size", validation_size)):
        if not 0 < size < 1:
            raise ValueError(f"{name} must be between 0 and 1 (exclusive), got {size}")
    if test_size + validation_size >= 1:
        raise ValueError(f"test_size + validation_size must be below 1 to leave problems for train, "
                         f"got {test_size} + {validation_size}")
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    ids = [p["src_uid"] for p in problems]
    y = targets(problems)
    strata = np.column_stack([y, ~y.any(1)])
    index = np.arange(len(ids))
    dev, test = next(MultilabelStratifiedShuffleSplit(1, test_size=test_size, random_state=seed).split(index, strata))
    val_share = validation_size / (1 - test_size)  # validation as a share of what is left after the test
    train, val = next(MultilabelStratifiedShuffleSplit(1, test_size=val_share, random_state=seed)
                      .split(dev, strata[dev]))
    parts = {"train": dev[train], "validation": dev[val], "test": test}
    log.info("split: %s", {name: len(part) for name, part in parts.items()})
    return {ids[i]: name for name, part in parts.items() for i in part}


def write_split(split: dict[str, str], path) -> None:
    """Write the split as a two-column CSV (``src_uid,split``), sorted by ``src_uid``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["src_uid", "split"])
        writer.writerows(sorted(split.items()))
    log.info("split written to %s", path)
