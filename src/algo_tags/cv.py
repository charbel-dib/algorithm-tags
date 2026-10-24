"""Repeated cross-validation used to compare models, on train + validation pooled (3,978 problems).

For each repetition r (seed r), the problems are cut into 5 folds. Each fold is predicted by a
model trained on the 4 others, whose threshold is itself chosen inside those 4 folds
(:meth:`TagModel.fit`). Every problem is therefore predicted once by a model that never saw it,
and the scores of a repetition are computed on all pooled predictions. Repeating with several
seeds measures how much a result depends on the cut.
"""

import csv
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

from . import plots
from .data import TAGS
from .log import get_logger
from .metrics import scores_for
from .model import TagModel

log = get_logger("cv")
GLOBAL = ("f1_macro9", "f1_macro8", "f1_micro", "f1_no_tag", "exact_match", "false_alarms_no_tag")


def _fold(name, problems, y, train, test):
    model = TagModel.preset(name).fit([problems[i] for i in train], y[train])
    return test, model.predict([problems[i] for i in test])


def cross_validate(name, problems, y, repeats=4, folds=5, jobs=-1) -> list[dict]:
    """Scores of the preset ``name``, one dict per repetition (see :func:`metrics.scores_for`).

    The folds depend only on the seeds and on the order of ``problems`` (sorted by ``src_uid``),
    so two models evaluated with the same arguments see exactly the same folds.
    """
    start = time.perf_counter()
    tasks = [(r, train, test) for r in range(repeats)
             for train, test in KFold(folds, shuffle=True, random_state=r).split(problems)]
    outputs = Parallel(n_jobs=jobs)(delayed(_fold)(name, problems, y, tr, te) for _, tr, te in tasks)
    results = []
    for r in range(repeats):
        pred = np.zeros(y.shape, dtype=bool)
        for (rep, _, _), (test, p) in zip(tasks, outputs):
            if rep == r:
                pred[test] = p
        results.append(scores_for(y, pred))
    macro9 = [round(100 * res["f1_macro9"], 2) for res in results]
    log.info("%s: macro9 %.2f (per repetition %s) in %.0f s", name, np.mean(macro9), macro9,
             time.perf_counter() - start)
    return results


def write_results(results: dict[str, list[dict]], folder) -> None:
    """Write ``cv_global.csv`` (mean and std over repetitions), ``cv_per_tag.csv`` and two figures."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "cv_global.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["model"] + [f"{k}_{s}" for k in GLOBAL for s in ("mean", "std")] + ["repeats"])
        for name, runs in results.items():
            row = [name]
            for k in GLOBAL:
                values = np.array([run[k] for run in runs], dtype=float) * (1 if k.startswith("false") else 100)
                row += [round(values.mean(), 2), round(values.std(), 2)]
            w.writerow(row + [len(runs)])
    with open(folder / "cv_per_tag.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["model", "tag", "precision", "recall", "f1", "f1_std", "support"])
        for name, runs in results.items():
            for tag in TAGS:
                s = {k: np.array([run["per_tag"][tag][k] for run in runs]) for k in ("precision", "recall", "f1")}
                w.writerow([name, tag] + [round(100 * s[k].mean(), 2) for k in ("precision", "recall", "f1")]
                           + [round(100 * s["f1"].std(), 2), runs[0]["per_tag"][tag]["support"]])
    plots.cv_per_tag(folder)
    plots.cv_ablation(folder)
    log.info("cross-validation results written to %s", folder)
