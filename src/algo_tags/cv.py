"""Repeated cross-validation used to compare models (train + validation pooled).

For every repetition, each problem is predicted once by a model that never saw it. The decision
threshold is chosen inside the training folds only (see TagModel.fit), so the scores are honest.
"""

import csv
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

from .data import TAGS
from .metrics import scores_for
from .model import TagModel

GLOBAL = ("f1_macro9", "f1_macro8", "f1_micro", "f1_no_tag", "exact_match", "false_alarms_no_tag")


def _fold(name, problems, y, train, test):
    model = TagModel.preset(name).fit([problems[i] for i in train], y[train])
    return test, model.predict([problems[i] for i in test])


def cross_validate(name, problems, y, repeats=4, folds=5, jobs=-1) -> list[dict]:
    """One result per repetition (seeds 0, 1, ...), each computed on all pooled predictions."""
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
    return results


def write_results(results: dict[str, list[dict]], folder):
    """cv_global.csv (mean and std over repetitions), cv_per_tag.csv and a per-tag F1 chart."""
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
    plot_per_tag(folder)


def plot_per_tag(folder):
    """Per-tag F1 of the first model (baseline) and of the best model by macro9, from the CSV files."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(folder)
    with open(folder / "cv_global.csv") as f:
        rows = list(csv.DictReader(f))
    baseline = rows[0]["model"]
    best = max(rows, key=lambda r: float(r["f1_macro9_mean"]))["model"]
    with open(folder / "cv_per_tag.csv") as f:
        per_tag = {(r["model"], r["tag"]): r for r in csv.DictReader(f)}
    order = sorted(TAGS, key=lambda t: float(per_tag[best, t]["f1"]))
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, (name, color) in enumerate([(baseline, "#9aa5b1"), (best, "#2f6db5")]):
        f1 = [float(per_tag[name, t]["f1"]) for t in order]
        std = [float(per_tag[name, t]["f1_std"]) for t in order]
        ax.barh(y + (0.2 if k else -0.2), f1, 0.4, xerr=std, color=color, label=name, capsize=2)
    ax.set_yticks(y, order)
    ax.set_xlim(0, 100)
    ax.set_xlabel("F1 (%), cross-validation (mean ± std over repetitions)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(folder / "cv_per_tag_f1.png", dpi=150)
    plt.close(fig)
