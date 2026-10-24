"""Figures written by ``evaluate`` (one model on one split) and ``cv`` (model comparison).

All figures use the same two colours: grey for the reference, blue for the model being shown.
"""

import csv
from pathlib import Path

import numpy as np

from .data import TAGS

GREY, BLUE = "#9aa5b1", "#2f6db5"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    _plt().close(fig)


def confusion_per_tag(result: dict, path) -> None:
    """One 2x2 confusion matrix per tag (true vs predicted), with counts and the tag's F1."""
    plt = _plt()
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
    for ax, tag in zip(axes.ravel(), TAGS):
        s = result["per_tag"][tag]
        m = np.array([[s["tn"], s["fp"]], [s["fn"], s["tp"]]])
        ax.imshow(m / m.sum(1, keepdims=True).clip(1), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{m[i, j]}", ha="center", va="center", fontsize=11,
                        color="white" if m[i, j] / max(m[i].sum(), 1) > 0.6 else "black")
        ax.set_xticks([0, 1], ["predicted no", "predicted yes"], fontsize=8)
        ax.set_yticks([0, 1], ["true no", "true yes"], fontsize=8)
        ax.set_title(f"{tag}  (F1 {100 * s['f1']:.1f})", fontsize=10)
    fig.suptitle("Confusion matrix per tag (counts; colour = share of the true row)", fontsize=11)
    _save(fig, path)


def tag_confusion(matrix: np.ndarray, path) -> None:
    """Heatmap of :func:`algo_tags.metrics.tag_confusion`: recall on the diagonal, wrong tags elsewhere."""
    plt = _plt()
    labels = list(TAGS) + ["no tag"]
    fig, ax = plt.subplots(figsize=(8, 6.8))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = 100 * matrix[i, j]
            if v >= 1:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 60 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.set_xlabel("predicted tag")
    ax.set_ylabel("true tag")
    ax.set_title("Diagonal: recall of each tag (%). Elsewhere: share (%) wrongly given another tag", fontsize=10)
    _save(fig, path)


def per_tag_scores(result: dict, path) -> None:
    """Precision, recall and F1 of every tag, sorted from best to worst F1."""
    plt = _plt()
    order = sorted(TAGS, key=lambda t: result["per_tag"][t]["f1"])
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for k, (metric, color, alpha) in enumerate([("precision", GREY, 1), ("recall", BLUE, 0.45), ("f1", BLUE, 1)]):
        values = [100 * result["per_tag"][t][metric] for t in order]
        ax.barh(y + (k - 1) * 0.27, values, 0.27, color=color, alpha=alpha, label=metric)
    ax.set_yticks(y, [f"{t} ({result['per_tag'][t]['support']})" for t in order])
    ax.set_xlim(0, 100)
    ax.set_xlabel("% (support in brackets)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    _save(fig, path)


def cv_per_tag(folder) -> None:
    """Per-tag F1 of the first model (baseline) and of the best model by macro9, from the CV files."""
    plt = _plt()
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
    for k, (name, color) in enumerate([(baseline, GREY), (best, BLUE)]):
        f1 = [float(per_tag[name, t]["f1"]) for t in order]
        std = [float(per_tag[name, t]["f1_std"]) for t in order]
        ax.barh(y + (0.2 if k else -0.2), f1, 0.4, xerr=std, color=color, label=name, capsize=2)
    ax.set_yticks(y, order)
    ax.set_xlim(0, 100)
    ax.set_xlabel("F1 (%), cross-validation (mean ± std over repetitions)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, folder / "cv_per_tag_f1.png")


def cv_ablation(folder) -> None:
    """Macro9 of every compared model (dot = mean, bar = std over repetitions), in the CV file order."""
    plt = _plt()
    folder = Path(folder)
    with open(folder / "cv_global.csv") as f:
        rows = list(csv.DictReader(f))
    names = [r["model"] for r in rows]
    mean = np.array([float(r["f1_macro9_mean"]) for r in rows])
    std = np.array([float(r["f1_macro9_std"]) for r in rows])
    best = int(mean.argmax())
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(rows) + 1.2))
    y = np.arange(len(rows))[::-1]
    for i, (yi, m, s) in enumerate(zip(y, mean, std)):
        color = BLUE if i == best else GREY
        ax.errorbar(m, yi, xerr=s, fmt="o", color=color, ecolor=color, capsize=3, markersize=7)
        ax.text(m + s + 0.2, yi, f"{m:.2f}", va="center", fontsize=9)
    ax.set_yticks(y, names)
    ax.set_xlim(mean.min() - 2, mean.max() + 2)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("F1 macro9 (%), cross-validation")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, folder / "cv_ablation.png")
