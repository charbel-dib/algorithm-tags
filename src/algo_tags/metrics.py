"""Multilabel metrics, per tag and global.

* F1 of a tag: harmonic mean of precision and recall for that tag.
* macro8: mean of the eight F1 scores, every tag counts the same, however rare.
* macro9: macro8 extended with the F1 of the derived state "no tag" (all eight outputs negative).
  46 % of the problems have none of the eight tags, so a model that tags everything must be
  penalised: this is the selection criterion.
* micro: F1 computed on all decisions pooled, dominated by frequent tags such as math.
"""

import numpy as np

from .data import TAGS


def _f1(tp, fp, fn):
    return np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)


def scores_for(y, pred) -> dict:
    """Global and per-tag scores for boolean matrices of shape (n_problems, 8).

    Returns a dict with ``f1_macro9``, ``f1_macro8``, ``f1_micro``, ``f1_no_tag``, ``exact_match``,
    ``false_alarms_no_tag`` (problems without any target that receive at least one tag),
    ``no_tag_support`` and ``per_tag`` (precision, recall, F1, support, tp, fp, fn, tn per tag).
    """
    y, pred = np.asarray(y, dtype=bool), np.asarray(pred, dtype=bool)
    tp, fp, fn = (y & pred).sum(0), (~y & pred).sum(0), (y & ~pred).sum(0)
    tn = (~y & ~pred).sum(0)
    f1 = _f1(tp, fp, fn)
    empty, pred_empty = ~y.any(1), ~pred.any(1)
    f1_empty = float(_f1((empty & pred_empty).sum(), (~empty & pred_empty).sum(), (empty & ~pred_empty).sum()))
    per_tag = {
        tag: dict(precision=float(tp[j] / max(tp[j] + fp[j], 1)), recall=float(tp[j] / max(tp[j] + fn[j], 1)),
                  f1=float(f1[j]), support=int(y[:, j].sum()),
                  tp=int(tp[j]), fp=int(fp[j]), fn=int(fn[j]), tn=int(tn[j]))
        for j, tag in enumerate(TAGS)
    }
    return dict(
        f1_macro9=float((f1.sum() + f1_empty) / (len(TAGS) + 1)),
        f1_macro8=float(f1.mean()),
        f1_micro=float(_f1(tp.sum(), fp.sum(), fn.sum())),
        f1_no_tag=f1_empty,
        exact_match=float((y == pred).all(1).mean()),
        false_alarms_no_tag=int((empty & pred.any(1)).sum()),
        no_tag_support=int(empty.sum()),
        per_tag=per_tag,
    )


def tag_confusion(y, pred) -> np.ndarray:
    """Where the problems of each true tag end up, to see which tags are mistaken for which.

    Rows: true tags, then "no tag". Columns: predicted tags, then "no tag".
    Diagonal cell (i, i): share of the problems of tag i that receive tag i (the recall of i).
    Off-diagonal cell (i, j): share of the problems of tag i that wrongly receive tag j, that is
    j is predicted but is not one of their true tags. Correct co-tags are not counted, so a
    number theory problem truly tagged math as well does not show up as a confusion.
    """
    y, pred = np.asarray(y, dtype=bool), np.asarray(pred, dtype=bool)
    true = np.column_stack([y, ~y.any(1)]).astype(float)
    predicted = np.column_stack([pred, ~pred.any(1)]).astype(float)
    counts = true.T @ (predicted * (1 - true))
    np.fill_diagonal(counts, (true * predicted).sum(0))
    return counts / np.maximum(true.sum(0), 1)[:, None]


def table(result: dict) -> str:
    """Readable summary for the terminal: global scores, then tags from best to worst F1."""
    lines = [f"{k:20s} {100 * result[k]:6.2f} %" for k in ("f1_macro9", "f1_macro8", "f1_micro", "f1_no_tag",
                                                           "exact_match")]
    lines.append(f"{'no-tag false alarms':20s} {result['false_alarms_no_tag']} / {result['no_tag_support']}")
    lines.append(f"\n{'tag':15s} {'precision':>9s} {'recall':>7s} {'F1':>7s} {'support':>8s}")
    for tag, s in sorted(result["per_tag"].items(), key=lambda kv: -kv[1]["f1"]):
        lines.append(f"{tag:15s} {100 * s['precision']:9.2f} {100 * s['recall']:7.2f} {100 * s['f1']:7.2f} "
                     f"{s['support']:8d}")
    return "\n".join(lines)
