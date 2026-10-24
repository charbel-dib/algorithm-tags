"""Multilabel metrics, per tag and global.

macro8 = mean F1 of the eight tags. macro9 = macro8 extended with the F1 of the derived state
"no tag" (all eight outputs negative): it rewards a model that leaves out-of-scope problems empty.
"""

import numpy as np

from .data import TAGS


def _f1(tp, fp, fn):
    return np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)


def scores_for(y, pred) -> dict:
    """Global and per-tag scores for boolean matrices of shape (n, 8)."""
    y, pred = np.asarray(y, dtype=bool), np.asarray(pred, dtype=bool)
    tp, fp, fn = (y & pred).sum(0), (~y & pred).sum(0), (y & ~pred).sum(0)
    f1 = _f1(tp, fp, fn)
    empty, pred_empty = ~y.any(1), ~pred.any(1)
    f1_empty = float(_f1((empty & pred_empty).sum(), (~empty & pred_empty).sum(), (empty & ~pred_empty).sum()))
    per_tag = {
        tag: dict(precision=float(tp[j] / max(tp[j] + fp[j], 1)), recall=float(tp[j] / max(tp[j] + fn[j], 1)),
                  f1=float(f1[j]), support=int(y[:, j].sum()))
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


def table(result: dict) -> str:
    """Readable summary: global scores, then tags from best to worst F1."""
    lines = [f"{k:20s} {100 * result[k]:6.2f} %" for k in ("f1_macro9", "f1_macro8", "f1_micro", "f1_no_tag",
                                                           "exact_match")]
    lines.append(f"{'no-tag false alarms':20s} {result['false_alarms_no_tag']} / {result['no_tag_support']}")
    lines.append(f"\n{'tag':15s} {'precision':>9s} {'recall':>7s} {'F1':>7s} {'support':>8s}")
    for tag, s in sorted(result["per_tag"].items(), key=lambda kv: -kv[1]["f1"]):
        lines.append(f"{tag:15s} {100 * s['precision']:9.2f} {100 * s['recall']:7.2f} {100 * s['f1']:7.2f} "
                     f"{s['support']:8d}")
    return "\n".join(lines)
