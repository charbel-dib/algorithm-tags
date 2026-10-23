"""Évaluation multilabel globale, par tag et pour les sorties vides."""

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support

from .data import TARGET_TAGS


def evaluate(y_true, y_pred) -> dict:
    """Évalue deux matrices binaires (nombre d'exemples, huit tags)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    for values in (y_true, y_pred):
        if values.ndim != 2 or values.shape[1] != len(TARGET_TAGS) or not len(values):
            raise ValueError("Une matrice non vide à huit colonnes est attendue")
        if not np.isin(values, [0, 1]).all():
            raise ValueError("Les cibles et prédictions doivent être binaires")
    if y_true.shape != y_pred.shape:
        raise ValueError("Les cibles et prédictions doivent avoir la même forme")

    overall = {}
    for average in ("macro", "micro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        overall.update(
            {
                f"precision_{average}": float(precision),
                f"recall_{average}": float(recall),
                f"f1_{average}": float(f1),
            }
        )
    overall["exact_match"] = float((y_true == y_pred).all(axis=1).mean())
    no_target = ~y_true.any(axis=1)
    false_alarms = int(y_pred[no_target].any(axis=1).sum())
    overall["without_target_count"] = int(no_target.sum())
    overall["without_target_false_positives"] = false_alarms
    overall["without_target_false_positive_rate"] = (
        float(false_alarms / no_target.sum()) if no_target.any() else None
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    matrices = multilabel_confusion_matrix(y_true, y_pred)
    per_label, warnings = {}, []
    for i, tag in enumerate(TARGET_TAGS):
        tn, fp, fn, tp = map(int, matrices[i].ravel())
        per_label[tag] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "predicted_positive": tp + fp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
        if support[i] == 0 or support[i] == len(y_true):
            warnings.append(f"{tag} : une classe binaire est absente des cibles")
    true_empty = no_target.astype(int)
    predicted_empty = (~y_pred.any(axis=1)).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        true_empty, predicted_empty, average="binary", zero_division=0
    )
    tn, fp, fn, tp = map(
        int,
        multilabel_confusion_matrix(
            np.column_stack([1 - true_empty, true_empty]),
            np.column_stack([1 - predicted_empty, predicted_empty]),
        )[1].ravel(),
    )
    no_target_metrics = {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "support": int(no_target.sum()),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    overall["f1_no_target"] = float(f)
    overall["f1_macro9"] = (8 * overall["f1_macro"] + float(f)) / 9
    return {
        "records": len(y_true),
        "overall": overall,
        "per_label": per_label,
        "no_target": no_target_metrics,
        "warnings": warnings,
    }
