"""Exports lisibles des métriques et vérification des prédictions enregistrées."""

import csv
from pathlib import Path

import numpy as np

from .data import TARGET_TAGS
from .evaluation import evaluate
from .storage import read_json, write_csv, write_json


def save_evaluation(folder, rows, scores, predictions):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    ids = np.array([row["src_uid"] for row in rows])
    y = np.array([row["targets"] for row in rows], dtype=np.int8)
    payload = dict(ids=ids, y=y, predictions=np.asarray(predictions, dtype=np.int8))
    if scores is not None:
        payload["scores"] = scores
    np.savez_compressed(folder / "predictions.npz", **payload)
    report = evaluate(y, predictions)
    write_json(folder / "metrics.json", report)
    write_csv(
        folder / "per_label.csv", [{"tag": tag, **values} for tag, values in report["per_label"].items()]
    )
    plot_confusions(report, folder / "confusions.png")
    return report


def plot_confusions(report, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, tag in zip(axes.flat, TARGET_TAGS):
        values = report["per_label"][tag]
        matrix = np.array([[values["tn"], values["fp"]], [values["fn"], values["tp"]]])
        ax.imshow(matrix, vmin=0, vmax=report["records"], cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > 0.6 * report["records"] else "black",
            )
        ax.set_title(tag)
        ax.set_xticks([0, 1], ["prédit 0", "prédit 1"])
        ax.set_yticks([0, 1], ["vrai 0", "vrai 1"])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def comparison_table(reports, folder):
    folder = Path(folder)
    write_csv(folder / "global.csv", [{"model": name, **r["overall"]} for name, r in reports.items()])
    write_csv(
        folder / "per_label.csv",
        [
            {"model": name, "tag": tag, **values}
            for name, r in reports.items()
            for tag, values in r["per_label"].items()
        ],
    )
    lines = [
        "| Modèle | F1 macro8 | F1 macro9 | F1 micro | F1 sans cible | FP sans cible |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, report in reports.items():
        m = report["overall"]
        lines.append(
            f"| {name} | {m['f1_macro']:.2%} | {m['f1_macro9']:.2%} | "
            f"{m['f1_micro']:.2%} | {m['f1_no_target']:.2%} | "
            f"{m['without_target_false_positives']} / {m['without_target_count']} |"
        )
    text = "\n".join(lines) + "\n"
    (folder / "comparison.md").write_text(text, encoding="utf-8")
    return text


def stored_table(results, partition="test"):
    path = Path(results) / partition / "global.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    winner = read_json(Path(results) / "selection.json")["winner"]
    rows.sort(key=lambda row: row["model"] != winner)
    lines = [f"{partition} — F1 macro8 / F1 macro9 / F1 micro / FP sans cible"]
    for row in rows:
        lines.append(
            f"{row['model']:<20} {float(row['f1_macro']):.2%}  "
            f"{float(row['f1_macro9']):.2%}  "
            f"{float(row['f1_micro']):.2%}  "
            f"{row['without_target_false_positives']} / {row['without_target_count']}"
        )
    return "\n".join(lines)


def verify_results(results, registry):
    """Recalcule les métriques depuis les prédictions, sans lire le corpus test."""
    results = Path(results)
    registry = read_json(registry)
    models = registry["models"]
    selection = read_json(results / "selection.json")
    winner = selection["winner"]
    if winner != registry["default_model"] or selection["thresholds"] != models[winner]["thresholds"]:
        raise ValueError("La sélection ne correspond pas au modèle par défaut ou à ses seuils")
    if selection["model_sha256"] != models[winner]["sha256"]:
        raise ValueError("Le modèle sélectionné diffère de l'artefact enregistré")
    for partition in ("validation", "test"):
        reference_ids, reference_y = None, None
        for name, meta in models.items():
            folder = results / partition / name
            with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
                ids, y, prediction = saved["ids"], saved["y"], saved["predictions"]
                expected = (
                    saved["scores"] >= np.asarray(meta["thresholds"])
                    if meta["score_kind"] != "binary_rule"
                    else None
                )
            if len(set(ids.tolist())) != len(ids):
                raise ValueError("Identifiants répétés dans les prédictions")
            if reference_ids is None:
                reference_ids, reference_y = ids, y
            elif not np.array_equal(reference_ids, ids) or not np.array_equal(reference_y, y):
                raise ValueError("Exemples différents entre comparateurs")
            if meta["score_kind"] != "binary_rule":
                if not np.array_equal(prediction, expected):
                    raise ValueError("Prédictions différentes des seuils : " + name)
            report = evaluate(y, prediction)
            published = read_json(folder / "metrics.json")
            if report != published:
                raise ValueError("Métriques incohérentes : " + str(folder))
    return "16 évaluations vérifiées ; métriques et seuils cohérents."
