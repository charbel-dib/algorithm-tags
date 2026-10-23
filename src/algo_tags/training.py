"""Entraînement sur train, seuil global hors pli, puis évaluation sur validation."""

from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from .classifiers import fit_linear
from .data import TARGET_TAGS
from .environment import package_versions
from .evaluation import evaluate
from .features import Sequences, TfidfBlocks
from .inputs import arrays, load_records, read_assignments
from .model import checked_scores
from .reporting import comparison_table, save_evaluation
from .storage import read_json, sha256, write_json


def f1_objective(y, prediction, metric):
    if metric == "f1_macro9":
        y = np.column_stack([y, ~y.any(1)])
        prediction = np.column_stack([prediction, ~prediction.any(1)])
    elif metric != "f1_macro":
        raise ValueError("Métrique attendue : f1_macro ou f1_macro9")
    numerator = 2.0 * (y & prediction).sum(0)
    denominator = y.sum(0) + prediction.sum(0)
    return float(np.divide(numerator, denominator, out=np.zeros(y.shape[1]), where=denominator != 0).mean())


def tune_threshold(y, scores, kind, metric="f1_macro9"):
    scores = checked_scores(scores, len(y), kind)
    natural = 0.0 if kind == "margin" else 0.5
    if kind == "margin":
        upper = np.nextafter(scores.max(), np.inf)
        if not np.isfinite(upper):
            raise ValueError("Borne de seuil non finie")
        values = np.unique(np.r_[np.quantile(scores, np.linspace(0, 1, 41)), natural, upper])
    else:
        values = np.unique(np.r_[np.arange(0.05, 1.0, 0.05), natural, np.nextafter(1.0, np.inf)])
    # Égalité : seuil le plus proche du seuil naturel, puis seuil le plus élevé.
    ranks = [(round(f1_objective(y, scores >= t, metric), 12), -abs(t - natural), float(t)) for t in values]
    return [max(ranks)[2]] * 8


def fit_model(rows, config, device):
    inputs = [row["inputs"] for row in rows]
    _, y = arrays(rows)
    if config["model"] == "cnn":
        from .neural import fit_cnn

        encoder = Sequences(config).fit(inputs)
        classifier, history = fit_cnn(encoder, inputs, y, config, device)
    else:
        encoder = TfidfBlocks(config).fit(inputs)
        classifier = fit_linear(encoder.transform(inputs), y, config)
        history = []
    return {"encoder": encoder, "classifier": classifier}, history


def bundle_scores(bundle, rows):
    inputs = [row["inputs"] for row in rows]
    return checked_scores(
        bundle["classifier"].scores(bundle["encoder"].transform(inputs)), len(rows), bundle["classifier"].kind
    )


def train_one(name, config, train, validation, folds, folder, provenance, metric, device):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    ids, y = arrays(train)
    oof = np.empty(y.shape, dtype=np.float64)
    for fold in range(3):
        fit = [row for i, row in enumerate(train) if folds[i] != fold]
        held_indices = np.flatnonzero(folds == fold)
        held = [train[i] for i in held_indices]
        print(f"{name} : pli {fold + 1}/3, apprentissage={len(fit)}, hors pli={len(held)}", flush=True)
        bundle, _ = fit_model(fit, config, device)
        oof[held_indices] = bundle_scores(bundle, held)
    kind = bundle["classifier"].kind
    thresholds = tune_threshold(y, oof, kind, metric)
    np.savez_compressed(folder / "oof.npz", ids=ids, y=y, scores=oof, folds=folds)
    print(f"{name} : entraînement sur tout le train", flush=True)
    bundle, history = fit_model(train, config, device)
    metadata = {
        "name": name,
        "parameters": config,
        "target_order": list(TARGET_TAGS),
        "fit_ids": ids.tolist(),
        "fit_partition": "train",
        "versions": package_versions(),
        "threshold_source": "three_fold_oof_train",
        "threshold_metric": metric,
        "thresholds": thresholds,
        "score_kind": kind,
        **provenance,
    }
    bundle["metadata"] = metadata
    model_path = folder / "model.joblib"
    joblib.dump(bundle, model_path, compress=3)
    write_json(folder / "metadata.json", metadata)
    if history:
        write_json(folder / "history.json", history)
    scores = bundle_scores(bundle, validation)
    report = save_evaluation(folder / "validation", validation, scores, scores >= thresholds)
    registry = {
        "target_order": list(TARGET_TAGS),
        "default_model": name,
        "models": {
            name: {
                "path": "model.joblib",
                "sha256": sha256(model_path),
                "score_kind": kind,
                "thresholds": thresholds,
            }
        },
    }
    write_json(folder / "registry.json", registry)
    write_json(
        folder / "complete.json",
        {
            "provenance": provenance,
            "parameters": config,
            "metric": metric,
            "model_sha256": sha256(model_path),
        },
    )
    return report


def run_experiments(
    data,
    splits,
    config_path,
    names,
    output,
    *,
    metric="f1_macro9",
    device="cpu",
    resume=False,
    reference_registry=None,
):
    """Même split et mêmes plis pour tous les candidats. Le test n'est pas chargé."""
    output = Path(output)
    if len(names) != len(set(names)):
        raise ValueError("Un modèle ne peut apparaître qu'une fois")
    catalog = read_json(config_path)
    if not names or any(name not in catalog for name in names):
        raise ValueError("Modèle absent de la configuration")
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("Dossier non vide : choisir un autre --output ou ajouter --resume")
    assignments = read_assignments(splits)
    train = load_records(data, labels=True, assignments=assignments, partition="train")
    validation = load_records(data, labels=True, assignments=assignments, partition="validation")
    ids, y = arrays(train)
    strata = np.column_stack([y, ~y.any(1)])
    folds = np.full(len(y), -1, dtype=np.int8)
    splitter = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for fold, (_, held) in enumerate(splitter.split(np.zeros(len(y)), strata)):
        folds[held] = fold
    # Une modification du code ou des versions empêche de mélanger des reprises.
    source_hashes = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    provenance = {
        "dataset_sha256": sha256(data),
        "split_sha256": sha256(splits),
        "code_sha256": source_hashes,
        "versions": package_versions(),
        "fold_seed": 42,
        "folds": 3,
        "device": device,
        "train_records": len(train),
        "validation_records": len(validation),
    }
    output.mkdir(parents=True, exist_ok=True)
    reports = {}
    if reference_registry is not None:
        from .model import Tagger

        reference = read_json(reference_registry)
        for key in ("dataset_sha256", "split_sha256"):
            if key in reference and reference[key] != provenance[key]:
                raise ValueError("Les données diffèrent de la référence enregistrée : " + key)
        tagger = Tagger(reference_registry)
        name = "reference_" + tagger.name
        scores, prediction = tagger.predict(validation)
        reports[name] = save_evaluation(output / name / "validation", validation, scores, prediction)
    for name in names:
        config = deepcopy(catalog[name])
        folder = output / name
        done = folder / "complete.json"
        if resume and done.exists():
            saved = read_json(done)
            if (
                saved["provenance"] != provenance
                or saved["parameters"] != config
                or saved["metric"] != metric
                or saved["model_sha256"] != sha256(folder / "model.joblib")
            ):
                raise ValueError("Reprise incompatible : " + name)
            print(name + " : résultat existant", flush=True)
            with np.load(folder / "validation" / "predictions.npz", allow_pickle=False) as pred:
                reports[name] = evaluate(pred["y"], pred["predictions"])
        else:
            reports[name] = train_one(
                name, config, train, validation, folds, folder, provenance, metric, device
            )
    write_json(
        output / "protocol.json",
        {
            **provenance,
            "models": names,
            "selection_metric": metric,
            "test_used": False,
            "reference_registry_sha256": sha256(reference_registry) if reference_registry else None,
        },
    )
    return comparison_table(reports, output)
