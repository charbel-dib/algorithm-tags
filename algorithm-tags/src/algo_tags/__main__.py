"""Interface en ligne de commande."""

import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, "2")
    parser = argparse.ArgumentParser(
        prog="algo_tags", description="Prédire les tags d'exercices d'algorithmique"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("preprocess", help="Préparer les JSON bruts")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, default=Path("data/processed"))
    split = commands.add_parser("split", help="Créer un nouveau découpage 60/20/20")
    split.add_argument("--data", type=Path, required=True)
    split.add_argument("--output", type=Path, default=Path("data/splits"))
    split.add_argument("--seed", type=int, default=42)
    sample = commands.add_parser("sample", help="Exporter un exemple réel du test pour la démonstration")
    sample.add_argument("--data", type=Path, required=True)
    sample.add_argument("--splits", type=Path, required=True)
    sample.add_argument("--output", type=Path, default=Path("examples/sample_0.json"))
    predict = commands.add_parser("predict", help="Prédire un JSON, un dossier ou un JSONL")
    evaluate = commands.add_parser("evaluate", help="Évaluer un modèle sur des exemples annotés")
    for command in (predict, evaluate):
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--registry", type=Path, default=Path("models/registry.json"))
        command.add_argument("--model", help="Par défaut : modèle indiqué dans le registre")
    predict.add_argument("--output", type=Path, help="Fichier JSON de prédictions ; sinon sortie standard")
    evaluate.add_argument("--output", type=Path, default=Path("runs/evaluation"))
    evaluate.add_argument("--splits", type=Path)
    evaluate.add_argument("--split", choices=("train", "validation", "test"))
    train = commands.add_parser("train", help="Entraîner un modèle et l'évaluer sur validation")
    experiments = commands.add_parser("experiment", help="Comparer les variantes de texte sur validation")
    for command in (train, experiments):
        command.add_argument("--data", type=Path, default=Path("data/processed/dataset.jsonl"))
        command.add_argument("--splits", type=Path, default=Path("data/splits/splits.csv"))
        command.add_argument("--config", type=Path, default=Path("configs/models.json"))
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        command.add_argument("--metric", choices=("f1_macro", "f1_macro9"), default="f1_macro9")
        command.add_argument("--resume", action="store_true")
    train.add_argument("--model", required=True)
    train.add_argument("--output", type=Path, required=True)
    experiments.add_argument(
        "--models", nargs="+", default=["tfidf_svm", "tfidf_manual_rules", "tfidf_stopwords"]
    )
    experiments.add_argument("--output", type=Path, default=Path("runs/text_variants"))
    experiments.add_argument("--reference-registry", type=Path, help="Ajouter un modèle déjà entraîné à comparer")
    results = commands.add_parser("results", help="Afficher les résultats enregistrés")
    results.add_argument("--results", type=Path, default=Path("results"))
    results.add_argument("--split", choices=("validation", "test"), default="test")
    verify = commands.add_parser("verify", help="Vérifier les modèles et les résultats enregistrés")
    verify.add_argument("--results", type=Path, default=Path("results"))
    verify.add_argument("--registry", type=Path, default=Path("models/registry.json"))
    args = parser.parse_args(argv)
    try:
        if args.command == "preprocess":
            from .data import preprocess

            summary = preprocess(args.input, args.output)
            print(f"{summary['records_written']} exercices ; {summary['records_without_target']} sans cible")
        elif args.command == "split":
            from .split import create_split

            summary = create_split(args.data, args.output, args.seed)
            for name, values in summary["splits"].items():
                print(f"{name}: {values['records']} exemples, {values['without_target']} sans cible")
        elif args.command == "sample":
            from .inputs import load_records, read_assignments
            from .storage import write_json

            if args.output.exists():
                raise ValueError("L'exemple existe déjà ; le conserver ou choisir un autre --output")
            rows = load_records(args.data, assignments=read_assignments(args.splits), partition="test")
            write_json(args.output, rows[0])
            print(f"Exemple test {rows[0]['src_uid']} exporté dans {args.output}")
        elif args.command in ("train", "experiment"):
            from .training import run_experiments

            names = [args.model] if args.command == "train" else args.models
            print(
                run_experiments(
                    args.data,
                    args.splits,
                    args.config,
                    names,
                    args.output,
                    metric=args.metric,
                    device=args.device,
                    resume=args.resume,
                    reference_registry=args.reference_registry if args.command == "experiment" else None,
                )
            )
        elif args.command == "results":
            from .reporting import stored_table

            print(stored_table(args.results, args.split))
        elif args.command == "verify":
            from .reporting import verify_results
            from .storage import read_json, sha256

            registry = read_json(args.registry)
            installed, missing = 0, 0
            for name, model in registry["models"].items():
                if "path" not in model:
                    continue
                path = (args.registry.parent / model["path"]).resolve()
                if not path.is_relative_to(args.registry.parent.resolve()):
                    raise ValueError("Chemin du modèle hors du registre")
                if not path.exists():
                    missing += 1
                    continue
                if sha256(path) != model["sha256"]:
                    raise ValueError("Modèle modifié : " + name)
                installed += 1
            print(verify_results(args.results, args.registry))
            print(f"Poids : {installed} fichier(s) vérifié(s), {missing} absent(s), facultatifs pour ce contrôle.")
        else:
            from .inputs import load_records, read_assignments
            from .model import Tagger
            from .storage import write_json

            tagger = Tagger(args.registry, args.model)
            if args.command == "predict":
                rows = load_records(args.input)
                records = tagger.records(rows)
                value = records[0] if len(records) == 1 else records
                if args.output:
                    write_json(args.output, value)
                else:
                    print(json.dumps(value, ensure_ascii=False, indent=2))
            else:
                from .reporting import save_evaluation

                if args.output.exists() and any(args.output.iterdir()):
                    raise ValueError("Le dossier d'évaluation doit être vide")
                assignments = read_assignments(args.splits) if args.splits else None
                rows = load_records(args.input, labels=True, assignments=assignments, partition=args.split)
                scores, predictions = tagger.predict(rows)
                report = save_evaluation(args.output, rows, scores, predictions)
                print(json.dumps(report, ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError, ModuleNotFoundError) as error:
        parser.exit(2, f"Erreur : {error}\n")


if __name__ == "__main__":
    main()
