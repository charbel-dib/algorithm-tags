"""Command line: train, evaluate, predict, explain, cross-validate, split."""

import argparse
import json
from pathlib import Path

from .data import TAGS, load_problems, read_split, select, targets
from .metrics import scores_for, table
from .model import PRESETS, TagModel

SPLIT = Path("data/splits/splits.csv")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="algo_tags", description="Predict algorithm tags of programming problems")
    cmd = parser.add_subparsers(dest="command", required=True)

    p = cmd.add_parser("train", help="train a model on one or more parts of the split")
    p.add_argument("--data", type=Path, required=True, help="folder of raw JSON problems")
    p.add_argument("--splits", type=Path, default=SPLIT)
    p.add_argument("--model", choices=PRESETS, default="final")
    p.add_argument("--train-on", nargs="+", default=["train"], choices=("train", "validation"))
    p.add_argument("--output", type=Path, default=Path("models/final.joblib"))

    p = cmd.add_parser("evaluate", help="scores of a trained model on a part of the split")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--splits", type=Path, default=SPLIT)
    p.add_argument("--on", default="test", choices=("train", "validation", "test"))
    p.add_argument("--output", type=Path, help="optional JSON file with all the scores")

    p = cmd.add_parser("predict", help="tags of one JSON problem or a folder of problems")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)

    p = cmd.add_parser("explain", help="features with the largest weight for each tag")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--top", type=int, default=15)

    p = cmd.add_parser("cv", help="repeated cross-validation on train + validation (model comparison)")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--splits", type=Path, default=SPLIT)
    p.add_argument("--models", nargs="+", choices=PRESETS, default=["reference", "final"])
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--output", type=Path, default=Path("results"))

    p = cmd.add_parser("split", help="rebuild the stratified 60/20/20 split")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, default=SPLIT)
    p.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)

    if args.command == "train":
        problems = select(load_problems(args.data), read_split(args.splits), args.train_on)
        model = TagModel.preset(args.model).fit(problems, targets(problems))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.output)
        print(f"{args.model}: trained on {len(problems)} problems, threshold {model.threshold:.4f} -> {args.output}")

    elif args.command == "evaluate":
        problems = select(load_problems(args.data), read_split(args.splits), [args.on])
        result = scores_for(targets(problems), TagModel.load(args.model).predict(problems))
        print(f"{args.model} on {args.on} ({len(problems)} problems)\n{table(result)}")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")

    elif args.command == "predict":
        model = TagModel.load(args.model)
        problems = load_problems(args.input)
        for problem, scores in zip(problems, model.scores(problems)):
            print(json.dumps({
                "src_uid": problem["src_uid"],
                "tags": [t for t, s in zip(TAGS, scores) if s >= model.threshold],
                "margins": {t: round(float(s), 3) for t, s in zip(TAGS, scores)},
                "threshold": round(model.threshold, 4),
            }, indent=2))

    elif args.command == "explain":
        print("| Tag | Features with the largest weight |\n|---|---|")
        for tag, names in TagModel.load(args.model).top_features(args.top).items():
            print(f"| {tag} | " + ", ".join(f"`{name}`" for name in names) + " |")

    elif args.command == "cv":
        from .cv import cross_validate, write_results

        problems = select(load_problems(args.data), read_split(args.splits), ["train", "validation"])
        results = {}
        for name in args.models:
            results[name] = cross_validate(name, problems, targets(problems), args.repeats, jobs=args.jobs)
            macro9 = [100 * r["f1_macro9"] for r in results[name]]
            print(f"{name:15s} macro9 {sum(macro9) / len(macro9):.2f}  per repetition {[round(m, 2) for m in macro9]}")
        write_results(results, args.output)

    elif args.command == "split":
        from .split import make_split, write_split

        if args.output.exists():
            parser.error(f"{args.output} exists; choose another --output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_split(make_split(load_problems(args.data), args.seed), args.output)
        print(f"split written to {args.output}")


if __name__ == "__main__":
    main()
