"""Command line.

Typical sequence (only the raw data path has to be given)::

    python -m algo_tags preprocess --raw path/to/raw_json
    python -m algo_tags train
    python -m algo_tags evaluate --on test
    python -m algo_tags predict --input examples/problem.json

Every command logs to the console and to ``logs/algo_tags.log``.
"""

import argparse
import json
import sys
from pathlib import Path

from . import __version__, plots
from .data import load_problems, preprocess, read_split, select, targets
from .log import get_logger, setup_logging, timed
from .metrics import scores_for, table, tag_confusion
from .model import PRESETS, TagModel

DATA = Path("data/processed/problems.jsonl")
SPLIT = Path("data/splits/splits.csv")
MODEL = Path("models/final.joblib")
log = get_logger("cli")


def build_parser() -> argparse.ArgumentParser:
    """All commands and their options. Defaults follow the layout of the repository."""
    parser = argparse.ArgumentParser(prog="algo_tags",
                                     description="Predict the algorithm tags of programming problems.")
    parser.add_argument("--version", action="version", version=f"algo_tags {__version__}")
    parser.add_argument("--log-file", type=Path, default=Path("logs/algo_tags.log"),
                        help="log file (default: %(default)s)")
    cmd = parser.add_subparsers(dest="command", required=True)

    p = cmd.add_parser("preprocess", help="validate the raw JSON files and write the prepared JSONL")
    p.add_argument("--raw", type=Path, required=True, help="raw JSON file, folder of JSON files, or JSONL")
    p.add_argument("--output", type=Path, default=DATA)

    p = cmd.add_parser("split", help="build a stratified train/validation/test split (the reference split is provided)")
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--output", type=Path, required=True, help="CSV to write, e.g. data/splits/my_split.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2, help="share of all problems for test (default: 0.2)")
    p.add_argument("--validation-size", type=float, default=0.2,
                   help="share of all problems for validation (default: 0.2); train gets the rest")

    p = cmd.add_parser("train", help="train a model on one or more parts of the split")
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--splits", type=Path, default=SPLIT)
    p.add_argument("--model", choices=PRESETS, default="final")
    p.add_argument("--train-on", nargs="+", default=["train"], choices=("train", "validation"))
    p.add_argument("--output", type=Path, help="default: models/<model>.joblib")

    p = cmd.add_parser("evaluate", help="scores, tables and figures of a trained model on a part of the split")
    p.add_argument("--model", type=Path, default=MODEL)
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--splits", type=Path, default=SPLIT)
    p.add_argument("--on", default="test", choices=("train", "validation", "test"))
    p.add_argument("--output", type=Path, help="default: results/<on>_<model name>/")

    p = cmd.add_parser("predict", help="tags of a raw JSON problem, a folder of problems or a JSONL file")
    p.add_argument("--model", type=Path, default=MODEL)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, help="write one JSON line per problem instead of printing")

    p = cmd.add_parser("explain", help="features with the largest weight for each tag")
    p.add_argument("--model", type=Path, default=MODEL)
    p.add_argument("--top", type=int, default=15)

    p = cmd.add_parser("cv", help="repeated cross-validation on train + validation (model comparison)")
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--splits", type=Path, default=SPLIT)
    p.add_argument("--models", nargs="+", choices=PRESETS, default=["reference", "final"])
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--jobs", type=int, default=-1, help="parallel processes (-1 = all cores)")
    p.add_argument("--output", type=Path, default=Path("results/cv"))
    return parser


def run(args) -> None:
    """Execute the command parsed by :func:`build_parser`."""
    if args.command == "preprocess":
        summary = preprocess(args.raw, args.output)
        print(json.dumps(summary, indent=2))

    elif args.command == "split":
        from .split import make_split, write_split

        if args.output.exists():
            raise ValueError(f"{args.output} already exists; choose another --output")
        split = make_split(load_problems(args.data), args.seed, args.test_size, args.validation_size)
        write_split(split, args.output)

    elif args.command == "train":
        problems = select(load_problems(args.data), read_split(args.splits), args.train_on)
        model = TagModel.preset(args.model).fit(problems, targets(problems))
        model.save(args.output or Path("models") / f"{args.model}.joblib")

    elif args.command == "evaluate":
        model = TagModel.load(args.model)
        problems = select(load_problems(args.data), read_split(args.splits), [args.on])
        y = targets(problems)
        pred = model.predict(problems)
        result = scores_for(y, pred)
        out = args.output or Path("results") / f"{args.on}_{model.name}"
        with timed(log, f"tables and figures in {out}"):
            out.mkdir(parents=True, exist_ok=True)
            report = {"model": model.card, "split": args.on, **result}
            (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
            plots.confusion_per_tag(result, out / "confusion_per_tag.png")
            plots.tag_confusion(tag_confusion(y, pred), out / "tag_confusion.png")
            plots.per_tag_scores(result, out / "per_tag_scores.png")
        print(f"{model.name} on {args.on} ({len(problems)} problems)\n{table(result)}")

    elif args.command == "predict":
        model = TagModel.load(args.model)
        problems = load_problems(args.input)
        raws = [{k: v for k, v in p.items() if k != "targets"} for p in problems]
        answers = [model.tag_problem(p) for p in raws]
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text("".join(json.dumps(a) + "\n" for a in answers))
            log.info("%d predictions written to %s", len(answers), args.output)
        else:
            for answer in answers:
                print(json.dumps(answer, indent=2))

    elif args.command == "explain":
        print("| Tag | Features with the largest weight |\n|---|---|")
        for tag, names in TagModel.load(args.model).top_features(args.top).items():
            print(f"| {tag} | " + ", ".join(f"`{name}`" for name in names) + " |")

    elif args.command == "cv":
        from .cv import cross_validate, write_results

        problems = select(load_problems(args.data), read_split(args.splits), ["train", "validation"])
        y = targets(problems)
        results = {name: cross_validate(name, problems, y, args.repeats, jobs=args.jobs) for name in args.models}
        write_results(results, args.output)


def main(argv=None) -> int:
    """Entry point of ``python -m algo_tags`` and ``algo-tags``. Returns 0 on success, 1 on a data or input error."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_file)
    log.info("algo_tags %s: %s", __version__, " ".join(sys.argv[1:] if argv is None else argv))
    try:
        run(args)
    except (ValueError, FileNotFoundError, KeyError) as error:
        log.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
