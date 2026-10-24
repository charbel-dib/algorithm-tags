import csv
import json
import logging
import os
import tempfile
import unittest
from pathlib import Path

from algo_tags.__main__ import main

from helpers import raw_problems


class CommandLineTest(unittest.TestCase):
    """The whole sequence a new user runs, on synthetic data: preprocess, train, evaluate, predict, explain, cv."""

    def test_full_sequence(self):
        with tempfile.TemporaryDirectory() as folder:
            cwd = os.getcwd()
            os.chdir(folder)
            try:
                Path("raw").mkdir()
                raws = raw_problems(80)
                for i, raw in enumerate(raws):
                    Path("raw", f"sample_{i}.json").write_text(json.dumps(raw))
                Path("data/splits").mkdir(parents=True)
                with open("data/splits/splits.csv", "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["src_uid", "split"])
                    w.writerows((r["src_uid"], "test" if i % 5 == 0 else "validation" if i % 5 == 1 else "train")
                                for i, r in enumerate(raws))
                self.assertEqual(main(["preprocess", "--raw", "raw"]), 0)
                self.assertEqual(main(["train", "--train-on", "train", "validation"]), 0)
                self.assertTrue(Path("models/final.joblib").exists())
                self.assertEqual(main(["evaluate", "--on", "test"]), 0)
                for figure in ("confusion_per_tag.png", "tag_confusion.png", "per_tag_scores.png"):
                    self.assertTrue(Path("results/test_final", figure).exists())
                self.assertEqual(main(["predict", "--input", "raw/sample_0.json", "--output", "out/pred.jsonl"]), 0)
                self.assertIn("tags", json.loads(Path("out/pred.jsonl").read_text().splitlines()[0]))
                self.assertEqual(main(["explain", "--top", "3"]), 0)
                self.assertEqual(main(["cv", "--models", "reference", "--repeats", "1", "--jobs", "1"]), 0)
                self.assertTrue(Path("results/cv/cv_ablation.png").exists())
                self.assertTrue(Path("logs/algo_tags.log").read_text())
                self.assertEqual(main(["evaluate", "--model", "missing.joblib"]), 1)  # clean error, no traceback
            finally:
                for handler in logging.getLogger("algo_tags").handlers:  # release the log file (Windows)
                    handler.close()
                logging.getLogger("algo_tags").handlers.clear()
                os.chdir(cwd)

    def test_unwritable_log_file_does_not_stop_the_command(self):
        with tempfile.TemporaryDirectory() as folder:
            blocker = Path(folder, "not_a_folder")
            blocker.write_text("")  # a file where the log folder should be: mkdir fails
            missing_model = str(Path(folder, "missing.joblib"))  # the command fails on its input, not on the log
            log_file = str(blocker / "algo_tags.log")
            self.assertEqual(main(["--log-file", log_file, "explain", "--model", missing_model]), 1)
            for handler in logging.getLogger("algo_tags").handlers:
                handler.close()


    def test_split_rejects_impossible_sizes(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder, "problems.jsonl")
            data.write_text("".join(json.dumps(r) + "\n" for r in raw_problems(10)))
            base = ["--log-file", str(Path(folder, "log.txt")), "split", "--data", str(data),
                    "--output", str(Path(folder, "split.csv"))]
            for sizes in (["--test-size", "0"], ["--validation-size", "1.2"],
                          ["--test-size", "0.5", "--validation-size", "0.5"]):
                self.assertEqual(main(base + sizes), 1)  # clean error before any split is drawn
            for handler in logging.getLogger("algo_tags").handlers:
                handler.close()


if __name__ == "__main__":
    unittest.main()
