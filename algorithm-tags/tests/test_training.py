"""Petits jeux synthétiques : absence de fuite, reprise et utilisation du modèle exporté."""

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from algo_tags.__main__ import main
from algo_tags.data import TARGET_TAGS
from algo_tags.model import Tagger
from algo_tags.storage import read_json, write_json
from algo_tags.training import fit_model, run_experiments, tune_threshold


class TrainingTests(unittest.TestCase):
    def test_cli_from_raw_json_to_prediction_without_bundled_weights(self):
        with tempfile.TemporaryDirectory(prefix="tag pipeline ") as temporary:
            root = Path(temporary)
            raw, prepared, splits, run = (root / n for n in ("raw inputs", "prepared", "splits", "run"))
            raw.mkdir()
            for i in range(64):
                write_json(raw / f"{i:03}.json", {
                    "src_uid": str(i), "code_uid": f"code-{i}", "source_code": "print(1)",
                    "prob_desc_description": f"group{i % 4} common text",
                    "tags": [tag for j, tag in enumerate(TARGET_TAGS) if (i + j) % 4 == 0]})
            with redirect_stdout(io.StringIO()):
                main(["preprocess", "--input", str(raw), "--output", str(prepared)])
                dataset = str(prepared / "dataset.jsonl")
                main(["split", "--data", dataset, "--output", str(splits)])
                assignments = str(splits / "splits.csv")
                main(["train", "--data", dataset, "--splits", assignments,
                      "--config", str(Path(__file__).resolve().parents[1] / "configs/models.json"),
                      "--model", "tfidf_svm", "--output", str(run)])
                registry = str(run / "tfidf_svm/registry.json")
                main(["evaluate", "--input", dataset, "--splits", assignments, "--split", "test",
                      "--registry", registry, "--output", str(root / "test")])
                sample = str(root / "sample.json")
                main(["sample", "--data", dataset, "--splits", assignments, "--output", sample])
                main(["predict", "--registry", registry, "--input", sample,
                      "--output", str(root / "prediction.json")])
            sample_uid = read_json(sample)["src_uid"]
            with Path(assignments).open(newline="") as stream:
                parts = {row["src_uid"]: row["split"] for row in csv.DictReader(stream)}
            self.assertEqual(parts[sample_uid], "test")
            self.assertEqual(read_json(root / "prediction.json")["src_uid"], sample_uid)
            self.assertTrue((root / "test/metrics.json").is_file())
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(["sample", "--data", dataset, "--splits", assignments, "--output", sample])
            self.assertEqual(error.exception.code, 2)

    def test_threshold_can_abstain_and_ties_prefer_natural(self):
        y = np.zeros((4, 8), dtype=np.int8)
        scores = np.full((4, 8), 0.8)
        thresholds = tune_threshold(y, scores, "probability")
        self.assertFalse((scores >= thresholds).any())
        y[:, 0] = 1
        scores[:, 0] = 0.9
        self.assertEqual(tune_threshold(y, scores, "probability", "f1_macro"), [0.5] * 8)

    def test_training_excludes_validation_and_test_and_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, split, config = (root / name for name in ("data.jsonl", "splits.csv", "config.json"))
            rows, assignments = [], []
            for i in range(48):
                part = "train" if i < 32 else "validation" if i < 40 else "test"
                y = [int((i + j) % 4 == 0) for j in range(8)]
                rows.append(
                    {"src_uid": str(i), "inputs": {"statement": f"group{i % 4} common text"}, "targets": y}
                )
                assignments.append([str(i), part])
            data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with split.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["src_uid", "split"])
                writer.writerows(assignments)
            base = {"model": "linear_svm", "min_df": 1, "C": 1.0, "seed": 42}
            write_json(
                config,
                {
                    "svm": base,
                    "rules": {**base, "manual_rules": True},
                    "stopwords": {**base, "stopwords": True},
                },
            )
            names = ["svm", "rules", "stopwords"]
            fit_ids = []

            def observed_fit(learning_rows, parameters, device):
                fit_ids.append({row["src_uid"] for row in learning_rows})
                return fit_model(learning_rows, parameters, device)

            with patch("algo_tags.training.fit_model", side_effect=observed_fit):
                run_experiments(data, split, config, names, root / "run")
            self.assertEqual(len(fit_ids), 12)
            self.assertEqual(fit_ids[-1], {str(i) for i in range(32)})
            self.assertTrue(all(ids <= fit_ids[-1] for ids in fit_ids))
            with np.load(root / "run/svm/oof.npz", allow_pickle=False) as saved:
                self.assertEqual(set(saved["ids"]), fit_ids[-1])
                self.assertEqual(set(saved["folds"]), {0, 1, 2})
            model = Tagger(root / "run/svm/registry.json")
            self.assertEqual(model.predict(rows[32:40])[1].shape, (8, 8))
            with patch("algo_tags.training.fit_model", side_effect=AssertionError("retraining")):
                run_experiments(data, split, config, names, root / "run", resume=True)
            metadata = read_json(root / "run/svm/metadata.json")
            self.assertEqual(set(metadata["fit_ids"]), fit_ids[-1])
            data.write_text(data.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Reprise incompatible"):
                run_experiments(data, split, config, ["svm"], root / "run", resume=True)
