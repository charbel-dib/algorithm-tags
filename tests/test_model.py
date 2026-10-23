"""Lecture d'entrées non annotées et cohérence des résultats publiés."""

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from algo_tags.data import TARGET_TAGS
from algo_tags.inputs import load_records
from algo_tags.model import Tagger
from algo_tags.reporting import verify_results
from algo_tags.storage import read_json, sha256, write_json
from algo_tags.training import fit_model

ROOT = Path(__file__).resolve().parents[1]


class ModelTests(unittest.TestCase):
    def test_selected_threshold_changes_decisions_without_changing_scores(self):
        selected = read_json(ROOT / "models/registry.json")["models"]["tfidf_svm"]
        reference = read_json(ROOT / "models/reference_macro8.json")["models"]["tfidf_svm"]
        with np.load(ROOT / "results/test/tfidf_svm/predictions.npz", allow_pickle=False) as saved:
            scores, prediction = saved["scores"], saved["predictions"]
        with np.load(ROOT / "results/thresholds_macro8/svm_macro8/test/predictions.npz",
                     allow_pickle=False) as saved:
            np.testing.assert_array_equal(scores, saved["scores"])
            reference_prediction = saved["predictions"]
        np.testing.assert_array_equal(prediction, scores >= selected["thresholds"])
        np.testing.assert_array_equal(reference_prediction, scores >= reference["thresholds"])
        self.assertTrue((np.array(selected["thresholds"]) > reference["thresholds"]).all())
        self.assertTrue((prediction <= reference_prediction).all())
        self.assertFalse(np.array_equal(prediction, reference_prediction))

    def test_published_metrics_match_predictions(self):
        self.assertIn("16 évaluations", verify_results(ROOT / "results", ROOT / "models/registry.json"))

    def test_prediction_needs_neither_annotation_nor_source_code(self):
        rows = [{"src_uid": str(i), "inputs": {"statement": f"group{i % 4} common text"},
                 "targets": [int((i + j) % 4 == 0) for j in range(8)]} for i in range(16)]
        bundle, _ = fit_model(rows, {"model": "linear_svm", "min_df": 1, "C": 1.0}, "cpu")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            joblib.dump(bundle, root / "model.joblib")
            write_json(root / "registry.json", {
                "target_order": list(TARGET_TAGS), "default_model": "svm",
                "models": {"svm": {"path": "model.joblib", "sha256": sha256(root / "model.joblib"),
                                   "thresholds": [0.0] * 8, "score_kind": "margin"}}})
            write_json(root / "input.json", {"prob_desc_description": "group0 common text"})
            record = Tagger(root / "registry.json").records(load_records(root / "input.json"))[0]
            self.assertEqual(record["model"], "svm")
            self.assertEqual(record["score_kind"], "margin")
            self.assertEqual(len(record["scores"]), 8)

    def test_missing_optional_weights_give_actionable_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "registry.json"
            write_json(path, read_json(ROOT / "models/registry.json"))
            with self.assertRaisesRegex(ValueError, "Poids absents.*commande train"):
                Tagger(path)

    def test_rules_can_return_no_tags(self):
        model = Tagger(ROOT / "models/registry.json", "rules")
        rows = [{"src_uid": "x", "inputs": {"statement": "Sort the values."}}]
        self.assertEqual(model.records(rows)[0]["tags"], [])

    def test_missing_annotation_is_not_treated_as_negative(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "problem.json"
            write_json(path, {"src_uid": "x", "prob_desc_description": "Sort the values."})
            self.assertEqual(len(load_records(path)), 1)
            with self.assertRaises(ValueError):
                load_records(path, labels=True)

    def test_bad_model_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bad.joblib").write_bytes(b"not a model")
            from algo_tags.data import TARGET_TAGS

            write_json(
                root / "registry.json",
                {
                    "target_order": list(TARGET_TAGS),
                    "default_model": "x",
                    "models": {"x": {"path": "bad.joblib", "sha256": "wrong", "score_kind": "margin"}},
                },
            )
            with self.assertRaisesRegex(ValueError, "Empreinte"):
                Tagger(root / "registry.json")
