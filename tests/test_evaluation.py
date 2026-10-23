"""Métriques calculées à la main et règles retenues."""

import unittest

import numpy as np

from algo_tags.baselines import predict_rules
from algo_tags.evaluation import evaluate


class MetricsTests(unittest.TestCase):
    def test_hand_calculated_metrics(self):
        truth = np.zeros((4, 8), dtype=int)
        predicted = np.zeros_like(truth)
        truth[0, 0] = truth[1, 0] = truth[2, 1] = 1
        predicted[0, 0] = predicted[2, 0] = predicted[2, 1] = predicted[3, 1] = 1
        report = evaluate(truth, predicted)
        math = report["per_label"]["math"]
        self.assertEqual([math[k] for k in ("tn", "fp", "fn", "tp")], [1, 1, 1, 1])
        self.assertAlmostEqual(report["per_label"]["graphs"]["f1"], 2 / 3)
        self.assertAlmostEqual(report["overall"]["f1_macro"], (0.5 + 2 / 3) / 8)
        self.assertAlmostEqual(report["overall"]["precision_micro"], 2 / 4)
        self.assertAlmostEqual(report["overall"]["recall_micro"], 2 / 3)
        self.assertAlmostEqual(report["overall"]["f1_micro"], 4 / 7)
        self.assertEqual(report["overall"]["exact_match"], 0.25)
        self.assertEqual(report["overall"]["without_target_false_positive_rate"], 1)

    def test_empty_annotations_are_scored_as_negative(self):
        report = evaluate([[0] * 8, [1] + [0] * 7], [[0] * 8] * 2)
        self.assertEqual(report["overall"]["f1_macro"], 0)
        self.assertEqual(report["overall"]["exact_match"], 0.5)
        self.assertEqual(report["overall"]["without_target_false_positive_rate"], 0)
        report = evaluate([[1] * 8], [[0] * 8])
        self.assertIsNone(report["overall"]["without_target_false_positive_rate"])

    def test_invalid_predictions_are_rejected(self):
        for invalid in ([[0] * 7], [[0.3] * 8], [[float("nan")] * 8], [], [[0] * 8] * 2):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                evaluate([[0] * 8], invalid)

    def test_rules_use_boundaries_and_allow_multiple_or_no_tags(self):
        predicted, matches = predict_rules(
            [
                "A TREE has vertices. Find the probability.",
                "The mainstream paragraph describes a street.",
                "Players take\nturns.",
            ]
        )
        self.assertEqual(predicted[0].tolist(), [0, 1, 0, 0, 1, 0, 0, 1])
        self.assertEqual(predicted[1].sum(), 0)
        self.assertEqual(predicted[2, 6], 1)
        self.assertEqual(matches[0]["trees"], ["tree"])


if __name__ == "__main__":
    unittest.main()
