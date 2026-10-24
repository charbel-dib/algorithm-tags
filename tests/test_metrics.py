import unittest

import numpy as np

from algo_tags.metrics import scores_for


class MetricsTest(unittest.TestCase):
    def test_hand_computed_example(self):
        y = np.zeros((4, 8), dtype=bool)
        pred = np.zeros((4, 8), dtype=bool)
        y[0, 0] = y[1, 0] = True  # two "math" problems
        pred[0, 0] = pred[2, 0] = True  # one hit, one false alarm on a problem without tags
        result = scores_for(y, pred)
        self.assertAlmostEqual(result["per_tag"]["math"]["f1"], 0.5)
        self.assertAlmostEqual(result["per_tag"]["math"]["precision"], 0.5)
        self.assertEqual(result["per_tag"]["math"]["support"], 2)
        # "no tag" state: true for rows 2 and 3, predicted for rows 1 and 3 -> F1 = 0.5
        self.assertAlmostEqual(result["f1_no_tag"], 0.5)
        self.assertAlmostEqual(result["f1_macro8"], 0.5 / 8)
        self.assertAlmostEqual(result["f1_macro9"], (0.5 + 0.5) / 9)
        self.assertEqual(result["false_alarms_no_tag"], 1)
        self.assertAlmostEqual(result["exact_match"], 0.5)

    def test_perfect_prediction(self):
        y = np.eye(8, dtype=bool)
        result = scores_for(y, y)
        self.assertEqual(result["f1_macro8"], 1.0)
        self.assertEqual(result["exact_match"], 1.0)


if __name__ == "__main__":
    unittest.main()
