import unittest

import numpy as np

from algo_tags.metrics import scores_for, tag_confusion


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

    def test_tag_confusion_shows_which_tag_is_taken_for_which(self):
        y = np.zeros((2, 8), dtype=bool)
        pred = np.zeros((2, 8), dtype=bool)
        y[0, 4] = True  # a "trees" problem...
        pred[0, 1] = True  # ...predicted "graphs"
        m = tag_confusion(y, pred)
        self.assertEqual(m.shape, (9, 9))
        self.assertEqual(m[4, 1], 1.0)  # true trees -> predicted graphs
        self.assertEqual(m[8, 8], 1.0)  # the empty problem stays empty

    def test_perfect_prediction(self):
        y = np.eye(8, dtype=bool)
        result = scores_for(y, y)
        self.assertEqual(result["f1_macro8"], 1.0)
        self.assertEqual(result["exact_match"], 1.0)


if __name__ == "__main__":
    unittest.main()
