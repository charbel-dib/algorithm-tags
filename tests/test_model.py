import tempfile
import unittest
from pathlib import Path

import numpy as np

from algo_tags.data import TAGS, targets, to_problem
from algo_tags.metrics import scores_for
from algo_tags.model import PRESETS, TagModel

from helpers import raw_problems


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problems = [to_problem(raw) for raw in raw_problems(80)]
        cls.y = targets(cls.problems)
        cls.model = TagModel.preset("final").fit(cls.problems, cls.y)

    def test_learns_the_synthetic_topics(self):
        pred = self.model.predict(self.problems)
        self.assertEqual(pred.shape, (80, len(TAGS)))
        self.assertEqual(scores_for(self.y, pred)["f1_macro8"], 1.0)

    def test_threshold_is_chosen_out_of_fold(self):
        self.assertTrue(np.isfinite(self.model.threshold))

    def test_save_and_load_give_identical_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "model.joblib")
            self.model.save(path)
            loaded = TagModel.load(path)
        np.testing.assert_allclose(loaded.scores(self.problems), self.model.scores(self.problems))
        self.assertEqual(loaded.threshold, self.model.threshold)

    def test_top_features_name_the_blocks(self):
        top = self.model.top_features(5)
        self.assertEqual(list(top), list(TAGS))
        self.assertTrue(all(":" in name for names in top.values() for name in names))

    def test_a_tag_without_positive_example_is_rejected(self):
        with self.assertRaises(ValueError):
            TagModel.preset("reference").fit(self.problems, self.y & False)

    def test_every_preset_is_buildable(self):
        for name in PRESETS:
            self.assertIsInstance(TagModel.preset(name), TagModel)


if __name__ == "__main__":
    unittest.main()
