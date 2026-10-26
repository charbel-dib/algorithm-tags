import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from sklearn.model_selection import KFold

from algo_tags.data import TAGS, targets, to_problem
from algo_tags.metrics import scores_for
from algo_tags.model import PRESETS, TagModel, best_threshold

from helpers import raw_problem, raw_problems


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
        seen = {}

        def spy(y, scores):
            seen["scores"] = scores.copy()
            return best_threshold(y, scores)

        with mock.patch("algo_tags.model.best_threshold", side_effect=spy):
            model = TagModel.preset("final").fit(self.problems, self.y)
        expected = np.zeros(self.y.shape)
        for train, held_out in KFold(3, shuffle=True, random_state=42).split(self.problems):
            part = TagModel(model.blocks, model.C)._fit_svms([self.problems[i] for i in train], self.y[train])
            expected[held_out] = part.scores([self.problems[i] for i in held_out])
        # the threshold is chosen on margins of problems the scoring model did not see...
        np.testing.assert_allclose(seen["scores"], expected)
        # ...which differ from the margins of the final model on its own training problems
        self.assertFalse(np.allclose(seen["scores"], model.scores(self.problems)))
        self.assertEqual(model.threshold, best_threshold(self.y, expected))

    def test_save_and_load_give_identical_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "model.joblib")
            self.model.save(path)
            loaded = TagModel.load(path)
            card = json.loads(path.with_suffix(".json").read_text())
        self.assertEqual(card["model"], "final")
        self.assertEqual(card["trained_on"], 80)
        np.testing.assert_allclose(loaded.scores(self.problems), self.model.scores(self.problems))
        self.assertEqual(loaded.threshold, self.model.threshold)

    def test_tag_reads_a_raw_problem_and_ignores_its_tags(self):
        raw = raw_problem(999, "geometry")
        raw["tags"] = ["strings"]
        answer = self.model.tag(raw)
        self.assertEqual(answer["tags"], ["geometry"])
        self.assertEqual(set(answer["margins"]), set(TAGS))
        self.assertGreater(answer["latency_ms"], 0)

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
