import importlib.util
import unittest

from algo_tags.data import TAGS, to_problem

from helpers import raw_problems

HAS_ITERSTRAT = importlib.util.find_spec("iterstrat") is not None


@unittest.skipUnless(HAS_ITERSTRAT, "iterative-stratification is not installed (extra [split])")
class SplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from algo_tags.split import make_split

        cls.make_split = staticmethod(make_split)
        cls.problems = sorted((to_problem(raw) for raw in raw_problems(200)), key=lambda p: p["src_uid"])

    def test_every_problem_is_in_exactly_one_part(self):
        split = self.make_split(self.problems)
        self.assertEqual(set(split), {p["src_uid"] for p in self.problems})
        self.assertEqual(set(split.values()), {"train", "validation", "test"})

    def test_same_seed_gives_the_same_split(self):
        self.assertEqual(self.make_split(self.problems, seed=3), self.make_split(self.problems, seed=3))
        self.assertNotEqual(self.make_split(self.problems, seed=3), self.make_split(self.problems, seed=4))

    def test_sizes_and_tag_proportions_follow_the_request(self):
        split = self.make_split(self.problems, test_size=0.15, validation_size=0.15)
        sizes = {part: sum(v == part for v in split.values()) for part in ("train", "validation", "test")}
        self.assertAlmostEqual(sizes["test"] / len(self.problems), 0.15, delta=0.03)
        self.assertAlmostEqual(sizes["validation"] / len(self.problems), 0.15, delta=0.03)
        for j, tag in enumerate(TAGS):  # every tag is present in every part
            for part in sizes:
                self.assertTrue(any(split[p["src_uid"]] == part and p["targets"][j] for p in self.problems),
                                (tag, part))


if __name__ == "__main__":
    unittest.main()
