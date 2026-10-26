import json
import tempfile
import unittest
from pathlib import Path

from algo_tags.data import TAGS, load_problems, preprocess, select, targets, to_problem

from helpers import raw_problem, raw_problems


class DataTest(unittest.TestCase):
    def test_statement_keeps_sections_in_order_and_skips_empty_ones(self):
        problem = to_problem(raw_problem(0, "graphs"))
        self.assertTrue(problem["statement"].startswith("[DESCRIPTION]\nGiven an undirected graph"))
        self.assertIn("[INPUT]", problem["statement"])
        self.assertNotIn("[NOTES]", problem["statement"])
        self.assertEqual(problem["sample_in"], "3\n1 2\n2 3")

    def test_targets_ignore_out_of_scope_tags(self):
        raw = raw_problem(0, "strings")
        self.assertEqual(to_problem(raw)["targets"], [int(t == "strings") for t in TAGS])
        raw["tags"] = ["greedy"]
        self.assertEqual(to_problem(raw)["targets"], [0] * 8)
        raw["tags"] = []
        self.assertEqual(to_problem(raw)["targets"], [0] * 8)

    def test_prediction_input_needs_no_tags(self):
        raw = raw_problem(0, "math")
        del raw["tags"]
        raw["difficulty"] = None
        problem = to_problem(raw)
        self.assertNotIn("targets", problem)
        self.assertIsNone(problem["difficulty"])

    def test_missing_description_is_rejected(self):
        with self.assertRaises(ValueError):
            to_problem({"prob_desc_description": "  "})

    def test_folder_loading_and_split_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            for i, tag in enumerate(["math", "graphs", "strings"]):
                Path(folder, f"sample_{2 - i}.json").write_text(json.dumps(raw_problem(i, tag)))
            problems = load_problems(folder)
        self.assertEqual([p["src_uid"] for p in problems], ["p000", "p001", "p002"])
        self.assertEqual(targets(problems).shape, (3, 8))
        split = {"p000": "train", "p001": "validation", "p002": "test"}
        self.assertEqual([p["src_uid"] for p in select(problems, split, ["train", "validation"])], ["p000", "p001"])
        with self.assertRaises(ValueError):
            select(problems, {"p000": "train"}, ["train"])

    def test_preprocess_writes_a_jsonl_that_loads_back_identically(self):
        with tempfile.TemporaryDirectory() as folder:
            raw_dir = Path(folder, "raw")
            raw_dir.mkdir()
            for i, raw in enumerate(raw_problems(16)):
                Path(raw_dir, f"sample_{i}.json").write_text(json.dumps(raw))
            out = Path(folder, "processed", "problems.jsonl")
            summary = preprocess(raw_dir, out)
            self.assertEqual(summary["problems"], 16)
            self.assertTrue(out.with_name("problems_summary.json").exists())
            self.assertEqual(load_problems(out), load_problems(raw_dir))
            # the prepared folder (JSONL + summary) can be read back as a whole
            self.assertEqual(load_problems(out.parent), load_problems(out))


if __name__ == "__main__":
    unittest.main()
