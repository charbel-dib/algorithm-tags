"""Tests des choix de préparation qui influencent les données d'apprentissage."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from algo_tags.data import TARGET_TAGS, encode_tags, prepare_inputs, prepare_record, preprocess


def example(uid="problem-1", tags=None):
    return {
        "src_uid": uid,
        "code_uid": f"code-{uid}",
        "tags": ["math"] if tags is None else tags,
        "prob_desc_description": "  Cafe\u0301 : $$$a_1 <= 2$$$\r\nKeep  two spaces.  ",
        "prob_desc_input_spec": "  integer n  ",
        "prob_desc_output_spec": None,
        "prob_desc_notes": None,
        "source_code": "\r\nif n:\r\n\tprint(n)  # keep comment\r\n\r\n",
        "difficulty": -1,
        "lang": "Python 2",
    }


class LabelsAndTextTests(unittest.TestCase):
    def test_empty_and_outside_targets_have_same_vector(self):
        self.assertEqual(encode_tags([]), [0] * 8)
        self.assertEqual(encode_tags(["greedy", "implementation"]), [0] * 8)

    def test_order_duplicates_and_exact_matching(self):
        self.assertEqual(encode_tags(["trees", "math", "math", "greedy"]), [1, 0, 0, 0, 1, 0, 0, 0])
        self.assertEqual(encode_tags(["Math", " math"]), [0] * 8)

    def test_malformed_labels_are_not_negative_examples(self):
        for tags in (None, "math", {}, [None], [" "], [1]):
            with self.subTest(tags=tags), self.assertRaises(ValueError):
                encode_tags(tags)

    def test_inputs_preserve_math_code_and_unicode(self):
        record = example()
        before = copy.deepcopy(record)
        inputs = prepare_inputs(record)
        self.assertEqual(inputs["description"], "Café : $$$a_1 <= 2$$$\nKeep  two spaces.")
        self.assertEqual(inputs["source_code"], "\nif n:\n\tprint(n)  # keep comment\n\n")
        self.assertEqual(inputs["output_spec"], "")
        self.assertEqual(inputs["notes"], "")
        self.assertEqual(
            inputs["statement"],
            "[DESCRIPTION]\nCafé : $$$a_1 <= 2$$$\nKeep  two spaces.\n\n[INPUT]\ninteger n",
        )
        self.assertEqual(record, before)

    def test_inputs_do_not_read_targets_or_metadata(self):
        a = example()
        b = dict(a, tags=["graphs"], difficulty=3500, lang="PyPy 3")
        del b["tags"]
        self.assertEqual(prepare_inputs(a), prepare_inputs(b))

    def test_missing_and_wrong_type_fields(self):
        for field, value in (
            ("src_uid", ""),
            ("code_uid", 3),
            ("source_code", None),
            ("prob_desc_description", "  "),
            ("prob_desc_notes", []),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                prepare_record(dict(example(), **{field: value}), source_file="a.json", source_sha256="x")


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw"
        self.output = self.root / "processed"
        self.raw.mkdir()
        for name, row in (
            ("z.json", example("p3", ["math", "math", "graphs"])),
            ("a.json", example("p1", [])),
            ("m.json", example("p2", ["greedy"])),
        ):
            (self.raw / name).write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def test_counts_traceability_and_reproducibility(self):
        raw_before = {p.name: p.read_bytes() for p in self.raw.iterdir()}
        summary = preprocess(self.raw, self.output)
        self.assertEqual(summary["records_written"], 3)
        self.assertEqual(summary["records_without_target"], 2)
        self.assertEqual(summary["records_with_repeated_tags"], 1)
        self.assertEqual(
            summary["annotation_counts"], {"empty_original": 1, "outside_targets": 1, "with_target": 1}
        )
        self.assertEqual(
            summary["positive_counts"], {tag: int(tag in ("math", "graphs")) for tag in TARGET_TAGS}
        )
        rows = [
            json.loads(line)
            for line in (self.output / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual([r["src_uid"] for r in rows], ["p1", "p2", "p3"])
        self.assertEqual(rows[0]["metadata"]["difficulty"], -1)
        self.assertNotIn("tags", rows[0])
        self.assertNotIn("difficulty", rows[0]["inputs"])
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        self.assertEqual(preprocess(self.raw, self.output), summary)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir()})
        self.assertEqual(raw_before, {p.name: p.read_bytes() for p in self.raw.iterdir()})

    def test_validation_error_keeps_existing_outputs(self):
        preprocess(self.raw, self.output)
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        (self.raw / "broken.json").write_text("{bad", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "broken.json"):
            preprocess(self.raw, self.output)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir()})

    def test_duplicate_id_is_reported(self):
        (self.raw / "duplicate.json").write_bytes((self.raw / "a.json").read_bytes())
        with self.assertRaisesRegex(ValueError, "src_uid répété"):
            preprocess(self.raw, self.output)
        self.assertFalse(self.output.exists())

    def test_output_inside_raw_is_rejected(self):
        with self.assertRaises(ValueError):
            preprocess(self.raw, self.raw / "processed")

    def test_cli_success_and_failure(self):
        command = [
            sys.executable,
            "-m",
            "algo_tags",
            "preprocess",
            "--input",
            str(self.raw),
            "--output",
            str(self.output),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        (self.raw / "broken.json").write_text("null", encoding="utf-8")
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("broken.json", result.stderr)


if __name__ == "__main__":
    unittest.main()
