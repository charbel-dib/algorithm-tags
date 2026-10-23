"""Contrôles du découpage, indépendants des identifiants choisis par l'algorithme."""

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from algo_tags.data import TARGET_TAGS
from algo_tags.split import create_split


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset.jsonl"
        # Chaque tag et le cas sans cible ont assez d'exemples pour trois ensembles.
        patterns = [[0] * 8] + [[int(i == j) for i in range(8)] for j in range(8)]
        patterns += [[1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0]]
        self.rows = [{"src_uid": f"p{i:04}", "targets": pattern} for i, pattern in enumerate(patterns * 30)]
        self.write_dataset()

    def tearDown(self):
        self.temporary.cleanup()

    def write_dataset(self):
        content = "".join(json.dumps(row) + "\n" for row in self.rows).encode()
        self.dataset.write_bytes(content)
        summary = {
            "dataset_sha256": hashlib.sha256(content).hexdigest(),
            "target_order": list(TARGET_TAGS),
            "records_written": len(self.rows),
        }
        (self.root / "preprocessing_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    def test_complete_disjoint_and_balanced(self):
        before = self.dataset.read_bytes()
        output = self.root / "split"
        summary = create_split(self.dataset, output)
        with (output / "splits.csv").open(encoding="utf-8", newline="") as stream:
            assignments = list(csv.DictReader(stream))
        self.assertEqual(len(assignments), len(self.rows))
        self.assertEqual({r["src_uid"] for r in assignments}, {r["src_uid"] for r in self.rows})
        self.assertEqual(len({r["src_uid"] for r in assignments}), len(self.rows))
        for name, requested in summary["requested_fractions"].items():
            counts = summary["splits"][name]
            self.assertAlmostEqual(counts["fraction"], requested, delta=0.02)
            self.assertAlmostEqual(counts["without_target"] / counts["records"], 1 / 11, delta=0.02)
            for i, tag in enumerate(TARGET_TAGS):
                total = sum(r["targets"][i] for r in self.rows)
                self.assertAlmostEqual(
                    counts["positive_counts"][tag] / counts["records"], total / len(self.rows), delta=0.03
                )
        self.assertFalse(summary["warnings"])
        self.assertEqual(self.dataset.read_bytes(), before)

    def test_same_seed_and_reordered_rows_keep_assignments(self):
        create_split(self.dataset, self.root / "a", seed=42)
        self.rows.reverse()
        self.write_dataset()
        create_split(self.dataset, self.root / "b", seed=42)
        self.assertEqual((self.root / "a/splits.csv").read_bytes(), (self.root / "b/splits.csv").read_bytes())

    def test_existing_split_is_not_overwritten(self):
        output = self.root / "split"
        create_split(self.dataset, output)
        before = (output / "splits.csv").read_bytes()
        with self.assertRaises(FileExistsError):
            create_split(self.dataset, output, seed=1)
        self.assertEqual((output / "splits.csv").read_bytes(), before)

    def test_changed_dataset_is_rejected(self):
        self.dataset.write_bytes(self.dataset.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "ne correspond pas"):
            create_split(self.dataset, self.root / "split")

    def test_duplicate_id_and_invalid_labels_are_rejected(self):
        self.rows[1]["src_uid"] = self.rows[0]["src_uid"]
        self.write_dataset()
        with self.assertRaisesRegex(ValueError, "répété"):
            create_split(self.dataset, self.root / "split")
        self.rows[1]["src_uid"] = "fixed-id"
        self.rows[1]["targets"] = [0] * 7 + [2]
        self.write_dataset()
        with self.assertRaisesRegex(ValueError, "0 ou 1"):
            create_split(self.dataset, self.root / "split")

    def test_cli(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "algo_tags",
                "split",
                "--data",
                str(self.dataset),
                "--output",
                str(self.root / "split"),
                "--seed",
                "42",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "split/split_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
