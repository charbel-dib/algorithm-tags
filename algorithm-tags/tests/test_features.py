"""Contraintes textuelles et séparation des annotations et caractéristiques."""

import unittest

import numpy as np

from algo_tags.features import PROTECTED, TfidfBlocks, rule_features


class FeatureTests(unittest.TestCase):
    def test_negations_and_quantifiers_are_preserved(self):
        rows = [
            {"statement": "the graph must not have more than one edge"},
            {"statement": "the graph must have exactly one edge"},
        ]
        encoder = TfidfBlocks({"stopwords": True, "min_df": 1}).fit(rows)
        vectorizer = encoder.blocks[0][1]
        stop = set(vectorizer.stop_words)
        self.assertTrue(PROTECTED.isdisjoint(stop))
        self.assertIn("the", stop)
        self.assertIn("not", vectorizer.vocabulary_)
        self.assertIn("exactly", vectorizer.vocabulary_)

    def test_vocabulary_does_not_learn_from_evaluation(self):
        encoder = TfidfBlocks({"min_df": 1}).fit([{"statement": "graph edges"}])
        encoder.transform([{"statement": "unseenword"}])
        self.assertNotIn("unseenword", encoder.blocks[0][1].vocabulary_)

    def test_rules_are_counts_and_respect_word_boundaries(self):
        values = rule_features(["a graph with vertices and edges", "a photograph", "prime gcd"])
        self.assertEqual(values[0, 2], 3)
        self.assertEqual(values[1].sum(), 0)
        self.assertEqual(values[2, 6], 2)
        self.assertTrue((values[:, 1::2] == 0).all())

    def test_additional_features_do_not_read_targets(self):
        rows = [{"statement": "graph edges", "targets": [1] * 8}]
        encoder = TfidfBlocks({"min_df": 1, "manual_rules": True, "normalize_blocks": True}).fit(rows)
        left = encoder.transform(rows)
        right = encoder.transform([{**rows[0], "targets": [0] * 8}])
        np.testing.assert_array_equal(left.toarray(), right.toarray())
        self.assertAlmostEqual(float(np.linalg.norm(left.toarray())), 1.0, places=6)
