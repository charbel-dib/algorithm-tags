"""Contrôles optionnels du CNN sur des données synthétiques."""

import unittest
from importlib.util import find_spec

import numpy as np


@unittest.skipUnless(find_spec("torch"), "PyTorch optionnel non installé")
class NeuralTests(unittest.TestCase):
    def test_padding_is_excluded_from_pooling(self):
        import torch

        from algo_tags.neural import TextNetwork

        torch.manual_seed(42)
        network = TextNetwork("cnn", 12, width=8).eval()
        short = torch.tensor([[2, 3, 4, 5, 6, 0, 0, 0]])
        long = torch.tensor([[2, 3, 4, 5, 6, 0, 0, 0, 0, 0]])
        with torch.no_grad():
            torch.testing.assert_close(network(short), network(long))

    def test_one_synthetic_epoch_produces_finite_probabilities(self):
        from algo_tags.features import Sequences
        from algo_tags.neural import fit_cnn

        rows = [{"statement": "graph tree edge vertices"}, {"statement": "prime divisor math formula"}] * 4
        y = np.array([[1, 0] * 4, [0, 1] * 4] * 4, dtype=np.int8)
        config = {"max_tokens": 8, "epochs": 1, "width": 8, "batch_size": 4, "jobs": 1}
        encoder = Sequences(config).fit(rows)
        classifier, history = fit_cnn(encoder, rows, y, config)
        scores = classifier.scores(encoder.transform(rows))
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(((0 <= scores) & (scores <= 1)).all())
        self.assertEqual(scores.shape, (8, 8))
        self.assertEqual(len(history), 1)
