import unittest

from algo_tags.data import to_problem
from algo_tags.features import Features, ast_tokens, clean_code, difficulty_features, sample_features

from helpers import raw_problems


class FeaturesTest(unittest.TestCase):
    def test_clean_code_drops_fast_io_and_comments(self):
        code = "import sys\ninput = sys.stdin.readline\n# read\nn = int(input())"
        self.assertEqual(clean_code(code), "n = int(input())")

    def test_ast_tokens_python3_and_python2(self):
        tokens = ast_tokens("from math import gcd\nprint(gcd(4, 6) % 998244353)").split()
        for token in ("I_math", "I_gcd", "C_gcd", "OP_Mod", "K_MOD"):
            self.assertIn(token, tokens)
        self.assertIn("C_input", ast_tokens("n = raw_input()\nprint n").split())
        self.assertEqual(ast_tokens("def ("), "PARSE_ERROR")

    def test_dense_features(self):
        problems = [to_problem(raw) for raw in raw_problems(3)]
        problems[0]["difficulty"] = None
        problems[1]["sample_out"] = "0.500000"
        self.assertEqual(difficulty_features(problems)[0, 1], 1.0)  # missing flag
        samples = sample_features(problems)
        self.assertEqual(samples.shape, (3, 11))
        self.assertEqual(samples[1, 0], 1.0)  # real-valued answer

    def test_blocks_are_fitted_then_concatenated(self):
        problems = [to_problem(raw) for raw in raw_problems(30)]
        features = Features(("words", "code", "ast", "difficulty", "samples")).fit(problems)
        x = features.transform(problems[:5])
        self.assertEqual(x.shape[0], 5)
        self.assertGreater(x.shape[1], 8 + 11)
        with self.assertRaises(ValueError):
            Features(("words", "unknown"))


if __name__ == "__main__":
    unittest.main()
