"""Tests for cached cross-run SWAN localization."""

import unittest

from analyze_swan_positive_localization import describe, triple_count


class PositiveLocalizationTests(unittest.TestCase):
    def test_triple_count(self):
        self.assertEqual(triple_count("(a / alpha :ARG0 (b / beta) :quant 2)"), 5)

    def test_describe_keeps_allocated_slots(self):
        rows = [
            {"words": 5, "candidate_triples": 8, "target_triples": 8,
             "triple_ratio": 1.0, "score": .8, "accepted": True},
            {"words": 15, "candidate_triples": 24, "target_triples": 8,
             "triple_ratio": 3.0, "score": .2, "accepted": False},
        ]
        result = describe(rows)
        self.assertEqual(result["slots"], 2)
        self.assertEqual(result["accepted"], 1)
        self.assertEqual(result["words"]["mean"], 10)
        self.assertEqual(result["candidate_target_ratio_mean"], 2)


if __name__ == "__main__":
    unittest.main()
