"""No-network tests for bank100 dual-distribution calibration."""

import unittest
from unittest.mock import Mock

from run_segmentation_intervention import text_sha
from run_swan_bank100_null_calibration import (
    empirical_threshold,
    factual_pass,
    proportion_z,
    score_bank100_sentence,
    structural_metrics,
    style_pass,
)


def tokenize(text):
    return [sentence.strip() + "." for sentence in text.split(".") if sentence.strip()]


class Bank100NullCalibrationTests(unittest.TestCase):
    def test_structural_gate_requires_five_sentences_and_length(self):
        sentence = " ".join(["word"] * 20) + "."
        result = structural_metrics(" ".join([sentence] * 5), "stop", tokenize)
        self.assertTrue(result["pass"])
        self.assertFalse(
            structural_metrics(" ".join([sentence] * 4), "stop", tokenize)["pass"]
        )

    def test_quality_gates_are_strict(self):
        style = {
            "fluency": 4, "coherence": 4, "nonredundancy": 4,
            "severe_problem": False, "reason": "fixture",
        }
        facts = {
            "faithfulness": 4, "unsupported_claim": False,
            "contradiction": False, "issues": [],
        }
        self.assertTrue(style_pass(style))
        self.assertTrue(factual_pass(facts))
        self.assertFalse(style_pass({**style, "coherence": 3}))
        self.assertFalse(factual_pass({**facts, "unsupported_claim": True}))

    def test_empirical_threshold_uses_strict_upper_tail(self):
        values = list(range(100))
        threshold = empirical_threshold(values, 0.05)
        self.assertEqual(threshold, 94)
        self.assertEqual(sum(value > threshold for value in values), 5)

    def test_proportion_z_uses_supplied_lambda(self):
        self.assertAlmostEqual(proportion_z(1, 5, 0.2), 0.0)
        self.assertGreater(proportion_z(2, 5, 0.2), 0)

    def test_bank_score_stops_on_first_green(self):
        candidate = "(a / alpha)"
        bank = ["(a / beta)", "(a / gamma)", "(a / delta)"]
        counts = {text_sha(template): [1, 1, 0] for template in bank}
        scorer = Mock(side_effect=[0.2, 0.7, 0.9])
        result = score_bank100_sentence(candidate, bank, counts, scorer)
        self.assertTrue(result["green"])
        self.assertEqual(result["s2match_calls"], 2)
        self.assertEqual(result["witness"]["template_index"], 1)
        self.assertEqual(scorer.call_count, 2)


if __name__ == "__main__":
    unittest.main()
