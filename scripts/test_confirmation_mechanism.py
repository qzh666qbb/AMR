"""Tests for the frozen sentence-boundary mechanism experiment."""

import json
from pathlib import Path
import unittest

from analyze_confirmation_mechanism import decomposition_summary, transitions
from analyze_sentence_count_decomposition import decompose
from run_confirmation_mechanism import lexical_tokens, punctuation_merge


class MechanismPreparationTests(unittest.TestCase):
    def test_punctuation_merge_preserves_lexical_sequence(self):
        sentences = ["Alice called Bob.", "Bob answered!", "They met today?"]
        merged = punctuation_merge(sentences)
        self.assertEqual(merged, "Alice called Bob; Bob answered; They met today?")
        self.assertEqual(lexical_tokens(merged), lexical_tokens(" ".join(sentences)))

    def test_punctuation_merge_requires_terminal_punctuation(self):
        with self.assertRaises(ValueError):
            punctuation_merge(["No punctuation", "Second sentence."])
        with self.assertRaises(ValueError):
            punctuation_merge(["One sentence."])


class MechanismAnalysisTests(unittest.TestCase):
    def test_transitions_and_exact_pairing(self):
        result = transitions([True, True, False, False], [True, False, True, False])
        self.assertEqual(result["both_escape"], 1)
        self.assertEqual(result["before_only_escape"], 1)
        self.assertEqual(result["after_only_escape"], 1)
        self.assertEqual(result["neither_escape"], 1)
        self.assertEqual(result["mcnemar_exact_two_sided_p"], 1.)

    def test_decomposition_summary_sums_observed_drop(self):
        row = decompose(5, 5, 1, 0)
        result = decomposition_summary([row])
        self.assertAlmostEqual(
            result["mean_count_drop"] + result["mean_green_rate_drop"],
            result["mean_observed_drop"])
        self.assertEqual(result["actual_escape"], 1)

    def test_completed_mechanism_summary(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "experiments/confirmation_mechanism30/summary.json"
        if not path.exists():
            self.skipTest("Complete mechanism experiment is not present")
        result = json.loads(path.read_text())
        self.assertEqual(result["status"], "complete_verified")
        self.assertEqual(result["coverage"]["records"], 134)
        fixed = result["punctuation_control"]["fixed_2_33"]["escape_rates"]
        self.assertEqual(fixed["merged_sentence"]["count"], 30)
        self.assertEqual(fixed["merged_exact_clause"]["count"], 1)
        boundary = result["selected_resegmentation"]["boundary_only"]["feedback"]
        self.assertEqual(boundary["fresh_clause_fixed_escape"]["count"], 20)
        self.assertEqual(boundary["fresh_clause_calibrated_escape"]["count"], 11)


if __name__ == "__main__":
    unittest.main()
