"""No-network tests for the SWAN bank-scale calibration pilot."""

from unittest.mock import Mock, patch
import unittest

from run_segmentation_intervention import text_sha
from run_swan_bank_scale_pilot import (
    cohort_metrics,
    load_banks,
    score_candidate,
    wilson_interval,
    z_score,
)


class BankScalePilotTests(unittest.TestCase):
    def test_official_bank_topology(self):
        banks = load_banks()
        self.assertEqual([len(banks[size]) for size in (50, 100, 500, 800)],
                         [50, 100, 500, 800])
        self.assertEqual(banks[500][:100], banks[100])
        self.assertEqual(banks[800][:500], banks[500])
        self.assertEqual(len(set(banks[50]) & set(banks[100])), 38)

    def test_wilson_interval_contains_observed_rate(self):
        lower, upper = wilson_interval(25, 100)
        self.assertLess(lower, 0.25)
        self.assertGreater(upper, 0.25)
        self.assertEqual(wilson_interval(0, 0), [None, None])

    def test_z_score_uses_frozen_null(self):
        self.assertAlmostEqual(z_score(5, 5), 3.872983346207417)
        self.assertLess(z_score(1, 5), 0)

    def test_score_candidate_reuses_shared_template_observation(self):
        candidate = "(a / alpha)"
        first = "(a / beta)"
        second = "(a / gamma)"
        banks = {2: [first, second], 3: [first, second, "(a / delta)"]}
        counts = {text_sha(template): [1, 1, 0]
                  for templates in banks.values() for template in templates}
        scorer = Mock(side_effect=lambda _, template: 0.7 if template == first else 0.1)
        with patch("run_swan_bank_scale_pilot.BANK_SIZES", (2, 3)):
            result = score_candidate(candidate, banks, counts, scorer)
        self.assertTrue(result["banks"]["2"]["green"])
        self.assertTrue(result["banks"]["3"]["green"])
        self.assertEqual(result["unique_s2match_calls"], 1)
        self.assertEqual(scorer.call_count, 1)

    def test_cohort_metrics_preserves_document_denominator(self):
        sentence = lambda green: {
            "banks": {"50": {
                "green": green,
                "count_compatible_fraction": 0.2,
                "new_s2match_calls": 1,
            }}
        }
        documents = [
            {"document_id": "a", "sentences": [sentence(True)] * 5},
            {"document_id": "b", "sentences": [sentence(False)] * 5},
        ]
        result = cohort_metrics(documents, 50)
        self.assertEqual(result["documents"], 2)
        self.assertEqual(result["sentences"], 10)
        self.assertEqual(result["detected_documents"], 1)
        self.assertEqual(result["fixed_threshold_document_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
