"""Tests for the deterministic partial-coverage power audit."""

import math
import unittest

import run_swan_partial_coverage_power as power


class PowerTests(unittest.TestCase):
    def test_fixed_rule_requires_seven_of_twelve(self):
        self.assertEqual(power.minimum_green(12, 0.25, 2.33), 7)
        self.assertLessEqual(power.z_score(6, 12, 0.25), 2.33)
        self.assertGreater(power.z_score(7, 12, 0.25), 2.33)

    def test_calibrated_rule_requires_four_of_twelve(self):
        snapshot = power.input_snapshot()
        required = power.minimum_green(
            12,
            snapshot["calibrated_lambda"],
            snapshot["calibrated_threshold"],
        )
        self.assertEqual(required, 4)

    def test_binomial_probability_normalizes(self):
        for probability in (0.0, 0.25, 0.9, 1.0):
            total = sum(
                power.binomial_pmf(k, 12, probability)
                for k in range(13)
            )
            self.assertTrue(math.isclose(total, 1.0, abs_tol=1e-12))

    def test_convolution_extremes(self):
        self.assertEqual(
            power.convolution_tail(6, 6, 1.0, 0.0, 7),
            0.0,
        )
        self.assertEqual(
            power.convolution_tail(6, 6, 1.0, 1.0, 7),
            1.0,
        )

    def test_rule_maps_quality_documents_without_reselection(self):
        documents = [
            {"prompt_id": "a", "short_slots": 6},
            {"prompt_id": "b", "short_slots": 5},
            {"prompt_id": "c", "short_slots": 4},
        ]
        result = power.analyze_rule(
            "fixed",
            {"lambda": 0.25, "threshold": 2.33},
            documents,
        )
        self.assertEqual(
            [
                row["prompt_id"]
                for row in result["quality_cohort"]["documents"]
            ],
            ["a", "b", "c"],
        )
        self.assertEqual(
            result["quality_cohort"]["guaranteed_detectable_documents"],
            0,
        )

    def test_frozen_quality_cohort_has_expected_slot_range(self):
        snapshot = power.input_snapshot()
        slots = [
            row["short_slots"] for row in snapshot["quality_documents"]
        ]
        self.assertEqual(len(slots), 14)
        self.assertEqual(max(slots), 6)
        self.assertEqual(sum(slots), 35)


if __name__ == "__main__":
    unittest.main()
