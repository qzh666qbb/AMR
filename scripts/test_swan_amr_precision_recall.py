"""Tests for S2Match precision/recall decomposition."""

import math
import unittest

import run_swan_amr_precision_recall as experiment


class MetricTests(unittest.TestCase):
    def test_valid_metric_reconstructs_f1(self):
        experiment.validate_metric((0.5, 1.0, 2 / 3))

    def test_invalid_f1_is_rejected(self):
        with self.assertRaises(ValueError):
            experiment.validate_metric((0.5, 1.0, 0.5))

    def test_frozen_budget_is_exact(self):
        self.assertEqual(experiment.EXPECTED_TARGETS, 108)
        self.assertEqual(experiment.EXPECTED_S2MATCH, 1620)


class StatisticsTests(unittest.TestCase):
    def test_sign_test_confirms_all_negative(self):
        result = experiment.sign_test([-0.1] * 30)
        self.assertTrue(result["confirmed_negative"])
        self.assertTrue(
            math.isclose(result["one_sided_sign_p"], 2**-30)
        )

    def test_sign_test_excludes_ties(self):
        result = experiment.sign_test([-0.1] * 20 + [0.0] * 10)
        self.assertEqual(result["non_ties"], 20)
        self.assertEqual(result["ties"], 10)
        self.assertTrue(result["confirmed_negative"])

    def test_balanced_changes_are_not_confirmed(self):
        result = experiment.sign_test([-0.1] * 15 + [0.1] * 15)
        self.assertFalse(result["confirmed_negative"])


if __name__ == "__main__":
    unittest.main()
