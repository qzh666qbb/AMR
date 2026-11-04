"""Offline tests for post-certificate calibration helpers."""

import math
import unittest

from certified_resegmentation_calibration import (
    binomial_cdf,
    clopper_pearson_upper,
    split_conformal_calibrate,
    strict_exceedance_count,
)


class SplitConformalTests(unittest.TestCase):
    def test_continuous_rank_and_strict_exceedance(self):
        result = split_conformal_calibrate(range(999), 0.05)
        self.assertEqual(result.order_rank, 950)
        self.assertEqual(result.threshold, 949)
        self.assertEqual(result.maximum_calibration_exceedances, 49)
        self.assertEqual(result.strict_exceedances, 49)
        self.assertTrue(math.isclose(result.empirical_fpr, 49 / 999))
        self.assertTrue(math.isclose(result.marginal_fpr_bound, 0.05))

    def test_ties_are_conservative(self):
        result = split_conformal_calibrate([0] * 990 + [1] * 9, 0.05)
        self.assertEqual(result.threshold, 0)
        self.assertEqual(result.maximum_calibration_exceedances, 49)
        self.assertEqual(result.strict_exceedances, 9)
        self.assertLess(result.empirical_fpr, result.marginal_fpr_bound)

    def test_small_sample_uses_infinite_threshold(self):
        result = split_conformal_calibrate(range(9), 0.05)
        self.assertEqual(result.order_rank, 10)
        self.assertEqual(result.threshold, math.inf)
        self.assertEqual(result.maximum_calibration_exceedances, 0)
        self.assertEqual(result.strict_exceedances, 0)
        self.assertEqual(result.marginal_fpr_bound, 0)
        self.assertEqual(strict_exceedance_count(range(9), math.inf), 0)

    def test_invalid_calibration_fails_closed(self):
        invalid_cases = (
            ([], 0.05),
            ([1, float("nan")], 0.05),
            ([1], 0),
            ([1], 1),
            ([1], float("nan")),
        )
        for scores, target in invalid_cases:
            with self.subTest(scores=scores, target=target):
                with self.assertRaises(ValueError):
                    split_conformal_calibrate(scores, target)

    def test_invalid_test_scores_fail_closed(self):
        with self.assertRaises(ValueError):
            strict_exceedance_count([0, float("inf")], 0)
        with self.assertRaises(ValueError):
            strict_exceedance_count([0], float("-inf"))


class ExactBinomialTests(unittest.TestCase):
    def test_binomial_cdf_matches_direct_sum(self):
        expected = sum(
            math.comb(10, index) * 0.3 ** index * 0.7 ** (10 - index)
            for index in range(4)
        )
        self.assertTrue(math.isclose(
            binomial_cdf(3, 10, 0.3),
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ))

    def test_clopper_pearson_zero_success_closed_form(self):
        upper = clopper_pearson_upper(0, 250, 0.95)
        expected = 1 - 0.05 ** (1 / 250)
        self.assertTrue(math.isclose(
            upper,
            expected,
            rel_tol=1e-11,
            abs_tol=1e-11,
        ))

    def test_clopper_pearson_interior_root(self):
        upper = clopper_pearson_upper(50, 1000, 0.95)
        self.assertGreater(upper, 0.05)
        self.assertTrue(math.isclose(
            binomial_cdf(50, 1000, upper),
            0.05,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ))

    def test_clopper_pearson_all_successes(self):
        self.assertEqual(clopper_pearson_upper(10, 10, 0.95), 1)

    def test_invalid_binomial_inputs_fail_closed(self):
        invalid_cdf = (
            (-2, 10, 0.5),
            (11, 10, 0.5),
            (1, -1, 0.5),
            (1, 10, -0.1),
            (1, 10, 1.1),
        )
        for successes, trials, probability in invalid_cdf:
            with self.subTest(
                successes=successes,
                trials=trials,
                probability=probability,
            ):
                with self.assertRaises(ValueError):
                    binomial_cdf(successes, trials, probability)
        with self.assertRaises(ValueError):
            clopper_pearson_upper(1, 0)
        with self.assertRaises(ValueError):
            clopper_pearson_upper(1, 10, 1)


if __name__ == "__main__":
    unittest.main()
