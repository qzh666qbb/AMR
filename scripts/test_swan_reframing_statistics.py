import itertools
import math
import unittest

from swan_reframing_statistics import (
    conservative_threshold, exact_mcnemar, holm_adjust, minimum_green,
    paired_power, verify_official, z_max, z_score,
)


class StatisticalContracts(unittest.TestCase):
    def test_actual_official_functions_with_only_mocks(self):
        audit = verify_official()
        self.assertEqual(audit["comparisons_with_mocked_graph_functions"], 3366)
        self.assertEqual(audit["n1_direct"], "NaN")
        self.assertAlmostEqual(audit["n1_preparsed_max"], math.sqrt(3))

    def test_upper_bound_and_green_discreteness(self):
        self.assertLess(z_max(1), 2.33)
        self.assertIsNone(minimum_green(1))
        self.assertEqual(minimum_green(8), 5)
        for n in range(1, 20):
            self.assertTrue(all(z_score(g, n) <= z_max(n) for g in range(n + 1)))
        for args in [(1, 0), (-1, 5), (1.5, 5), (2, 1)]:
            with self.assertRaises(ValueError):
                z_score(*args)

    def test_exact_direction_and_no_discordance(self):
        self.assertEqual(exact_mcnemar(0, 0), 1)
        self.assertAlmostEqual(exact_mcnemar(29, 0), 2 ** -29)
        self.assertEqual(exact_mcnemar(0, 29), 1)

    def test_holm_order(self):
        self.assertEqual(holm_adjust([.04, .001, .02]), [.04, .003, .04])

    def test_calibration_ties_and_empty_are_not_dropped(self):
        scores = [3.0] * 10 + [1.0] * 990
        threshold = conservative_threshold(scores)
        self.assertEqual(sum(x > threshold for x in scores), 10)
        self.assertEqual(conservative_threshold([3.0] * 11 + [1.0] * 989), 3.0)
        for scores in ([], [1.0, math.nan], [math.inf]):
            with self.assertRaises(ValueError):
                conservative_threshold(scores)

    def test_exact_power_against_enumerating_all_outcomes(self):
        n, pl, pg, alpha, effect = 6, .30, .10, .10, .10
        expected = 0
        for outcomes in itertools.product((-1, 0, 1), repeat=n):
            losses, gains = outcomes.count(1), outcomes.count(-1)
            if exact_mcnemar(losses, gains) <= alpha and (losses-gains)/n >= effect:
                expected += pl**losses * pg**gains * (1-pl-pg)**outcomes.count(0)
        self.assertAlmostEqual(paired_power(n, pl, pg, alpha, effect), expected)


if __name__ == "__main__":
    unittest.main()
