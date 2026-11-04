"""Statistical helpers for the completed confirmation analysis."""

import unittest
from unittest.mock import patch

from analyze_confirmation_results import analyze, distribution, holm_adjust


class FinalAnalysisTests(unittest.TestCase):
    def test_holm_two_comparisons_and_monotonicity(self):
        self.assertEqual(holm_adjust({"a": .04, "b": .03}), {"b": .06, "a": .06})
        self.assertEqual(holm_adjust({"a": .8, "b": .7}), {"b": 1., "a": 1.})
        self.assertEqual(holm_adjust({}), {})

    def test_holm_observed_pairs(self):
        result = holm_adjust({"plain": .000518798828125, "boundary": .0001220703125})
        self.assertEqual(result["boundary"], .000244140625)
        self.assertEqual(result["plain"], .000518798828125)

    def test_distribution(self):
        self.assertEqual(distribution([1, 2, 3, 4]),
                         {"n": 4, "min": 1, "median": 2.5, "mean": 2.5, "max": 4})
        self.assertIsNone(distribution([]))

    def test_incomplete_cohort_rejected(self):
        with patch("analyze_confirmation_results.audit", return_value={"status": "partial"}):
            with self.assertRaisesRegex(ValueError, "complete frozen cohort"):
                analyze(None)


if __name__ == "__main__":
    unittest.main()
