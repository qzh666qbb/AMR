"""Tests for the WP2-v7 SimMark probe sweep."""

import unittest

from run_certified_resegmentation_wp2_v7_probe import select_winner


def result(batch_size, rate, passed=True):
    return {
        "batch_size": batch_size,
        "feature_seconds_per_input_token": rate,
        "passed": passed,
    }


class ProbeSweepTests(unittest.TestCase):
    def test_selects_fastest_passing_candidate(self):
        baseline, winner, speedup = select_winner([
            result(64, 4.0),
            result(128, 3.0),
            result(256, 2.0),
            result(384, 2.5),
        ])
        self.assertEqual(baseline["batch_size"], 64)
        self.assertEqual(winner["batch_size"], 256)
        self.assertEqual(speedup, 2.0)

    def test_half_percent_tie_prefers_smaller_batch(self):
        _, winner, _ = select_winner([
            result(64, 4.0),
            result(128, 2.01),
            result(256, 2.0),
            result(384, 3.0),
        ])
        self.assertEqual(winner["batch_size"], 128)

    def test_failed_candidates_are_ignored(self):
        _, winner, speedup = select_winner([
            result(64, 4.0),
            result(128, 1.0, passed=False),
            result(256, 2.5),
            result(384, 3.0),
        ])
        self.assertEqual(winner["batch_size"], 256)
        self.assertEqual(speedup, 1.6)

    def test_no_winner_when_baseline_fails(self):
        baseline, winner, speedup = select_winner([
            result(64, 4.0, passed=False),
            result(128, 2.0),
            result(256, 1.0),
            result(384, 3.0),
        ])
        self.assertFalse(baseline["passed"])
        self.assertIsNone(winner)
        self.assertIsNone(speedup)


if __name__ == "__main__":
    unittest.main()
