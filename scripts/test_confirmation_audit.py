"""Tests for independent detection-score reconstruction."""

import copy
import math
import unittest

from audit_confirmation_experiment import verify_score


class ScoreAuditTests(unittest.TestCase):
    def setUp(self):
        self.score = {"n": 1, "green": 1, "z": math.sqrt(3),
                      "details": [{"score": .8, "bank_scores": [.8] + [0.] * 49}]}

    def test_valid_score(self):
        verify_score(self.score)
        verify_score(None)

    def test_reject_invalid_template_vector(self):
        for vector in ([.8] * 49, [.8] * 49 + [float("nan")], [.8] * 49 + [1.1]):
            bad = copy.deepcopy(self.score)
            bad["details"][0]["bank_scores"] = vector
            with self.assertRaises(ValueError):
                verify_score(bad)

    def test_reject_wrong_aggregates(self):
        for key, value in (("n", 2), ("green", 0), ("z", 0.)):
            bad = copy.deepcopy(self.score)
            bad[key] = value
            with self.assertRaises(ValueError):
                verify_score(bad)
        bad = copy.deepcopy(self.score)
        bad["details"][0]["score"] = .9
        with self.assertRaises(ValueError):
            verify_score(bad)

    def test_threshold_equality_is_a_hit(self):
        score = copy.deepcopy(self.score)
        score["details"][0] = {"score": .65, "bank_scores": [.65] * 50}
        verify_score(score)


if __name__ == "__main__":
    unittest.main()
