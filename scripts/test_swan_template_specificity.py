"""Tests for template-specific SWAN background calibration."""

import math
import unittest

import run_swan_template_specificity as experiment


class FeatureTests(unittest.TestCase):
    def candidate(self, bank, f1, coverage, precision):
        return {
            "bank_index": bank,
            "local_f1": f1,
            "coverage": coverage,
            "local_precision": precision,
        }

    def checkpoint(self, candidates):
        return {
            "graphs": [
                {
                    "known_target_template_indices": [],
                    "candidates": candidates,
                }
            ]
        }

    def test_document_score_takes_maximum_per_template(self):
        checkpoint = self.checkpoint(
            [
                self.candidate(3, 0.7, 0.8, 0.6),
                self.candidate(3, 0.9, 0.7, 0.9),
                self.candidate(4, 0.8, 0.6, 1.0),
            ]
        )
        scores = experiment.document_template_scores(
            checkpoint,
            "local_f1",
        )
        self.assertEqual(scores[3], 0.9)
        self.assertEqual(scores[4], 0.8)
        self.assertEqual(scores[2], 0.0)

    def test_balanced_min_requires_precision_and_coverage(self):
        checkpoint = self.checkpoint(
            [
                self.candidate(3, 0.7, 0.9, 0.6),
                self.candidate(3, 0.7, 0.7, 0.8),
            ]
        )
        scores = experiment.document_template_scores(
            checkpoint,
            "balanced_min",
        )
        self.assertEqual(scores[3], 0.7)


class StatisticTests(unittest.TestCase):
    def matrix(self):
        return [[0.0] * 50 for _ in range(250)]

    def test_unseen_extreme_has_smoothed_minimum_tail(self):
        scores = [0.0] * 50
        scores[7] = 1.0
        result = experiment.document_statistic(
            scores,
            self.matrix(),
            1,
        )
        self.assertAlmostEqual(
            result["statistic"],
            math.log(250),
        )
        self.assertEqual(
            result["top_evidence"][0]["bank_index"],
            7,
        )

    def test_background_floor_has_zero_surprisal(self):
        result = experiment.document_statistic(
            [0.0] * 50,
            self.matrix(),
            4,
        )
        self.assertEqual(result["statistic"], 0.0)

    def test_leave_one_out_excludes_self(self):
        matrix = self.matrix()
        matrix[0][5] = 1.0
        result = experiment.document_statistic(
            matrix[0],
            matrix,
            1,
            leave_out=0,
        )
        self.assertAlmostEqual(
            result["top_evidence"][0]["tail_probability"],
            1 / 250,
        )

    def test_top_k_sums_largest_template_surprisals(self):
        scores = [0.0] * 50
        scores[3] = 1.0
        scores[9] = 1.0
        result = experiment.document_statistic(
            scores,
            self.matrix(),
            2,
        )
        self.assertAlmostEqual(
            result["statistic"],
            2 * math.log(250),
        )
        self.assertEqual(
            [row["bank_index"] for row in result["top_evidence"]],
            [3, 9],
        )

    def test_strict_threshold_respects_ties(self):
        values = [3.0, 3.0, 2.0] + [0.0] * 247
        threshold, false_positives = (
            experiment.calibrated_threshold(values)
        )
        self.assertEqual(threshold, 2.0)
        self.assertEqual(false_positives, 2)


class ProtocolTests(unittest.TestCase):
    def test_grid_is_frozen_and_unique(self):
        self.assertEqual(len(experiment.GRID), 9)
        self.assertEqual(
            len(
                {
                    experiment.rule_id(rule)
                    for rule in experiment.GRID
                }
            ),
            9,
        )

    def test_rank_prioritizes_boundary_robustness(self):
        weak = {
            "rule": {"feature": "balanced_min", "top_k": 1},
            "calibration_false_positives": 0,
            "positive_development": {
                "original": {"detected": 15},
                "composition": {"detected": 15},
                "reparse": {"detected": 7},
            },
        }
        balanced = {
            "rule": {"feature": "coverage", "top_k": 4},
            "calibration_false_positives": 2,
            "positive_development": {
                "original": {"detected": 8},
                "composition": {"detected": 8},
                "reparse": {"detected": 8},
            },
        }
        self.assertGreater(
            experiment.development_rank(balanced),
            experiment.development_rank(weak),
        )

    def test_attack_inventory_is_fixed(self):
        inventory = experiment.attack_inventory()
        self.assertEqual(len(inventory), 30)
        self.assertEqual(
            sum(row["quality_valid"] for row in inventory),
            24,
        )
        self.assertEqual(
            sum(
                row["quality_valid"]
                and row["historically_successful"]
                for row in inventory
            ),
            23,
        )


if __name__ == "__main__":
    unittest.main()
