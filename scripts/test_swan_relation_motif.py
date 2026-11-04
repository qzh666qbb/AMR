"""Tests for the SWAN relation-anchored motif preflight."""

import math
import unittest

import run_swan_relation_motif as experiment


def unit_weights(template, default=1.0):
    return {
        experiment.signature_key(unit["signature"]): default
        for unit in template["units"]
    }


class ParsingTests(unittest.TestCase):
    def test_parser_strips_concepts_and_excludes_top(self):
        graph = experiment.parse_graph(
            '(a / V :ARG0 (b / N) :mode imperative)'
        )
        self.assertEqual(graph["concepts"], ["V", "N"])
        self.assertIn(("mode", 0, "imperative"), graph["attributes"])
        self.assertFalse(
            any(role == "TOP" for role, _, _ in graph["attributes"])
        )

    def test_template_structure_inventory_is_frozen(self):
        inventory = experiment.structural_inventory()
        self.assertEqual(inventory["templates"], 50)
        self.assertEqual(inventory["tree_templates"], 49)
        self.assertEqual(inventory["unicyclic_templates"], 1)
        self.assertEqual(inventory["spanning_tree_counts"][35], 3)


class MatcherTests(unittest.TestCase):
    def score(self, template_amr, candidate_amr, feature):
        template = experiment.prepare_template(template_amr, 0)
        candidate = experiment.parse_graph(candidate_amr)
        result = experiment.score_template(
            template,
            candidate,
            unit_weights(template),
        )
        return result["features"][feature]

    def test_x_is_template_side_wildcard(self):
        result = self.score(
            "(a / V :ARG1 (b / X))",
            "(x / V :ARG1 (y / NE))",
            "relation_coverage",
        )
        self.assertEqual(result["score"], 1.0)

    def test_extra_candidate_background_does_not_reduce_score(self):
        result = self.score(
            "(a / V :ARG0 (b / N) :ARG1 (c / N))",
            """
            (x / V
               :ARG0 (y / N)
               :ARG1 (z / N)
               :time (q / N :mod (m / ADJ)))
            """,
            "motif_coverage",
        )
        self.assertEqual(result["score"], 1.0)

    def test_relation_direction_is_exact(self):
        result = self.score(
            "(a / V :ARG0 (b / N))",
            "(x / N :ARG0 (y / V))",
            "relation_coverage",
        )
        self.assertEqual(result["score"], 0.0)

    def test_mapping_is_globally_injective(self):
        result = self.score(
            "(a / V :ARG1 (b / N) :ARG1 (c / N))",
            "(x / V :ARG1 (y / N))",
            "relation_coverage",
        )
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["matched_relations"], 1)

    def test_wedge_requires_one_consistent_center(self):
        result = self.score(
            "(a / V :ARG1 (b / N) :ARG1 (c / N))",
            """
            (x / V
               :ARG1 (y / N)
               :mod (q / V :ARG1 (z / N)))
            """,
            "motif_coverage",
        )
        self.assertLess(result["score"], 1.0)
        self.assertEqual(result["matched_wedges"], 0)

    def test_unicyclic_template_can_match_fully(self):
        template_amr = experiment.bank_templates()[35]
        result = self.score(
            template_amr,
            template_amr,
            "idf_motif_coverage",
        )
        self.assertAlmostEqual(result["score"], 1.0)


class CalibrationTests(unittest.TestCase):
    def test_rare_signature_gets_larger_idf(self):
        rare = 1 + math.log(251 / 2)
        common = 1 + math.log(251 / 251)
        self.assertGreater(rare, common)

    def test_strict_threshold_preserves_false_positive_budget(self):
        values = [3.0, 3.0, 2.0] + [0.0] * 247
        threshold, false_positives = (
            experiment.calibrated_threshold(values)
        )
        self.assertEqual(threshold, 2.0)
        self.assertEqual(false_positives, 2)

    def test_candidate_rank_prioritizes_boundary_robustness(self):
        weak = {
            "feature": "idf_motif_coverage",
            "positive_development": {
                "original": {"localized_targets": 56},
                "composition": {"localized_targets": 56},
                "reparse": {"localized_targets": 20},
            },
        }
        balanced = {
            "feature": "relation_coverage",
            "positive_development": {
                "original": {"localized_targets": 40},
                "composition": {"localized_targets": 30},
                "reparse": {"localized_targets": 40},
            },
        }
        self.assertGreater(
            experiment.candidate_feature_rank(balanced),
            experiment.candidate_feature_rank(weak),
        )

    def test_protocol_constants_match_preregistered_gates(self):
        self.assertEqual(experiment.FEATURES, (
            "relation_coverage",
            "motif_coverage",
            "idf_motif_coverage",
        ))
        self.assertEqual(experiment.CANDIDATE_TOP_K, 5)
        self.assertEqual(experiment.CANDIDATE_MIN_ORIGINAL, 42)
        self.assertEqual(experiment.CANDIDATE_MIN_REPARSE, 34)
        self.assertEqual(experiment.DOCUMENT_TOP_KS, (1, 2, 4))


if __name__ == "__main__":
    unittest.main()
