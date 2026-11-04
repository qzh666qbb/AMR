"""Tests for the SWAN local multi-instance evidence preflight."""

import random
import unittest

import run_swan_local_multi_instance as experiment


class LocalAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s2match = experiment.load_s2match()

    def test_local_denominator_ignores_unaligned_component(self):
        graph = experiment.parse_graph(
            """
            (m / multi-sentence
               :snt1 (w / V :ARG0 (p / N))
               :snt2 (s / V :ARG0 (c / N)))
            """,
            "a",
            self.s2match,
        )
        first_component = {1, 2}
        count = experiment.local_triple_count(
            graph,
            first_component,
        )
        self.assertEqual(count, 3)
        self.assertLess(count, graph["triples"])

    def test_active_nodes_include_matched_relation_endpoints(self):
        mapping = [0, 1, -1]
        weights = {
            (0, 0): {-1: 0, (1, 1): 1},
            (1, 1): {-1: 0, (0, 0): 1},
        }
        self.assertEqual(
            experiment.active_nodes(mapping, weights),
            {0, 1},
        )

    def test_weight_mask_removes_blocked_nodes_and_relations(self):
        weights = {
            (0, 0): {-1: 1, (1, 1): 1},
            (1, 1): {-1: 1, (0, 0): 1, (2, 2): 1},
            (2, 2): {-1: 1, (1, 1): 1},
        }
        masked = experiment.mask_weight_dict(weights, {0})
        self.assertNotIn((0, 0), masked)
        self.assertNotIn((0, 0), masked[(1, 1)])
        self.assertIn((2, 2), masked[(1, 1)])

    def test_alignment_recovers_identical_tiny_graph(self):
        first = experiment.parse_graph(
            "(w / V :ARG0 (p / N) :ARG1 (x / N))",
            "a",
            self.s2match,
        )
        second = experiment.parse_graph(
            "(w / V :ARG0 (p / N) :ARG1 (x / N))",
            "b",
            self.s2match,
        )
        candidates, weights = self.s2match.compute_pool(
            first["instances"],
            first["attributes"],
            first["relations"],
            second["instances"],
            second["attributes"],
            second["relations"],
            "a",
            "b",
            {},
            0.5,
            0.5,
            self.s2match.get_sim_fun("cosine"),
            "split",
        )
        random.seed(7)
        mapping, match = experiment.best_match_with_pool(
            self.s2match,
            candidates,
            weights,
            first["instances"],
            second["instances"],
        )
        active = experiment.active_nodes(mapping, weights)
        self.assertEqual(len(active), 3)
        self.assertEqual(
            experiment.local_triple_count(first, active),
            first["triples"],
        )
        self.assertAlmostEqual(match, first["triples"])


class EvidenceSelectionTests(unittest.TestCase):
    def row(self, bank, nodes, score=0.9, occurrence=0):
        return {
            "bank_index": bank,
            "occurrence": occurrence,
            "soft_match": score * 10,
            "template_triples": 10,
            "local_candidate_triples": 10,
            "coverage": score,
            "local_precision": score,
            "local_f1": score,
            "active_nodes": nodes,
        }

    def graph(self, rows):
        return {
            "graph_id": "g",
            "known_target_template_indices": [],
            "candidates": rows,
        }

    def rule(self, overlap):
        return {
            "coverage": 0.8,
            "local_precision": 0.8,
            "min_active_nodes": 2,
            "max_overlap": overlap,
        }

    def test_overlapping_bank_candidates_are_deduplicated(self):
        graph = self.graph(
            [
                self.row(1, [0, 1, 2], 0.95),
                self.row(2, [0, 1, 3], 0.90),
                self.row(3, [4, 5, 6], 0.85),
            ]
        )
        selected = experiment.select_graph_evidence(
            graph,
            self.rule(0.5),
        )
        self.assertEqual(
            [row["bank_index"] for row in selected],
            [1, 3],
        )

    def test_disjoint_repeated_template_occurrences_both_count(self):
        graph = self.graph(
            [
                self.row(7, [0, 1, 2], 0.95, 0),
                self.row(7, [3, 4, 5], 0.90, 1),
            ]
        )
        selected = experiment.select_graph_evidence(
            graph,
            self.rule(0.0),
        )
        self.assertEqual(len(selected), 2)

    def test_document_count_sums_across_graphs(self):
        checkpoint = {
            "graphs": [
                self.graph([self.row(1, [0, 1], 0.9)]),
                self.graph([self.row(2, [0, 1], 0.9)]),
            ]
        }
        result = experiment.document_evidence(
            checkpoint,
            self.rule(0.0),
        )
        self.assertEqual(result["count"], 2)


class LockingTests(unittest.TestCase):
    def test_calibration_selects_smallest_threshold_under_budget(self):
        threshold, false_positives = (
            experiment.calibrated_threshold(
                [0] * 240 + [1] * 7 + [2] * 2 + [3]
            )
        )
        self.assertEqual(threshold, 3)
        self.assertEqual(false_positives, 1)

    def test_calibration_can_use_zero_false_positive_threshold(self):
        threshold, false_positives = (
            experiment.calibrated_threshold([0] * 250)
        )
        self.assertEqual(threshold, 1)
        self.assertEqual(false_positives, 0)

    def test_rank_prioritizes_worst_case_boundary_recovery(self):
        base = {
            "rule": experiment.GRID[0],
            "calibration_false_positives": 0,
            "positive_development": {
                "original": 15,
                "composition": 15,
                "reparse": 8,
            },
        }
        balanced = {
            "rule": experiment.GRID[-1],
            "calibration_false_positives": 2,
            "positive_development": {
                "original": 12,
                "composition": 12,
                "reparse": 12,
            },
        }
        self.assertGreater(
            experiment.development_rank(balanced),
            experiment.development_rank(base),
        )

    def test_grid_size_is_frozen(self):
        self.assertEqual(len(experiment.GRID), 36)
        self.assertEqual(
            len({experiment.rule_id(rule) for rule in experiment.GRID}),
            36,
        )


class SourceSplitTests(unittest.TestCase):
    def test_positive_split_is_disjoint_and_complete(self):
        self.assertEqual(len(experiment.DEVELOPMENT_IDS), 15)
        self.assertEqual(len(experiment.HOLDOUT_IDS), 15)
        self.assertFalse(
            set(experiment.DEVELOPMENT_IDS)
            & set(experiment.HOLDOUT_IDS)
        )

    def test_positive_payloads_have_expected_graph_counts(self):
        payloads = experiment.positive_payloads(
            [experiment.DEVELOPMENT_IDS[0]]
        )
        by_representation = {
            row["representation"]: row for row in payloads
        }
        self.assertEqual(
            len(by_representation["original"]["graphs"]),
            4,
        )
        self.assertEqual(
            len(by_representation["composition"]["graphs"]),
            2,
        )
        self.assertEqual(
            len(by_representation["reparse"]["graphs"]),
            2,
        )


if __name__ == "__main__":
    unittest.main()
