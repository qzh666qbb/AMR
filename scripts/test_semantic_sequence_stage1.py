"""Meaningful invariance and oracle checks for the pilot prerequisite."""
import itertools
import math
import unittest
from fractions import Fraction

import numpy as np

from semantic_sequence_stage1_common import (
    POOL, balanced_assignment, jaccard_distance, structural_features,
)
from run_semantic_sequence_stage1a import (
    build_universe, summarize, symbol_weights, top_credit, validate_row_identities,
)


class StructuralTests(unittest.TestCase):
    def test_variable_order_inverse_and_literal_invariance(self):
        left = '(a / V :ARG1 (b / N) :ARG2 b :mode imperative :value 123)'
        right = '(q / V :value 987 :mode interrogative :ARG1 (r / N :ARG2-of q))'
        self.assertEqual(structural_features(left), structural_features(right))

    def test_lexical_concepts_removed_but_roles_retained(self):
        left = structural_features('(a / eat-01 :ARG0 (b / boy))')
        same = structural_features('(x / walk-02 :ARG0 (z / girl))')
        different = structural_features('(x / walk-02 :ARG1 (z / girl))')
        self.assertEqual(left, same)
        self.assertGreater(jaccard_distance(left, different), 0)

    def test_exact_balancing_matches_exhaustive_oracle(self):
        costs = [[Fraction((i + j) % 3, i + 2) for j in range(3)] for i in range(6)]
        oracle = min(
            (sum(costs[i][a] for i, a in enumerate(assignment)), assignment)
            for assignment in itertools.product(range(3), repeat=6)
            if all(assignment.count(group) == 2 for group in range(3))
        )
        self.assertEqual(balanced_assignment(costs, 2), oracle)

    def test_all_ties_have_stable_lexicographic_solution(self):
        cost, assignment = balanced_assignment([[Fraction(0)] * 3 for _ in range(6)], 2)
        self.assertEqual(cost, 0)
        self.assertEqual(assignment, (0, 0, 1, 1, 2, 2))


class MetricTests(unittest.TestCase):
    def test_source_slot_gaps_join_by_accepted_ordinal(self):
        docs = [{"document_id": "d", "slots": [
            {"slot": 0, "template_index": 9, "text": "First."},
            {"slot": 3, "template_index": 11, "text": "Second."},
        ]}]
        universe = build_universe(docs)
        self.assertEqual([u["slot"] for u in universe], [0, 1])
        self.assertEqual([u["source_slot"] for u in universe], [0, 3])
        rows = {("d", 0, "baseline"): {"template_index": 9},
                ("d", 1, "baseline"): {"template_index": 11}}
        validate_row_identities(universe, rows)
        rows["d", 1, "baseline"]["template_index"] = 9
        with self.assertRaisesRegex(ValueError, "Template mismatch"):
            validate_row_identities(universe, rows)

    def test_equal_symbols_do_not_gain_from_id_ties(self):
        self.assertEqual(top_credit([1, 1, 1, 1], 3, 1), 0.25)
        self.assertEqual(top_credit([1, 1, 1, 1], 0, 2), 0.5)
        q = symbol_weights([1000] * 24, [list(range(i * 6, i * 6 + 6)) for i in range(4)], 0.1)
        np.testing.assert_allclose(q, [0.25] * 4)

    def test_erasure_retained_and_not_renormalized(self):
        groups = [list(range(i * 6, i * 6 + 6)) for i in range(4)]
        mapping = {str(t): i // 6 for i, t in enumerate(POOL)}
        universe = [
            {"doc": "d", "slot": i, "template_index": POOL[i * 6], "unique_text_weight": 1.0}
            for i in range(4)
        ]
        rows = {
            ("d", i, "baseline"): {"scores": [0.5] * 100}
            for i in range(4)
        }
        rows["d", 3, "baseline"]["scores"][POOL[0]] = None
        result = summarize(universe, rows, "scores", mapping, groups, "baseline", 0.1, False)
        self.assertEqual(result["planned_carriers"], 4)
        self.assertEqual(result["erased_carriers"], 1)
        self.assertAlmostEqual(result["balanced_accuracy"], 0.1875)
        self.assertAlmostEqual(result["nll"], math.log(4))
        self.assertAlmostEqual(result["brier"], 0.75)


if __name__ == "__main__":
    unittest.main()
