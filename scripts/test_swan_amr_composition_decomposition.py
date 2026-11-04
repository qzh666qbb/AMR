"""Tests for deterministic AMR composition decomposition."""

import math
import unittest

import penman

import run_swan_amr_composition_decomposition as experiment


FIRST = """
(w / want-01
   :ARG0 (p / person)
   :ARG1 (g / go-01 :ARG0 p))
"""

SECOND = """
(s / sleep-01
   :ARG0 (c / cat))
"""


class CompositionTests(unittest.TestCase):
    def test_composition_preserves_both_graphs_under_new_root(self):
        composed = experiment.compose_raw_amrs(FIRST, SECOND)
        graph = penman.decode(composed)
        self.assertEqual(graph.top, "m")
        self.assertIn(("m", ":instance", "multi-sentence"), graph.triples)
        self.assertTrue(any(role == ":snt1" for _, role, _ in graph.triples))
        self.assertTrue(any(role == ":snt2" for _, role, _ in graph.triples))
        expected = (
            len(penman.decode(FIRST).triples)
            + len(penman.decode(SECOND).triples)
            + 3
        )
        self.assertEqual(len(graph.triples), expected)

    def test_component_variables_are_disjoint(self):
        graph = penman.decode(experiment.compose_raw_amrs(FIRST, SECOND))
        variables = set(graph.variables())
        first = {value for value in variables if value.startswith("a")}
        second = {value for value in variables if value.startswith("b")}
        self.assertTrue(first)
        self.assertTrue(second)
        self.assertFalse(first & second)

    def test_composition_is_deterministic(self):
        self.assertEqual(
            experiment.compose_raw_amrs(FIRST, SECOND),
            experiment.compose_raw_amrs(FIRST, SECOND),
        )


class StatisticsTests(unittest.TestCase):
    def test_sign_test_confirms_consistent_loss(self):
        result = experiment.sign_test_loss([-0.1] * 30)
        self.assertTrue(result["confirmed_loss"])
        self.assertEqual(result["losses"], 30)
        self.assertTrue(
            math.isclose(result["one_sided_sign_p"], 2**-30)
        )

    def test_sign_test_does_not_confirm_balanced_changes(self):
        result = experiment.sign_test_loss([-0.1] * 15 + [0.1] * 15)
        self.assertFalse(result["confirmed_loss"])
        self.assertEqual(result["losses"], 15)
        self.assertEqual(result["gains"], 15)

    def test_frozen_budget_is_exact(self):
        per_pair = (
            50
            + 2 * 3 * experiment.REPEATS
            + experiment.REPEATS
        )
        self.assertEqual(per_pair, 85)
        self.assertEqual(per_pair * 60, experiment.EXPECTED_S2MATCH)


if __name__ == "__main__":
    unittest.main()
