"""Regression tests for offline retry validation and score decomposition."""

import copy
import unittest

from analyze_sentence_count_decomposition import decompose
from evaluate_adaptive_retry import best_valid, evaluate_rows, gates


def candidate(index, number=0):
    return {"index": index, "candidate": number, "source": "A person runs.",
            "attack": "A person runs."}


def reviewed(row):
    return {**row, "review": {"score": 5, "major_error": False}}


class RetryTests(unittest.TestCase):
    def test_second_group_uses_its_own_graph(self):
        rows = [candidate(10), candidate(20)]
        parsed = [["(a / run-01)"], ["not a graph"]]
        result = evaluate_rows(rows, [0., 0.], parsed, [reviewed(r) for r in rows])
        self.assertTrue(result[0]["quality_valid"])
        self.assertFalse(result[1]["quality_valid"])
        self.assertEqual(result[1]["position"], 1)

    def test_review_order_does_not_affect_alignment(self):
        rows = [candidate(10), candidate(20)]
        reviews = [reviewed(r) for r in rows]
        reviews[1]["review"]["major_error"] = True
        result = evaluate_rows(rows, [0., 0.], [["(a / run)"]] * 2, reviews[::-1])
        self.assertTrue(result[0]["quality_valid"])
        self.assertFalse(result[1]["quality_valid"])

    def test_empty_and_malformed_amr_fail(self):
        for graphs in ([], [""], ["(a / run"], ["(a / run)", "bad"]):
            self.assertFalse(gates(candidate(0), graphs)["amr_ok"])

    def test_reject_misaligned_inputs(self):
        row = candidate(0)
        for scores, parsed, reviews in (
            ([], [["(a / run)"]], [reviewed(row)]),
            ([0.], [], [reviewed(row)]),
            ([0.], [["(a / run)"]], [reviewed(candidate(1))]),
        ):
            with self.assertRaises(ValueError):
                evaluate_rows([row], scores, parsed, reviews)
        bad = reviewed(row)
        bad["attack"] = "Different content."
        with self.assertRaises(ValueError):
            evaluate_rows([row], [0.], [["(a / run)"]], [bad])

    def test_duplicate_identity_and_stale_score_rejected(self):
        row = candidate(0)
        with self.assertRaises(ValueError):
            evaluate_rows([row, row], [0., 0.], [["(a / run)"]] * 2, [reviewed(row)] * 2)
        with self.assertRaises(ValueError):
            evaluate_rows([row], [1.], [["(a / run)"]], [{**reviewed(row), "z_score": 0.}])

    def test_no_mutation_and_best_valid_tie(self):
        rows = [candidate(0, 1), candidate(0, 0), candidate(1)]
        original = copy.deepcopy(rows)
        result = evaluate_rows(rows, [0., 0., -1.], [["(a / run)"], ["(b / run)"], []],
                               [reviewed(r) for r in rows])
        self.assertEqual(rows, original)
        self.assertEqual([(r["index"], r["candidate"]) for r in best_valid(result)], [(0, 0)])


class DecompositionTests(unittest.TestCase):
    def test_no_changes(self):
        result = decompose(5, 4, 5, 4)
        self.assertEqual(result["count_drop"], 0)
        self.assertEqual(result["rate_drop"], 0)

    def test_count_only(self):
        result = decompose(6, 3, 2, 1)
        self.assertAlmostEqual(result["rate_drop"], 0)
        self.assertAlmostEqual(result["count_drop"], result["baseline"] - result["observed"])

    def test_rate_only(self):
        result = decompose(5, 5, 5, 0)
        self.assertAlmostEqual(result["count_drop"], 0)
        self.assertAlmostEqual(result["rate_drop"], result["baseline"] - result["observed"])

    def test_fractional_hybrid_and_additivity(self):
        result = decompose(5, 4, 3, 1)
        self.assertAlmostEqual(result["count_drop"] + result["rate_drop"],
                               result["baseline"] - result["observed"])
        self.assertGreater(result["count_drop"], 0)
        self.assertGreater(result["rate_drop"], 0)

    def test_count_reduction_can_raise_negative_score(self):
        result = decompose(5, 0, 1, 0)
        self.assertLess(result["count_drop"], 0)
        self.assertEqual(result["rate_drop"], 0)


if __name__ == "__main__":
    unittest.main()
