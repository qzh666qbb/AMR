"""Independent small-path oracles and leakage/denominator regression checks."""
import itertools
import math
from pathlib import Path
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/core"))
from semantic_sequence import (
    average_window_weights, equal_composition_sequences, exact_costs,
    rank_truth, shared_candidates, symbol_weights, unordered_costs,
)
from run_semantic_sequence_cycle import carrier_messages, required_markers


def exhaustive_cost(observations, candidate):
    """Enumerate every edit path, without a DP table or memoization."""
    if not observations:
        return len(candidate)
    if len(candidate) == 0:
        return len(observations)
    q = observations[0]
    cost = 1 if q is None else -math.log(max(q[candidate[0]], 1e-6)) / math.log(4)
    return min(cost + exhaustive_cost(observations[1:], candidate[1:]),
               1 + exhaustive_cost(observations[1:], candidate),
               1 + exhaustive_cost(observations, candidate[1:]))


class CycleChecks(unittest.TestCase):
    def test_all_arrangements_and_unordered_control(self):
        rows = equal_composition_sequences()
        self.assertEqual(rows.shape, (2520, 8))
        self.assertEqual(len({tuple(r) for r in rows}), 2520)
        self.assertTrue(all(list(r).count(i) == 2 for r in rows for i in range(4)))
        losses = unordered_costs([[.1, .2, .3, .4], None], rows)
        self.assertTrue(np.all(losses == losses[0]))
        rank = rank_truth(losses, rows, rows[37])
        self.assertEqual(rank["alternative_count"], 2519)
        self.assertEqual(rank["pairwise_win_rate"], .5)
        self.assertEqual(rank["average_rank"], 1260.5)
        self.assertEqual(rank["top1_credit"], 1 / 2520)

    def test_soft_dp_exhaustive_all_small_paths(self):
        options = [None, [.6, .2, .1, .1], [.1, .2, .3, .4]]
        count = 0
        for m in range(4):
            for observations in itertools.product(options, repeat=m):
                for n in range(1, 4):
                    candidates = np.asarray(list(itertools.product(range(2), repeat=n)))
                    actual = exact_costs(observations, candidates)
                    expected = [exhaustive_cost(observations, row) for row in candidates]
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
                    count += len(candidates)
        self.assertEqual(count, 560)

    def test_perfect_order_and_loss_of_order(self):
        rows = equal_composition_sequences()
        truth = rows[137]
        weights = np.eye(4)[truth]
        for mode in ("hard", "soft"):
            losses = exact_costs(weights, rows, mode)
            self.assertEqual(rank_truth(losses, rows, truth)["pairwise_win_rate"], 1)
            self.assertEqual(rank_truth(losses, rows, truth)["top1_credit"], 1)
            replay = exact_costs(weights, rows, mode)
            self.assertTrue(np.array_equal(losses, replay))
        flat = exact_costs([[.25] * 4] * 8, rows, "hard")
        self.assertEqual(rank_truth(flat, rows, truth)["pairwise_win_rate"], .5)

    def test_missing_window_erases_anchor_not_other_anchors(self):
        q = np.asarray([.1, .2, .3, .4])
        self.assertIsNone(average_window_weights([q, None]))
        np.testing.assert_array_equal(average_window_weights([q, q]), q)
        self.assertIsNone(symbol_weights([.3, np.nan, .4, .5], [[0], [1], [2], [3]], .1))
        with self.assertRaises(ValueError):
            symbol_weights([.3] * 4, [[0], [1], [2], [2]], .1)

    def test_dedup_by_occurrence_and_common_windows(self):
        text = "Mira repairs carts. Mira repairs carts."
        positions = [5, 25]
        nlp = lambda _: [types.SimpleNamespace(idx=i, text="repairs", lemma_="repair", pos_="VERB")
                         for i in positions]
        result = shared_candidates(text, nlp)
        self.assertEqual(len(result["observations"]), 2)
        self.assertNotEqual(result["observations"][0]["observation_id"],
                            result["observations"][1]["observation_id"])
        self.assertEqual(result["observations"][0]["windows"][1]["text"],
                         result["observations"][1]["windows"][1]["text"])
        for row in result["observations"]:
            for window in row["windows"]:
                a, b = window["span"]
                self.assertEqual(window["text"], text[a:b])
                self.assertLessEqual(a, row["anchor_span"][0])
                self.assertGreaterEqual(b, row["anchor_span"][1])

    def test_budget_is_abstention_not_prefix_truncation(self):
        text = "run " * 33
        nlp = lambda _: [types.SimpleNamespace(idx=i * 4, text="run", lemma_="run", pos_="VERB")
                         for i in range(33)]
        result = shared_candidates(text, nlp)
        self.assertEqual(result["status"], "budget_abstain")
        self.assertEqual(result["observed_anchor_count"], 33)
        self.assertEqual(result["observations"], [])
        self.assertFalse(result["truncated"])
        with self.assertRaises(TypeError):
            shared_candidates({"text": text, "gold_slots": [1]}, nlp)

    def test_prompt_repairs_and_matched_surface_contract(self):
        template = '(p / V :ARG1 (x / NE :name (n / N :op1 "NE" :op2 "NE")) :polarity (u / X))'
        markers = required_markers(template)
        self.assertEqual(markers["name_sizes"], [2])
        self.assertEqual(markers["unknown_nodes"], 1)
        positive = carrier_messages("Fictional scene.", "", 0, markers, template)[1]["content"]
        negative = carrier_messages("Fictional scene.", "", 0, markers)[1]["content"]
        self.assertIn("amr-unknown", positive)
        self.assertIn("unknown polarity", positive)
        self.assertIn("unknown polarity", negative)
        self.assertIn("12 to 22", negative)
        self.assertNotIn("TARGET", negative)
        self.assertNotIn(":ARG1", negative)
        self.assertNotIn("another unspecified concept", positive)


if __name__ == "__main__":
    unittest.main()
