"""Synthetic exhaustive reference tests for three-arm observation capacity."""
import itertools
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.core import atomic_alignment_arms as arms
from src.core.multiview_views import Event


def candidate(i):
    return {"candidate_index": i, "start_char": i * 10, "end_char": i * 10 + 8,
            "span_sha256": str(i), "graph_body_sha256": "graph" + str(i), "usable": True}


def event(i, ci=0):
    return Event(str(i), ci, "x", "give-01", ((":ARG0", "person[Ada]"),),
                 False, (), (), "Ada gives", "same-identity", False)


class ArmTests(unittest.TestCase):
    def test_a0_no_solver_or_grouping(self):
        with patch.object(arms, "milp", side_effect=AssertionError):
            result = arms.no_alignment([candidate(0)], [event(i) for i in range(5)])
        self.assertEqual([p["event_id"] for p in result["selected"]], ["0", "1", "2"])

    def test_a1_does_not_merge_same_signature_observations(self):
        result = arms.predicted([candidate(0)], [event(0), event(1)],
                                np.array([[1., 0]]), np.array([[1., 0], [1., 0]]), [{"clauses": []}])
        self.assertEqual(len(result["selected"]), 2)

    def test_oracle_recovers_edges_excluded_by_prediction(self):
        cs, es = [candidate(0), candidate(1)], [event(0)]
        pred = arms.predicted(cs, es, np.array([[1., 0], [0., 1]]),
                              np.array([[0., 1]]), [{"clauses": []}] * 2)
        self.assertEqual(pred["selected"], [])
        result = arms.oracle(cs, es, {(0, "0"): False, (1, "0"): True})
        self.assertEqual(result["selected"][0]["candidate_index"], 1)

    def test_oracle_entire_universe_required(self):
        with self.assertRaises(ValueError):
            arms.oracle([candidate(0), candidate(1)], [event(0)], {(0, "0"): True})
        with self.assertRaises(ValueError):
            arms.oracle([candidate(0)], [event(0)], {(0, "0"): None})

    def test_oracle_matches_exhaustive_all_small_graphs(self):
        cs, es = [candidate(0), candidate(1)], [event(i) for i in range(3)]
        common = arms.universe(cs, es)
        for flags in itertools.product((False, True), repeat=6):
            valid = [i for i, yes in enumerate(flags) if yes]
            feasible = []
            for selected_flags in itertools.product((False, True), repeat=len(valid)):
                selected = [common[i] for i, yes in zip(valid, selected_flags) if yes]
                if len({e["event_id"] for e in selected}) != len(selected):
                    continue
                if any(sum(e["candidate_index"] == ci for e in selected) > 1 for ci in range(2)):
                    continue
                feasible.append(selected)
            best = sorted(feasible, key=lambda x: (-len(x), [(e["event_id"], e["candidate_index"]) for e in x]))[0]
            support = {(e["candidate_index"], e["event_id"]): v for e, v in zip(common, flags)}
            actual = arms.oracle(cs, es, support, arms.AlignmentConfig(max_events_per_candidate=1))
            self.assertEqual(actual["selected"], best)

    def test_candidate_payload_rejects_gold_and_scores(self):
        for field in ("target_template", "source_fact_id", "bank_scores"):
            with self.assertRaises(ValueError):
                arms.no_alignment([{**candidate(0), field: 1}], [event(0)])

    def test_graph_count_is_shared_deduplication(self):
        observations = [{"usable": True, "graph_body_sha256": "a",
                         "bank_scores": [{"score": .8}]}] * 2
        result = arms.fixed_statistic([{"event_candidate_index": 0}, {"event_candidate_index": 1}], observations)
        self.assertEqual(result["unique_graphs"], 1)
        self.assertAlmostEqual(result["statistic"], .15)


if __name__ == "__main__":
    unittest.main()
