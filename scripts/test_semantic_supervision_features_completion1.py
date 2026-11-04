import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts")]
import run_semantic_supervision_features_completion1 as feature
from src.core.semantic_fact_core_matcher import representation
from test_semantic_fact_core_matcher import CONTRACT, GRAPH


def options():
    return [{"key": str(scene) + symbol, "candidate_index": 0, "scene_id": scene, "symbol": symbol}
            for scene in range(12) for symbol in "ABCD"]


class FeatureTests(unittest.TestCase):
    def test_balanced_complete_dictionary(self):
        selected = feature.choose_templates(options(), range(12))
        self.assertEqual(len(selected), 24)
        self.assertEqual(feature.Counter(r["symbol"] for r in selected), dict.fromkeys("ABCD", 6))
        self.assertEqual(feature.Counter(r["scene_id"] for r in selected), dict.fromkeys(range(12), 2))
        self.assertEqual(selected, feature.choose_templates(list(reversed(options())), range(12)))

    def test_insufficient_human_coverage_no_padding(self):
        with self.assertRaises(RuntimeError):
            feature.choose_templates([r for r in options() if r["symbol"] != "D"], range(12))
        with self.assertRaises(ValueError):
            feature.choose_templates(options() + [{"scene_id": 99, "symbol": "A"}], range(12))

    def test_maxflow_can_reassign_greedy_scene_choices(self):
        rows = [r for r in options() if r["scene_id"] < 6 or r["symbol"] in "AB"]
        selected = feature.choose_templates(rows, range(12))
        self.assertEqual(len(selected), 24)
        self.assertEqual({r["symbol"] for r in selected if r["scene_id"] < 6}, set("CD"))

    def test_full_shape_and_view_abstention(self):
        rep = representation(GRAPH, CONTRACT, "receive", ["receive"])
        bank = {"templates": [rep] * 24}
        doc = {"candidates": [None, None]}
        def observe(_doc, ci, vi, _lemma):
            value = rep if (ci, vi) != (1, 1) else {"status": "invalid", "reason": "raw_graph_ineligible"}
            return value, {"amr_sha256": "a" * 64}
        x, evidence = feature.score_document(doc, [None] * 2, bank, observe=observe)
        self.assertEqual(x.shape, (2, 3, 24))
        self.assertEqual(x.dtype, np.float32)
        self.assertTrue(np.isnan(x[1, 1]).all())
        self.assertTrue((x[0] == 1).all())
        self.assertEqual(evidence[1]["status"], "invalid")
        x, evidence = feature.score_document(doc, [None] * 2, bank, 1, observe)
        self.assertTrue(np.isnan(x).all())
        self.assertEqual(evidence[0]["status"], "budget_abstain")

    def test_labels_cannot_change_features(self):
        rep = representation(GRAPH, CONTRACT, "receive", ["receive"])
        bank = {"templates": [rep] * 24}
        observe = lambda *args: (rep, {"amr_sha256": "a" * 64})
        one = {"candidates": [None], "planned_symbol": "A", "actual_symbol": "B"}
        two = {**one, "planned_symbol": "D", "actual_symbol": "C"}
        a = feature.score_document(one, [None], bank, observe=observe)
        b = feature.score_document(two, [None], bank, observe=observe)
        np.testing.assert_array_equal(a[0], b[0])
        self.assertEqual(a[1], b[1])

    def test_training_selector_only_opens_train_labels(self):
        # Synthetic receipts test the access boundary, not scientific results.
        labels = {r["key"]: {"condition": "original", "scene_id": r["scene_id"],
                  "candidate_labels": [{"index": 0, "kind": "carrier", "symbol": r["symbol"]}],
                  "planned_symbol": "UNREADABLE_TARGET"} for r in options()}
        docs = [{"key": k, "contract": {}, "candidates": [None]} for k in labels]
        class Runner:
            def role_labels(self, role):
                if role != "train":
                    raise AssertionError("Heldout labels opened")
                return labels
            def gold_seal(self):
                return {"role_label_sha256": {"train": {}}}
        values = {
            "FEATURE_REQUEST_NO_TARGET.json": {"documents": docs},
            "BLIND_LEMMAS.json": {k: [None] for k in labels},
            "MEMBERSHIP.json": {k: r["scene_id"] for k, r in labels.items()},
            "SPLITS.json": {"groups": {"train": list(range(12))}},
        }
        actual_by_key = {k: r["candidate_labels"][0]["symbol"] for k, r in labels.items()}
        def observe(doc, *args):
            return {"status": "complete"}, {"amr": actual_by_key[doc["key"]], "amr_sha256": "hash"}
        with patch.object(feature, "verify"), patch.object(feature, "graph_seal"), \
                patch.object(feature, "bind", return_value=(None, Runner())), \
                patch.object(Path, "exists", return_value=False), \
                patch.object(feature, "load", side_effect=lambda p: copy.deepcopy(values[p.name])), \
                patch.object(feature, "observation", side_effect=observe), \
                patch.object(feature, "read_graph_facts", side_effect=lambda amr, _: {"prediction": amr}), \
                patch.object(feature, "sha", return_value="sealed"), \
                patch.object(feature, "immutable") as save, \
                patch.object(feature, "verify_templates", return_value={}):
            feature.templates()
        bank = save.call_args_list[0].args[1]
        self.assertEqual(len(bank["templates"]), 24)
        self.assertTrue(all(r["symbol"] in "ABCD" for r in bank["provenance"]))

    def test_empty_candidate_denominator_retained(self):
        x, evidence = feature.score_document({"candidates": []}, [], {"templates": []})
        self.assertEqual(x.shape, (0, 3, 24))
        self.assertEqual(evidence, [])


if __name__ == "__main__":
    unittest.main()
