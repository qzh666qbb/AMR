"""Synthetic evidence only: payment replay, count neutrality, pretest selection."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_swan_normal_completion1 as normal
import swan_normal_alignment1 as aligned
from swan_final1_common import freeze, sha256
from swan_atomic_calibration1 import reject


class NormalTests(unittest.TestCase):
    def test_same_count_and_exact_words(self):
        sentences = [f"Researcher Alice carefully studied sample number {i}." for i in range(16)]
        tokenize = lambda text: [s.strip() + "." for s in text.split(".") if s.strip()]
        variants, evidence = normal.pair_rows("s", sentences, lambda _: {"complete": True}, tokenize)
        self.assertEqual([len(tokenize(variants[c])) for c in ("A", "B")], [15, 15])
        self.assertEqual(normal.WORD.findall(variants["A"]), normal.WORD.findall(variants["B"]))
        self.assertTrue(evidence["checks"]["punctuation_multiset_equal"])

    def test_no_complete_clause_retains_failure(self):
        variants, evidence = normal.pair_rows("s", ["Sentence."]*16,
                                             lambda _: {"complete": False},
                                             lambda t: ["Sentence."]*16)
        self.assertEqual(variants, {"A": None, "B": None})
        self.assertIn("reason", evidence)

    def test_calibration_ties_and_failure_denominator(self):
        rows = [{"source_id": str(i), "role": "calibration", "condition": c,
                 "score": 1 if i else None} for i in range(100) for c in normal.CONDITIONS]
        cuts = normal.calibration(rows)
        self.assertEqual(cuts["A"]["calibration_n"], 100)
        self.assertFalse(reject(1, cuts["A"]))
        self.assertTrue(reject(2, cuts["A"]))
        with self.assertRaises(ValueError):
            normal.calibration(rows[:-1])
        with self.assertRaises(ValueError):
            normal.calibration(rows + rows[:1])

    def test_evaluation_all_generation_failures_remain(self):
        rows = [{"source_id": role+str(i), "role": role, "condition": c, "score": None}
                for role in ("positive", "null_test") for i in range(10) for c in normal.CONDITIONS]
        cuts = {c: {"value": 1, "infinity": None} for c in normal.CONDITIONS}
        result = normal.evaluate(rows, cuts)
        self.assertEqual(result["positive"]["B"]["own_condition_threshold"]["n"], 10)
        self.assertEqual(result["count_neutral"]["complete_pairs"], 0)

    def test_null_paid_call_replay_and_identity_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            freeze(out / "FREEZE.json", {"synthetic": True})
            calls = []
            def send(body):
                calls.append(body)
                return {"id": "test", "model": normal.budget.MODEL,
                        "usage": {"prompt_tokens": 10, "completion_tokens": 12},
                        "choices": [{"message": {"content": "Alice carefully inspected the newly repaired laboratory equipment."},
                                     "finish_reason": "stop"}]}
            with patch.object(normal.budget, "transport", side_effect=lambda body, key: send(body)):
                j = normal.journal(out, "llm_null", "SYNTHETIC_NO_KEY")
                response = normal.null_slot(out, {"allocation_id": "x"}, 0, "context", j, lambda t: [t])
                j2 = normal.journal(out, "llm_null", "SYNTHETIC_NO_KEY")
                replay = normal.null_slot(out, {"allocation_id": "x"}, 0, "context", j2, lambda t: [t])
                self.assertEqual(response, replay)
                self.assertEqual(len(calls), 1)
                with self.assertRaises(RuntimeError):
                    normal.null_slot(out, {"allocation_id": "x"}, 0, "changed", j2, lambda t: [t])

    def test_null_truncation_never_becomes_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            freeze(out / "FREEZE.json", {"synthetic": True})
            class J:
                def request(self, *args):
                    return {"text": "Alice carefully inspected the newly repaired laboratory equipment.",
                            "stop_reason": "length"}
            r = normal.null_slot(out, {"allocation_id": "x"}, 0, "context", J(), lambda t: [t])
            self.assertFalse(r["accepted"])
            self.assertEqual(len(r["call_ids"]), 50)

    def test_seal_detects_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            freeze(out / "FREEZE.json", {"x": 1})
            freeze(out / "input.json", [1])
            normal.file_seal(out, out / "SEAL.json", [out / "input.json"])
            normal.verify_seal(out, out / "SEAL.json")
            (out / "input.json").write_text("[2]")
            with self.assertRaises(RuntimeError):
                normal.verify_seal(out, out / "SEAL.json")


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.expected = {f"cal{i}": {"role": "calibration", "kind": "human"} for i in range(100)}
        self.expected.update({"dev": {"role": "development", "kind": "watermarked"}})
        self.meta = [{"row_id": sid+c, "source_id": sid, "condition": c, **m}
                     for sid, m in self.expected.items() for c in normal.CONDITIONS]
        self.pred = []
        for m in self.meta:
            zero = {"statistic": 0, "unique_graphs": 1}
            families = {k: dict(zero) for k in normal.FAMILIES}
            if m["source_id"] == "dev":
                families["word_window_24"]["statistic"] = 1
            self.pred.append({"row_id": m["row_id"], "A0": zero, "A1": zero,
                              "single_view_families": families})

    def test_strong_baseline_chosen_only_on_development(self):
        selected = aligned.baseline_and_cuts(self.pred, self.meta, self.expected)
        self.assertEqual(selected["selected_family"], "word_window_24")
        self.assertFalse(selected["test_rows_read"])
        self.meta[-1]["role"] = "positive_test"
        with self.assertRaises(ValueError):
            aligned.baseline_and_cuts(self.pred, self.meta, self.expected)

    def test_missing_condition_cannot_select_baseline(self):
        with self.assertRaises(ValueError):
            aligned.baseline_and_cuts(self.pred[:-1], self.meta, self.expected)

    def test_missing_oracle_not_false_or_abstain(self):
        with self.assertRaises(ValueError):
            aligned.endpoint_rows(self.pred, self.meta, "whole_text", {})
        oracle = {p["row_id"]: None for p in self.pred}
        with self.assertRaises(ValueError):
            aligned.endpoint_rows(self.pred, self.meta, "whole_text", oracle)

    def test_empty_observation_has_explicit_abstention(self):
        for p in self.pred:
            p["A0"] = {"statistic": 0, "unique_graphs": 0}
        rows = aligned.endpoint_rows(self.pred, self.meta, "whole_text")
        self.assertTrue(all(r["scores"]["A0"] is None for r in rows))
        self.assertTrue(all(r["abstentions"]["A0"] == "no_usable_observation" for r in rows))


if __name__ == "__main__":
    unittest.main()
