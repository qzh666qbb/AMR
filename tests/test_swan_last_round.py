"""Check leakage, finite-sample limits and retention of missing documents."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_swan_last_round import CONDITIONS, analyze, calibrate, cp, q95
from scripts.repair_swan_last_round_inputs import problems
from scripts.run_swan_blind_eval import sha_file
from scripts.run_swan_last_round import validate_input_readiness
from src.core.swan_blind_protocol import ProtocolConfig, digest


class FinalRoundStatistics(unittest.TestCase):
    def test_actual_freeze_transaction_accepts_script_filename(self):
        import scripts.run_swan_last_round as runner
        docs = []
        for source, arm, split, n in (
                ("human", "negative", "calibration", 100),
                ("human", "negative", "test", 200),
                ("generated", "negative", "calibration", 100),
                ("generated", "negative", "test", 100),
                ("generated", "positive", "test", 50)):
            docs.extend({"document_id": f"{source}:{arm}:{split}:{i}",
                         "source": source, "arm": arm, "split": split,
                         "eligible": True} for i in range(n))
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp), Path(tmp)/"repair"
            (out/"private").mkdir(parents=True)
            (root/"scripts").mkdir()
            (root/"scripts/analyze_swan_last_round.py").write_text("# fixture\n")
            old = root/"experiments/swan_multiview_development1/blind_eval1/run2"
            old.mkdir(parents=True)
            (old/"heartbeat.json").write_text('{"state":"complete"}')
            data = out/"private/DOCUMENTS.json"
            data.write_text(json.dumps({"documents": docs}))
            files = {
                "CONTROL.json": {"deadline_at": "2099-01-01T00:00:00+00:00"},
                "CANDIDATE_SELECTION.json": {"method": "S8"},
                "SOURCE_FREEZE.json": {}, "GENERATION_FREEZE.json": {},
                "INPUT_AUDIT.json": {"all_frozen_documents_retained": True,
                    "source_family_overlap": 0, "dispatch_cap_pass": True,
                    "generated_eligible": 250, "input_manifest_sha256": sha_file(data)},
            }
            for name, value in files.items():
                (out/name).write_text(json.dumps(value))
            tokenizer = lambda texts, **kwargs: {"input_ids": [[1, 2, 3] for _ in texts]}
            fake_transformers = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda path: tokenizer))
            configs = {m: ProtocolConfig(m, parser_context_tokens=512) for m in ("P0", "S8")}
            with mock.patch.multiple(runner, ROOT=root, OUT=out, CACHE=root/"cache"), \
                    mock.patch.object(runner, "assets", return_value={}), \
                    mock.patch.object(runner, "configs_for", return_value=configs), \
                    mock.patch.object(runner, "bank_load", return_value=[]), \
                    mock.patch.object(runner, "inputs", return_value=([], {"key": "text"}, [])), \
                    mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
                value = runner.freeze()
            self.assertTrue((out/"EVAL_FREEZE.json").exists())
            self.assertEqual(value["implementation_sha256"], sha_file(runner.__file__))
            self.assertEqual(value["native_max_input_tokens"], 3)

    def test_sha_accepts_string_and_path_for_real_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture"
            path.write_bytes(b"abc")
            expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
            self.assertEqual(sha_file(path), expected)
            self.assertEqual(sha_file(str(path)), expected)

    def test_zero_eligible_generation_cannot_start_confirmation(self):
        docs = []
        for source, arm, split, n in (
                ("human", "negative", "calibration", 100),
                ("human", "negative", "test", 200),
                ("generated", "negative", "calibration", 100),
                ("generated", "negative", "test", 100),
                ("generated", "positive", "test", 50)):
            docs.extend({"document_id": f"{source}:{arm}:{split}:{i}",
                         "source": source, "arm": arm, "split": split,
                         "eligible": source == "human"} for i in range(n))
        audit = {"all_frozen_documents_retained": True, "source_family_overlap": 0,
                 "dispatch_cap_pass": True, "generated_eligible": 0}
        with self.assertRaisesRegex(RuntimeError, "Input readiness"):
            validate_input_readiness(audit, docs)
        for d in docs:
            d["eligible"] = True
        audit["generated_eligible"] = 250
        validate_input_readiness(audit, docs)

    def test_surface_repair_rejects_hidden_newline_and_wrong_finish(self):
        slot = {"text": "The local team completed repairs before sunset.",
                "finish_reason": "stop"}
        self.assertEqual(problems(slot, set()), [])
        slot["text"] = "The local team\ncompleted repairs before sunset."
        self.assertIn("newline", problems(slot, set()))
        slot["finish_reason"] = "length"
        self.assertIn("finish_reason", problems(slot, set()))

    def negatives(self, n=100):
        return [{
            "method": "P0_strict", "split": "calibration", "arm": "negative",
            "source": source, "attack": condition, "complete": True,
            "record_id": f"{source}:{condition}:{i}",
            "text_sha256": digest([source, condition, i]),
            "unit_bank_max": [.1, .7] if i % 10 == 0 else [.1, .1],
            "N_expected": 2,
        } for source in ("human", "generated") for condition in CONDITIONS
            for i in range(n)]

    def test_nontrivial_quantile_requires_19_complete_documents(self):
        self.assertIsNone(q95(list(range(18))))
        self.assertEqual(q95(list(range(19))), 18)
        self.assertEqual(q95(list(range(100))), 95)

    def test_test_text_and_positive_labels_cannot_change_calibration(self):
        calibration = self.negatives()
        test = {**calibration[0], "split": "test", "unit_bank_max": [1., 1.]}
        positive = {**calibration[1], "arm": "positive", "unit_bank_max": [1., 1.]}
        before = calibrate(calibration, "P0_strict")
        self.assertEqual(before, calibrate(calibration+[test, positive], "P0_strict"))
        altered = copy.deepcopy(test)
        altered["unit_bank_max"] = [0., 0.]
        self.assertEqual(before, calibrate(calibration+[altered], "P0_strict"))

    def test_insufficient_condition_is_unavailable(self):
        rows = self.negatives()
        for r in rows:
            if r["source"] == "human" and r["attack"] == "newline" \
                    and int(r["record_id"].split(":")[-1]) >= 18:
                r["complete"] = False
        result = calibrate(rows, "P0_strict")
        self.assertFalse(result["available"])
        self.assertIsNone(result["document_threshold"])
        self.assertEqual(result["groups"]["human:newline"]["frozen_denominator"], 100)
        self.assertEqual(result["groups"]["human:newline"]["complete"], 18)

    def test_simultaneous_fpr_interval_does_not_reuse_95pct_bound(self):
        self.assertGreater(cp(0, 100, .01)[1], .05)
        self.assertLess(cp(0, 160, .01)[1], .05)
        self.assertGreater(cp(0, 100, .01)[1], cp(0, 100, .05)[1])

    def test_partial_score_table_cannot_be_presented_as_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            frozen = {"candidate_method": "S8"}
            (p/"EVAL_FREEZE.json").write_text(json.dumps(frozen))
            rows = [{
                "method": m, "source": "human", "arm": "negative", "split": "test",
                "attack": "baseline", "record_id": m+":missing-rest",
            } for m in ("P0_raw", "P0_strict", "S8")]
            (p/"DOCUMENT_SCORES.json").write_text(json.dumps({
                "freeze_sha256": digest(frozen), "documents": rows}))
            with self.assertRaisesRegex(RuntimeError, "denominators"):
                analyze(p)


if __name__ == "__main__":
    unittest.main()
