"""Offline tests for paired grounded prompts, budget and quality gates."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from run_confirmation_experiment import request_body
from run_grounded_generation_comparison import (
    COMMON, Journal, decode_fact_review, evaluate, fact_body, generation_body, source_document, summarize,
)
from test_confirmation_execution import response


class GroundedTests(unittest.TestCase):
    def style(self):
        return {"fluency": 4, "coherence": 4, "nonredundancy": 4, "severe_problem": False}

    def facts(self):
        return {"faithfulness": 4, "unsupported_claim": False, "contradiction": False, "issues": []}

    def fixture(self):
        sentences = [" ".join([f"word{i}"] * 20) + "." for i in range(5)]
        text = " ".join(sentences)
        responses = [{"content": s, "finish_reason": "stop"} for s in sentences]
        tokenize = lambda value: [s.strip() + "." for s in value.split(".") if s.strip()]
        return text, responses, tokenize

    def test_shared_source_rules_and_total_token_cap(self):
        seq = generation_body("SOURCE FIXTURE", "sequential", 1, ["Earlier fact."])
        full = generation_body("SOURCE FIXTURE", "onepass", 0, [])
        for body in (seq, full):
            self.assertTrue(body["messages"][-1]["content"].startswith(COMMON))
            self.assertTrue(body["messages"][-1]["content"].endswith("SOURCE FIXTURE"))
        self.assertIn("Earlier fact.", seq["messages"][-1]["content"])
        self.assertNotIn("Earlier fact.", full["messages"][-1]["content"])
        self.assertEqual(seq["max_tokens"] * 5, full["max_tokens"])
        self.assertEqual(seq["messages"][0], full["messages"][0])
        with self.assertRaises(ValueError):
            generation_body("s", "onepass", 0, ["leaked paired output"])

    def test_bounded_identical_excerpt(self):
        record = {"text": " ".join(str(i) for i in range(1100))}
        self.assertEqual(len(source_document(record).split()), 1000)
        self.assertEqual(source_document(record).split()[-1], "999")

    def test_fact_review_rejects_invalid_types_and_missing_evidence(self):
        self.assertEqual(decode_fact_review(json.dumps(self.facts())), self.facts())
        for override in ({"faithfulness": True}, {"unsupported_claim": "false"},
                         {"contradiction": True}, {"issues": [{}]}):
            with self.assertRaises(ValueError):
                decode_fact_review(json.dumps({**self.facts(), **override}))

    def test_factual_reviews_contain_source_without_arm_label(self):
        body = fact_body("supported source", "candidate")
        prompt = body["messages"][-1]["content"]
        self.assertIn("supported source", prompt)
        self.assertIn("candidate", prompt)
        self.assertNotIn("sequential", prompt)
        self.assertNotIn("onepass", prompt)

    def test_joint_gate_and_failure_dimensions(self):
        text, responses, tokenize = self.fixture()
        good = evaluate(text, responses, "sequential", self.style(), self.facts(), tokenize)
        self.assertTrue(good["source_feasibility_pass"])
        self.assertEqual(good["words"], 100)
        bad = evaluate(text, responses, "sequential", self.style(), {
            **self.facts(), "issues": [{"claim": "c", "evidence": "", "reason": "r"}]}, tokenize)
        self.assertFalse(bad["gates"]["factual_pass"])
        responses[0]["finish_reason"] = "length"
        self.assertFalse(evaluate(text, responses, "sequential", self.style(), self.facts(), tokenize)["gates"]["structure_pass"])

    def test_onepass_wording_limits_use_same_sentence_gate(self):
        text, responses, tokenize = self.fixture()
        merged = [{"content": text, "finish_reason": "stop"}]
        self.assertTrue(evaluate(text, merged, "onepass", self.style(), self.facts(), tokenize)["source_feasibility_pass"])
        responses[0]["content"] = "Too short."
        bad_text = " ".join(r["content"] for r in responses)
        result = evaluate(bad_text, [{"content": bad_text, "finish_reason": "stop"}],
                          "onepass", self.style(), self.facts(), tokenize)
        self.assertFalse(result["gates"]["sentence_length_pass"])

    def test_cumulative_cap_replay_has_no_new_call(self):
        with tempfile.TemporaryDirectory() as d, patch("run_grounded_generation_comparison.CAP", 1):
            folder = Path(d)
            body = request_body("OK", 0, 16)
            with patch("requests.post", return_value=response()):
                Journal(folder, "", Mock()).request("probe", body)
            resumed = Journal(folder, "", Mock())
            with patch("requests.post", side_effect=AssertionError("No network")):
                resumed.request("probe", body)
                with self.assertRaises(RuntimeError):
                    resumed.request("another", body)
            self.assertEqual(resumed.sent, 0)

    def test_partial_results_do_not_trigger_advance(self):
        from run_confirmation_experiment import freeze
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            freeze(folder / "manifest.json", {"prompts": [{"prompt_id": f"p{i}"} for i in range(3)]})
            for pid in ("p0", "p1"):
                freeze(folder / "evaluated" / f"{pid}-sequential.json", {
                    "prompt_id": pid, "arm": "sequential", "source_feasibility_pass": True,
                    "gates": {k: True for k in ("style_pass", "factual_pass", "structure_pass",
                                               "total_length_pass", "sentence_length_pass")}})
            result = summarize(folder)
            self.assertEqual(result["groups"]["sequential"]["joint_pass"], 2)
            self.assertEqual(result["ready_for_injection_pilot_preparation"], [])


if __name__ == "__main__":
    unittest.main()
