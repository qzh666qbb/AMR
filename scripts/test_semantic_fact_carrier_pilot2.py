"""Meaningful isolated v2 checks; synthetic unit fixtures are not study evidence."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import run_semantic_fact_carrier_pilot2 as v2
from semantic_fact_carrier_v2 import classify_facts, generation_messages, paraphrase_messages, plan

runtime = v2.runtime


class CarrierTests(unittest.TestCase):
    def test_fixed_denominator_and_distinct_version_ids(self):
        rows = plan()
        self.assertEqual(len(rows), 96)
        self.assertEqual(len({r["item_id"] for r in rows}), 96)
        self.assertTrue(all(r["item_id"].startswith("v2-") for r in rows))
        self.assertEqual([r["intended_symbol"] for r in rows[:4]], list("ABCD"))

    def test_both_plan_assertions_are_narrator_level(self):
        prompt = generation_messages(plan()[2])[1]["content"]
        for required in ["CURRENT", "NOT happened yet", "outside speech",
                         "reserving it", "past abandoned plan"]:
            self.assertIn(required, prompt)

    def test_completion_is_received_not_reported(self):
        prompt = generation_messages(plan()[0])[1]["content"]
        self.assertIn("recipient has received it", prompt)
        self.assertIn("not something somebody says", prompt)

    def test_paraphrase_uses_only_actual_source(self):
        text = "UNIT FIXTURE ONLY; no target."
        messages = paraphrase_messages(text)
        self.assertTrue(messages[1]["content"].endswith(text))
        self.assertNotIn("intended_symbol", json.dumps(messages))
        self.assertNotIn("intended_facts", json.dumps(messages))
        self.assertIn("Do not add missing facts", messages[1]["content"])

    def test_simultaneity_is_not_replaced_by_and(self):
        messages = paraphrase_messages("UNIT FIXTURE ONLY")
        self.assertIn("explicitly simultaneous", messages[1]["content"])
        self.assertIn("not just become an 'and' list", messages[1]["content"])

    def test_all_directions_and_states_map(self):
        for row in plan()[:4]:
            self.assertEqual(classify_facts(row["intended_facts"],
                                           row["public_contract"]).symbol,
                             row["intended_symbol"])

    def test_uncertain_facts_cannot_be_saved_by_target(self):
        row = plan()[2]
        for target in "ABCD":
            facts = {**row["intended_facts"], "transfer_status": "unknown",
                     "intended_symbol": target}
            self.assertIsNone(classify_facts(facts, row["public_contract"]).symbol)

    def test_unknown_cannot_be_marked_clear_in_formal_schema(self):
        facts = {"text": "UNIT FIXTURE ONLY", "readability": "clear",
                 "sender": "unknown", "recipient": "unknown", "object": "unknown",
                 "transfer_status": "unknown", "explicit_status": "unknown",
                 "quality": "unknown", "notes": "Unit fixture",
                 "evidence_json": json.dumps(dict.fromkeys(runtime.FIELDS, []))}
        with self.assertRaises(ValueError):
            runtime.validate_fact(facts)
        runtime.validate_fact({**facts, "readability": "unclear"})


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="amr_carrier_v2_test_")
        self.base = Path(self.temporary.name)
        self.out, self.private = self.base / "out", self.base / "private"
        self.out.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def freeze(self):
        (self.out / "PROTOCOL.md").write_text("Unit fixture, not an experiment.")
        runtime.atomic(self.out / "PRICE_SOURCE.json", {"fixture_only": True})
        runtime.atomic(self.out / "TEST_RESULT.json", {
            "status": "PASS", "tests_run": 14, "fixture_only": True,
            "code_sha256": {s: runtime.sha(v2.ROOT / s) for s in v2.SOURCES}})
        return v2.prepare(self.out, self.private)

    @staticmethod
    def transport(calls, completion_tokens=4):
        def call(*args, **kwargs):
            calls.append(kwargs["json"])
            class Response:
                status_code = 200
                text = json.dumps({
                    "model": "UNIT_FIXTURE", "choices": [{
                        "finish_reason": "stop", "message": {"content": "Unit fixture only."}}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": completion_tokens}})
            return Response()
        return call

    def test_new_freeze_is_immutable_and_checks_all_sources(self):
        self.freeze()
        self.assertTrue(all(runtime.verify(self.out, self.private).values()))
        with self.assertRaises(RuntimeError):
            v2.prepare(self.out, self.private)
        runtime.atomic(self.private / "PLAN.json", [])
        with self.assertRaises(runtime.TerminalHalt):
            runtime.verify(self.out, self.private)

    def test_at_most_once_replay_on_restart(self):
        calls = []
        provider = runtime.Provider(self.out, self.private, "UNIT_FIXTURE", self.transport(calls))
        messages = [{"role": "user", "content": "Unit fixture"}]
        first = provider.request("fixture", messages)
        restart = runtime.Provider(self.out, self.private, "UNIT_FIXTURE", self.transport(calls))
        self.assertEqual(first, restart.request("fixture", messages))
        self.assertEqual(len(calls), 1)

    def test_uncertain_request_permanently_halts(self):
        runtime.atomic(self.private / "requests/fixture/INTENT.json", {"body_sha256": "fixture"})
        with self.assertRaises(runtime.TerminalHalt):
            runtime.Provider(self.out, self.private, "UNIT_FIXTURE")
        self.assertTrue((self.private / "HALT.json").is_file())

    def test_capacity_rejects_before_dispatch(self):
        calls = []
        provider = runtime.Provider(self.out, self.private, "UNIT_FIXTURE", self.transport(calls))
        with self.assertRaises(runtime.TerminalHalt):
            provider.request("fixture", [{"role": "user", "content": "x" * 5000}])
        self.assertEqual(calls, [])
        self.assertEqual(list((self.private / "requests").glob("*/INTENT.json")), [])

    def test_actual_usage_above_bound_halts_without_retry(self):
        calls = []
        provider = runtime.Provider(self.out, self.private, "UNIT_FIXTURE",
                                    self.transport(calls, completion_tokens=321))
        with self.assertRaises(runtime.TerminalHalt):
            provider.request("fixture", [{"role": "user", "content": "Unit fixture"}])
        self.assertEqual(len(calls), 1)
        with self.assertRaises(runtime.TerminalHalt):
            runtime.Provider(self.out, self.private, "UNIT_FIXTURE", self.transport(calls))

    def test_source_failure_remains_in_complete_denominator(self):
        self.freeze()
        runtime.summarize(self.out, self.private, "UNIT_FIXTURE_NOT_GENERATED")
        summary = runtime.load(self.out / "GENERATION_SUMMARY.json")
        self.assertEqual(summary["counts"], {"original:not_generated": 96,
                                           "paraphrase:not_generated": 96})
        self.assertIsNone(summary["semantic_writing_success"])
        self.assertEqual(summary["human_gate"], "NOT_EVALUATED")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if len(sys.argv) == 3 and sys.argv[1] == "--evidence":
        runtime.atomic(Path(sys.argv[2]), {
            "at": runtime.now(), "status": "PASS" if result.wasSuccessful() else "FAIL",
            "tests_run": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "api_calls": 0,
            "unit_fixtures_not_experiment_results": True,
            "code_sha256": {s: runtime.sha(v2.ROOT / s) for s in v2.SOURCES}})
    raise SystemExit(0 if result.wasSuccessful() else 1)
