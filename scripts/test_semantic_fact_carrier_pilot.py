"""Unit fixtures only, never experimental evidence or human labels."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_semantic_fact_carrier_pilot as pilot
from semantic_fact_carrier import (classify_facts, generation_messages,
                                   plan, validate_evidence)


class FactTests(unittest.TestCase):
    def test_alphabet_directions_and_states(self):
        contract = {"role_one": "Mina", "role_two": "Theo", "object": "label box"}
        for symbol, (sender, recipient, state) in pilot.SYMBOLS.items():
            facts = {"sender": contract[sender], "recipient": contract[recipient],
                     "object": "label box", "transfer_status": state}
            self.assertEqual(classify_facts(facts, contract).symbol, symbol)

    def test_ambiguous_status_and_confusers_are_erased(self):
        contract = {"role_one": "Mina", "role_two": "Theo", "object": "label box"}
        facts = {"sender": "Mina", "recipient": "Theo", "object": "label box",
                 "transfer_status": "completed"}
        for patch in ({"transfer_status": "future"}, {"transfer_status": "promised"},
                      {"transfer_status": "unknown"}, {"sender": "Nora"},
                      {"recipient": "Mina"}, {"object": "lens case"},
                      {"contradiction": True}, {"object": ""}):
            self.assertIsNone(classify_facts({**facts, **patch}, contract).symbol)
        self.assertIsNone(classify_facts(facts, {**contract, "role_two": "Mina"}).symbol)

    def test_target_cannot_change_fact_prediction(self):
        row = plan()[0]
        facts = row["intended_facts"]
        for fake in "ABCD":
            self.assertEqual(classify_facts({**facts, "intended_symbol": fake},
                                            row["public_contract"]).symbol, "A")

    def test_exact_character_evidence(self):
        text = "Mina received it."
        self.assertTrue(validate_evidence(text, [{"start": 0, "end": 4, "quote": "Mina"}]))
        for evidence in ([], [{"start": -1, "end": 4, "quote": "Mina"}],
                         [{"start": 0, "end": 4, "quote": "Theo"}],
                         [{"start": True, "end": 4, "quote": "ina"}],
                         [{"start": 0, "end": 99, "quote": text}]):
            self.assertFalse(validate_evidence(text, evidence))

    def test_plan_and_explicit_nonoccurrence(self):
        rows = plan()
        self.assertEqual(len(rows), 96)
        self.assertEqual(len({r["item_id"] for r in rows}), 96)
        for i in range(0, 96, 4):
            self.assertEqual([r["intended_symbol"] for r in rows[i:i+4]], list("ABCD"))
        content = generation_messages(rows[2])[1]["content"]
        self.assertIn("explicitly not yet occurred", content)
        self.assertIn("BOTH", content)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="amr_fact_unit_")
        self.base = Path(self.temporary.name)
        self.out, self.private = self.base / "public", self.base / "private"
        self.out.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def freeze(self):
        (self.out / "PROTOCOL.md").write_text("Unit fixture; not a research run.")
        pilot.atomic(self.out / "PRICE_SOURCE.json", {"fixture_only": True})
        sources = ["scripts/run_semantic_fact_carrier_pilot.py",
                   "src/core/semantic_fact_carrier.py",
                   "scripts/test_semantic_fact_carrier_pilot.py"]
        pilot.atomic(self.out / "TEST_RESULT.json", {
            "status": "PASS", "tests_run": 12, "fixture_only": True,
            "code_sha256": {s: pilot.sha(pilot.ROOT / s) for s in sources}})
        return pilot.prepare(self.out, self.private)

    @staticmethod
    def transport(calls):
        def call(*args, **kwargs):
            calls.append(kwargs["json"])
            class Receipt:
                status_code = 200
                text = json.dumps({"model": "UNIT_FIXTURE",
                                   "choices": [{"finish_reason": "stop",
                                                "message": {"content": "Unit fixture only."}}],
                                   "usage": {"prompt_tokens": 20, "completion_tokens": 4}})
            return Receipt()
        return call

    def test_freeze_and_input_drift(self):
        self.freeze()
        self.assertTrue(all(pilot.verify(self.out, self.private).values()))
        with self.assertRaises(RuntimeError):
            pilot.prepare(self.out, self.private)
        pilot.atomic(self.private / "PLAN.json", [])
        with self.assertRaises(pilot.TerminalHalt):
            pilot.verify(self.out, self.private)

    def test_at_most_once_complete_receipt_replay(self):
        calls = []
        provider = pilot.Provider(self.out, self.private, "UNIT_FIXTURE", self.transport(calls))
        messages = [{"role": "user", "content": "Unit fixture only"}]
        first = provider.request("unit0", messages)
        second = provider.request("unit0", messages)
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)
        restarted = pilot.Provider(self.out, self.private, "UNIT_FIXTURE", self.transport(calls))
        self.assertEqual(restarted.request("unit0", messages), first)
        self.assertEqual(len(calls), 1)
        with self.assertRaises(pilot.TerminalHalt):
            restarted.request("unit0", [{"role": "user", "content": "changed"}])
        self.assertEqual(len(calls), 1)

    def test_missing_receipt_permanently_halts(self):
        pilot.atomic(self.private / "requests/unit0/INTENT.json",
                     {"body_sha256": "unit", "reserved_usd": 0.1})
        calls = []
        with self.assertRaises(pilot.TerminalHalt):
            pilot.Provider(self.out, self.private, "UNIT_FIXTURE", self.transport(calls))
        self.assertTrue((self.private / "HALT.json").exists())
        self.assertEqual(calls, [])

    def test_transport_uncertainty_never_retries(self):
        calls = []
        def lost(*args, **kwargs):
            calls.append(1)
            raise TimeoutError("unit simulated lost response")
        provider = pilot.Provider(self.out, self.private, "UNIT_FIXTURE", lost)
        with self.assertRaises(pilot.TerminalHalt):
            provider.request("unit0", [{"role": "user", "content": "unit"}])
        with self.assertRaises(pilot.TerminalHalt):
            pilot.Provider(self.out, self.private, "UNIT_FIXTURE", lost)
        self.assertEqual(len(calls), 1)

    def test_capacity_prevents_dispatch(self):
        calls = []
        provider = pilot.Provider(self.out, self.private, "UNIT_FIXTURE", self.transport(calls))
        with self.assertRaises(pilot.TerminalHalt):
            provider.request("unit0", [{"role": "user", "content": "x" * 5000}])
        self.assertEqual(calls, [])
        self.assertEqual(list((self.private / "requests").glob("*/INTENT.json")), [])

    def test_truncation_and_http_failure_are_not_success(self):
        receipt = {"kind": "http_receipt", "http_status": 200, "payload": {
            "model": "UNIT_FIXTURE", "choices": [{"finish_reason": "length",
                                               "message": {"content": "partial"}}]}}
        self.assertEqual(pilot.extract(receipt)["status"], "truncated")
        self.assertEqual(pilot.extract({**receipt, "http_status": 429})["status"],
                         "provider_failure")

    def test_review_blanks_no_target_and_full_denominator(self):
        self.freeze()
        for entry in pilot.load(self.private / "PLAN.json"):
            identity = entry["row"]["item_id"]
            fixture = {"status": "complete", "text": "UNIT FIXTURE ONLY.",
                       "model": "UNIT_FIXTURE", "usage": {}}
            pilot.atomic(self.private / "documents" / (identity + ".json"),
                         {"original": fixture, "paraphrase": fixture})
        pilot.summarize(self.out, self.private, "GENERATION_FINISHED")
        pilot.review(self.out, self.private)
        for reviewer in (1, 2):
            directory = self.out / "private_review" / f"reviewer{reviewer}"
            for task, count, labels in (("facts", 192, pilot.FACT_LABELS),
                                        ("pairs", 96, pilot.PAIR_LABELS)):
                with (directory / (task + ".csv")).open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), count)
                self.assertNotIn("intended_symbol", rows[0])
                self.assertNotIn("intended_facts", rows[0])
                self.assertTrue(all(r[k] == "" for r in rows for k in labels))
        result = pilot.collect(self.out, self.private)
        self.assertEqual(result["status"], "INCOMPLETE_OR_INVALID")
        path = self.out / "private_review/reviewer1/facts.csv"
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines[:-1]) + "\n")
        self.assertEqual(pilot.collect(self.out, self.private)["status"],
                         "INCOMPLETE_OR_INVALID")

    def test_human_evidence_and_time_are_validated(self):
        row = {"text": "", "readability": "unavailable", "explicit_status": "unknown",
               "quality": "unknown", "notes": "Unit fixture only",
               **dict.fromkeys(pilot.FIELDS, "unknown"),
               "evidence_json": json.dumps(dict.fromkeys(pilot.FIELDS, []))}
        self.assertEqual(pilot.validate_fact(row)["readability"], "unavailable")
        with self.assertRaises(ValueError):
            pilot.validate_fact({**row, "evidence_json": json.dumps(dict.fromkeys(pilot.FIELDS, 0))})
        self.assertFalse(pilot.valid_completed_at({"completed_at": ""}))
        self.assertFalse(pilot.valid_completed_at({"completed_at": "2026-10-26"}))
        self.assertFalse(pilot.valid_completed_at({"completed_at": "2099-01-01T00:00:00Z"}))
        self.assertTrue(pilot.valid_completed_at({"completed_at": "2020-01-01T00:00:00Z"}))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if len(sys.argv) == 3 and sys.argv[1] == "--evidence":
        sources = ["scripts/run_semantic_fact_carrier_pilot.py",
                   "src/core/semantic_fact_carrier.py",
                   "scripts/test_semantic_fact_carrier_pilot.py"]
        pilot.atomic(Path(sys.argv[2]), {
            "at": pilot.now(), "status": "PASS" if result.wasSuccessful() else "FAIL",
            "tests_run": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "api_calls": 0,
            "unit_fixtures_not_experiment_results": True,
            "code_sha256": {s: pilot.sha(pilot.ROOT / s) for s in sources}})
    raise SystemExit(0 if result.wasSuccessful() else 1)
