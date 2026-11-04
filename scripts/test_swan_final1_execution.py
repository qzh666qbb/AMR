"""Synthetic runtime tests; never load a model, call a scorer or send a request."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import run_swan_positive_collection as swan
from swan_final1_budget import (
    DocumentFailure, GlobalStop, Journal, MAX_RESERVE, MODEL, Pause, URL,
    body_for, budget_estimate, reserve_for,
)
from swan_final1_common import digest, freeze, load, seed, sha256, single_writer
from swan_final1_generation import load_slots, run_documents, verify_generation
from swan_final1_quality import verify_decision
from swan_final1_runtime import final1_instrument
from swan_final1_workflow import verify_inputs
import test_swan_final1_quality_sources as quality_fixtures


def response(text="Alice reports the news.", model=MODEL, inputs=100, outputs=10):
    return {"model": model, "id": "mock-response", "system_fingerprint": "fixture",
            "usage": {"prompt_tokens": inputs, "completion_tokens": outputs},
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}]}


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def journal(self, send, limit=100):
        return Journal(self.root, "pilot", limit, send, "contract")

    def request(self, journal, call_id="call0"):
        return journal.request(call_id, "user", "system", 50, .7, .9)

    def test_fsync_reservation_exists_before_send(self):
        def send(body):
            events = (self.root / "private/pilot/journal.jsonl").read_text().splitlines()
            self.assertEqual(json.loads(events[-1])["kind"], "reserve")
            self.assertEqual(json.loads(events[-1])["request"], body)
            return response()
        journal = self.journal(send)
        self.request(journal)
        self.assertEqual(journal.totals()["actual_peak_nanodollars"], 42000)
        restored = self.journal(Mock(side_effect=AssertionError("network")))
        self.assertEqual(restored.totals()["actual_peak_nanodollars"], 42000)

    def test_uncertainty_retains_reserve_and_no_retry(self):
        send = Mock(side_effect=TimeoutError("secret not to be logged"))
        journal = self.journal(send)
        with self.assertRaises(DocumentFailure):
            self.request(journal)
        expected = reserve_for(body_for("user", "system"))["nanodollars"]
        restored = self.journal(send)
        self.assertEqual(restored.totals()["exposure_nanodollars"], expected)
        with self.assertRaises(GlobalStop):
            self.request(restored)
        self.assertEqual(send.call_count, 1)
        self.assertNotIn("secret", (self.root / "private/pilot/journal.jsonl").read_text())

    def test_crash_after_reserve_never_resends(self):
        journal = self.journal(Mock())
        body = body_for("user", "system")
        journal.event("reserve", call_id="call0", request=body, reserve=reserve_for(body))
        recovered = self.journal(Mock(side_effect=AssertionError("network")))
        with self.assertRaises(GlobalStop):
            self.request(recovered)
        self.assertEqual(recovered.totals()["uncertain_or_invalid_calls"], 1)

    def test_partial_journal_fails_closed(self):
        self.request(self.journal(lambda _: response()))
        with (self.root / "private/pilot/journal.jsonl").open("ab") as f:
            f.write(b'{"partial"')
        with self.assertRaises(GlobalStop):
            self.journal(Mock())

    def test_model_drift_permanently_halts(self):
        with self.assertRaises(GlobalStop):
            self.request(self.journal(lambda _: response(model="different")))
        with self.assertRaises(GlobalStop):
            self.journal(Mock())

    def test_usage_over_reserve_halts(self):
        with self.assertRaises(GlobalStop):
            self.request(self.journal(lambda _: response(inputs=100000)))

    def test_invocation_budget_before_transport(self):
        send = Mock(return_value=response())
        journal = self.journal(send, 1)
        self.request(journal)
        with self.assertRaises(Pause):
            self.request(journal, "call1")
        self.assertEqual(send.call_count, 1)

    def test_oversize_request_not_reserved_or_sent(self):
        send = Mock()
        journal = self.journal(send)
        with self.assertRaises(DocumentFailure):
            journal.request("call0", "\U0001f600" * 1000, "system", 50, .7, .9)
        self.assertEqual(journal.totals()["calls"], 0)
        send.assert_not_called()

    def test_phase_cost_limit_and_fixed_maximum(self):
        self.assertEqual(MAX_RESERVE, 2824800)
        self.assertEqual(budget_estimate()["phases"]["pilot"]["max_cost_usd"], 27.13)
        journal = self.journal(Mock())
        with patch("swan_final1_budget.CAPS", {"pilot": (9601, 1)}):
            self.assertEqual(journal.capacity(), "phase_money_cap")
            with self.assertRaises(Pause):
                self.request(journal)

    def test_single_writer_excludes_second_process_lock(self):
        with single_writer(self.root):
            with self.assertRaises(RuntimeError):
                with single_writer(self.root):
                    self.fail("Second lock entered")


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        freeze(self.root / "manifest.json", {"fixture": True})
        freeze(self.root / "source_manifest.json", {"fixture": True})
        freeze(self.root / "prior_identity_snapshot.json", {"normalized_text_sha256": []})
        self.examples = self.root / "examples.json"
        freeze(self.examples, [{"normalized_template_amr": "(e / N)", "sentence": "Example."}] * 3)
        assets = patch("swan_final1_generation.injection_assets",
                       return_value=(["(t / N)"], load(self.examples)))
        assets.start()
        self.addCleanup(assets.stop)
        self.source = {"allocation_id": "final1-pilot-0000", "ordinal": 0,
                       "locator": "synthetic:0", "domain": "fixture.invalid",
                       "generation_prompt": "Alice prepares a report."}
        # 13 words, 8 slots to cross 100; generated document qualifies without requiring 16.
        self.text = "Alice carefully reviews the detailed report before sharing all her findings with colleagues."
        self.contract = sha256(self.root / "manifest.json")

    def tearDown(self):
        self.temp.cleanup()

    def module(self, score=.8):
        return swan.official_function_module_for_test({
            "BedrockManager": object, "parse_amr": lambda _: "(a / N)",
            "normalize_amr_variables": lambda g: g, "generate_template_amr": lambda g: g,
            "decide_s2match_match": lambda a, b, threshold: (score >= threshold, score),
        })

    def run_fixture(self, journal, sources=None, score=.8):
        with patch.multiple(swan, SOURCE_MANIFEST=self.root / "source_manifest.json",
                            CONTRACT=self.root / "manifest.json", MODEL=MODEL, REGION=URL,
                            freeze_json=freeze, slot_seed=lambda a, s: seed("injection-template", a, s),
                            score_seed=lambda a, s, t: seed("injection-score", a, s, t),
                            instrument_official_injection=final1_instrument):
            return run_documents(self.root, "pilot", sources or [self.source], journal,
                                 self.module(score), swan.run_official_slot,
                                 ["(t / N)"], self.examples)

    def journal(self, send, cap=9601):
        return Journal(self.root, "pilot", cap, send, self.contract)

    def test_fallback_survives_50_attempts_and_stops_at_first_100(self):
        send = Mock(return_value=response(self.text))
        result = self.run_fixture(self.journal(send), score=.2)
        self.assertEqual(result["state"], "phase_terminal")
        slots, failure = load_slots(self.root, "pilot", self.source["allocation_id"])
        self.assertIsNone(failure)
        self.assertTrue(all(s["fallback"] and not s["accepted"] for s in slots))
        self.assertEqual(len(slots), 8)
        self.assertEqual(send.call_count, 400)
        documents = verify_generation(self.root, "pilot", [self.source])
        self.assertTrue(documents[0]["structural_pass"])

    def test_pause_before_start_not_document_failure(self):
        send = Mock(return_value=response(self.text))
        result = self.run_fixture(self.journal(send, cap=50))
        self.assertEqual(result["state"], "paused_before_slot")
        self.assertEqual(send.call_count, 1)
        result = self.run_fixture(self.journal(send))
        self.assertEqual(result["state"], "phase_terminal")
        self.assertEqual(send.call_count, 8)

    def test_first_100_even_when_fewer_than_8_units(self):
        text = "Alice " + " ".join(["carefully"] * 100) + " reports."
        send = Mock(return_value=response(text))
        result = self.run_fixture(self.journal(send))
        self.assertEqual(send.call_count, 1)
        self.assertEqual(result["structural_pass"], 0)
        row = verify_generation(self.root, "pilot", [self.source])[0]
        self.assertEqual(row["termination"], "first_100_words")

    def test_interrupted_slot_abandons_doc_then_next_source(self):
        private = self.root / "private/pilot"
        freeze(private / "slot_starts/final1-pilot-0000-slot-00.json",
               {"allocation_id": self.source["allocation_id"], "slot": 0,
                "context_sha256": digest(self.source["generation_prompt"]),
                "manifest_sha256": self.contract})
        second = {**self.source, "allocation_id": "final1-pilot-0001", "ordinal": 1}
        send = Mock(return_value=response(self.text))
        result = self.run_fixture(self.journal(send), [self.source, second])
        self.assertEqual(result["terminal"], 2)
        self.assertEqual(send.call_count, 8)
        first = load(self.root / "documents/pilot/final1-pilot-0000.json")
        self.assertEqual(first["termination"], "interrupted_slot_no_retry")
        self.assertFalse(first["generation_complete"])

    def test_timeout_is_terminal_but_following_allocation_runs(self):
        second = {**self.source, "allocation_id": "final1-pilot-0001", "ordinal": 1}
        send = Mock(side_effect=[TimeoutError()] + [response(self.text)] * 8)
        result = self.run_fixture(self.journal(send), [self.source, second])
        self.assertEqual(result["terminal"], 2)
        self.assertEqual(result["uncertain_or_invalid_calls"], 1)
        self.assertEqual(send.call_count, 9)
        verify_generation(self.root, "pilot", [self.source, second])

    def test_tampered_structural_gate_rejected(self):
        self.run_fixture(self.journal(lambda _: response("Alice " + "carefully " * 100 + "reports.")))
        path = self.root / "documents/pilot/final1-pilot-0000.json"
        row = load(path)
        row["structural_pass"] = True
        path.write_text(json.dumps(row))
        with self.assertRaises(GlobalStop):
            verify_generation(self.root, "pilot", [self.source])


class QualityReplayTests(unittest.TestCase):
    setUp = quality_fixtures.QualityTests.setUp
    tearDown = quality_fixtures.QualityTests.tearDown
    write_reviews = quality_fixtures.QualityTests.write_reviews

    def test_hand_edited_decision_cannot_unlock(self):
        self.write_reviews(fail_b=True)
        from swan_final1_quality import finalize
        finalize(self.folder, "contract", self.root, quality_fixtures.FROZEN)
        path = self.folder / "decision.json"
        result = load(path)
        result["quality_pass_count"] = 10
        path.write_text(json.dumps(result))
        with self.assertRaises(ValueError):
            verify_decision(self.folder, "contract")

    def test_packet_must_match_actual_transform(self):
        verify_inputs(self.folder, self.records)
        changed = [{**r, "transformed_text": r["transformed_text"] + " Extra."} for r in self.records]
        with self.assertRaises(ValueError):
            verify_inputs(self.folder, changed)


if __name__ == "__main__":
    unittest.main()
