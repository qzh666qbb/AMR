"""Network-free tests for the resilient length-controlled confirmation."""

from __future__ import annotations

from contextlib import nullcontext
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import run_swan_positive_deepseek_length_confirmation3 as runner


def provider_response(text: str = "A complete generated sentence.") -> dict:
    return {
        "id": "fixture-response",
        "model": "deepseek-flash",
        "usage": {
            "prompt_tokens": 40,
            "completion_tokens": 6,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
        "choices": [
            {
                "message": {"content": text},
                "finish_reason": "stop",
            }
        ],
    }


class RequestAndJournalTests(unittest.TestCase):
    def authorization(self, calls: int = 2, cost: float = 1.0) -> dict:
        return {
            "max_model_calls": calls,
            "max_cost_usd": cost,
        }

    def test_request_contract_is_frozen_non_thinking_model(self):
        body = runner.request_body("user", "system", 50, 0.7, 0.9)
        self.assertEqual(body["model"], "deepseek-flash")
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(body["temperature"], 0.7)
        self.assertEqual(body["top_p"], 0.9)

    def test_complete_call_replays_without_secret_or_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            transport = Mock(return_value=provider_response())
            journal = runner.DeepSeekJournal(
                folder,
                "fixture-secret",
                "pilot",
                self.authorization(),
                max_new_calls=1,
                transport=transport,
            )
            first = journal.request(
                "call-0", "user", "system", 50, 0.7, 0.9
            )
            replay = runner.DeepSeekJournal(
                folder,
                "fixture-secret",
                "pilot",
                self.authorization(),
                max_new_calls=0,
                transport=Mock(
                    side_effect=AssertionError("unexpected transport")
                ),
            ).request("call-0", "user", "system", 50, 0.7, 0.9)
            saved_text = (folder / "deepseek_calls/call-0.json").read_text()
        self.assertEqual(transport.call_count, 1)
        self.assertFalse(first["replayed"])
        self.assertTrue(replay["replayed"])
        self.assertNotIn("fixture-secret", saved_text)
        self.assertNotIn("sk-", saved_text)

    def test_per_command_cap_blocks_before_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = Mock(
                side_effect=AssertionError("unexpected transport")
            )
            journal = runner.DeepSeekJournal(
                Path(directory),
                "fixture-secret",
                "pilot",
                self.authorization(),
                max_new_calls=0,
                transport=transport,
            )
            with self.assertRaises(runner.InvocationBudgetExhausted):
                journal.request(
                    "call-0", "user", "system", 50, 0.7, 0.9
                )
        self.assertEqual(transport.call_count, 0)

    def test_usd_cap_blocks_before_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = Mock(
                side_effect=AssertionError("unexpected transport")
            )
            journal = runner.DeepSeekJournal(
                Path(directory),
                "fixture-secret",
                "pilot",
                self.authorization(cost=0.000001),
                max_new_calls=1,
                transport=transport,
            )
            with self.assertRaisesRegex(
                runner.InvocationBudgetExhausted,
                "USD cap",
            ):
                journal.request(
                    "call-0", "user", "system", 50, 0.7, 0.9
                )
        self.assertEqual(transport.call_count, 0)

    def test_uncertain_call_is_redacted_and_never_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            failed = Mock(side_effect=TimeoutError("private network detail"))
            journal = runner.DeepSeekJournal(
                folder,
                "fixture-secret",
                "pilot",
                self.authorization(),
                max_new_calls=1,
                transport=failed,
            )
            with self.assertRaises(runner.UncertainCallExists):
                journal.request(
                    "call-0", "user", "system", 50, 0.7, 0.9
                )
            saved_text = (folder / "deepseek_calls/call-0.json").read_text()
            retry = Mock(side_effect=AssertionError("unexpected retry"))
            with self.assertRaises(runner.UncertainCallExists):
                runner.DeepSeekJournal(
                    folder,
                    "fixture-secret",
                    "pilot",
                    self.authorization(),
                    max_new_calls=1,
                    transport=retry,
                ).request("call-0", "user", "system", 50, 0.7, 0.9)
        self.assertNotIn("fixture-secret", saved_text)
        self.assertNotIn("private network detail", saved_text)
        self.assertEqual(retry.call_count, 0)

    def test_uncertain_call_reserve_is_charged_and_new_id_can_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            transport = Mock(
                side_effect=[
                    TimeoutError("private detail"),
                    provider_response(),
                ]
            )
            journal = runner.DeepSeekJournal(
                folder,
                "fixture-secret",
                "pilot",
                self.authorization(calls=2, cost=1.0),
                max_new_calls=2,
                transport=transport,
            )
            with self.assertRaises(runner.UncertainCallExists):
                journal.request(
                    "call-0", "user", "system", 50, 0.7, 0.9
                )
            first = runner.load_json(
                folder / "deepseek_calls/call-0.json"
            )
            response = journal.request(
                "call-1", "user", "system", 50, 0.7, 0.9
            )
            expected = (
                first["cost"]["reserve"][
                    "peak_all_cache_miss_cost_usd"
                ]
                + runner.actual_cost(response["usage"])[
                    "conservative_peak_cost_usd"
                ]
            )
            accounted = journal.authorization_accounting_cost()
        self.assertEqual(transport.call_count, 2)
        self.assertAlmostEqual(accounted, expected)


class LengthGateTests(unittest.TestCase):
    def run_attempt(self, candidate: str, matcher: Mock):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            examples = folder / "examples.json"
            examples.write_text(
                json.dumps(
                    [
                        {
                            "normalized_template_amr": "example",
                            "sentence": "An example sentence.",
                        }
                    ]
                )
            )
            module = runner.pure_prompt_module()
            module.decide_s2match_match = matcher
            journal = Mock()
            journal.request.return_value = {
                "call_sha256": "fixture-call-sha256",
                "text": candidate,
                "text_sha256": runner.text_sha256(candidate),
                "stop_reason": "stop",
                "usage": {
                    "prompt_tokens": 40,
                    "completion_tokens": 8,
                },
                "response_metadata": {"provider": "fixture"},
            }
            with runner.instrument_length_controlled_injection(
                module,
                folder,
                "fixture-pilot-000",
                0,
                ["template"],
                journal,
            ) as trace:
                result = module.amr_instruction_completion(
                    context="A source sentence.",
                    amr_bank=["template"],
                    model_id=runner.swan.MODEL,
                    region=runner.swan.REGION,
                    max_new_tokens=runner.swan.MAX_NEW_TOKENS,
                    temperature=runner.swan.TEMPERATURE,
                    top_p=runner.swan.DECLARED_TOP_P,
                    max_trials=1,
                    amr_examples_path=str(examples),
                    s2match_threshold=runner.MATCH_THRESHOLD,
                    max_templates_to_try=1,
                    max_retries_per_template=1,
                )
                event = dict(trace["attempts"][0])
                request = journal.request.call_args.args
        return result, event, request

    def test_prompt_appends_frozen_length_instruction_once(self):
        prompt = runner.length_controlled_user_prompt("Official prompt.")
        self.assertTrue(prompt.startswith("Official prompt."))
        self.assertEqual(prompt.count(runner.LENGTH_INSTRUCTION), 1)
        with self.assertRaisesRegex(ValueError, "more than once"):
            runner.length_controlled_user_prompt(prompt)

    def test_short_candidate_skips_s2match_even_if_matcher_would_pass(self):
        matcher = Mock(return_value=(True, 1.0))
        _result, event, request = self.run_attempt(
            "Alice quickly built a sturdy device.",
            matcher,
        )
        self.assertFalse(event["candidate_gate"]["candidate_gate_pass"])
        self.assertEqual(event["candidate_gate"]["word_count"], 6)
        self.assertFalse(event["s2match_evaluated"])
        self.assertFalse(event["accepted"])
        self.assertEqual(event["score"], 0.0)
        self.assertEqual(matcher.call_count, 0)
        self.assertIn(runner.LENGTH_INSTRUCTION, request[1])

    def test_valid_candidate_reaches_s2match_and_can_be_accepted(self):
        matcher = Mock(return_value=(True, 0.8))
        result, event, _request = self.run_attempt(
            "Alice quickly built a sturdy device yesterday.",
            matcher,
        )
        self.assertTrue(event["candidate_gate"]["candidate_gate_pass"])
        self.assertEqual(event["candidate_gate"]["word_count"], 7)
        self.assertTrue(event["s2match_evaluated"])
        self.assertTrue(event["accepted"])
        self.assertEqual(event["score"], 0.8)
        self.assertEqual(matcher.call_count, 1)
        self.assertEqual(
            result[0],
            "Alice quickly built a sturdy device yesterday.",
        )

    def test_gate_rejects_multiple_sentences_and_missing_terminal_mark(self):
        multiple = runner.candidate_gate(
            "Alice built the device. Bob tested it carefully."
        )
        unterminated = runner.candidate_gate(
            "Alice quickly built a sturdy device yesterday"
        )
        self.assertFalse(multiple["single_sentence_pass"])
        self.assertFalse(multiple["candidate_gate_pass"])
        self.assertFalse(unterminated["terminal_punctuation_pass"])
        self.assertFalse(unterminated["candidate_gate_pass"])


class SourceAllocationTests(unittest.TestCase):
    class Response:
        def __init__(self, rows):
            payload = b"".join(
                json.dumps(row).encode("utf-8") + b"\n"
                for row in rows
            )
            self.raw = self.Raw(gzip.compress(payload))

        class Raw(io.BytesIO):
            decode_content = False

        def raise_for_status(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    @staticmethod
    def empty_snapshot() -> dict:
        return {
            "schema_version": 1,
            "created_at": "fixture",
            "status": "frozen_before_new_source_allocation",
            "selection": {},
            "scan_counts": {},
            "artifact_inventory": [],
            "artifact_inventory_digest": runner.digest([]),
            "field_value_counts": {},
            "normalized_text_sha256": [],
            "url_sha256": [],
            "counts": {
                "artifacts": 0,
                "normalized_text_identities": 0,
                "url_identities": 0,
            },
        }

    def test_mocked_c4_allocation_freezes_five_plus_thirty(self):
        rows = [
            {
                "text": (
                    f"Document {index} contains a complete opening sentence. "
                    "It also contains additional source context."
                ),
                "url": f"https://example.test/{index}",
            }
            for index in range(35)
        ]
        empty_snapshot = self.empty_snapshot()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            protocol = folder / "protocol.md"
            protocol.write_text("fixture protocol")
            with (
                patch.object(runner, "OUTPUT", folder),
                patch.object(
                    runner,
                    "SOURCE_SNAPSHOT",
                    folder / "prior_identity_snapshot.json",
                ),
                patch.object(
                    runner,
                    "SOURCE_MANIFEST",
                    folder / "source_manifest.json",
                ),
                patch.object(runner, "PROTOCOL", protocol),
                patch.object(
                    runner,
                    "build_history_snapshot",
                    return_value=empty_snapshot,
                ),
            ):
                manifest = runner.allocate_sources(
                    lambda _url: self.Response(rows)
                )
        self.assertEqual(manifest["cohort_counts"]["pilot"], 5)
        self.assertEqual(manifest["cohort_counts"]["main"], 30)
        self.assertEqual(len(manifest["sources"]), 35)
        self.assertEqual(manifest["dataset"]["shard"], runner.shard_name(12))
        self.assertEqual(
            {row["shard_index"] for row in manifest["sources"]},
            {12},
        )
        self.assertTrue(
            all(
                row["allocation_id"].startswith("dslength3-")
                for row in manifest["sources"]
            )
        )
        self.assertEqual(
            [row["cohort"] for row in manifest["sources"][:5]],
            ["pilot"] * 5,
        )
        self.assertEqual(
            [row["cohort"] for row in manifest["sources"][5:]],
            ["main"] * 30,
        )
        self.assertEqual(manifest["resource_counts"]["model_calls"], 0)

    def test_historical_identity_is_excluded_before_allocation(self):
        rows = [
            {
                "text": (
                    f"Document {index} contains a complete opening sentence. "
                    "It also contains additional source context."
                ),
                "url": f"https://example.test/{index}",
            }
            for index in range(36)
        ]
        snapshot = self.empty_snapshot()
        snapshot["normalized_text_sha256"] = [
            runner.normalized_hash(rows[0]["text"])
        ]
        snapshot["url_sha256"] = [
            runner.normalized_url_hash(rows[0]["url"])
        ]
        snapshot["counts"]["normalized_text_identities"] = 1
        snapshot["counts"]["url_identities"] = 1
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            protocol = folder / "protocol.md"
            protocol.write_text("fixture protocol")
            with (
                patch.object(runner, "OUTPUT", folder),
                patch.object(
                    runner,
                    "SOURCE_SNAPSHOT",
                    folder / "prior_identity_snapshot.json",
                ),
                patch.object(
                    runner,
                    "SOURCE_MANIFEST",
                    folder / "source_manifest.json",
                ),
                patch.object(runner, "PROTOCOL", protocol),
                patch.object(
                    runner,
                    "build_history_snapshot",
                    return_value=snapshot,
                ),
            ):
                manifest = runner.allocate_sources(
                    lambda _url: self.Response(rows)
                )
        self.assertEqual(manifest["screened_rows"][0]["decision"], "excluded")
        self.assertEqual(manifest["sources"][0]["shard_row"], 1)
        self.assertEqual(len(manifest["sources"]), 35)


class PauseAndQualityTests(unittest.TestCase):
    def test_pause_before_full_slot_writes_no_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            authorization = {
                "max_model_calls": runner.PILOT_CALL_CAP,
                "max_cost_usd": 1000.0,
            }
            with (
                patch.object(runner, "OUTPUT", folder),
                patch.object(runner, "check"),
                patch.object(
                    runner,
                    "require_authorization",
                    return_value=authorization,
                ),
                patch.object(
                    runner,
                    "read_credential",
                    return_value="fixture-secret",
                ),
                patch.object(runner, "probe", return_value={}),
                patch.object(
                    runner,
                    "status",
                    return_value={"status": "paused"},
                ),
            ):
                result = runner.run_phase(
                    "pilot",
                    folder / "unused-key",
                    runner.MAX_TRIALS - 1,
                    transport=Mock(),
                )
            self.assertEqual(result, {"status": "paused"})
            self.assertFalse((folder / "generation_failures").exists())

    def test_uncertain_call_fails_current_document_and_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source_manifest = folder / "source_manifest.json"
            manifest = folder / "manifest.json"
            source_manifest.write_text("{}")
            manifest.write_text("{}")
            sources = [
                {
                    "allocation_id": f"dslength3-pilot-{index:03d}",
                    "cohort": "pilot",
                    "generation_prompt": "Source context.",
                }
                for index in range(2)
            ]

            def fake_slot(
                _module,
                output,
                source,
                slot,
                _context,
                _bank,
                _examples,
                _journal,
            ):
                allocation_id = source["allocation_id"]
                if allocation_id.endswith("000"):
                    call_id = f"{allocation_id}-slot-00-attempt-00"
                    runner.write_json(
                        output / "deepseek_calls" / f"{call_id}.json",
                        {
                            "call_id": call_id,
                            "phase": "pilot",
                            "status": "failed_or_uncertain",
                            "error_type": "TimeoutError",
                            "cost": {
                                "reserve": {
                                    "peak_all_cache_miss_cost_usd": 0.01
                                },
                                "actual": None,
                            },
                        },
                    )
                    runner.write_json(
                        output
                        / "generation_failures"
                        / f"{allocation_id}-slot-00.json",
                        {
                            "status": "failed_or_uncertain",
                            "attempts_recorded": 1,
                        },
                    )
                    raise runner.UncertainCallExists("fixture uncertainty")
                result = {
                    "allocation_id": allocation_id,
                    "slot": slot,
                    "status": "exhausted",
                    "attempts": 1,
                    "text": "Fallback.",
                    "chosen_template": "template",
                    "accepted": False,
                    "fallback": True,
                }
                runner.write_json(
                    output
                    / "generation_slots"
                    / f"{allocation_id}-slot-00.json",
                    result,
                )
                return result

            with (
                patch.object(runner, "OUTPUT", folder),
                patch.object(runner, "SOURCE_MANIFEST", source_manifest),
                patch.object(runner, "MANIFEST", manifest),
                patch.object(runner, "check"),
                patch.object(
                    runner,
                    "require_authorization",
                    return_value={
                        "max_model_calls": runner.PILOT_CALL_CAP,
                        "max_cost_usd": 1000.0,
                    },
                ),
                patch.object(
                    runner,
                    "read_credential",
                    return_value="fixture-secret",
                ),
                patch.object(runner, "probe", return_value={}),
                patch.object(runner, "phase_sources", return_value=sources),
                patch.object(
                    runner,
                    "patched_swan_runtime",
                    return_value=nullcontext(object()),
                ),
                patch.object(
                    runner.swan,
                    "run_official_slot",
                    side_effect=fake_slot,
                ) as run_slot,
                patch.object(
                    runner,
                    "summarize",
                    return_value={"status": "complete"},
                ),
            ):
                result = runner.run_phase(
                    "pilot",
                    folder / "unused-key",
                    1000,
                )
            first = runner.load_json(
                folder
                / "generated_documents"
                / "dslength3-pilot-000.json"
            )
            second = runner.load_json(
                folder
                / "generated_documents"
                / "dslength3-pilot-001.json"
            )
        self.assertEqual(result, {"status": "complete"})
        self.assertEqual(run_slot.call_count, 2)
        self.assertEqual(first["status"], "operational_failure")
        self.assertEqual(
            first["operational_failure"]["uncertain_call_status"],
            "failed_or_uncertain",
        )
        self.assertEqual(second["status"], "injection_failed")

    def test_review_requires_complete_typed_ratings(self):
        packet = {
            "phase": "pilot",
            "documents": [{"blind_id": "blind-1"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            packet_path = folder / "blind_packet.json"
            review_path = folder / "review.json"
            packet_path.write_text(json.dumps(packet))
            review_path.write_text(
                json.dumps(
                    {
                        "reviewer_id": "reviewer-a",
                        "independent_review": True,
                        "packet_sha256": runner.sha256(packet_path),
                        "reviews": [
                            {
                                "blind_id": "blind-1",
                                "fluency": 4,
                                "coherence": 4,
                                "nonredundancy": 4,
                                "source_faithfulness": 4,
                                "major_error": False,
                                "amr_or_template_leakage": False,
                                "overall_pass": True,
                            }
                        ],
                    }
                )
            )
            with patch.object(
                runner,
                "quality_paths",
                return_value={"packet": packet_path},
            ):
                review = runner.validate_review(review_path, packet)
        self.assertTrue(runner.review_row_pass(review["reviews"][0]))


if __name__ == "__main__":
    unittest.main()
