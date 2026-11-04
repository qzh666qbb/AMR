"""Network-free tests for the independent DeepSeek confirmation runner."""

from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import run_swan_positive_deepseek_confirmation as runner


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
        empty_snapshot = {
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
        self.assertEqual(
            [row["cohort"] for row in manifest["sources"][:5]],
            ["pilot"] * 5,
        )
        self.assertEqual(
            [row["cohort"] for row in manifest["sources"][5:]],
            ["main"] * 30,
        )
        self.assertEqual(manifest["resource_counts"]["model_calls"], 0)


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
