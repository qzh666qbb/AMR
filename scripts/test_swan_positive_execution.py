"""Network-free tests for the corrected SWAN positive execution wrapper."""

from __future__ import annotations

import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

import run_swan_positive_collection as runner


def bedrock_response(text: str, request_id: str = "request-1") -> dict:
    body = {
        "id": "message-1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "claude-sonnet-4-5",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }
    return {
        "body": io.BytesIO(json.dumps(body).encode("utf-8")),
        "contentType": "application/json",
        "performanceConfigLatency": "standard",
        "ResponseMetadata": {
            "RequestId": request_id,
            "HTTPStatusCode": 200,
        },
    }


def fake_module(scores):
    iterator = iter(scores)

    def decide(*args):
        threshold = args[2]
        score = next(iterator)
        return score >= threshold, score

    return runner.official_function_module_for_test(
        {
            "BedrockManager": object,
            "parse_amr": lambda text: f"(x / parsed-{len(text)})",
            "normalize_amr_variables": lambda graph: graph,
            "generate_template_amr": lambda graph: graph,
            "decide_s2match_match": decide,
        }
    )


class RequestContractTest(unittest.TestCase):
    def test_claude_45_declares_but_effectively_omits_top_p(self):
        body = runner.anthropic_request_body(
            "user",
            "system",
            50,
            0.7,
            0.9,
        )
        self.assertNotIn("top_p", body)
        self.assertEqual(body["temperature"], 0.7)
        self.assertEqual(body["system"], "system")

    def test_journal_replays_complete_call_without_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            transport = Mock(return_value=bedrock_response("First response."))
            journal = runner.BedrockCallJournal(
                folder,
                max_new_calls=1,
                approved_call_cap=1,
                approved_cost_usd=100.0,
                transport=transport,
            )
            first = journal.request("call-0", "user", "system", 50, 0.7, 0.9)
            self.assertFalse(first["replayed"])
            self.assertNotIn(
                "top_p",
                json.loads(
                    (folder / "calls/call-0.json").read_text()
                )["request"]["body"],
            )
            replay = runner.BedrockCallJournal(
                folder,
                max_new_calls=0,
                approved_call_cap=1,
                approved_cost_usd=100.0,
                transport=Mock(
                    side_effect=AssertionError("Unexpected transport")
                ),
            ).request("call-0", "user", "system", 50, 0.7, 0.9)
        self.assertTrue(replay["replayed"])
        self.assertEqual(replay["text"], first["text"])
        self.assertLessEqual(
            first["cost"]["actual"]["estimated_billed_cost_usd"],
            first["cost"]["reserve"]["reserved_cost_usd"],
        )
        self.assertEqual(transport.call_count, 1)

    def test_usd_budget_blocks_before_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            transport = Mock(
                side_effect=AssertionError("Unexpected transport")
            )
            journal = runner.BedrockCallJournal(
                folder,
                max_new_calls=1,
                approved_call_cap=1,
                approved_cost_usd=0.000001,
                transport=transport,
            )
            with self.assertRaisesRegex(
                runner.InvocationBudgetExhausted,
                "US-dollar",
            ):
                journal.request(
                    "call-0",
                    "user",
                    "system",
                    50,
                    0.7,
                    0.9,
                )
            self.assertFalse((folder / "calls").exists())
        self.assertEqual(transport.call_count, 0)

    def test_uncertain_call_is_frozen_and_never_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            failed = Mock(side_effect=TimeoutError("private transport detail"))
            journal = runner.BedrockCallJournal(
                folder,
                max_new_calls=1,
                approved_call_cap=1,
                approved_cost_usd=100.0,
                transport=failed,
            )
            with self.assertRaises(runner.UncertainCallExists):
                journal.request(
                    "call-0",
                    "user",
                    "system",
                    50,
                    0.7,
                    0.9,
                )
            saved = json.loads((folder / "calls/call-0.json").read_text())
            self.assertEqual(saved["status"], "failed_or_uncertain")
            self.assertNotIn("private transport detail", json.dumps(saved))
            retry_transport = Mock(
                side_effect=AssertionError("Unexpected retry")
            )
            replay = runner.BedrockCallJournal(
                folder,
                max_new_calls=1,
                approved_call_cap=1,
                approved_cost_usd=100.0,
                transport=retry_transport,
            )
            with self.assertRaises(runner.UncertainCallExists):
                replay.request(
                    "call-0",
                    "user",
                    "system",
                    50,
                    0.7,
                    0.9,
                )
        self.assertEqual(retry_transport.call_count, 0)


class OfficialInjectionTraceTest(unittest.TestCase):
    def run_slot(self, folder: Path, scores, texts):
        source_manifest = folder / "source_manifest.json"
        contract = folder / "execution_contract_v3.json"
        runner.write_json(source_manifest, {"fixture": "source"})
        runner.write_json(contract, {"fixture": "contract"})
        responses = iter(texts)

        def next_response(region, request):
            self.assertEqual(region, runner.REGION)
            self.assertEqual(request["model_id"], runner.MODEL)
            return bedrock_response(next(responses))

        transport = Mock(side_effect=next_response)
        journal = runner.BedrockCallJournal(
            folder,
            max_new_calls=len(texts),
            approved_call_cap=len(texts),
            approved_cost_usd=100.0,
            transport=transport,
        )
        examples = folder / "examples.json"
        runner.write_json(
            examples,
            [
                {
                    "normalized_template_amr": "(e / example)",
                    "sentence": "Example.",
                }
            ]
            * 3,
        )
        module = fake_module(scores)
        with (
            patch.object(runner, "SOURCE_MANIFEST", source_manifest),
            patch.object(runner, "CONTRACT", contract),
        ):
            result = runner.run_official_slot(
                module,
                folder,
                {"allocation_id": "pilot-000"},
                0,
                "Frozen source prompt.",
                ["(t / alpha)", "(t / beta)"],
                examples,
                journal,
            )
        return result, transport

    def test_every_attempt_is_traced_and_acceptance_is_reconstructed(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            result, transport = self.run_slot(
                folder,
                [0.2, 0.8],
                ["Rejected sentence.", "Accepted sentence."],
            )
            traces = sorted((folder / "generation_attempts").glob("*.json"))
            rows = [json.loads(path.read_text()) for path in traces]
        self.assertEqual(result["attempts"], 2)
        self.assertTrue(result["accepted"])
        self.assertFalse(result["fallback"])
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["accepted"] for row in rows], [False, True])
        self.assertTrue(all(row["raw_amr"] for row in rows))
        self.assertEqual(transport.call_count, 2)

    def test_exhausted_official_fallback_is_not_marked_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with (
                patch.object(runner, "MAX_TRIALS", 3),
                patch.object(runner, "MAX_TEMPLATES", 1),
                patch.object(runner, "RETRIES_PER_TEMPLATE", 3),
            ):
                result, transport = self.run_slot(
                    folder,
                    [0.1, 0.2, 0.3],
                    ["First.", "Second.", "Fallback."],
                )
        self.assertEqual(result["attempts"], 3)
        self.assertFalse(result["accepted"])
        self.assertTrue(result["fallback"])
        self.assertEqual(result["text"], "Fallback.")
        self.assertEqual(result["score"], 0.3)
        self.assertEqual(transport.call_count, 3)

    def test_template_sequence_is_deterministic(self):
        outputs = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as directory:
                result, transport = self.run_slot(
                    Path(directory),
                    [0.1, 0.8],
                    ["First.", "Second."],
                )
                outputs.append(result["templates_chosen"])
                self.assertEqual(transport.call_count, 2)
        self.assertEqual(outputs[0], outputs[1])

    def test_score_seed_blocks_internal_no_argument_reseeding(self):
        random.seed(91)
        outer_state = random.getstate()
        with runner.fixed_score_seed(7):
            random.seed()
            observed = random.random()
        expected = random.Random(7).random()
        self.assertEqual(observed, expected)
        self.assertEqual(random.getstate(), outer_state)


class AuthorizationTest(unittest.TestCase):
    def test_paid_execution_requires_explicit_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(PermissionError, "not authorized"):
                runner.require_authorization(
                    Path(directory),
                    "pilot_collection",
                )

    def test_probe_freezes_requested_profile_without_claiming_returned_id(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "source_manifest.json"
            contract = folder / "execution_contract_v3.json"
            runner.write_json(source, {"fixture": "source"})
            runner.write_json(
                contract,
                {
                    "status": "frozen_execution_contract",
                    "execution_ready": True,
                    "runner_sha256": runner.sha256(Path(runner.__file__)),
                    "source_manifest_sha256": runner.sha256(source),
                    "python_environment": {"prefix": runner.sys.prefix},
                },
            )
            authorization = {
                "status": "explicitly_authorized",
                "phase": "pilot_collection",
                "model_id": runner.MODEL,
                "region": runner.REGION,
                "source_manifest_sha256": runner.sha256(source),
                "execution_contract_sha256": runner.sha256(contract),
                "max_model_calls": 1,
                "max_cost_usd": 1.0,
            }
            runner.write_json(
                folder / runner.AUTHORIZATION.name,
                authorization,
            )
            contract_audit = folder / "execution_contract_audit_v3.json"
            runner.write_json(
                contract_audit,
                {
                    "status": (
                        "verified_local_execution_ready_remote_unauthorized"
                    ),
                    "all_checks_pass": True,
                    "execution_contract_sha256": runner.sha256(contract),
                },
            )
            profile_transport = Mock(
                return_value={
                    "inferenceProfileId": runner.MODEL,
                    "inferenceProfileArn": "arn:fixture",
                    "models": [{"modelArn": "arn:model-fixture"}],
                    "status": "ACTIVE",
                }
            )
            model_transport = Mock(
                return_value=bedrock_response("OK", "probe-request")
            )
            with (
                patch.object(runner, "SOURCE_MANIFEST", source),
                patch.object(runner, "CONTRACT", contract),
                patch.object(runner, "CONTRACT_AUDIT", contract_audit),
            ):
                probe = runner.run_model_probe(
                    folder,
                    1,
                    profile_transport,
                    model_transport,
                )
        self.assertEqual(probe["model_id_requested"], runner.MODEL)
        self.assertEqual(
            probe["profile_id_returned_by_control_plane"],
            runner.MODEL,
        )
        self.assertEqual(probe["aws_request_id"], "probe-request")
        self.assertIn(
            "no top-level returned modelId",
            probe["identity_interpretation"],
        )
        self.assertEqual(profile_transport.call_count, 1)
        self.assertEqual(model_transport.call_count, 1)

    def test_offline_check_reports_zero_calls_and_no_authorization(self):
        result = runner.check_contract()
        self.assertFalse(result["authorization_present"])
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(result["generation_slots"], 0)


if __name__ == "__main__":
    unittest.main()
