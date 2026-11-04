"""Network-free tests for the bounded DeepSeek SWAN preflight."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import run_swan_positive_deepseek_preflight as deepseek


def provider_response(text="A generated sentence."):
    return {
        "id": "fixture-response",
        "model": "deepseek-flash",
        "usage": {
            "prompt_tokens": 40,
            "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
        "choices": [
            {
                "message": {"content": text},
                "finish_reason": "stop",
            }
        ],
    }


class DeepSeekContractTest(unittest.TestCase):
    def test_request_contract_uses_current_nonthinking_model(self):
        body = deepseek.request_body(
            "user",
            "system",
            50,
            0.7,
            0.9,
        )
        self.assertEqual(body["model"], "deepseek-flash")
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(body["max_tokens"], 50)
        self.assertEqual(body["top_p"], 0.9)

    def test_call_is_journaled_without_credential_and_replayed(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            transport = Mock(return_value=provider_response())
            authorization = {
                "max_model_calls": 2,
                "max_cost_usd": 1.0,
            }
            journal = deepseek.DeepSeekJournal(
                folder,
                "fixture-secret",
                authorization,
                transport=transport,
            )
            first = journal.request(
                "call-0",
                "user",
                "system",
                50,
                0.7,
                0.9,
            )
            replay = journal.request(
                "call-0",
                "user",
                "system",
                50,
                0.7,
                0.9,
            )
            saved_text = next(
                (folder / "deepseek_calls").glob("*.json")
            ).read_text()
        self.assertEqual(transport.call_count, 1)
        self.assertFalse(first["replayed"])
        self.assertTrue(replay["replayed"])
        self.assertNotIn("fixture-secret", saved_text)
        saved = json.loads(saved_text)
        self.assertLessEqual(
            saved["cost"]["actual"]["conservative_peak_cost_usd"],
            saved["cost"]["reserve"]["peak_all_cache_miss_cost_usd"],
        )

    def test_usd_cap_blocks_before_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = Mock(
                side_effect=AssertionError("Unexpected transport")
            )
            journal = deepseek.DeepSeekJournal(
                Path(directory),
                "fixture-secret",
                {"max_model_calls": 1, "max_cost_usd": 0.000001},
                transport=transport,
            )
            with self.assertRaisesRegex(RuntimeError, "USD cap"):
                journal.request(
                    "call-0",
                    "user",
                    "system",
                    50,
                    0.7,
                    0.9,
                )
        self.assertEqual(transport.call_count, 0)

    def test_prepare_freezes_zero_remote_calls_and_inactive_template(self):
        state = deepseek.prepare()
        manifest = state["manifest"]
        budget = state["budget"]
        template = deepseek.load_json(deepseek.AUTH_TEMPLATE)
        self.assertEqual(manifest["status"], "frozen_remote_unauthorized")
        self.assertEqual(manifest["resource_counts"]["deepseek_calls"], 0)
        self.assertEqual(budget["resource_counts"]["deepseek_calls"], 0)
        self.assertEqual(
            template["status"],
            "awaiting_explicit_user_approval",
        )
        self.assertIsNone(template["credential_rotated"])
        self.assertFalse(deepseek.AUTHORIZATION.exists())

    def test_frozen_artifacts_contain_no_secret_reference(self):
        state = deepseek.prepare()
        text = json.dumps(state)
        self.assertNotIn("api.txt", text)
        self.assertNotIn("DEEPSEEK_API_KEY", text)
        self.assertNotIn("sk-", text)


if __name__ == "__main__":
    unittest.main()
