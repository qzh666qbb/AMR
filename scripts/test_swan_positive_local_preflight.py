"""Network-free tests for the local SWAN backend preflight."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import run_swan_positive_local_preflight as local


class FakeGenerator:
    def generate(
        self,
        system_text,
        user_text,
        max_tokens,
        temperature,
        top_p,
        seed,
    ):
        self.observed = {
            "system_text": system_text,
            "user_text": user_text,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
        }
        return {
            "text": "A deterministic local sentence.",
            "text_sha256": local.text_sha256(
                "A deterministic local sentence."
            ),
            "stop_reason": "eos",
            "usage": {"input_tokens": 20, "output_tokens": 6},
            "response_metadata": {"backend": "fixture"},
            "rendered_prompt_sha256": "fixture",
        }


class LocalJournalTest(unittest.TestCase):
    def test_local_call_is_checkpointed_and_replayed(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            generator = FakeGenerator()
            journal = local.LocalJournal(folder, generator)
            first = journal.request(
                "pilot-000-slot-00-attempt-00",
                "user",
                "system",
                50,
                0.7,
                0.9,
            )
            replay = journal.request(
                "pilot-000-slot-00-attempt-00",
                "user",
                "system",
                50,
                0.7,
                0.9,
            )
            saved = json.loads(
                next((folder / "local_calls").glob("*.json")).read_text()
            )
        self.assertFalse(first["replayed"])
        self.assertTrue(replay["replayed"])
        self.assertEqual(saved["status"], "complete")
        self.assertEqual(first["text"], replay["text"])
        self.assertEqual(generator.observed["max_tokens"], 50)

    def test_manifest_declares_non_equivalence_and_zero_remote_calls(self):
        manifest = local.prepare()
        self.assertTrue(manifest["backend"]["not_official_equivalent"])
        self.assertEqual(manifest["protocol"]["documents"], 5)
        self.assertEqual(manifest["protocol"]["slots_per_document"], 1)
        self.assertEqual(manifest["protocol"]["maximum_local_generations"], 50)
        self.assertEqual(manifest["resource_counts"]["deepseek_calls"], 0)
        self.assertEqual(manifest["resource_counts"]["bedrock_calls"], 0)

    def test_secret_file_is_not_a_frozen_dependency(self):
        manifest = local.check()
        serialized = json.dumps(manifest)
        self.assertNotIn("api.txt", serialized)
        self.assertNotIn("DEEPSEEK_API_KEY", serialized)


if __name__ == "__main__":
    unittest.main()
