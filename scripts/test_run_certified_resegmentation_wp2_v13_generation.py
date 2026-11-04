import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import run_certified_resegmentation_wp2_v13_generation as runner


class DummyTokenizer:
    def __call__(self, value, **_kwargs):
        return {"input_ids": [0] + value.split() + [1]}


def split_sentences(value):
    return [
        sentence.strip() + "."
        for sentence in value.split(".")
        if sentence.strip()
    ]


class SemStampV13GenerationTests(unittest.TestCase):
    def test_seed_contract_is_inherited_from_v12(self):
        identity = "0" * 64
        self.assertEqual(runner.smoke.generation_seed(identity), 1575591264)

    def test_validate_output_accepts_two_units(self):
        summary, codes = runner.validate_output(
            "First sentence. Second sentence.",
            "Prompt.",
            split_sentences,
            DummyTokenizer(),
            set(),
            set(),
        )
        self.assertEqual(codes, [])
        self.assertEqual(summary["unit_count"], 2)
        self.assertFalse(summary["output_equals_prompt"])

    def test_validate_output_marks_prompt_only(self):
        prompt = "Only prompt."
        summary, codes = runner.validate_output(
            prompt,
            prompt,
            split_sentences,
            DummyTokenizer(),
            set(),
            set(),
        )
        self.assertEqual(codes, ["invalid_or_short_p0"])
        self.assertTrue(summary["output_equals_prompt"])

    def test_validate_output_rejects_forbidden_and_duplicate(self):
        text = "First sentence. Second sentence."
        identity = hashlib.sha256(text.encode("utf-8")).hexdigest()
        _, codes = runner.validate_output(
            text,
            "Prompt.",
            split_sentences,
            DummyTokenizer(),
            {identity},
            {identity},
        )
        self.assertEqual(
            codes,
            [
                "duplicate_generation_output",
                "identity_overlaps_forbidden_set",
            ],
        )

    def test_jsonl_writer_is_canonical(self):
        rows = [{"b": 2, "a": 1}, {"value": "x"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            first = runner.atomic_write_jsonl(path, rows)
            first_bytes = path.read_bytes()
            second = runner.atomic_write_jsonl(path, rows)
        self.assertEqual(first, second)
        self.assertEqual(first["rows"], 2)
        self.assertEqual(
            hashlib.sha256(first_bytes).hexdigest(),
            first["sha256"],
        )

    def test_checkpoint_counts_and_public_redaction(self):
        attempt = {
            "attempt_index": 0,
            "exclusion_codes": [],
            "source_text": "source",
            "prompt": "prompt",
            "generated_text": "output",
            "output": {"raw_text_sha256": "a" * 64},
        }
        checkpoint = {"attempts": [attempt]}
        self.assertEqual(runner.checkpoint_counts(checkpoint), (1, 1))
        public = runner.public_attempt(attempt)
        self.assertNotIn("source_text", public)
        self.assertNotIn("prompt", public)
        self.assertNotIn("generated_text", public)
        self.assertEqual(
            json.loads(json.dumps(public))["attempt_index"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
