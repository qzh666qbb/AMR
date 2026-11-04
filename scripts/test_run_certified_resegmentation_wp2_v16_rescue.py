import copy
import json
from pathlib import Path
import tempfile
import unittest

import run_certified_resegmentation_wp2_v16_rescue as runner


ROOT = Path(__file__).resolve().parents[1]
V15 = (
    ROOT
    / "experiments"
    / "certified_resegmentation_wp2_v15_causal_prefix1"
)
V16 = (
    ROOT
    / "experiments"
    / "certified_resegmentation_wp2_v16_prompt_only_rescue1"
)


def tokenizer(_text, **_kwargs):
    return {"input_ids": [1, 2, 3]}


class ParentEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads(
            (V15 / "G1_RESULT.json").read_text(encoding="utf-8")
        )
        cls.positives = runner.parent.read_jsonl(
            V15 / "POSITIVE_IDENTITIES.jsonl"
        )
        cls.h0_stop = json.loads(
            (V15 / "H0_STOP.json").read_text(encoding="utf-8")
        )

    def test_frozen_parent_reclassifies_to_995(self):
        retained, all_output_ids = runner.validate_parent_evidence(
            self.result, self.positives, self.h0_stop
        )
        self.assertEqual(len(retained), runner.RETAINED_PARENT)
        self.assertTrue({
            row["output_raw_text_sha256"] for row in retained
        } <= all_output_ids)
        self.assertFalse(
            set(runner.PROMPT_ONLY_ATTEMPTS)
            & {row["attempt_index"] for row in retained}
        )

    def test_parent_rejects_anomaly_set_drift(self):
        result = copy.deepcopy(self.result)
        result["attempts"][8]["output"]["output_equals_prompt"] = False
        with self.assertRaisesRegex(
            ValueError, "prompt-only attempt set mismatch"
        ):
            runner.validate_parent_evidence(
                result, self.positives, self.h0_stop
            )

    def test_parent_rejects_h0_drift(self):
        h0_stop = copy.deepcopy(self.h0_stop)
        h0_stop["protocol_violation"]["affected_count"] = 4
        with self.assertRaisesRegex(
            ValueError, "H0 Stop evidence mismatch"
        ):
            runner.validate_parent_evidence(
                self.result, self.positives, h0_stop
            )


class RescueValidationTests(unittest.TestCase):
    def test_prompt_only_output_is_excluded(self):
        text = "First sentence. Second sentence."
        output, codes = runner.validate_rescue_output(
            text,
            text,
            lambda _text: ["First sentence.", "Second sentence."],
            tokenizer,
            set(),
            set(),
            set(),
        )
        self.assertTrue(output["output_equals_prompt"])
        self.assertIn("prompt_only_output", codes)

    def test_valid_continuation_is_eligible(self):
        output, codes = runner.validate_rescue_output(
            "First sentence. Second sentence.",
            "Prompt.",
            lambda _text: ["First sentence.", "Second sentence."],
            tokenizer,
            set(),
            set(),
            set(),
        )
        self.assertFalse(output["output_equals_prompt"])
        self.assertEqual(codes, [])

    def test_v15_and_supplement_duplicates_are_distinct(self):
        text = "First sentence. Second sentence."
        identity = runner.smoke.sha256_text(text)
        _, v15_codes = runner.validate_rescue_output(
            text,
            "Prompt.",
            lambda _text: ["First sentence.", "Second sentence."],
            tokenizer,
            set(),
            {identity},
            set(),
        )
        _, supplement_codes = runner.validate_rescue_output(
            text,
            "Prompt.",
            lambda _text: ["First sentence.", "Second sentence."],
            tokenizer,
            set(),
            set(),
            {identity},
        )
        self.assertIn("duplicate_v15_output", v15_codes)
        self.assertIn(
            "duplicate_supplement_output", supplement_codes
        )


class FreezeAndCheckpointTests(unittest.TestCase):
    def test_frozen_document_hashes(self):
        self.assertEqual(
            runner.smoke.sha256_file(V16 / "PROTOCOL.md"),
            runner.PROTOCOL_SHA256,
        )
        self.assertEqual(
            runner.smoke.sha256_file(V16 / "G1R_CONTRACT.md"),
            runner.CONTRACT_SHA256,
        )

    def test_new_checkpoint_is_bound_and_private(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sealed" / "checkpoint.json"
            checkpoint = runner.load_or_create_checkpoint(
                path, "a" * 64
            )
            self.assertEqual(checkpoint["start_attempt"], 1111)
            self.assertEqual(checkpoint["attempts"], [])
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_checkpoint_rejects_runner_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            runner.load_or_create_checkpoint(path, "a" * 64)
            with self.assertRaisesRegex(
                ValueError, "input/code identity mismatch"
            ):
                runner.load_or_create_checkpoint(path, "b" * 64)


if __name__ == "__main__":
    unittest.main()
