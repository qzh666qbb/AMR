"""Network-free tests for the SIRA-adapted comparison runner."""

import json
from pathlib import Path
import tempfile
import unittest

from run_sira_adapted_baseline import (
    ATTACK_PROMPT, K, MASK_PERCENTILE, REFERENCE_PROMPT, mask_tokens, prepare,
    summarize,
)


class SiraMaskTests(unittest.TestCase):
    def test_official_percentile_masking_and_chunk_format(self):
        result = mask_tokens([" A", " B", " C", " D"], [1., 2., 3., 4.], percentile=50)
        self.assertEqual(result["mask"], [False, False, True, True])
        self.assertEqual(result["masked_tokens"], 2)
        self.assertEqual(result["blank_text"], "( A  B)__")

    def test_mask_validation(self):
        with self.assertRaises(ValueError):
            mask_tokens([], [])
        with self.assertRaises(ValueError):
            mask_tokens(["a"], [float("nan")])
        with self.assertRaises(ValueError):
            mask_tokens(["a"], [1.], percentile=101)

    def test_protocol_constants(self):
        self.assertEqual(K, 4)
        self.assertEqual(MASK_PERCENTILE, 30)
        self.assertIn("You are a paraphraser", REFERENCE_PROMPT)
        self.assertIn("incomplete paragraph", ATTACK_PROMPT)


class SiraArtifactTests(unittest.TestCase):
    def test_prepare_and_partial_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            prepare(folder)
            manifest = json.loads((folder / "manifest.json").read_text())
            self.assertEqual(len(manifest["records"]), 30)
            self.assertTrue(manifest["protocol"]["not_official_reproduction"])
            result = summarize(folder)
            self.assertEqual(result["status"], "incomplete")
            self.assertIsNone(result["feedback_valid_asr"])
            self.assertEqual(result["artifact_counts"]["attacks"], 0)

    def test_completed_audit_regression(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "experiments/sira_adapted30/audit.json"
        if not path.exists():
            self.skipTest("Complete SIRA-adapted experiment is not present")
        result = json.loads(path.read_text())
        self.assertEqual(result["status"], "complete_independently_verified")
        self.assertEqual(result["coverage"]["candidates"], 120)
        self.assertEqual(result["coverage"]["quality_candidates"], 53)
        self.assertEqual(
            result["paragraph_rates"]["feedback_valid_asr"]["count"], 13)
        self.assertEqual(
            result["paragraph_rates"]["no_feedback_valid_asr"]["count"], 8)
        self.assertEqual(
            result["mask_statistics"]["fresh_value_mismatches_at_1e_5"], 0)


if __name__ == "__main__":
    unittest.main()
