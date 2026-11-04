"""Network-free tests for the cross-watermark transfer runner."""

import json
import math
from pathlib import Path
import tempfile
import unittest

from audit_cross_watermark_transfer import paired_test
from run_cross_watermark_transfer import (
    DenseCpuSynthIDProcessor,
    GENERATION,
    K,
    SYNTHID,
    WATERMARKS,
    calibrated_threshold,
    generation_seed,
    make_synthid_processor,
    prepare,
    summarize,
    synthid_score_token_ids,
)


class SynthIDAdapterTests(unittest.TestCase):
    def test_dense_wrapper_matches_official_sparse_cpu_result(self):
        import torch

        torch.manual_seed(7)
        input_ids = torch.tensor([[3, 5, 8, 13, 21]], dtype=torch.long)
        scores = torch.randn(1, 64)
        official = make_synthid_processor()
        expected, indices, _ = official.watermarked_call(input_ids, scores)
        wrapper = DenseCpuSynthIDProcessor()
        actual = wrapper(input_ids, scores)
        self.assertTrue(torch.allclose(actual.gather(1, indices), expected))
        selected = torch.zeros_like(actual, dtype=torch.bool)
        selected.scatter_(1, indices, True)
        self.assertTrue(torch.isneginf(actual[~selected]).all())
        self.assertEqual(wrapper.processor.device.type, "cpu")

    def test_detector_is_finite_and_cpu_only(self):
        result = synthid_score_token_ids(list(range(1, 35)))
        self.assertTrue(math.isfinite(result["score"]))
        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 1)
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["depth"], len(SYNTHID["keys"]))

    def test_detector_rejects_short_text(self):
        with self.assertRaises(ValueError):
            synthid_score_token_ids([1, 2, 3, 4])


class ProtocolTests(unittest.TestCase):
    def test_frozen_budget_and_shared_sampling(self):
        self.assertEqual(WATERMARKS, ("kgw", "synthid"))
        self.assertEqual(K, 4)
        self.assertEqual(GENERATION["temperature"], 0.5)
        self.assertEqual(GENERATION["top_k"], 40)
        self.assertEqual(GENERATION["top_p"], 0.99)
        self.assertEqual(SYNTHID["ngram_len"], 5)
        self.assertEqual(len(SYNTHID["keys"]), 30)

    def test_seed_is_stable_and_prompt_specific(self):
        self.assertEqual(generation_seed("a"), generation_seed("a"))
        self.assertNotEqual(generation_seed("a"), generation_seed("b"))

    def test_strict_one_percent_threshold(self):
        scores = [float(index) for index in range(250)]
        threshold = calibrated_threshold(scores)
        self.assertEqual(threshold, 247.0)
        self.assertEqual(sum(score > threshold for score in scores), 2)
        with self.assertRaises(ValueError):
            calibrated_threshold([])
        with self.assertRaises(ValueError):
            calibrated_threshold([float("nan")])

    def test_paired_exact_test(self):
        result = paired_test(
            {"a": True, "b": True, "c": False},
            {"a": True, "b": False, "c": True},
        )
        self.assertEqual(result["both_success"], 1)
        self.assertEqual(result["kgw_only_success"], 1)
        self.assertEqual(result["synthid_only_success"], 1)
        self.assertEqual(result["mcnemar_exact_two_sided_p"], 1.0)
        with self.assertRaises(ValueError):
            paired_test({"a": True}, {"b": True})


class ArtifactTests(unittest.TestCase):
    def test_prepare_freezes_independent_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            prepare(folder)
            manifest = json.loads((folder / "manifest.json").read_text())
            self.assertEqual(len(manifest["prompts"]), 30)
            self.assertEqual(len(manifest["human_records"]), 500)
            self.assertEqual(
                sum(
                    row["cohort"] == "human_calibration"
                    for row in manifest["human_records"]
                ),
                250,
            )
            self.assertIn(
                "boundary_only",
                manifest["protocol"]["attack"]["name"],
            )
            self.assertIn(
                "before any target attack score",
                manifest["protocol"]["quality"]["selection"],
            )
            summary = summarize(folder)
            self.assertEqual(summary["status"], "incomplete")
            self.assertIsNone(
                summary["watermarks"]["kgw"]["no_feedback_valid_asr"]
            )


if __name__ == "__main__":
    unittest.main()
