"""Offline tests for normal-length coherent baseline detection."""
import unittest

import run_swan_normal_length_coherent_baseline as baseline


class SourceTests(unittest.TestCase):
    def test_source_is_strict_normal_length_and_quality_passed(self):
        inputs = baseline.source_inputs()
        self.assertEqual(len(inputs["selected"]), 20)
        self.assertEqual(inputs["document"]["word_count"], 126)
        self.assertEqual(len(inputs["document"]["segmentation_spans"]), 20)
        self.assertTrue(
            inputs["quality_summary"]["model_assisted_gate_pass"]
        )
        self.assertTrue(
            all(row["target_score"] >= 0.75 for row in inputs["selected"])
        )

    def test_equivalence_contract_is_fixed(self):
        self.assertEqual(
            baseline.EQUIVALENCE_PAIRS,
            ((0, 0), (9, 25), (19, 47)),
        )
        self.assertEqual(baseline.EQUIVALENCE_TOLERANCE, 1e-12)

    def test_score_seed_namespace_is_distinct(self):
        self.assertNotEqual(
            baseline.seed("detect", "doc", 0, 47),
            baseline.base.seed("detect", "doc", 0, 47),
        )


if __name__ == "__main__":
    unittest.main()
