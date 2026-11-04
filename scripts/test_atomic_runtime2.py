"""Frozen cohort mapping and common erasure checks; synthetic inputs only."""
import unittest
import tempfile
from pathlib import Path
import numpy as np

from run_atomic_alignment_completion2 import CONDITIONS, inputs, native_erasure, raw_input, observe
from run_multiscale_transfer import digest, atomic, sha
from semantic_fact_raw_amr import audit_raw


class RuntimeTests(unittest.TestCase):
    def rows(self):
        return [{"group_id": "synthetic", "variant_id": c, "row_id": c,
                 "cohort": "synthetic", "text": "A test sentence.",
                 "text_sha256": digest("A test sentence."),
                 "metadata": {"target_template": "MUST_NOT_PASS"}}
                for c in CONDITIONS]

    def test_blind_mapping_drops_metadata(self):
        rows, meta = inputs(self.rows())
        self.assertEqual(len(rows), 7)
        self.assertTrue(all("metadata" not in r and "cohort" not in r and "group_id" not in r for r in rows))
        self.assertEqual(len(meta), 7)
        self.assertTrue(all(len(r["candidates"]) <= 24 for r in rows))

    def test_missing_condition_or_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            inputs(self.rows()[:-1])
        with self.assertRaises(ValueError):
            inputs(self.rows() + self.rows()[:1])

    def test_native_erasure_is_ineligible_and_keeps_identity(self):
        text = "A test sentence."
        record = native_erasure(text, "f", 1024)
        result = audit_raw(record, raw_input(text), "f")
        self.assertFalse(result["structurally_eligible"])
        self.assertEqual(record["case_id"], digest(text))
        record["text_sha256"] = "foreign"
        with self.assertRaises(ValueError):
            audit_raw(record, raw_input(text), "f")

    def test_all_erased_observations_keep_empty_arms_and_denominator(self):
        class FakeEncoder:
            np = np
            def fits(self, text):
                return True
            def encode(self, texts):
                return np.zeros((len(texts), 3), dtype=np.float32)
            class model:
                @staticmethod
                def get_sentence_embedding_dimension():
                    return 3
        class FakeSurface:
            def analyze_many(self, texts):
                return [{} for _ in texts]
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            atomic(out / "FREEZE.json", {})
            rows, _ = inputs(self.rows())
            for candidate in rows[0]["candidates"]:
                text = candidate["text"]
                atomic(out / "private/raw_graphs" / (digest(text) + ".json"),
                       native_erasure(text, sha(out / "FREEZE.json"), 1024))
            result = observe(rows[0], out, FakeEncoder(), FakeSurface(), None)
            self.assertEqual(len(result["observations"]), len(rows[0]["candidates"]))
            self.assertEqual(result["A0"]["statistic"], 0)
            self.assertEqual(result["A1"]["statistic"], 0)
            self.assertEqual(result["universe"], [])


if __name__ == "__main__":
    unittest.main()
