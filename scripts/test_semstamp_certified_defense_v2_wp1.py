"""Model-free tests for the SemStamp certified-defense v2 WP1 runner."""

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_semstamp_certified_defense_v2_wp1 as wp1
from semstamp_refined_atom_renderer import canonicalize_refined_bytes


def identity(value):
    return hashlib.sha256(str(value).encode("ascii")).hexdigest()


class RendererRowTests(unittest.TestCase):
    def test_renderer_row_binds_refinement_without_text(self):
        text = "  Alpha ends.Next begins. \t"
        document = canonicalize_refined_bytes(
            text.encode("utf-8"),
            lambda _value: ["Alpha ends.", "Next begins."],
        )
        planned = {
            "cohort": "calibration_null",
            "cohort_index": 0,
            "raw_text_sha256": hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest(),
            "unit_count": 2,
            "max_native_tokens": 5,
        }
        row = wp1.renderer_row(0, planned, document, (0,))
        self.assertTrue(row["legacy_v1_alignment_failure"])
        self.assertEqual(row["refinement_count"], 1)
        self.assertEqual(row["zero_byte_gap_count"], 1)
        self.assertTrue(row["zero_byte_gaps_all_original"])
        self.assertFalse(row["contains_raw_text"])
        self.assertEqual(row["score_access_count_during_construction"], 0)
        self.assertEqual(
            row["byte_reconstruction_sha256"],
            planned["raw_text_sha256"],
        )
        wp1.reject_raw_public_keys(row)

    def test_summary_counts_legacy_repairs(self):
        rows = []
        for index, cohort in enumerate(wp1.COHORTS):
            row = {
                "status": "pass",
                "cohort": cohort,
                "raw_text_sha256": identity(index),
                "lexical_span_count": 2,
                "atom_count": 3,
                "refinement_count": 1,
                "zero_byte_gap_count": 1,
                "legacy_v1_alignment_failure": index < 2,
                "topology_sha256": identity(f"topology-{index}"),
                "legality_sha256": identity(f"legality-{index}"),
                "byte_reconstruction_sha256": identity(index),
            }
            rows.append(row)
        summary = wp1.summarize_rows(rows)
        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["valid_rows"], 3)
        self.assertEqual(summary["refinement_total"], 3)
        self.assertEqual(summary["legacy_v1_failures_repaired"], 2)


class ArtifactTests(unittest.TestCase):
    def test_atomic_output_is_canonical_and_private(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            wp1.atomic_write_json(path, {"b": 2, "a": 1})
            self.assertEqual(path.read_bytes(), b'{"a":1,"b":2}\n')
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_public_key_guard_rejects_text(self):
        with self.assertRaisesRegex(ValueError, "raw text key"):
            wp1.reject_raw_public_keys({"nested": {"text": "secret"}})

    def test_runner_import_does_not_import_detector_or_torch(self):
        self.assertNotIn("torch", vars(wp1))
        self.assertNotIn(
            "run_certified_resegmentation_wp2_detector_equivalence",
            sys.modules,
        )


if __name__ == "__main__":
    unittest.main()
