"""Model-free tests for the SemStamp certified-defense v2 WP3 P0 runner."""

from collections import Counter
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_semstamp_certified_defense_v2_wp3_p0 as wp3
from semstamp_refined_atom_renderer import canonicalize_refined_bytes


def digest(value):
    return hashlib.sha256(str(value).encode("ascii")).hexdigest()


def synthetic_rows(mandatory_counts):
    rows = []
    plan_index = 0
    for cohort, mandatory_count in zip(wp3.COHORTS, mandatory_counts):
        for cohort_index in range(wp3.COHORT_SIZE):
            mandatory = cohort_index < mandatory_count
            rows.append({
                "status": "pass",
                "plan_index": plan_index,
                "cohort": cohort,
                "cohort_index": cohort_index,
                "raw_text_sha256": digest(plan_index),
                "atom_count": 2 + (cohort_index % 31),
                "unit_count": 2 + (cohort_index % 11),
                "refinement_count": 1 if mandatory else 0,
                "zero_byte_gap_count": 1 if mandatory else 0,
                "topology_sha256": digest(f"topology-{plan_index}"),
                "legality_sha256": digest(f"legality-{plan_index}"),
                "wp1_reference_row_sha256": digest(
                    f"parent-{plan_index}"
                ),
            })
            plan_index += 1
    return rows


class SmokeSelectionTests(unittest.TestCase):
    def test_selection_is_balanced_risk_complete_and_deterministic(self):
        mandatory_counts = (17, 20, 5)
        rows = synthetic_rows(mandatory_counts)
        selected = wp3.select_smoke_rows(rows)

        self.assertEqual(len(selected), wp3.SMOKE_ROWS)
        self.assertEqual(
            Counter(row["cohort"] for row in selected),
            {cohort: wp3.SMOKE_PER_COHORT for cohort in wp3.COHORTS},
        )
        risk = {
            row["plan_index"]
            for row in rows
            if row["refinement_count"] or row["zero_byte_gap_count"]
        }
        self.assertEqual(len(risk), wp3.LEGACY_FAILURE_COUNT)
        self.assertTrue(risk.issubset({
            row["plan_index"] for row in selected
        }))
        self.assertEqual(
            wp3.jsonl_bytes(selected),
            wp3.jsonl_bytes(wp3.select_smoke_rows(list(reversed(rows)))),
        )

        for cohort, mandatory_count in zip(
            wp3.COHORTS, mandatory_counts
        ):
            cohort_rows = [
                row for row in rows
                if row["cohort"] == cohort
                and row["refinement_count"] == 0
            ]
            cohort_rows.sort(key=lambda row: (
                row["atom_count"],
                row["unit_count"],
                row["raw_text_sha256"],
                row["plan_index"],
            ))
            fill_count = wp3.SMOKE_PER_COHORT - mandatory_count
            expected_ranks = [
                ((2 * index + 1) * len(cohort_rows))
                // (2 * fill_count)
                for index in range(fill_count)
            ]
            observed_ranks = [
                row["structural_rank"]
                for row in selected
                if (
                    row["cohort"] == cohort
                    and row["selection_class"] == "structural_quantile"
                )
            ]
            self.assertEqual(observed_ranks, expected_ranks)

    def test_selection_stops_if_mandatory_rows_exceed_budget(self):
        rows = synthetic_rows((wp3.SMOKE_PER_COHORT + 1, 0, 0))
        with self.assertRaisesRegex(
            ValueError, "mandatory smoke rows exceed budget"
        ):
            wp3.select_smoke_rows(rows)


class ParentBindingTests(unittest.TestCase):
    def setUp(self):
        self.text = "  Alpha ends.Next begins. \t"
        self.document = canonicalize_refined_bytes(
            self.text.encode("utf-8"),
            lambda _value: ["Alpha ends.", "Next begins."],
        )
        self.planned = {
            "cohort": "calibration_null",
            "cohort_index": 0,
            "raw_text_sha256": hashlib.sha256(
                self.text.encode("utf-8")
            ).hexdigest(),
            "unit_count": 2,
            "max_native_tokens": 5,
        }

    def parent_row(self):
        structural = wp3.structural_row(
            0, self.planned, self.document, (0,)
        )
        return {
            "schema_version": 1,
            "protocol_sha256": wp3.WP1_PROTOCOL_SHA256,
            **structural,
        }

    def test_p0_row_binds_exact_wp1_parent_without_raw_text(self):
        parent = self.parent_row()
        row = wp3.p0_row(
            0, self.planned, self.document, (0,), parent
        )
        self.assertTrue(row["wp1_reference_exact"])
        self.assertEqual(
            row["wp1_reference_row_sha256"],
            wp3.sha256_bytes(wp3.canonical_bytes(parent)),
        )
        self.assertEqual(row["stage"], wp3.STAGE)
        self.assertEqual(row["protocol_sha256"], wp3.PROTOCOL_SHA256)
        self.assertFalse(row["contains_raw_text"])
        wp3.reject_raw_public_keys(row)

    def test_p0_row_rejects_parent_structural_drift(self):
        parent = self.parent_row()
        parent["topology_sha256"] = digest("drift")
        with self.assertRaisesRegex(ValueError, "structural drift"):
            wp3.p0_row(
                0, self.planned, self.document, (0,), parent
            )


class ArtifactAndIsolationTests(unittest.TestCase):
    def test_atomic_output_is_canonical_and_private(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            wp3.atomic_write_json(path, {"b": 2, "a": 1})
            self.assertEqual(path.read_bytes(), b'{"a":1,"b":2}\n')
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_public_key_guard_rejects_text(self):
        with self.assertRaisesRegex(ValueError, "raw text key"):
            wp3.reject_raw_public_keys({"nested": {"text": "secret"}})

    def test_runner_import_does_not_import_detector_or_model_stack(self):
        forbidden = (
            "sentence_transformers",
            "torch",
            "transformers",
        )
        loaded = tuple(sys.modules)
        self.assertFalse(any(
            module == prefix or module.startswith(f"{prefix}.")
            for module in loaded
            for prefix in forbidden
        ))
        self.assertNotIn(
            "run_certified_resegmentation_wp2_detector_equivalence",
            sys.modules,
        )


if __name__ == "__main__":
    unittest.main()
