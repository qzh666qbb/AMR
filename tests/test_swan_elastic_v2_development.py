from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.core.elastic_v2_development import (
    DEVELOPMENT_COHORTS,
    VARIANTS,
    prepare,
    read_json,
    read_rows,
    summarize,
    verify_inputs,
)


def detection(status="ok", invalid=0, selected_invalid=0):
    return {
        "status": status,
        "invalid_edge_count": invalid,
        "selected_invalid_edges": selected_invalid,
    }


def development_row(cohort, group, variant, detected, status="ok"):
    return {
        "cohort": cohort,
        "group_id": group,
        "variant_id": variant,
        "calibration": {"is_watermarked": detected},
        "detection": detection(status=status),
    }


class ElasticV2DevelopmentTests(unittest.TestCase):
    def test_prepare_freezes_only_declared_development_cohorts(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            manifest = prepare(output)
            self.assertEqual(manifest["outputs"]["calibration"]["rows"], 1000)
            self.assertEqual(manifest["outputs"]["development"]["rows"], 210)
            rows = read_rows(output / "inputs/development_rows.jsonl")
            self.assertEqual(
                {row["cohort"] for row in rows},
                set(DEVELOPMENT_COHORTS),
            )
            self.assertEqual(verify_inputs(output), read_json(
                output / "INPUT_MANIFEST.json"
            ))

    def test_summary_uses_baseline_conditional_all_five_denominator(self):
        rows = [
            development_row(
                "confirmation_baseline_positive", "a", "baseline", True
            ),
            development_row(
                "confirmation_baseline_positive", "b", "baseline", False
            ),
            development_row(
                "confirmation_boundary_attack",
                "a",
                "boundary_only_candidate_0",
                True,
            ),
            development_row(
                "confirmation_boundary_attack",
                "b",
                "boundary_only_candidate_0",
                True,
            ),
        ]
        for variant in VARIANTS:
            rows.extend([
                development_row(
                    "confirmation_punctuation_stress",
                    "a",
                    variant,
                    variant != "space_all",
                ),
                development_row(
                    "confirmation_punctuation_stress",
                    "b",
                    variant,
                    True,
                ),
            ])
        calibration = [{
            "detection": detection(status="abstain", invalid=2),
        }]
        summary = summarize(calibration, rows)
        self.assertEqual(
            summary["baseline_detection"],
            {"successes": 1, "trials": 2, "rate": 0.5},
        )
        self.assertEqual(
            summary["survival_all_five_variants"],
            {"successes": 0, "trials": 1, "rate": 0.0},
        )
        self.assertEqual(summary["calibration"]["abstain"]["successes"], 1)
        self.assertEqual(summary["invalid_candidate_edges"], 2)

    def test_abstain_is_retained_in_development_denominator(self):
        rows = [
            development_row(
                "confirmation_baseline_positive",
                "a",
                "baseline",
                False,
                status="abstain",
            ),
        ]
        summary = summarize([], rows)
        self.assertEqual(
            summary["development_abstain"],
            {"successes": 1, "trials": 1, "rate": 1.0},
        )
        self.assertEqual(summary["baseline_detection"]["trials"], 1)


if __name__ == "__main__":
    unittest.main()

