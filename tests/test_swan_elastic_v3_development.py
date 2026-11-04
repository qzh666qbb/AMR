from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

from src.core.elastic_v2 import CountStatePathOptimizer
from src.core.elastic_v3 import (
    InvalidTolerantAMRExtractor,
    SwanAMRBankScorerV3,
    V3EdgeEvidence,
)
from src.core.elastic_v3_development import (
    build_v3_detector,
    detection_payload,
    load_checkpoint,
    prepare,
    summarize,
    verify_inputs,
)
from src.core.models import BankMatch, Boundary, Lattice, LatticeEdge


def match(score: float) -> BankMatch:
    return BankMatch(0, score, score, score, score)


def lattice() -> Lattice:
    return Lattice(
        text="alpha beta",
        boundaries=(
            Boundary(0, 0, "terminal"),
            Boundary(1, 5, "test"),
            Boundary(2, 10, "terminal"),
        ),
        edges=(
            LatticeEdge(0, 1, "alpha", 1),
            LatticeEdge(1, 2, "beta", 1),
            LatticeEdge(0, 2, "alpha beta", 2),
        ),
    )


def audit(span_records=1, graph_valid=1, roundtrip_valid=1):
    return {
        "span_records": span_records,
        "graph_valid_records": graph_valid,
        "roundtrip_valid_records": roundtrip_valid,
        "warning_span_records": 0,
        "warning_records": 0,
        "warning_details": [],
    }


def detection(
    status="ok",
    invalid=0,
    reasons=None,
    selected_invalid=0,
):
    return {
        "status": status,
        "invalid_edge_count": invalid,
        "invalid_edge_reasons": reasons or {},
        "selected_invalid_edges": selected_invalid,
    }


def development_row(cohort, group, variant, detected, **kwargs):
    return {
        "cohort": cohort,
        "group_id": group,
        "variant_id": variant,
        "calibration": {"is_watermarked": detected},
        "detection": detection(**kwargs),
        "penman_audit": audit(
            span_records=2,
            graph_valid=1 if kwargs.get("invalid") else 2,
            roundtrip_valid=1 if kwargs.get("invalid") else 2,
        ),
    }


class ElasticV3DevelopmentTests(unittest.TestCase):
    def test_prepare_starts_new_calibration_at_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            manifest = prepare(output)
            self.assertEqual(manifest["outputs"]["calibration"]["rows"], 1000)
            self.assertEqual(manifest["outputs"]["development"]["rows"], 210)
            self.assertEqual(
                manifest["selection"]["calibration_start_index"],
                0,
            )
            self.assertFalse(manifest["selection"]["v2_partial_reused"])
            self.assertEqual(verify_inputs(output), manifest)

    def test_build_uses_versioned_extractor_scorer_and_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            args = Namespace(
                backend="heuristic",
                model_dir="",
                tokenizer_dir="",
                device="cpu",
                model_batch_size=1,
                request_batch_size=8,
                cache_db=str(Path(folder) / "v3.db"),
                bank=(
                    "third_party/SWAN_official_2ca0c207_clean/"
                    "amr_bank/banks/amr_bank_100.json"
                ),
                max_boundary_hops=3,
                max_span_words=80,
                capitalization_gap_words=12,
                long_gap_stride_words=40,
                score_center=0.65,
                green_threshold=0.75,
            )
            detector = build_v3_detector(args)
            self.assertIsInstance(detector.scorer, SwanAMRBankScorerV3)
            self.assertIsInstance(
                detector.scorer.extractor,
                InvalidTolerantAMRExtractor,
            )
            self.assertEqual(
                Path(detector.scorer.extractor.cache.db_path),
                Path(args.cache_db).resolve(),
            )

    def test_detection_payload_reports_and_excludes_invalid_edge(self):
        graph = lattice()
        evidence = (
            V3EdgeEvidence(True, (), True, True, 0),
            V3EdgeEvidence(True, (), True, True, 0),
            V3EdgeEvidence(
                False,
                ("graph_invalid", "roundtrip_invalid"),
                False,
                False,
                0,
            ),
        )
        result = CountStatePathOptimizer(score_center=0.65).optimize(
            graph,
            (match(0.85), match(0.85), match(1.0)),
            tuple(item.valid for item in evidence),
        )
        payload = detection_payload(result, graph, evidence)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["invalid_edge_count"], 1)
        self.assertEqual(payload["selected_invalid_edges"], 0)
        self.assertEqual(
            payload["invalid_edge_reasons"],
            {"graph_invalid": 1, "roundtrip_invalid": 1},
        )
        self.assertEqual(len(payload["units"]), 2)

    def test_summary_keeps_abstain_and_invalid_rows_in_denominators(self):
        calibration = [{
            "detection": detection(
                status="abstain",
                invalid=1,
                reasons={"graph_invalid": 1},
            ),
            "penman_audit": audit(2, 1, 1),
        }]
        rows = [
            development_row(
                "confirmation_baseline_positive",
                "a",
                "baseline",
                False,
                status="abstain",
                invalid=1,
                reasons={"roundtrip_invalid": 1},
            ),
        ]
        for variant in (
            "semicolon_all",
            "semicolon_pairs",
            "comma_all",
            "newline_all",
            "space_all",
        ):
            rows.append(development_row(
                "confirmation_punctuation_stress",
                "a",
                variant,
                False,
            ))
        summary = summarize(calibration, rows)
        self.assertEqual(summary["calibration"]["abstain"]["trials"], 1)
        self.assertEqual(summary["development_abstain"]["trials"], 6)
        self.assertEqual(summary["baseline_detection"]["trials"], 1)
        self.assertEqual(summary["invalid_candidate_edges"], 2)
        self.assertEqual(
            summary["invalid_candidate_edge_reasons"],
            {"graph_invalid": 1, "roundtrip_invalid": 1},
        )
        self.assertEqual(
            summary["row_local_graph_invalid_span_records"],
            2,
        )

    def test_checkpoint_must_be_an_input_prefix(self):
        inputs = [
            {"window_sha256": "first"},
            {"window_sha256": "second"},
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.jsonl"
            path.write_text(
                '{"window_sha256":"wrong"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                load_checkpoint(path, inputs, "calibration")


if __name__ == "__main__":
    unittest.main()
