"""No-model regression tests for the WP2-v10 SemStamp runner."""

import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

import run_certified_resegmentation_wp2_v10_semstamp as runner


def test_relation():
    values = []
    for previous in range(8):
        accepted = {previous, (previous + 3) % 8}
        values.extend(int(current in accepted) for current in range(8))
    return bytes(values)


def source_row(index=200):
    if index < 600:
        cohort = "certificate_calibration"
        role = "null"
    elif index < 1000:
        cohort = "independent_test"
        role = "null"
    else:
        cohort = "positive"
        role = "positive"
    return {
        "cohort": cohort,
        "role": role,
        "row_index": index,
        "raw_text_sha256": "a" * 64,
        "structural_tape_sha256": "b" * 64,
        "atom_count": 8,
        "unit_count": 4,
    }


def p0_row(index, score, relation):
    source = source_row(index)
    regions = [0, 0, 0, 0]
    raw_score = runner.raw_sum_from_regions(regions, relation)
    return {
        "schema_version": 1,
        "protocol_sha256": runner.PROTOCOL_SHA256,
        "relation_sha256": runner.RELATION_SHA256,
        "stage": "p0_detector_health",
        "manifest_index": index,
        **runner.identity_fields(source),
        "edge_count": 3,
        "regions": regions,
        "regions_sha256": runner.sha256_bytes(bytes(regions)),
        "raw_score": raw_score,
        "z_score": float(score),
        "elapsed_seconds": 0.1,
        "contains_raw_text": False,
    }


def summary_p0_row(index, score):
    return {
        "manifest_index": index,
        "z_score": float(score),
    }


def certificate_row(index, score):
    certificates = {}
    for radius in range(4):
        certificates[str(radius)] = {
            "z_score": float(score - radius),
            "threat_set_size": radius + 1,
        }
    return {
        "manifest_index": index,
        "certificates": certificates,
    }


class ScoreAndP0Tests(unittest.TestCase):
    def test_semstamp_score_matches_closed_form(self):
        self.assertEqual(runner.score_from_raw(1, 4), 0.0)
        self.assertAlmostEqual(
            runner.score_from_raw(0, 4),
            -1.0 / (0.75 ** 0.5),
        )

    def test_raw_sum_uses_directed_relation(self):
        relation = test_relation()
        self.assertEqual(
            runner.raw_sum_from_regions([0, 0, 3, 6], relation),
            3,
        )
        self.assertEqual(
            runner.raw_sum_from_regions([0, 3, 0], relation),
            1,
        )

    def test_p0_checkpoint_recomputes_regions_and_score(self):
        relation = test_relation()
        row = p0_row(200, runner.score_from_raw(3, 3), relation)
        self.assertIs(
            runner.validate_p0_checkpoint(
                row,
                200,
                source_row(200),
                relation,
            ),
            row,
        )
        row["raw_score"] = 2
        with self.assertRaises(ValueError):
            runner.validate_p0_checkpoint(
                row,
                200,
                source_row(200),
                relation,
            )

    def test_p0_summary_passes_only_both_health_gates(self):
        rows = []
        rows.extend(
            summary_p0_row(index, index - 200)
            for index in runner.CALIBRATION_RANGE
        )
        rows.extend(
            summary_p0_row(index, -100)
            for index in runner.TEST_RANGE
        )
        rows.extend(
            summary_p0_row(index, 1000)
            for index in runner.POSITIVE_RANGE
            if index != runner.EXCLUDED_MANIFEST_INDEX
        )
        summary = runner.summarize_p0(rows)
        self.assertEqual(summary["test"]["strict_exceedances"], 0)
        self.assertEqual(summary["positive"]["strict_exceedances"], 999)
        self.assertTrue(all(summary["gates"].values()))
        self.assertEqual(summary["status"], "semstamp_p0_pass")

        for row in rows:
            if row["manifest_index"] >= 1000:
                row["z_score"] = -100
        summary = runner.summarize_p0(rows)
        self.assertFalse(
            summary["gates"]["semstamp_clean_baseline_tpr_ge_0_80"]
        )
        self.assertEqual(summary["status"], "semstamp_p0_stop")


class SidecarTests(unittest.TestCase):
    def test_mode5_sidecar_preserves_illegal_records(self):
        relation = test_relation()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.bin"
            codes = np.asarray([1, 7, 13], dtype=np.uint64)
            allowed = np.asarray([True, False, True])
            regions = np.asarray([2, 255, 7], dtype=np.uint8)
            size = runner.write_mode5_sidecar(
                path,
                4,
                codes,
                allowed,
                regions,
                relation,
                chunk_size=1,
            )
            self.assertEqual(size, 8 + 20 + 64 + 10 + 9 + 10)
            with path.open("rb") as handle:
                self.assertEqual(handle.read(8), runner.MAGIC)
                self.assertEqual(
                    struct.unpack("<IIQI", handle.read(20)),
                    (runner.SEMSTAMP_MODE, 4, 3, 1),
                )
                self.assertEqual(handle.read(64), relation)
                self.assertEqual(
                    struct.unpack("<IIBB", handle.read(10)),
                    (0, 1, 1, 2),
                )
                self.assertEqual(
                    struct.unpack("<IIB", handle.read(9)),
                    (1, 2, 0),
                )
                self.assertEqual(
                    struct.unpack("<IIBB", handle.read(10)),
                    (2, 3, 1, 7),
                )
                self.assertEqual(handle.read(), b"")

    def test_subset_rejects_missing_span(self):
        superset = np.asarray([2, 4, 8, 16], dtype=np.uint64)
        subset = np.asarray([2, 8], dtype=np.uint64)
        self.assertTrue(np.array_equal(
            runner.assert_subset(subset, superset),
            np.asarray([0, 2]),
        ))
        with self.assertRaises(ValueError):
            runner.assert_subset(
                np.asarray([3], dtype=np.uint64),
                superset,
            )


class CertificateSummaryTests(unittest.TestCase):
    def test_final_summary_reproduces_p0_and_primary_gates(self):
        rows = []
        p0_rows = []
        for index in runner.CALIBRATION_RANGE:
            score = index - 200
            rows.append(certificate_row(index, score))
            p0_rows.append(summary_p0_row(index, score))
        for index in runner.TEST_RANGE:
            rows.append(certificate_row(index, -100))
            p0_rows.append(summary_p0_row(index, -100))
        for index in runner.POSITIVE_RANGE:
            if index == runner.EXCLUDED_MANIFEST_INDEX:
                continue
            rows.append(certificate_row(index, 1000))
            p0_rows.append(summary_p0_row(index, 1000))
        p0_summary = runner.summarize_p0(p0_rows)
        summary = runner.summarize_results(rows, p0_summary)
        self.assertEqual(summary["calibration"]["0"], p0_summary["calibration"])
        self.assertEqual(summary["test"]["1"]["strict_exceedances"], 0)
        self.assertEqual(summary["positive"]["1"]["strict_exceedances"], 999)
        self.assertTrue(all(summary["gates"].values()))
        self.assertEqual(summary["status"], "semstamp_stage_pass")

    def test_atomic_json_has_no_partial_temp_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            runner.atomic_write_json(path, {"value": 3})
            self.assertEqual(json.loads(path.read_text()), {"value": 3})
            self.assertEqual(list(Path(directory).glob("*.tmp.*")), [])


if __name__ == "__main__":
    unittest.main()
