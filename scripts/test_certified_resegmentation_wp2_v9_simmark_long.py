"""No-model regression tests for the WP2-v9 SimMark long runner."""

import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

import run_certified_resegmentation_wp2_v9_simmark_long as runner


def source_row(index=200):
    return {
        "cohort": "certificate_calibration",
        "role": "null",
        "row_index": 17,
        "raw_text_sha256": "a" * 64,
        "structural_tape_sha256": "b" * 64,
        "atom_count": 8,
        "unit_count": 4,
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


class GammaAndScoreTests(unittest.TestCase):
    def test_gamma_reproduces_histogram_bin_center_rule(self):
        distances = np.linspace(-0.25, 0.95, 2001)
        observed = runner.fit_gamma(distances)
        counts, bins = np.histogram(distances, bins=1000)
        centers = (bins[:-1] + bins[1:]) / 2
        expected = counts[
            (centers >= 0.68) & (centers <= 0.76)
        ].sum() / len(distances)
        self.assertEqual(observed["gamma"], expected)
        self.assertEqual(observed["pair_count"], len(distances))

    def test_degenerate_zero_gamma_uses_official_epsilon(self):
        observed = runner.fit_gamma([0.9, 0.95])
        self.assertEqual(observed["gamma"], 0.0)
        self.assertEqual(runner.score_from_raw(2e-6, 3, 0.0), 2.0)

    def test_score_is_affine_and_monotone_in_raw_sum(self):
        gamma = 0.2
        low = runner.score_from_raw(1.0, 10, gamma)
        high = runner.score_from_raw(2.0, 10, gamma)
        self.assertLess(low, high)
        self.assertTrue(math.isclose(
            high,
            (2.0 - 2.0) / math.sqrt(1.6 + 1e-12),
        ))

    def test_clopper_pearson_lower_boundaries(self):
        self.assertEqual(runner.clopper_pearson_lower(0, 100), 0.0)
        lower = runner.clopper_pearson_lower(20, 100, 0.95)
        self.assertGreater(lower, 0.1)
        self.assertLess(lower, 0.2)


class SidecarTests(unittest.TestCase):
    def test_streamed_sidecar_has_exact_binary_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.bin"
            features_path = Path(directory) / "features.f32"
            features = np.memmap(
                features_path,
                dtype="<f4",
                mode="w+",
                shape=(3, runner.SIMMARK_DIMENSION),
            )
            features[:] = 0
            features[0, 0] = 1
            features[1, 1] = 1
            features[2, 2] = 1
            features.flush()
            codes = np.asarray([1, 8], dtype=np.uint64)
            positions = np.asarray([0, 2], dtype=np.int64)
            size = runner.write_mode6_sidecar(
                path,
                4,
                codes,
                positions,
                features,
                chunk_size=1,
            )
            expected_size = (
                8 + 20 + 2 * runner.SIDECAR_RECORD_DTYPE.itemsize
            )
            self.assertEqual(size, expected_size)
            with path.open("rb") as handle:
                self.assertEqual(handle.read(8), runner.MAGIC)
                self.assertEqual(
                    struct.unpack("<IIQI", handle.read(20)),
                    (runner.SIMMARK_MODE, 4, 2, runner.SIMMARK_DIMENSION),
                )
                records = np.fromfile(
                    handle,
                    dtype=runner.SIDECAR_RECORD_DTYPE,
                    count=2,
                )
            self.assertEqual(
                list(zip(records["start"], records["end"])),
                [(0, 1), (1, 3)],
            )
            self.assertTrue(np.array_equal(
                records["embedding"],
                np.asarray(features)[positions],
            ))

    def test_sorted_subset_positions(self):
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

    def test_mode5_reader_rejects_trailing_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topology.bin"
            payload = bytearray(runner.MAGIC)
            payload.extend(struct.pack("<IIQI", 5, 4, 1, 1))
            payload.extend(bytes(64))
            payload.extend(struct.pack("<IIBB", 0, 2, 1, 3))
            payload.extend(b"x")
            path.write_bytes(payload)
            with self.assertRaises(ValueError):
                runner.read_mode5_codes(path, 4)


class CheckpointAndSummaryTests(unittest.TestCase):
    def test_certificate_checkpoint_validation(self):
        source = source_row()
        gamma = 0.25
        gamma_sha = "c" * 64
        certificates = {}
        for radius, raw in enumerate((1.0, 0.8, 0.7, 0.6)):
            certificates[str(radius)] = {
                "requested_radius": radius,
                "effective_radius": radius,
                "raw_score": raw,
                "z_score": runner.score_from_raw(raw, 3, gamma),
                "worst_boundaries": [1, 3, 5],
                "original_replay_abs_error": 0.0,
                "witness_replay_abs_error": 0.0,
            }
        row = {
            "schema_version": 1,
            "protocol_sha256": runner.PROTOCOL_SHA256,
            "gamma_artifact_sha256": gamma_sha,
            "stage": "exact_certificate",
            "manifest_index": 200,
            **runner.identity_fields(source),
            "gamma": gamma,
            "certificates": certificates,
        }
        self.assertIs(
            runner.validate_certificate_checkpoint(
                row,
                200,
                source,
                gamma,
                gamma_sha,
            ),
            row,
        )
        row["certificates"]["2"]["raw_score"] = 0.9
        with self.assertRaises(ValueError):
            runner.validate_certificate_checkpoint(
                row,
                200,
                source,
                gamma,
                gamma_sha,
            )

    def test_final_summary_uses_frozen_cohort_ranges(self):
        rows = []
        for index in runner.CALIBRATION_RANGE:
            rows.append(certificate_row(index, index - 200))
        for index in runner.TEST_RANGE:
            rows.append(certificate_row(index, -100))
        for index in runner.POSITIVE_RANGE:
            rows.append(certificate_row(index, 1000))
        gamma = {
            "gamma": 0.25,
            "pair_count": 100,
        }
        summary = runner.summarize_results(rows, gamma)
        self.assertEqual(summary["calibration"]["1"]["calibration_size"], 400)
        self.assertEqual(summary["test"]["1"]["strict_exceedances"], 0)
        self.assertEqual(summary["positive"]["1"]["strict_exceedances"], 1000)
        self.assertTrue(all(summary["gates"].values()))
        self.assertEqual(summary["status"], "simmark_stage_pass")

    def test_atomic_json_has_no_partial_temp_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            runner.atomic_write_json(path, {"value": 3})
            self.assertEqual(json.loads(path.read_text()), {"value": 3})
            self.assertEqual(list(Path(directory).glob("*.tmp.*")), [])


if __name__ == "__main__":
    unittest.main()
