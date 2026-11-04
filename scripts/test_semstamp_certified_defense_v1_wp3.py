"""Model-free tests for the SemStamp certified-defense WP3 runner."""

import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_semstamp_certified_defense_v1_wp3 as wp3
from semstamp_fixed_byte_renderer import canonicalize_fixed_bytes


def identity(value):
    return hashlib.sha256(str(value).encode("ascii")).hexdigest()


class SelectionTests(unittest.TestCase):
    def test_smoke_selection_is_score_blind_and_stratified(self):
        rows = []
        for cohort_index, cohort in enumerate(wp3.COHORTS):
            for index in range(wp3.COHORT_SIZE):
                rows.append({
                    "plan_index": cohort_index * wp3.COHORT_SIZE + index,
                    "cohort": cohort,
                    "cohort_index": index,
                    "raw_text_sha256": identity(
                        cohort_index * wp3.COHORT_SIZE + index
                    ),
                    "atom_count": (index * 17) % 211 + 2,
                    "unit_count": index % 37 + 2,
                    "topology_sha256": identity(f"topology-{cohort}-{index}"),
                    "legality_sha256": identity(f"legality-{cohort}-{index}"),
                })
        selected = wp3.select_smoke_rows(rows)
        self.assertEqual(len(selected), wp3.SMOKE_ROWS)
        self.assertEqual(
            [row["structural_rank"] for row in selected[:32]],
            [
                ((2 * index + 1) * 1000) // 64
                for index in range(32)
            ],
        )
        self.assertTrue(all(
            row["score_access_count_during_construction"] == 0
            for row in selected
        ))
        altered = copy.deepcopy(rows)
        altered.reverse()
        self.assertEqual(
            wp3.canonical_bytes(wp3.select_smoke_rows(altered)),
            wp3.canonical_bytes(selected),
        )


class RendererBridgeTests(unittest.TestCase):
    def test_renderer_row_binds_plan_without_text(self):
        text = "  First sentence.\n\nSecond sentence. \t"
        document = canonicalize_fixed_bytes(
            text.encode("utf-8"),
            lambda _value: ["First sentence.", "Second sentence."],
        )
        planned = {
            "plan_index": 0,
            "cohort": "calibration_null",
            "cohort_index": 0,
            "raw_text_sha256": hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest(),
            "unit_count": 2,
            "max_native_tokens": 5,
        }
        row = wp3.renderer_row(0, planned, document)
        self.assertFalse(row["contains_raw_text"])
        self.assertEqual(row["score_access_count_during_construction"], 0)
        self.assertEqual(
            row["byte_reconstruction_sha256"],
            planned["raw_text_sha256"],
        )
        wp3.reject_raw_public_keys(row)


class ReplayTests(unittest.TestCase):
    def test_official_replay_model_requires_exact_consumption(self):
        model = wp3.ReplayLSHModel([2, 5])
        self.assertEqual(model.get_hash(["a"]), [2])
        with self.assertRaises(AssertionError):
            model.assert_exhausted()
        self.assertEqual(model.get_hash(["b"]), [5])
        model.assert_exhausted()
        with self.assertRaises(AssertionError):
            model.get_hash(["c"])

    def test_checkpoint_rejects_replay_drift(self):
        indices = [3, 8]
        checkpoint = {
            "schema_version": 1,
            "stage": wp3.STAGE,
            "phase": "smoke",
            "contract_sha256": identity("contract"),
            "p0_readiness_sha256": identity("p0"),
            "indices": indices,
            "replays": {
                "0": [{
                    "public": {
                        "plan_index": 3,
                        "v17_reference_exact": True,
                    },
                    "regions": [1, 2],
                }],
                "1": [],
            },
        }
        wp3.validate_checkpoint(
            checkpoint,
            "smoke",
            indices,
            identity("contract"),
            identity("p0"),
        )
        checkpoint["replays"]["1"] = copy.deepcopy(
            checkpoint["replays"]["0"]
        )
        checkpoint["replays"]["1"][0]["regions"][0] = 7
        with self.assertRaisesRegex(ValueError, "replay mismatch"):
            wp3.validate_checkpoint(
                checkpoint,
                "smoke",
                indices,
                identity("contract"),
                identity("p0"),
            )

    def test_atomic_output_is_canonical_and_private(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            wp3.atomic_write_json(path, {"b": 2, "a": 1})
            self.assertEqual(path.read_bytes(), b'{"a":1,"b":2}\n')
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_runner_import_does_not_import_detector_or_torch(self):
        self.assertNotIn("torch", vars(wp3))
        self.assertNotIn(
            "run_certified_resegmentation_wp2_detector_equivalence",
            sys.modules,
        )


if __name__ == "__main__":
    unittest.main()
