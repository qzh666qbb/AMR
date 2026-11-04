"""Model-free tests for the v2 WP3 full detector-replay runner."""

import ast
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_semstamp_certified_defense_v2_wp3_full_replay as full


class AuthorizationTests(unittest.TestCase):
    def make_paths(self, directory):
        output = Path(directory) / "output"
        output.mkdir()
        paths = {
            "output_dir": output,
            "contract": output / "CONTRACT.json",
            "authorization": output / "EXECUTION_AUTHORIZATION.json",
            "resume_authorization": output / "RESUME_AUTHORIZATION.json",
            "checkpoint": output / "sealed/FULL_REPLAY_CHECKPOINT.json",
            "heartbeat": output / "HEARTBEAT.json",
            "last_failure": output / "LAST_FAILURE.json",
            "runner": Path(full.__file__).resolve(),
        }
        paths["contract"].write_bytes(b'{"design":true}\n')
        return paths

    def authorization(self, paths):
        return {
            "schema_version": 1,
            "stage": full.STAGE,
            "status": "FULL_REPLAY_EXECUTION_AUTHORIZED",
            "scope": "3000_row_detector_full_replay_only",
            "protocol_sha256": full.PROTOCOL_SHA256,
            "contract_sha256": full.smoke.sha256_file(paths["contract"]),
            "runner_sha256": full.smoke.sha256_file(paths["runner"]),
            "smoke_final_sha256": full.SMOKE_FINAL_SHA256,
            "authorized_replays": full.REPLAY_COUNT,
            "authorized_rows_per_replay": full.TOTAL_ROWS,
            "deterministic_environment": full.DETERMINISTIC_ENVIRONMENT,
            "wp4_authorized": False,
            "certificate_authorized": False,
        }

    def test_absent_execution_authorization_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_paths(directory)
            with self.assertRaisesRegex(PermissionError, "not authorized"):
                full.validate_authorization(
                    paths,
                    {"execution_authorized": False, "resume_authorized": False},
                )

    def test_exact_execution_authorization_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_paths(directory)
            value = self.authorization(paths)
            paths["authorization"].write_text(
                json.dumps(value), encoding="utf-8"
            )
            self.assertEqual(
                full.validate_authorization(
                    paths,
                    {"execution_authorized": False, "resume_authorized": False},
                ),
                value,
            )

    def test_execution_authorization_cannot_enable_wp4(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_paths(directory)
            value = self.authorization(paths)
            value["wp4_authorized"] = True
            paths["authorization"].write_text(
                json.dumps(value), encoding="utf-8"
            )
            with self.assertRaisesRegex(PermissionError, "is invalid"):
                full.validate_authorization(
                    paths,
                    {"execution_authorized": False, "resume_authorized": False},
                )


class EnvironmentTests(unittest.TestCase):
    def test_exact_deterministic_environment_passes(self):
        with mock.patch.dict(
            os.environ, full.DETERMINISTIC_ENVIRONMENT, clear=True
        ):
            self.assertEqual(
                full.validate_pre_model_environment(),
                full.DETERMINISTIC_ENVIRONMENT,
            )

    def test_missing_cublas_environment_fails_before_import(self):
        values = {
            key: value
            for key, value in full.DETERMINISTIC_ENVIRONMENT.items()
            if key != "CUBLAS_WORKSPACE_CONFIG"
        }
        with mock.patch.dict(os.environ, values, clear=True):
            with self.assertRaisesRegex(RuntimeError, "environment mismatch"):
                full.validate_pre_model_environment()


class ResumeTests(unittest.TestCase):
    def make_checkpoint(self):
        return {
            "attempts": [],
            "replays": {"0": [], "1": []},
        }

    def test_initial_execution_rejects_resume_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            paths = {
                "resume_authorization": output / "RESUME_AUTHORIZATION.json",
            }
            paths["resume_authorization"].write_text(
                "{}\n", encoding="ascii"
            )
            with self.assertRaisesRegex(
                PermissionError, "forbidden for initial"
            ):
                full.validate_resume_authorization(
                    paths, self.make_checkpoint()
                )

    def test_restart_requires_resume_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            paths = {
                "resume_authorization": output / "RESUME_AUTHORIZATION.json",
            }
            checkpoint = self.make_checkpoint()
            checkpoint["attempts"].append({"attempt_index": 0})
            with self.assertRaisesRegex(
                PermissionError, "requires a separate"
            ):
                full.validate_resume_authorization(paths, checkpoint)

    def test_completed_checkpoint_cannot_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            paths = {
                "resume_authorization": output / "RESUME_AUTHORIZATION.json",
            }
            checkpoint = self.make_checkpoint()
            checkpoint["attempts"].append({"attempt_index": 0})
            with mock.patch.object(full, "TOTAL_ROWS", 0):
                with self.assertRaisesRegex(
                    PermissionError, "completed.*cannot resume"
                ):
                    full.validate_resume_authorization(paths, checkpoint)

    def test_exact_resume_authorization_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = AuthorizationTests().make_paths(directory)
            paths["checkpoint"].parent.mkdir()
            paths["authorization"].write_bytes(b'{"authorized":true}\n')
            paths["checkpoint"].write_bytes(b'{"checkpoint":true}\n')
            paths["last_failure"].write_bytes(b'{"failed":true}\n')
            checkpoint = self.make_checkpoint()
            checkpoint["attempts"].append({"attempt_index": 0})
            checkpoint["replays"]["0"].append({"row": 0})
            value = {
                "schema_version": 1,
                "stage": full.STAGE,
                "status": "FULL_REPLAY_RESUME_AUTHORIZED",
                "scope": "exact_checkpoint_prefix_only",
                "contract_sha256": full.smoke.sha256_file(paths["contract"]),
                "execution_authorization_sha256": full.smoke.sha256_file(
                    paths["authorization"]
                ),
                "checkpoint_sha256": full.smoke.sha256_file(
                    paths["checkpoint"]
                ),
                "completed_prefix_lengths": [1, 0],
                "next_attempt_index": 1,
                "evidence_relative_path": "LAST_FAILURE.json",
                "evidence_sha256": full.smoke.sha256_file(
                    paths["last_failure"]
                ),
                "no_score_based_retry": True,
                "wp4_authorized": False,
                "certificate_authorized": False,
            }
            paths["resume_authorization"].write_text(
                json.dumps(value), encoding="utf-8"
            )
            self.assertEqual(
                full.validate_resume_authorization(paths, checkpoint),
                value,
            )


class CheckpointTests(unittest.TestCase):
    def synthetic(self):
        planned = {
            "plan_index": 0,
            "cohort": "calibration_null",
            "cohort_index": 0,
            "raw_text_sha256": "0" * 64,
            "unit_count": 2,
            "max_native_tokens": 4,
        }
        renderer = {
            **planned,
            "atom_count": 2,
            "refinement_count": 0,
            "zero_byte_gap_count": 0,
            "topology_sha256": "1" * 64,
            "legality_sha256": "2" * 64,
            "byte_reconstruction_sha256": "0" * 64,
        }
        reference = dict(planned)
        return [planned], [renderer], [reference]

    def checkpoint(self):
        return {
            "schema_version": 1,
            "stage": full.STAGE,
            "phase": "full_replay",
            "protocol_sha256": full.PROTOCOL_SHA256,
            "contract_sha256": "3" * 64,
            "authorization_sha256": "4" * 64,
            "p0_rows_sha256": full.P0_ROWS_SHA256,
            "v17_rows_sha256": full.V17_ROWS_SHA256,
            "attempts": [],
            "replays": {"0": [], "1": []},
        }

    def test_replay_one_cannot_start_before_replay_zero_complete(self):
        plan, renderer, reference = self.synthetic()
        checkpoint = self.checkpoint()
        checkpoint["replays"]["1"].append({
            "public": {
                "plan_index": 0,
                "full_replay_index": 0,
                "v17_reference_exact": True,
            },
            "regions": [1, 2],
        })
        checkpoint["replays"]["0"] = copy.deepcopy(
            checkpoint["replays"]["1"]
        )
        with mock.patch.object(full, "TOTAL_ROWS", 1):
            full.validate_checkpoint(
                checkpoint,
                "3" * 64,
                "4" * 64,
                plan,
                renderer,
                reference,
            )
        checkpoint["replays"]["0"] = []
        with self.assertRaisesRegex(ValueError, "prefix"):
            full.validate_checkpoint(
                checkpoint,
                "3" * 64,
                "4" * 64,
                plan,
                renderer,
                reference,
            )

    def test_checkpoint_rejects_replay_drift(self):
        plan, renderer, reference = self.synthetic()
        checkpoint = self.checkpoint()
        entry = {
            "public": {
                "plan_index": 0,
                "full_replay_index": 0,
                "v17_reference_exact": True,
            },
            "regions": [1, 2],
        }
        checkpoint["replays"]["0"] = [entry]
        checkpoint["replays"]["1"] = [copy.deepcopy(entry)]
        checkpoint["replays"]["1"][0]["regions"][0] = 7
        with mock.patch.object(full, "TOTAL_ROWS", 1):
            with self.assertRaisesRegex(full.ReplayStop, "replay drift"):
                full.validate_checkpoint(
                    checkpoint,
                    "3" * 64,
                    "4" * 64,
                    plan,
                    renderer,
                    reference,
                )


class IsolationTests(unittest.TestCase):
    def test_import_does_not_load_model_stack(self):
        forbidden = ("sentence_transformers", "torch", "transformers")
        self.assertFalse(any(
            module == prefix or module.startswith(f"{prefix}.")
            for module in sys.modules
            for prefix in forbidden
        ))
        self.assertNotIn(
            "run_certified_resegmentation_wp2_detector_equivalence",
            sys.modules,
        )

    def test_runner_has_only_full_replay_phase(self):
        source = Path(full.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertIn("run_full_replay", functions)
        self.assertNotIn("run_wp4", functions)
        self.assertNotIn("run_certificate", functions)

    def test_parser_rejects_wp4_phase(self):
        original = sys.argv
        try:
            sys.argv = [
                "runner",
                "--root", "/tmp/root",
                "--output-dir", "/tmp/root/output",
                "--nltk-data", "/tmp/nltk",
                "--embedder-dir", "/tmp/model",
                "--phase", "wp4",
            ]
            with self.assertRaises(SystemExit):
                full.parse_args()
        finally:
            sys.argv = original


if __name__ == "__main__":
    unittest.main()
