"""Model-free tests for the v2 WP3 detector-smoke runner."""

import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_semstamp_certified_defense_v2_wp3_smoke as smoke


def digest(value):
    return hashlib.sha256(str(value).encode("ascii")).hexdigest()


class AuthorizationTests(unittest.TestCase):
    def make_paths(self, directory):
        root = Path(directory)
        output = root / "output"
        output.mkdir()
        paths = {
            "contract": output / "CONTRACT.json",
            "authorization": output / "EXECUTION_AUTHORIZATION.json",
            "runner": Path(smoke.__file__).resolve(),
        }
        paths["contract"].write_bytes(b'{"design":true}\n')
        return paths

    def authorization(self, paths):
        return {
            "schema_version": 1,
            "stage": smoke.STAGE,
            "status": "SMOKE_EXECUTION_AUTHORIZED",
            "scope": "96_row_detector_smoke_only",
            "protocol_sha256": smoke.PROTOCOL_SHA256,
            "contract_sha256": smoke.sha256_file(paths["contract"]),
            "runner_sha256": smoke.sha256_file(paths["runner"]),
            "smoke_plan_sha256": smoke.SMOKE_PLAN_SHA256,
            "authorized_replays": smoke.REPLAY_COUNT,
            "authorized_rows_per_replay": smoke.SMOKE_ROWS,
            "full_replay_authorized": False,
            "wp4_authorized": False,
            "certificate_authorized": False,
        }

    def test_absent_authorization_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_paths(directory)
            with self.assertRaisesRegex(PermissionError, "not authorized"):
                smoke.validate_authorization(
                    paths, {"execution_authorized": False}
                )

    def test_exact_authorization_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_paths(directory)
            value = self.authorization(paths)
            paths["authorization"].write_text(
                json.dumps(value), encoding="utf-8"
            )
            self.assertEqual(
                smoke.validate_authorization(
                    paths, {"execution_authorized": False}
                ),
                value,
            )

    def test_authorization_cannot_enable_full_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_paths(directory)
            value = self.authorization(paths)
            value["full_replay_authorized"] = True
            paths["authorization"].write_text(
                json.dumps(value), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                PermissionError, "authorization is invalid"
            ):
                smoke.validate_authorization(
                    paths, {"execution_authorized": False}
                )


class ReplayTests(unittest.TestCase):
    def test_replay_model_requires_exact_consumption(self):
        model = smoke.ReplayLSHModel([2, 5])
        self.assertEqual(model.get_hash(["a"]), [2])
        with self.assertRaises(AssertionError):
            model.assert_exhausted()
        self.assertEqual(model.get_hash(["b"]), [5])
        model.assert_exhausted()
        with self.assertRaises(AssertionError):
            model.get_hash(["c"])

    def checkpoint(self):
        indices = [3, 8]
        return indices, {
            "schema_version": 1,
            "stage": smoke.STAGE,
            "phase": "smoke",
            "contract_sha256": digest("contract"),
            "authorization_sha256": digest("authorization"),
            "smoke_plan_sha256": smoke.SMOKE_PLAN_SHA256,
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

    def test_checkpoint_rejects_replay_drift(self):
        indices, checkpoint = self.checkpoint()
        smoke.validate_checkpoint(
            checkpoint,
            indices,
            digest("contract"),
            digest("authorization"),
        )
        checkpoint["replays"]["1"] = copy.deepcopy(
            checkpoint["replays"]["0"]
        )
        checkpoint["replays"]["1"][0]["regions"][0] = 7
        with self.assertRaisesRegex(ValueError, "replay mismatch"):
            smoke.validate_checkpoint(
                checkpoint,
                indices,
                digest("contract"),
                digest("authorization"),
            )


class ArtifactAndIsolationTests(unittest.TestCase):
    def test_atomic_output_is_canonical_and_private(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            smoke.atomic_write_json(path, {"b": 2, "a": 1})
            self.assertEqual(path.read_bytes(), b'{"a":1,"b":2}\n')
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_public_key_guard_rejects_text(self):
        with self.assertRaisesRegex(ValueError, "raw text key"):
            smoke.reject_raw_public_keys({"nested": {"text": "secret"}})

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

    def test_runner_has_only_smoke_phase_and_no_full_entrypoint(self):
        source = Path(smoke.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        function_names = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertNotIn("run_full", function_names)

    def test_parser_rejects_full_phase(self):
        original = sys.argv
        try:
            sys.argv = [
                "runner",
                "--root", "/tmp/root",
                "--output-dir", "/tmp/root/output",
                "--nltk-data", "/tmp/nltk",
                "--embedder-dir", "/tmp/model",
                "--phase", "full",
            ]
            with self.assertRaises(SystemExit):
                smoke.parse_args()
        finally:
            sys.argv = original


if __name__ == "__main__":
    unittest.main()
