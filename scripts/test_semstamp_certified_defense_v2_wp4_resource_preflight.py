"""Model-free tests for the WP4 resource-preflight design."""

import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_semstamp_certified_defense_v2_wp4_resource_preflight as wp4


class AuthorizationTests(unittest.TestCase):
    def make_paths(self, directory):
        output = Path(directory)
        paths = {
            "contract": output / "CONTRACT.json",
            "manifest": output / "INPUT_MANIFEST.sha256",
            "runner": Path(wp4.__file__).resolve(),
            "authorization": output / "EXECUTION_AUTHORIZATION.json",
            "resume_authorization": output / "RESUME_AUTHORIZATION.json",
            "rows": output / "PREFLIGHT_ROWS.jsonl",
        }
        paths["contract"].write_bytes(b'{"contract":true}\n')
        paths["manifest"].write_bytes(b"0" * 64 + b"  input\n")
        contract = {
            "execution_authorized": False,
            "selected_hardware": {
                "device": wp4.SELECTED_DEVICE,
                "driver_version": wp4.SELECTED_DRIVER,
                "memory_mib": wp4.SELECTED_MEMORY_MIB,
                "cuda_visible_devices": "0",
            },
            "resource_budget": {
                "span_gate": wp4.SPAN_GATE,
                "pair_gate": wp4.PAIR_GATE,
                "legacy_pair_diagnostic": wp4.LEGACY_PAIR_DIAGNOSTIC,
                "sidecar_gate_bytes": wp4.SIDECAR_GATE_BYTES,
                "work_disk_gate_bytes": wp4.WORK_DISK_GATE_BYTES,
                "free_disk_reserve_bytes": wp4.FREE_DISK_RESERVE_BYTES,
                "gpu_hour_gate": wp4.GPU_HOUR_GATE,
            },
        }
        return paths, contract

    def test_absent_authorization_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, contract = self.make_paths(directory)
            with self.assertRaisesRegex(PermissionError, "not authorized"):
                wp4.validate_authorization(paths, contract)

    def test_exact_authorization_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, contract = self.make_paths(directory)
            expected = wp4.expected_authorization(paths, contract)
            paths["authorization"].write_text(
                json.dumps(expected), encoding="ascii"
            )
            self.assertEqual(
                wp4.validate_authorization(paths, contract), expected
            )

    def test_authorization_cannot_enable_certificate(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, contract = self.make_paths(directory)
            value = wp4.expected_authorization(paths, contract)
            value["certificate_authorized"] = True
            paths["authorization"].write_text(
                json.dumps(value), encoding="ascii"
            )
            with self.assertRaisesRegex(PermissionError, "invalid"):
                wp4.validate_authorization(paths, contract)

    def test_contract_cannot_self_authorize(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, contract = self.make_paths(directory)
            contract["execution_authorized"] = True
            paths["authorization"].write_text(
                json.dumps(wp4.expected_authorization(paths, contract)),
                encoding="ascii",
            )
            with self.assertRaisesRegex(PermissionError, "does not prohibit"):
                wp4.validate_authorization(paths, contract)


class ResumeTests(unittest.TestCase):
    def test_initial_run_rejects_resume_file(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, _ = AuthorizationTests().make_paths(directory)
            paths["resume_authorization"].write_text("{}", encoding="ascii")
            with self.assertRaisesRegex(PermissionError, "forbidden initially"):
                wp4.validate_resume_authorization(paths, [])

    def test_partial_run_requires_resume_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, _ = AuthorizationTests().make_paths(directory)
            paths["rows"].write_bytes(b'{"plan_index":0}\n')
            with self.assertRaisesRegex(PermissionError, "requires"):
                wp4.validate_resume_authorization(
                    paths, [{"plan_index": 0, "attempt_index": 0}]
                )

    def test_exact_partial_resume_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, _ = AuthorizationTests().make_paths(directory)
            paths["authorization"].write_bytes(b'{"authorized":true}\n')
            paths["rows"].write_bytes(b'{"plan_index":0}\n')
            expected = {
                "schema_version": 1,
                "stage": wp4.STAGE,
                "status": "WP4_RESOURCE_PREFLIGHT_RESUME_AUTHORIZED",
                "scope": "exact_completed_prefix_only",
                "execution_authorization_sha256": wp4.sha256_file(
                    paths["authorization"]
                ),
                "completed_rows_sha256": wp4.sha256_file(paths["rows"]),
                "completed_prefix_length": 1,
                "next_attempt_index": 1,
                "no_score_based_retry": True,
                "sbert_load_authorized": False,
                "detector_score_authorized": False,
                "certificate_authorized": False,
                "wp5_authorized": False,
            }
            paths["resume_authorization"].write_text(
                json.dumps(expected), encoding="ascii"
            )
            self.assertEqual(
                wp4.validate_resume_authorization(
                    paths, [{"plan_index": 0, "attempt_index": 0}]
                ),
                expected,
            )


class ProjectionTests(unittest.TestCase):
    def profiles(self):
        return [
            {
                "atom_count": 10,
                "selection_slot": "low",
                "feature_seconds_per_input_token": 2.0,
                "sidecar_seconds_per_span": 3.0,
                "core_seconds_per_pair": 4.0,
            },
            {
                "atom_count": 20,
                "selection_slot": "high",
                "feature_seconds_per_input_token": 5.0,
                "sidecar_seconds_per_span": 7.0,
                "core_seconds_per_pair": 11.0,
            },
        ]

    def test_endpoint_profile_selection_is_conservative(self):
        rate = wp4.conservative_rate(
            self.profiles(), 100, "core_seconds_per_pair"
        )
        self.assertEqual(rate["selected_slot"], "high")
        self.assertEqual(rate["rate"], 11.0)

    def test_projection_counts_distinct_effective_radii_once(self):
        radii = {
            str(radius): {
                "effective_radius": 0,
                "unique_span_count": 3,
                "unique_pair_count": 5,
            }
            for radius in wp4.RADII
        }
        projection = wp4.project_document(
            10,
            radii,
            {"maximum_legal_input_tokens": 2},
            self.profiles(),
        )
        self.assertEqual(projection["feature_seconds"], 10.0)
        self.assertEqual(projection["sidecar_seconds"], 21.0)
        self.assertEqual(projection["core_seconds"], 55.0)

    def test_core_cap_failure_has_dedicated_terminal(self):
        gates = {
            "max_pruned_spans_le_2000000": True,
            "max_pruned_pairs_le_50000000": False,
            "max_all_allowed_spans_le_2000000": True,
            "max_all_allowed_pairs_le_50000000": True,
            "budget": True,
        }
        self.assertEqual(
            wp4.classify_terminal(gates),
            "WP4_RESOURCE_PREFLIGHT_READINESS_STOP_CORE_CAP",
        )

    def test_budget_failure_does_not_raise_core_cap(self):
        gates = {
            "max_pruned_spans_le_2000000": True,
            "max_pruned_pairs_le_50000000": True,
            "max_all_allowed_spans_le_2000000": True,
            "max_all_allowed_pairs_le_50000000": True,
            "budget": False,
        }
        self.assertEqual(
            wp4.classify_terminal(gates),
            "WP4_RESOURCE_PREFLIGHT_READINESS_STOP_BUDGET",
        )

    def test_all_pass_has_wp5_design_terminal(self):
        gates = {
            "max_pruned_spans_le_2000000": True,
            "max_pruned_pairs_le_50000000": True,
            "max_all_allowed_spans_le_2000000": True,
            "max_all_allowed_pairs_le_50000000": True,
            "budget": True,
        }
        self.assertEqual(
            wp4.classify_terminal(gates),
            "WP4_RESOURCE_PREFLIGHT_GO_FOR_WP5_DESIGN",
        )


class StaticSafetyTests(unittest.TestCase):
    def test_protocol_hash_is_exact(self):
        protocol = (
            Path(wp4.__file__).resolve().parents[1]
            / "experiments/"
            "semstamp_certified_defense_v2_wp4_resource_preflight_design1/"
            "PROTOCOL.md"
        )
        self.assertEqual(wp4.sha256_file(protocol), wp4.PROTOCOL_SHA256)

    def test_no_sbert_weight_is_a_tokenizer_input(self):
        self.assertNotIn("pytorch_model.bin", wp4.TOKENIZER_SHA256)
        self.assertEqual(len(wp4.TOKENIZER_SHA256), 5)

    def test_forbidden_imports_are_not_top_level(self):
        tree = ast.parse(Path(wp4.__file__).read_text(encoding="utf-8"))
        names = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        self.assertTrue(
            names.isdisjoint(
                {
                    "numpy",
                    "torch",
                    "transformers",
                    "sentence_transformers",
                    "semstamp_refined_atom_renderer",
                }
            )
        )

    def test_run_validates_authorization_before_deferred_imports(self):
        source = Path(wp4.__file__).read_text(encoding="utf-8")
        authorization = source.index(
            "validate_authorization(paths, contract)", source.index("def run(")
        )
        deferred = source.index(
            "import run_semstamp_certified_defense_v2_wp3_p0",
            source.index("def run("),
        )
        self.assertLess(authorization, deferred)

    def test_public_raw_text_key_rejected(self):
        with self.assertRaisesRegex(ValueError, "raw text key"):
            wp4.reject_raw_public_keys({"nested": {"text": "secret"}})

    def test_hardware_inventory_mismatch_fails(self):
        completed = mock.Mock(stdout="Other GPU, 595.80, 24576\n")
        with mock.patch("subprocess.run", return_value=completed):
            with mock.patch.dict(
                "os.environ", {"CUDA_VISIBLE_DEVICES": "0"}, clear=True
            ):
                with self.assertRaisesRegex(RuntimeError, "mismatch"):
                    wp4.validate_hardware_inventory()


if __name__ == "__main__":
    unittest.main()
