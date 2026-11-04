"""No-model tests for corrected SWAN positive-collection provenance."""

from __future__ import annotations

import json
import socket
import unittest
from unittest.mock import patch

import audit_swan_positive_runtime_v2 as audit_v2
import prepare_swan_positive_runtime_v2 as prepare_v2


class OfficialCheckoutTest(unittest.TestCase):
    def test_checkout_is_exact_clean_official_commit(self):
        evidence = prepare_v2.official_checkout_evidence()
        self.assertEqual(evidence["commit"], prepare_v2.EXPECTED_COMMIT)
        self.assertEqual(evidence["tree_oid"], prepare_v2.EXPECTED_TREE)
        self.assertTrue(evidence["clean"])
        self.assertEqual(evidence["tracked_file_count"], 36)
        self.assertEqual(len(evidence["tracked_files"]), 36)

    def test_legacy_mismatch_set_and_prompt_extractor(self):
        evidence = prepare_v2.official_checkout_evidence()
        comparison = prepare_v2.compare_legacy_vendor(evidence)
        self.assertEqual(comparison["matched_file_count"], 31)
        self.assertEqual(comparison["mismatched_file_count"], 5)
        self.assertEqual(
            {row["path"] for row in comparison["mismatches"]},
            prepare_v2.EXPECTED_MISMATCHES,
        )
        self.assertTrue(comparison["prompt_extractor"]["byte_identical"])


class FrozenRuntimeV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = prepare_v2.OUTPUT
        cls.available = all(
            (cls.folder / name).is_file()
            for name in (
                "provenance_correction.json",
                "runtime_v2.json",
                "pre_execution_manifest_v2.json",
                "audit_v2.json",
                "report_v2.md",
            )
        )

    def require_runtime(self):
        if not self.available:
            self.skipTest("Corrected v2 runtime has not been frozen")

    def load(self, name: str):
        return json.loads((self.folder / name).read_text(encoding="utf-8"))

    def test_v1_artifacts_are_preserved_but_execution_superseded(self):
        self.require_runtime()
        correction = self.load("provenance_correction.json")
        supersession = correction["supersession"]
        self.assertEqual(supersession["scope"], "execution_provenance_only")
        self.assertFalse(supersession["runtime_v1_valid_for_execution"])
        self.assertFalse(
            supersession[
                "pre_execution_manifest_v1_valid_for_execution"
            ]
        )
        self.assertTrue(supersession["source_allocation_v1_valid"])
        for name, expected in {
            **supersession["legacy_artifact_sha256"],
            **supersession["source_artifact_sha256"],
        }.items():
            self.assertEqual(
                prepare_v2.sha256(self.folder / name),
                expected,
            )

    def test_runtime_v2_uses_official_tree_and_keeps_source_lineage(self):
        self.require_runtime()
        runtime = self.load("runtime_v2.json")
        source = self.load("source_manifest.json")
        self.assertEqual(
            runtime["official_swan"]["local_root"],
            "third_party/SWAN_official_2ca0c207_clean",
        )
        self.assertEqual(
            runtime["official_swan"]["commit"],
            prepare_v2.EXPECTED_COMMIT,
        )
        self.assertEqual(
            runtime["official_swan"]["tracked_file_count"],
            36,
        )
        self.assertEqual(runtime["source_lineage"]["source_count"], 35)
        self.assertEqual(
            runtime["source_lineage"]["cohorts"],
            {"pilot": 5, "main": 30},
        )
        self.assertEqual(len(source["sources"]), 35)
        self.assertTrue(runtime["source_lineage"]["valid_under_correction"])

    def test_readiness_is_not_overstated(self):
        self.require_runtime()
        runtime = self.load("runtime_v2.json")
        self.assertFalse(runtime["execution_authorized"])
        self.assertFalse(runtime["execution_ready"])
        self.assertFalse(runtime["python_environment"]["all_exact"])
        self.assertGreater(
            runtime["python_environment"]["mismatch_count"],
            0,
        )
        self.assertEqual(
            {
                gap["id"]
                for gap in runtime["protocol_implementation_gaps"]
                if not gap["resolved"]
            },
            {
                "template_rng_unseeded",
                "per_attempt_trace_unavailable",
                "acceptance_state_not_persisted",
                "claude_4_5_top_p_omitted",
            },
        )
        self.assertIsNone(
            runtime["effective_official_behavior"]["template_rng_seed"]
        )
        self.assertIsNone(
            runtime["effective_official_behavior"]["remote_top_p"]
        )

    def test_resource_hashes_include_required_local_assets(self):
        self.require_runtime()
        runtime = self.load("runtime_v2.json")
        dependencies = runtime["dependencies"]
        self.assertEqual(
            len(dependencies["official_swan_tracked_files"]),
            36,
        )
        self.assertEqual(len(dependencies["amr_parser_weights"]), 4)
        self.assertEqual(len(dependencies["amr_tokenizer"]), 5)
        self.assertEqual(len(dependencies["glove"]), 1)
        self.assertEqual(len(dependencies["nltk"]), 5)

    def test_independent_audit_is_offline_and_zero_call(self):
        self.require_runtime()

        saved = self.load("audit_v2.json")
        with patch.object(
            socket,
            "socket",
            side_effect=AssertionError(
                "Network access attempted by v2 audit"
            ),
        ):
            live = audit_v2.build_audit(created_at=saved["created_at"])
        self.assertTrue(live["all_checks_pass"])
        self.assertEqual(saved, live)
        self.assertFalse(live["readiness"]["execution_ready"])
        for key in audit_v2.REMOTE_ZERO_KEYS:
            self.assertEqual(live["resource_counts"][key], 0)

    def test_no_model_or_downstream_artifact_exists(self):
        self.require_runtime()
        forbidden = (
            "model_probe.json",
            "quality_packet.json",
            "quality_labels.json",
            "original_detection.json",
            "calls",
            "generation_traces",
            "baselines",
        )
        self.assertFalse(
            any((self.folder / relative).exists() for relative in forbidden)
        )


if __name__ == "__main__":
    unittest.main()
