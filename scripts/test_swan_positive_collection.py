"""No-model tests for SWAN positive-collection preparation and audit."""

from __future__ import annotations

import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import audit_swan_positive_collection as audit
import prepare_swan_positive_collection as prepare


class PriorIdentitySnapshotTest(unittest.TestCase):
    def test_discovers_all_identity_artifacts_including_terminated_run(self):
        with tempfile.TemporaryDirectory(dir=prepare.ROOT) as directory:
            root = Path(directory)
            experiments = root / "experiments"
            terminated = (
                experiments
                / "swan_final_confirmation1"
                / "source_manifest.json"
            )
            excluded = (
                experiments
                / "swan_positive_collection1"
                / "source_manifest.json"
            )
            irrelevant = experiments / "other" / "summary.json"
            for path in (terminated, excluded, irrelevant):
                path.parent.mkdir(parents=True, exist_ok=True)
            terminated.write_text(
                json.dumps(
                    {
                        "source_text": "A prior C4 article.",
                        "url": "https://example.test/prior",
                    }
                ),
                encoding="utf-8",
            )
            excluded.write_text(
                json.dumps({"text": "Must not self-enter the snapshot."}),
                encoding="utf-8",
            )
            irrelevant.write_text(
                json.dumps({"metric": 0.5}),
                encoding="utf-8",
            )

            paths, scanned = prepare.discover_history_artifacts(
                root=root,
                excluded_tree=excluded.parent,
            )
            snapshot = prepare.build_prior_identity_snapshot(
                paths,
                root=root,
                all_files_scanned=scanned,
                created_at="2026-09-16T00:00:00+00:00",
            )

        self.assertEqual(scanned, 2)
        self.assertEqual(
            [row["path"] for row in snapshot["artifact_inventory"]],
            [
                (
                    "experiments/swan_final_confirmation1/"
                    "source_manifest.json"
                )
            ],
        )
        self.assertIn(
            prepare.normalized_hash("A prior C4 article."),
            snapshot["normalized_text_sha256"],
        )
        self.assertIn(
            prepare.normalized_url_hash("https://example.test/prior"),
            snapshot["url_sha256"],
        )

    def test_collects_declared_normalized_hash_and_text_values(self):
        text_hashes = set()
        url_hashes = set()
        counts = {}
        declared = "a" * 64
        prepare._collect_identity_values(
            {
                "source": "Prior source.",
                "normalized_source_sha256": declared,
                "generation_prompt": "Prior prompt.",
                "url": "https://example.test/source",
                "unrelated": "not an identity",
            },
            text_hashes,
            url_hashes,
            counts,
        )
        self.assertIn(declared, text_hashes)
        self.assertIn(
            prepare.normalized_hash("Prior source."),
            text_hashes,
        )
        self.assertIn(
            prepare.normalized_hash("Prior prompt."),
            text_hashes,
        )
        self.assertEqual(len(url_hashes), 1)


class AllocationReplayTest(unittest.TestCase):
    def test_scan_preserves_order_and_freezes_pilot_before_main(self):
        rows = [
            {
                "text": "This is a previously allocated source article.",
                "url": "https://example.test/prior",
            },
            {"text": "   ", "url": ""},
            {
                "text": "Fresh article alpha has enough prompt words here.",
                "url": "https://example.test/alpha",
            },
            {
                "text": "Different text but a repeated source URL appears.",
                "url": "https://example.test/alpha",
            },
            {
                "text": "Fresh article beta has enough prompt words here.",
                "url": "https://example.test/beta",
            },
        ]
        snapshot = {
            "normalized_text_sha256": [
                prepare.normalized_hash(rows[0]["text"])
            ],
            "url_sha256": [
                prepare.normalized_url_hash(rows[0]["url"])
            ],
        }

        def stream(index):
            self.assertEqual(index, prepare.SOURCE_START_SHARD)
            yield from rows

        with patch.object(prepare, "stream_train_shard", side_effect=stream):
            sources, screened, summary = prepare.scan_sources(
                snapshot,
                source_count=2,
                pilot_count=1,
            )

        self.assertEqual(
            [row["shard_row"] for row in sources],
            [2, 4],
        )
        self.assertEqual(
            [row["cohort"] for row in sources],
            ["pilot", "main"],
        )
        self.assertEqual(len(screened), 5)
        self.assertEqual(summary["selected"], 2)
        self.assertEqual(summary["excluded"], 3)
        self.assertEqual(
            summary["exclusion_condition_counts"],
            {
                "empty_source": 1,
                "empty_prompt": 1,
                "normalized_text_collision": 1,
                "url_collision": 2,
            },
        )

    def test_independent_screen_rebuild_matches_preparer(self):
        row = {
            "text": "A fresh source provides a stable prompt for generation.",
            "url": "https://example.test/fresh",
        }
        prepared = prepare._screening_record(row, 8, 4, set(), set())
        rebuilt = audit.rebuild_screening(prepared, set(), set())
        self.assertEqual(prepared, rebuilt)
        self.assertEqual(prepared["decision"], "selected")


class FrozenCollectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = prepare.OUTPUT
        cls.available = (
            (cls.folder / "pre_execution_manifest.json").is_file()
            and (cls.folder / "source_manifest.json").is_file()
        )

    def require_collection(self):
        if not self.available:
            self.skipTest("Frozen source allocation has not been acquired")

    def test_actual_allocation_is_five_pilot_plus_thirty_main(self):
        self.require_collection()
        manifest = prepare.check_pre_execution(self.folder)
        sources = json.loads(
            (self.folder / "source_manifest.json").read_text()
        )
        self.assertFalse(manifest["execution_authorized"])
        self.assertEqual(sources["cohort_counts"], {"pilot": 5, "main": 30})
        self.assertEqual(
            [row["cohort"] for row in sources["sources"][:5]],
            ["pilot"] * 5,
        )
        self.assertEqual(
            [row["cohort"] for row in sources["sources"][5:]],
            ["main"] * 30,
        )

    def test_snapshot_inventory_contains_terminated_source_manifest(self):
        self.require_collection()
        snapshot = json.loads(
            (self.folder / "prior_identity_snapshot.json").read_text()
        )
        inventory = {row["path"] for row in snapshot["artifact_inventory"]}
        self.assertIn(
            "experiments/swan_final_confirmation1/source_manifest.json",
            inventory,
        )

    def test_runtime_freezes_full_upstream_tree_and_remote_lock(self):
        self.require_collection()
        runtime = json.loads((self.folder / "runtime.json").read_text())
        upstream = runtime["dependencies"]["upstream_swan_commit_files"]
        self.assertEqual(len(upstream), 36)
        self.assertEqual(
            runtime["upstream_swan"]["commit"],
            prepare.SWAN_COMMIT,
        )
        self.assertIsNone(runtime["remote_model"]["returned"])
        self.assertFalse(runtime["remote_model"]["model_probe_authorized"])
        self.assertFalse(runtime["remote_model"]["generation_authorized"])

    def test_independent_audit_is_offline_and_reports_zero_remote_calls(self):
        self.require_collection()

        def forbidden(*_args, **_kwargs):
            raise AssertionError("Network access attempted by offline audit")

        with patch.object(socket, "socket", forbidden):
            result = audit.build_audit(self.folder)
        self.assertTrue(result["all_checks_pass"])
        self.assertEqual(result["allocation"]["pilot"], 5)
        self.assertEqual(result["allocation"]["main"], 30)
        for key in (
            "paid_api_calls",
            "model_probe_calls",
            "generation_calls",
            "parser_calls",
            "s2match_calls",
            "detector_calls",
            "attack_calls",
        ):
            self.assertEqual(result["resource_counts"][key], 0)

    def test_no_model_or_downstream_artifact_exists(self):
        self.require_collection()
        forbidden = [
            "model_probe.json",
            "quality_packet.json",
            "quality_labels.json",
            "original_detection.json",
            "calls",
            "generation_traces",
            "baselines",
        ]
        self.assertFalse(
            any((self.folder / relative).exists() for relative in forbidden)
        )


if __name__ == "__main__":
    unittest.main()
