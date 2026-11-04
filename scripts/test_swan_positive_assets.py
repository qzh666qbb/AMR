"""No-network tests for the SWAN positive-asset audit."""

import json
from pathlib import Path
import socket
import unittest
from unittest.mock import patch

import audit_swan_positive_assets as audit


class PositiveAssetAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = (
            Path(__file__).resolve().parents[1]
            / "experiments/swan_positive_asset_audit1"
        )
        cls.manifest = json.loads(
            (cls.output / "manifest.json").read_text()
        )

    def test_public_snapshot_has_no_generated_watermarked_text(self):
        result = audit.verify_official_snapshot(self.manifest)
        self.assertEqual(result["main_commit"], (
            "2ca0c2071c8d0c6bb3739b2b2bcfb34cfe63eb2a"
        ))
        self.assertEqual(result["tracked_files"], 36)
        self.assertEqual(result["generated_watermarked_text_files"], 0)
        self.assertEqual(result["releases"], 0)
        self.assertEqual(result["tags"], 0)

    def test_local_inventory_reconstructs_zero_joint_assets(self):
        inventory = audit.build_inventory(self.manifest)
        self.assertEqual(inventory, self.manifest["expected_inventory"])
        self.assertTrue(all(
            row["joint_qualified_documents"] == 0
            for row in inventory.values()
        ))
        self.assertEqual(
            inventory["confirmation_three_arm30"][
                "normal_form_documents"
            ],
            0,
        )
        self.assertEqual(
            inventory["swan_quality_first_windows4"][
                "saved_automated_quality_pass_documents"
            ],
            14,
        )
        self.assertEqual(
            inventory["swan_quality_first_windows4"][
                "traceable_injected_documents"
            ],
            0,
        )

    def test_collection_is_separate_and_execution_locked(self):
        checks = audit.validate_collection_contract(self.manifest)
        self.assertTrue(all(checks.values()))
        collection = self.manifest["new_collection"]
        self.assertFalse(collection["execution_authorized"])
        self.assertEqual(collection["allocation"]["pilot"], 5)
        self.assertEqual(collection["allocation"]["main"], 30)
        self.assertFalse(collection["allocation"]["pilot_in_main"])
        self.assertFalse(
            collection["qualification"][
                "original_detector_is_inclusion_gate"
            ]
        )

    def test_summary_build_uses_no_network(self):
        def forbidden(*_args, **_kwargs):
            raise AssertionError("Network access attempted")

        with patch.object(socket, "socket", forbidden):
            rebuilt = audit.build_summary(self.manifest)
        saved = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(rebuilt, saved)
        self.assertEqual(
            rebuilt["decision"],
            "no_existing_asset_meets_joint_gate",
        )
        self.assertEqual(
            rebuilt["joint_qualifying_existing_documents"],
            0,
        )
        self.assertTrue(all(
            value == 0 for value in rebuilt["resource_counts"].values()
        ))

    def test_report_is_deterministic(self):
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(
            audit.render_report(summary),
            (self.output / "report.md").read_text(),
        )


if __name__ == "__main__":
    unittest.main()
