"""Tests for the versioned Branch C reframing audit."""
from pathlib import Path
import tempfile
import unittest

import audit_swan_reframing_v2 as audit


class ReframingAuditV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = audit.build()

    def test_real_repository_reconciles_only_the_superseded_report(self):
        self.assertTrue(self.payload["ok"])
        self.assertFalse(self.payload["legacy_audit"]["ok"])
        self.assertEqual(
            self.payload["legacy_audit"][
                "expected_unexpected_historical_changes"
            ],
            [audit.REPORT],
        )
        self.assertTrue(
            self.payload["checks"]["removing_notice_reconstructs_p0_bytes"]
        )
        self.assertEqual(
            self.payload["supersession"]["p0_sha256"],
            audit.P0_REPORT_SHA256,
        )

    def test_reuses_preservation_and_endpoint_results(self):
        legacy = self.payload["legacy_audit"]
        self.assertEqual(
            legacy["historical_preservation"]["preserved_originals_verified"],
            82,
        )
        self.assertEqual(len(legacy["endpoint_gates"]), 16)
        self.assertEqual(legacy["bad_endpoints"], [])
        self.assertEqual(
            legacy["new_model_parser_s2match_detector_observations"], 0
        )

    def test_auditor_has_no_prohibited_runtime_or_sealed_source_reference(self):
        source = Path(audit.__file__).read_text(encoding="utf-8")
        for value in (
            "official_runtime",
            "parse_amr(",
            "decide_s2match",
            "transport(",
            "main_sources.json",
            "api.txt",
        ):
            self.assertNotIn(value, source)

    def test_frozen_bytes_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            expected = audit.json_bytes({"ok": True})
            path.write_bytes(expected)
            audit.frozen_bytes(path, expected)
            path.write_bytes(expected + b" ")
            with self.assertRaises(ValueError):
                audit.frozen_bytes(path, expected)


if __name__ == "__main__":
    unittest.main()
