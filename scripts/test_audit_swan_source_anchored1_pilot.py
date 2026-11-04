"""Tests for the source-anchored no-compute closure audit."""
import json
from pathlib import Path
import unittest

import audit_swan_source_anchored1_pilot as audit


class SourceAnchoredClosureTests(unittest.TestCase):
    def test_real_pilot_rebuilds_expected_branch_c_closure(self):
        outputs = audit.build()
        payload = json.loads(outputs["audit.json"])
        attrition = json.loads(outputs["attrition.json"])

        self.assertFalse(payload["pilot_passed"])
        self.assertEqual(payload["gate"]["branch"], "C")
        self.assertEqual(attrition["counts"]["structural_pass"], 5)
        self.assertEqual(attrition["counts"]["generation_attempts"], 1890)
        self.assertEqual(attrition["counts"]["responses"], 1891)
        self.assertEqual(attrition["counts"]["fallback_slots"], 60)
        self.assertEqual(attrition["counts"]["accepted_slots"], 0)
        self.assertFalse(payload["checks"]["parser_warning_stop_resolved"])

    def test_auditor_has_no_prohibited_runtime_or_input_reference(self):
        source = Path(audit.__file__).read_text(encoding="utf-8")
        prohibited = (
            "official_runtime",
            "parse_amr(",
            "decide_s2match",
            "transport(",
            "main_sources.json",
        )
        for value in prohibited:
            self.assertNotIn(value, source)


if __name__ == "__main__":
    unittest.main()
