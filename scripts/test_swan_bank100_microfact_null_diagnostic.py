"""Tests for the post-hoc matched-form generation diagnostic."""

import json
import unittest

from diagnose_swan_bank100_microfact_null import (
    minimally_repair_missing_array_close,
)


class Bank100MicrofactNullDiagnosticTests(unittest.TestCase):
    def facts(self):
        return [
            f"Subject{i} records one verified source detail clearly today."
            for i in range(15)
        ]

    def test_repairs_only_missing_array_close(self):
        content = json.dumps(
            {"facts": self.facts()},
            separators=(",", ":"),
        )
        broken = content[:-2] + "}"
        facts, repaired = minimally_repair_missing_array_close(broken)
        self.assertTrue(repaired)
        self.assertEqual(facts, self.facts())

    def test_does_not_repair_other_schema_errors(self):
        facts, repaired = minimally_repair_missing_array_close(
            json.dumps({"items": self.facts()})
        )
        self.assertIsNone(facts)
        self.assertFalse(repaired)

    def test_preserves_strict_json(self):
        content = json.dumps({"facts": self.facts()})
        facts, repaired = minimally_repair_missing_array_close(content)
        self.assertFalse(repaired)
        self.assertEqual(facts, self.facts())


if __name__ == "__main__":
    unittest.main()
