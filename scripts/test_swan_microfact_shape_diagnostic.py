"""Tests for the post-hoc microfact graph-shape diagnostic."""

import json
import unittest

from diagnose_swan_microfact_shapes import describe, recover_microfacts


class MicrofactShapeDiagnosticTests(unittest.TestCase):
    def test_recover_accepts_valid_json_without_change(self):
        facts = [f"Fact {index}." for index in range(15)]
        recovered, repair = recover_microfacts(json.dumps({"facts": facts}))
        self.assertEqual(recovered, facts)
        self.assertEqual(repair, "none")

    def test_recover_only_inserts_missing_array_close(self):
        facts = [f"Fact {index}." for index in range(15)]
        broken = json.dumps({"facts": facts}, separators=(",", ":"))
        broken = broken[:-2] + "}"
        recovered, repair = recover_microfacts(broken)
        self.assertEqual(recovered, facts)
        self.assertEqual(repair, "insert_array_close_before_final_object_close")
        with self.assertRaises(ValueError):
            recover_microfacts("not json")

    def test_describe_uses_fixed_scale_labels(self):
        rows = [
            {"scale_compatible": True, "template_triples": 10,
             "compatible_bank_templates": 20},
            {"scale_compatible": False, "template_triples": 30,
             "compatible_bank_templates": 0},
        ]
        result = describe(rows)
        self.assertEqual(result["facts"], 2)
        self.assertEqual(result["scale_compatible"], 1)
        self.assertEqual(result["scale_compatible_fraction"], 0.5)
        self.assertEqual(result["template_triples_median"], 20)


if __name__ == "__main__":
    unittest.main()
