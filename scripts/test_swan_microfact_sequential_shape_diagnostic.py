"""Tests for the sequential microfact graph-scale diagnostic."""

import unittest

from diagnose_swan_microfact_sequential_shapes import (
    describe,
    metric_line,
)


class SequentialMicrofactShapeDiagnosticTests(unittest.TestCase):
    def test_describe_uses_fixed_scale_labels(self):
        rows = [
            {
                "scale_compatible": True,
                "template_triples": 10,
                "compatible_bank_templates": 20,
            },
            {
                "scale_compatible": False,
                "template_triples": 30,
                "compatible_bank_templates": 0,
            },
        ]
        result = describe(rows)
        self.assertEqual(result["facts"], 2)
        self.assertEqual(result["scale_compatible"], 1)
        self.assertEqual(result["count_feasible"], 1)
        self.assertEqual(result["scale_compatible_fraction"], 0.5)
        self.assertEqual(result["template_triples_median"], 20)

    def test_empty_metric_line_uses_na(self):
        line = metric_line("Empty", describe([]))
        self.assertIn("0/0", line)
        self.assertIn("n/a", line)


if __name__ == "__main__":
    unittest.main()
