"""Tests for the post-hoc atomic-fact length diagnostic."""

import unittest

from analyze_swan_atomic_fact_preflight import describe, pearson


class AtomicFactLengthDiagnosticTests(unittest.TestCase):
    def test_pearson_detects_linear_relation(self):
        self.assertAlmostEqual(pearson([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertIsNone(pearson([1], [2]))

    def test_describe_uses_all_rows(self):
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
        self.assertEqual(result["template_triples_mean"], 20)


if __name__ == "__main__":
    unittest.main()
