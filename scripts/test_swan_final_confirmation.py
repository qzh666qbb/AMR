"""Tests for the frozen final SWAN segmentation-audit primitives."""

import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import swan_final_structural as structural
import prepare_swan_final_confirmation as preparation


TEXT = (
    "Alpha analysts reviewed the quarterly revenue report carefully. "
    "Beta managers approved the revised operating plan yesterday. "
    "Gamma engineers tested the new server under load. "
    "Delta researchers documented every unexpected system response. "
    "Epsilon auditors compared the records with prior filings. "
    "Zeta directors discussed the remaining financial risks privately. "
    "Eta lawyers revised the final disclosure before publication. "
    "Theta editors released the completed article on Monday."
)


class StructuralConditionsTest(unittest.TestCase):
    def test_all_conditions_preserve_lexical_tokens(self):
        expected = structural.lexical_tokens(TEXT)
        for condition in (
            "C0", "C1", "C2", "C3", "C4a", "C4b", "C4c", "C4d", "C5",
        ):
            with self.subTest(condition=condition):
                result = structural.apply_condition(TEXT, condition, 7)
                self.assertEqual(
                    structural.lexical_tokens(result["text"]),
                    expected,
                )
                self.assertTrue(result["lexical_tokens_preserved"])

    def test_boundary_exchange_preserves_unit_count(self):
        result = structural.apply_condition(TEXT, "C2", 7)
        self.assertEqual(result["source_units"], 8)
        self.assertEqual(result["surface_units"], 8)
        self.assertEqual(result["detector_units"], 8)

    def test_strict_merge_dose_is_nested_and_monotone(self):
        outputs = {
            name: structural.apply_condition(TEXT, name, 7)
            for name in ("C4a", "C4b", "C4c", "C4d")
        }
        counts = [outputs[name]["surface_units"] for name in outputs]
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertEqual(outputs["C4d"]["surface_units"], 1)

        semicolons = {
            name: {
                index
                for index, character in enumerate(outputs[name]["text"])
                if character == ";"
            }
            for name in outputs
        }
        self.assertTrue(semicolons["C4a"] <= semicolons["C4b"])
        self.assertTrue(semicolons["C4b"] <= semicolons["C4c"])
        self.assertTrue(semicolons["C4c"] <= semicolons["C4d"])

    def test_exact_restoration_reconstructs_surface(self):
        result = structural.apply_condition(TEXT, "C5", 7)
        self.assertEqual(result["surface_units"], 1)
        self.assertEqual(result["detector_units"], 8)
        self.assertEqual(" ".join(result["supplied_spans"]), result["text"])

    def test_invalid_condition_and_fraction_fail(self):
        with self.assertRaises(ValueError):
            structural.apply_condition(TEXT, "unknown", 7)
        with self.assertRaises(ValueError):
            structural.strict_semicolon_merge(TEXT, math.nan)


class PreregistrationStatisticsTest(unittest.TestCase):
    def test_exact_mcnemar_power_is_frozen(self):
        power = preparation.unconditional_mcnemar_power(
            sample_size=104,
            favorable_probability=0.25,
            reverse_probability=0.05,
            alpha=0.025,
        )
        self.assertAlmostEqual(power, 0.9678883268813759, places=14)

    def test_clopper_pearson_maximum_width(self):
        result = preparation.maximum_clopper_pearson_width(104)
        self.assertEqual(result["successes_at_maximum"], 52)
        self.assertLess(result["maximum_width"], 0.20)
        self.assertAlmostEqual(
            result["maximum_width"],
            0.1993221491987659,
            places=14,
        )


if __name__ == "__main__":
    unittest.main()
