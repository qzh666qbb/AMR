"""Offline tests for the WP2-v2 score-blind design audit."""

import itertools
import unittest

from audit_certified_resegmentation_wp2_v2_design import (
    INSTRUCTION,
    audit_identity,
    candidate_codes,
    official_tokenizer_text,
    p0_codes,
    threat_size_upper,
    unit_layout,
)
from certified_resegmentation import boundary_moves
from certified_resegmentation_wp2_adapter import make_source_record


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return {"input_ids": list(range(len(text.split()) + 2))}


class P0ContractTests(unittest.TestCase):
    def test_edge_whitespace_fails_strict_but_passes_lossless(self):
        units = (" alpha beta ", " gamma delta ")
        layouts = unit_layout(units)
        base, strict, lossless, missing = p0_codes(
            "list", units, layouts
        )
        self.assertEqual(base, [])
        self.assertEqual(strict, ["unit_edge_whitespace"])
        self.assertEqual(lossless, [])
        self.assertEqual(missing, 0)

    def test_lossless_rejects_atom_gluing_across_list_boundary(self):
        units = ("alpha beta", "gamma delta")
        base, strict, lossless, missing = p0_codes(
            "list", units, unit_layout(units)
        )
        self.assertEqual(base, [])
        self.assertEqual(strict, [])
        self.assertEqual(lossless, ["missing_cross_unit_whitespace"])
        self.assertEqual(missing, 1)

    def test_native_candidate_is_independent_of_legacy_atom_gate(self):
        codes = candidate_codes(
            base_codes=[],
            strict_codes=[],
            lossless_codes=[],
            unit_count=2,
            atom_counts=[1, 40],
            native_token_counts=[3, 42],
        )
        self.assertFalse(codes["L_STRICT_NATIVE_512"])
        self.assertFalse(codes["V2_LOSSLESS_NATIVE_512"])
        self.assertIn("unit_below_5_atoms", codes["V1_STRICT_5_35"])
        self.assertIn("unit_above_35_atoms", codes["V1_STRICT_5_35"])

    def test_simmark_prompt_and_fix_space_are_exact(self):
        self.assertEqual(
            official_tokenizer_text("simmark", "first", 0),
            INSTRUCTION + "first",
        )
        self.assertEqual(
            official_tokenizer_text("simmark", "second", 1),
            INSTRUCTION + " second",
        )
        self.assertEqual(
            official_tokenizer_text("semstamp", " sentence ", 1),
            "sentence",
        )


class ComplexityTests(unittest.TestCase):
    def test_closed_form_matches_small_partition_enumeration(self):
        atom_count = 8
        original = (2, 5, 7)
        unit_count = len(original) + 1
        all_boundaries = range(1, atom_count)
        for radius in range(4):
            observed = sum(
                boundary_moves(candidate, original) <= radius
                for candidate in itertools.combinations(
                    all_boundaries, len(original)
                )
            )
            self.assertEqual(
                observed,
                threat_size_upper(atom_count, unit_count, radius),
            )


class RowAuditTests(unittest.TestCase):
    def test_row_contains_hashes_and_no_raw_text(self):
        value = [" alpha beta ", " gamma delta "]
        record = make_source_record(
            "semstamp", "revision", 7, value
        )
        tokenizer = FakeTokenizer()
        row = audit_identity(
            {
                "cohort": "positive",
                "role": "positive",
                "record": record,
                "raw_value": value,
            },
            "semstamp",
            tokenizer,
        )
        self.assertTrue(
            row["candidate_eligible"]["V2_LOSSLESS_NATIVE_512"]
        )
        self.assertFalse(row["candidate_eligible"]["V1_STRICT_5_35"])
        self.assertFalse(row["contains_raw_text"])
        self.assertNotIn("raw_value", row)
        self.assertNotIn("original_units", row)
        self.assertEqual(len(tokenizer.calls), 2)
        for _, kwargs in tokenizer.calls:
            self.assertFalse(kwargs["truncation"])


if __name__ == "__main__":
    unittest.main()
