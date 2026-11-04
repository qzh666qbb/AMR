"""Unit tests for the refined-atom raw-byte renderer."""

import hashlib
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))

from semstamp_refined_atom_renderer import (
    RefinedAtomRendererError,
    canonicalize_refined_bytes,
)


def splitter_for(*units):
    return lambda _text: list(units)


class RefinedAtomRendererTests(unittest.TestCase):
    def test_refines_no_whitespace_nltk_boundary(self):
        raw = b"Alpha ends.Next begins."
        document = canonicalize_refined_bytes(
            raw,
            splitter_for("Alpha ends.", "Next begins."),
        )
        self.assertEqual(document.lexical_span_count, 3)
        self.assertEqual(document.atom_count, 4)
        self.assertEqual(document.refinement_count, 1)
        self.assertEqual(document.original_boundaries, (2,))
        self.assertEqual(document.zero_byte_gap_boundaries, (2,))
        self.assertEqual(document.gap_after(2), b"")
        self.assertEqual(
            document.render_partition(document.original_boundaries),
            ("Alpha ends.", "Next begins."),
        )
        self.assertEqual(document.reconstruct(document.original_boundaries), raw)

    def test_deleting_zero_byte_b0_boundary_preserves_raw_bytes(self):
        raw = b"Alpha ends.Next begins."
        document = canonicalize_refined_bytes(
            raw,
            splitter_for("Alpha ends.", "Next begins."),
        )
        self.assertEqual(
            document.render_partition(()),
            ("Alpha ends.Next begins.",),
        )
        self.assertEqual(document.reconstruct(()), raw)

    def test_preserves_whitespace_unicode_and_outer_gaps(self):
        text = " \tAlpha  beta.\r\n\r\nGamma\u00a0delta! \n"
        raw = text.encode("utf-8")
        document = canonicalize_refined_bytes(
            raw,
            splitter_for("Alpha  beta.", "Gamma\u00a0delta!"),
        )
        self.assertEqual(document.refinement_count, 0)
        self.assertEqual(document.zero_byte_gap_boundaries, ())
        self.assertEqual(document.prefix_gap, b" \t")
        self.assertEqual(document.suffix_gap, b" \n")
        self.assertEqual(document.reconstruct(document.original_boundaries), raw)
        self.assertEqual(
            document.public_evidence()["byte_reconstruction_sha256"],
            hashlib.sha256(raw).hexdigest(),
        )

    def test_multiple_internal_boundaries_refine_one_lexical_span(self):
        raw = b"A.B.C."
        document = canonicalize_refined_bytes(
            raw,
            splitter_for("A.", "B.", "C."),
        )
        self.assertEqual(document.lexical_span_count, 1)
        self.assertEqual(document.atom_count, 3)
        self.assertEqual(document.refinement_count, 2)
        self.assertEqual(document.original_boundaries, (1, 2))
        self.assertEqual(document.zero_byte_gap_boundaries, (1, 2))
        self.assertEqual(document.render_partition((1, 2)), ("A.", "B.", "C."))
        self.assertEqual(document.reconstruct((1, 2)), raw)

    def test_candidate_partition_keeps_unselected_whitespace_inside_unit(self):
        raw = b"One  two. \n Three four."
        document = canonicalize_refined_bytes(
            raw,
            splitter_for("One  two.", "Three four."),
        )
        self.assertEqual(document.original_boundaries, (2,))
        self.assertEqual(
            document.render_partition((1, 3)),
            ("One", "two. \n Three", "four."),
        )
        self.assertEqual(document.reconstruct((1, 3)), raw)

    def test_duplicate_units_align_sequentially(self):
        raw = b"Same sentence.  Same sentence."
        document = canonicalize_refined_bytes(
            raw,
            splitter_for("Same sentence.", "Same sentence."),
        )
        self.assertEqual(document.original_boundaries, (2,))
        self.assertEqual(document.reconstruct((2,)), raw)

    def test_rejects_non_whitespace_omission_between_units(self):
        raw = b"Alpha beta. Hidden Gamma delta."
        with self.assertRaisesRegex(ValueError, "P0_ALIGNMENT_FAILED"):
            canonicalize_refined_bytes(
                raw,
                splitter_for("Alpha beta.", "Gamma delta."),
            )

    def test_rejects_non_whitespace_trailing_omission(self):
        with self.assertRaisesRegex(ValueError, "P0_ALIGNMENT_FAILED"):
            canonicalize_refined_bytes(
                b"Alpha.Beta.Hidden",
                splitter_for("Alpha.", "Beta."),
            )

    def test_rejects_invalid_utf8_and_single_unit_segmentation(self):
        with self.assertRaises(RefinedAtomRendererError) as context:
            canonicalize_refined_bytes(b"\xff", splitter_for("x", "y"))
        self.assertEqual(context.exception.code, "INVALID_UTF8")
        with self.assertRaisesRegex(ValueError, "P0_SEGMENTATION_INVALID"):
            canonicalize_refined_bytes(
                b"single",
                splitter_for("single"),
            )

    def test_rejects_invalid_candidate_boundaries(self):
        document = canonicalize_refined_bytes(
            b"One two. Three four.",
            splitter_for("One two.", "Three four."),
        )
        for boundaries in ((2, 1), (2, 2), (0,), (4,), (True,)):
            with self.assertRaises(ValueError):
                document.render_partition(boundaries)

    def test_public_evidence_contains_no_payload(self):
        document = canonicalize_refined_bytes(
            b"Alpha ends.Next begins.",
            splitter_for("Alpha ends.", "Next begins."),
        )
        evidence = document.public_evidence()
        self.assertNotIn("Alpha", repr(evidence))
        self.assertEqual(evidence["refinement_count"], 1)
        self.assertEqual(evidence["zero_byte_gap_count"], 1)


if __name__ == "__main__":
    unittest.main()
