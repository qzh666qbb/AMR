"""Unit tests for the WP1 fixed-byte renderer contract."""

import hashlib
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))

from semstamp_fixed_byte_renderer import canonicalize_fixed_bytes


def splitter_for(*units):
    return lambda _text: list(units)


class FixedByteRendererTests(unittest.TestCase):
    def test_preserves_all_raw_whitespace_and_unicode_bytes(self):
        text = " \tAlpha  beta.\r\n\r\nGamma\u00a0delta! \n"
        raw = text.encode("utf-8")
        document = canonicalize_fixed_bytes(
            raw,
            splitter_for("Alpha  beta.", "Gamma\u00a0delta!"),
        )
        self.assertEqual(
            document.render_partition(document.original_boundaries),
            ("Alpha  beta.", "Gamma\u00a0delta!"),
        )
        self.assertEqual(document.prefix_gap, b" \t")
        self.assertEqual(document.suffix_gap, b" \n")
        self.assertEqual(document.reconstruct(document.original_boundaries), raw)
        self.assertEqual(
            document.public_evidence()["byte_reconstruction_sha256"],
            hashlib.sha256(raw).hexdigest(),
        )

    def test_candidate_partition_keeps_unselected_gaps_inside_units(self):
        raw = b"One  two. \n Three four."
        document = canonicalize_fixed_bytes(
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
        document = canonicalize_fixed_bytes(
            raw,
            splitter_for("Same sentence.", "Same sentence."),
        )
        self.assertEqual(document.original_boundaries, (2,))
        self.assertEqual(document.reconstruct((2,)), raw)

    def test_rejects_non_whitespace_between_splitter_units(self):
        raw = b"Alpha beta. Hidden Gamma delta."
        with self.assertRaisesRegex(ValueError, "P0_ALIGNMENT_FAILED"):
            canonicalize_fixed_bytes(
                raw,
                splitter_for("Alpha beta.", "Gamma delta."),
            )

    def test_rejects_boundary_that_splits_an_atom(self):
        raw = b"Alpha beta.Gamma delta."
        with self.assertRaisesRegex(ValueError, "P0_ALIGNMENT_FAILED"):
            canonicalize_fixed_bytes(
                raw,
                splitter_for("Alpha beta.", "Gamma delta."),
            )

    def test_rejects_invalid_utf8_and_insufficient_atoms(self):
        with self.assertRaises(UnicodeDecodeError):
            canonicalize_fixed_bytes(b"\xff", splitter_for("x", "y"))
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_ATOMS"):
            canonicalize_fixed_bytes(b"single", splitter_for("single"))

    def test_rejects_invalid_candidate_boundaries(self):
        document = canonicalize_fixed_bytes(
            b"One two. Three four.",
            splitter_for("One two.", "Three four."),
        )
        for boundaries in ((2, 1), (2, 2), (0,), (4,)):
            with self.assertRaises(ValueError):
                document.render_partition(boundaries)


if __name__ == "__main__":
    unittest.main()
