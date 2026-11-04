"""Unit tests for the v2 refined-atom variable-N wrapper."""

import hashlib
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))

from certified_resegmentation_wp2_v3 import (
    SparsePairwiseProblem,
    exact_sparse_pairwise_certificate,
)
from semstamp_certified_defense_v1_variable_n import (
    ExactScore,
    InvalidOrIncompleteError,
)
from semstamp_certified_defense_v2_variable_n import (
    RefinedVariableNPairwiseProblem,
    brute_force_refined_variable_n_pairwise_certificate,
    exact_refined_fixed_count_reduction_certificate,
    exact_refined_variable_n_pairwise_certificate,
    refined_problem_from_document,
    verify_refined_result_witness,
)
from semstamp_refined_atom_renderer import (
    LexicalSpan,
    RefinedAtom,
    RefinedAtomDocument,
    canonicalize_refined_bytes,
)


def deterministic_hit(left, middle, right):
    return int((left * 31 + middle * 17 + right * 13) % 7 < 3)


def base_problem(**overrides):
    values = {
        "atom_count": 6,
        "original_boundaries": (2, 4),
        "gap_byte_lengths": (1, 0, 2, 1, 3),
        "radius": 1,
        "span_allowed": lambda start, end: True,
        "pair_hit": deterministic_hit,
        "threshold": ExactScore(1, 4),
    }
    values.update(overrides)
    return RefinedVariableNPairwiseProblem(**values)


def manual_two_atom_document(gap: bytes, original_boundaries=(1,)):
    raw = b"A" + gap + b"B"
    second_start = 1 + len(gap)
    digest_a = hashlib.sha256(b"A").hexdigest()
    digest_b = hashlib.sha256(b"B").hexdigest()
    return RefinedAtomDocument(
        raw_utf8_bytes=raw,
        raw_utf8_sha256=hashlib.sha256(raw).hexdigest(),
        decoded_text=raw.decode("utf-8"),
        lexical_spans=(
            LexicalSpan(0, 1, 0, 1, digest_a),
            LexicalSpan(
                second_start,
                second_start + 1,
                second_start,
                second_start + 1,
                digest_b,
            ),
        ),
        atoms=(
            RefinedAtom(0, 1, 0, 1, 0, digest_a),
            RefinedAtom(
                second_start,
                second_start + 1,
                second_start,
                second_start + 1,
                1,
                digest_b,
            ),
        ),
        refinement_cut_codepoints=(),
        original_units=("A", "B"),
        original_unit_codepoint_spans=((0, 1), (second_start, second_start + 1)),
        original_unit_atom_spans=((0, 1), (1, 2)),
        original_boundaries=original_boundaries,
    )


class RefinedMetadataTests(unittest.TestCase):
    def assert_error_code(self, problem, code):
        with self.assertRaises(InvalidOrIncompleteError) as context:
            exact_refined_variable_n_pairwise_certificate(problem)
        self.assertEqual(context.exception.code, code)

    def test_gap_count_mismatch_fails_before_callbacks(self):
        span_calls = []
        hit_calls = []
        problem = base_problem(
            gap_byte_lengths=(1, 0),
            span_allowed=lambda start, end: span_calls.append(
                (start, end)
            ) or True,
            pair_hit=lambda left, middle, right: hit_calls.append(
                (left, middle, right)
            ) or 0,
        )
        self.assert_error_code(problem, "REFINED_GAP_COUNT_MISMATCH")
        self.assertEqual(span_calls, [])
        self.assertEqual(hit_calls, [])

    def test_invalid_gap_lengths_fail_closed(self):
        for value in (-1, True, 1.5):
            with self.subTest(value=value):
                lengths = (1, 0, 2, 1, value)
                self.assert_error_code(
                    base_problem(gap_byte_lengths=lengths),
                    "INVALID_GAP_BYTE_LENGTH",
                )

    def test_unauthorized_zero_gap_fails_before_callbacks(self):
        span_calls = []
        problem = base_problem(
            gap_byte_lengths=(0, 1, 2, 1, 3),
            span_allowed=lambda start, end: span_calls.append(
                (start, end)
            ) or True,
        )
        self.assert_error_code(problem, "UNAUTHORIZED_ZERO_BYTE_GAP")
        self.assertEqual(span_calls, [])

    def test_gap_metadata_is_bound_into_semantic_payload(self):
        first = exact_refined_variable_n_pairwise_certificate(base_problem())
        second = exact_refined_variable_n_pairwise_certificate(
            base_problem(gap_byte_lengths=(3, 0, 2, 1, 3))
        )
        self.assertEqual(
            first.core_result.semantic_payload(),
            second.core_result.semantic_payload(),
        )
        self.assertNotEqual(
            first.gap_metadata_sha256, second.gap_metadata_sha256
        )
        self.assertNotEqual(
            first.semantic_payload(), second.semantic_payload()
        )


class ZeroGapOperationTests(unittest.TestCase):
    def test_dp_matches_oracle_with_authorized_zero_gap(self):
        problem = base_problem()
        exact = exact_refined_variable_n_pairwise_certificate(problem)
        oracle = brute_force_refined_variable_n_pairwise_certificate(problem)
        self.assertEqual(exact.semantic_payload(), oracle.semantic_payload())
        self.assertTrue(verify_refined_result_witness(problem, exact))
        self.assertEqual(exact.zero_byte_gap_boundaries, (2,))

    def test_zero_gap_identity(self):
        problem = base_problem(radius=0)
        result = exact_refined_variable_n_pairwise_certificate(problem)
        self.assertEqual(result.worst_boundaries, (2, 4))
        self.assertEqual(result.threat_set_size, 1)
        self.assertEqual(result.operation_cost, 0)

    def test_zero_gap_deletion_is_a_distinct_threat_member(self):
        original = (1, 3)
        legal = {(0, 1), (1, 3), (3, 5), (0, 3)}
        problem = RefinedVariableNPairwiseProblem(
            atom_count=5,
            original_boundaries=original,
            gap_byte_lengths=(0, 1, 1, 2),
            radius=1,
            span_allowed=lambda start, end: (start, end) in legal,
            pair_hit=lambda left, middle, right: int(
                (left, middle, right) != (0, 3, 5)
            ),
        )
        result = exact_refined_variable_n_pairwise_certificate(problem)
        self.assertEqual(result.worst_boundaries, (3,))
        self.assertEqual((result.insertions, result.deletions), (0, 1))
        self.assertIn((0, 1, 1), result.histogram_by_insertions_deletions)

    def test_zero_gap_relocation_is_a_distinct_threat_member(self):
        original = (1, 3)
        target = (2, 3)
        legal = {
            (0, 1), (1, 3), (3, 5),
            (0, 2), (2, 3), (1, 2),
        }
        target_hits = {(0, 2, 3), (2, 3, 5)}
        problem = RefinedVariableNPairwiseProblem(
            atom_count=5,
            original_boundaries=original,
            gap_byte_lengths=(0, 2, 1, 1),
            radius=1,
            span_allowed=lambda start, end: (start, end) in legal,
            pair_hit=lambda left, middle, right: int(
                (left, middle, right) not in target_hits
            ),
        )
        result = exact_refined_variable_n_pairwise_certificate(problem)
        self.assertEqual(result.worst_boundaries, target)
        self.assertEqual((result.insertions, result.deletions), (1, 1))
        self.assertIn((1, 1, 1), result.histogram_by_insertions_deletions)


class BridgeTests(unittest.TestCase):
    def test_renderer_bridge_preserves_refined_topology(self):
        document = canonicalize_refined_bytes(
            b"Alpha ends.Next begins.",
            lambda _text: ["Alpha ends.", "Next begins."],
        )
        problem = refined_problem_from_document(
            document,
            radius=1,
            span_allowed=lambda start, end: True,
            pair_hit=deterministic_hit,
        )
        self.assertEqual(problem.atom_count, document.atom_count)
        self.assertEqual(
            problem.original_boundaries, document.original_boundaries
        )
        self.assertEqual(problem.gap_byte_lengths, (1, 0, 1))
        result = exact_refined_variable_n_pairwise_certificate(problem)
        self.assertEqual(result.zero_byte_gap_boundaries, (2,))

    def test_bridge_rejects_nonwhitespace_internal_gap(self):
        with self.assertRaises(InvalidOrIncompleteError) as context:
            refined_problem_from_document(
                manual_two_atom_document(b"-"),
                radius=0,
                span_allowed=lambda start, end: True,
                pair_hit=lambda left, middle, right: 0,
            )
        self.assertEqual(
            context.exception.code, "NONWHITESPACE_REFINED_GAP"
        )

    def test_bridge_rejects_unauthorized_zero_gap(self):
        with self.assertRaises(InvalidOrIncompleteError) as context:
            refined_problem_from_document(
                manual_two_atom_document(b"", original_boundaries=()),
                radius=0,
                span_allowed=lambda start, end: True,
                pair_hit=lambda left, middle, right: 0,
            )
        self.assertEqual(
            context.exception.code, "UNAUTHORIZED_ZERO_BYTE_GAP"
        )


class ReductionTests(unittest.TestCase):
    def test_refined_fixed_count_matches_existing_sparse_dp(self):
        problem = base_problem(radius=2)
        reduced = exact_refined_fixed_count_reduction_certificate(problem)
        existing = exact_sparse_pairwise_certificate(
            SparsePairwiseProblem(
                token_count=problem.atom_count,
                original_boundaries=problem.original_boundaries,
                max_boundary_moves=problem.radius,
                span_allowed=problem.span_allowed,
                pair_scorer=lambda left, middle, right: float(
                    problem.pair_hit(left, middle, right)
                ),
                threshold=float(problem.threshold.hits),
                normalization=1.0,
            )
        )
        self.assertEqual(
            reduced.exact_score.hits, int(existing.certified_score)
        )
        self.assertEqual(reduced.worst_boundaries, existing.worst_boundaries)
        self.assertEqual(reduced.threat_set_size, existing.threat_set_size)


if __name__ == "__main__":
    unittest.main()
