"""Unit tests for the WP2 variable-N exact certificate."""

import json
import random
import unittest

from certified_resegmentation_wp2_v3 import (
    SparsePairwiseProblem,
    exact_sparse_pairwise_certificate,
)
from semstamp_certified_defense_v1_variable_n import (
    CERTIFIED_WATERMARKED,
    NOT_CERTIFIED,
    ExactScore,
    InvalidOrIncompleteError,
    VariableNPairwiseProblem,
    brute_force_variable_n_pairwise_certificate,
    compare_exact_scores,
    exact_fixed_count_reduction_certificate,
    exact_variable_n_pairwise_certificate,
    verify_result_witness,
)


ZERO_THRESHOLD = ExactScore(1, 4)


def deterministic_hit(left, middle, right):
    return int(((left * 31 + middle * 17 + right * 13) % 7) < 3)


class ExactScoreTests(unittest.TestCase):
    def test_sign_square_and_cross_m_ties(self):
        ordered = [
            ExactScore(0, 8),
            ExactScore(0, 3),
            ExactScore(0, 2),
            ExactScore(1, 4),
            ExactScore(1, 3),
            ExactScore(1, 2),
        ]
        for left, right in zip(ordered, ordered[1:]):
            self.assertLess(compare_exact_scores(left, right), 0)
            self.assertGreater(compare_exact_scores(right, left), 0)
        self.assertEqual(
            compare_exact_scores(
                ExactScore(0, 2), ExactScore(1, 8)
            ),
            0,
        )
        self.assertEqual(
            compare_exact_scores(
                ExactScore(1, 4), ExactScore(2, 8)
            ),
            0,
        )

    def test_exact_score_validation(self):
        for hits, edges in ((-1, 2), (3, 2), (0, 0), (True, 2)):
            with self.subTest(hits=hits, edges=edges):
                with self.assertRaises(ValueError):
                    ExactScore(hits, edges)


class VariableNCorrectnessTests(unittest.TestCase):
    def assert_semantically_equal(self, left, right):
        self.assertEqual(left.semantic_payload(), right.semantic_payload())

    def test_all_zero_hits_use_canonical_variable_n_witness(self):
        problem = VariableNPairwiseProblem(
            token_count=6,
            original_boundaries=(2, 4),
            radius=1,
            span_allowed=lambda start, end: True,
            pair_hit=lambda left, middle, right: 0,
            threshold=ZERO_THRESHOLD,
        )
        result = exact_variable_n_pairwise_certificate(problem)
        self.assertEqual(result.exact_score, ExactScore(0, 3))
        self.assertEqual(result.worst_boundaries, (1, 2, 4))
        self.assertEqual((result.insertions, result.deletions), (1, 0))
        self.assertEqual(result.decision, NOT_CERTIFIED)
        self.assertTrue(verify_result_witness(problem, result))

    def test_radius_one_contains_all_four_operation_classes(self):
        problem = VariableNPairwiseProblem(
            token_count=6,
            original_boundaries=(2, 4),
            radius=1,
            span_allowed=lambda start, end: True,
            pair_hit=deterministic_hit,
        )
        result = exact_variable_n_pairwise_certificate(problem)
        observed = {
            (insertions, deletions)
            for insertions, deletions, count
            in result.histogram_by_insertions_deletions
            if count
        }
        self.assertEqual(observed, {(0, 0), (1, 0), (0, 1), (1, 1)})
        self.assertEqual(result.threat_set_size, 12)

    def test_radius_two_contains_mixed_operations(self):
        problem = VariableNPairwiseProblem(
            token_count=7,
            original_boundaries=(2, 4),
            radius=2,
            span_allowed=lambda start, end: True,
            pair_hit=deterministic_hit,
        )
        result = exact_variable_n_pairwise_certificate(problem)
        observed = {
            (insertions, deletions)
            for insertions, deletions, count
            in result.histogram_by_insertions_deletions
            if count
        }
        self.assertIn((2, 1), observed)
        self.assertIn((1, 2), observed)

    def test_two_unit_original_excludes_zero_edge_deletion(self):
        problem = VariableNPairwiseProblem(
            token_count=5,
            original_boundaries=(2,),
            radius=1,
            span_allowed=lambda start, end: True,
            pair_hit=deterministic_hit,
        )
        result = exact_variable_n_pairwise_certificate(problem)
        self.assertEqual(
            tuple(edge_count for edge_count, _ in result.histogram_by_edge_count),
            (1, 2),
        )
        self.assertNotIn(
            (0, 1),
            {
                (insertions, deletions)
                for insertions, deletions, _
                in result.histogram_by_insertions_deletions
            },
        )

    def test_illegal_spans_are_filtered(self):
        original = (2, 4)
        original_spans = {(0, 2), (2, 4), (4, 6)}
        problem = VariableNPairwiseProblem(
            token_count=6,
            original_boundaries=original,
            radius=1,
            span_allowed=lambda start, end: (
                (start, end) in original_spans or end - start <= 2
            ),
            pair_hit=deterministic_hit,
            threshold=ExactScore(0, 1),
        )
        exact = exact_variable_n_pairwise_certificate(problem)
        oracle = brute_force_variable_n_pairwise_certificate(problem)
        self.assert_semantically_equal(exact, oracle)
        self.assertLess(exact.threat_set_size, 12)

    def test_cross_m_exact_tie_uses_canonical_order(self):
        token_count = 9
        original = (2, 4, 6, 7, 8)
        tied_low = (2, 6)
        tied_high = tuple(range(1, token_count))
        legal_spans = set()
        for boundaries in (original, tied_low, tied_high):
            endpoints = (0, *boundaries, token_count)
            legal_spans.update(zip(endpoints, endpoints[1:]))
        hit_triples = {
            (0, 1, 2),
            (0, 2, 3),
            (2, 4, 5),
            (2, 4, 6),
            (2, 6, 7),
        }
        problem = VariableNPairwiseProblem(
            token_count=token_count,
            original_boundaries=original,
            radius=3,
            span_allowed=lambda start, end: (
                (start, end) in legal_spans
            ),
            pair_hit=lambda left, middle, right: int(
                (left, middle, right) in hit_triples
            ),
        )
        result = exact_variable_n_pairwise_certificate(problem)
        self.assertEqual(result.exact_score, ExactScore(0, 2))
        self.assertEqual(
            compare_exact_scores(result.exact_score, ExactScore(1, 8)),
            0,
        )
        self.assertEqual(result.worst_boundaries, tied_low)
        self.assertEqual((result.insertions, result.deletions), (0, 3))

    def test_strict_threshold_tie_is_not_certified(self):
        base_problem = VariableNPairwiseProblem(
            token_count=7,
            original_boundaries=(2, 5),
            radius=2,
            span_allowed=lambda start, end: True,
            pair_hit=deterministic_hit,
        )
        base = exact_variable_n_pairwise_certificate(base_problem)
        tied_problem = VariableNPairwiseProblem(
            token_count=base_problem.token_count,
            original_boundaries=base_problem.original_boundaries,
            radius=base_problem.radius,
            span_allowed=base_problem.span_allowed,
            pair_hit=base_problem.pair_hit,
            threshold=base.exact_score,
        )
        result = exact_variable_n_pairwise_certificate(tied_problem)
        self.assertEqual(result.decision, NOT_CERTIFIED)

    def test_radius_zero_equals_direct_clean_score(self):
        problem = VariableNPairwiseProblem(
            token_count=8,
            original_boundaries=(1, 4, 6),
            radius=0,
            span_allowed=lambda start, end: True,
            pair_hit=deterministic_hit,
            threshold=ZERO_THRESHOLD,
        )
        result = exact_variable_n_pairwise_certificate(problem)
        self.assertEqual(result.worst_boundaries, problem.original_boundaries)
        self.assertEqual(result.exact_score, result.original_exact_score)
        self.assertEqual(result.threat_set_size, 1)

    def test_radius_monotonicity(self):
        results = []
        for radius in range(4):
            results.append(exact_variable_n_pairwise_certificate(
                VariableNPairwiseProblem(
                    token_count=8,
                    original_boundaries=(2, 4, 6),
                    radius=radius,
                    span_allowed=lambda start, end: end - start <= 5,
                    pair_hit=deterministic_hit,
                )
            ))
        for left, right in zip(results, results[1:]):
            self.assertGreaterEqual(
                compare_exact_scores(left.exact_score, right.exact_score),
                0,
            )

    def test_random_dp_matches_independent_oracle(self):
        rng = random.Random(20261008)
        for _ in range(300):
            token_count = rng.randint(2, 10)
            original_count = rng.randint(1, token_count - 1)
            original = tuple(sorted(rng.sample(
                range(1, token_count), original_count
            )))
            original_spans = set(zip(
                (0, *original), (*original, token_count)
            ))
            legal = {
                (start, end): (
                    (start, end) in original_spans
                    or rng.random() >= 0.25
                )
                for start in range(token_count)
                for end in range(start + 1, token_count + 1)
            }
            salt = rng.randrange(1_000_000)
            problem = VariableNPairwiseProblem(
                token_count=token_count,
                original_boundaries=original,
                radius=rng.randint(0, 3),
                span_allowed=lambda start, end, table=legal: (
                    table[(start, end)]
                ),
                pair_hit=lambda left, middle, right, value=salt: int(
                    (
                        left * 1009
                        + middle * 9176
                        + right * 65537
                        + value
                    )
                    % 11
                    < 4
                ),
                threshold=ZERO_THRESHOLD,
            )
            exact = exact_variable_n_pairwise_certificate(problem)
            oracle = brute_force_variable_n_pairwise_certificate(problem)
            self.assert_semantically_equal(exact, oracle)
            self.assertTrue(verify_result_witness(problem, exact))
            self.assertEqual(
                exact.score_access_count_during_construction, 0
            )

    def test_replay_is_byte_deterministic(self):
        problem = VariableNPairwiseProblem(
            token_count=9,
            original_boundaries=(2, 5, 7),
            radius=3,
            span_allowed=lambda start, end: end - start <= 6,
            pair_hit=deterministic_hit,
            threshold=ZERO_THRESHOLD,
        )
        first = json.dumps(
            exact_variable_n_pairwise_certificate(problem).to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        second = json.dumps(
            exact_variable_n_pairwise_certificate(problem).to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        self.assertEqual(first, second)


class FixedCountReductionTests(unittest.TestCase):
    def test_reduction_matches_existing_sparse_pairwise_dp(self):
        problem = VariableNPairwiseProblem(
            token_count=9,
            original_boundaries=(2, 5, 7),
            radius=2,
            span_allowed=lambda start, end: end - start <= 6,
            pair_hit=deterministic_hit,
            threshold=ExactScore(1, 3),
        )
        reduced = exact_fixed_count_reduction_certificate(problem)
        existing = exact_sparse_pairwise_certificate(
            SparsePairwiseProblem(
                token_count=problem.token_count,
                original_boundaries=problem.original_boundaries,
                max_boundary_moves=problem.radius,
                span_allowed=problem.span_allowed,
                pair_scorer=lambda left, middle, right: float(
                    problem.pair_hit(left, middle, right)
                ),
                normalization=1.0,
                threshold=float(problem.threshold.hits),
            )
        )
        self.assertEqual(
            reduced.exact_score.hits, int(existing.certified_score)
        )
        self.assertEqual(reduced.worst_boundaries, existing.worst_boundaries)
        self.assertEqual(reduced.threat_set_size, existing.threat_set_size)
        self.assertEqual(
            tuple(
                (insertions, count)
                for insertions, deletions, count
                in reduced.histogram_by_insertions_deletions
                if insertions == deletions
            ),
            existing.exact_move_histogram,
        )


class FailClosedTests(unittest.TestCase):
    def base_problem(self, span_allowed, pair_hit):
        return VariableNPairwiseProblem(
            token_count=5,
            original_boundaries=(2, 4),
            radius=1,
            span_allowed=span_allowed,
            pair_hit=pair_hit,
        )

    def test_nonboolean_legality_fails_closed(self):
        with self.assertRaises(InvalidOrIncompleteError) as context:
            exact_variable_n_pairwise_certificate(
                self.base_problem(lambda start, end: 1, deterministic_hit)
            )
        self.assertEqual(context.exception.code, "NONBOOLEAN_SPAN_LEGALITY")

    def test_nonbinary_hit_fails_closed(self):
        with self.assertRaises(InvalidOrIncompleteError) as context:
            exact_variable_n_pairwise_certificate(
                self.base_problem(
                    lambda start, end: True,
                    lambda left, middle, right: 2,
                )
            )
        self.assertEqual(context.exception.code, "NONBINARY_PAIR_HIT")

    def test_missing_hit_fails_closed(self):
        def missing(left, middle, right):
            raise KeyError((left, middle, right))

        with self.assertRaises(InvalidOrIncompleteError) as context:
            exact_variable_n_pairwise_certificate(
                self.base_problem(lambda start, end: True, missing)
            )
        self.assertEqual(context.exception.code, "MISSING_PAIR_HIT")

    def test_pair_hit_is_not_read_during_topology_construction(self):
        span_calls = []
        hit_calls = []

        def span_allowed(start, end):
            self.assertEqual(hit_calls, [])
            span_calls.append((start, end))
            return True

        def pair_hit(left, middle, right):
            hit_calls.append((left, middle, right))
            return 0

        result = exact_variable_n_pairwise_certificate(
            self.base_problem(span_allowed, pair_hit)
        )
        self.assertTrue(span_calls)
        self.assertTrue(hit_calls)
        self.assertEqual(
            result.score_access_count_during_construction, 0
        )


if __name__ == "__main__":
    unittest.main()
