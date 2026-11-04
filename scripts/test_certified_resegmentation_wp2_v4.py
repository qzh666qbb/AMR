"""Regression tests for the WP2-v4 sparse implementations."""

from itertools import combinations
import math
import random
import unittest

from certified_resegmentation_wp2_v3 import (
    SparsePairwiseProblem,
    brute_force_sparse_pairwise_certificate,
    exact_sparse_pairwise_certificate,
    rank_candidate_boundaries,
)
from certified_resegmentation_wp2_v4 import (
    exact_grouped_sparse_pairwise_certificate,
    rank_candidate_boundaries_grouped,
)


def score_function(left, middle, right):
    return float(
        (
            (left + 3) * 1009
            + (middle + 5) * 9176
            + (right + 7) * 65537
        )
        % 1000003
    )


class CandidateProjectionTests(unittest.TestCase):
    def test_closed_form_matches_v3_projection(self):
        for token_count in range(3, 13):
            gaps = tuple(range(1, token_count))
            for boundary_count in range(
                1, min(5, token_count - 1) + 1
            ):
                for original in combinations(gaps, boundary_count):
                    for radius in range(
                        min(3, boundary_count) + 1
                    ):
                        self.assertEqual(
                            rank_candidate_boundaries_grouped(
                                token_count, original, radius
                            ),
                            rank_candidate_boundaries(
                                token_count, original, radius
                            ),
                        )


class GroupedReferenceTests(unittest.TestCase):
    def test_random_three_way_results_and_metrics(self):
        rng = random.Random(20260922)
        for _ in range(500):
            token_count = rng.randint(4, 11)
            boundary_count = rng.randint(
                1, min(4, token_count - 1)
            )
            original = tuple(sorted(rng.sample(
                range(1, token_count), boundary_count
            )))
            radius = rng.randint(0, min(3, boundary_count))
            original_spans = set(zip(
                (0, *original),
                (*original, token_count),
            ))
            legal = {
                (start, end): (
                    (start, end) in original_spans
                    or rng.random() >= 0.3
                )
                for start in range(token_count)
                for end in range(start + 1, token_count + 1)
            }
            problem = SparsePairwiseProblem(
                token_count=token_count,
                original_boundaries=original,
                max_boundary_moves=radius,
                span_allowed=lambda start, end: legal[(start, end)],
                pair_scorer=score_function,
                threshold=29.0,
            )
            v3 = exact_sparse_pairwise_certificate(problem)
            v4 = exact_grouped_sparse_pairwise_certificate(problem)
            oracle = brute_force_sparse_pairwise_certificate(problem)
            for field in (
                "original_score",
                "certified_score",
                "worst_boundaries",
                "exact_move_histogram",
                "threat_set_size",
            ):
                self.assertEqual(
                    getattr(v4, field),
                    getattr(oracle, field),
                )
            for field in (
                "reachable_state_count",
                "transitions_evaluated",
                "peak_layer_states",
                "candidate_boundary_positions",
                "unique_span_checks",
                "unique_pair_scores",
            ):
                self.assertEqual(
                    getattr(v4, field),
                    getattr(v3, field),
                )

    def test_zero_score_uses_canonical_lexicographic_witness(self):
        problem = SparsePairwiseProblem(
            token_count=9,
            original_boundaries=(2, 5, 7),
            max_boundary_moves=3,
            span_allowed=lambda start, end: True,
            pair_scorer=lambda left, middle, right: 0.0,
        )
        result = exact_grouped_sparse_pairwise_certificate(problem)
        self.assertEqual(result.worst_boundaries, (2, 5, 7))
        self.assertEqual(result.certified_score, 0.0)

    def test_invalid_callbacks_fail_closed(self):
        illegal_original = SparsePairwiseProblem(
            token_count=5,
            original_boundaries=(2,),
            max_boundary_moves=1,
            span_allowed=lambda start, end: False,
            pair_scorer=score_function,
        )
        with self.assertRaises(ValueError):
            exact_grouped_sparse_pairwise_certificate(illegal_original)

        non_bool_span = SparsePairwiseProblem(
            token_count=5,
            original_boundaries=(2,),
            max_boundary_moves=1,
            span_allowed=lambda start, end: 1,
            pair_scorer=score_function,
        )
        with self.assertRaises(TypeError):
            exact_grouped_sparse_pairwise_certificate(non_bool_span)

        non_finite_score = SparsePairwiseProblem(
            token_count=5,
            original_boundaries=(2,),
            max_boundary_moves=1,
            span_allowed=lambda start, end: True,
            pair_scorer=lambda left, middle, right: math.inf,
        )
        with self.assertRaises(ValueError):
            exact_grouped_sparse_pairwise_certificate(non_finite_score)


if __name__ == "__main__":
    unittest.main()
