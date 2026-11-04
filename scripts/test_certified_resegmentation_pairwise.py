"""Offline tests for exact adjacent-pair resegmentation certificates."""

from dataclasses import replace
from itertools import combinations
import math
import random
import unittest

from certified_resegmentation import (
    CertificateProblem,
    PairwiseCertificateProblem,
    boundary_moves,
    brute_force_pairwise_certificate,
    build_pair_scores,
    build_span_scores,
    exact_certificate,
    exact_pairwise_certificate,
)


def valid_partitions(token_count, minimum, maximum):
    for count in range(token_count):
        for boundaries in combinations(range(1, token_count), count):
            endpoints = (0, *boundaries, token_count)
            lengths = [
                endpoints[index + 1] - endpoints[index]
                for index in range(len(endpoints) - 1)
            ]
            if all(minimum <= length <= maximum for length in lengths):
                yield boundaries


def make_pairwise_problem(
    token_count,
    original,
    minimum,
    maximum,
    radius,
    scorer,
    threshold=0.0,
    normalization=None,
    allowed=None,
):
    allowed = (
        frozenset(range(1, token_count))
        if allowed is None
        else frozenset(allowed)
    )
    if normalization is None:
        normalization = len(original)
    return PairwiseCertificateProblem(
        token_count=token_count,
        original_boundaries=tuple(original),
        min_unit_words=minimum,
        max_unit_words=maximum,
        max_boundary_moves=radius,
        pair_scores=build_pair_scores(
            token_count, minimum, maximum, scorer, allowed
        ),
        threshold=threshold,
        normalization=normalization,
        allowed_boundaries=allowed,
    )


def score_pair_partition(problem, boundaries):
    endpoints = (0, *boundaries, problem.token_count)
    pair_sum = sum(
        problem.pair_scores[
            (endpoints[index - 1], endpoints[index], endpoints[index + 1])
        ]
        for index in range(1, len(endpoints) - 1)
    )
    return problem.bias + pair_sum / problem.normalization


def compare_problem(testcase, problem):
    dynamic = exact_pairwise_certificate(problem)
    oracle = brute_force_pairwise_certificate(problem)
    testcase.assertTrue(math.isclose(
        dynamic.certified_score,
        oracle.certified_score,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ))
    testcase.assertEqual(
        dynamic.exact_move_histogram, oracle.exact_move_histogram
    )
    testcase.assertEqual(dynamic.threat_set_size, oracle.threat_set_size)
    testcase.assertEqual(
        dynamic.certified_positive, oracle.certified_positive
    )
    testcase.assertTrue(math.isclose(
        score_pair_partition(problem, dynamic.worst_boundaries),
        dynamic.certified_score,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ))
    testcase.assertLessEqual(
        boundary_moves(
            dynamic.worst_boundaries,
            dynamic.original_boundaries,
        ),
        dynamic.requested_radius,
    )
    return dynamic


class HandCheckedTests(unittest.TestCase):
    def setUp(self):
        original_pairs = {(0, 2, 4), (2, 4, 6)}
        low_pairs = {
            (0, 1, 4): 0,
            (1, 4, 6): 10,
        }

        def scorer(start, middle, end):
            if (start, middle, end) in low_pairs:
                return low_pairs[(start, middle, end)]
            if (start, middle, end) in original_pairs:
                return 15
            return 100

        self.problem = make_pairwise_problem(
            token_count=6,
            original=(2, 4),
            minimum=1,
            maximum=3,
            radius=1,
            scorer=scorer,
            threshold=10,
            normalization=2,
        )

    def test_known_worst_partition_flips_decision(self):
        dynamic = compare_problem(self, self.problem)
        self.assertEqual(dynamic.worst_boundaries, (1, 4))
        self.assertEqual(dynamic.original_score, 15)
        self.assertEqual(dynamic.certified_score, 5)
        self.assertTrue(dynamic.original_positive)
        self.assertFalse(dynamic.certified_positive)
        self.assertEqual(dynamic.worst_case_drop, 10)
        self.assertGreater(dynamic.reachable_state_count, 0)
        self.assertGreater(dynamic.transitions_evaluated, 0)
        self.assertGreater(dynamic.peak_layer_states, 0)

    def test_zero_radius_is_original_only(self):
        result = exact_pairwise_certificate(
            replace(self.problem, max_boundary_moves=0)
        )
        self.assertEqual(result.worst_boundaries, (2, 4))
        self.assertEqual(result.threat_set_size, 1)
        self.assertEqual(result.exact_move_histogram, ((0, 1),))
        self.assertEqual(result.certified_score, result.original_score)

    def test_restricted_lattice_reports_vacuity(self):
        allowed = frozenset(self.problem.original_boundaries)
        problem = replace(
            self.problem,
            max_boundary_moves=1,
            allowed_boundaries=allowed,
            pair_scores=build_pair_scores(
                6, 1, 3, lambda *_: 1, allowed
            ),
        )
        result = exact_pairwise_certificate(problem)
        self.assertTrue(result.singleton_threat_set)
        self.assertFalse(result.budget_reaches_requested_radius)
        self.assertEqual(result.max_realized_moves, 0)

    def test_strict_threshold_does_not_certify_tie(self):
        problem = replace(
            self.problem,
            threshold=15,
            max_boundary_moves=0,
        )
        result = exact_pairwise_certificate(problem)
        self.assertFalse(result.original_positive)
        self.assertFalse(result.certified_positive)
        self.assertEqual(result.certificate_margin, 0)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.problem = make_pairwise_problem(
            6,
            (2, 4),
            1,
            3,
            1,
            lambda start, middle, end: start + 2 * middle + end,
        )

    def test_invalid_normalization_fails_closed(self):
        for value in (0, -1, float("inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    exact_pairwise_certificate(
                        replace(self.problem, normalization=value)
                    )

    def test_single_unit_fails_closed(self):
        problem = PairwiseCertificateProblem(
            token_count=3,
            original_boundaries=(),
            min_unit_words=1,
            max_unit_words=3,
            max_boundary_moves=0,
            pair_scores={},
            threshold=0,
            normalization=1,
        )
        with self.assertRaises(ValueError):
            exact_pairwise_certificate(problem)

    def test_missing_and_nonfinite_scores_fail_closed(self):
        scores = dict(self.problem.pair_scores)
        scores.pop(next(iter(scores)))
        with self.assertRaises(ValueError):
            exact_pairwise_certificate(
                replace(self.problem, pair_scores=scores)
            )
        scores = dict(self.problem.pair_scores)
        scores[next(iter(scores))] = float("nan")
        with self.assertRaises(ValueError):
            exact_pairwise_certificate(
                replace(self.problem, pair_scores=scores)
            )

    def test_invalid_structure_fails_closed(self):
        with self.assertRaises(ValueError):
            exact_pairwise_certificate(
                replace(self.problem, max_boundary_moves=3)
            )
        with self.assertRaises(ValueError):
            exact_pairwise_certificate(
                replace(
                    self.problem,
                    allowed_boundaries=frozenset({1, 2, 3, 5}),
                )
            )
        with self.assertRaises(ValueError):
            exact_pairwise_certificate(
                replace(self.problem, original_boundaries=(1, 5))
            )

    def test_builder_rejects_nonfinite_score(self):
        with self.assertRaises(ValueError):
            build_pair_scores(6, 1, 3, lambda *_: float("inf"))

    def test_brute_force_limit_prevents_accidental_large_search(self):
        with self.assertRaises(ValueError):
            brute_force_pairwise_certificate(
                self.problem, combination_limit=1
            )


class EquivalenceTests(unittest.TestCase):
    def test_every_small_legal_base_partition(self):
        comparisons = 0
        for token_count in range(2, 8):
            for minimum in range(1, min(3, token_count) + 1):
                for maximum in range(minimum, min(4, token_count) + 1):
                    partitions = [
                        partition
                        for partition in valid_partitions(
                            token_count, minimum, maximum
                        )
                        if partition
                    ]
                    for original in partitions:
                        for radius in range(len(original) + 1):
                            scorer = (
                                lambda start, middle, end, n=token_count: (
                                    11 * start
                                    + 17 * middle
                                    + 31 * end
                                    + 13 * n
                                ) % 23 - 11
                            )
                            problem = make_pairwise_problem(
                                token_count,
                                original,
                                minimum,
                                maximum,
                                radius,
                                scorer,
                                threshold=0.25,
                            )
                            compare_problem(self, problem)
                            comparisons += 1
        self.assertGreater(comparisons, 1000)

    def test_random_lattices_and_tied_scores(self):
        rng = random.Random(20260922)
        comparisons = 0
        while comparisons < 500:
            token_count = rng.randint(2, 10)
            minimum = rng.randint(1, min(3, token_count))
            maximum = rng.randint(minimum, min(token_count, minimum + 4))
            partitions = [
                partition
                for partition in valid_partitions(
                    token_count, minimum, maximum
                )
                if partition
            ]
            if not partitions:
                continue
            original = rng.choice(partitions)
            allowed = set(original)
            allowed.update(
                gap
                for gap in range(1, token_count)
                if rng.random() < 0.55
            )
            values = {}

            def scorer(start, middle, end):
                key = (start, middle, end)
                if key not in values:
                    values[key] = rng.randint(-4, 4) / 2
                return values[key]

            problem = make_pairwise_problem(
                token_count,
                original,
                minimum,
                maximum,
                rng.randint(0, len(original)),
                scorer,
                threshold=rng.randint(-2, 2) / 2,
                normalization=rng.randint(1, 5) / 2,
                allowed=allowed,
            )
            compare_problem(self, problem)
            comparisons += 1
        self.assertEqual(comparisons, 500)

    def test_current_span_degeneracy_matches_additive_dp(self):
        comparisons = 0
        for token_count in range(2, 8):
            minimum = 1
            maximum = min(4, token_count)
            partitions = [
                partition
                for partition in valid_partitions(
                    token_count, minimum, maximum
                )
                if partition
            ]
            allowed = frozenset(range(1, token_count))
            for original in partitions:
                unit_count = len(original) + 1

                def span_scorer(start, end, n=token_count):
                    if start == 0:
                        return 0
                    return (19 * start + 29 * end + 7 * n) % 17 - 8

                span_scores = build_span_scores(
                    token_count,
                    minimum,
                    maximum,
                    span_scorer,
                    allowed,
                )

                def current_span_scorer(previous_start, start, end):
                    del previous_start
                    return span_scorer(start, end)

                pair_scores = build_pair_scores(
                    token_count,
                    minimum,
                    maximum,
                    current_span_scorer,
                    allowed,
                )
                for radius in range(len(original) + 1):
                    additive = exact_certificate(CertificateProblem(
                        token_count=token_count,
                        original_boundaries=original,
                        min_unit_words=minimum,
                        max_unit_words=maximum,
                        max_boundary_moves=radius,
                        span_scores=span_scores,
                        threshold=0.25,
                        allowed_boundaries=allowed,
                    ))
                    pairwise = exact_pairwise_certificate(
                        PairwiseCertificateProblem(
                            token_count=token_count,
                            original_boundaries=original,
                            min_unit_words=minimum,
                            max_unit_words=maximum,
                            max_boundary_moves=radius,
                            pair_scores=pair_scores,
                            threshold=0.25,
                            normalization=unit_count,
                            allowed_boundaries=allowed,
                        )
                    )
                    self.assertTrue(math.isclose(
                        additive.certified_score,
                        pairwise.certified_score,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ))
                    self.assertEqual(
                        additive.exact_move_histogram,
                        pairwise.exact_move_histogram,
                    )
                    self.assertEqual(
                        additive.certified_positive,
                        pairwise.certified_positive,
                    )
                    comparisons += 1
        self.assertGreater(comparisons, 100)


if __name__ == "__main__":
    unittest.main()
