"""Offline tests for exact fixed-count resegmentation certificates."""

from dataclasses import replace
from itertools import combinations
import math
import random
import unittest

from certified_resegmentation import (
    CertificateProblem,
    boundary_moves,
    brute_force_certificate,
    build_span_scores,
    exact_certificate,
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


def make_problem(
    token_count,
    original,
    minimum,
    maximum,
    radius,
    scorer,
    threshold=0.0,
    allowed=None,
):
    allowed = (
        frozenset(range(1, token_count))
        if allowed is None
        else frozenset(allowed)
    )
    return CertificateProblem(
        token_count=token_count,
        original_boundaries=tuple(original),
        min_unit_words=minimum,
        max_unit_words=maximum,
        max_boundary_moves=radius,
        span_scores=build_span_scores(
            token_count, minimum, maximum, scorer, allowed
        ),
        threshold=threshold,
        allowed_boundaries=allowed,
    )


def assert_results_match(testcase, dynamic, oracle):
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


class HandCheckedTests(unittest.TestCase):
    def setUp(self):
        original_spans = {(0, 2), (2, 4), (4, 6)}
        low_spans = {(0, 1), (1, 4)}

        def scorer(start, end):
            if (start, end) in low_spans:
                return 0
            if (start, end) in original_spans:
                return 15
            return 100

        self.problem = make_problem(
            token_count=6,
            original=(2, 4),
            minimum=1,
            maximum=3,
            radius=1,
            scorer=scorer,
            threshold=10,
        )

    def test_known_worst_partition_flips_decision(self):
        dynamic = exact_certificate(self.problem)
        oracle = brute_force_certificate(self.problem)
        assert_results_match(self, dynamic, oracle)
        self.assertEqual(dynamic.worst_boundaries, (1, 4))
        self.assertEqual(dynamic.original_score, 15)
        self.assertEqual(dynamic.certified_score, 5)
        self.assertTrue(dynamic.original_positive)
        self.assertFalse(dynamic.certified_positive)
        self.assertEqual(dynamic.worst_case_drop, 10)

    def test_zero_radius_is_original_only(self):
        result = exact_certificate(
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
            span_scores=build_span_scores(6, 1, 3, lambda *_: 1, allowed),
        )
        result = exact_certificate(problem)
        self.assertTrue(result.singleton_threat_set)
        self.assertFalse(result.budget_reaches_requested_radius)
        self.assertEqual(result.max_realized_moves, 0)

    def test_strict_threshold_does_not_certify_tie(self):
        problem = replace(
            self.problem,
            threshold=15,
            max_boundary_moves=0,
        )
        result = exact_certificate(problem)
        self.assertFalse(result.original_positive)
        self.assertFalse(result.certified_positive)
        self.assertEqual(result.certificate_margin, 0)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.problem = make_problem(
            6, (2, 4), 1, 3, 1, lambda start, end: start + end
        )

    def test_boundary_move_requires_fixed_count(self):
        self.assertEqual(boundary_moves((1, 4), (2, 4)), 1)
        with self.assertRaises(ValueError):
            boundary_moves((1,), (2, 4))

    def test_invalid_radius_fails_closed(self):
        with self.assertRaises(ValueError):
            exact_certificate(
                replace(self.problem, max_boundary_moves=3)
            )

    def test_missing_original_boundary_from_lattice_fails_closed(self):
        with self.assertRaises(ValueError):
            exact_certificate(
                replace(
                    self.problem,
                    allowed_boundaries=frozenset({1, 2, 3, 5}),
                )
            )

    def test_illegal_original_lengths_fail_closed(self):
        with self.assertRaises(ValueError):
            exact_certificate(
                replace(self.problem, original_boundaries=(1, 5))
            )

    def test_missing_and_nonfinite_scores_fail_closed(self):
        scores = dict(self.problem.span_scores)
        scores.pop(next(iter(scores)))
        with self.assertRaises(ValueError):
            exact_certificate(replace(self.problem, span_scores=scores))
        scores = dict(self.problem.span_scores)
        scores[next(iter(scores))] = float("nan")
        with self.assertRaises(ValueError):
            exact_certificate(replace(self.problem, span_scores=scores))

    def test_brute_force_limit_prevents_accidental_large_search(self):
        with self.assertRaises(ValueError):
            brute_force_certificate(self.problem, combination_limit=1)


class EquivalenceTests(unittest.TestCase):
    def test_every_small_legal_base_partition(self):
        comparisons = 0
        for token_count in range(2, 8):
            for minimum in range(1, min(3, token_count) + 1):
                for maximum in range(minimum, min(4, token_count) + 1):
                    partitions = list(
                        valid_partitions(token_count, minimum, maximum)
                    )
                    for original in partitions:
                        for radius in range(len(original) + 1):
                            scorer = lambda start, end, n=token_count: (
                                (17 * start + 31 * end + 13 * n) % 23 - 11
                            ) / 7
                            problem = make_problem(
                                token_count,
                                original,
                                minimum,
                                maximum,
                                radius,
                                scorer,
                                threshold=0.25,
                            )
                            dynamic = exact_certificate(problem)
                            oracle = brute_force_certificate(problem)
                            assert_results_match(self, dynamic, oracle)
                            comparisons += 1
        self.assertGreater(comparisons, 1000)

    def test_random_lattices_and_tied_scores(self):
        rng = random.Random(20260922)
        comparisons = 0
        while comparisons < 500:
            token_count = rng.randint(2, 10)
            minimum = rng.randint(1, min(3, token_count))
            maximum = rng.randint(minimum, min(token_count, minimum + 4))
            partitions = list(
                valid_partitions(token_count, minimum, maximum)
            )
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

            def scorer(start, end):
                key = (start, end)
                if key not in values:
                    values[key] = rng.randint(-4, 4) / 2
                return values[key]

            problem = make_problem(
                token_count,
                original,
                minimum,
                maximum,
                rng.randint(0, len(original)),
                scorer,
                threshold=rng.randint(-2, 2) / 2,
                allowed=allowed,
            )
            dynamic = exact_certificate(problem)
            oracle = brute_force_certificate(problem)
            assert_results_match(self, dynamic, oracle)
            score_sum = 0
            endpoints = (0, *dynamic.worst_boundaries, token_count)
            for index in range(len(endpoints) - 1):
                score_sum += problem.span_scores[
                    (endpoints[index], endpoints[index + 1])
                ]
            self.assertTrue(math.isclose(
                problem.bias + score_sum / dynamic.unit_count,
                dynamic.certified_score,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ))
            self.assertLessEqual(
                boundary_moves(
                    dynamic.worst_boundaries,
                    dynamic.original_boundaries,
                ),
                dynamic.requested_radius,
            )
            comparisons += 1
        self.assertEqual(comparisons, 500)


if __name__ == "__main__":
    unittest.main()
