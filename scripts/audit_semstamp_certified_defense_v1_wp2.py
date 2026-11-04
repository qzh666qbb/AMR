"""Run the frozen WP2 pure-CPU variable-N correctness audit."""

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
import platform
import random
import sys
import time

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


SEED = 20261008
SYSTEMATIC_CASES = 5_000
RANDOM_CASES = 2_000
FIXED_REDUCTION_CASES = 1_000
MAX_TOKEN_COUNT = 10
ORACLE_COMBINATION_LIMIT = 1_000_000


def canonical_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def threshold_for(index):
    thresholds = (
        ExactScore(0, 1),
        ExactScore(1, 4),
        ExactScore(1, 3),
        ExactScore(1, 2),
    )
    return thresholds[index % len(thresholds)]


def make_systematic_problem(
    token_count,
    original,
    radius,
    legality_profile,
    hit_profile,
    case_index,
):
    endpoints = (0, *original, token_count)
    original_spans = set(zip(endpoints, endpoints[1:]))
    maximum_original_span = max(
        end - start for start, end in original_spans
    )

    if legality_profile == 0:
        def span_allowed(start, end):
            return True
    elif legality_profile == 1:
        def span_allowed(start, end, cap=maximum_original_span):
            return end - start <= cap
    else:
        def span_allowed(start, end, frozen=original_spans):
            return (
                (start, end) in frozen
                or (start * 17 + end * 31 + token_count) % 5 != 0
            )

    if hit_profile == 0:
        def pair_hit(left, middle, right):
            return 0
    elif hit_profile == 1:
        def pair_hit(left, middle, right):
            return 1
    elif hit_profile == 2:
        def pair_hit(left, middle, right):
            return int(
                (left * 31 + middle * 17 + right * 13) % 7 < 3
            )
    else:
        def pair_hit(left, middle, right):
            return int(
                (
                    left * 1009
                    + middle * 9176
                    + right * 65537
                    + case_index
                )
                % 11
                < 5
            )

    return VariableNPairwiseProblem(
        token_count=token_count,
        original_boundaries=original,
        radius=radius,
        span_allowed=span_allowed,
        pair_hit=pair_hit,
        threshold=threshold_for(case_index),
    )


def systematic_problems():
    case_index = 0
    for token_count in range(2, MAX_TOKEN_COUNT + 1):
        gaps = tuple(range(1, token_count))
        for original_count in range(
            1, min(4, token_count - 1) + 1
        ):
            for original in combinations(gaps, original_count):
                for radius in range(4):
                    for legality_profile in range(3):
                        for hit_profile in range(4):
                            yield make_systematic_problem(
                                token_count,
                                original,
                                radius,
                                legality_profile,
                                hit_profile,
                                case_index,
                            )
                            case_index += 1


def random_problem(rng, case_index):
    token_count = rng.randint(2, MAX_TOKEN_COUNT)
    original_count = rng.randint(1, token_count - 1)
    original = tuple(sorted(rng.sample(
        range(1, token_count), original_count
    )))
    endpoints = (0, *original, token_count)
    original_spans = set(zip(endpoints, endpoints[1:]))
    legal = {
        (start, end): (
            (start, end) in original_spans
            or rng.random() >= 0.30
        )
        for start in range(token_count)
        for end in range(start + 1, token_count + 1)
    }
    salt = rng.randrange(10_000_000)
    modulus = rng.choice((5, 7, 11, 13))
    cutoff = rng.randint(1, modulus - 1)
    return VariableNPairwiseProblem(
        token_count=token_count,
        original_boundaries=original,
        radius=rng.randint(0, 3),
        span_allowed=lambda start, end, table=legal: table[(start, end)],
        pair_hit=lambda left, middle, right: int(
            (
                left * 1009
                + middle * 9176
                + right * 65537
                + salt
            )
            % modulus
            < cutoff
        ),
        threshold=threshold_for(case_index),
    )


def compare_dp_and_oracle(problem):
    exact = exact_variable_n_pairwise_certificate(problem)
    oracle = brute_force_variable_n_pairwise_certificate(
        problem,
        combination_limit=ORACLE_COMBINATION_LIMIT,
    )
    if exact.semantic_payload() != oracle.semantic_payload():
        raise AssertionError(
            "variable-N DP semantic payload differs from oracle"
        )
    if exact.certified_score != oracle.certified_score:
        raise AssertionError("reported float score differs from oracle")
    if exact.score_access_count_during_construction != 0:
        raise AssertionError("DP accessed score during topology construction")
    if oracle.score_access_count_during_construction != 0:
        raise AssertionError(
            "oracle accessed score during topology construction"
        )
    if not verify_result_witness(problem, exact):
        raise AssertionError("independent witness verification failed")
    return exact


def run_correctness_suite():
    digest = hashlib.sha256()
    operation_pairs = set()
    total_threat_members = 0
    maximum_states = 0
    maximum_transitions = 0
    maximum_pair_hits = 0

    generated = systematic_problems()
    for case_index in range(SYSTEMATIC_CASES):
        try:
            problem = next(generated)
        except StopIteration as exc:
            raise AssertionError(
                "systematic generator produced too few cases"
            ) from exc
        result = compare_dp_and_oracle(problem)
        digest.update(canonical_bytes(result.semantic_payload()) + b"\n")
        operation_pairs.update(
            (insertions, deletions)
            for insertions, deletions, count
            in result.histogram_by_insertions_deletions
            if count
        )
        total_threat_members += result.threat_set_size
        maximum_states = max(
            maximum_states, result.reachable_state_count
        )
        maximum_transitions = max(
            maximum_transitions, result.transitions_evaluated
        )
        maximum_pair_hits = max(
            maximum_pair_hits, result.unique_pair_hits
        )

    rng = random.Random(SEED)
    for random_index in range(RANDOM_CASES):
        problem = random_problem(rng, SYSTEMATIC_CASES + random_index)
        result = compare_dp_and_oracle(problem)
        digest.update(canonical_bytes(result.semantic_payload()) + b"\n")
        operation_pairs.update(
            (insertions, deletions)
            for insertions, deletions, count
            in result.histogram_by_insertions_deletions
            if count
        )
        total_threat_members += result.threat_set_size
        maximum_states = max(
            maximum_states, result.reachable_state_count
        )
        maximum_transitions = max(
            maximum_transitions, result.transitions_evaluated
        )
        maximum_pair_hits = max(
            maximum_pair_hits, result.unique_pair_hits
        )

    required_pairs = {(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2)}
    if not required_pairs.issubset(operation_pairs):
        raise AssertionError("operation-class coverage is incomplete")
    return {
        "systematic_cases": SYSTEMATIC_CASES,
        "random_cases": RANDOM_CASES,
        "total_cases": SYSTEMATIC_CASES + RANDOM_CASES,
        "total_threat_members": total_threat_members,
        "maximum_reachable_states": maximum_states,
        "maximum_transitions_evaluated": maximum_transitions,
        "maximum_unique_pair_hits": maximum_pair_hits,
        "operation_pairs_observed": [
            [insertions, deletions]
            for insertions, deletions in sorted(operation_pairs)
        ],
        "semantic_sha256": digest.hexdigest(),
        "score_match": True,
        "exact_score_match": True,
        "decision_match": True,
        "witness_match": True,
        "threat_set_size_match": True,
        "all_histograms_match": True,
        "witness_verification": True,
        "score_blind_topology": True,
    }


def run_fixed_reduction_suite():
    rng = random.Random(SEED + 1)
    digest = hashlib.sha256()
    maximum_threat_size = 0
    for case_index in range(FIXED_REDUCTION_CASES):
        token_count = rng.randint(2, MAX_TOKEN_COUNT)
        original_count = rng.randint(1, token_count - 1)
        original = tuple(sorted(rng.sample(
            range(1, token_count), original_count
        )))
        endpoints = (0, *original, token_count)
        original_spans = set(zip(endpoints, endpoints[1:]))
        legal = {
            (start, end): (
                (start, end) in original_spans
                or rng.random() >= 0.25
            )
            for start in range(token_count)
            for end in range(start + 1, token_count + 1)
        }
        salt = rng.randrange(10_000_000)
        radius = rng.randint(0, min(3, original_count))
        threshold_hits = rng.randint(0, original_count)

        def span_allowed(start, end, table=legal):
            return table[(start, end)]

        def pair_hit(left, middle, right, value=salt):
            return int(
                (
                    left * 1009
                    + middle * 9176
                    + right * 65537
                    + value
                )
                % 7
                < 3
            )

        problem = VariableNPairwiseProblem(
            token_count=token_count,
            original_boundaries=original,
            radius=radius,
            span_allowed=span_allowed,
            pair_hit=pair_hit,
            threshold=ExactScore(threshold_hits, original_count),
        )
        reduced = exact_fixed_count_reduction_certificate(problem)
        existing = exact_sparse_pairwise_certificate(
            SparsePairwiseProblem(
                token_count=token_count,
                original_boundaries=original,
                max_boundary_moves=radius,
                span_allowed=span_allowed,
                pair_scorer=lambda left, middle, right: float(
                    pair_hit(left, middle, right)
                ),
                threshold=float(threshold_hits),
                normalization=1.0,
            )
        )
        if reduced.exact_score.hits != int(existing.certified_score):
            raise AssertionError("fixed-N minimum hit count differs")
        if (
            reduced.original_exact_score.hits
            != int(existing.original_score)
        ):
            raise AssertionError("fixed-N original hit count differs")
        if reduced.worst_boundaries != existing.worst_boundaries:
            raise AssertionError("fixed-N canonical witness differs")
        if reduced.threat_set_size != existing.threat_set_size:
            raise AssertionError("fixed-N threat-set size differs")
        reduced_histogram = tuple(
            (insertions, count)
            for insertions, deletions, count
            in reduced.histogram_by_insertions_deletions
            if insertions == deletions
        )
        if reduced_histogram != existing.exact_move_histogram:
            raise AssertionError("fixed-N move histogram differs")
        expected_decision = (
            CERTIFIED_WATERMARKED
            if existing.certified_score > threshold_hits
            else NOT_CERTIFIED
        )
        if reduced.decision != expected_decision:
            raise AssertionError("fixed-N strict decision differs")
        digest.update(canonical_bytes(reduced.semantic_payload()) + b"\n")
        maximum_threat_size = max(
            maximum_threat_size, reduced.threat_set_size
        )
    return {
        "cases": FIXED_REDUCTION_CASES,
        "minimum_hits_match": True,
        "original_hits_match": True,
        "witness_match": True,
        "threat_set_size_match": True,
        "move_histogram_match": True,
        "strict_decision_match": True,
        "maximum_threat_set_size": maximum_threat_size,
        "semantic_sha256": digest.hexdigest(),
    }


def cross_m_tie_problem():
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
    return VariableNPairwiseProblem(
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


def run_specialized_gates():
    cross_problem = cross_m_tie_problem()
    cross_result = compare_dp_and_oracle(cross_problem)
    if cross_result.exact_score != ExactScore(0, 2):
        raise AssertionError("cross-m tie selected the wrong exact score")
    if compare_exact_scores(
        cross_result.exact_score, ExactScore(1, 8)
    ) != 0:
        raise AssertionError("cross-m symbolic tie was not exact")
    if cross_result.worst_boundaries != (2, 6):
        raise AssertionError("cross-m canonical witness is wrong")

    base_problem = VariableNPairwiseProblem(
        token_count=7,
        original_boundaries=(2, 5),
        radius=2,
        span_allowed=lambda start, end: True,
        pair_hit=lambda left, middle, right: int(
            (left * 31 + middle * 17 + right * 13) % 7 < 3
        ),
    )
    base_result = exact_variable_n_pairwise_certificate(base_problem)
    tied_problem = VariableNPairwiseProblem(
        token_count=base_problem.token_count,
        original_boundaries=base_problem.original_boundaries,
        radius=base_problem.radius,
        span_allowed=base_problem.span_allowed,
        pair_hit=base_problem.pair_hit,
        threshold=base_result.exact_score,
    )
    if (
        exact_variable_n_pairwise_certificate(tied_problem).decision
        != NOT_CERTIFIED
    ):
        raise AssertionError("strict threshold tie was certified")

    radius_zero_cases = 100
    monotonicity_cases = 200
    rng = random.Random(SEED + 2)
    for case_index in range(monotonicity_cases):
        token_count = rng.randint(3, MAX_TOKEN_COUNT)
        original_count = rng.randint(1, token_count - 1)
        original = tuple(sorted(rng.sample(
            range(1, token_count), original_count
        )))
        salt = rng.randrange(1_000_000)

        def pair_hit(left, middle, right, value=salt):
            return int(
                (left * 31 + middle * 17 + right * 13 + value) % 7
                < 3
            )

        results = [
            exact_variable_n_pairwise_certificate(
                VariableNPairwiseProblem(
                    token_count=token_count,
                    original_boundaries=original,
                    radius=radius,
                    span_allowed=lambda start, end: True,
                    pair_hit=pair_hit,
                )
            )
            for radius in range(4)
        ]
        for left, right in zip(results, results[1:]):
            if compare_exact_scores(
                left.exact_score, right.exact_score
            ) < 0:
                raise AssertionError("radius monotonicity failed")
        if case_index < radius_zero_cases:
            if (
                results[0].exact_score
                != results[0].original_exact_score
                or results[0].worst_boundaries != original
                or results[0].threat_set_size != 1
            ):
                raise AssertionError("radius zero differs from clean score")

    class_problem = VariableNPairwiseProblem(
        token_count=7,
        original_boundaries=(2, 4),
        radius=2,
        span_allowed=lambda start, end: True,
        pair_hit=lambda left, middle, right: 0,
    )
    class_result = exact_variable_n_pairwise_certificate(class_problem)
    operation_pairs = {
        (insertions, deletions)
        for insertions, deletions, count
        in class_result.histogram_by_insertions_deletions
        if count
    }
    if not {
        (0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2)
    }.issubset(operation_pairs):
        raise AssertionError("operation classes are incomplete")

    two_unit = exact_variable_n_pairwise_certificate(
        VariableNPairwiseProblem(
            token_count=5,
            original_boundaries=(2,),
            radius=1,
            span_allowed=lambda start, end: True,
            pair_hit=lambda left, middle, right: 0,
        )
    )
    if 0 in dict(two_unit.histogram_by_edge_count):
        raise AssertionError("zero-edge candidate entered two-unit threat")

    original_spans = {(0, 2), (2, 4), (4, 6)}
    illegal_problem = VariableNPairwiseProblem(
        token_count=6,
        original_boundaries=(2, 4),
        radius=1,
        span_allowed=lambda start, end: (
            (start, end) in original_spans or end - start <= 2
        ),
        pair_hit=lambda left, middle, right: 0,
    )
    illegal_result = compare_dp_and_oracle(illegal_problem)
    if illegal_result.threat_set_size >= 12:
        raise AssertionError("illegal-span filter did not remove members")

    replay_problem = VariableNPairwiseProblem(
        token_count=9,
        original_boundaries=(2, 5, 7),
        radius=3,
        span_allowed=lambda start, end: end - start <= 6,
        pair_hit=lambda left, middle, right: int(
            (left * 31 + middle * 17 + right * 13) % 7 < 3
        ),
        threshold=ExactScore(1, 4),
    )
    first_replay = canonical_bytes(
        exact_variable_n_pairwise_certificate(replay_problem).to_dict()
    )
    second_replay = canonical_bytes(
        exact_variable_n_pairwise_certificate(replay_problem).to_dict()
    )
    if first_replay != second_replay:
        raise AssertionError("single-problem replay differs")

    failure_codes = {}
    invalid_problems = {
        "nonboolean_legality": (
            VariableNPairwiseProblem(
                5, (2, 4), 1,
                lambda start, end: 1,
                lambda left, middle, right: 0,
            ),
            "NONBOOLEAN_SPAN_LEGALITY",
        ),
        "nonbinary_hit": (
            VariableNPairwiseProblem(
                5, (2, 4), 1,
                lambda start, end: True,
                lambda left, middle, right: 2,
            ),
            "NONBINARY_PAIR_HIT",
        ),
        "missing_hit": (
            VariableNPairwiseProblem(
                5, (2, 4), 1,
                lambda start, end: True,
                lambda left, middle, right: {}[
                    (left, middle, right)
                ],
            ),
            "MISSING_PAIR_HIT",
        ),
    }
    for name, (problem, expected_code) in invalid_problems.items():
        try:
            exact_variable_n_pairwise_certificate(problem)
        except InvalidOrIncompleteError as exc:
            if exc.code != expected_code:
                raise AssertionError(
                    f"{name} produced {exc.code}, expected {expected_code}"
                ) from exc
            failure_codes[name] = exc.code
        else:
            raise AssertionError(f"{name} did not fail closed")

    return {
        "cross_m_exact_tie": True,
        "cross_m_tied_scores": [
            ExactScore(0, 2).to_dict(),
            ExactScore(1, 8).to_dict(),
        ],
        "cross_m_canonical_witness": list(
            cross_result.worst_boundaries
        ),
        "strict_threshold_tie": True,
        "radius_zero_cases": radius_zero_cases,
        "radius_zero_equals_clean": True,
        "radius_monotonicity_cases": monotonicity_cases,
        "radius_monotonicity": True,
        "operation_classes": True,
        "two_unit_case": True,
        "illegal_span_filtering": True,
        "single_problem_replay": True,
        "fail_closed_codes": failure_codes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    first_correctness = run_correctness_suite()
    first_reduction = run_fixed_reduction_suite()
    second_correctness = run_correctness_suite()
    second_reduction = run_fixed_reduction_suite()
    if first_correctness != second_correctness:
        raise AssertionError("full correctness replay differs")
    if first_reduction != second_reduction:
        raise AssertionError("fixed-N reduction replay differs")
    specialized = run_specialized_gates()
    elapsed = time.perf_counter() - started

    output = {
        "schema_version": 1,
        "project_id": "semstamp_certified_defense_v1",
        "stage": "WP2",
        "audit": "algorithm1_pure_cpu_correctness",
        "created_on": "2026-10-08",
        "status": "pass",
        "parameters": {
            "seed": SEED,
            "systematic_cases": SYSTEMATIC_CASES,
            "random_cases": RANDOM_CASES,
            "fixed_n_reduction_cases": FIXED_REDUCTION_CASES,
            "maximum_synthetic_atom_count": MAX_TOKEN_COUNT,
            "oracle_combination_limit": ORACLE_COMBINATION_LIMIT,
            "full_replays": 2,
        },
        "correctness": first_correctness,
        "fixed_n_reduction": first_reduction,
        "specialized_gates": specialized,
        "replay": {
            "correctness_semantic_sha256": (
                first_correctness["semantic_sha256"]
            ),
            "fixed_n_semantic_sha256": (
                first_reduction["semantic_sha256"]
            ),
            "byte_identical_semantic_summaries": True,
        },
        "zero_model_attestation": {
            "model_loads": 0,
            "gpu_calls": 0,
            "remote_calls": 0,
            "network_downloads": 0,
            "sealed_examples_read": 0,
            "generated_text_examples": 0,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "runtime_seconds": elapsed,
        "gates": {
            "systematic_minimum_met": (
                first_correctness["systematic_cases"] >= 5_000
            ),
            "random_minimum_met": (
                first_correctness["random_cases"] >= 2_000
            ),
            "fixed_n_minimum_met": (
                first_reduction["cases"] >= 1_000
            ),
            "dp_oracle_exact_agreement": True,
            "fixed_n_reduction_agreement": True,
            "specialized_gates_pass": True,
            "full_replay_match": True,
            "zero_model_boundary_respected": True,
        },
        "terminal": "WP2_ALGORITHM_GO_FOR_WP3_DESIGN",
        "wp3_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(output) + b"\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
