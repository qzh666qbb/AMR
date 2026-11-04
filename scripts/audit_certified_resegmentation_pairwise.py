"""Run deterministic WP1b pairwise checks and synthetic CPU benchmarks."""

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import platform
import random
import time

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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "certified_resegmentation_pairwise_preflight1"
    / "audit_results.json"
)
PER_RUN_LIMIT_SECONDS = 10.0


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_partitions(token_count, minimum, maximum):
    for count in range(1, token_count):
        for boundaries in combinations(range(1, token_count), count):
            endpoints = (0, *boundaries, token_count)
            if all(
                minimum <= endpoints[index + 1] - endpoints[index] <= maximum
                for index in range(len(endpoints) - 1)
            ):
                yield boundaries


def make_problem(
    token_count,
    original,
    minimum,
    maximum,
    radius,
    allowed,
    scorer,
    threshold,
    normalization=None,
):
    allowed = frozenset(allowed)
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


def score_partition(problem, boundaries):
    endpoints = (0, *boundaries, problem.token_count)
    pair_sum = sum(
        problem.pair_scores[
            (endpoints[index - 1], endpoints[index], endpoints[index + 1])
        ]
        for index in range(1, len(endpoints) - 1)
    )
    return problem.bias + pair_sum / problem.normalization


def compare(problem):
    dynamic = exact_pairwise_certificate(problem)
    oracle = brute_force_pairwise_certificate(problem)
    if abs(dynamic.certified_score - oracle.certified_score) > 1e-12:
        raise AssertionError("pairwise DP and oracle certified scores differ")
    if dynamic.exact_move_histogram != oracle.exact_move_histogram:
        raise AssertionError("pairwise DP and oracle move histograms differ")
    if dynamic.threat_set_size != oracle.threat_set_size:
        raise AssertionError("pairwise DP and oracle threat-set sizes differ")
    if dynamic.certified_positive != oracle.certified_positive:
        raise AssertionError("pairwise DP and oracle decisions differ")
    if abs(
        score_partition(problem, dynamic.worst_boundaries)
        - dynamic.certified_score
    ) > 1e-12:
        raise AssertionError("pairwise DP witness does not reproduce score")
    if boundary_moves(
        dynamic.worst_boundaries, dynamic.original_boundaries
    ) > dynamic.requested_radius:
        raise AssertionError("pairwise DP witness exceeds declared radius")
    return dynamic


def systematic_checks():
    comparisons = 0
    enumerated_threat_members = 0
    configurations = 0
    for token_count in range(2, 9):
        for minimum in range(1, min(3, token_count) + 1):
            for maximum in range(minimum, min(4, token_count) + 1):
                partitions = list(
                    valid_partitions(token_count, minimum, maximum)
                )
                for original in partitions:
                    full = frozenset(range(1, token_count))
                    parity = frozenset(
                        set(original)
                        | {
                            gap
                            for gap in range(1, token_count)
                            if gap % 2 == token_count % 2
                        }
                    )
                    lattices = {full, parity, frozenset(original)}
                    for allowed in lattices:
                        configurations += 1
                        for radius in range(len(original) + 1):
                            scorer = (
                                lambda start, middle, end, n=token_count: (
                                    11 * start
                                    + 17 * middle
                                    + 31 * end
                                    + 13 * n
                                ) % 23 - 11
                            )
                            problem = make_problem(
                                token_count,
                                original,
                                minimum,
                                maximum,
                                radius,
                                allowed,
                                scorer,
                                threshold=0.25,
                            )
                            result = compare(problem)
                            comparisons += 1
                            enumerated_threat_members += (
                                result.threat_set_size
                            )
    return {
        "token_count_range": [2, 8],
        "all_min_unit_words": [1, 2, 3],
        "all_max_unit_words_up_to": 4,
        "lattice_variants": [
            "all_internal_gaps",
            "original_plus_parity_gaps",
            "original_boundaries_only",
        ],
        "all_legal_multi_unit_original_partitions": True,
        "all_radii_for_each_original": True,
        "configuration_count": configurations,
        "dp_oracle_comparisons": comparisons,
        "oracle_threat_members_checked": enumerated_threat_members,
        "status": "pass",
    }


def random_checks(case_count=2000):
    rng = random.Random(20260922)
    comparisons = 0
    enumerated_threat_members = 0
    while comparisons < case_count:
        token_count = rng.randint(2, 12)
        minimum = rng.randint(1, min(4, token_count))
        maximum = rng.randint(minimum, min(token_count, minimum + 5))
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
            if rng.random() < 0.5
        )
        values = {}

        def scorer(start, middle, end):
            key = (start, middle, end)
            if key not in values:
                values[key] = rng.randint(-8, 8) / 4
            return values[key]

        problem = make_problem(
            token_count,
            original,
            minimum,
            maximum,
            rng.randint(0, len(original)),
            allowed,
            scorer,
            threshold=rng.randint(-4, 4) / 4,
            normalization=rng.randint(1, 8) / 4,
        )
        result = compare(problem)
        comparisons += 1
        enumerated_threat_members += result.threat_set_size
    return {
        "seed": 20260922,
        "token_count_range": [2, 12],
        "case_count": comparisons,
        "oracle_threat_members_checked": enumerated_threat_members,
        "tied_pair_scores": True,
        "random_positive_normalization": True,
        "status": "pass",
    }


def degeneracy_checks(case_count=1000):
    rng = random.Random(22092026)
    comparisons = 0
    while comparisons < case_count:
        token_count = rng.randint(2, 12)
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
            if rng.random() < 0.5
        )
        allowed = frozenset(allowed)
        values = {}

        def span_scorer(start, end):
            if start == 0:
                return 0
            key = (start, end)
            if key not in values:
                values[key] = rng.randint(-8, 8) / 4
            return values[key]

        def current_span_scorer(previous_start, start, end):
            del previous_start
            return span_scorer(start, end)

        radius = rng.randint(0, len(original))
        threshold = rng.randint(-4, 4) / 4
        span_scores = build_span_scores(
            token_count, minimum, maximum, span_scorer, allowed
        )
        pair_scores = build_pair_scores(
            token_count, minimum, maximum, current_span_scorer, allowed
        )
        additive = exact_certificate(CertificateProblem(
            token_count=token_count,
            original_boundaries=original,
            min_unit_words=minimum,
            max_unit_words=maximum,
            max_boundary_moves=radius,
            span_scores=span_scores,
            threshold=threshold,
            allowed_boundaries=allowed,
        ))
        pairwise = exact_pairwise_certificate(PairwiseCertificateProblem(
            token_count=token_count,
            original_boundaries=original,
            min_unit_words=minimum,
            max_unit_words=maximum,
            max_boundary_moves=radius,
            pair_scores=pair_scores,
            threshold=threshold,
            normalization=len(original) + 1,
            allowed_boundaries=allowed,
        ))
        if abs(additive.certified_score - pairwise.certified_score) > 1e-12:
            raise AssertionError("pairwise additive degeneracy score differs")
        if additive.exact_move_histogram != pairwise.exact_move_histogram:
            raise AssertionError(
                "pairwise additive degeneracy histogram differs"
            )
        if additive.certified_positive != pairwise.certified_positive:
            raise AssertionError(
                "pairwise additive degeneracy decision differs"
            )
        comparisons += 1
    return {
        "seed": 22092026,
        "case_count": comparisons,
        "pair_score": "q(h,i,j)=s(i,j)",
        "all_possible_first_span_scores": 0,
        "shared_normalization": "unit_count",
        "status": "pass",
    }


def synthetic_benchmarks():
    records = []
    observed_300_times = []
    for token_count in (100, 200, 300):
        original = tuple(range(20, token_count, 20))
        minimum = 5
        maximum = 35
        allowed = frozenset(range(1, token_count))

        def scorer(start, middle, end):
            previous_length = middle - start
            current_length = end - middle
            return (
                37 * start
                + 19 * middle
                + 11 * end
                + 13 * previous_length * current_length
            ) % 101 - 50

        precompute_started = time.perf_counter()
        pair_scores = build_pair_scores(
            token_count, minimum, maximum, scorer, allowed
        )
        precompute_elapsed = time.perf_counter() - precompute_started
        radius_records = []
        previous_score = None
        previous_size = None
        for radius in range(0, min(3, len(original)) + 1):
            problem = PairwiseCertificateProblem(
                token_count=token_count,
                original_boundaries=original,
                min_unit_words=minimum,
                max_unit_words=maximum,
                max_boundary_moves=radius,
                pair_scores=pair_scores,
                threshold=0.0,
                normalization=len(original),
                allowed_boundaries=allowed,
            )
            started = time.perf_counter()
            result = exact_pairwise_certificate(problem)
            elapsed = time.perf_counter() - started
            if (
                previous_score is not None
                and result.certified_score > previous_score + 1e-12
            ):
                raise AssertionError("pairwise score increased with radius")
            if (
                previous_size is not None
                and result.threat_set_size < previous_size
            ):
                raise AssertionError("pairwise threat set shrank with radius")
            if abs(
                score_partition(problem, result.worst_boundaries)
                - result.certified_score
            ) > 1e-12:
                raise AssertionError("benchmark witness score differs")
            previous_score = result.certified_score
            previous_size = result.threat_set_size
            if token_count == 300:
                observed_300_times.append(elapsed)
            radius_records.append({
                "radius": radius,
                "elapsed_seconds": elapsed,
                "performance_gate_seconds": (
                    PER_RUN_LIMIT_SECONDS if token_count == 300 else None
                ),
                "performance_gate_passed": (
                    elapsed < PER_RUN_LIMIT_SECONDS
                    if token_count == 300
                    else None
                ),
                "result": result.to_dict(),
            })
        records.append({
            "token_count": token_count,
            "unit_count": len(original) + 1,
            "min_unit_words": minimum,
            "max_unit_words": maximum,
            "precomputed_pair_score_count": len(pair_scores),
            "pair_score_precompute_seconds": precompute_elapsed,
            "radii": radius_records,
        })

    max_300_time = max(observed_300_times)
    return {
        "records": records,
        "performance_gate": {
            "token_count": 300,
            "per_run_limit_seconds": PER_RUN_LIMIT_SECONDS,
            "maximum_observed_seconds": max_300_time,
            "passed": max_300_time < PER_RUN_LIMIT_SECONDS,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    started = time.perf_counter()
    systematic = systematic_checks()
    random_result = random_checks()
    degeneracy = degeneracy_checks()
    benchmarks = synthetic_benchmarks()
    elapsed = time.perf_counter() - started

    source_paths = [
        ROOT / "scripts" / "certified_resegmentation.py",
        ROOT / "scripts" / "test_certified_resegmentation.py",
        ROOT / "scripts" / "test_certified_resegmentation_pairwise.py",
        ROOT / "scripts" / "audit_certified_resegmentation_pairwise.py",
        ROOT
        / "experiments"
        / "certified_resegmentation_pairwise_preflight1"
        / "PROTOCOL.md",
    ]
    status = (
        "pass"
        if benchmarks["performance_gate"]["passed"]
        else "stop_performance_gate_failed"
    )
    payload = {
        "schema_version": 1,
        "status": status,
        "scope": (
            "WP1b synthetic adjacent-pair detector correctness and "
            "CPU complexity only"
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in source_paths
        },
        "systematic_small_instances": systematic,
        "random_small_instances": random_result,
        "additive_degeneracy_checks": degeneracy,
        "synthetic_normal_length_benchmarks": benchmarks,
        "total_elapsed_seconds": elapsed,
        "vacuity_interpretation": {
            "implemented": [
                "threat_set_size",
                "exact_move_histogram",
                "max_realized_moves",
                "singleton_threat_set",
                "budget_reaches_requested_radius",
                "original_is_worst_case",
                "worst_case_drop",
                "certificate_margin",
                "reachable_state_count",
                "transitions_evaluated",
                "peak_layer_states",
            ],
            "real_detector_nonvacuity_tested": False,
        },
        "execution_attestation": {
            "provider_api_calls": 0,
            "watermark_generation_calls": 0,
            "llm_judge_calls": 0,
            "watermark_detector_calls": 0,
            "amr_parser_calls": 0,
            "s2match_calls": 0,
            "model_downloads": 0,
            "dataset_payload_downloads": 0,
            "external_gpu_jobs": 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "status": status,
        "systematic_comparisons": systematic["dp_oracle_comparisons"],
        "random_comparisons": random_result["case_count"],
        "degeneracy_comparisons": degeneracy["case_count"],
        "maximum_300_token_seconds": (
            benchmarks["performance_gate"]["maximum_observed_seconds"]
        ),
        "total_elapsed_seconds": elapsed,
    }, indent=2))
    if status != "pass":
        raise SystemExit("WP1b performance gate failed")


if __name__ == "__main__":
    main()
