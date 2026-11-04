"""Run one frozen v2 WP2 refined-atom pure-CPU correctness replay."""

import argparse
import hashlib
from itertools import combinations
import json
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
    compare_exact_scores,
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


SEED = 20261008
SYSTEMATIC_CASES = 5_000
RANDOM_CASES = 2_000
FIXED_REDUCTION_CASES = 1_000
MAX_ATOM_COUNT = 10
ORACLE_COMBINATION_LIMIT = 1_000_000
AUDIT_SAMPLE_CASES = 64
SYSTEMATIC_SAMPLE_INDICES = frozenset(
    (index * 157) % SYSTEMATIC_CASES for index in range(32)
)
RANDOM_SAMPLE_INDICES = frozenset(
    (index * 61) % RANDOM_CASES for index in range(32)
)


def canonical_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_value(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def atomic_write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_bytes(value) + b"\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def threshold_for(index):
    thresholds = (
        ExactScore(0, 1),
        ExactScore(1, 4),
        ExactScore(1, 3),
        ExactScore(1, 2),
    )
    return thresholds[index % len(thresholds)]


def gap_lengths_for(atom_count, original, case_index):
    lengths = [
        1 + ((boundary * 17 + case_index * 13) % 4)
        for boundary in range(1, atom_count)
    ]
    for boundary in original:
        if (case_index + boundary) % 3 == 0:
            lengths[boundary - 1] = 0
    return tuple(lengths)


def make_systematic_problem(
    atom_count,
    original,
    radius,
    legality_profile,
    hit_profile,
    case_index,
):
    endpoints = (0, *original, atom_count)
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
                or (start * 17 + end * 31 + atom_count) % 5 != 0
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

    return RefinedVariableNPairwiseProblem(
        atom_count=atom_count,
        original_boundaries=original,
        gap_byte_lengths=gap_lengths_for(
            atom_count, original, case_index
        ),
        radius=radius,
        span_allowed=span_allowed,
        pair_hit=pair_hit,
        threshold=threshold_for(case_index),
    )


def systematic_problems():
    case_index = 0
    for atom_count in range(2, MAX_ATOM_COUNT + 1):
        gaps = tuple(range(1, atom_count))
        for original_count in range(
            1, min(4, atom_count - 1) + 1
        ):
            for original in combinations(gaps, original_count):
                for radius in range(4):
                    for legality_profile in range(3):
                        for hit_profile in range(4):
                            yield make_systematic_problem(
                                atom_count,
                                original,
                                radius,
                                legality_profile,
                                hit_profile,
                                case_index,
                            )
                            case_index += 1


def random_problem(rng, case_index):
    atom_count = rng.randint(2, MAX_ATOM_COUNT)
    original_count = rng.randint(1, atom_count - 1)
    original = tuple(sorted(rng.sample(
        range(1, atom_count), original_count
    )))
    endpoints = (0, *original, atom_count)
    original_spans = set(zip(endpoints, endpoints[1:]))
    legal = {
        (start, end): (
            (start, end) in original_spans or rng.random() >= 0.30
        )
        for start in range(atom_count)
        for end in range(start + 1, atom_count + 1)
    }
    salt = rng.randrange(10_000_000)
    modulus = rng.choice((5, 7, 11, 13))
    cutoff = rng.randint(1, modulus - 1)
    return RefinedVariableNPairwiseProblem(
        atom_count=atom_count,
        original_boundaries=original,
        gap_byte_lengths=gap_lengths_for(
            atom_count, original, case_index
        ),
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
    exact = exact_refined_variable_n_pairwise_certificate(problem)
    oracle = brute_force_refined_variable_n_pairwise_certificate(
        problem,
        combination_limit=ORACLE_COMBINATION_LIMIT,
    )
    if exact.semantic_payload() != oracle.semantic_payload():
        raise AssertionError(
            "refined variable-N DP differs from combinations oracle"
        )
    if exact.certified_score != oracle.certified_score:
        raise AssertionError("reported float score differs from oracle")
    if exact.score_access_count_during_construction != 0:
        raise AssertionError("DP read score during topology construction")
    if oracle.score_access_count_during_construction != 0:
        raise AssertionError(
            "oracle read score during topology construction"
        )
    if not verify_refined_result_witness(problem, exact):
        raise AssertionError("refined witness verification failed")
    return exact


def disclosed_descriptor(case_id, problem, result):
    allowed_spans = [
        [start, end]
        for start in range(problem.atom_count)
        for end in range(start + 1, problem.atom_count + 1)
        if problem.span_allowed(start, end)
    ]
    hit_triples = [
        [left, middle, right]
        for left in range(problem.atom_count)
        for middle in range(left + 1, problem.atom_count)
        for right in range(middle + 1, problem.atom_count + 1)
        if problem.pair_hit(left, middle, right)
    ]
    return {
        "case_id": case_id,
        "atom_count": problem.atom_count,
        "original_boundaries": list(problem.original_boundaries),
        "gap_byte_lengths": list(problem.gap_byte_lengths),
        "radius": problem.radius,
        "allowed_spans": allowed_spans,
        "hit_triples": hit_triples,
        "threshold": (
            None
            if problem.threshold is None
            else problem.threshold.to_dict()
        ),
        "expected": result.semantic_payload(),
    }


def update_summary(summary, result):
    summary["total_threat_members"] += result.threat_set_size
    summary["maximum_reachable_states"] = max(
        summary["maximum_reachable_states"],
        result.reachable_state_count,
    )
    summary["maximum_transitions_evaluated"] = max(
        summary["maximum_transitions_evaluated"],
        result.transitions_evaluated,
    )
    summary["maximum_unique_pair_hits"] = max(
        summary["maximum_unique_pair_hits"],
        result.unique_pair_hits,
    )
    summary["zero_gap_cases"] += int(
        bool(result.zero_byte_gap_boundaries)
    )
    summary["operation_pairs"].update(
        (insertions, deletions)
        for insertions, deletions, count
        in result.histogram_by_insertions_deletions
        if count
    )


def run_correctness_suite():
    digest = hashlib.sha256()
    samples = []
    summary = {
        "total_threat_members": 0,
        "maximum_reachable_states": 0,
        "maximum_transitions_evaluated": 0,
        "maximum_unique_pair_hits": 0,
        "zero_gap_cases": 0,
        "operation_pairs": set(),
    }

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
        update_summary(summary, result)
        if case_index in SYSTEMATIC_SAMPLE_INDICES:
            samples.append(disclosed_descriptor(
                f"systematic-{case_index}", problem, result
            ))

    rng = random.Random(SEED)
    for random_index in range(RANDOM_CASES):
        problem = random_problem(
            rng, SYSTEMATIC_CASES + random_index
        )
        result = compare_dp_and_oracle(problem)
        digest.update(canonical_bytes(result.semantic_payload()) + b"\n")
        update_summary(summary, result)
        if random_index in RANDOM_SAMPLE_INDICES:
            samples.append(disclosed_descriptor(
                f"random-{random_index}", problem, result
            ))

    required_pairs = {(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2)}
    if not required_pairs.issubset(summary["operation_pairs"]):
        raise AssertionError("operation-class coverage is incomplete")
    if len(samples) != AUDIT_SAMPLE_CASES:
        raise AssertionError("independent audit sample count is wrong")
    if summary["zero_gap_cases"] == 0:
        raise AssertionError("correctness suite covered no zero-gap cases")

    return {
        "systematic_cases": SYSTEMATIC_CASES,
        "random_cases": RANDOM_CASES,
        "total_cases": SYSTEMATIC_CASES + RANDOM_CASES,
        "total_threat_members": summary["total_threat_members"],
        "maximum_reachable_states": (
            summary["maximum_reachable_states"]
        ),
        "maximum_transitions_evaluated": (
            summary["maximum_transitions_evaluated"]
        ),
        "maximum_unique_pair_hits": summary["maximum_unique_pair_hits"],
        "zero_gap_cases": summary["zero_gap_cases"],
        "operation_pairs_observed": [
            [insertions, deletions]
            for insertions, deletions
            in sorted(summary["operation_pairs"])
        ],
        "semantic_sha256": digest.hexdigest(),
        "score_match": True,
        "exact_score_match": True,
        "decision_match": True,
        "witness_match": True,
        "threat_set_size_match": True,
        "all_histograms_match": True,
        "gap_metadata_match": True,
        "witness_verification": True,
        "score_blind_topology": True,
    }, samples


def run_fixed_reduction_suite():
    rng = random.Random(SEED + 1)
    digest = hashlib.sha256()
    maximum_threat_size = 0
    zero_gap_cases = 0
    for case_index in range(FIXED_REDUCTION_CASES):
        atom_count = rng.randint(2, MAX_ATOM_COUNT)
        original_count = rng.randint(1, atom_count - 1)
        original = tuple(sorted(rng.sample(
            range(1, atom_count), original_count
        )))
        endpoints = (0, *original, atom_count)
        original_spans = set(zip(endpoints, endpoints[1:]))
        legal = {
            (start, end): (
                (start, end) in original_spans
                or rng.random() >= 0.25
            )
            for start in range(atom_count)
            for end in range(start + 1, atom_count + 1)
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

        problem = RefinedVariableNPairwiseProblem(
            atom_count=atom_count,
            original_boundaries=original,
            gap_byte_lengths=gap_lengths_for(
                atom_count, original, case_index
            ),
            radius=radius,
            span_allowed=span_allowed,
            pair_hit=pair_hit,
            threshold=ExactScore(threshold_hits, original_count),
        )
        reduced = exact_refined_fixed_count_reduction_certificate(
            problem
        )
        existing = exact_sparse_pairwise_certificate(
            SparsePairwiseProblem(
                token_count=atom_count,
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
        if reduced.original_exact_score.hits != int(existing.original_score):
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
        zero_gap_cases += int(bool(reduced.zero_byte_gap_boundaries))

    if zero_gap_cases == 0:
        raise AssertionError("fixed-N suite covered no zero-gap cases")
    return {
        "cases": FIXED_REDUCTION_CASES,
        "zero_gap_cases": zero_gap_cases,
        "minimum_hits_match": True,
        "original_hits_match": True,
        "witness_match": True,
        "threat_set_size_match": True,
        "move_histogram_match": True,
        "strict_decision_match": True,
        "maximum_threat_set_size": maximum_threat_size,
        "semantic_sha256": digest.hexdigest(),
    }


def refined_problem(
    atom_count,
    original,
    radius,
    span_allowed,
    pair_hit,
    threshold=None,
    zero_boundaries=(),
):
    lengths = [1] * (atom_count - 1)
    for boundary in zero_boundaries:
        lengths[boundary - 1] = 0
    return RefinedVariableNPairwiseProblem(
        atom_count=atom_count,
        original_boundaries=original,
        gap_byte_lengths=tuple(lengths),
        radius=radius,
        span_allowed=span_allowed,
        pair_hit=pair_hit,
        threshold=threshold,
    )


def cross_m_tie_problem():
    atom_count = 9
    original = (2, 4, 6, 7, 8)
    tied_low = (2, 6)
    tied_high = tuple(range(1, atom_count))
    legal_spans = set()
    for boundaries in (original, tied_low, tied_high):
        endpoints = (0, *boundaries, atom_count)
        legal_spans.update(zip(endpoints, endpoints[1:]))
    hit_triples = {
        (0, 1, 2),
        (0, 2, 3),
        (2, 4, 5),
        (2, 4, 6),
        (2, 6, 7),
    }
    return refined_problem(
        atom_count,
        original,
        3,
        lambda start, end: (start, end) in legal_spans,
        lambda left, middle, right: int(
            (left, middle, right) in hit_triples
        ),
        zero_boundaries=(4,),
    )


def manual_two_atom_document(gap, original_boundaries=(1,)):
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
        original_unit_codepoint_spans=(
            (0, 1), (second_start, second_start + 1)
        ),
        original_unit_atom_spans=((0, 1), (1, 2)),
        original_boundaries=original_boundaries,
    )


def expect_failure(problem, expected_code):
    try:
        exact_refined_variable_n_pairwise_certificate(problem)
    except InvalidOrIncompleteError as exc:
        if exc.code != expected_code:
            raise AssertionError(
                f"got {exc.code}, expected {expected_code}"
            ) from exc
        return exc.code
    raise AssertionError(f"{expected_code} did not fail closed")


def run_specialized_gates():
    cross_result = compare_dp_and_oracle(cross_m_tie_problem())
    if cross_result.exact_score != ExactScore(0, 2):
        raise AssertionError("cross-m tie selected wrong score")
    if compare_exact_scores(
        cross_result.exact_score, ExactScore(1, 8)
    ) != 0:
        raise AssertionError("cross-m exact tie was not preserved")
    if cross_result.worst_boundaries != (2, 6):
        raise AssertionError("cross-m canonical witness is wrong")

    base_problem = refined_problem(
        7,
        (2, 5),
        2,
        lambda start, end: True,
        lambda left, middle, right: int(
            (left * 31 + middle * 17 + right * 13) % 7 < 3
        ),
    )
    base_result = exact_refined_variable_n_pairwise_certificate(
        base_problem
    )
    tied_problem = RefinedVariableNPairwiseProblem(
        atom_count=base_problem.atom_count,
        original_boundaries=base_problem.original_boundaries,
        gap_byte_lengths=base_problem.gap_byte_lengths,
        radius=base_problem.radius,
        span_allowed=base_problem.span_allowed,
        pair_hit=base_problem.pair_hit,
        threshold=base_result.exact_score,
    )
    if (
        exact_refined_variable_n_pairwise_certificate(
            tied_problem
        ).decision
        != NOT_CERTIFIED
    ):
        raise AssertionError("strict threshold tie was certified")

    radius_zero_cases = 100
    monotonicity_cases = 200
    rng = random.Random(SEED + 2)
    for case_index in range(monotonicity_cases):
        atom_count = rng.randint(3, MAX_ATOM_COUNT)
        original_count = rng.randint(1, atom_count - 1)
        original = tuple(sorted(rng.sample(
            range(1, atom_count), original_count
        )))
        salt = rng.randrange(1_000_000)
        lengths = gap_lengths_for(atom_count, original, case_index)

        def pair_hit(left, middle, right, value=salt):
            return int(
                (left * 31 + middle * 17 + right * 13 + value) % 7
                < 3
            )

        results = [
            exact_refined_variable_n_pairwise_certificate(
                RefinedVariableNPairwiseProblem(
                    atom_count=atom_count,
                    original_boundaries=original,
                    gap_byte_lengths=lengths,
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
                raise AssertionError("radius zero differs from clean")

    class_result = exact_refined_variable_n_pairwise_certificate(
        refined_problem(
            7, (2, 4), 2,
            lambda start, end: True,
            lambda left, middle, right: 0,
            zero_boundaries=(2,),
        )
    )
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

    two_unit = exact_refined_variable_n_pairwise_certificate(
        refined_problem(
            5, (2,), 1,
            lambda start, end: True,
            lambda left, middle, right: 0,
            zero_boundaries=(2,),
        )
    )
    if 0 in dict(two_unit.histogram_by_edge_count):
        raise AssertionError("zero-edge candidate entered threat set")

    original_spans = {(0, 2), (2, 4), (4, 6)}
    illegal_result = compare_dp_and_oracle(refined_problem(
        6,
        (2, 4),
        1,
        lambda start, end: (
            (start, end) in original_spans or end - start <= 2
        ),
        lambda left, middle, right: 0,
        zero_boundaries=(2,),
    ))
    if illegal_result.threat_set_size >= 12:
        raise AssertionError("illegal-span filter removed no members")

    replay_problem = refined_problem(
        9,
        (2, 5, 7),
        3,
        lambda start, end: end - start <= 6,
        lambda left, middle, right: int(
            (left * 31 + middle * 17 + right * 13) % 7 < 3
        ),
        threshold=ExactScore(1, 4),
        zero_boundaries=(5,),
    )
    first = canonical_bytes(
        exact_refined_variable_n_pairwise_certificate(
            replay_problem
        ).to_dict()
    )
    second = canonical_bytes(
        exact_refined_variable_n_pairwise_certificate(
            replay_problem
        ).to_dict()
    )
    if first != second:
        raise AssertionError("single-problem replay differs")

    deletion_legal = {(0, 1), (1, 3), (3, 5), (0, 3)}
    deletion_problem = refined_problem(
        5,
        (1, 3),
        1,
        lambda start, end: (start, end) in deletion_legal,
        lambda left, middle, right: int(
            (left, middle, right) != (0, 3, 5)
        ),
        zero_boundaries=(1,),
    )
    identity_result = exact_refined_variable_n_pairwise_certificate(
        RefinedVariableNPairwiseProblem(
            atom_count=deletion_problem.atom_count,
            original_boundaries=deletion_problem.original_boundaries,
            gap_byte_lengths=deletion_problem.gap_byte_lengths,
            radius=0,
            span_allowed=deletion_problem.span_allowed,
            pair_hit=deletion_problem.pair_hit,
        )
    )
    deletion_result = compare_dp_and_oracle(deletion_problem)
    if (
        identity_result.worst_boundaries != (1, 3)
        or deletion_result.worst_boundaries != (3,)
        or (deletion_result.insertions, deletion_result.deletions)
        != (0, 1)
    ):
        raise AssertionError("zero-gap identity/deletion gate failed")

    relocation_legal = {
        (0, 1), (1, 3), (3, 5),
        (0, 2), (2, 3), (1, 2),
    }
    target_hits = {(0, 2, 3), (2, 3, 5)}
    relocation_result = compare_dp_and_oracle(refined_problem(
        5,
        (1, 3),
        1,
        lambda start, end: (start, end) in relocation_legal,
        lambda left, middle, right: int(
            (left, middle, right) not in target_hits
        ),
        zero_boundaries=(1,),
    ))
    if (
        relocation_result.worst_boundaries != (2, 3)
        or (relocation_result.insertions, relocation_result.deletions)
        != (1, 1)
    ):
        raise AssertionError("zero-gap relocation gate failed")

    span_calls = []
    hit_calls = []
    failure_codes = {}
    malformed = RefinedVariableNPairwiseProblem(
        atom_count=5,
        original_boundaries=(2, 4),
        gap_byte_lengths=(1, 0),
        radius=1,
        span_allowed=lambda start, end: span_calls.append(
            (start, end)
        ) or True,
        pair_hit=lambda left, middle, right: hit_calls.append(
            (left, middle, right)
        ) or 0,
    )
    failure_codes["gap_count"] = expect_failure(
        malformed, "REFINED_GAP_COUNT_MISMATCH"
    )
    failure_codes["negative_gap"] = expect_failure(
        refined_problem(
            5, (2, 4), 1,
            lambda start, end: True,
            lambda left, middle, right: 0,
        ).__class__(
            5, (2, 4), (1, 0, 1, -1), 1,
            lambda start, end: True,
            lambda left, middle, right: 0,
        ),
        "INVALID_GAP_BYTE_LENGTH",
    )
    failure_codes["boolean_gap"] = expect_failure(
        RefinedVariableNPairwiseProblem(
            5, (2, 4), (1, 0, 1, True), 1,
            lambda start, end: True,
            lambda left, middle, right: 0,
        ),
        "INVALID_GAP_BYTE_LENGTH",
    )
    failure_codes["unauthorized_zero_gap"] = expect_failure(
        RefinedVariableNPairwiseProblem(
            5, (2, 4), (0, 1, 1, 1), 1,
            lambda start, end: True,
            lambda left, middle, right: 0,
        ),
        "UNAUTHORIZED_ZERO_BYTE_GAP",
    )
    if span_calls or hit_calls:
        raise AssertionError("metadata validation called a callback")

    bridge_document = canonicalize_refined_bytes(
        b"Alpha ends.Next begins.",
        lambda _text: ["Alpha ends.", "Next begins."],
    )
    bridge_problem = refined_problem_from_document(
        bridge_document,
        radius=1,
        span_allowed=lambda start, end: True,
        pair_hit=lambda left, middle, right: 0,
    )
    bridge_result = exact_refined_variable_n_pairwise_certificate(
        bridge_problem
    )
    if (
        bridge_problem.gap_byte_lengths != (1, 0, 1)
        or bridge_result.zero_byte_gap_boundaries != (2,)
    ):
        raise AssertionError("renderer bridge changed refined topology")

    try:
        refined_problem_from_document(
            manual_two_atom_document(b"-"),
            radius=0,
            span_allowed=lambda start, end: True,
            pair_hit=lambda left, middle, right: 0,
        )
    except InvalidOrIncompleteError as exc:
        if exc.code != "NONWHITESPACE_REFINED_GAP":
            raise
        failure_codes["nonwhitespace_gap"] = exc.code
    else:
        raise AssertionError("bridge accepted non-whitespace gap")

    try:
        refined_problem_from_document(
            manual_two_atom_document(b"", original_boundaries=()),
            radius=0,
            span_allowed=lambda start, end: True,
            pair_hit=lambda left, middle, right: 0,
        )
    except InvalidOrIncompleteError as exc:
        if exc.code != "UNAUTHORIZED_ZERO_BYTE_GAP":
            raise
        failure_codes["bridge_unauthorized_zero_gap"] = exc.code
    else:
        raise AssertionError("bridge accepted unauthorized zero gap")

    return {
        "cross_m_exact_tie": True,
        "cross_m_canonical_witness": [2, 6],
        "strict_threshold_tie": True,
        "radius_zero_cases": radius_zero_cases,
        "radius_zero_equals_clean": True,
        "radius_monotonicity_cases": monotonicity_cases,
        "radius_monotonicity": True,
        "operation_classes": True,
        "two_unit_case": True,
        "illegal_span_filtering": True,
        "single_problem_replay": True,
        "zero_gap_identity": True,
        "zero_gap_deletion": True,
        "zero_gap_deletion_witness": [3],
        "zero_gap_relocation": True,
        "zero_gap_relocation_witness": [2, 3],
        "metadata_validation_before_callbacks": True,
        "renderer_to_solver_bridge": True,
        "failure_codes": failure_codes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-id", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    correctness, audit_samples = run_correctness_suite()
    fixed_reduction = run_fixed_reduction_suite()
    specialized = run_specialized_gates()
    audit_sample = {
        "case_count": len(audit_samples),
        "cases": audit_samples,
        "cases_sha256": sha256_value(audit_samples),
    }
    semantic = {
        "correctness": correctness,
        "fixed_n_reduction": fixed_reduction,
        "specialized_gates": specialized,
        "independent_audit_sample": audit_sample,
    }
    output = {
        "schema_version": 1,
        "project_id": "semstamp_certified_defense_v2",
        "stage": "WP2",
        "audit": "refined_atom_algorithm1_pure_cpu_correctness",
        "created_on": "2026-10-08",
        "status": "pass",
        "replay_id": args.replay_id,
        "parameters": {
            "seed": SEED,
            "systematic_cases": SYSTEMATIC_CASES,
            "random_cases": RANDOM_CASES,
            "fixed_n_reduction_cases": FIXED_REDUCTION_CASES,
            "maximum_synthetic_atom_count": MAX_ATOM_COUNT,
            "oracle_combination_limit": ORACLE_COMBINATION_LIMIT,
            "independent_disclosed_sample": AUDIT_SAMPLE_CASES,
        },
        "semantic": semantic,
        "semantic_sha256": sha256_value(semantic),
        "zero_model_attestation": {
            "model_loads": 0,
            "gpu_calls": 0,
            "remote_calls": 0,
            "network_downloads": 0,
            "sealed_examples_read": 0,
            "detector_score_reads": 0,
            "generated_text_examples": 0,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "runtime_seconds": time.perf_counter() - started,
        "wp3_authorized": False,
    }
    atomic_write_json(args.output, output)
    print(json.dumps({
        "output": str(args.output),
        "replay_id": args.replay_id,
        "semantic_sha256": output["semantic_sha256"],
        "systematic_cases": correctness["systematic_cases"],
        "random_cases": correctness["random_cases"],
        "fixed_n_reduction_cases": fixed_reduction["cases"],
        "zero_gap_cases": correctness["zero_gap_cases"],
        "runtime_seconds": output["runtime_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
