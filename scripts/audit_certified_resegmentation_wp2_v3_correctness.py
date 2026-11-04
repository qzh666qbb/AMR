"""Audit WP2-v3 structured rendering and sparse-DP correctness."""

import argparse
from itertools import combinations
import json
import math
from pathlib import Path
import platform
import random
import re
import time

from certified_resegmentation_wp2_v3 import (
    SparsePairwiseProblem,
    brute_force_sparse_pairwise_certificate,
    canonicalize_structured,
    exact_sparse_pairwise_certificate,
)


SEED = 20260922
ADAPTER_CASES = 5_000
DP_CASES = 10_000
_ATOM_RE = re.compile(r"\S+")


def expected_join(document, boundaries):
    retained_original = set(boundaries) & set(
        document.original_boundaries
    )
    removed_positions = {
        document.boundary_cut_map[boundary][0]
        for boundary in retained_original
    }
    return "".join(
        character
        for index, character in enumerate(document.structural_tape)
        if index not in removed_positions
    )


def audit_adapter(rng):
    whitespace = ("", " ", "  ", "\t", "\n", " \t")
    candidate_partitions = 0
    max_atoms = 0
    max_units = 0
    for case_index in range(ADAPTER_CASES):
        units = []
        for unit_index in range(rng.randint(2, 8)):
            atoms = [
                f"c{case_index}u{unit_index}a{atom_index}"
                for atom_index in range(rng.randint(1, 7))
            ]
            pieces = [rng.choice(whitespace), atoms[0]]
            for atom in atoms[1:]:
                pieces.extend((rng.choice(whitespace[1:]), atom))
            pieces.append(rng.choice(whitespace))
            units.append("".join(pieces))

        document = canonicalize_structured(units)
        max_atoms = max(max_atoms, document.atom_count)
        max_units = max(max_units, document.unit_count)
        if document.render_partition(
            document.original_boundaries
        ) != tuple(units):
            raise AssertionError("k=0 unit round-trip failed")

        candidates = {document.original_boundaries}
        for _ in range(3):
            candidates.add(tuple(sorted(rng.sample(
                range(1, document.atom_count),
                document.unit_count - 1,
            ))))
        for boundaries in sorted(candidates):
            rendered = document.render_partition(boundaries)
            candidate_partitions += 1
            if document.partition_atoms(boundaries) != document.atom_payloads:
                raise AssertionError("candidate atom sequence changed")
            if "".join(rendered) != expected_join(document, boundaries):
                raise AssertionError(
                    "virtual separator rendering was not exact"
                )
            for system_id in ("semstamp", "simmark"):
                preprocessed = document.preprocess_partition(
                    system_id, boundaries
                )
                if len(preprocessed) != document.unit_count:
                    raise AssertionError("preprocessing changed unit count")
                if system_id == "semstamp" and preprocessed != rendered:
                    raise AssertionError("SemStamp preprocessing changed text")
                if (
                    system_id == "simmark"
                    and preprocessed[0] != rendered[0]
                ):
                    raise AssertionError("SimMark changed the first unit")
                if (
                    system_id == "simmark"
                    and any(
                        observed != " " + expected
                        for observed, expected in zip(
                            preprocessed[1:], rendered[1:]
                        )
                    )
                ):
                    raise AssertionError("SimMark prefix mismatch")
    return {
        "documents": ADAPTER_CASES,
        "candidate_partitions": candidate_partitions,
        "max_atoms": max_atoms,
        "max_units": max_units,
        "k0_exact": True,
        "atom_sequence_exact": True,
        "virtual_separator_exact": True,
        "preprocessing_exact": True,
    }


def score_function(left, middle, right):
    return float(
        (
            (left + 3) * 1009
            + (middle + 5) * 9176
            + (right + 7) * 65537
        )
        % 1000003
    )


def compare_problem(problem):
    exact = exact_sparse_pairwise_certificate(problem)
    oracle = brute_force_sparse_pairwise_certificate(problem)
    score_error = abs(exact.certified_score - oracle.certified_score)
    original_error = abs(exact.original_score - oracle.original_score)
    if score_error > 1e-12 or original_error > 1e-12:
        raise AssertionError("sparse DP score differs from oracle")
    if exact.exact_move_histogram != oracle.exact_move_histogram:
        raise AssertionError("sparse DP histogram differs from oracle")
    if exact.threat_set_size != oracle.threat_set_size:
        raise AssertionError("sparse DP threat size differs from oracle")
    if exact.worst_boundaries != oracle.worst_boundaries:
        raise AssertionError("sparse DP witness differs from oracle")
    return exact, max(score_error, original_error)


def systematic_problems():
    for atom_count in range(4, 10):
        gaps = tuple(range(1, atom_count))
        for boundary_count in range(
            1, min(4, atom_count - 1) + 1
        ):
            for original in combinations(gaps, boundary_count):
                endpoints = (0, *original, atom_count)
                original_max = max(
                    endpoints[index + 1] - endpoints[index]
                    for index in range(len(endpoints) - 1)
                )
                for radius in range(
                    min(3, boundary_count) + 1
                ):
                    for allowance in ("all", "length"):
                        yield SparsePairwiseProblem(
                            token_count=atom_count,
                            original_boundaries=original,
                            max_boundary_moves=radius,
                            span_allowed=(
                                (lambda start, end: True)
                                if allowance == "all"
                                else (
                                    lambda start, end, cap=original_max:
                                    end - start <= cap
                                )
                            ),
                            pair_scorer=score_function,
                            threshold=13.0,
                        )


def random_problem(rng):
    atom_count = rng.randint(4, 11)
    boundary_count = rng.randint(1, min(4, atom_count - 1))
    original = tuple(sorted(rng.sample(
        range(1, atom_count), boundary_count
    )))
    radius = rng.randint(0, min(3, boundary_count))
    original_spans = set(zip(
        (0, *original),
        (*original, atom_count),
    ))
    legal = {
        (start, end): (
            (start, end) in original_spans or rng.random() >= 0.30
        )
        for start in range(atom_count)
        for end in range(start + 1, atom_count + 1)
    }
    return SparsePairwiseProblem(
        token_count=atom_count,
        original_boundaries=original,
        max_boundary_moves=radius,
        span_allowed=lambda start, end: legal[(start, end)],
        pair_scorer=score_function,
        threshold=29.0,
    )


def audit_sparse_dp(rng):
    systematic = list(systematic_problems())
    problems = systematic[:DP_CASES]
    while len(problems) < DP_CASES:
        problems.append(random_problem(rng))

    max_error = 0.0
    max_states = 0
    max_transitions = 0
    max_pairs = 0
    total_threat_members = 0
    for problem in problems:
        result, error = compare_problem(problem)
        max_error = max(max_error, error)
        max_states = max(max_states, result.reachable_state_count)
        max_transitions = max(
            max_transitions, result.transitions_evaluated
        )
        max_pairs = max(max_pairs, result.unique_pair_scores)
        total_threat_members += result.threat_set_size
    return {
        "cases": len(problems),
        "systematic_cases": min(len(systematic), DP_CASES),
        "random_cases": max(0, DP_CASES - len(systematic)),
        "total_threat_members": total_threat_members,
        "max_score_error": max_error,
        "max_states": max_states,
        "max_transitions": max_transitions,
        "max_unique_pair_scores": max_pairs,
        "score_match": max_error <= 1e-12,
        "histogram_match": True,
        "threat_size_match": True,
        "witness_match": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rng = random.Random(SEED)
    started = time.perf_counter()
    adapter = audit_adapter(rng)
    sparse = audit_sparse_dp(rng)
    output = {
        "schema_version": 1,
        "created_on": "2026-09-22",
        "audit_mode": "wp2_v3_model_free_correctness",
        "status": "pass",
        "parameters": {
            "seed": SEED,
            "adapter_cases": ADAPTER_CASES,
            "dp_cases": DP_CASES,
        },
        "adapter": adapter,
        "sparse_dp": sparse,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "execution_attestation": {
            "model_loads": 0,
            "tokenizer_loads": 0,
            "watermark_detector_calls": 0,
            "detector_score_reads": 0,
            "watermark_generation_calls": 0,
            "provider_api_calls": 0,
        },
        "contains_raw_text": False,
    }
    if not all((
        adapter["k0_exact"],
        adapter["atom_sequence_exact"],
        adapter["virtual_separator_exact"],
        adapter["preprocessing_exact"],
        sparse["score_match"],
        sparse["histogram_match"],
        sparse["threat_size_match"],
        sparse["witness_match"],
        math.isclose(sparse["max_score_error"], 0.0, abs_tol=1e-12),
    )):
        output["status"] = "stop"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": output["status"],
        "elapsed_seconds": output["elapsed_seconds"],
        "adapter_documents": adapter["documents"],
        "dp_cases": sparse["cases"],
        "total_threat_members": sparse["total_threat_members"],
        "max_score_error": sparse["max_score_error"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
