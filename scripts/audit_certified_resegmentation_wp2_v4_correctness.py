"""Audit the compiled WP2-v4 core against v3 and the brute-force oracle."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import time

from audit_certified_resegmentation_wp2_v3_correctness import (
    ADAPTER_CASES,
    DP_CASES,
    SEED,
    audit_adapter,
    random_problem,
    systematic_problems,
)
from certified_resegmentation_wp2_v3 import (
    SparsePairwiseProblem,
    brute_force_sparse_pairwise_certificate,
    exact_sparse_pairwise_certificate,
)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def span_table(problem):
    width = problem.token_count + 1
    bits = bytearray(b"0" * (width * width))
    for start in range(problem.token_count):
        for end in range(start + 1, problem.token_count + 1):
            value = problem.span_allowed(start, end)
            if not isinstance(value, bool):
                raise TypeError("span_allowed must return bool")
            if value:
                bits[start * width + end] = ord("1")
    return bits.decode("ascii")


def serialize_problems(problems):
    lines = [str(len(problems))]
    for problem in problems:
        normalization = (
            len(problem.original_boundaries)
            if problem.normalization is None
            else float(problem.normalization)
        )
        fields = [
            problem.token_count,
            len(problem.original_boundaries),
            problem.max_boundary_moves,
            1,
            0,
            format(float(normalization), ".17g"),
            format(float(problem.bias), ".17g"),
            format(float(problem.threshold), ".17g"),
            *problem.original_boundaries,
            span_table(problem),
        ]
        lines.append(" ".join(map(str, fields)))
    return ("\n".join(lines) + "\n").encode("ascii")


def parse_histogram(value):
    return tuple(
        (int(radius), int(count))
        for item in value.split(",")
        for radius, count in (item.split(":", 1),)
        if int(count) != 0
    )


def parse_core_output(value, expected_cases):
    rows = []
    lines = value.decode("ascii").splitlines()
    if len(lines) != expected_cases:
        raise AssertionError("compiled core returned the wrong row count")
    for expected_index, line in enumerate(lines):
        fields = line.split("\t")
        if len(fields) != 12:
            raise AssertionError("compiled core returned a malformed row")
        case_index = int(fields[0])
        if case_index != expected_index:
            raise AssertionError("compiled core changed case order")
        rows.append({
            "original_score": float(fields[1]),
            "certified_score": float(fields[2]),
            "threat_set_size": int(fields[3]),
            "exact_move_histogram": parse_histogram(fields[4]),
            "worst_boundaries": tuple(
                map(int, fields[5].split(","))
            ),
            "reachable_state_count": int(fields[6]),
            "transitions_evaluated": int(fields[7]),
            "peak_layer_states": int(fields[8]),
            "candidate_boundary_positions": int(fields[9]),
            "unique_span_checks": int(fields[10]),
            "unique_pair_scores": int(fields[11]),
        })
    return rows


def run_core(core, payload, expected_cases):
    completed = subprocess.run(
        [str(core)],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "compiled core failed: "
            + completed.stderr.decode("utf-8", errors="replace")
        )
    return (
        parse_core_output(completed.stdout, expected_cases),
        completed.stdout,
    )


def make_problems(rng):
    systematic = list(systematic_problems())
    problems = systematic[:DP_CASES]
    while len(problems) < DP_CASES:
        problems.append(random_problem(rng))
    tie_problem = SparsePairwiseProblem(
        token_count=9,
        original_boundaries=(2, 5, 7),
        max_boundary_moves=3,
        span_allowed=lambda start, end: True,
        pair_scorer=lambda left, middle, right: 0.0,
        threshold=17.0,
    )
    return problems, tie_problem, len(systematic)


def tie_payload(problem):
    normalization = len(problem.original_boundaries)
    fields = [
        problem.token_count,
        len(problem.original_boundaries),
        problem.max_boundary_moves,
        3,
        1,
        normalization,
        problem.bias,
        problem.threshold,
        *problem.original_boundaries,
    ]
    return ("1\n" + " ".join(map(str, fields)) + "\n").encode("ascii")


def compare_result(observed, exact, oracle):
    max_error = max(
        abs(observed["original_score"] - oracle.original_score),
        abs(observed["certified_score"] - oracle.certified_score),
    )
    if max_error > 1e-12:
        raise AssertionError("compiled score differs from oracle")
    for field in (
        "worst_boundaries",
        "exact_move_histogram",
        "threat_set_size",
    ):
        if observed[field] != getattr(oracle, field):
            raise AssertionError(f"compiled {field} differs from oracle")
    for field in (
        "reachable_state_count",
        "transitions_evaluated",
        "peak_layer_states",
        "candidate_boundary_positions",
        "unique_span_checks",
        "unique_pair_scores",
    ):
        if observed[field] != getattr(exact, field):
            raise AssertionError(f"compiled {field} differs from v3")
    return max_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rng = random.Random(SEED)
    started = time.perf_counter()
    adapter = audit_adapter(rng)
    problems, tie_problem, systematic_count = make_problems(rng)
    payload = serialize_problems(problems)
    observed, first_stdout = run_core(
        args.core, payload, len(problems)
    )
    repeated, second_stdout = run_core(
        args.core, payload, len(problems)
    )
    if first_stdout != second_stdout or observed != repeated:
        raise AssertionError("compiled replay changed output")

    max_error = 0.0
    total_threat_members = 0
    maxima = {
        "reachable_state_count": 0,
        "transitions_evaluated": 0,
        "unique_pair_scores": 0,
    }
    for problem, compiled in zip(problems, observed):
        exact = exact_sparse_pairwise_certificate(problem)
        oracle = brute_force_sparse_pairwise_certificate(problem)
        max_error = max(
            max_error,
            compare_result(compiled, exact, oracle),
        )
        total_threat_members += compiled["threat_set_size"]
        for field in maxima:
            maxima[field] = max(maxima[field], compiled[field])

    compiled_tie, tie_stdout = run_core(
        args.core, tie_payload(tie_problem), 1
    )
    exact_tie = exact_sparse_pairwise_certificate(tie_problem)
    oracle_tie = brute_force_sparse_pairwise_certificate(tie_problem)
    compare_result(compiled_tie[0], exact_tie, oracle_tie)

    output = {
        "schema_version": 1,
        "created_on": "2026-09-22",
        "audit_mode": "wp2_v4_compiled_three_way_correctness",
        "status": "pass",
        "parameters": {
            "seed": SEED,
            "adapter_cases": ADAPTER_CASES,
            "dp_cases": len(problems),
            "systematic_cases": min(
                systematic_count, len(problems)
            ),
            "random_cases": max(
                0, len(problems) - systematic_count
            ),
            "tie_cases": 1,
        },
        "adapter": adapter,
        "sparse_dp": {
            "cases": len(problems),
            "total_threat_members": total_threat_members,
            "max_score_error": max_error,
            "score_match": max_error <= 1e-12,
            "histogram_match": True,
            "threat_size_match": True,
            "witness_match": True,
            "v3_instrumentation_match": True,
            "byte_identical_replay": True,
            "stdout_sha256": sha256_bytes(first_stdout),
            "tie_stdout_sha256": sha256_bytes(tie_stdout),
            "maxima": maxima,
        },
        "core": {
            "path": str(args.core),
            "sha256": sha256_bytes(args.core.read_bytes()),
        },
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": output["status"],
        "adapter_documents": adapter["documents"],
        "dp_cases": len(problems),
        "max_score_error": max_error,
        "stdout_sha256": output["sparse_dp"]["stdout_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
