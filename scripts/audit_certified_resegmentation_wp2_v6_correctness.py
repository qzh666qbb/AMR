"""Audit WP2-v6 factorized modes against Python and brute force."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import tempfile
import time

from audit_certified_resegmentation_wp2_v4_correctness import (
    parse_core_output,
)
from certified_resegmentation_wp2_v3 import (
    SparsePairwiseProblem,
    brute_force_sparse_pairwise_certificate,
)
from certified_resegmentation_wp2_v4 import (
    exact_grouped_sparse_pairwise_certificate,
)
from certified_resegmentation_wp2_v6_features import (
    SEMSTAMP_MODE,
    SIMMARK_MODE,
    encode_sidecar,
    feature_pair_score,
    synthetic_embedding,
    synthetic_region,
    synthetic_relation,
)


SEED = 20260922
CASES_PER_MODE = 1000


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def make_case(case_index, mode, rng):
    token_count = rng.randint(4, 11)
    boundary_count = rng.randint(1, min(4, token_count - 1))
    original = tuple(sorted(rng.sample(
        range(1, token_count),
        boundary_count,
    )))
    radius = rng.randint(0, min(3, boundary_count))
    original_spans = set(zip(
        (0, *original),
        (*original, token_count),
    ))
    relation = synthetic_relation()
    features = {}

    def is_allowed(start, end):
        return (
            (start, end) in original_spans
            or (
                start * 1009
                + end * 9176
                + case_index * 65537
                + mode * 31
            ) % 7 != 0
        )

    def feature(start, end):
        key = (start, end)
        if key not in features:
            features[key] = (
                synthetic_region(start, end)
                if mode == SEMSTAMP_MODE
                else synthetic_embedding(start, end)
            )
        return features[key]

    def score(left, middle, right):
        feature(left, middle)
        feature(middle, right)
        return feature_pair_score(
            mode,
            features,
            relation,
            left,
            middle,
            right,
        )

    queried = set()

    def recording_allowed(start, end):
        queried.add((start, end))
        return is_allowed(start, end)

    problem = SparsePairwiseProblem(
        token_count=token_count,
        original_boundaries=original,
        max_boundary_moves=radius,
        span_allowed=recording_allowed,
        pair_scorer=score,
        normalization=float(boundary_count),
        bias=0.125,
        threshold=0.25,
    )
    exact = exact_grouped_sparse_pairwise_certificate(problem)
    frozen_queries = tuple(sorted(queried))
    records = tuple(
        (
            start,
            end,
            is_allowed(start, end),
            feature(start, end) if is_allowed(start, end) else None,
        )
        for start, end in frozen_queries
    )

    oracle_problem = SparsePairwiseProblem(
        token_count=token_count,
        original_boundaries=original,
        max_boundary_moves=radius,
        span_allowed=is_allowed,
        pair_scorer=score,
        normalization=float(boundary_count),
        bias=0.125,
        threshold=0.25,
    )
    oracle = brute_force_sparse_pairwise_certificate(oracle_problem)
    return problem, exact, oracle, records, relation


def serialize_case(problem, mode, path):
    fields = [
        problem.token_count,
        len(problem.original_boundaries),
        problem.max_boundary_moves,
        mode,
        2,
        format(float(problem.normalization), ".17g"),
        format(float(problem.bias), ".17g"),
        format(float(problem.threshold), ".17g"),
        *problem.original_boundaries,
        str(path),
    ]
    return " ".join(map(str, fields))


def compare_python(exact, oracle):
    for field in (
        "original_score",
        "certified_score",
        "worst_boundaries",
        "exact_move_histogram",
        "threat_set_size",
    ):
        if getattr(exact, field) != getattr(oracle, field):
            raise AssertionError(f"Python {field} differs from oracle")


def compare_core(observed, exact, tolerance):
    error = max(
        abs(observed["original_score"] - exact.original_score),
        abs(observed["certified_score"] - exact.certified_score),
    )
    if error > tolerance:
        raise AssertionError("compiled score differs from Python")
    for field in (
        "worst_boundaries",
        "exact_move_histogram",
        "threat_set_size",
        "reachable_state_count",
        "transitions_evaluated",
        "peak_layer_states",
        "candidate_boundary_positions",
        "unique_span_checks",
        "unique_pair_scores",
    ):
        if observed[field] != getattr(exact, field):
            raise AssertionError(f"compiled {field} differs from Python")
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rng = random.Random(SEED)
    started = time.perf_counter()
    references = []
    payload_rows = []
    aggregate_sidecar_hash = hashlib.sha256()
    deterministic_sidecars = True
    totals = {
        "span_records": 0,
        "pair_scores": 0,
        "threat_members": 0,
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        case_index = 0
        for mode in (SEMSTAMP_MODE, SIMMARK_MODE):
            for _ in range(CASES_PER_MODE):
                problem, exact, oracle, records, relation = make_case(
                    case_index,
                    mode,
                    rng,
                )
                compare_python(exact, oracle)
                encoded = encode_sidecar(
                    mode,
                    problem.token_count,
                    records,
                    relation=relation,
                )
                repeated = encode_sidecar(
                    mode,
                    problem.token_count,
                    records,
                    relation=relation,
                )
                deterministic_sidecars &= encoded == repeated
                aggregate_sidecar_hash.update(encoded)
                path = root / f"case_{case_index:04d}.bin"
                path.write_bytes(encoded)
                payload_rows.append(serialize_case(
                    problem,
                    mode,
                    path,
                ))
                references.append((mode, exact))
                totals["span_records"] += len(records)
                totals["pair_scores"] += exact.unique_pair_scores
                totals["threat_members"] += exact.threat_set_size
                case_index += 1

        payload = (
            str(len(payload_rows))
            + "\n"
            + "\n".join(payload_rows)
            + "\n"
        ).encode("ascii")
        completed = subprocess.run(
            [str(args.core)],
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
        observed = parse_core_output(
            completed.stdout,
            len(references),
        )

    max_errors = {
        "semstamp": 0.0,
        "simmark": 0.0,
    }
    for compiled, (mode, exact) in zip(observed, references):
        name = "semstamp" if mode == SEMSTAMP_MODE else "simmark"
        tolerance = 1e-12 if mode == SEMSTAMP_MODE else 1e-8
        max_errors[name] = max(
            max_errors[name],
            compare_core(compiled, exact, tolerance),
        )
    if not deterministic_sidecars:
        raise AssertionError("sidecar replay changed bytes")

    output = {
        "schema_version": 1,
        "created_on": "2026-09-22",
        "audit_mode": "wp2_v6_factorized_three_way_correctness",
        "status": "pass",
        "parameters": {
            "seed": SEED,
            "cases_per_mode": CASES_PER_MODE,
            "total_cases": len(references),
            "simmark_dimension": 768,
        },
        "results": {
            "python_oracle_match": True,
            "histogram_match": True,
            "threat_size_match": True,
            "witness_match": True,
            "instrumentation_match": True,
            "deterministic_sidecars": deterministic_sidecars,
            "aggregate_sidecar_sha256": (
                aggregate_sidecar_hash.hexdigest()
            ),
            "stdout_sha256": sha256_bytes(completed.stdout),
            "max_score_errors": max_errors,
            "totals": totals,
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
            "model_score_reads": 0,
            "watermark_detector_calls": 0,
            "watermark_generation_calls": 0,
            "provider_api_calls": 0,
        },
        "contains_raw_text": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "status": output["status"],
        "cases": len(references),
        "max_score_errors": max_errors,
        "stdout_sha256": output["results"]["stdout_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
