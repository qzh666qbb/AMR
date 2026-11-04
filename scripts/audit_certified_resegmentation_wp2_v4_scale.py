"""Run one independent WP2-v4 compiled scale replay."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import subprocess
import time

from audit_certified_resegmentation_wp2_v4_correctness import (
    parse_core_output,
)
from certified_resegmentation_wp2_v3 import (
    SparsePairwiseProblem,
    exact_sparse_pairwise_certificate,
)


EXPECTED_TOKEN_COUNT = 14_169
EXPECTED_BOUNDARY_COUNT = 844
EXPECTED_STRUCTURAL_TAPE_SHA256 = (
    "d31e34cc1873d933c8100dd799af58adf2168ee930873bd0040a2523a7b6bfbc"
)
TIMEOUT_SECONDS = 70
WALL_GATE_SECONDS = 60
RSS_GATE_BYTES = 4 * 1024 ** 3
SPAN_GATE = 2_000_000
PAIR_GATE = 50_000_000


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def synthetic_score(left, middle, right):
    return float(
        ((left + 3) * 31 + (middle + 5) * 17 + (right + 7) * 13)
        % 97
    )


def serialize_problem(n, boundaries):
    fields = [
        n,
        len(boundaries),
        3,
        2,
        1,
        len(boundaries),
        0,
        0,
        *boundaries,
    ]
    return ("1\n" + " ".join(map(str, fields)) + "\n").encode("ascii")


def run_single(core, payload):
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(core)],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        elapsed = time.perf_counter() - started
        return {
            "completed": False,
            "elapsed_seconds": elapsed,
            "peak_rss_bytes": (
                resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                * 1024
            ),
            "error": f"compiled scale audit exceeded {TIMEOUT_SECONDS}s",
            "stdout_sha256": sha256_bytes(error.stdout or b""),
            "result": None,
        }
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            "compiled core failed: "
            + completed.stderr.decode("utf-8", errors="replace")
        )
    parsed = parse_core_output(completed.stdout, 1)[0]
    return {
        "completed": True,
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": (
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            * 1024
        ),
        "error": None,
        "stdout_sha256": sha256_bytes(completed.stdout),
        "result": parsed,
    }


def expected_histogram(n, boundary_count, k):
    nonoriginal = n - 1 - boundary_count
    return tuple(
        (
            radius,
            math.comb(boundary_count, radius)
            * math.comb(nonoriginal, radius),
        )
        for radius in range(k + 1)
    )


def audit_preflight(core):
    n = 300
    boundary_count = 60
    boundaries = tuple(
        round(index * n / (boundary_count + 1))
        for index in range(1, boundary_count + 1)
    )
    compiled = run_single(core, serialize_problem(n, boundaries))
    if not compiled["completed"]:
        raise AssertionError("n=300 compiled preflight timed out")
    exact = exact_sparse_pairwise_certificate(
        SparsePairwiseProblem(
            token_count=n,
            original_boundaries=boundaries,
            max_boundary_moves=3,
            span_allowed=lambda start, end: True,
            pair_scorer=synthetic_score,
        )
    )
    observed = compiled["result"]
    score_error = max(
        abs(observed["original_score"] - exact.original_score),
        abs(observed["certified_score"] - exact.certified_score),
    )
    exact_fields = (
        "worst_boundaries",
        "exact_move_histogram",
        "threat_set_size",
        "reachable_state_count",
        "transitions_evaluated",
        "peak_layer_states",
        "candidate_boundary_positions",
        "unique_span_checks",
        "unique_pair_scores",
    )
    if score_error > 1e-12 or any(
        observed[field] != getattr(exact, field)
        for field in exact_fields
    ):
        raise AssertionError("n=300 compiled preflight differs from v3")
    return {
        "status": "pass",
        "token_count": n,
        "boundary_count": boundary_count,
        "score_error": score_error,
        "compiled": compiled,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--shape", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-id", required=True)
    args = parser.parse_args()

    shape_bytes = args.shape.read_bytes()
    shape = json.loads(shape_bytes)
    boundaries = tuple(shape["boundaries"])
    if (
        shape["n"] != EXPECTED_TOKEN_COUNT
        or len(boundaries) != EXPECTED_BOUNDARY_COUNT
        or shape["structural_tape_sha256"]
        != EXPECTED_STRUCTURAL_TAPE_SHA256
    ):
        raise ValueError("scale shape differs from the frozen v3 maximum")
    if tuple(sorted(set(boundaries))) != boundaries:
        raise ValueError("scale boundaries are not strictly increasing")

    preflight = audit_preflight(args.core)
    scale = run_single(
        args.core,
        serialize_problem(shape["n"], boundaries),
    )
    if scale["completed"]:
        expected = expected_histogram(
            shape["n"], len(boundaries), 3
        )
        histogram_closed_form_match = (
            scale["result"]["exact_move_histogram"] == expected
        )
        gates = {
            "wall_seconds_le_60": (
                scale["elapsed_seconds"] <= WALL_GATE_SECONDS
            ),
            "peak_rss_le_4_gib": (
                scale["peak_rss_bytes"] <= RSS_GATE_BYTES
            ),
            "unique_spans_le_2000000": (
                scale["result"]["unique_span_checks"] <= SPAN_GATE
            ),
            "unique_pairs_le_50000000": (
                scale["result"]["unique_pair_scores"] <= PAIR_GATE
            ),
            "histogram_matches_closed_form": (
                histogram_closed_form_match
            ),
        }
    else:
        gates = {
            "wall_seconds_le_60": False,
            "peak_rss_le_4_gib": (
                scale["peak_rss_bytes"] <= RSS_GATE_BYTES
            ),
            "unique_spans_le_2000000": None,
            "unique_pairs_le_50000000": None,
            "histogram_matches_closed_form": None,
        }
    status = (
        "pass"
        if preflight["status"] == "pass"
        and all(value is True for value in gates.values())
        else "stop"
    )

    output = {
        "schema_version": 1,
        "created_on": "2026-09-22",
        "audit_mode": "wp2_v4_compiled_scale_replay",
        "replay_id": args.replay_id,
        "status": status,
        "parameters": {
            "max_boundary_moves": 3,
            "timeout_seconds": TIMEOUT_SECONDS,
            "wall_gate_seconds": WALL_GATE_SECONDS,
            "rss_gate_bytes": RSS_GATE_BYTES,
            "span_gate": SPAN_GATE,
            "pair_gate": PAIR_GATE,
            "score_mode": "frozen_synthetic_mod97",
            "span_mode": "all_allowed",
        },
        "preflight_n300": preflight,
        "scale": scale,
        "shape": {
            "path": str(args.shape),
            "sha256": sha256_bytes(shape_bytes),
            "token_count": shape["n"],
            "boundary_count": len(boundaries),
            "structural_tape_sha256": (
                shape["structural_tape_sha256"]
            ),
            "contains_raw_text": False,
        },
        "core": {
            "path": str(args.core),
            "sha256": sha256_bytes(args.core.read_bytes()),
        },
        "gates": gates,
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
        "replay_id": args.replay_id,
        "status": status,
        "elapsed_seconds": scale["elapsed_seconds"],
        "peak_rss_bytes": scale["peak_rss_bytes"],
        "gates": gates,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
