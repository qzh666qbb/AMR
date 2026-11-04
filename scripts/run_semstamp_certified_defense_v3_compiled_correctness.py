"""Run the frozen v3 compiled variable-N correctness matrix."""

import argparse
from itertools import combinations
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import tempfile
import time

from semstamp_certified_defense_v1_variable_n import (
    ExactScore,
    VariableNPairwiseProblem,
    compare_exact_scores,
    exact_variable_n_pairwise_certificate,
)


SYSTEMATIC_CASES = 5_000
RANDOM_CASES = 2_000
FIXED_REDUCTION_CASES = 1_000
MODE5_VARIABLE_CASES = 200
SYSTEMATIC_SEED = 20261011
RANDOM_SEED = 20261012
FIXED_SEED = 20261013
MODE5_SEED = 20261014


def canonical_bytes(value):
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, 0o600)
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def score_pair(left, middle, right):
    return (
        (left + 3) * 31
        + (middle + 5) * 17
        + (right + 7) * 13
    ) % 97


def synthetic_relation(previous, current):
    return int(((previous + 1) * 5 + (current + 3) * 3) % 4 == 0)


def synthetic_region(start, end):
    return ((start + 3) * 5 + (end + 7) * 3) % 8


def semstamp_hit(left, middle, right):
    return synthetic_relation(
        synthetic_region(left, middle),
        synthetic_region(middle, right),
    )


def always_allowed(start, end):
    return start < end


def span_bits(token_count, allowed):
    width = token_count + 1
    bits = bytearray(b"0" * (width * width))
    for start, end in allowed:
        bits[start * width + end] = ord("1")
    return bits.decode("ascii")


def parse_histogram(value):
    return {
        int(key): int(count)
        for item in value.split(",")
        for key, count in (item.split(":", 1),)
    }


def parse_output(payload, expected_count):
    lines = payload.decode("ascii").splitlines()
    if len(lines) != expected_count:
        raise AssertionError(
            f"compiled core returned {len(lines)} rows, "
            f"expected {expected_count}"
        )
    output = []
    for expected_index, line in enumerate(lines):
        fields = line.split("\t")
        if len(fields) != 12 or int(fields[0]) != expected_index:
            raise AssertionError("compiled core returned malformed output")
        original_sum = float(fields[1])
        minimum_sum = float(fields[2])
        if not original_sum.is_integer() or not minimum_sum.is_integer():
            raise AssertionError("synthetic score is not integral")
        output.append({
            "original_sum": int(original_sum),
            "minimum_sum": int(minimum_sum),
            "threat_set_size": int(fields[3]),
            "insertion_histogram": parse_histogram(fields[4]),
            "witness": tuple(map(int, fields[5].split(","))),
            "reachable_states": int(fields[6]),
            "transitions": int(fields[7]),
            "peak_layer_states": int(fields[8]),
            "candidate_positions": int(fields[9]),
            "unique_spans": int(fields[10]),
            "unique_pairs": int(fields[11]),
            "raw_fields": tuple(fields[1:]),
        })
    return output


def run_process(command, payload):
    return subprocess.run(
        [str(item) for item in command],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def v3_case_payload(case, score_mode=2, span_mode=0, suffix=None):
    token_count, original, target_count, radius, allowed = case
    fields = [
        token_count,
        len(original),
        target_count,
        radius,
        score_mode,
        span_mode,
        1,
        0,
        0,
        *original,
    ]
    if suffix is not None:
        fields.append(suffix)
    elif span_mode == 0:
        fields.append(span_bits(token_count, allowed))
    return " ".join(map(str, fields))


def v6_case_payload(case):
    token_count, original, _target_count, radius, allowed = case
    fields = [
        token_count,
        len(original),
        radius,
        2,
        0,
        1,
        0,
        0,
        *original,
        span_bits(token_count, allowed),
    ]
    return " ".join(map(str, fields))


def batch_payload(rows):
    return (str(len(rows)) + "\n" + "\n".join(rows) + "\n").encode(
        "ascii"
    )


def brute_target(case, scorer=score_pair):
    token_count, original, target_count, radius, allowed = case
    original_set = set(original)
    histogram = {index: 0 for index in range(4)}
    candidates = []
    for boundaries in combinations(range(1, token_count), target_count):
        selected = set(boundaries)
        insertions = len(selected - original_set)
        deletions = len(original_set - selected)
        if max(insertions, deletions) > radius:
            continue
        endpoints = (0, *boundaries, token_count)
        spans = tuple(zip(endpoints, endpoints[1:]))
        if any(span not in allowed for span in spans):
            continue
        total = sum(
            scorer(
                endpoints[index - 1],
                endpoints[index],
                endpoints[index + 1],
            )
            for index in range(1, len(endpoints) - 1)
        )
        histogram[insertions] += 1
        candidates.append((total, insertions, boundaries))
    if not candidates:
        return None
    candidates.sort()
    best = candidates[0]
    original_endpoints = (0, *original, token_count)
    original_sum = sum(
        scorer(
            original_endpoints[index - 1],
            original_endpoints[index],
            original_endpoints[index + 1],
        )
        for index in range(1, len(original_endpoints) - 1)
    )
    return {
        "original_sum": original_sum,
        "minimum_sum": best[0],
        "threat_set_size": len(candidates),
        "insertion_histogram": histogram,
        "witness": best[2],
    }


def compare_target(observed, expected):
    for key in (
        "original_sum",
        "minimum_sum",
        "threat_set_size",
        "insertion_histogram",
        "witness",
    ):
        if observed[key] != expected[key]:
            raise AssertionError(
                f"target oracle mismatch for {key}: "
                f"{observed[key]!r} != {expected[key]!r}"
            )


def all_spans(token_count):
    return {
        (start, end)
        for start in range(token_count)
        for end in range(start + 1, token_count + 1)
    }


def systematic_cases():
    output = []
    for token_count in range(3, 11):
        internal = range(1, token_count)
        allowed = all_spans(token_count)
        for original_count in range(1, token_count):
            for original in combinations(internal, original_count):
                for radius in range(min(3, original_count) + 1):
                    lower = max(1, original_count - radius)
                    upper = min(
                        token_count - 1,
                        original_count + radius,
                    )
                    for target_count in range(lower, upper + 1):
                        output.append((
                            token_count,
                            original,
                            target_count,
                            radius,
                            allowed,
                        ))
                        if len(output) == SYSTEMATIC_CASES:
                            return output
    raise AssertionError("systematic generator exhausted")


def random_case(rng, legality_probability=0.62):
    token_count = rng.randint(3, 10)
    original_count = rng.randint(1, token_count - 1)
    original = tuple(sorted(
        rng.sample(range(1, token_count), original_count)
    ))
    radius = rng.randint(0, min(3, original_count))
    lower = max(1, original_count - radius)
    upper = min(token_count - 1, original_count + radius)
    target_count = rng.randint(lower, upper)
    allowed = {
        (start, end)
        for start in range(token_count)
        for end in range(start + 1, token_count + 1)
        if rng.random() < legality_probability
    }
    endpoints = (0, *original, token_count)
    allowed.update(zip(endpoints, endpoints[1:]))
    return token_count, original, target_count, radius, allowed


def observation_case(case):
    return {
        "token_count": case[0],
        "original": list(case[1]),
        "target_count": case[2],
        "radius": case[3],
        "allowed": [list(span) for span in sorted(case[4])],
    }


def observation_result(row):
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in row.items()
        if key != "raw_fields"
    }


def update_observation(digest, family, case, row):
    digest.update(canonical_bytes({
        "case": observation_case(case),
        "family": family,
        "result": observation_result(row),
    }))


def run_target_family(core, family, cases, observation_digest):
    expected = [brute_target(case) for case in cases]
    successful = [
        (case, reference)
        for case, reference in zip(cases, expected)
        if reference is not None
    ]
    payload = batch_payload([
        v3_case_payload(case) for case, _reference in successful
    ])
    completed = run_process([core], payload)
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.decode("utf-8", errors="replace")
        )
    observed_rows = parse_output(completed.stdout, len(successful))
    for (case, reference), observed in zip(successful, observed_rows):
        compare_target(observed, reference)
        update_observation(
            observation_digest, family, case, observed
        )

    empty_count = 0
    for case, reference in zip(cases, expected):
        if reference is not None:
            continue
        empty_count += 1
        completed = run_process(
            [core],
            batch_payload([v3_case_payload(case)]),
        )
        if (
            completed.returncode == 0
            or completed.stdout
            or b"empty threat set" not in completed.stderr
        ):
            raise AssertionError("empty threat set did not fail closed")
        observation_digest.update(canonical_bytes({
            "case": observation_case(case),
            "family": family,
            "result": "empty_threat_set",
        }))
    return {
        "cases": len(cases),
        "nonempty": len(successful),
        "empty_fail_closed": empty_count,
    }


def run_fixed_reductions(core, v6_core, observation_digest):
    rng = random.Random(FIXED_SEED)
    cases = []
    for _ in range(FIXED_REDUCTION_CASES):
        case = random_case(rng, legality_probability=0.72)
        cases.append((case[0], case[1], len(case[1]), case[3], case[4]))
    current = run_process(
        [core],
        batch_payload([v3_case_payload(case) for case in cases]),
    )
    legacy = run_process(
        [v6_core],
        batch_payload([v6_case_payload(case) for case in cases]),
    )
    if current.returncode != 0 or legacy.returncode != 0:
        raise RuntimeError(
            "fixed-N replay failed: "
            + current.stderr.decode("utf-8", errors="replace")
            + legacy.stderr.decode("utf-8", errors="replace")
        )
    current_rows = parse_output(current.stdout, len(cases))
    legacy_rows = parse_output(legacy.stdout, len(cases))
    for case, observed, reference in zip(
        cases, current_rows, legacy_rows
    ):
        if observed["raw_fields"] != reference["raw_fields"]:
            raise AssertionError("fixed-N output differs from v6")
        update_observation(
            observation_digest,
            "fixed_n_reduction",
            case,
            observed,
        )
    return {
        "cases": len(cases),
        "byte_exact_rows": len(cases),
        "v3_stdout_sha256": hashlib.sha256(current.stdout).hexdigest(),
        "v6_stdout_sha256": hashlib.sha256(legacy.stdout).hexdigest(),
    }


def candidate_is_better(candidate, incumbent):
    if incumbent is None:
        return True
    comparison = compare_exact_scores(
        candidate["score"], incumbent["score"]
    )
    if comparison != 0:
        return comparison < 0
    return candidate["tie_key"] < incumbent["tie_key"]


def run_mode5_variables(core, observation_digest):
    rng = random.Random(MODE5_SEED)
    target_solves = 0
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        for case_index in range(MODE5_VARIABLE_CASES):
            token_count = rng.randint(4, 10)
            original_count = rng.randint(1, token_count - 1)
            original = tuple(sorted(
                rng.sample(range(1, token_count), original_count)
            ))
            radius = rng.randint(0, min(3, original_count))
            allowed = all_spans(token_count)
            target_rows = []
            lower = max(1, original_count - radius)
            upper = min(token_count - 1, original_count + radius)
            for target_count in range(lower, upper + 1):
                case = (
                    token_count,
                    original,
                    target_count,
                    radius,
                    allowed,
                )
                sidecar = directory / (
                    f"case{case_index}_m{target_count}.bin"
                )
                dump_row = v3_case_payload(
                    case,
                    score_mode=3,
                    span_mode=1,
                )
                dumped = run_process(
                    [
                        core,
                        "--dump-synthetic-sidecar",
                        sidecar,
                        "5",
                    ],
                    batch_payload([dump_row]),
                )
                if dumped.returncode != 0:
                    raise RuntimeError(
                        dumped.stderr.decode(
                            "utf-8", errors="replace"
                        )
                    )
                dump_result = parse_output(dumped.stdout, 1)[0]
                scored = run_process(
                    [core],
                    batch_payload([
                        v3_case_payload(
                            case,
                            score_mode=5,
                            span_mode=2,
                            suffix=sidecar,
                        )
                    ]),
                )
                sidecar.unlink()
                if scored.returncode != 0:
                    raise RuntimeError(
                        scored.stderr.decode(
                            "utf-8", errors="replace"
                        )
                    )
                observed = parse_output(scored.stdout, 1)[0]
                expected = brute_target(case, scorer=semstamp_hit)
                compare_target(observed, expected)
                for key in (
                    "threat_set_size",
                    "insertion_histogram",
                    "reachable_states",
                    "transitions",
                    "peak_layer_states",
                    "candidate_positions",
                    "unique_spans",
                    "unique_pairs",
                ):
                    if dump_result[key] != observed[key]:
                        raise AssertionError(
                            f"mode-5 topology changed for {key}"
                        )
                target_rows.append((target_count, case, observed))
                target_solves += 1

            reference = exact_variable_n_pairwise_certificate(
                VariableNPairwiseProblem(
                    token_count=token_count,
                    original_boundaries=original,
                    radius=radius,
                    span_allowed=always_allowed,
                    pair_hit=semstamp_hit,
                )
            )
            best = None
            threat_set_size = 0
            operation_histogram = {}
            for target_count, case, observed in target_rows:
                threat_set_size += observed["threat_set_size"]
                for insertions, count in observed[
                    "insertion_histogram"
                ].items():
                    if not count:
                        continue
                    deletions = (
                        original_count - target_count + insertions
                    )
                    operation_histogram[(insertions, deletions)] = count
                insertions = len(
                    set(observed["witness"]) - set(original)
                )
                deletions = len(
                    set(original) - set(observed["witness"])
                )
                candidate = {
                    "score": ExactScore(
                        observed["minimum_sum"], target_count
                    ),
                    "boundaries": observed["witness"],
                    "tie_key": (
                        max(insertions, deletions),
                        insertions,
                        deletions,
                        target_count,
                        observed["witness"],
                    ),
                }
                if candidate_is_better(candidate, best):
                    best = candidate
                update_observation(
                    observation_digest,
                    "mode5_target",
                    case,
                    observed,
                )
            reference_histogram = {
                (insertions, deletions): count
                for insertions, deletions, count
                in reference.histogram_by_insertions_deletions
            }
            if (
                best["score"] != reference.exact_score
                or best["boundaries"] != reference.worst_boundaries
                or threat_set_size != reference.threat_set_size
                or operation_histogram != reference_histogram
                or target_rows[0][2]["original_sum"]
                != reference.original_exact_score.hits
            ):
                raise AssertionError(
                    "mode-5 cross-m aggregate differs from oracle"
                )
            observation_digest.update(canonical_bytes({
                "case_index": case_index,
                "family": "mode5_cross_m",
                "operation_histogram": [
                    [insertions, deletions, count]
                    for (insertions, deletions), count
                    in sorted(operation_histogram.items())
                ],
                "score": {
                    "edge_count": best["score"].edge_count,
                    "hits": best["score"].hits,
                },
                "threat_set_size": threat_set_size,
                "witness": list(best["boundaries"]),
            }))
    return {
        "target_solves": target_solves,
        "variable_problems": MODE5_VARIABLE_CASES,
    }


def compiler_identity():
    completed = subprocess.run(
        ["c++", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        text=True,
    )
    return completed.stdout.splitlines()[0] if completed.stdout else "unknown"


def run(args):
    core = Path(args.core).resolve()
    v6_core = Path(args.v6_core).resolve()
    if not core.is_file() or not os.access(core, os.X_OK):
        raise ValueError("v3 core is not executable")
    if not v6_core.is_file() or not os.access(v6_core, os.X_OK):
        raise ValueError("v6 core is not executable")
    started = time.perf_counter()
    observation_digest = hashlib.sha256()

    systematic = run_target_family(
        core,
        "systematic",
        systematic_cases(),
        observation_digest,
    )
    rng = random.Random(RANDOM_SEED)
    random_cases = [random_case(rng) for _ in range(RANDOM_CASES)]
    random_result = run_target_family(
        core,
        "random_legality",
        random_cases,
        observation_digest,
    )
    if random_result["empty_fail_closed"] == 0:
        raise AssertionError("random matrix did not cover empty fail-closed")
    fixed = run_fixed_reductions(
        core, v6_core, observation_digest
    )
    mode5 = run_mode5_variables(core, observation_digest)

    result = {
        "counts": {
            "fixed_n_reduction": fixed,
            "mode5": mode5,
            "random_legality": random_result,
            "systematic": systematic,
            "total_declared_problems": (
                SYSTEMATIC_CASES
                + RANDOM_CASES
                + FIXED_REDUCTION_CASES
                + MODE5_VARIABLE_CASES
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "compiler": compiler_identity(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "gates": {
            "all_oracles_exact": True,
            "empty_cases_fail_closed": True,
            "fixed_n_byte_exact": True,
            "mode5_cross_m_exact": True,
            "problem_count_ge_8000": True,
        },
        "inputs": {
            "core_sha256": sha256_file(core),
            "protocol_sha256": sha256_file(args.protocol),
            "runner_sha256": sha256_file(Path(__file__)),
            "v6_core_sha256": sha256_file(v6_core),
        },
        "observation_stream_sha256": observation_digest.hexdigest(),
        "platform_label": args.platform_label,
        "replay_index": args.replay_index,
        "schema_version": 1,
        "seeds": {
            "fixed": FIXED_SEED,
            "mode5": MODE5_SEED,
            "random": RANDOM_SEED,
            "systematic": SYSTEMATIC_SEED,
        },
        "stage": "semstamp_certified_defense_v3_compiled_correctness",
        "status": "pass",
    }
    if not math.isfinite(result["elapsed_seconds"]):
        raise AssertionError("non-finite elapsed time")
    atomic_write_json(args.output, result)
    print(json.dumps({
        "elapsed_seconds": result["elapsed_seconds"],
        "observation_stream_sha256": result[
            "observation_stream_sha256"
        ],
        "status": result["status"],
    }, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", required=True)
    parser.add_argument("--v6-core", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--platform-label", required=True)
    parser.add_argument("--replay-index", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.replay_index not in (0, 1):
        parser.error("--replay-index must be 0 or 1")
    run(args)


if __name__ == "__main__":
    main()
