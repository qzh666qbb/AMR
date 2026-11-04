"""Run the frozen v3 Linux maximum-scale validation."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import time


PLAN_INDEX = 800
EXPECTED_RAW_SHA256 = (
    "bcd44daf7325ffafccb45f6cb6d3d5dcc91012724800d65980054a175d588d68"
)
EXPECTED_ATOM_COUNT = 5_554
EXPECTED_UNIT_COUNT = 174
EXPECTED_ORIGINAL_EDGES = 173
RADIUS = 3
TARGET_COUNTS = tuple(range(170, 177))
HISTORICAL_PAIR_COUNT = 83_540_900
HISTORICAL_SPAN_COUNT = 769_113
PAIR_CAP = 200_000_000
SPAN_CAP = 2_000_000
RSS_CAP_BYTES = 16 * 1024**3
TIMEOUT_SECONDS = 3_600


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


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


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


def read_jsonl_row(path, index):
    with Path(path).open("r", encoding="utf-8") as handle:
        for current, line in enumerate(handle):
            if current == index:
                return json.loads(line)
    raise ValueError(f"missing row {index}")


def parse_histogram(value):
    return {
        int(key): int(count)
        for item in value.split(",")
        for key, count in (item.split(":", 1),)
    }


def parse_output(payload):
    lines = payload.decode("ascii").splitlines()
    if len(lines) != 1:
        raise ValueError("core returned the wrong row count")
    fields = lines[0].split("\t")
    if len(fields) != 12 or fields[0] != "0":
        raise ValueError("core output schema mismatch")
    return {
        "candidate_boundary_positions": int(fields[9]),
        "certified_score": float(fields[2]),
        "insertion_histogram": parse_histogram(fields[4]),
        "original_score": float(fields[1]),
        "peak_layer_states": int(fields[8]),
        "reachable_state_count": int(fields[6]),
        "threat_set_size": int(fields[3]),
        "transitions_evaluated": int(fields[7]),
        "unique_pair_scores": int(fields[11]),
        "unique_span_checks": int(fields[10]),
        "witness_boundaries_sha256": sha256_bytes(
            fields[5].encode("ascii")
        ),
        "witness_edge_count": len(fields[5].split(",")),
    }


def parse_gnu_time(path):
    values = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            key, separator, value = line.strip().partition(":")
            if separator:
                values[key.strip()] = value.strip()
    key = "Maximum resident set size (kbytes)"
    if key not in values:
        raise ValueError("GNU time output lacks maximum RSS")
    return int(values[key]) * 1024


def payload(row, target_count, score_mode, span_mode, suffix=None):
    fields = [
        row["atom_count"],
        len(row["original_boundaries"]),
        target_count,
        RADIUS,
        score_mode,
        span_mode,
        1,
        0,
        0,
        *row["original_boundaries"],
    ]
    if suffix is not None:
        fields.append(suffix)
    return ("1\n" + " ".join(map(str, fields)) + "\n").encode("ascii")


def timed_core(core, arguments, input_payload, timing_path):
    has_gnu_time = Path("/usr/bin/time").is_file()
    command = [str(core), *map(str, arguments)]
    if has_gnu_time:
        command = [
            "/usr/bin/time",
            "-v",
            "-o",
            str(timing_path),
            *command,
        ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        input=input_payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=TIMEOUT_SECONDS,
    )
    elapsed = time.perf_counter() - started
    if has_gnu_time:
        rss_bytes = parse_gnu_time(timing_path)
        timing_path.unlink(missing_ok=True)
        rss_measurement = "gnu_time_maximum_rss"
    else:
        rss_bytes = (
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
        )
        rss_measurement = "rusage_children_running_upper_bound"
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.decode("utf-8", errors="replace")
        )
    return {
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": rss_bytes,
        "result": parse_output(completed.stdout),
        "rss_measurement": rss_measurement,
        "stdout_sha256": sha256_bytes(completed.stdout),
    }


def validate_identity(row):
    expected = {
        "atom_count": EXPECTED_ATOM_COUNT,
        "plan_index": PLAN_INDEX,
        "raw_text_sha256": EXPECTED_RAW_SHA256,
        "unit_count": EXPECTED_UNIT_COUNT,
    }
    mismatches = {
        key: (row.get(key), value)
        for key, value in expected.items()
        if row.get(key) != value
    }
    if mismatches:
        raise ValueError(f"maximum-scale identity mismatch: {mismatches}")
    if len(row.get("original_boundaries", ())) != EXPECTED_ORIGINAL_EDGES:
        raise ValueError("maximum-scale original edge count mismatch")


def topology_fields(result):
    return {
        key: result[key]
        for key in (
            "candidate_boundary_positions",
            "insertion_histogram",
            "peak_layer_states",
            "reachable_state_count",
            "threat_set_size",
            "transitions_evaluated",
            "unique_pair_scores",
            "unique_span_checks",
        )
    }


def run(args):
    core = Path(args.core).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(work_dir, 0o700)
    row = read_jsonl_row(args.v2_rows, PLAN_INDEX)
    validate_identity(row)

    target_profiles = []
    for target_count in TARGET_COUNTS:
        timed = timed_core(
            core,
            (),
            payload(row, target_count, 3, 1),
            work_dir / f"topology_m{target_count}.time",
        )
        target_profiles.append({
            "elapsed_seconds": timed["elapsed_seconds"],
            "peak_rss_bytes": timed["peak_rss_bytes"],
            "result": timed["result"],
            "rss_measurement": timed["rss_measurement"],
            "stdout_sha256": timed["stdout_sha256"],
            "target_edge_count": target_count,
        })

    fixed_profile = next(
        item for item in target_profiles
        if item["target_edge_count"] == EXPECTED_ORIGINAL_EDGES
    )
    if (
        fixed_profile["result"]["unique_pair_scores"]
        != HISTORICAL_PAIR_COUNT
        or fixed_profile["result"]["unique_span_checks"]
        != HISTORICAL_SPAN_COUNT
    ):
        raise AssertionError("fixed-N historical topology changed")

    qualifying = [
        item for item in target_profiles
        if (
            HISTORICAL_PAIR_COUNT
            < item["result"]["unique_pair_scores"]
            <= PAIR_CAP
        )
    ]
    if not qualifying:
        raise AssertionError(
            "no target exceeds the historical pair count within v3 cap"
        )
    selected = min(
        qualifying,
        key=lambda item: (
            -item["result"]["unique_pair_scores"],
            item["target_edge_count"],
        ),
    )
    selected_target = selected["target_edge_count"]
    sidecar = work_dir / "maximum_scale_mode5.bin"
    dumped = timed_core(
        core,
        ("--dump-synthetic-sidecar", sidecar, "5"),
        payload(row, selected_target, 3, 1),
        work_dir / "sidecar_dump.time",
    )
    if dumped["stdout_sha256"] != selected["stdout_sha256"]:
        raise AssertionError("sidecar dump changed score-blind topology")
    sidecar_bytes = sidecar.stat().st_size
    sidecar_sha256 = sha256_file(sidecar)

    scored = timed_core(
        core,
        (),
        payload(row, selected_target, 5, 2, suffix=sidecar),
        work_dir / "mode5.time",
    )
    sidecar.unlink()
    if topology_fields(scored["result"]) != topology_fields(
        selected["result"]
    ):
        raise AssertionError("mode-5 replay changed topology fields")

    all_runs = [*target_profiles, {
        "elapsed_seconds": dumped["elapsed_seconds"],
        "peak_rss_bytes": dumped["peak_rss_bytes"],
    }, {
        "elapsed_seconds": scored["elapsed_seconds"],
        "peak_rss_bytes": scored["peak_rss_bytes"],
    }]
    maximum_rss = max(item["peak_rss_bytes"] for item in all_runs)
    maximum_elapsed = max(item["elapsed_seconds"] for item in all_runs)
    selected_pairs = selected["result"]["unique_pair_scores"]
    selected_spans = selected["result"]["unique_span_checks"]
    gates = {
        "fixed_n_historical_counts_exact": True,
        "maximum_replay_le_3600_seconds": (
            maximum_elapsed <= TIMEOUT_SECONDS
        ),
        "maximum_rss_le_16_gib": maximum_rss <= RSS_CAP_BYTES,
        "mode5_topology_exact": True,
        "pair_count_gt_historical": (
            selected_pairs > HISTORICAL_PAIR_COUNT
        ),
        "pair_count_le_200m": selected_pairs <= PAIR_CAP,
        "sidecar_dump_stdout_exact": True,
        "span_count_le_2m": selected_spans <= SPAN_CAP,
    }
    if not all(gates.values()):
        raise AssertionError(f"maximum-scale gate failure: {gates}")
    if not all(
        math.isfinite(item["elapsed_seconds"]) for item in all_runs
    ):
        raise AssertionError("non-finite maximum-scale timing")

    result = {
        "caps": {
            "historical_pair_count": HISTORICAL_PAIR_COUNT,
            "pair_cap": PAIR_CAP,
            "replay_timeout_seconds": TIMEOUT_SECONDS,
            "rss_cap_bytes": RSS_CAP_BYTES,
            "span_cap": SPAN_CAP,
        },
        "gates": gates,
        "identity": {
            "atom_count": row["atom_count"],
            "original_edge_count": len(row["original_boundaries"]),
            "original_boundaries_sha256": sha256_bytes(canonical_bytes(
                row["original_boundaries"]
            )),
            "plan_index": row["plan_index"],
            "raw_text_sha256": row["raw_text_sha256"],
            "unit_count": row["unit_count"],
        },
        "inputs": {
            "core_sha256": sha256_file(core),
            "protocol_sha256": sha256_file(args.protocol),
            "runner_sha256": sha256_file(Path(__file__)),
            "v2_rows_sha256": sha256_file(args.v2_rows),
        },
        "maximum_observed_elapsed_seconds": maximum_elapsed,
        "maximum_observed_rss_bytes": maximum_rss,
        "mode5": {
            "elapsed_seconds": scored["elapsed_seconds"],
            "peak_rss_bytes": scored["peak_rss_bytes"],
            "result": scored["result"],
            "rss_measurement": scored["rss_measurement"],
            "sidecar_bytes": sidecar_bytes,
            "sidecar_sha256": sidecar_sha256,
            "stdout_sha256": scored["stdout_sha256"],
        },
        "replay_index": args.replay_index,
        "schema_version": 1,
        "selected_target_edge_count": selected_target,
        "sidecar_dump": {
            "elapsed_seconds": dumped["elapsed_seconds"],
            "peak_rss_bytes": dumped["peak_rss_bytes"],
            "rss_measurement": dumped["rss_measurement"],
            "stdout_sha256": dumped["stdout_sha256"],
        },
        "stage": "semstamp_certified_defense_v3_maximum_scale",
        "status": "pass",
        "target_profiles": target_profiles,
    }
    atomic_write_json(args.output, result)
    print(json.dumps({
        "mode5_stdout_sha256": scored["stdout_sha256"],
        "selected_pairs": selected_pairs,
        "selected_target_edge_count": selected_target,
        "status": "pass",
    }, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--v2-rows", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--replay-index", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.replay_index not in (0, 1):
        parser.error("--replay-index must be 0 or 1")
    run(args)


if __name__ == "__main__":
    main()
