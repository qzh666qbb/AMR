"""Run one WP2-v6 synthetic factorized-feature scale gate."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

from audit_certified_resegmentation_wp2_v4_correctness import (
    parse_core_output,
)


MODE_NAMES = {5: "semstamp_region", 6: "simmark_embedding_768"}
SIDE_CAR_GATE_BYTES = 50 * 1024 ** 3
RSS_GATES = {5: 4 * 1024 ** 3, 6: 16 * 1024 ** 3}
WALL_GATES = {
    ("n300", 5): 60,
    ("n300", 6): 60,
    ("maximum", 5): 60,
    ("maximum", 6): 600,
}
SPAN_GATE = 2_000_000
PAIR_GATE = 50_000_000
EXPECTED_MAX_N = 14_169
EXPECTED_MAX_M = 844
EXPECTED_MAX_TAPE_SHA256 = (
    "d31e34cc1873d933c8100dd799af58adf2168ee930873bd0040a2523a7b6bfbc"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def child_peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return value if platform.system() == "Darwin" else value * 1024


def shape_from_args(size, shape_path):
    if size == "n300":
        n = 300
        boundary_count = 60
        boundaries = tuple(
            round(index * n / (boundary_count + 1))
            for index in range(1, boundary_count + 1)
        )
        return {
            "n": n,
            "boundaries": boundaries,
            "structural_tape_sha256": None,
        }
    if shape_path is None:
        raise ValueError("--shape is required for maximum scale")
    shape = json.loads(shape_path.read_text(encoding="utf-8"))
    boundaries = tuple(shape["boundaries"])
    if (
        shape["n"] != EXPECTED_MAX_N
        or len(boundaries) != EXPECTED_MAX_M
        or shape["structural_tape_sha256"]
        != EXPECTED_MAX_TAPE_SHA256
    ):
        raise ValueError("maximum shape differs from frozen v4")
    return {
        "n": shape["n"],
        "boundaries": boundaries,
        "structural_tape_sha256": shape[
            "structural_tape_sha256"
        ],
    }


def baseline_payload(shape):
    boundaries = shape["boundaries"]
    fields = [
        shape["n"],
        len(boundaries),
        min(3, len(boundaries)),
        3,
        1,
        len(boundaries),
        0,
        0,
        *boundaries,
    ]
    return ("1\n" + " ".join(map(str, fields)) + "\n").encode("ascii")


def feature_payload(shape, mode, sidecar):
    boundaries = shape["boundaries"]
    fields = [
        shape["n"],
        len(boundaries),
        min(3, len(boundaries)),
        mode,
        2,
        len(boundaries),
        0,
        0,
        *boundaries,
        str(sidecar),
    ]
    return ("1\n" + " ".join(map(str, fields)) + "\n").encode("ascii")


def run_process(command, payload, timeout):
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "completed": False,
            "timed_out": True,
            "returncode": None,
            "elapsed_seconds": time.perf_counter() - started,
            "stdout_sha256": hashlib.sha256(
                error.stdout or b""
            ).hexdigest(),
            "stderr": (error.stderr or b"").decode(
                "utf-8",
                errors="replace",
            ),
            "peak_rss_bytes": child_peak_rss_bytes(),
            "result": None,
        }
    elapsed = time.perf_counter() - started
    result = None
    if completed.returncode == 0:
        result = parse_core_output(completed.stdout, 1)[0]
    return {
        "completed": completed.returncode == 0,
        "timed_out": False,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "stdout_sha256": hashlib.sha256(
            completed.stdout
        ).hexdigest(),
        "stderr": completed.stderr.decode(
            "utf-8",
            errors="replace",
        ),
        "peak_rss_bytes": child_peak_rss_bytes(),
        "result": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--mode", type=int, choices=(5, 6), required=True)
    parser.add_argument(
        "--size",
        choices=("n300", "maximum"),
        required=True,
    )
    parser.add_argument("--shape", type=Path)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shape = shape_from_args(args.size, args.shape)
    base_payload = baseline_payload(shape)
    args.sidecar.parent.mkdir(parents=True, exist_ok=True)
    replay_sidecar = args.sidecar.with_suffix(
        args.sidecar.suffix + ".replay"
    )
    for path in (args.sidecar, replay_sidecar):
        if path.exists():
            path.unlink()

    generation_timeout = 900 if args.size == "maximum" else 120
    generations = []
    for path in (args.sidecar, replay_sidecar):
        generation = run_process(
            [
                str(args.core),
                "--dump-synthetic-sidecar",
                str(path),
                str(args.mode),
            ],
            base_payload,
            generation_timeout,
        )
        if generation["completed"]:
            generation["sidecar_bytes"] = path.stat().st_size
            generation["sidecar_sha256"] = sha256_file(path)
        else:
            generation["sidecar_bytes"] = None
            generation["sidecar_sha256"] = None
        generations.append(generation)
        if not generation["completed"]:
            break

    sidecar_replay_identical = (
        len(generations) == 2
        and generations[0]["sidecar_bytes"]
        == generations[1]["sidecar_bytes"]
        and generations[0]["sidecar_sha256"]
        == generations[1]["sidecar_sha256"]
    )
    if replay_sidecar.exists():
        replay_sidecar.unlink()

    replays = []
    wall_gate = WALL_GATES[(args.size, args.mode)]
    if sidecar_replay_identical:
        payload = feature_payload(shape, args.mode, args.sidecar)
        for _ in range(2):
            replay = run_process(
                [str(args.core)],
                payload,
                wall_gate + 30,
            )
            replays.append(replay)
            if not replay["completed"]:
                break

    result_replay_identical = (
        len(replays) == 2
        and replays[0]["stdout_sha256"]
        == replays[1]["stdout_sha256"]
        and replays[0]["result"] == replays[1]["result"]
    )
    completed_results = [
        replay["result"]
        for replay in replays
        if replay["result"] is not None
    ]
    observed_spans = (
        completed_results[0]["unique_span_checks"]
        if completed_results
        else None
    )
    observed_pairs = (
        completed_results[0]["unique_pair_scores"]
        if completed_results
        else None
    )
    gates = {
        "sidecar_replay_identical": sidecar_replay_identical,
        "sidecar_le_50_gib": (
            bool(generations)
            and generations[0]["sidecar_bytes"] is not None
            and generations[0]["sidecar_bytes"] <= SIDE_CAR_GATE_BYTES
        ),
        "two_core_replays_completed": len(replays) == 2 and all(
            replay["completed"] for replay in replays
        ),
        "result_replay_identical": result_replay_identical,
        "each_wall_seconds_within_gate": len(replays) == 2 and all(
            replay["elapsed_seconds"] <= wall_gate
            for replay in replays
        ),
        "peak_rss_within_gate": len(replays) == 2 and all(
            replay["peak_rss_bytes"] <= RSS_GATES[args.mode]
            for replay in replays
        ),
        "unique_spans_le_2000000": (
            observed_spans is not None and observed_spans <= SPAN_GATE
        ),
        "unique_pairs_le_50000000": (
            observed_pairs is not None and observed_pairs <= PAIR_GATE
        ),
    }
    status = "pass" if all(gates.values()) else "stop"
    output = {
        "schema_version": 1,
        "created_on": "2026-09-22",
        "audit_mode": "wp2_v6_synthetic_factorized_scale",
        "status": status,
        "mode": args.mode,
        "mode_name": MODE_NAMES[args.mode],
        "size": args.size,
        "shape": {
            "token_count": shape["n"],
            "boundary_count": len(shape["boundaries"]),
            "structural_tape_sha256": (
                shape["structural_tape_sha256"]
            ),
        },
        "parameters": {
            "max_boundary_moves": min(3, len(shape["boundaries"])),
            "simmark_dimension": 768 if args.mode == 6 else None,
            "wall_gate_seconds": wall_gate,
            "rss_gate_bytes": RSS_GATES[args.mode],
            "sidecar_gate_bytes": SIDE_CAR_GATE_BYTES,
            "span_gate": SPAN_GATE,
            "pair_gate": PAIR_GATE,
        },
        "sidecar": {
            "path": str(args.sidecar),
            "bytes": (
                generations[0]["sidecar_bytes"]
                if generations
                else None
            ),
            "sha256": (
                generations[0]["sidecar_sha256"]
                if generations
                else None
            ),
            "replay_identical": sidecar_replay_identical,
        },
        "generations": generations,
        "core_replays": replays,
        "gates": gates,
        "core": {
            "path": str(args.core),
            "sha256": sha256_file(args.core),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
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
        "status": status,
        "mode": MODE_NAMES[args.mode],
        "size": args.size,
        "sidecar_bytes": output["sidecar"]["bytes"],
        "sidecar_sha256": output["sidecar"]["sha256"],
        "replay_seconds": [
            replay["elapsed_seconds"] for replay in replays
        ],
        "peak_rss_bytes": [
            replay["peak_rss_bytes"] for replay in replays
        ],
        "unique_spans": observed_spans,
        "unique_pairs": observed_pairs,
        "gates": gates,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
