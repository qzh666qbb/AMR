"""Run the frozen WP2-v7 SimMark batch-size probe sweep."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


PROTOCOL_SHA256 = (
    "436a0e8fdb7aa3d161cdaa5632d747a2467b6a506a82a55650c4759b44dafcb3"
)
BASE_PROTOCOL_SHA256 = (
    "ddab1e0f8d3714dda22ae383395ea91e14ff10ac55d59e63b0708806d533a350"
)
SELECTION_SHA256 = (
    "413bdd37780b9c3e9e41807fcf80b2e8843951fb26af4570d81312544649e399"
)
PROVIDER_RUNNER_SHA256 = (
    "d4715aef3516493983748ba29e0f89bf1886fdf791ad7610c4c9dfe530500a14"
)
CORE_SHA256 = (
    "d724a038f72d766b477cb1bd95c8318bc0032c3718e067005bd05921944847fc"
)
BATCH_SIZES = (64, 128, 256, 384)
MIN_PROBE_SPEEDUP = 1.15
GPU_BUDGET_HOURS = 2.0
PROBE_SLOT = "nearest_300"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_sha(path, expected, label):
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA256 mismatch: expected {expected}, got {observed}"
        )


def active_gpu_processes():
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "nvidia-smi process query failed: " + completed.stderr.strip()
        )
    return [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    ]


def candidate_paths(artifact_dir, batch_size):
    stem = f"simmark_probe_batch_{batch_size}"
    return {
        "artifact": artifact_dir / f"{stem}.json",
        "log": artifact_dir / f"{stem}.log",
        "telemetry": artifact_dir / f"{stem}_nvidia_dmon.log",
    }


def run_candidate(args, batch_size):
    paths = candidate_paths(args.artifact_dir, batch_size)
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise ValueError(
            "candidate outputs already exist: " + ", ".join(existing)
        )
    processes = active_gpu_processes()
    if processes:
        raise RuntimeError(
            "unexpected active GPU process before candidate: "
            + "; ".join(processes)
        )

    candidate_work_dir = args.work_dir / f"batch_{batch_size}"
    candidate_work_dir.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(args.provider_runner),
        "--root",
        str(args.root),
        "--system",
        "simmark",
        "--slot",
        PROBE_SLOT,
        "--model-dir",
        str(args.model_dir),
        "--nltk-data",
        str(args.nltk_data),
        "--protocol",
        str(args.base_protocol),
        "--selection",
        str(args.selection),
        "--core",
        str(args.core),
        "--work-dir",
        str(candidate_work_dir),
        "--output",
        str(paths["artifact"]),
        "--batch-size",
        str(batch_size),
    ]

    started = time.perf_counter()
    timed_out = False
    with paths["telemetry"].open("wb") as telemetry_handle:
        monitor = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "pucm", "-d", "1"],
            stdout=telemetry_handle,
            stderr=subprocess.STDOUT,
        )
        try:
            time.sleep(1.0)
            with paths["log"].open("wb") as log_handle:
                try:
                    completed = subprocess.run(
                        command,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        timeout=args.timeout_seconds,
                        check=False,
                    )
                    returncode = completed.returncode
                except subprocess.TimeoutExpired:
                    timed_out = True
                    returncode = 124
        finally:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
    elapsed_seconds = time.perf_counter() - started

    output = {
        "batch_size": batch_size,
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": elapsed_seconds,
        "artifact_path": str(paths["artifact"]),
        "artifact_exists": paths["artifact"].exists(),
        "artifact_sha256": (
            sha256_file(paths["artifact"])
            if paths["artifact"].exists()
            else None
        ),
        "log_path": str(paths["log"]),
        "log_sha256": sha256_file(paths["log"]),
        "telemetry_path": str(paths["telemetry"]),
        "telemetry_sha256": sha256_file(paths["telemetry"]),
        "passed": False,
        "feature_seconds_per_input_token": None,
    }
    if returncode != 0 or not paths["artifact"].exists():
        return output

    artifact = load_json(paths["artifact"])
    gates = artifact.get("gates", {})
    passed = (
        artifact.get("audit_mode") == "wp2_v6_real_feature_provider"
        and artifact.get("system") == "simmark"
        and artifact.get("selection_slot") == PROBE_SLOT
        and artifact.get("status") == "pass"
        and artifact.get("provider", {}).get("batch_size") == batch_size
        and gates
        and all(gates.values())
        and artifact.get("inputs", {}).get("protocol_sha256")
        == BASE_PROTOCOL_SHA256
        and artifact.get("inputs", {}).get("selection_sha256")
        == SELECTION_SHA256
        and artifact.get("inputs", {}).get("core_sha256") == CORE_SHA256
    )
    output["passed"] = bool(passed)
    if passed:
        provider = artifact["provider"]
        replay_seconds = artifact["timing"]["stages_seconds"][
            "two_batch_provider_replays"
        ]
        output["feature_seconds_per_input_token"] = (
            replay_seconds / provider["deduplicated_input_tokens"]
        )
        output["provider_replay_seconds"] = replay_seconds
        output["deduplicated_input_tokens"] = provider[
            "deduplicated_input_tokens"
        ]
        output["feature_max_abs_error"] = artifact["equivalence"][
            "feature_max_abs_error"
        ]
        output["cosine_max_abs_error"] = artifact["equivalence"][
            "cosine_max_abs_error"
        ]
        output["pair_value_max_abs_error"] = artifact["equivalence"][
            "pair_value_max_abs_error"
        ]
        output["core_score_max_abs_error"] = artifact["equivalence"][
            "core_score_max_abs_error"
        ]
        output["cuda_peak_reserved_bytes"] = artifact["resources"][
            "cuda_peak_reserved_bytes"
        ]
        output["process_peak_rss_bytes"] = artifact["resources"][
            "process_peak_rss_bytes"
        ]
    return output


def select_winner(results):
    baseline = next(
        row for row in results if row["batch_size"] == BATCH_SIZES[0]
    )
    candidates = [
        row
        for row in results
        if row["batch_size"] != BATCH_SIZES[0] and row["passed"]
    ]
    if not baseline["passed"] or not candidates:
        return baseline, None, None
    minimum_rate = min(
        row["feature_seconds_per_input_token"] for row in candidates
    )
    near_minimum = [
        row
        for row in candidates
        if row["feature_seconds_per_input_token"] <= minimum_rate * 1.005
    ]
    winner = min(near_minimum, key=lambda row: row["batch_size"])
    speedup = (
        baseline["feature_seconds_per_input_token"]
        / winner["feature_seconds_per_input_token"]
    )
    return baseline, winner, speedup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--v7-protocol", type=Path, required=True)
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--provider-runner", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()

    for path, expected, label in (
        (args.v7_protocol, PROTOCOL_SHA256, "v7 protocol"),
        (args.base_protocol, BASE_PROTOCOL_SHA256, "v6 base protocol"),
        (args.selection, SELECTION_SHA256, "selection"),
        (args.provider_runner, PROVIDER_RUNNER_SHA256, "provider runner"),
        (args.core, CORE_SHA256, "v6 core"),
    ):
        verify_sha(path, expected, label)
    if args.output.exists():
        raise ValueError(f"summary output already exists: {args.output}")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for batch_size in BATCH_SIZES:
        result = run_candidate(args, batch_size)
        results.append(result)
        print(json.dumps({
            "batch_size": batch_size,
            "returncode": result["returncode"],
            "passed": result["passed"],
            "feature_seconds_per_input_token": result[
                "feature_seconds_per_input_token"
            ],
        }, sort_keys=True), flush=True)

    baseline, winner, speedup = select_winner(results)
    total_elapsed_seconds = sum(row["elapsed_seconds"] for row in results)
    gates = {
        "baseline_passed": baseline["passed"],
        "at_least_one_optimized_candidate_passed": winner is not None,
        "winner_speedup_ge_1_15": (
            speedup is not None and speedup >= MIN_PROBE_SPEEDUP
        ),
        "probe_sweep_within_2_gpu_hours": (
            total_elapsed_seconds / 3600 <= GPU_BUDGET_HOURS
        ),
    }
    output = {
        "schema_version": 1,
        "created_on": "2026-09-23",
        "audit_mode": "wp2_v7_simmark_batch_probe",
        "status": "probe-ready" if all(gates.values()) else "stop",
        "inputs": {
            "v7_protocol_sha256": PROTOCOL_SHA256,
            "base_protocol_sha256": BASE_PROTOCOL_SHA256,
            "selection_sha256": SELECTION_SHA256,
            "provider_runner_sha256": PROVIDER_RUNNER_SHA256,
            "core_sha256": CORE_SHA256,
        },
        "parameters": {
            "batch_sizes": list(BATCH_SIZES),
            "minimum_probe_speedup": MIN_PROBE_SPEEDUP,
            "gpu_budget_hours": GPU_BUDGET_HOURS,
            "timeout_seconds": args.timeout_seconds,
        },
        "results": results,
        "baseline_batch_size": baseline["batch_size"],
        "winner_batch_size": (
            winner["batch_size"] if winner is not None else None
        ),
        "winner_speedup": speedup,
        "total_elapsed_seconds": total_elapsed_seconds,
        "gates": gates,
        "execution_attestation": {
            "model_load_attempts_upper_bound": len(results),
            "detector_document_score_calls": 0,
            "simmark_gamma_reads_or_fits": 0,
            "calibration_model_feature_reads": 0,
            "test_model_feature_reads": 0,
            "positive_model_feature_reads": 0,
            "watermark_generation_calls": 0,
            "provider_api_calls": 0,
        },
        "contains_raw_text": False,
    }
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "status": output["status"],
        "winner_batch_size": output["winner_batch_size"],
        "winner_speedup": output["winner_speedup"],
        "gates": gates,
    }, sort_keys=True))
    if output["status"] != "probe-ready":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
