"""Measure frozen SemStamp feature throughput across a fixed batch sweep."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time


STAGE = "semstamp_certified_defense_v3_wp2_throughput1"
CANDIDATE_BATCH_SIZES = (64, 128, 256, 512)
PROBE_SLOT = "nearest_300"
CONFIRM_SLOT = "maximum_remaining"
MAX_VRAM_FRACTION = 0.80
MAX_RSS_BYTES = 16 * 1024**3
MAX_TOTAL_GPU_SECONDS = 3600
PROBE_TIMEOUT_SECONDS = 900
CONFIRM_TIMEOUT_SECONDS = 1800
RATE_TIE_FRACTION = 0.005
DETERMINISTIC_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": "0",
    "HF_HUB_OFFLINE": "1",
    "PYTHONHASHSEED": "0",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
}


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


def load_json(path):
    return json.loads(Path(path).read_text(encoding="ascii"))


def resolve_inside(root, relative):
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError(f"contract path must be relative: {relative}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"contract path escapes root: {relative}")
    return resolved


def validate_contract(root, contract_path):
    root = Path(root).resolve()
    contract_path = Path(contract_path).resolve()
    contract = load_json(contract_path)
    expected_parameters = {
        "batch_sizes": list(CANDIDATE_BATCH_SIZES),
        "confirm_slot": CONFIRM_SLOT,
        "max_rss_bytes": MAX_RSS_BYTES,
        "max_total_gpu_seconds": MAX_TOTAL_GPU_SECONDS,
        "max_vram_fraction": MAX_VRAM_FRACTION,
        "probe_slot": PROBE_SLOT,
        "rate_tie_fraction": RATE_TIE_FRACTION,
    }
    if (
        contract.get("schema_version") != 1
        or contract.get("stage") != STAGE
        or contract.get("status") != "frozen"
        or contract.get("execution_authorized") is not True
        or contract.get("parameters") != expected_parameters
        or contract.get("deterministic_environment")
        != DETERMINISTIC_ENVIRONMENT
    ):
        raise ValueError("throughput contract identity mismatch")

    observed = {}
    for relative, expected in contract["inputs"].items():
        path = resolve_inside(root, relative)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"input SHA256 mismatch for {relative}: "
                f"expected {expected}, got {actual}"
            )
        observed[relative] = actual
    runner_relative = contract["paths"]["runner"]
    if observed.get(runner_relative) != sha256_file(Path(__file__).resolve()):
        raise ValueError("running throughput script is not the frozen runner")
    return contract


def validate_environment():
    observed = {
        key: os.environ.get(key)
        for key in DETERMINISTIC_ENVIRONMENT
    }
    if observed != DETERMINISTIC_ENVIRONMENT:
        raise RuntimeError(
            "deterministic/offline environment mismatch before model import"
        )
    return observed


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


def query_sequence_sha256(texts, span_to_text_index, token_counts):
    digest = hashlib.sha256()
    for span in sorted(span_to_text_index):
        index = span_to_text_index[span]
        text_sha = sha256_bytes(texts[index].encode("utf-8"))
        digest.update(
            (
                f"{span[0]}\t{span[1]}\t{index}\t"
                f"{token_counts[span]}\t{text_sha}\n"
            ).encode("ascii")
        )
    return digest.hexdigest()


def package_versions(names):
    from importlib import metadata

    output = {}
    for name in names:
        try:
            output[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            output[name] = None
    return output


def run_candidate(args):
    validate_environment()
    root = args.root.resolve()
    contract = validate_contract(root, args.contract)

    import numpy as np
    import torch

    from audit_certified_resegmentation_wp2_v2_design import (
        load_tokenizer,
    )
    from run_certified_resegmentation_wp2_detector_equivalence import (
        verify_inputs,
    )
    from run_certified_resegmentation_wp2_v6_provider import (
        deduplicate_texts,
        dump_all_allowed_spans,
        freeze_native_topology,
        load_selected_document,
        load_semstamp,
        read_dumped_span_keys,
        semstamp_features,
    )

    if args.batch_size not in CANDIDATE_BATCH_SIZES:
        raise ValueError("batch size is outside the frozen candidate set")
    if args.slot not in {PROBE_SLOT, CONFIRM_SLOT}:
        raise ValueError("slot is outside the frozen throughput set")
    if args.slot == PROBE_SLOT and args.batch_size not in CANDIDATE_BATCH_SIZES:
        raise ValueError("invalid probe batch")

    output = args.output.resolve()
    if not output.parent.is_relative_to(root) or output.exists():
        raise ValueError("candidate output must be new and inside root")

    paths = {
        key: resolve_inside(root, value)
        for key, value in contract["paths"].items()
    }
    reference = load_json(paths[f"{args.slot}_reference"])
    reference_identity = reference["identity"]
    reference_provider = reference["provider"]

    os.environ["NLTK_DATA"] = str(paths["nltk_data"])
    import nltk

    nltk.data.path.insert(0, str(paths["nltk_data"]))
    if nltk.__version__ != "3.8.1":
        raise ValueError("NLTK version mismatch")

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    selection = load_json(paths["selection"])
    frozen, document = load_selected_document(
        root,
        selection,
        "semstamp",
        args.slot,
    )
    observed_identity = {
        "atom_count": document.atom_count,
        "contains_raw_text": False,
        "raw_text_sha256": frozen["raw_text_sha256"],
        "row_index": frozen["row_index"],
        "structural_tape_sha256": document.structural_tape_sha256,
        "unit_count": document.unit_count,
    }
    if observed_identity != reference_identity:
        raise ValueError("candidate document identity drift")

    input_identity = verify_inputs(
        root,
        paths["model_dir"],
        "semstamp",
    )
    tokenizer = load_tokenizer(paths["model_dir"], "semstamp")

    args.work_dir.mkdir(parents=True, exist_ok=False)
    dump_path = args.work_dir / "all_allowed.bin"
    topology_started = time.perf_counter()
    baseline = dump_all_allowed_spans(
        paths["v6_core"],
        document,
        5,
        dump_path,
    )
    all_spans = read_dumped_span_keys(
        dump_path,
        5,
        document.atom_count,
    )
    dump_path.unlink()
    topology = freeze_native_topology(
        document,
        "semstamp",
        tokenizer,
        all_spans,
    )
    texts, span_to_text_index = deduplicate_texts(topology["texts"])
    deduplicated_token_counts = {}
    for span, text_index in span_to_text_index.items():
        token_count = topology["token_counts"][span]
        previous = deduplicated_token_counts.setdefault(
            text_index,
            token_count,
        )
        if previous != token_count:
            raise AssertionError(
                "deduplicated text has inconsistent token counts"
            )
    topology_seconds = time.perf_counter() - topology_started

    deduplicated_input_tokens = sum(deduplicated_token_counts.values())
    topology_observed = {
        "all_query_spans_legal": topology["all_allowed"],
        "baseline_unique_pairs": baseline["unique_pair_scores"],
        "baseline_unique_spans": baseline["unique_span_checks"],
        "deduplicated_feature_texts": len(texts),
        "deduplicated_input_tokens": deduplicated_input_tokens,
        "legal_query_spans": len(topology["texts"]),
        "query_sequence_sha256": query_sequence_sha256(
            texts,
            span_to_text_index,
            topology["token_counts"],
        ),
        "topology_source": topology["topology_source"],
        "unique_query_spans": len(topology["spans"]),
    }
    expected_topology = {
        "all_query_spans_legal": (
            reference_provider["all_query_spans_legal"]
        ),
        "baseline_unique_pairs": reference["topology"][
            "baseline_unique_pairs"
        ],
        "baseline_unique_spans": reference["topology"][
            "baseline_unique_spans"
        ],
        "deduplicated_feature_texts": (
            reference_provider["deduplicated_feature_texts"]
        ),
        "deduplicated_input_tokens": (
            reference_provider["deduplicated_input_tokens"]
        ),
        "legal_query_spans": reference_provider["legal_query_spans"],
        "topology_source": reference_provider["topology_source"],
        "unique_query_spans": reference_provider["unique_query_spans"],
    }
    for key, expected in expected_topology.items():
        if topology_observed[key] != expected:
            raise ValueError(f"candidate topology drift at {key}")

    model_load_started = time.perf_counter()
    model, relation = load_semstamp(
        root,
        paths["model_dir"],
        args.batch_size,
    )
    if len(relation) != 64 or any(value not in (0, 1) for value in relation):
        raise AssertionError("invalid frozen SemStamp relation table")
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - model_load_started
    torch.cuda.reset_peak_memory_stats()

    replay_rows = []
    first_features = None
    for replay_index in range(2):
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            features = semstamp_features(model, texts)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if features.shape != (len(texts),):
            raise AssertionError("unexpected SemStamp feature shape")
        if not np.all(features <= 7):
            raise AssertionError("SemStamp feature escaped region range")
        feature_sha = sha256_bytes(
            np.ascontiguousarray(features).tobytes(order="C")
        )
        if first_features is None:
            first_features = features.copy()
        elif not np.array_equal(first_features, features):
            raise AssertionError("provider replay changed region bytes")
        replay_rows.append({
            "elapsed_seconds": elapsed,
            "feature_sha256": feature_sha,
            "replay_index": replay_index,
        })

    reference_feature_sha = reference_provider["first_feature_sha256"]
    total_feature_seconds = sum(
        row["elapsed_seconds"] for row in replay_rows
    )
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    gpu_total = int(torch.cuda.get_device_properties(0).total_memory)
    peak_rss = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    gates = {
        "all_query_spans_legal": topology_observed[
            "all_query_spans_legal"
        ],
        "cuda_reserved_le_80_percent": (
            peak_reserved <= MAX_VRAM_FRACTION * gpu_total
        ),
        "feature_shape_exact": first_features.shape == (len(texts),),
        "input_identity_exact": observed_identity == reference_identity,
        "process_rss_le_16_gib": peak_rss <= MAX_RSS_BYTES,
        "reference_region_sha256_exact": all(
            row["feature_sha256"] == reference_feature_sha
            for row in replay_rows
        ),
        "replay_region_bytes_exact": (
            replay_rows[0]["feature_sha256"]
            == replay_rows[1]["feature_sha256"]
        ),
        "topology_counts_exact": all(
            topology_observed[key] == expected
            for key, expected in expected_topology.items()
        ),
    }
    result = {
        "audit_mode": "semstamp_feature_throughput_candidate",
        "batch_size": args.batch_size,
        "contains_raw_text": False,
        "deterministic_environment": validate_environment(),
        "environment": {
            "gpu_name": torch.cuda.get_device_name(0),
            "package_versions": package_versions((
                "datasets",
                "nltk",
                "numpy",
                "sentence-transformers",
                "torch",
                "transformers",
            )),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch_cuda": torch.version.cuda,
        },
        "execution_attestation": {
            "certificate_computations": 0,
            "detector_score_computations": 0,
            "generation_calls": 0,
            "model_loads": 1,
            "network_provider_calls": 0,
            "provider_feature_replays": 2,
            "threshold_reads": 0,
        },
        "gates": gates,
        "identity": observed_identity,
        "inputs": {
            "contract_sha256": sha256_file(args.contract),
            "official": input_identity,
            "reference_artifact_sha256": sha256_file(
                paths[f"{args.slot}_reference"]
            ),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        },
        "model_load_seconds": model_load_seconds,
        "provider": {
            "feature_seconds_per_input_token": (
                total_feature_seconds / deduplicated_input_tokens
            ),
            "reference_feature_sha256": reference_feature_sha,
            "replays": replay_rows,
            "total_feature_seconds": total_feature_seconds,
        },
        "resources": {
            "cuda_peak_allocated_bytes": peak_allocated,
            "cuda_peak_reserved_bytes": peak_reserved,
            "gpu_total_bytes": gpu_total,
            "process_peak_rss_bytes": peak_rss,
        },
        "schema_version": 1,
        "slot": args.slot,
        "stage": STAGE,
        "status": "pass" if all(gates.values()) else "stop",
        "topology": topology_observed,
        "topology_seconds": topology_seconds,
    }
    atomic_write_json(output, result)
    print(json.dumps({
        "batch_size": args.batch_size,
        "feature_seconds_per_input_token": result["provider"][
            "feature_seconds_per_input_token"
        ],
        "slot": args.slot,
        "status": result["status"],
    }, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(3)


def candidate_paths(output_dir, batch_size, slot):
    stem = f"{slot}_batch_{batch_size}"
    return {
        "artifact": output_dir / f"{stem}.json",
        "log": output_dir / f"{stem}.log",
        "telemetry": output_dir / f"{stem}_nvidia_dmon.log",
        "work": output_dir / "sealed" / stem,
    }


def run_child(root, contract_path, output_dir, batch_size, slot, timeout):
    paths = candidate_paths(output_dir, batch_size, slot)
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
    paths["telemetry"].parent.mkdir(parents=True, exist_ok=True)
    paths["work"].parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "candidate",
        "--root",
        str(root),
        "--contract",
        str(contract_path),
        "--batch-size",
        str(batch_size),
        "--slot",
        slot,
        "--work-dir",
        str(paths["work"]),
        "--output",
        str(paths["artifact"]),
    ]
    environment = os.environ.copy()
    environment.update(DETERMINISTIC_ENVIRONMENT)
    started = time.perf_counter()
    timed_out = False
    with paths["telemetry"].open("wb") as telemetry_handle:
        monitor = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "pucm", "-d", "1"],
            stdout=telemetry_handle,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        try:
            time.sleep(1)
            with paths["log"].open("wb") as log_handle:
                try:
                    completed = subprocess.run(
                        command,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        env=environment,
                        timeout=timeout,
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

    summary = {
        "artifact_sha256": (
            sha256_file(paths["artifact"])
            if paths["artifact"].exists()
            else None
        ),
        "batch_size": batch_size,
        "elapsed_seconds": elapsed_seconds,
        "feature_seconds_per_input_token": None,
        "log_sha256": sha256_file(paths["log"]),
        "passed": False,
        "returncode": returncode,
        "slot": slot,
        "telemetry_sha256": sha256_file(paths["telemetry"]),
        "timed_out": timed_out,
    }
    if returncode == 0 and paths["artifact"].exists():
        artifact = load_json(paths["artifact"])
        passed = (
            artifact.get("stage") == STAGE
            and artifact.get("status") == "pass"
            and artifact.get("batch_size") == batch_size
            and artifact.get("slot") == slot
            and artifact.get("gates")
            and all(artifact["gates"].values())
        )
        summary["passed"] = bool(passed)
        if passed:
            summary["feature_seconds_per_input_token"] = artifact[
                "provider"
            ]["feature_seconds_per_input_token"]
            summary["total_feature_seconds"] = artifact["provider"][
                "total_feature_seconds"
            ]
            summary["cuda_peak_reserved_bytes"] = artifact["resources"][
                "cuda_peak_reserved_bytes"
            ]
            summary["process_peak_rss_bytes"] = artifact["resources"][
                "process_peak_rss_bytes"
            ]
            summary["feature_sha256"] = artifact["provider"][
                "reference_feature_sha256"
            ]
            summary["query_sequence_sha256"] = artifact["topology"][
                "query_sequence_sha256"
            ]
    return summary


def select_winner(results):
    passed = [row for row in results if row["passed"]]
    if not passed:
        return None
    minimum_rate = min(
        row["feature_seconds_per_input_token"] for row in passed
    )
    near_minimum = [
        row for row in passed
        if row["feature_seconds_per_input_token"]
        <= minimum_rate * (1 + RATE_TIE_FRACTION)
    ]
    return min(near_minimum, key=lambda row: row["batch_size"])


def run_sweep(args):
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_relative_to(root):
        raise ValueError("output directory must be inside root")
    contract = validate_contract(root, args.contract)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "THROUGHPUT_RESULT.json"
    if result_path.exists():
        raise ValueError("throughput result already exists")

    probe_results = []
    for batch_size in CANDIDATE_BATCH_SIZES:
        row = run_child(
            root,
            args.contract,
            output_dir,
            batch_size,
            PROBE_SLOT,
            PROBE_TIMEOUT_SECONDS,
        )
        probe_results.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    winner = select_winner(probe_results)
    confirm = None
    if winner is not None:
        confirm = run_child(
            root,
            args.contract,
            output_dir,
            winner["batch_size"],
            CONFIRM_SLOT,
            CONFIRM_TIMEOUT_SECONDS,
        )
        print(json.dumps(confirm, sort_keys=True), flush=True)

    baseline = next(
        row for row in probe_results
        if row["batch_size"] == CANDIDATE_BATCH_SIZES[0]
    )
    speedup = (
        baseline["feature_seconds_per_input_token"]
        / winner["feature_seconds_per_input_token"]
        if baseline["passed"] and winner is not None
        else None
    )
    total_gpu_seconds = sum(
        row.get("total_feature_seconds", 0.0)
        for row in probe_results
    ) + (
        confirm.get("total_feature_seconds", 0.0)
        if confirm is not None
        else 0.0
    )
    sequence_hashes = {
        row["query_sequence_sha256"]
        for row in probe_results
        if row["passed"]
    }
    gates = {
        "baseline_batch_64_passed": baseline["passed"],
        "confirm_maximum_remaining_passed": (
            confirm is not None and confirm["passed"]
        ),
        "probe_region_sha_exact_for_every_passed_candidate": all(
            row.get("feature_sha256")
            == contract["references"][PROBE_SLOT]["feature_sha256"]
            for row in probe_results
            if row["passed"]
        ),
        "probe_sequence_identity_exact_across_passed_candidates": (
            bool(sequence_hashes) and len(sequence_hashes) == 1
        ),
        "total_feature_gpu_seconds_le_3600": (
            total_gpu_seconds <= MAX_TOTAL_GPU_SECONDS
        ),
        "winner_selected": winner is not None,
    }
    result = {
        "audit_mode": "semstamp_feature_throughput_sweep",
        "baseline_batch_size": 64,
        "contains_raw_text": False,
        "contract_sha256": sha256_file(args.contract),
        "execution_attestation": {
            "certificate_computations": 0,
            "detector_score_computations": 0,
            "generation_calls": 0,
            "model_loads_upper_bound": (
                len(probe_results) + int(confirm is not None)
            ),
            "network_provider_calls": 0,
            "threshold_reads": 0,
        },
        "gates": gates,
        "maximum_confirmation": confirm,
        "parameters": contract["parameters"],
        "probe_results": probe_results,
        "schema_version": 1,
        "stage": STAGE,
        "status": "go" if all(gates.values()) else "stop",
        "total_feature_gpu_seconds": total_gpu_seconds,
        "winner_batch_size": (
            winner["batch_size"] if winner is not None else None
        ),
        "winner_probe_speedup_over_batch_64": speedup,
    }
    atomic_write_json(result_path, result)
    print(json.dumps({
        "gates": gates,
        "status": result["status"],
        "winner_batch_size": result["winner_batch_size"],
        "winner_probe_speedup_over_batch_64": speedup,
    }, sort_keys=True))
    if result["status"] != "go":
        raise SystemExit(3)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    candidate = subparsers.add_parser("candidate")
    candidate.add_argument("--root", type=Path, required=True)
    candidate.add_argument("--contract", type=Path, required=True)
    candidate.add_argument("--batch-size", type=int, required=True)
    candidate.add_argument(
        "--slot",
        choices=(PROBE_SLOT, CONFIRM_SLOT),
        required=True,
    )
    candidate.add_argument("--work-dir", type=Path, required=True)
    candidate.add_argument("--output", type=Path, required=True)

    sweep = subparsers.add_parser("sweep")
    sweep.add_argument("--root", type=Path, required=True)
    sweep.add_argument("--contract", type=Path, required=True)
    sweep.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "candidate":
        run_candidate(args)
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
