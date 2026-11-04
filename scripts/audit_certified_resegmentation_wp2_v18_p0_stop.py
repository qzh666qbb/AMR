"""Independently verify and seal the frozen WP2-v18 P0 Stop decision."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile


STAGE = "WP2-v18-B1-P0-stop-audit"
PROTOCOL_SHA256 = (
    "0da27bc53e06e76ba5f1d25aefe3bf787893c0fa4421113dddb1a995892ab82d"
)
CONTRACT_SHA256 = (
    "117ea262c253299f2762f167a2ebfcfe4145efb76fb3eee48b54f5f0b6de2481"
)
RUNNER_SHA256 = (
    "f251b6aefbb719e5a181b0142207897e3e294eb263ea799838168dab4bde9f29"
)
B0_PLAN_SHA256 = (
    "a1427a9978751c756ad5072aeb45bb6cb393713d8339d62ff6ef4397591fc4e2"
)
PROFILE_SHA256 = (
    "4701a4b6d77f2d0c20e189853ec896d70ac205df05659ee38e576b5010119f6d"
)
CORE_SHA256 = (
    "d724a038f72d766b477cb1bd95c8318bc0032c3718e067005bd05921944847fc"
)
TOTAL_ROWS = 3000
SPAN_GATE = 2_000_000
PAIR_GATE = 50_000_000
LEGACY_PAIR_DIAGNOSTIC = 5_000_000
CACHE_GATE_BYTES = 50 * 1024**3
MIN_FREE_BYTES = 5 * 1024**3
GPU_HOUR_GATE = 72.0
FORBIDDEN_KEYS = {
    "text",
    "raw_text",
    "raw_value",
    "prompt",
    "generated_text",
    "original_units",
    "structural_tape",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line
    ]


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(canonical_bytes(value))
            handle.flush()
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def no_forbidden_keys(value):
    if isinstance(value, dict):
        return (
            not FORBIDDEN_KEYS.intersection(value)
            and all(no_forbidden_keys(item) for item in value.values())
        )
    if isinstance(value, list):
        return all(no_forbidden_keys(item) for item in value)
    return True


def paths_for(root, output_dir):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    v17 = (
        root
        / "experiments/certified_resegmentation_wp2_v17_b0_detector_health1"
    )
    return {
        "root": root,
        "output": output_dir,
        "protocol": output_dir / "PROTOCOL.md",
        "contract": output_dir / "B1_CONTRACT.json",
        "runner": root / "scripts/run_certified_resegmentation_wp2_v18_b1.py",
        "b0_plan": v17 / "B0_PLAN.jsonl",
        "profile": (
            root
            / "experiments/certified_resegmentation_wp2_v6_feature1"
            / "semstamp_workload.json"
        ),
        "core": root / "build/wp2_v6/certified_resegmentation_wp2_v6_core",
        "rows": output_dir / "P0_WORKLOAD_ROWS.jsonl",
        "timings": output_dir / "sealed/p0_workload_timings.jsonl",
        "readiness": output_dir / "P0_READINESS.json",
        "heartbeat": output_dir / "B1_HEARTBEAT.json",
        "gate": output_dir / "B1_GATE_RESULT.json",
        "audit": output_dir / "P0_STOP_AUDIT.json",
    }


def validate_static_files(paths):
    expected = {
        "protocol": PROTOCOL_SHA256,
        "contract": CONTRACT_SHA256,
        "runner": RUNNER_SHA256,
        "b0_plan": B0_PLAN_SHA256,
        "profile": PROFILE_SHA256,
        "core": CORE_SHA256,
    }
    checks = {}
    for key, digest in expected.items():
        checks[f"{key}_sha_exact"] = (
            paths[key].is_file() and sha256_file(paths[key]) == digest
        )
    return checks


def validate_rows(rows, plan):
    if len(rows) != TOTAL_ROWS or len(plan) != TOTAL_ROWS:
        raise ValueError("P0 workload/plan must contain 3,000 rows")
    for index, (row, planned) in enumerate(zip(rows, plan)):
        if (
            row.get("schema_version") != 1
            or row.get("protocol_sha256") != PROTOCOL_SHA256
            or row.get("plan_index") != index
            or row.get("cohort") != planned["cohort"]
            or row.get("cohort_index") != planned["cohort_index"]
            or row.get("raw_text_sha256") != planned["raw_text_sha256"]
            or row.get("unit_count") != planned["unit_count"]
            or row.get("edge_count") != planned["unit_count"] - 1
            or row.get("contains_raw_text") is not False
            or set(row.get("radii", {})) != {"0", "1", "2", "3"}
            or not no_forbidden_keys(row)
        ):
            raise ValueError(f"invalid P0 workload identity at row {index}")
        for radius in range(4):
            value = row["radii"][str(radius)]
            if (
                value.get("requested_radius") != radius
                or value.get("effective_radius")
                != min(radius, row["edge_count"])
                or not 1 <= value["legal_span_count"]
                <= value["unique_span_count"]
                <= value["all_allowed_unique_spans"]
                or not 0 <= value["unique_pair_count"]
                <= value["all_allowed_unique_pairs"]
                or value["threat_set_size"] < 1
            ):
                raise ValueError(
                    f"invalid P0 topology at row {index}, k={radius}"
                )


def recompute_counts(rows):
    radius_values = [
        value for row in rows for value in row["radii"].values()
    ]
    return {
        "cohorts": dict(Counter(row["cohort"] for row in rows)),
        "total_maximum_legal_input_tokens": sum(
            row["resource_counts"]["maximum_legal_input_tokens"]
            for row in rows
        ),
        "total_radius_unique_spans": sum(
            value["unique_span_count"] for value in radius_values
        ),
        "total_radius_unique_pairs": sum(
            value["unique_pair_count"] for value in radius_values
        ),
        "max_document_pruned_spans": max(
            value["unique_span_count"] for value in radius_values
        ),
        "max_document_pruned_pairs": max(
            value["unique_pair_count"] for value in radius_values
        ),
        "max_document_all_allowed_spans": max(
            value["all_allowed_unique_spans"] for value in radius_values
        ),
        "max_document_all_allowed_pairs": max(
            value["all_allowed_unique_pairs"] for value in radius_values
        ),
        "max_document_sidecar_bytes": max(
            row["resource_counts"]["maximum_sidecar_bytes"] for row in rows
        ),
        "documents_over_legacy_5m_pair_diagnostic": sum(
            max(
                value["unique_pair_count"]
                for value in row["radii"].values()
            )
            > LEGACY_PAIR_DIAGNOSTIC
            for row in rows
        ),
    }


def recompute_runtime(rows, timings, profiles):
    if len(timings) != TOTAL_ROWS:
        raise ValueError("P0 timing row count mismatch")
    for index, timing in enumerate(timings):
        if (
            timing.get("plan_index") != index
            or not math.isfinite(timing.get("elapsed_seconds", math.nan))
            or timing["elapsed_seconds"] < 0
        ):
            raise ValueError(f"invalid P0 timing at row {index}")
    components = {
        key: sum(row["estimate"][key] for row in rows)
        for key in (
            "feature_seconds",
            "sidecar_seconds",
            "core_seconds",
            "total_gpu_seconds_without_topology",
        )
    }
    topology = sum(row["elapsed_seconds"] for row in timings)
    model_load = max(row["model_load_seconds"] for row in profiles)
    total = 2 * (
        topology + components["total_gpu_seconds_without_topology"]
    ) + model_load
    return {
        "replay_count": 2,
        "measured_topology_seconds_per_replay": topology,
        **components,
        "model_load_seconds": model_load,
        "conservative_two_replay_seconds": total,
        "conservative_two_replay_gpu_hours": total / 3600,
    }


def recompute_gates(counts, runtime, readiness):
    return {
        "all_3000_rows_completed": True,
        "all_identity_and_structure_exact": True,
        "all_renderers_restore_original_units": True,
        "max_pruned_spans_le_2000000": (
            counts["max_document_pruned_spans"] <= SPAN_GATE
        ),
        "max_pruned_pairs_le_50000000": (
            counts["max_document_pruned_pairs"] <= PAIR_GATE
        ),
        "max_all_allowed_spans_le_2000000": (
            counts["max_document_all_allowed_spans"] <= SPAN_GATE
        ),
        "max_all_allowed_pairs_le_50000000": (
            counts["max_document_all_allowed_pairs"] <= PAIR_GATE
        ),
        "max_sidecar_le_50_gib": (
            counts["max_document_sidecar_bytes"] <= CACHE_GATE_BYTES
        ),
        "free_disk_ge_5_gib": (
            readiness["resource_gate"]["observed_free_bytes"]
            >= MIN_FREE_BYTES
        ),
        "two_replay_estimate_le_72_gpu_hours": (
            runtime["conservative_two_replay_gpu_hours"]
            <= GPU_HOUR_GATE
        ),
    }


def top_pair_rows(rows, limit=20):
    ranked = []
    for row in rows:
        radius, value = max(
            row["radii"].items(),
            key=lambda item: item[1]["unique_pair_count"],
        )
        ranked.append({
            "plan_index": row["plan_index"],
            "cohort": row["cohort"],
            "cohort_index": row["cohort_index"],
            "raw_text_sha256": row["raw_text_sha256"],
            "atom_count": row["atom_count"],
            "unit_count": row["unit_count"],
            "radius": int(radius),
            "unique_span_count": value["unique_span_count"],
            "unique_pair_count": value["unique_pair_count"],
            "all_allowed_unique_pairs": value[
                "all_allowed_unique_pairs"
            ],
        })
    return sorted(
        ranked,
        key=lambda row: row["unique_pair_count"],
        reverse=True,
    )[:limit]


def audit(paths):
    readiness = read_json(paths["readiness"])
    heartbeat = read_json(paths["heartbeat"])
    gate = read_json(paths["gate"])
    plan = read_jsonl(paths["b0_plan"])
    rows = read_jsonl(paths["rows"])
    timings = read_jsonl(paths["timings"])
    profile = read_json(paths["profile"])
    validate_rows(rows, plan)
    counts = recompute_counts(rows)
    runtime = recompute_runtime(
        rows, timings, profile["inputs"]["provider_profiles"]
    )
    gates = recompute_gates(counts, runtime, readiness)
    failed_gates = sorted(key for key, value in gates.items() if not value)
    certificate_artifacts = [
        path.name
        for pattern in (
            "B1_RESULT.json",
            "B1_RUNTIME.json",
            "B1_ROWS_REPLAY*.jsonl",
        )
        for path in paths["output"].glob(pattern)
    ]
    replay_directories = sorted(
        path.name
        for path in (paths["output"] / "sealed").glob("replay*")
        if path.is_dir()
    )
    checks = {
        **validate_static_files(paths),
        "workload_rows_sha_exact": (
            readiness["workload_rows"]["sha256"]
            == sha256_file(paths["rows"])
        ),
        "workload_rows_count_exact": len(rows) == TOTAL_ROWS,
        "timing_rows_count_exact": len(timings) == TOTAL_ROWS,
        "counts_recomputed_exact": readiness["counts"] == counts,
        "runtime_recomputed_exact": (
            readiness["runtime_upper_bound"] == runtime
        ),
        "gates_recomputed_exact": readiness["gates"] == gates,
        "stop_decision_exact": (
            readiness["status"] == "stop"
            and readiness["terminal_reason"] == "b1_readiness_stop"
            and bool(failed_gates)
        ),
        "expected_frozen_gates_failed": failed_gates == [
            "max_all_allowed_pairs_le_50000000",
            "max_pruned_pairs_le_50000000",
            "two_replay_estimate_le_72_gpu_hours",
        ],
        "automatic_gate_stopped_execution": (
            gate.get("status") == "stop"
            and gate.get("terminal_reason") == "readiness_not_go"
        ),
        "heartbeat_terminal_stop": (
            heartbeat.get("status") == "stop"
            and heartbeat.get("completed") == TOTAL_ROWS
        ),
        "score_blind_attestation_zero": all(
            readiness["execution_attestation"].get(key) == 0
            for key in (
                "detector_source_imports",
                "detector_model_loads",
                "region_computations",
                "certificate_score_computations",
                "threshold_computations",
                "generation_calls",
            )
        ),
        "no_certificate_artifacts": not certificate_artifacts,
        "no_replay_directories": not replay_directories,
        "public_outputs_contain_no_raw_text": all(
            no_forbidden_keys(value)
            for value in (readiness, heartbeat, gate, rows)
        ),
        "frozen_protocol_contract_runner_bound": (
            readiness["protocol_sha256"] == PROTOCOL_SHA256
            and readiness["contract_sha256"] == CONTRACT_SHA256
            and readiness["runner_sha256"] == RUNNER_SHA256
            and readiness["core_executable_sha256"] == CORE_SHA256
        ),
    }
    return {
        "schema_version": 1,
        "stage": STAGE,
        "status": "pass" if all(checks.values()) else "fail",
        "decision": "stop",
        "terminal_reason": "b1_readiness_stop",
        "protocol_sha256": PROTOCOL_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "readiness_sha256": sha256_file(paths["readiness"]),
        "workload_rows_sha256": sha256_file(paths["rows"]),
        "timings_sha256": sha256_file(paths["timings"]),
        "counts": counts,
        "runtime_upper_bound": runtime,
        "failed_gates": failed_gates,
        "top_pair_rows": top_pair_rows(rows),
        "certificate_artifacts": certificate_artifacts,
        "replay_directories": replay_directories,
        "checks": checks,
        "audited_at": utc_now(),
        "contains_raw_text": False,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = paths_for(args.root, args.output_dir)
    result = audit(paths)
    atomic_write_json(paths["audit"], result)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
