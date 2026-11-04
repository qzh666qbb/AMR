"""Run the score-blind WP4 fixed-N subset lower-bound correction."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import time


STAGE = "semstamp_certified_defense_v2_wp4_resource_preflight_correction"
PROTOCOL_SHA256 = (
    "f4083699cd963e4913c4103d9ef1cf3d5798d48a1eda4f79e0d70831e24f2361"
)
WP1_THREAT_SHA256 = (
    "ae58c2830fc71f4fc8c4c732f33966f58f5648e6e87c9dedff259910ed3f7bb4"
)
WP2_RESULT_SHA256 = (
    "bc36969f6c482c68d89895dc059c27b6a74f3ff828af8f8c11919f3d1ec98c46"
)
V2_ROWS_SHA256 = (
    "8625f6b074150d4ec099b9e57da00d6b08167b42f25ad95d34b975682e5a7474"
)
V18_ROWS_SHA256 = (
    "a6e76b846981b12cf2eab197f81931f51a63309582167079d1ac926386ececc4"
)
INVALIDATION_SHA256 = (
    "46e42adc1fe0aae9bb2e322d3dc112e2c530f91c6c64d9fd2a2eae4bff55fae6"
)
LINUX_CORE_SHA256 = (
    "d724a038f72d766b477cb1bd95c8318bc0032c3718e067005bd05921944847fc"
)
PLAN_INDEX = 800
RADIUS = 3
PAIR_GATE = 50_000_000
EXPECTED_SPANS = 769_113
EXPECTED_PAIRS = 83_540_900
EXPECTED_RAW_SHA256 = (
    "bcd44daf7325ffafccb45f6cb6d3d5dcc91012724800d65980054a175d588d68"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


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


def canonical_digest(value):
    return sha256_bytes(canonical_bytes(value))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl_row(path, index):
    with Path(path).open("r", encoding="utf-8") as handle:
        for current, line in enumerate(handle):
            if current == index:
                return json.loads(line)
    raise ValueError(f"missing JSONL row {index}: {path}")


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


def paths_for(root, output_dir):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_relative_to(root):
        raise ValueError("output directory must remain inside root")
    return {
        "root": root,
        "output_dir": output_dir,
        "protocol": output_dir / "PROTOCOL.md",
        "contract": output_dir / "CONTRACT.json",
        "authorization": output_dir / "CORRECTION_EXECUTION_AUTHORIZATION.json",
        "result": output_dir / "CORRECTIVE_RESULT.json",
        "runner": Path(__file__).resolve(),
        "wp1_threat": (
            root
            / "experiments/semstamp_certified_defense_v2_wp1_threat_model1"
            / "THREAT_MODEL.md"
        ),
        "wp2_result": (
            root
            / "experiments/semstamp_certified_defense_v2_wp2_algorithm1"
            / "RESULT.json"
        ),
        "v2_rows": (
            root
            / "experiments/semstamp_certified_defense_v2_wp3_detector_health1"
            / "P0_RENDERER_ROWS_REPLAY0.jsonl"
        ),
        "v18_rows": (
            root
            / "experiments/certified_resegmentation_wp2_v18_b1_exact_certificate1"
            / "P0_WORKLOAD_ROWS.jsonl"
        ),
        "invalidation": (
            root
            / "experiments/semstamp_certified_defense_v2_wp4_resource_preflight_design1"
            / "EXECUTION_INVALIDATION.json"
        ),
    }


def expected_authorization(paths, core):
    return {
        "certificate_authorized": False,
        "contract_sha256": sha256_file(paths["contract"]),
        "core_sha256": sha256_file(core),
        "detector_score_authorized": False,
        "model_load_authorized": False,
        "pair_gate": PAIR_GATE,
        "plan_index": PLAN_INDEX,
        "protocol_sha256": PROTOCOL_SHA256,
        "radius": RADIUS,
        "runner_sha256": sha256_file(paths["runner"]),
        "schema_version": 1,
        "scope": "single_score_blind_fixed_n_subset_lower_bound_only",
        "stage": STAGE,
        "status": "WP4_CORRECTIVE_CORE_CAP_EXECUTION_AUTHORIZED",
        "wp5_authorized": False,
    }


def validate_inputs(paths, core):
    expected = {
        "protocol": PROTOCOL_SHA256,
        "wp1_threat": WP1_THREAT_SHA256,
        "wp2_result": WP2_RESULT_SHA256,
        "v2_rows": V2_ROWS_SHA256,
        "v18_rows": V18_ROWS_SHA256,
        "invalidation": INVALIDATION_SHA256,
    }
    for name, digest in expected.items():
        if sha256_file(paths[name]) != digest:
            raise ValueError(f"{name} SHA256 mismatch")
    if sha256_file(core) != LINUX_CORE_SHA256:
        raise ValueError("Linux core SHA256 mismatch")
    contract = read_json(paths["contract"])
    required_contract = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "frozen",
        "protocol_sha256": PROTOCOL_SHA256,
        "plan_index": PLAN_INDEX,
        "radius": RADIUS,
        "pair_gate": PAIR_GATE,
        "fixed_n_subset_of_variable_n": True,
        "execution_authorized": False,
        "model_load_authorized": False,
        "detector_score_authorized": False,
        "certificate_authorized": False,
        "wp5_authorized": False,
    }
    mismatches = {
        key: (contract.get(key), expected_value)
        for key, expected_value in required_contract.items()
        if contract.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"contract mismatch: {mismatches}")
    if contract.get("runner_sha256") != sha256_file(paths["runner"]):
        raise ValueError("runner binding mismatch")
    if stat.S_IMODE(paths["authorization"].stat().st_mode) != 0o600:
        raise PermissionError("authorization mode must be 0600")
    authorization = read_json(paths["authorization"])
    if authorization != expected_authorization(paths, core):
        raise PermissionError("correction execution authorization mismatch")
    return contract, authorization


def parse_histogram(value):
    return {
        key: int(count)
        for item in value.split(",")
        for key, count in (item.split(":", 1),)
    }


def parse_core_output(payload):
    lines = payload.decode("ascii").splitlines()
    if len(lines) != 1:
        raise ValueError("core returned the wrong row count")
    fields = lines[0].split("\t")
    if len(fields) != 12 or fields[0] != "0":
        raise ValueError("core output schema mismatch")
    witness = tuple(map(int, fields[5].split(",")))
    return {
        "original_score": float(fields[1]),
        "synthetic_certified_score": float(fields[2]),
        "threat_set_size": int(fields[3]),
        "move_histogram": parse_histogram(fields[4]),
        "witness_boundaries_sha256": canonical_digest(list(witness)),
        "witness_edge_count": len(witness),
        "reachable_state_count": int(fields[6]),
        "transitions_evaluated": int(fields[7]),
        "peak_layer_states": int(fields[8]),
        "candidate_boundary_positions": int(fields[9]),
        "all_allowed_unique_spans": int(fields[10]),
        "all_allowed_unique_pairs": int(fields[11]),
    }


def validate_identity(v2_row, v18_row):
    expected_v2 = {
        "plan_index": PLAN_INDEX,
        "raw_text_sha256": EXPECTED_RAW_SHA256,
        "atom_count": 5554,
        "unit_count": 174,
        "refinement_count": 0,
        "zero_byte_gap_count": 0,
    }
    mismatches = {
        key: (v2_row.get(key), value)
        for key, value in expected_v2.items()
        if v2_row.get(key) != value
    }
    if mismatches:
        raise ValueError(f"v2 decisive identity mismatch: {mismatches}")
    if (
        v18_row.get("plan_index") != PLAN_INDEX
        or v18_row.get("raw_text_sha256") != EXPECTED_RAW_SHA256
        or v18_row.get("atom_count") != v2_row["atom_count"]
        or v18_row.get("unit_count") != v2_row["unit_count"]
        or v18_row.get("original_boundaries")
        != v2_row.get("original_boundaries")
    ):
        raise ValueError("v18/v2 decisive topology identity mismatch")
    if len(v2_row["original_boundaries"]) != 173:
        raise ValueError("decisive edge count mismatch")


def run_core(core, v2_row):
    fields = [
        v2_row["atom_count"],
        len(v2_row["original_boundaries"]),
        RADIUS,
        3,
        1,
        1,
        0,
        0,
        *v2_row["original_boundaries"],
    ]
    payload = (
        "1\n" + " ".join(map(str, fields)) + "\n"
    ).encode("ascii")
    started = time.perf_counter()
    completed = subprocess.run(
        [str(core)],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            "core failed: "
            + completed.stderr.decode("utf-8", errors="replace")
        )
    return parse_core_output(completed.stdout), elapsed, sha256_bytes(
        completed.stdout
    )


def run(args):
    paths = paths_for(args.root, args.output_dir)
    core = Path(args.core).resolve()
    validate_inputs(paths, core)
    invalidation = read_json(paths["invalidation"])
    wp2_result = read_json(paths["wp2_result"])
    if (
        invalidation.get("terminal")
        != "WP4_RESOURCE_PREFLIGHT_EXECUTION_INVALIDATED_THREAT_MODEL_MISMATCH"
        or invalidation.get("completed_prefix_length") != 83
        or invalidation.get("resume_authorized") is not False
    ):
        raise ValueError("invalidated execution evidence mismatch")
    if (
        wp2_result.get("status") != "pass"
        or wp2_result.get("terminal")
        != "WP2_ALGORITHM_GO_FOR_V2_WP3_DESIGN"
    ):
        raise ValueError("WP2 variable-N parent is not Go")

    v2_row = read_jsonl_row(paths["v2_rows"], PLAN_INDEX)
    v18_row = read_jsonl_row(paths["v18_rows"], PLAN_INDEX)
    validate_identity(v2_row, v18_row)
    observed, elapsed, stdout_sha256 = run_core(core, v2_row)
    historical = v18_row["radii"][str(RADIUS)]
    if (
        observed["all_allowed_unique_spans"] != EXPECTED_SPANS
        or observed["all_allowed_unique_pairs"] != EXPECTED_PAIRS
        or historical.get("all_allowed_unique_spans") != EXPECTED_SPANS
        or historical.get("all_allowed_unique_pairs") != EXPECTED_PAIRS
    ):
        raise ValueError("fresh/historical topology count mismatch")

    pair_gate_pass = observed["all_allowed_unique_pairs"] <= PAIR_GATE
    if pair_gate_pass:
        raise AssertionError("frozen core cap unexpectedly passed")
    checks = {
        "authorization_exact": True,
        "inputs_sha256_exact": True,
        "wp2_variable_n_go_exact": True,
        "invalid_execution_prefix_frozen": True,
        "invalid_execution_resume_forbidden": True,
        "decisive_identity_exact": True,
        "fixed_n_subset_of_variable_n": True,
        "fresh_core_execution_succeeded": True,
        "fresh_historical_spans_exact": True,
        "fresh_historical_pairs_exact": True,
        "pair_gate_le_50000000": pair_gate_pass,
        "zero_model_score_certificate_calls": True,
        "public_output_contains_no_raw_text": True,
    }
    result = {
        "authorization_sha256": sha256_file(paths["authorization"]),
        "bindings": {
            "contract_sha256": sha256_file(paths["contract"]),
            "core_sha256": sha256_file(core),
            "invalidation_sha256": INVALIDATION_SHA256,
            "protocol_sha256": PROTOCOL_SHA256,
            "runner_sha256": sha256_file(paths["runner"]),
            "v18_rows_sha256": V18_ROWS_SHA256,
            "v2_rows_sha256": V2_ROWS_SHA256,
            "wp1_threat_sha256": WP1_THREAT_SHA256,
            "wp2_result_sha256": WP2_RESULT_SHA256,
        },
        "completed_at": utc_now(),
        "contains_raw_text": False,
        "decision": {
            "certificate_authorized": False,
            "wp4": "readiness_stop",
            "wp5_authorized": False,
        },
        "execution_attestation": {
            "certificate_computations": 0,
            "cuda_initializations": 0,
            "detector_score_computations": 0,
            "generation_calls": 0,
            "model_loads": 0,
            "score_mode": "synthetic_zero_topology_only",
            "wp5_calls": 0,
        },
        "fixed_n_subset_lower_bound": {
            "atom_count": v2_row["atom_count"],
            "fresh_core_elapsed_seconds": elapsed,
            "fresh_core_stdout_sha256": stdout_sha256,
            "original_boundaries_sha256": canonical_digest(
                v2_row["original_boundaries"]
            ),
            "original_edge_count": len(v2_row["original_boundaries"]),
            "plan_index": PLAN_INDEX,
            "radius": RADIUS,
            "raw_text_sha256": EXPECTED_RAW_SHA256,
            **observed,
        },
        "gates": checks,
        "pair_gate": PAIR_GATE,
        "schema_version": 1,
        "stage": STAGE,
        "status": "stop",
        "terminal": "WP4_RESOURCE_PREFLIGHT_READINESS_STOP_CORE_CAP",
    }
    atomic_write_json(paths["result"], result)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    return parser.parse_args()


def main():
    result = run(parse_args())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
