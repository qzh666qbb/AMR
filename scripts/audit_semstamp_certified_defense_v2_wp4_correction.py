"""Independently audit the WP4 corrective fixed-N subset lower bound."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile


STAGE = "semstamp_certified_defense_v2_wp4_resource_preflight_correction_audit"
PROTOCOL_SHA256 = (
    "f4083699cd963e4913c4103d9ef1cf3d5798d48a1eda4f79e0d70831e24f2361"
)
CONTRACT_SHA256 = (
    "3fe03b8410398c78dd488920262e6cc2019b9e07939cf83bcc3cd30e141d8793"
)
RUNNER_SHA256 = (
    "75ab0443ecf450f9eea6914977d78d82c88c9a85cc4ab52c29d9aec9a07c21af"
)
AUTHORIZATION_SHA256 = (
    "2b3df26ed1b45d6df82c2114a3dbaadfee46ba167a87b98527dac04a3c466f2f"
)
RESULT_SHA256 = (
    "1b7458ab1cee8317b3afe1d0c186804527e962625c536d57aada865b21971db1"
)
INVALIDATION_SHA256 = (
    "46e42adc1fe0aae9bb2e322d3dc112e2c530f91c6c64d9fd2a2eae4bff55fae6"
)
PARTIAL_ROWS_SHA256 = (
    "ffcc98ec5c2b707f9165988f6b34c3ec9c9977ee07bffc00d8f93701a44faf5d"
)
PARTIAL_TIMINGS_SHA256 = (
    "d307235bef548ec5cb57c93bee8b55b42134411cfb54bc32e4d63cec8408adb0"
)
V2_ROWS_SHA256 = (
    "8625f6b074150d4ec099b9e57da00d6b08167b42f25ad95d34b975682e5a7474"
)
V18_ROWS_SHA256 = (
    "a6e76b846981b12cf2eab197f81931f51a63309582167079d1ac926386ececc4"
)
LINUX_CORE_SHA256 = (
    "d724a038f72d766b477cb1bd95c8318bc0032c3718e067005bd05921944847fc"
)
ORIGINAL_RUNNER_SHA256 = (
    "37c183640f84c8a9a9e803414b527201e836f6ce6bb39ee72c4e3f3fbd5ce606"
)
CORE_STDOUT_SHA256 = (
    "4c37491d61cf1ed09e22d8dcb380eb1b189c6efadabf999efa2dcba858990d9c"
)
PAIR_GATE = 50_000_000
EXPECTED_PAIRS = 83_540_900
EXPECTED_SPANS = 769_113
PLAN_INDEX = 800
FORBIDDEN_PUBLIC_KEYS = {
    "generated_text",
    "prompt",
    "raw_text",
    "source_text",
    "text",
}


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


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_jsonl_row(path, index):
    with Path(path).open("r", encoding="utf-8") as handle:
        for current, line in enumerate(handle):
            if current == index:
                return json.loads(line)
    raise ValueError(f"missing row {index}")


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


def reject_raw_public_keys(value, location="root"):
    if isinstance(value, dict):
        overlap = FORBIDDEN_PUBLIC_KEYS & set(value)
        if overlap:
            raise ValueError(f"raw text key at {location}: {overlap}")
        for key, child in value.items():
            reject_raw_public_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_raw_public_keys(child, f"{location}[{index}]")


def paths_for(root, output_dir):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    original = (
        root
        / "experiments/semstamp_certified_defense_v2_wp4_resource_preflight_design1"
    )
    return {
        "root": root,
        "output_dir": output_dir,
        "protocol": output_dir / "PROTOCOL.md",
        "contract": output_dir / "CONTRACT.json",
        "authorization": output_dir / "CORRECTION_EXECUTION_AUTHORIZATION.json",
        "result": output_dir / "CORRECTIVE_RESULT.json",
        "audit": output_dir / "CORRECTIVE_AUDIT.json",
        "runner": (
            root / "scripts/run_semstamp_certified_defense_v2_wp4_correction.py"
        ),
        "original_runner": (
            root
            / "scripts/run_semstamp_certified_defense_v2_wp4_resource_preflight.py"
        ),
        "invalidation": original / "EXECUTION_INVALIDATION.json",
        "partial_rows": original / "PREFLIGHT_ROWS.jsonl",
        "partial_timings": original / "sealed/PREFLIGHT_TIMINGS.jsonl",
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
    }


def static_runner_mismatch(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    return {
        "calls_fixed_n_sparse_solver": (
            "exact_grouped_sparse_pairwise_certificate" in names
        ),
        "calls_v2_variable_n_solver": (
            "exact_refined_variable_n_pairwise_certificate" in names
        ),
    }


def run_core(core, row):
    fields = [
        row["atom_count"],
        len(row["original_boundaries"]),
        3,
        3,
        1,
        1,
        0,
        0,
        *row["original_boundaries"],
    ]
    payload = (
        "1\n" + " ".join(map(str, fields)) + "\n"
    ).encode("ascii")
    completed = subprocess.run(
        [str(core)],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.decode("utf-8", errors="replace")
        )
    lines = completed.stdout.decode("ascii").splitlines()
    if len(lines) != 1:
        raise ValueError("independent core replay returned wrong row count")
    fields = lines[0].split("\t")
    if len(fields) != 12 or fields[0] != "0":
        raise ValueError("independent core replay schema mismatch")
    return {
        "stdout_sha256": sha256_bytes(completed.stdout),
        "threat_set_size": int(fields[3]),
        "unique_spans": int(fields[10]),
        "unique_pairs": int(fields[11]),
    }


def run(args):
    paths = paths_for(args.root, args.output_dir)
    core = Path(args.core).resolve()
    expected_hashes = {
        "protocol": PROTOCOL_SHA256,
        "contract": CONTRACT_SHA256,
        "authorization": AUTHORIZATION_SHA256,
        "result": RESULT_SHA256,
        "runner": RUNNER_SHA256,
        "original_runner": ORIGINAL_RUNNER_SHA256,
        "invalidation": INVALIDATION_SHA256,
        "partial_rows": PARTIAL_ROWS_SHA256,
        "partial_timings": PARTIAL_TIMINGS_SHA256,
        "v2_rows": V2_ROWS_SHA256,
        "v18_rows": V18_ROWS_SHA256,
    }
    hashes_exact = all(
        sha256_file(paths[name]) == digest
        for name, digest in expected_hashes.items()
    )
    if not hashes_exact or sha256_file(core) != LINUX_CORE_SHA256:
        raise ValueError("audit input SHA256 mismatch")

    result = read_json(paths["result"])
    invalidation = read_json(paths["invalidation"])
    partial_rows = read_jsonl(paths["partial_rows"])
    partial_timings = read_jsonl(paths["partial_timings"])
    v2_row = read_jsonl_row(paths["v2_rows"], PLAN_INDEX)
    v18_row = read_jsonl_row(paths["v18_rows"], PLAN_INDEX)
    source_audit = static_runner_mismatch(paths["original_runner"])
    independent = run_core(core, v2_row)

    reject_raw_public_keys(result)
    reject_raw_public_keys(invalidation)
    reject_raw_public_keys(partial_rows)
    checks = {
        "all_input_sha256_exact": hashes_exact,
        "authorization_mode_0600": (
            stat.S_IMODE(paths["authorization"].stat().st_mode) == 0o600
        ),
        "result_mode_0600": (
            stat.S_IMODE(paths["result"].stat().st_mode) == 0o600
        ),
        "original_runner_uses_fixed_n_solver": (
            source_audit["calls_fixed_n_sparse_solver"]
        ),
        "original_runner_omits_variable_n_solver": (
            not source_audit["calls_v2_variable_n_solver"]
        ),
        "invalidated_prefix_is_83": (
            invalidation.get("completed_prefix_length") == 83
            and len(partial_rows) == 83
            and len(partial_timings) == 83
        ),
        "partial_prefix_order_exact": (
            [row.get("plan_index") for row in partial_rows]
            == list(range(83))
            and [row.get("plan_index") for row in partial_timings]
            == list(range(83))
        ),
        "partial_prefix_score_access_zero": all(
            row.get("score_access_count") == 0 for row in partial_rows
        ),
        "resume_forbidden": (
            invalidation.get("resume_authorized") is False
        ),
        "decisive_v2_identity_exact": (
            v2_row.get("plan_index") == PLAN_INDEX
            and v2_row.get("atom_count") == 5554
            and v2_row.get("unit_count") == 174
            and v2_row.get("refinement_count") == 0
            and v2_row.get("zero_byte_gap_count") == 0
            and len(v2_row.get("original_boundaries", [])) == 173
        ),
        "v18_v2_all_allowed_topology_identity": (
            v18_row.get("atom_count") == v2_row.get("atom_count")
            and v18_row.get("original_boundaries")
            == v2_row.get("original_boundaries")
        ),
        "independent_core_stdout_exact": (
            independent["stdout_sha256"] == CORE_STDOUT_SHA256
        ),
        "independent_spans_exact": (
            independent["unique_spans"] == EXPECTED_SPANS
        ),
        "independent_pairs_exact": (
            independent["unique_pairs"] == EXPECTED_PAIRS
        ),
        "historical_spans_exact": (
            v18_row["radii"]["3"]["all_allowed_unique_spans"]
            == EXPECTED_SPANS
        ),
        "historical_pairs_exact": (
            v18_row["radii"]["3"]["all_allowed_unique_pairs"]
            == EXPECTED_PAIRS
        ),
        "fixed_n_subset_pair_gate_fails": EXPECTED_PAIRS > PAIR_GATE,
        "result_terminal_exact": (
            result.get("status") == "stop"
            and result.get("terminal")
            == "WP4_RESOURCE_PREFLIGHT_READINESS_STOP_CORE_CAP"
        ),
        "wp5_and_certificate_not_authorized": (
            result.get("decision", {}).get("wp5_authorized") is False
            and result.get("decision", {}).get("certificate_authorized")
            is False
        ),
        "zero_model_score_certificate_calls": (
            result.get("execution_attestation") == {
                "certificate_computations": 0,
                "cuda_initializations": 0,
                "detector_score_computations": 0,
                "generation_calls": 0,
                "model_loads": 0,
                "score_mode": "synthetic_zero_topology_only",
                "wp5_calls": 0,
            }
        ),
        "public_outputs_contain_no_raw_text": True,
    }
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise AssertionError(f"corrective audit failed: {failed}")
    audit = {
        "audited_at": utc_now(),
        "bindings": {
            **expected_hashes,
            "core": LINUX_CORE_SHA256,
        },
        "checks": checks,
        "contains_raw_text": False,
        "counts": {
            "checks_passed": sum(checks.values()),
            "checks_total": len(checks),
            "fixed_n_subset_pairs": independent["unique_pairs"],
            "fixed_n_subset_spans": independent["unique_spans"],
            "invalidated_prefix_rows": len(partial_rows),
            "pair_gate": PAIR_GATE,
            "threat_set_size": independent["threat_set_size"],
        },
        "decision": {
            "certificate_authorized": False,
            "wp5_authorized": False,
        },
        "schema_version": 1,
        "scope": "independent_score_blind_core_cap_lower_bound_audit",
        "source_audit": source_audit,
        "stage": STAGE,
        "status": "pass",
        "terminal": "WP4_RESOURCE_PREFLIGHT_READINESS_STOP_CORE_CAP",
    }
    reject_raw_public_keys(audit)
    atomic_write_json(paths["audit"], audit)
    return audit


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    return parser.parse_args()


def main():
    audit = run(parse_args())
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
