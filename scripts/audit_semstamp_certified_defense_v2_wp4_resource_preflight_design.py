"""Independently audit the frozen WP4 resource-preflight design."""

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile


STAGE = "semstamp_certified_defense_v2_wp4_resource_preflight"
AUDIT_STAGE = f"{STAGE}_design_audit"
PROTOCOL_SHA256 = (
    "3334b8d5567db674f0350d37b8b1d345e27466d26e2dea4df44d4a7e2105f904"
)
WP3_FINAL_SHA256 = (
    "b51733c497713389ca91c7e14b4313f44c9ae94d19e40d5be88e9292be0a0951"
)
WP3_AUDIT_SHA256 = (
    "1ced8396070eabdac75f40ebac1d34d5a38229e7bf93f44a916e23eeb1dc91a2"
)
P0_ROWS_SHA256 = (
    "8625f6b074150d4ec099b9e57da00d6b08167b42f25ad95d34b975682e5a7474"
)
V6_PROFILE_SHA256 = (
    "4701a4b6d77f2d0c20e189853ec896d70ac205df05659ee38e576b5010119f6d"
)
V18_READINESS_SHA256 = (
    "bee8324acf47e7772f7628127082159d73d6d0040f9b0c7f1a95fe64efdf0d56"
)
TOTAL_ROWS = 3000
COHORTS = ("calibration_null", "independent_test_null", "positive")
FORBIDDEN_PUBLIC_KEYS = {
    "generated_text",
    "prompt",
    "raw_text",
    "source_text",
    "text",
}
FORBIDDEN_TOP_LEVEL_IMPORTS = {
    "numpy",
    "sentence_transformers",
    "torch",
    "transformers",
    "semstamp_refined_atom_renderer",
    "run_semstamp_certified_defense_v2_wp3_p0",
}


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


def parse_manifest(path):
    entries = {}
    for line_number, line in enumerate(
        Path(path).read_text(encoding="ascii").splitlines(), 1
    ):
        if not line:
            continue
        digest, separator, relative = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not relative
            or relative in entries
        ):
            raise ValueError(f"invalid manifest line {line_number}")
        entries[relative] = digest
    return entries


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


def paths_for(root, output_dir, output):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    output = Path(output).resolve()
    if (
        not output_dir.is_relative_to(root)
        or not output.is_relative_to(output_dir)
    ):
        raise ValueError("audit paths must remain inside root")
    return {
        "root": root,
        "output_dir": output_dir,
        "protocol": output_dir / "PROTOCOL.md",
        "contract": output_dir / "CONTRACT.json",
        "manifest": output_dir / "INPUT_MANIFEST.sha256",
        "authorization": output_dir / "EXECUTION_AUTHORIZATION.json",
        "resume_authorization": output_dir / "RESUME_AUTHORIZATION.json",
        "rows": output_dir / "PREFLIGHT_ROWS.jsonl",
        "timings": output_dir / "sealed/PREFLIGHT_TIMINGS.jsonl",
        "result": output_dir / "PREFLIGHT_RESULT.json",
        "heartbeat": output_dir / "HEARTBEAT.json",
        "failure": output_dir / "LAST_FAILURE.json",
        "lock": output_dir / "EXECUTION.lock",
        "local_test": output_dir / "LOCAL_TEST_RESULT.json",
        "remote_test": output_dir / "REMOTE_TEST_RESULT.json",
        "runner": (
            root
            / "scripts/"
            "run_semstamp_certified_defense_v2_wp4_resource_preflight.py"
        ),
        "test": (
            root
            / "scripts/"
            "test_semstamp_certified_defense_v2_wp4_resource_preflight.py"
        ),
        "auditor": Path(__file__).resolve(),
        "wp3_final": (
            root
            / "experiments/"
            "semstamp_certified_defense_v2_wp3_detector_full_replay_design1/"
            "FINAL_RESULT.json"
        ),
        "wp3_audit": (
            root
            / "experiments/"
            "semstamp_certified_defense_v2_wp3_detector_full_replay_design1/"
            "EXECUTION_AUDIT.json"
        ),
        "p0_rows": (
            root
            / "experiments/semstamp_certified_defense_v2_wp3_detector_health1/"
            "P0_RENDERER_ROWS_REPLAY0.jsonl"
        ),
        "v6_profile": (
            root
            / "experiments/certified_resegmentation_wp2_v6_feature1/"
            "semstamp_workload.json"
        ),
        "v18_readiness": (
            root
            / "experiments/"
            "certified_resegmentation_wp2_v18_b1_exact_certificate1/"
            "P0_READINESS.json"
        ),
        "output": output,
    }


def top_level_imports(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def assigned_literals(tree):
    def evaluate(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Pow):
                return left ** right
        raise ValueError("not a supported literal expression")

    values = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                values[node.targets[0].id] = evaluate(node.value)
            except (ValueError, TypeError):
                pass
    return values


def audit(root, output_dir):
    paths = paths_for(root, output_dir, Path(output_dir) / "DESIGN_AUDIT.json")
    contract = read_json(paths["contract"])
    manifest = parse_manifest(paths["manifest"])
    runner_source = paths["runner"].read_text(encoding="utf-8")
    runner_tree = ast.parse(runner_source)
    literals = assigned_literals(runner_tree)

    manifest_exact = True
    for relative, expected in manifest.items():
        path = (paths["root"] / relative).resolve()
        metadata = contract.get("inputs", {}).get(relative, {})
        if (
            not path.is_file()
            or sha256_file(path) != expected
            or metadata.get("sha256") != expected
            or metadata.get("bytes") != path.stat().st_size
        ):
            manifest_exact = False
            break
        expected_mode = metadata.get("mode")
        if (
            expected_mode is not None
            and f"{stat.S_IMODE(path.stat().st_mode):04o}" != expected_mode
        ):
            manifest_exact = False
            break

    p0_rows = read_jsonl(paths["p0_rows"])
    wp3_final = read_json(paths["wp3_final"])
    wp3_audit = read_json(paths["wp3_audit"])
    v18 = read_json(paths["v18_readiness"])
    profile = read_json(paths["v6_profile"])
    local_test = read_json(paths["local_test"])
    remote_test = read_json(paths["remote_test"])

    run_start = runner_source.index("def run(")
    authorization_position = runner_source.index(
        "validate_authorization(paths, contract)", run_start
    )
    deferred_import_position = runner_source.index(
        "import run_semstamp_certified_defense_v2_wp3_p0", run_start
    )
    checks = {
        "protocol_sha_exact": (
            sha256_file(paths["protocol"]) == PROTOCOL_SHA256
            and contract.get("protocol_sha256") == PROTOCOL_SHA256
        ),
        "contract_artifact_hashes_exact": all(
            contract.get(field) == sha256_file(path)
            for field, path in {
                "runner_sha256": paths["runner"],
                "test_sha256": paths["test"],
                "auditor_sha256": paths["auditor"],
                "input_manifest_sha256": paths["manifest"],
            }.items()
        ),
        "input_manifest_complete_and_exact": (
            bool(manifest)
            and set(manifest) == set(contract.get("inputs", {}))
            and manifest_exact
        ),
        "wp3_terminal_exact": (
            sha256_file(paths["wp3_final"]) == WP3_FINAL_SHA256
            and sha256_file(paths["wp3_audit"]) == WP3_AUDIT_SHA256
            and wp3_final.get("terminal")
            == "WP3_FULL_REPLAY_GO_FOR_WP4_DESIGN"
            and wp3_audit.get("status") == "go"
        ),
        "p0_complete_denominator_exact": (
            sha256_file(paths["p0_rows"]) == P0_ROWS_SHA256
            and len(p0_rows) == TOTAL_ROWS
            and [row["plan_index"] for row in p0_rows]
            == list(range(TOTAL_ROWS))
            and Counter(row["cohort"] for row in p0_rows)
            == {cohort: 1000 for cohort in COHORTS}
        ),
        "risk_rows_and_zero_gaps_retained": (
            sum(row["refinement_count"] > 0 for row in p0_rows) == 42
            and sum(row["zero_byte_gap_count"] for row in p0_rows) == 59
        ),
        "v6_profile_exact_and_complete": (
            sha256_file(paths["v6_profile"]) == V6_PROFILE_SHA256
            and profile.get("status") == "pass"
            and len(profile.get("inputs", {}).get("provider_profiles", []))
            == 12
        ),
        "v18_stop_preserved": (
            sha256_file(paths["v18_readiness"]) == V18_READINESS_SHA256
            and v18.get("status") == "stop"
            and v18.get("terminal_reason") == "b1_readiness_stop"
        ),
        "old_core_limits_not_raised": (
            literals.get("SPAN_GATE") == 2_000_000
            and literals.get("PAIR_GATE") == 50_000_000
            and contract.get("resource_budget", {}).get("span_gate")
            == 2_000_000
            and contract.get("resource_budget", {}).get("pair_gate")
            == 50_000_000
        ),
        "storage_limits_exact": (
            literals.get("SIDECAR_GATE_BYTES") == 50 * 1024**3
            and literals.get("WORK_DISK_GATE_BYTES") == 15 * 1024**3
            and literals.get("FREE_DISK_RESERVE_BYTES") == 5 * 1024**3
        ),
        "two_replay_budget_exact": (
            literals.get("REPLAY_COUNT") == 2
            and literals.get("GPU_HOUR_GATE") == 72.0
        ),
        "selected_hardware_and_driver_exact": (
            contract.get("selected_hardware") == {
                "device": "NVIDIA GeForce RTX 3090",
                "driver_version": "595.80",
                "memory_mib": 24576,
                "cuda_visible_devices": "0",
            }
        ),
        "authorization_absent": not paths["authorization"].exists(),
        "resume_authorization_absent": (
            not paths["resume_authorization"].exists()
        ),
        "execution_outputs_absent": not any(
            path.exists()
            for path in (
                paths["rows"],
                paths["timings"],
                paths["result"],
                paths["heartbeat"],
                paths["failure"],
                paths["lock"],
            )
        ),
        "contract_execution_flags_false": all(
            contract.get(key) is False
            for key in (
                "execution_authorized",
                "sbert_load_authorized",
                "detector_score_authorized",
                "certificate_authorized",
                "wp5_authorized",
            )
        ),
        "authorization_precedes_deferred_imports": (
            authorization_position < deferred_import_position
        ),
        "forbidden_top_level_imports_absent": top_level_imports(
            runner_tree
        ).isdisjoint(FORBIDDEN_TOP_LEVEL_IMPORTS),
        "tokenizer_binding_excludes_weight_file": (
            "pytorch_model.bin" not in runner_source[
                runner_source.index("TOKENIZER_SHA256"):
                runner_source.index("COHORTS =")
            ]
        ),
        "score_blind_terminal_set_exact": all(
            terminal in runner_source
            for terminal in (
                "WP4_RESOURCE_PREFLIGHT_GO_FOR_WP5_DESIGN",
                "WP4_RESOURCE_PREFLIGHT_READINESS_STOP_CORE_CAP",
                "WP4_RESOURCE_PREFLIGHT_READINESS_STOP_BUDGET",
                "WP4_RESOURCE_PREFLIGHT_INVALID_OR_INCOMPLETE",
            )
        ),
        "local_tests_pass": (
            local_test.get("status") == "pass"
            and local_test.get("tests_run") == 18
            and local_test.get("failures") == 0
        ),
        "remote_tests_pass": (
            remote_test.get("status") == "pass"
            and remote_test.get("tests_run") == 18
            and remote_test.get("failures") == 0
        ),
        "public_inputs_contain_no_raw_text": all(
            row.get("contains_raw_text") is False for row in p0_rows
        ),
        "design_has_no_runtime_artifacts": (
            contract.get("design_execution_attestation") == {
                "authorization_files": 0,
                "tokenizer_loads": 0,
                "sbert_weight_reads": 0,
                "sbert_model_loads": 0,
                "cuda_initializations": 0,
                "region_computations": 0,
                "hit_computations": 0,
                "detector_score_reads": 0,
                "detector_score_computations": 0,
                "threshold_computations": 0,
                "certificate_computations": 0,
                "generation_calls": 0,
                "preflight_executions": 0,
                "wp5_calls": 0,
            }
        ),
    }
    result = {
        "schema_version": 1,
        "stage": AUDIT_STAGE,
        "status": "go" if all(checks.values()) else "stop",
        "terminal": (
            "WP4_RESOURCE_PREFLIGHT_DESIGN_AUDIT_PASS"
            if all(checks.values())
            else "WP4_RESOURCE_PREFLIGHT_DESIGN_AUDIT_STOP"
        ),
        "contains_raw_text": False,
        "checks": checks,
        "counts": {
            "checks_passed": sum(checks.values()),
            "checks_total": len(checks),
            "input_manifest_entries": len(manifest),
            "p0_rows": len(p0_rows),
            "refinement_risk_rows": sum(
                row["refinement_count"] > 0 for row in p0_rows
            ),
            "zero_byte_gaps": sum(
                row["zero_byte_gap_count"] for row in p0_rows
            ),
            "provider_profiles": len(
                profile.get("inputs", {}).get("provider_profiles", [])
            ),
        },
        "bindings": {
            "protocol": sha256_file(paths["protocol"]),
            "contract": sha256_file(paths["contract"]),
            "manifest": sha256_file(paths["manifest"]),
            "runner": sha256_file(paths["runner"]),
            "test": sha256_file(paths["test"]),
            "auditor": sha256_file(paths["auditor"]),
            "wp3_final": sha256_file(paths["wp3_final"]),
            "wp3_audit": sha256_file(paths["wp3_audit"]),
            "p0_rows": sha256_file(paths["p0_rows"]),
            "v6_profile": sha256_file(paths["v6_profile"]),
            "v18_readiness": sha256_file(paths["v18_readiness"]),
        },
        "decision": {
            "wp4_design": (
                "frozen_pending_execution_authorization"
                if all(checks.values())
                else "stop"
            ),
            "wp4_execution_authorized": False,
            "wp5_authorized": False,
            "certificate_authorized": False,
        },
        "audit_execution_attestation": {
            "tokenizer_loads": 0,
            "sbert_weight_reads": 0,
            "sbert_model_loads": 0,
            "cuda_initializations": 0,
            "region_computations": 0,
            "hit_computations": 0,
            "detector_score_reads": 0,
            "detector_score_computations": 0,
            "certificate_computations": 0,
            "generation_calls": 0,
            "preflight_executions": 0,
            "wp5_calls": 0,
        },
    }
    reject_raw_public_keys(result)
    atomic_write_json(paths["output"], result)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    result = audit(args.root, args.output_dir)
    atomic_write_json(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
