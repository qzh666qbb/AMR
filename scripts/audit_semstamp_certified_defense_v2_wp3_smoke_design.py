"""Audit the frozen v2 WP3 detector-smoke design without executing it."""

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile


STAGE = "semstamp_certified_defense_v2_wp3_detector_smoke"
AUDIT_STAGE = f"{STAGE}_design_audit"
PROTOCOL_SHA256 = (
    "49cc738c67bc41f28e05095a983a966c52150e5c3c7a77d9706e49601f33d627"
)
SEMSTAMP_REVISION = "97db73d11fd80f376a02b0a604d500627622f7e6"
SEMSTAMP_DATA_REVISION = "bd6e948de2a6fc30f7670dfad347c2b79e252687"
MODEL_REVISION = "a77dce8530e100ac1101981e4be390546c1cb282"
RELATION_SHA256 = (
    "4a8c517498d800a398414485a7e218e871a24d79a72238920b0c3f4ca29c53f1"
)
PUNKT_SHA256 = (
    "5cad3758596392364e3be9803dbd7ebeda384b68937b488a01365f5551bb942c"
)
P0_FINAL_SHA256 = (
    "6a03448655d0084fe5e55505490c9018dedf79c66609dfd504183eeea9c3d6fe"
)
P0_READINESS_SHA256 = (
    "eb4da58d35be349a53aa4580884d23e6ecbb9d534a34c4f19d09769378b7a8f9"
)
P0_AUDIT_SHA256 = (
    "af21e2c408bbb5936c1c646b84ac8ddc2dafba8cb53f3f47ac592755e0041d78"
)
P0_ROWS_SHA256 = (
    "8625f6b074150d4ec099b9e57da00d6b08167b42f25ad95d34b975682e5a7474"
)
SMOKE_PLAN_SHA256 = (
    "f4354914023a62927931e1423fa4db119cc202b6c002e765b3a79a1bd80509bf"
)
COHORTS = ("calibration_null", "independent_test_null", "positive")
COHORT_SIZE = 1000
TOTAL_ROWS = 3000
SMOKE_PER_COHORT = 32
SMOKE_ROWS = 96
EXPECTED_MANDATORY_ROWS = 42
EXPECTED_ZERO_GAPS = 59
FORBIDDEN_PUBLIC_KEYS = {
    "generated_text",
    "prompt",
    "raw_text",
    "source_text",
    "text",
}
FORBIDDEN_TOP_LEVEL_IMPORTS = {
    "sentence_transformers",
    "torch",
    "transformers",
    "run_certified_resegmentation_wp2_detector_equivalence",
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


def jsonl_bytes(rows):
    return b"".join(canonical_bytes(row) for row in rows)


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
    if not entries:
        raise ValueError("input manifest is empty")
    return entries


def git_head(path):
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def paths_for(root, output_dir, output):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    output = Path(output).resolve()
    if (
        not output_dir.is_relative_to(root)
        or not output.is_relative_to(output_dir)
    ):
        raise ValueError("output paths must remain inside the experiment")
    p0_dir = (
        root
        / "experiments/semstamp_certified_defense_v2_wp3_detector_health1"
    )
    return {
        "root": root,
        "output_dir": output_dir,
        "protocol": output_dir / "PROTOCOL.md",
        "contract": output_dir / "CONTRACT.json",
        "manifest": output_dir / "INPUT_MANIFEST.sha256",
        "authorization": output_dir / "EXECUTION_AUTHORIZATION.json",
        "rows0": output_dir / "SMOKE_ROWS_REPLAY0.jsonl",
        "rows1": output_dir / "SMOKE_ROWS_REPLAY1.jsonl",
        "checkpoint": output_dir / "sealed/SMOKE_CHECKPOINT.json",
        "heartbeat": output_dir / "HEARTBEAT.json",
        "smoke_result": output_dir / "SMOKE_RESULT.json",
        "last_failure": output_dir / "LAST_FAILURE.json",
        "audit": output,
        "runner": (
            root / "scripts/run_semstamp_certified_defense_v2_wp3_smoke.py"
        ),
        "renderer": root / "scripts/semstamp_refined_atom_renderer.py",
        "p0_runner": (
            root / "scripts/run_semstamp_certified_defense_v2_wp3_p0.py"
        ),
        "test": (
            root / "scripts/test_semstamp_certified_defense_v2_wp3_smoke.py"
        ),
        "auditor": Path(__file__).resolve(),
        "p0_final": p0_dir / "FINAL_RESULT.json",
        "p0_readiness": p0_dir / "P0_READINESS.json",
        "p0_audit": p0_dir / "AUDIT.json",
        "p0_rows": p0_dir / "P0_RENDERER_ROWS_REPLAY0.jsonl",
        "smoke_plan": p0_dir / "SMOKE_PLAN.jsonl",
        "repo": root / "repos/SemStamp",
        "data_repo": root / "repos/semstamp-data",
    }


def independently_select_smoke(rows):
    selected = []
    for cohort in COHORTS:
        cohort_rows = [row for row in rows if row["cohort"] == cohort]
        if len(cohort_rows) != COHORT_SIZE:
            raise ValueError(f"cohort count mismatch: {cohort}")
        mandatory = sorted(
            (
                row for row in cohort_rows
                if (
                    row["refinement_count"] > 0
                    or row["zero_byte_gap_count"] > 0
                )
            ),
            key=lambda row: row["plan_index"],
        )
        if len(mandatory) > SMOKE_PER_COHORT:
            raise ValueError(f"mandatory budget exceeded: {cohort}")
        mandatory_indices = {row["plan_index"] for row in mandatory}
        remaining = sorted(
            (
                row for row in cohort_rows
                if row["plan_index"] not in mandatory_indices
            ),
            key=lambda row: (
                row["atom_count"],
                row["unit_count"],
                row["raw_text_sha256"],
                row["plan_index"],
            ),
        )
        fill_count = SMOKE_PER_COHORT - len(mandatory)
        ranks = [
            ((2 * index + 1) * len(remaining)) // (2 * fill_count)
            for index in range(fill_count)
        ]
        choices = [
            ("mandatory_refinement", None, row)
            for row in mandatory
        ] + [
            ("structural_quantile", rank, remaining[rank])
            for rank in ranks
        ]
        for cohort_selection_index, (
            selection_class,
            structural_rank,
            row,
        ) in enumerate(choices):
            selected.append({
                "schema_version": 1,
                "stage": (
                    "semstamp_certified_defense_v2_wp3_detector_health_p0"
                ),
                "protocol_sha256": (
                    "3d6d071e64a9bba12d5cf97d78b1f8ca8ce73c8318def7cbc2fef6bc0299acb3"
                ),
                "smoke_index": len(selected),
                "cohort": cohort,
                "cohort_selection_index": cohort_selection_index,
                "selection_class": selection_class,
                "structural_rank": structural_rank,
                "plan_index": row["plan_index"],
                "cohort_index": row["cohort_index"],
                "raw_text_sha256": row["raw_text_sha256"],
                "atom_count": row["atom_count"],
                "unit_count": row["unit_count"],
                "refinement_count": row["refinement_count"],
                "zero_byte_gap_count": row["zero_byte_gap_count"],
                "topology_sha256": row["topology_sha256"],
                "legality_sha256": row["legality_sha256"],
                "wp1_reference_row_sha256": (
                    row["wp1_reference_row_sha256"]
                ),
                "score_access_count_during_construction": 0,
                "contains_raw_text": False,
            })
    return selected


def imported_root(node):
    if isinstance(node, ast.Import):
        return {alias.name.split(".", 1)[0] for alias in node.names}
    if isinstance(node, ast.ImportFrom) and node.module:
        return {node.module.split(".", 1)[0]}
    return set()


def called_function_names(function):
    names = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.append((node.lineno, node.func.id))
        elif isinstance(node.func, ast.Attribute):
            names.append((node.lineno, node.func.attr))
    return [name for _line, name in sorted(names)]


def audit_runner_source(path):
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_imports = set()
    functions = {}
    for node in tree.body:
        top_imports.update(imported_root(node))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = node
    if top_imports & FORBIDDEN_TOP_LEVEL_IMPORTS:
        raise ValueError("model stack imported at module scope")
    if "run_full" in functions:
        raise ValueError("runner exposes a full replay entry point")
    required = {
        "validate_authorization",
        "initialize_detector",
        "run_smoke",
        "parse_args",
    }
    if not required.issubset(functions):
        raise ValueError("runner entrypoint functions are incomplete")
    calls = called_function_names(functions["run_smoke"])
    try:
        authorization_index = calls.index("validate_authorization")
        detector_index = calls.index("initialize_detector")
        execute_index = calls.index("execute_replays")
    except ValueError as error:
        raise ValueError("runner call sequence is incomplete") from error
    if not authorization_index < detector_index < execute_index:
        raise ValueError("authorization does not precede model execution")
    initialize_imports = set()
    for node in functions["initialize_detector"].body:
        initialize_imports.update(imported_root(node))
    if (
        "torch" not in initialize_imports
        or "run_certified_resegmentation_wp2_detector_equivalence"
        not in initialize_imports
    ):
        raise ValueError("detector imports are not isolated as expected")
    full_string = any(
        isinstance(node, ast.Constant) and node.value == "full"
        for node in ast.walk(functions["parse_args"])
    )
    if full_string:
        raise ValueError("parser contains a full execution phase")
    return {
        "top_level_model_imports": 0,
        "authorization_precedes_detector": True,
        "smoke_only_entrypoint": True,
        "detector_imports_isolated": True,
    }


def audit(root, output_dir, output):
    paths = paths_for(root, output_dir, output)
    contract = read_json(paths["contract"])
    required_contract = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "frozen_design",
        "protocol_sha256": PROTOCOL_SHA256,
        "renderer_contract": "refined_nltk_boundary_atoms_v2",
        "cohorts": list(COHORTS),
        "cohort_size": COHORT_SIZE,
        "total_parent_rows": TOTAL_ROWS,
        "smoke_per_cohort": SMOKE_PER_COHORT,
        "smoke_rows": SMOKE_ROWS,
        "replay_count": 2,
        "batch_size": 64,
        "native_token_limit": 512,
        "minimum_free_gib": 5.0,
        "maximum_vram_fraction": 0.8,
        "maximum_gpu_seconds": 900,
        "semstamp_revision": SEMSTAMP_REVISION,
        "semstamp_data_revision": SEMSTAMP_DATA_REVISION,
        "model_revision": MODEL_REVISION,
        "relation_sha256": RELATION_SHA256,
        "punkt_payload_sha256": PUNKT_SHA256,
        "p0_final_sha256": P0_FINAL_SHA256,
        "p0_readiness_sha256": P0_READINESS_SHA256,
        "p0_audit_sha256": P0_AUDIT_SHA256,
        "p0_rows_sha256": P0_ROWS_SHA256,
        "smoke_plan_sha256": SMOKE_PLAN_SHA256,
        "execution_authorized": False,
        "full_replay_authorized": False,
        "wp4_authorized": False,
    }
    if any(
        contract.get(key) != value
        for key, value in required_contract.items()
    ):
        raise ValueError("contract constants mismatch")
    artifact_fields = {
        "protocol_sha256": paths["protocol"],
        "runner_sha256": paths["runner"],
        "renderer_sha256": paths["renderer"],
        "p0_runner_sha256": paths["p0_runner"],
        "test_sha256": paths["test"],
        "auditor_sha256": paths["auditor"],
        "input_manifest_sha256": paths["manifest"],
    }
    if any(
        contract.get(field) != sha256_file(path)
        for field, path in artifact_fields.items()
    ):
        raise ValueError("contract artifact binding mismatch")

    manifest = parse_manifest(paths["manifest"])
    if set(manifest) != set(contract.get("inputs", {})):
        raise ValueError("manifest/contract input set mismatch")
    for relative, expected_sha in manifest.items():
        relative_path = Path(relative)
        path = (paths["root"] / relative_path).resolve()
        value = contract["inputs"][relative]
        if (
            relative_path.is_absolute()
            or not path.is_relative_to(paths["root"])
            or not path.is_file()
            or sha256_file(path) != expected_sha
            or path.stat().st_size != value.get("bytes")
            or value.get("sha256") != expected_sha
        ):
            raise ValueError(f"input drift: {relative}")
        if (
            value.get("mode") is not None
            and f"{stat.S_IMODE(path.stat().st_mode):04o}" != value["mode"]
        ):
            raise ValueError(f"input mode drift: {relative}")
    if git_head(paths["repo"]) != SEMSTAMP_REVISION:
        raise ValueError("SemStamp revision drift")
    if git_head(paths["data_repo"]) != SEMSTAMP_DATA_REVISION:
        raise ValueError("semstamp-data revision drift")

    parent_hashes = {
        paths["p0_final"]: P0_FINAL_SHA256,
        paths["p0_readiness"]: P0_READINESS_SHA256,
        paths["p0_audit"]: P0_AUDIT_SHA256,
        paths["p0_rows"]: P0_ROWS_SHA256,
        paths["smoke_plan"]: SMOKE_PLAN_SHA256,
    }
    if any(
        sha256_file(path) != digest
        for path, digest in parent_hashes.items()
    ):
        raise ValueError("P0 parent artifact drift")
    final = read_json(paths["p0_final"])
    readiness = read_json(paths["p0_readiness"])
    parent_audit = read_json(paths["p0_audit"])
    renderer_rows = read_jsonl(paths["p0_rows"])
    smoke_plan = read_jsonl(paths["smoke_plan"])
    reject_raw_public_keys([
        final, readiness, parent_audit, renderer_rows, smoke_plan
    ])
    if (
        final.get("status") != "go"
        or final.get("terminal") != "WP3_P0_GO_FOR_SMOKE_DESIGN"
        or readiness.get("status") != "go"
        or not all(readiness.get("gates", {}).values())
        or parent_audit.get("status") != "go"
        or not all(parent_audit.get("checks", {}).values())
        or len(renderer_rows) != TOTAL_ROWS
    ):
        raise ValueError("P0 parent terminal mismatch")
    rebuilt = independently_select_smoke(renderer_rows)
    if (
        jsonl_bytes(rebuilt) != paths["smoke_plan"].read_bytes()
        or smoke_plan != rebuilt
    ):
        raise ValueError("smoke plan is not independently reproducible")
    risk_indices = {
        row["plan_index"]
        for row in renderer_rows
        if row["refinement_count"] or row["zero_byte_gap_count"]
    }
    selected_indices = {row["plan_index"] for row in smoke_plan}
    if (
        len(smoke_plan) != SMOKE_ROWS
        or len(selected_indices) != SMOKE_ROWS
        or Counter(row["cohort"] for row in smoke_plan)
        != {cohort: SMOKE_PER_COHORT for cohort in COHORTS}
        or len(risk_indices) != EXPECTED_MANDATORY_ROWS
        or not risk_indices.issubset(selected_indices)
        or sum(row["zero_byte_gap_count"] for row in smoke_plan)
        != EXPECTED_ZERO_GAPS
        or any(
            row["score_access_count_during_construction"] != 0
            for row in smoke_plan
        )
    ):
        raise ValueError("smoke plan coverage mismatch")

    source_audit = audit_runner_source(paths["runner"])
    forbidden_outputs = (
        "authorization",
        "rows0",
        "rows1",
        "checkpoint",
        "heartbeat",
        "smoke_result",
        "last_failure",
    )
    existing_forbidden = [
        name for name in forbidden_outputs if paths[name].exists()
    ]
    if existing_forbidden:
        raise ValueError(
            f"execution artifacts exist during design: {existing_forbidden}"
        )

    checks = {
        "protocol_and_contract_frozen": True,
        "all_input_bytes_sha_modes_exact": True,
        "repository_revisions_exact": True,
        "p0_go_terminal_exact": True,
        "p0_independent_audit_exact": True,
        "smoke_plan_byte_exact": True,
        "smoke_plan_independently_recomputed": True,
        "smoke_96_unique_and_balanced": True,
        "all_42_risk_rows_covered": True,
        "all_59_zero_gaps_covered": True,
        "selection_score_access_zero": True,
        "authorization_file_absent": True,
        "execution_outputs_absent": True,
        "authorization_precedes_detector_import": True,
        "detector_imports_isolated": True,
        "runner_has_smoke_only_entrypoint": True,
        "runner_has_no_full_replay_entrypoint": True,
        "public_artifacts_contain_no_raw_text": True,
        "model_gpu_score_certificate_calls_zero": True,
    }
    result = {
        "schema_version": 1,
        "stage": AUDIT_STAGE,
        "status": "go" if all(checks.values()) else "stop",
        "terminal": (
            "WP3_SMOKE_DESIGN_FROZEN_PENDING_EXECUTION_AUTHORIZATION"
        ),
        "scope": "model_free_static_and_input_binding_design_audit",
        "bindings": {
            "protocol": sha256_file(paths["protocol"]),
            "contract": sha256_file(paths["contract"]),
            "manifest": sha256_file(paths["manifest"]),
            "runner": sha256_file(paths["runner"]),
            "renderer": sha256_file(paths["renderer"]),
            "p0_runner": sha256_file(paths["p0_runner"]),
            "test": sha256_file(paths["test"]),
            "auditor": sha256_file(paths["auditor"]),
            "p0_final": sha256_file(paths["p0_final"]),
            "p0_readiness": sha256_file(paths["p0_readiness"]),
            "p0_audit": sha256_file(paths["p0_audit"]),
            "p0_rows": sha256_file(paths["p0_rows"]),
            "smoke_plan": sha256_file(paths["smoke_plan"]),
        },
        "checks": checks,
        "counts": {
            "input_manifest_entries": len(manifest),
            "parent_rows": len(renderer_rows),
            "smoke_rows": len(smoke_plan),
            "smoke_cohorts": dict(Counter(
                row["cohort"] for row in smoke_plan
            )),
            "mandatory_risk_rows": len(risk_indices),
            "zero_byte_gaps": sum(
                row["zero_byte_gap_count"] for row in smoke_plan
            ),
        },
        "source_audit": source_audit,
        "execution_attestation": {
            "authorization_files_created": 0,
            "certificate_calls": 0,
            "detector_imports": 0,
            "detector_model_loads": 0,
            "detector_score_reads": 0,
            "detector_scores_computed": 0,
            "generation_calls": 0,
            "gpu_calls": 0,
            "smoke_executed": False,
        },
        "decision": {
            "smoke_design": "frozen",
            "smoke_execution": "not_authorized",
            "smoke_started": False,
            "full_replay_authorized": False,
            "full_replay_started": False,
            "wp4_authorized": False,
            "wp4_started": False,
        },
        "contains_raw_text": False,
    }
    reject_raw_public_keys(result)
    atomic_write_json(paths["audit"], result)
    print(json.dumps(result, sort_keys=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.output_dir, args.output)


if __name__ == "__main__":
    main()
