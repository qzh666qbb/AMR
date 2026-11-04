"""Audit the frozen v2 WP3 full detector-replay design without executing it."""

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


STAGE = "semstamp_certified_defense_v2_wp3_detector_full_replay"
AUDIT_STAGE = f"{STAGE}_design_audit"
PROTOCOL_SHA256 = (
    "c4102127400bf1b20c7aa9a9849252bc7f737e8b628e67699bb4269213fb0d91"
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
SMOKE_FINAL_SHA256 = (
    "64e509330781183057c7c479dacc3ae5862a39923f18dbfa3401c53236baa637"
)
SMOKE_RESULT_SHA256 = (
    "b604c5ce22be68a1fb08d0774b39f37aa19007c5d5d9b9bc998594aa696b0fe8"
)
SMOKE_AUDIT_SHA256 = (
    "163933e86d171fe5f289441580606d7bce66cfe6867baf59553438579f9badf5"
)
SMOKE_REPLAY_SHA256 = (
    "a4b7403bae9095be1a5235b447a57bfc722f19c6c20556c6aae0df867dc2cc7b"
)
SMOKE_INPUT_MANIFEST_SHA256 = (
    "326f36d14e8dcb7c12ce019a2aa3668d67b30ad3f5a9284bbf4e4451fde69697"
)
P0_ROWS_SHA256 = (
    "8625f6b074150d4ec099b9e57da00d6b08167b42f25ad95d34b975682e5a7474"
)
V17_ROWS_SHA256 = (
    "75d0fe012215958727da715a0e916240837dd2ef0daac6b02c78c6500410824f"
)
V17_RESULT_SHA256 = (
    "c17a382d6946fe93e09df34f420b2ff07a9ccc9daad184b55c71d23118cf9e9e"
)
COHORTS = ("calibration_null", "independent_test_null", "positive")
COHORT_SIZE = 1000
TOTAL_ROWS = 3000
EXPECTED_MANDATORY_ROWS = 42
EXPECTED_ZERO_GAPS = 59
DETERMINISTIC_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_VISIBLE_DEVICES": "0",
    "HF_HUB_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
}
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
    smoke_dir = (
        root
        / "experiments/semstamp_certified_defense_v2_wp3_detector_smoke_design1"
    )
    v17_dir = (
        root
        / "experiments/certified_resegmentation_wp2_v17_b0_detector_health1"
    )
    return {
        "root": root,
        "output_dir": output_dir,
        "protocol": output_dir / "PROTOCOL.md",
        "contract": output_dir / "CONTRACT.json",
        "manifest": output_dir / "INPUT_MANIFEST.sha256",
        "authorization": output_dir / "EXECUTION_AUTHORIZATION.json",
        "resume_authorization": output_dir / "RESUME_AUTHORIZATION.json",
        "rows0": output_dir / "FULL_REPLAY_ROWS_REPLAY0.jsonl",
        "rows1": output_dir / "FULL_REPLAY_ROWS_REPLAY1.jsonl",
        "checkpoint": output_dir / "sealed/FULL_REPLAY_CHECKPOINT.json",
        "heartbeat": output_dir / "HEARTBEAT.json",
        "result": output_dir / "FULL_REPLAY_RESULT.json",
        "last_failure": output_dir / "LAST_FAILURE.json",
        "lock": output_dir / "EXECUTION.lock",
        "audit": output,
        "runner": (
            root
            / "scripts/run_semstamp_certified_defense_v2_wp3_full_replay.py"
        ),
        "smoke_runner": (
            root / "scripts/run_semstamp_certified_defense_v2_wp3_smoke.py"
        ),
        "renderer": root / "scripts/semstamp_refined_atom_renderer.py",
        "p0_runner": (
            root / "scripts/run_semstamp_certified_defense_v2_wp3_p0.py"
        ),
        "test": (
            root
            / "scripts/test_semstamp_certified_defense_v2_wp3_full_replay.py"
        ),
        "auditor": Path(__file__).resolve(),
        "smoke_final": smoke_dir / "FINAL_RESULT.json",
        "smoke_result": smoke_dir / "SMOKE_RESULT.json",
        "smoke_audit": smoke_dir / "EXECUTION_AUDIT.json",
        "smoke_contract": smoke_dir / "CONTRACT.json",
        "smoke_manifest": smoke_dir / "INPUT_MANIFEST.sha256",
        "p0_rows": p0_dir / "P0_RENDERER_ROWS_REPLAY0.jsonl",
        "v17_plan": v17_dir / "B0_PLAN.jsonl",
        "v17_rows": v17_dir / "B0_ROWS_REPLAY0.jsonl",
        "v17_result": v17_dir / "B0_RESULT.json",
        "repo": root / "repos/SemStamp",
        "data_repo": root / "repos/semstamp-data",
    }


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
    return [name for _, name in sorted(names)]


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
    required = {
        "validate_authorization",
        "validate_pre_model_environment",
        "validate_resume_authorization",
        "validate_checkpoint",
        "run_full_replay",
        "parse_args",
    }
    if not required.issubset(functions):
        raise ValueError("runner entrypoint functions are incomplete")
    if "run_wp4" in functions or "run_certificate" in functions:
        raise ValueError("runner exposes an out-of-scope entry point")
    calls = called_function_names(functions["run_full_replay"])
    try:
        authorization_index = calls.index("validate_authorization")
        environment_index = calls.index("validate_pre_model_environment")
        detector_index = calls.index("initialize_detector")
        execute_index = calls.index("execute_replays")
    except ValueError as error:
        raise ValueError("runner call sequence is incomplete") from error
    if not (
        authorization_index
        < environment_index
        < detector_index
        < execute_index
    ):
        raise ValueError("authorization/environment do not precede detector")
    load_calls = called_function_names(functions["load_or_create_checkpoint"])
    if "validate_resume_authorization" not in load_calls:
        raise ValueError("checkpoint resume authorization is not enforced")
    if (
        '"CUBLAS_WORKSPACE_CONFIG": ":4096:8"' not in source
        or '"HF_HUB_OFFLINE": "1"' not in source
        or '"TRANSFORMERS_OFFLINE": "1"' not in source
    ):
        raise ValueError("deterministic/offline environment contract missing")
    parse_constants = {
        node.value
        for node in ast.walk(functions["parse_args"])
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    if "full_replay" not in parse_constants or "wp4" in parse_constants:
        raise ValueError("runner phase scope is invalid")
    return {
        "top_level_model_imports": 0,
        "authorization_precedes_environment_and_detector": True,
        "deterministic_environment_precedes_detector": True,
        "resume_authorization_enforced": True,
        "full_replay_only_entrypoint": True,
    }


def validate_parent_terminals(paths):
    expected = {
        paths["smoke_final"]: SMOKE_FINAL_SHA256,
        paths["smoke_result"]: SMOKE_RESULT_SHA256,
        paths["smoke_audit"]: SMOKE_AUDIT_SHA256,
        paths["p0_rows"]: P0_ROWS_SHA256,
        paths["v17_rows"]: V17_ROWS_SHA256,
        paths["v17_result"]: V17_RESULT_SHA256,
    }
    if any(sha256_file(path) != digest for path, digest in expected.items()):
        raise ValueError("parent artifact SHA drift")
    final = read_json(paths["smoke_final"])
    smoke_result = read_json(paths["smoke_result"])
    smoke_audit = read_json(paths["smoke_audit"])
    v17_result = read_json(paths["v17_result"])
    reject_raw_public_keys([final, smoke_result, smoke_audit, v17_result])
    if (
        final.get("status") != "go"
        or final.get("terminal") != "WP3_SMOKE_GO_FOR_FULL_REPLAY_DESIGN"
        or final.get("decision", {}).get("full_replay_execution_authorized")
        is not False
        or smoke_result.get("status") != "go"
        or not all(smoke_result.get("gates", {}).values())
        or smoke_result.get("outputs", {}).get("replay0", {}).get("sha256")
        != SMOKE_REPLAY_SHA256
        or smoke_result.get("outputs", {}).get("replay1", {}).get("sha256")
        != SMOKE_REPLAY_SHA256
        or smoke_audit.get("status") != "go"
        or not all(smoke_audit.get("checks", {}).values())
        or v17_result.get("status") != "go"
        or v17_result.get("terminal_reason") != "b0_detector_health_go"
        or not all(v17_result.get("gates", {}).values())
    ):
        raise ValueError("parent terminal mismatch")
    return final, smoke_result, smoke_audit, v17_result


def validate_inherited_inputs(paths):
    if sha256_file(paths["smoke_manifest"]) != SMOKE_INPUT_MANIFEST_SHA256:
        raise ValueError("inherited smoke manifest SHA drift")
    inherited_manifest = parse_manifest(paths["smoke_manifest"])
    inherited_contract = read_json(paths["smoke_contract"])
    inherited_inputs = inherited_contract.get("inputs", {})
    if (
        len(inherited_manifest) != 46
        or set(inherited_manifest) != set(inherited_inputs)
    ):
        raise ValueError("inherited smoke input set mismatch")
    for relative, expected_sha in inherited_manifest.items():
        relative_path = Path(relative)
        path = (paths["root"] / relative_path).resolve()
        value = inherited_inputs[relative]
        if (
            relative_path.is_absolute()
            or not path.is_relative_to(paths["root"])
            or not path.is_file()
            or sha256_file(path) != expected_sha
            or path.stat().st_size != value.get("bytes")
            or value.get("sha256") != expected_sha
        ):
            raise ValueError(f"inherited input drift: {relative}")
        if (
            value.get("mode") is not None
            and f"{stat.S_IMODE(path.stat().st_mode):04o}" != value["mode"]
        ):
            raise ValueError(f"inherited input mode drift: {relative}")
    return inherited_manifest


def validate_denominator(paths):
    renderer_rows = read_jsonl(paths["p0_rows"])
    reference_rows = read_jsonl(paths["v17_rows"])
    plan = read_jsonl(paths["v17_plan"])
    reject_raw_public_keys([renderer_rows, reference_rows, plan])
    if (
        len(renderer_rows) != TOTAL_ROWS
        or len(reference_rows) != TOTAL_ROWS
        or len(plan) != TOTAL_ROWS
    ):
        raise ValueError("full denominator row count mismatch")
    for index, (renderer, reference, planned) in enumerate(
        zip(renderer_rows, reference_rows, plan)
    ):
        expected = {
            "plan_index": index,
            "cohort": COHORTS[index // COHORT_SIZE],
            "cohort_index": index % COHORT_SIZE,
        }
        if any(
            row.get(key) != value
            for row in (renderer, reference, planned)
            for key, value in expected.items()
        ):
            raise ValueError(f"full denominator order mismatch at {index}")
        if not (
            renderer["raw_text_sha256"]
            == reference["raw_text_sha256"]
            == planned["raw_text_sha256"]
        ):
            raise ValueError(f"full denominator identity mismatch at {index}")
        if not (
            renderer["unit_count"]
            == reference["unit_count"]
            == planned["unit_count"]
        ):
            raise ValueError(f"full denominator unit mismatch at {index}")
    cohort_counts = Counter(row["cohort"] for row in renderer_rows)
    risk_rows = sum(
        bool(row["refinement_count"] or row["zero_byte_gap_count"])
        for row in renderer_rows
    )
    zero_gaps = sum(row["zero_byte_gap_count"] for row in renderer_rows)
    if (
        cohort_counts != {cohort: COHORT_SIZE for cohort in COHORTS}
        or risk_rows != EXPECTED_MANDATORY_ROWS
        or zero_gaps != EXPECTED_ZERO_GAPS
    ):
        raise ValueError("full denominator coverage mismatch")
    return cohort_counts, risk_rows, zero_gaps


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
        "rows_per_replay": TOTAL_ROWS,
        "replay_count": 2,
        "batch_size": 64,
        "native_token_limit": 512,
        "minimum_free_gib": 5.0,
        "maximum_vram_fraction": 0.8,
        "maximum_gpu_seconds": 1800,
        "deterministic_environment": DETERMINISTIC_ENVIRONMENT,
        "semstamp_revision": SEMSTAMP_REVISION,
        "semstamp_data_revision": SEMSTAMP_DATA_REVISION,
        "model_revision": MODEL_REVISION,
        "relation_sha256": RELATION_SHA256,
        "punkt_payload_sha256": PUNKT_SHA256,
        "p0_rows_sha256": P0_ROWS_SHA256,
        "v17_rows_sha256": V17_ROWS_SHA256,
        "v17_result_sha256": V17_RESULT_SHA256,
        "smoke_final_sha256": SMOKE_FINAL_SHA256,
        "smoke_result_sha256": SMOKE_RESULT_SHA256,
        "smoke_audit_sha256": SMOKE_AUDIT_SHA256,
        "smoke_replay_sha256": SMOKE_REPLAY_SHA256,
        "inherited_input_manifest_sha256": SMOKE_INPUT_MANIFEST_SHA256,
        "inherited_input_count": 46,
        "execution_authorized": False,
        "resume_authorized": False,
        "wp4_authorized": False,
        "certificate_authorized": False,
    }
    if any(
        contract.get(key) != value
        for key, value in required_contract.items()
    ):
        raise ValueError("contract constants mismatch")
    artifact_fields = {
        "protocol_sha256": paths["protocol"],
        "runner_sha256": paths["runner"],
        "smoke_runner_sha256": paths["smoke_runner"],
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
    inherited_manifest = validate_inherited_inputs(paths)
    if git_head(paths["repo"]) != SEMSTAMP_REVISION:
        raise ValueError("SemStamp revision drift")
    if git_head(paths["data_repo"]) != SEMSTAMP_DATA_REVISION:
        raise ValueError("semstamp-data revision drift")

    validate_parent_terminals(paths)
    cohort_counts, risk_rows, zero_gaps = validate_denominator(paths)
    source_audit = audit_runner_source(paths["runner"])
    forbidden_outputs = (
        "authorization",
        "resume_authorization",
        "rows0",
        "rows1",
        "checkpoint",
        "heartbeat",
        "result",
        "last_failure",
        "lock",
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
        "all_inherited_input_bytes_sha_modes_exact": True,
        "repository_revisions_exact": True,
        "smoke_go_terminal_exact": True,
        "smoke_independent_execution_audit_exact": True,
        "v17_health_terminal_exact": True,
        "full_3000_row_denominator_exact": True,
        "balanced_1000_per_cohort": True,
        "all_42_risk_rows_retained": True,
        "all_59_zero_gaps_retained": True,
        "execution_authorization_absent": True,
        "resume_authorization_absent": True,
        "execution_outputs_absent": True,
        "authorization_precedes_detector_import": True,
        "cublas_and_offline_environment_precede_detector_import": True,
        "checkpoint_resume_requires_separate_authorization": True,
        "runner_has_full_replay_only_entrypoint": True,
        "runner_has_no_wp4_or_certificate_entrypoint": True,
        "public_artifacts_contain_no_raw_text": True,
        "model_gpu_score_certificate_calls_zero": True,
    }
    result = {
        "schema_version": 1,
        "stage": AUDIT_STAGE,
        "status": "go" if all(checks.values()) else "stop",
        "terminal": (
            "WP3_FULL_REPLAY_DESIGN_FROZEN_PENDING_EXECUTION_AUTHORIZATION"
        ),
        "scope": "model_free_static_and_input_binding_design_audit",
        "bindings": {
            "protocol": sha256_file(paths["protocol"]),
            "contract": sha256_file(paths["contract"]),
            "manifest": sha256_file(paths["manifest"]),
            "runner": sha256_file(paths["runner"]),
            "smoke_runner": sha256_file(paths["smoke_runner"]),
            "renderer": sha256_file(paths["renderer"]),
            "p0_runner": sha256_file(paths["p0_runner"]),
            "test": sha256_file(paths["test"]),
            "auditor": sha256_file(paths["auditor"]),
            "smoke_final": sha256_file(paths["smoke_final"]),
            "smoke_result": sha256_file(paths["smoke_result"]),
            "smoke_audit": sha256_file(paths["smoke_audit"]),
            "p0_rows": sha256_file(paths["p0_rows"]),
            "v17_rows": sha256_file(paths["v17_rows"]),
            "v17_result": sha256_file(paths["v17_result"]),
        },
        "checks": checks,
        "counts": {
            "input_manifest_entries": len(manifest),
            "inherited_input_manifest_entries": len(inherited_manifest),
            "rows_per_replay": TOTAL_ROWS,
            "replays": 2,
            "cohorts": dict(cohort_counts),
            "mandatory_risk_rows": risk_rows,
            "zero_byte_gaps": zero_gaps,
        },
        "source_audit": source_audit,
        "execution_attestation": {
            "authorization_files_created": 0,
            "resume_authorization_files_created": 0,
            "certificate_calls": 0,
            "detector_imports": 0,
            "detector_model_loads": 0,
            "detector_score_reads": 0,
            "detector_scores_computed": 0,
            "generation_calls": 0,
            "gpu_calls": 0,
            "full_replay_executed": False,
            "wp4_executed": False,
        },
        "decision": {
            "full_replay_design": "frozen",
            "full_replay_execution": "not_authorized",
            "full_replay_started": False,
            "wp4_authorized": False,
            "wp4_started": False,
            "certificate_authorized": False,
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
