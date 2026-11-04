"""Freeze v4 source identities and check local resource-audit inputs."""

import json
from pathlib import Path

from run_semstamp_certified_defense_v4_full_radius_profile import (
    FEATURE_BATCH_SIZE, GPU_HOUR_CAP, NATIVE_TOKEN_LIMIT, PAIR_CAP,
    RADII, RSS_CAP_BYTES, SIDECAR_CAP_BYTES, SPAN_CAP, STAGE,
    atomic_write_bytes, atomic_write_json, sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / STAGE


def prepare():
    old = ROOT / "experiments/semstamp_certified_defense_v3_wp3_variable_n_profile1"
    base = json.loads((old / "CONTRACT.json").read_text())
    # Model files and sealed inputs are required for execution, even if absent.
    inputs = {k: v for k, v in base["inputs"].items()
              if k not in {
                  "build/semstamp_v3/semstamp_certified_defense_v3_variable_n_core",
                  "scripts/run_semstamp_certified_defense_v3_variable_n_profile.py",
                  "scripts/semstamp_certified_defense_v3_variable_n_core.cpp",
              }}
    for relative in (
        "scripts/run_semstamp_certified_defense_v4_full_radius_profile.py",
        "scripts/semstamp_certified_defense_v4_full_radius_core.cpp",
        "scripts/audit_semstamp_certified_defense_v4_radius_correctness.py",
        "scripts/audit_semstamp_certified_defense_v4_resource_readiness.py",
        "scripts/prepare_semstamp_certified_defense_v4.py",
        "scripts/run_semstamp_certified_defense_v3_compiled_correctness.py",
        "scripts/semstamp_certified_defense_v1_variable_n.py",
        "scripts/semstamp_certified_defense_v3_variable_n_core.cpp",
        "scripts/certified_resegmentation_wp2_v13_b0_support.py",
        "experiments/semantic_watermark_defense_plan_20261007/THREAT_MODEL.md",
        "experiments/semantic_watermark_defense_plan_20261007/ALGORITHM_CONTRACT.md",
        "experiments/semstamp_certified_defense_v3_wp3_variable_n_profile1/PROFILE_ROWS.jsonl",
        "experiments/semstamp_certified_defense_v3_wp3_variable_n_profile1/PROFILE_RESULT.json",
        "experiments/certified_resegmentation_wp2_v17_b0_detector_health1/B0_PLAN.jsonl",
        "experiments/certified_resegmentation_wp2_v16_prompt_only_rescue1/H0_RESULT.json",
        "experiments/certified_resegmentation_wp2_v16_prompt_only_rescue1/COMBINED_POSITIVE_IDENTITIES.jsonl",
        "experiments/certified_resegmentation_wp2_v12_fresh_assets1/SOURCE_IDENTITIES.jsonl",
    ):
        inputs[relative] = sha256_file(ROOT / relative)
    legacy_p0 = json.loads((
        ROOT / "experiments/semstamp_certified_defense_v2_wp3_detector_health1/CONTRACT.json"
    ).read_text())
    sealed_names = (
        "experiments/certified_resegmentation_wp2_v12_fresh_assets1/sealed/source_rows.jsonl",
        "experiments/certified_resegmentation_wp2_v15_causal_prefix1/sealed/generation_checkpoint.json",
        "experiments/certified_resegmentation_wp2_v16_prompt_only_rescue1/sealed/supplement_checkpoint.json",
    )
    for relative in sealed_names:
        meta = legacy_p0["inputs"][relative]
        inputs[relative] = meta["sha256"]
    core = OUT / "LINUX_BUILD.json"
    linux_core_sha = json.loads(core.read_text())["core_sha256"] if core.is_file() else None
    if linux_core_sha:
        inputs["build/semstamp_v4/semstamp_certified_defense_v4_full_radius_core"] = linux_core_sha
    manifest = b"".join(
        f"{digest}  {relative}\n".encode("ascii")
        for relative, digest in sorted(inputs.items())
    )
    atomic_write_bytes(OUT / "INPUT_MANIFEST.sha256", manifest)
    contract = {
        "schema_version": 1, "stage": STAGE,
        "status": "frozen_and_authorized",
        "authorization": "contract_alignment_and_score_blind_resource_audit_only",
        "total_rows": 3000, "radii": list(RADII),
        "radius_rule": "requested_b_without_original_edge_cap",
        "operation_cost": "max(insertions,deletions)",
        "renderer": "refined_nltk_boundary_atoms_v2",
        "feature_batch_size": FEATURE_BATCH_SIZE,
        "native_token_limit": NATIVE_TOKEN_LIMIT,
        "pair_cap_per_layer": PAIR_CAP,
        "span_cap_per_radius_union": SPAN_CAP,
        "rss_cap_bytes": RSS_CAP_BYTES, "sidecar_cap_bytes": SIDECAR_CAP_BYTES,
        "gpu_hour_cap": GPU_HOUR_CAP,
        "score_blind": True, "certificate_authorized": False,
        "successor_execution_authorized": False,
        "execution": {"workers": 6, "core_timeout_seconds": 3600,
                      "checkpoint": "atomic_combined_row_and_timing_bound_to_contract"},
        "protocol_sha256": sha256_file(OUT / "PROTOCOL.md"),
        "runner_sha256": sha256_file(
            ROOT / "scripts/run_semstamp_certified_defense_v4_full_radius_profile.py"),
        "linux_core_sha256": linux_core_sha,
        "manifest_sha256": sha256_file(OUT / "INPUT_MANIFEST.sha256"),
        "inputs": inputs,
        "original_v3_rows_sha256": base.get("rows_sha256") or sha256_file(old / "PROFILE_ROWS.jsonl"),
        "historical_v3_terminal": "V3_VARIABLE_N_PROFILE_READINESS_STOP",
        "full_profile_requires_readiness_go": True,
    }
    atomic_write_json(OUT / "CONTRACT.json", contract)
    checks = [{
        "relative_path": relative, "expected_sha256": expected,
        "present": (ROOT / relative).is_file(),
        "sha256_exact": (
            sha256_file(ROOT / relative) == expected
            if (ROOT / relative).is_file() else False),
    } for relative, expected in sorted(inputs.items())]
    # The Linux executable is frozen separately; a local Mach-O binary is
    # intentionally not used as an execution substitute.
    atomic_write_json(OUT / "LOCAL_INPUT_READINESS.json", {
        "schema": "semstamp-v4-input-readiness/v1",
        "status": "pass" if all(row["sha256_exact"] for row in checks) else "stop",
        "checks": checks,
        "sealed_input_count": len(sealed_names),
        "sealed_input_availability": [{
            "relative_path": name, "present": (ROOT / name).is_file(),
        } for name in sealed_names],
        "semantic_model_loads": 0, "full_profile_started": False,
    })
    print(json.dumps({
        "contract_frozen": True,
        "missing_inputs": [row["relative_path"] for row in checks if not row["present"]],
        "mismatched_present_inputs": [row["relative_path"] for row in checks
                                     if row["present"] and not row["sha256_exact"]],
    }, sort_keys=True))


if __name__ == "__main__":
    prepare()
