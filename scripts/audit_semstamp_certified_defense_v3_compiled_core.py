"""Independently audit v3 compiled-core correctness and scale evidence."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


OBSERVATION_SHA256 = (
    "7390f0e621279891703f287c7faf1d9aa428e9b53546c04f55be1dd15f3fde68"
)
HISTORICAL_PAIRS = 83_540_900
HISTORICAL_SPANS = 769_113
PAIR_CAP = 200_000_000
SPAN_CAP = 2_000_000
RSS_CAP = 16 * 1024**3
TIME_CAP = 3_600
FORBIDDEN_KEYS = {
    "generated_text",
    "prompt",
    "raw_text",
    "source_text",
    "text",
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


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def reject_raw_keys(value, location="root"):
    if isinstance(value, dict):
        overlap = FORBIDDEN_KEYS & set(value)
        if overlap:
            raise ValueError(f"raw key at {location}: {sorted(overlap)}")
        for key, child in value.items():
            reject_raw_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_raw_keys(child, f"{location}[{index}]")


def comparable_profile(profile):
    return {
        "result": profile["result"],
        "stdout_sha256": profile["stdout_sha256"],
        "target_edge_count": profile["target_edge_count"],
    }


def audit(args):
    root = Path(args.root).resolve()
    core1 = root / args.core1_dir
    scale2 = root / args.scale2_dir
    source = root / "scripts/semstamp_certified_defense_v3_variable_n_core.cpp"
    correctness_runner = (
        root
        / "scripts/run_semstamp_certified_defense_v3_compiled_correctness.py"
    )
    base_scale_runner = (
        root / "scripts/run_semstamp_certified_defense_v3_maximum_scale.py"
    )
    scale2_runner = (
        root / "scripts/run_semstamp_certified_defense_v3_maximum_scale_v2.py"
    )
    v2_rows = (
        root
        / "experiments/semstamp_certified_defense_v2_wp3_detector_health1"
        / "P0_RENDERER_ROWS_REPLAY0.jsonl"
    )
    core1_contract = read_json(core1 / "CONTRACT.json")
    scale2_contract = read_json(scale2 / "CONTRACT.json")
    invalidation = read_json(
        core1 / "MAXIMUM_SCALE_ATTEMPT0_INVALIDATION.json"
    )

    correctness_names = (
        "CORRECTNESS_MACOS_REPLAY0.json",
        "CORRECTNESS_MACOS_REPLAY1.json",
        "CORRECTNESS_LINUX_REPLAY0.json",
        "CORRECTNESS_LINUX_REPLAY1.json",
    )
    correctness = {
        name: read_json(core1 / name) for name in correctness_names
    }
    scale_replays = [
        read_json(scale2 / f"MAXIMUM_SCALE_LINUX_REPLAY{index}.json")
        for index in (0, 1)
    ]

    input_hashes = core1_contract["inputs"]
    checks = {}
    checks["core1_protocol_bound"] = (
        sha256_file(core1 / "PROTOCOL.md")
        == input_hashes["protocol_sha256"]
    )
    checks["core_source_bound"] = (
        sha256_file(source) == input_hashes["core_source_sha256"]
    )
    checks["correctness_runner_bound"] = (
        sha256_file(correctness_runner)
        == input_hashes["correctness_runner_sha256"]
    )
    checks["base_scale_runner_bound"] = (
        sha256_file(base_scale_runner)
        == scale2_contract["inputs"]["base_maximum_runner_sha256"]
    )
    checks["scale2_runner_bound"] = (
        sha256_file(scale2_runner)
        == scale2_contract["inputs"]["runner_sha256"]
    )
    checks["scale2_protocol_bound"] = (
        sha256_file(scale2 / "PROTOCOL.md")
        == scale2_contract["inputs"]["protocol_sha256"]
    )
    checks["v2_rows_bound"] = (
        sha256_file(v2_rows)
        == input_hashes["v2_renderer_rows_sha256"]
        == scale2_contract["inputs"]["v2_renderer_rows_sha256"]
    )

    observations = {
        value["observation_stream_sha256"]
        for value in correctness.values()
    }
    checks["four_correctness_replays_pass"] = all(
        value["status"] == "pass"
        and all(value["gates"].values())
        for value in correctness.values()
    )
    checks["correctness_observation_sha_exact"] = (
        observations == {OBSERVATION_SHA256}
    )
    checks["correctness_counts_exact"] = all(
        value["counts"]["systematic"]["cases"] == 5_000
        and value["counts"]["random_legality"]["cases"] == 2_000
        and value["counts"]["random_legality"]["empty_fail_closed"] == 171
        and value["counts"]["fixed_n_reduction"]["cases"] == 1_000
        and value["counts"]["fixed_n_reduction"]["byte_exact_rows"] == 1_000
        and value["counts"]["mode5"]["variable_problems"] == 200
        and value["counts"]["mode5"]["target_solves"] == 569
        and value["counts"]["total_declared_problems"] == 8_200
        for value in correctness.values()
    )
    checks["fixed_n_stdout_exact"] = all(
        value["counts"]["fixed_n_reduction"]["v3_stdout_sha256"]
        == value["counts"]["fixed_n_reduction"]["v6_stdout_sha256"]
        for value in correctness.values()
    )
    checks["correctness_artifact_hashes_bound"] = all(
        sha256_file(core1 / name)
        == scale2_contract["inputs"][
            name.lower().replace(".json", "_sha256")
        ]
        for name in correctness_names
    )

    checks["core1_stop_exact"] = (
        invalidation["status"] == "invalidated_and_stopped"
        and invalidation["terminal"]
        == "V3_COMPILED_CORE1_MAXIMUM_SCALE_SELECTION_STOP"
        and invalidation["completed_topology_target_count"] == 7
        and invalidation["resume_forbidden"] is True
        and invalidation["maximum_scale_result_created"] is False
    )
    checks["core1_stop_score_blind"] = (
        invalidation["execution_attestation"] == {
            "certificate_computations": 0,
            "cuda_initializations": 0,
            "detector_score_computations": 0,
            "generation_calls": 0,
            "mode5_replays": 0,
            "model_loads": 0,
            "sidecar_dumps": 0,
            "topology_only_core_calls": 7,
        }
    )

    checks["two_scale_replays_pass"] = all(
        value["status"] == "pass"
        and all(value["gates"].values())
        for value in scale_replays
    )
    checks["scale_replay_indices_exact"] = [
        value["replay_index"] for value in scale_replays
    ] == [0, 1]
    checks["real_profiles_replay_exact"] = (
        [
            comparable_profile(item)
            for item in scale_replays[0]["real_target_profiles"]
        ]
        == [
            comparable_profile(item)
            for item in scale_replays[1]["real_target_profiles"]
        ]
    )
    real_profiles = scale_replays[0]["real_target_profiles"]
    fixed = next(
        item for item in real_profiles
        if item["target_edge_count"] == 173
    )
    checks["real_fixed_counts_exact"] = (
        fixed["result"]["unique_pair_scores"] == HISTORICAL_PAIRS
        and fixed["result"]["unique_span_checks"] == HISTORICAL_SPANS
    )
    checks["real_selector_stop_reconstructed"] = not any(
        HISTORICAL_PAIRS
        < item["result"]["unique_pair_scores"]
        <= PAIR_CAP
        for item in real_profiles
    )

    source_row = read_jsonl_row(v2_rows, 800)
    amplified_boundaries = [
        boundary * 6_500 // 5_554
        for boundary in source_row["original_boundaries"]
    ]
    amplified_sha = sha256_bytes(canonical_bytes(amplified_boundaries))
    checks["amplified_boundary_formula_exact"] = all(
        value["amplification"]["amplified_boundaries_sha256"]
        == amplified_sha
        and value["amplification"]["formula"]
        == "floor(6500 * boundary / 5554)"
        for value in scale_replays
    )
    topologies = [
        value["amplified_topology"] for value in scale_replays
    ]
    mode5 = [value["amplified_mode5"] for value in scale_replays]
    checks["amplified_topology_replay_exact"] = (
        comparable_profile(topologies[0])
        == comparable_profile(topologies[1])
    )
    checks["amplified_sidecar_replay_exact"] = (
        mode5[0]["sidecar_bytes"] == mode5[1]["sidecar_bytes"]
        and mode5[0]["sidecar_sha256"] == mode5[1]["sidecar_sha256"]
    )
    checks["amplified_mode5_replay_exact"] = (
        mode5[0]["stdout_sha256"] == mode5[1]["stdout_sha256"]
        and mode5[0]["result"] == mode5[1]["result"]
    )
    amplified_result = topologies[0]["result"]
    checks["amplified_scale_bounds_exact"] = (
        HISTORICAL_PAIRS
        < amplified_result["unique_pair_scores"]
        <= PAIR_CAP
        and amplified_result["unique_span_checks"] <= SPAN_CAP
    )
    checks["amplified_expected_counts_exact"] = (
        amplified_result["unique_pair_scores"] == 133_713_262
        and amplified_result["unique_span_checks"] == 1_052_797
    )
    checks["resource_gates_exact"] = all(
        value["maximum_observed_rss_bytes"] <= RSS_CAP
        and value["maximum_observed_elapsed_seconds"] <= TIME_CAP
        for value in scale_replays
    )
    checks["zero_model_score_certificate_calls"] = all(
        all(count == 0 for count in value["execution_attestation"].values())
        for value in scale_replays
    )

    public_values = [
        core1_contract,
        scale2_contract,
        invalidation,
        *correctness.values(),
        *scale_replays,
    ]
    for value in public_values:
        reject_raw_keys(value)
    checks["public_outputs_contain_no_raw_text"] = True

    source_text = source.read_text(encoding="utf-8")
    checks["source_pair_cap_exact"] = (
        "constexpr uint64_t kMaximumPairScores = 200000000;"
        in source_text
    )
    checks["source_has_distinct_edge_counts"] = (
        "uint32_t original_m;" in source_text
        and "uint32_t target_m;" in source_text
    )

    failed = sorted(key for key, value in checks.items() if not value)
    audit = {
        "bindings": {
            "core1_contract_sha256": sha256_file(
                core1 / "CONTRACT.json"
            ),
            "core1_invalidation_sha256": sha256_file(
                core1 / "MAXIMUM_SCALE_ATTEMPT0_INVALIDATION.json"
            ),
            "scale2_contract_sha256": sha256_file(
                scale2 / "CONTRACT.json"
            ),
            "scale2_replay0_sha256": sha256_file(
                scale2 / "MAXIMUM_SCALE_LINUX_REPLAY0.json"
            ),
            "scale2_replay1_sha256": sha256_file(
                scale2 / "MAXIMUM_SCALE_LINUX_REPLAY1.json"
            ),
        },
        "checks": checks,
        "counts": {
            "amplified_pairs": amplified_result["unique_pair_scores"],
            "amplified_spans": amplified_result["unique_span_checks"],
            "checks_passed": sum(checks.values()),
            "checks_total": len(checks),
            "correctness_declared_problems_per_replay": 8_200,
            "correctness_replays": 4,
            "mode5_target_solves_per_replay": 569,
        },
        "failed_checks": failed,
        "schema_version": 1,
        "stage": "semstamp_certified_defense_v3_compiled_core_audit",
        "status": "pass" if not failed else "fail",
        "terminal": (
            "V3_WP2_COMPILED_CORE_GO_FOR_THROUGHPUT_PREFLIGHT"
            if not failed
            else "V3_WP2_COMPILED_CORE_AUDIT_STOP"
        ),
    }
    atomic_write_json(args.output, audit)
    print(json.dumps({
        "checks_passed": audit["counts"]["checks_passed"],
        "checks_total": audit["counts"]["checks_total"],
        "status": audit["status"],
    }, sort_keys=True))
    if failed:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--core1-dir",
        default=(
            "experiments/"
            "semstamp_certified_defense_v3_wp2_compiled_core1"
        ),
    )
    parser.add_argument(
        "--scale2-dir",
        default=(
            "experiments/"
            "semstamp_certified_defense_v3_wp2_compiled_core_scale2"
        ),
    )
    parser.add_argument("--output", required=True)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
