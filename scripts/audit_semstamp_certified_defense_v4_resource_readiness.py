"""Independently re-account every frozen row and the v4 budget lower bound."""

from bisect import bisect_left
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import tempfile
import os


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/semstamp_certified_defense_v4_full_radius_profile1"
OLD = ROOT / "experiments/semstamp_certified_defense_v3_wp3_variable_n_profile1"
OLD_ROWS_SHA = "c3a941ce4d8b205090452fc31c290c0b186cf56e8b89ff10f49162dc60c08fb5"


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    data = (json.dumps(data, sort_keys=True, ensure_ascii=True,
                       allow_nan=False, separators=(",", ":")) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        os.chmod(handle.name, 0o600)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = handle.name
    os.replace(temporary, path)


def rate(profiles, atoms, field):
    anchors = [p["atom_count"] for p in profiles]
    pos = bisect_left(anchors, atoms)
    if pos == 0:
        indices = (0, 1)
    elif pos == len(anchors):
        indices = (len(anchors)-2, len(anchors)-1)
    elif anchors[pos] != atoms:
        indices = (pos-1, pos)
    else:
        indices = tuple(i for i in (pos-1, pos, pos+1)
                        if 0 <= i < len(anchors))
    return max(profiles[i][field] for i in indices)


def close(left, right):
    return math.isclose(left, right, rel_tol=1e-14, abs_tol=1e-8)


def audit():
    old_rows = rows(OLD / "PROFILE_ROWS.jsonl")
    frozen = rows(ROOT / "experiments/semstamp_certified_defense_v2_wp3_detector_health1/P0_RENDERER_ROWS_REPLAY0.jsonl")
    plan = rows(ROOT / "experiments/certified_resegmentation_wp2_v17_b0_detector_health1/B0_PLAN.jsonl")
    old_result = read(OLD / "PROFILE_RESULT.json")
    profiles_path = ROOT / "experiments/certified_resegmentation_wp2_v6_feature1/semstamp_workload.json"
    profiles = sorted(read(profiles_path)["inputs"]["provider_profiles"],
                      key=lambda p: p["atom_count"])
    contract = read(OUT / "CONTRACT.json")
    local = read(OUT / "LOCAL_CORRECTNESS.json")
    linux = read(OUT / "LINUX_CORRECTNESS.json")
    local_inputs = read(OUT / "LOCAL_INPUT_READINESS.json")
    remote = read(OUT / "REMOTE_INPUT_READINESS.json")
    checks = {
        "historical_rows_sha_exact": sha(OLD / "PROFILE_ROWS.jsonl") == OLD_ROWS_SHA,
        "full_frozen_denominator": len(old_rows) == len(frozen) == len(plan) == 3000,
        "cohorts_1000_each": Counter(r["cohort"] for r in old_rows) == {
            "calibration_null": 1000, "independent_test_null": 1000, "positive": 1000,
        },
        "refinement_rows_42": sum(r["refinement_count"] > 0 for r in old_rows) == 42,
        "zero_byte_gaps_59": sum(r["zero_byte_gap_count"] for r in old_rows) == 59,
        "plan_indices_exact": [r["plan_index"] for r in old_rows] == list(range(3000)),
        "v4_original_radius_contract": (
            contract["radius_rule"] == "requested_b_without_original_edge_cap"
            and contract["operation_cost"] == "max(insertions,deletions)"
            and contract["radii"] == [0, 1, 2, 3]),
        "local_linux_oracles_pass": local["status"] == linux["status"] == "pass",
        "local_linux_oracle_stream_exact": (
            local["observation_sha256"] == linux["observation_sha256"]),
        "local_linux_legacy_compatibility": (
            local["legacy_outputs_byte_exact"] and linux["legacy_outputs_byte_exact"]),
        "core_change_only_admits_uncapped_radius": (
            local["core_change_only_removes_b_gt_m0_rejection"]
            and linux["core_change_only_removes_b_gt_m0_rejection"]),
        "no_successor_authorization": (
            contract["certificate_authorized"] is False
            and contract["successor_execution_authorized"] is False),
        "contract_protocol_sha": contract["protocol_sha256"] == sha(OUT / "PROTOCOL.md"),
        "contract_runner_sha": contract["runner_sha256"] == sha(
            ROOT / "scripts/run_semstamp_certified_defense_v4_full_radius_profile.py"),
        "contract_manifest_sha": contract["manifest_sha256"] == sha(
            OUT / "INPUT_MANIFEST.sha256"),
        "frozen_canonical_profile_batch": all(
            p.get("batch_size") in (None, 64) for p in profiles) and len(profiles) == 12,
    }
    per_row_checks = Counter()
    corrected_plan = []
    components = {"feature_seconds": 0.0, "union_sidecar_rate_seconds": 0.0,
                  "pair_rate_core_seconds": 0.0}
    integer_counts = Counter()
    affected = Counter()
    old_target_layers = 0
    for index, (r, f, p) in enumerate(zip(old_rows, frozen, plan)):
        m0 = len(f["original_boundaries"])
        n = f["atom_count"]
        if r["original_edge_count"] != m0 or r["atom_count"] != n \
                or r["raw_text_sha256"] != f["raw_text_sha256"] \
                or r["raw_text_sha256"] != p["raw_text_sha256"] \
                or r["topology_sha256"] != f["topology_sha256"]:
            per_row_checks["frozen_structure_mismatches"] += 1
        distinct = []
        revised = {}
        for b in range(4):
            historical = r["radii"][str(b)]
            k = min(b, m0)
            targets = list(range(max(1, m0-b), min(n-1, m0+b)+1))
            revised[str(b)] = {"requested_radius": b, "effective_radius": b,
                               "target_edge_counts": targets}
            if b > m0:
                affected[str(b)] += 1
            if historical["effective_radius"] != k \
                    or historical["requested_radius"] != b \
                    or historical["target_edge_counts"] != list(range(
                        max(1, m0-k), min(n-1, m0+k)+1)):
                per_row_checks["historical_radius_mismatches"] += 1
            if b != k:
                continue
            distinct.append(historical)
            if [layer["target_edge_count"] for layer in historical["layers"]] \
                    != historical["target_edge_counts"]:
                per_row_checks["missing_target_layers"] += 1
            for layer in historical["layers"]:
                m = layer["target_edge_count"]
                expected_hist = {}
                for a in range(4):
                    d = m0-m+a
                    value = (math.comb(n-1-m0, a) * math.comb(m0, m-a)
                             if a <= k and 0 <= d <= k and a <= n-1-m0
                             and 0 <= m-a <= m0 else 0)
                    expected_hist[str(a)] = value
                observed = layer["all_allowed"]
                if observed["insertion_histogram"] != expected_hist \
                        or observed["threat_set_size"] != sum(expected_hist.values()):
                    per_row_checks["all_allowed_binomial_mismatches"] += 1
                native = layer["native_pruned"]
                if any(native["insertion_histogram"][a] > expected_hist[a]
                       for a in expected_hist):
                    per_row_checks["native_not_subset"] += 1
                if observed["unique_pair_scores"] > 200_000_000 \
                        or native["unique_pair_scores"] > 200_000_000:
                    per_row_checks["historical_pair_cap_failures"] += 1
            old_target_layers += len(historical["layers"])
        unions = sum(radius["all_allowed_union_span_count"] for radius in distinct)
        pairs = sum(layer["native_pruned"]["unique_pair_scores"]
                    for radius in distinct for layer in radius["layers"])
        feature_tokens = r["resource_counts"]["deduplicated_legal_input_tokens"]
        calculated = {
            "feature_seconds": feature_tokens * rate(
                profiles, n, "feature_seconds_per_input_token"),
            "union_sidecar_rate_seconds": unions * rate(
                profiles, n, "sidecar_seconds_per_span"),
            "pair_rate_core_seconds": pairs * rate(
                profiles, n, "core_seconds_per_pair"),
        }
        if pairs != r["resource_counts"]["sum_distinct_layer_pruned_pairs"] \
                or unions != r["resource_counts"]["sum_distinct_radius_union_spans"]:
            per_row_checks["row_integer_count_mismatches"] += 1
        for key, value in calculated.items():
            components[key] += value
            if not close(value, r["projection"][key]):
                per_row_checks["row_rate_projection_mismatches"] += 1
        integer_counts.update({
            "deduplicated_legal_input_tokens": feature_tokens,
            "distinct_radius_union_spans": unions, "native_pruned_pairs": pairs,
        })
        corrected_plan.append({
            "plan_index": index, "cohort": r["cohort"],
            "raw_text_sha256": r["raw_text_sha256"], "atom_count": n,
            "original_edge_count": m0, "topology_sha256": r["topology_sha256"],
            "radii": revised, "score_access_count": 0,
            "profile_executed": False,
        })
    checks["all_3000_row_verifications"] = not per_row_checks
    checks["historical_46501_layers_reaccounted"] = old_target_layers == 46501
    checks["historical_component_totals_recomputed"] = all(
        close(value, old_result["runtime"][key]) for key, value in components.items())
    checks["corrected_affected_denominators"] = dict(affected) == {"2": 28, "3": 155}
    corrected_layers = sum(
        len(radius["target_edge_counts"]) for r in corrected_plan for radius in r["radii"].values())
    checks["corrected_plan_all_47332_target_layers"] = corrected_layers == 47332
    model_load = max(p["model_load_seconds"] for p in profiles)
    lower_seconds = model_load + 2 * sum(components.values())
    lower_hours = lower_seconds / 3600
    budget_stop = lower_hours > contract["gpu_hour_cap"]
    local_ready = local_inputs["status"] == "pass"
    remote_ready = remote["frozen_full_profile_inputs_available"]
    checks["budget_lower_bound_already_exceeds_gate"] = budget_stop
    checks["resource_profile_not_started"] = (
        not (OUT / "PROFILE_ROWS.jsonl").exists()
        and remote["full_profile_started"] is False)
    manifest = {}
    for line in (OUT / "INPUT_MANIFEST.sha256").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        manifest[relative] = digest
    checks["manifest_contract_entries_exact"] = manifest == contract["inputs"]
    available = [
        (name, expected) for name, expected in manifest.items()
        if (ROOT / name).is_file()
        and name != "build/semstamp_v4/semstamp_certified_defense_v4_full_radius_core"
    ]
    checks["all_available_inputs_sha_exact"] = all(
        sha(ROOT / name) == expected for name, expected in available)
    with (OUT / "CORRECTED_RADIUS_PLAN.jsonl").open("wb") as handle:
        for row in corrected_plan:
            handle.write((json.dumps(row, sort_keys=True, separators=(",", ":"))
                          + "\n").encode("ascii"))
    result = {
        "schema": "semstamp-v4-resource-readiness/v1",
        "audit_status": "pass" if all(checks.values()) else "fail",
        "terminal": "V4_FULL_RADIUS_READINESS_STOP",
        "decision": "stop",
        "stop_reasons": [
            "FROZEN_RATE_TWO_REPLAY_LOWER_BOUND_EXCEEDS_72_HOURS",
            "FROZEN_SEALED_INPUTS_AND_TOKENIZER_UNAVAILABLE",
        ],
        "checks": checks, "row_verification_failures": dict(per_row_checks),
        "coverage": {
            "rows_verified": len(old_rows), "cohorts": dict(Counter(r["cohort"] for r in old_rows)),
            "affected_requested_radius_2_rows": affected["2"],
            "affected_requested_radius_3_rows": affected["3"],
            "historical_distinct_target_layers": old_target_layers,
            "corrected_target_layers_planned": corrected_layers,
            "additional_target_layers_planned": corrected_layers-old_target_layers,
            "new_native_pruned_layers_measured": 0,
        },
        "budget": {
            "accounting_type": "frozen_protocol_projection_lower_bound",
            "inherited_model_load_seconds": model_load,
            "old_components_independently_reaccounted": components,
            "old_integer_counts_reaccounted": dict(integer_counts),
            "two_replay_lower_bound_seconds": lower_seconds,
            "two_replay_lower_bound_hours": lower_hours,
            "cap_hours": contract["gpu_hour_cap"],
            "lower_bound_excess_hours": lower_hours-contract["gpu_hour_cap"],
            "lower_bound_excess_percent": 100*(lower_hours/contract["gpu_hour_cap"]-1),
            "new_full_projection_hours": None,
            "new_full_measured_wall_seconds": None,
            "includes_measured_topology_time": False,
            "bound_depends_on_frozen_rates_and_renderer": True,
        },
        "input_readiness": {"local": local_ready, "current_remote": remote_ready},
        "full_profile_executed": False, "certificate_authorized": False,
        "successor_execution_authorized": False,
        "detector_computations": 0, "certificate_computations": 0,
        "formal_detection_results": "NOT_COMPUTED",
        "historical_v3_terminal_unchanged": old_result["terminal"],
        "verification_limit": (
            "All 3000 historical ledgers and rates were reaccounted. "
            "Native text/tokenizer computation was not independently replayed; "
            "new full-radius workload and resource peaks were not measured."),
        "sources": {
            "old_rows_sha256": sha(OLD / "PROFILE_ROWS.jsonl"),
            "frozen_renderer_rows_sha256": sha(ROOT / "experiments/semstamp_certified_defense_v2_wp3_detector_health1/P0_RENDERER_ROWS_REPLAY0.jsonl"),
            "profiles_sha256": sha(profiles_path),
            "contract_sha256": sha(OUT / "CONTRACT.json"),
            "corrected_plan_sha256": sha(OUT / "CORRECTED_RADIUS_PLAN.jsonl"),
            "local_correctness_sha256": sha(OUT / "LOCAL_CORRECTNESS.json"),
            "linux_correctness_sha256": sha(OUT / "LINUX_CORRECTNESS.json"),
            "auditor_sha256": sha(Path(__file__)),
        },
    }
    write(OUT / "RESOURCE_AUDIT.json", result)
    print(json.dumps({
        "audit_status": result["audit_status"], "terminal": result["terminal"],
        "checks_passed": sum(checks.values()), "checks_total": len(checks),
        "failed_checks": [k for k, v in checks.items() if not v],
        "row_verification_failures": dict(per_row_checks),
        "two_replay_lower_bound_hours": lower_hours,
        "corrected_target_layers": corrected_layers,
    }, sort_keys=True))
    if result["audit_status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    audit()
