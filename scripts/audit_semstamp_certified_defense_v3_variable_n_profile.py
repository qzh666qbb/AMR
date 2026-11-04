"""Independently audit the v3 variable-N score-blind profile."""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


STAGE = "semstamp_certified_defense_v3_wp3_variable_n_profile1"
PAIR_CAP = 200_000_000
SPAN_CAP = 2_000_000
RSS_CAP_BYTES = 16 * 1024**3
GPU_HOUR_CAP = 72.0
TIMING_KEYS = {
    "topology_seconds",
    "pruned_core_seconds",
    "union_sidecar_seconds",
    "measured_topology_seconds",
    "measured_union_sidecar_seconds",
    "measured_pruned_core_seconds",
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


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def strip_measurements(value):
    if isinstance(value, dict):
        return {
            key: strip_measurements(child)
            for key, child in value.items()
            if key not in TIMING_KEYS
        }
    if isinstance(value, list):
        return [strip_measurements(child) for child in value]
    return value


def deterministic_rows_sha256(rows):
    return hashlib.sha256(canonical_bytes(
        [strip_measurements(row) for row in rows]
    )).hexdigest()


def distinct_radius_rows(rows):
    return [
        radius
        for row in rows
        for requested, radius in row["radii"].items()
        if int(requested) == radius["effective_radius"]
    ]


def audit_experiment(experiment):
    experiment = Path(experiment)
    contract = read_json(experiment / "CONTRACT.json")
    rows = read_jsonl(experiment / "PROFILE_ROWS.jsonl")
    timings = read_jsonl(experiment / "sealed/PROFILE_TIMINGS.jsonl")
    result = read_json(experiment / "PROFILE_RESULT.json")
    profile_path = (
        experiment.parents[0].parents[0]
        / "certified_resegmentation_wp2_v6_feature1/semstamp_workload.json"
    )
    if not profile_path.is_file():
        root = experiment
        while root.parent != root and not (root / "experiments").is_dir():
            root = root.parent
        profile_path = (
            root
            / "experiments/certified_resegmentation_wp2_v6_feature1/"
            "semstamp_workload.json"
        )
    profiles = read_json(profile_path)["inputs"]["provider_profiles"]
    radius_rows = distinct_radius_rows(rows)
    components = {
        key: sum(row["projection"][key] for row in rows)
        for key in (
            "feature_seconds",
            "union_sidecar_rate_seconds",
            "pair_rate_core_seconds",
            "measured_topology_seconds",
            "measured_union_sidecar_seconds",
            "measured_pruned_core_seconds",
        )
    }
    model_load = max(row["model_load_seconds"] for row in profiles)
    formula_a = model_load + 2 * (
        components["feature_seconds"]
        + max(
            components["union_sidecar_rate_seconds"],
            components["measured_union_sidecar_seconds"],
        )
        + components["measured_pruned_core_seconds"]
        + components["measured_topology_seconds"]
    )
    formula_b = model_load + 2 * (
        components["feature_seconds"]
        + components["union_sidecar_rate_seconds"]
        + components["pair_rate_core_seconds"]
        + components["measured_topology_seconds"]
    )
    upper = max(formula_a, formula_b)
    full = len(rows) == 3000
    expected_terminal = (
        "V3_VARIABLE_N_PROFILE_GO_FOR_NEW_WP4"
        if full and upper / 3600 <= GPU_HOUR_CAP
        else "V3_VARIABLE_N_PROFILE_READINESS_STOP"
        if full
        else "V3_VARIABLE_N_PROFILE_SMOKE_PASS"
    )
    checks = {
        "contract_frozen_authorized_score_blind": (
            contract["stage"] == STAGE
            and contract["status"] == "frozen_and_authorized"
            and contract["score_blind"] is True
            and contract["certificate_authorized"] is False
        ),
        "row_timing_counts_exact": (
            len(rows) == len(timings) == len(result["selected_indices"])
        ),
        "plan_indices_exact": (
            [row["plan_index"] for row in rows]
            == result["selected_indices"]
            == [row["plan_index"] for row in timings]
        ),
        "all_layer_pair_caps": all(
            row["resource_counts"]["maximum_layer_pair_count"] <= PAIR_CAP
            for row in rows
        ),
        "all_radius_union_caps": all(
            radius["all_allowed_union_span_count"] <= SPAN_CAP
            for radius in radius_rows
        ),
        "rss_cap_exact": (
            max(row["worker_peak_rss_bytes"] for row in timings)
            <= RSS_CAP_BYTES
            and max(row["child_peak_rss_bytes"] for row in timings)
            <= RSS_CAP_BYTES
        ),
        "score_blind_rows": all(
            row["score_access_count"] == 0
            and row["contains_raw_text"] is False
            for row in rows
        ),
        "feature_sha_collision_buckets_zero": all(
            row["resource_counts"]["sha256_collision_bucket_count"] == 0
            for row in rows
        ),
        "component_sums_exact": all(
            math.isclose(
                result["runtime"][key],
                value,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for key, value in components.items()
        ),
        "formula_a_exact": math.isclose(
            result["runtime"]["two_replay_measured_core_formula_seconds"],
            formula_a,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "formula_b_exact": math.isclose(
            result["runtime"]["two_replay_pair_rate_formula_seconds"],
            formula_b,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "upper_bound_exact": math.isclose(
            result["runtime"][
                "conservative_two_replay_upper_bound_seconds"
            ],
            upper,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "terminal_recomputed": result["terminal"] == expected_terminal,
    }
    if full:
        checks.update({
            "full_cohorts_exact": Counter(row["cohort"] for row in rows)
            == {
                "calibration_null": 1000,
                "independent_test_null": 1000,
                "positive": 1000,
            },
            "all_42_refinement_rows_retained": sum(
                row["refinement_count"] > 0 for row in rows
            ) == 42,
            "all_59_zero_byte_gaps_retained": sum(
                row["zero_byte_gap_count"] for row in rows
            ) == 59,
            "gpu_hour_gate_recomputed": (
                result["gates"][
                    "two_replay_upper_bound_le_72_gpu_hours"
                ]
                == (upper / 3600 <= GPU_HOUR_CAP)
            ),
        })
    return {
        "schema_version": 1,
        "stage": STAGE,
        "status": "pass" if all(checks.values()) else "fail",
        "experiment": str(experiment),
        "rows_sha256": sha256_file(experiment / "PROFILE_ROWS.jsonl"),
        "deterministic_rows_sha256": deterministic_rows_sha256(rows),
        "recomputed": {
            **components,
            "model_load_seconds": model_load,
            "formula_a_seconds": formula_a,
            "formula_b_seconds": formula_b,
            "upper_bound_seconds": upper,
            "upper_bound_gpu_hours": upper / 3600,
            "terminal": expected_terminal,
        },
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--replay-experiment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = audit_experiment(args.experiment)
    if args.replay_experiment is not None:
        replay = audit_experiment(args.replay_experiment)
        audit["replay"] = {
            "experiment": replay["experiment"],
            "audit_status": replay["status"],
            "deterministic_rows_sha256": replay[
                "deterministic_rows_sha256"
            ],
            "deterministic_rows_exact": (
                audit["deterministic_rows_sha256"]
                == replay["deterministic_rows_sha256"]
            ),
        }
        if (
            replay["status"] != "pass"
            or not audit["replay"]["deterministic_rows_exact"]
        ):
            audit["status"] = "fail"
    atomic_write_json(args.output, audit)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
