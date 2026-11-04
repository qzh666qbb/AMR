#!/usr/bin/env python3
"""Audit frozen v4 results and explore alpha without modifying source artifacts."""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    "semicolon_all", "semicolon_pairs", "comma_all", "newline_all", "space_all"
)
NULL_COHORT = "independent_null_punctuation_stress"
BASELINE_COHORT = "confirmation_baseline_positive"
ABSTAIN_STATISTIC = -1e12


def reject_constant(value):
    raise ValueError(f"Non-standard JSON numeric value: {value}")


def read_json(path):
    return json.loads(path.read_text(), parse_constant=reject_constant)


def rows(path):
    return [
        json.loads(line, parse_constant=reject_constant)
        for line in path.read_text().splitlines() if line.strip()
    ]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratio(hits, total):
    return {"successes": hits, "trials": total, "rate": hits / total if total else None}


def json_output(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_experiment(path):
    return {
        "path": path,
        "manifest": read_json(path / "INPUT_MANIFEST.json"),
        "runtime": read_json(path / "RUNTIME_MANIFEST.json"),
        "result": read_json(path / "RESULT.json"),
        "calibrator": read_json(path / "results/CALIBRATION.json"),
        "calibration_inputs": rows(path / "inputs/calibration_rows.jsonl"),
        "calibration_rows": rows(path / "results/calibration_rows.jsonl"),
        "development_inputs": rows(path / "inputs/development_rows.jsonl"),
        "development_rows": rows(path / "results/development_rows.jsonl"),
    }


def bucket(label_definition, word_count):
    for item in label_definition:
        if word_count >= item["min_words"] and (
            item["max_words"] is None or word_count <= item["max_words"]
        ):
            return item["label"]
    raise ValueError(f"No word bucket for {word_count}")


def probability(calibrator, row):
    label = bucket(calibrator["word_bins"], row["word_count"])
    values = calibrator["bucket_statistics"].get(label, [])
    scope = "bucket"
    if len(values) < calibrator["min_bucket_size"]:
        values = calibrator["global_statistics"]
        scope = "global-fallback"
    if row["detection"]["status"] == "abstain":
        return {
            "p_value": 1.0, "bucket": label, "scope": "abstain",
            "null_count": len(values), "statistic": ABSTAIN_STATISTIC,
        }
    statistic = row["detection"]["statistic"]
    if not math.isfinite(statistic):
        raise ValueError("Non-finite detection statistic")
    return {
        "p_value": (1 + len(values) - bisect_left(values, statistic)) / (len(values) + 1),
        "bucket": label, "scope": scope, "null_count": len(values),
        "statistic": statistic,
    }


def assert_source_bindings(exp):
    for item in exp["manifest"]["outputs"].values():
        path = ROOT / item["path"]
        assert sha(path) == item["sha256"], path
        assert len(rows(path)) == item["rows"], path
    for name, expected in exp["result"]["artifacts"].items():
        assert sha(exp["path"] / name) == expected, name
    runtime = exp["runtime"]
    assert runtime["input_manifest_sha256"] == sha(exp["path"] / "INPUT_MANIFEST.json")
    provenance = exp["calibrator"]["v4_provenance"]
    for key, name in (
        ("calibration_rows_sha256", "results/calibration_rows.jsonl"),
        ("input_manifest_sha256", "INPUT_MANIFEST.json"),
        ("runtime_manifest_sha256", "RUNTIME_MANIFEST.json"),
    ):
        assert provenance[key] == sha(exp["path"] / name), key
    for field, artifact_field in (
        ("detector", "detector_fingerprint"),
        ("bank", "bank_fingerprint"),
        ("parser", "parser_fingerprint"),
        ("bank_sha256", "bank_sha256"),
    ):
        assert runtime["fingerprints"][field] == exp["calibrator"][artifact_field]
        assert exp["result"]["fingerprints"][field] == runtime["fingerprints"][field]
    for stage in ("calibration", "development"):
        source = exp[f"{stage}_inputs"]
        output = exp[f"{stage}_rows"]
        assert len(source) == len(output), stage
        identities = []
        for left, right in zip(source, output):
            field = "window_sha256" if stage == "calibration" else "text_sha256"
            assert left[field] == right[field], (stage, field)
            assert left["word_count"] == right["word_count"]
            assert left["cohort"] == right["cohort"]
            identity_field = "allocation_id" if stage == "calibration" else "row_id"
            assert left[identity_field] == right[identity_field]
            identities.append(left[identity_field])
            text = left["window"] if stage == "calibration" else left["text"]
            assert hashlib.sha256(text.encode()).hexdigest() == left[field]
        assert len(identities) == len(set(identities)), stage


def assert_calibration_distribution(exp):
    actual = [
        ABSTAIN_STATISTIC if row["detection"]["status"] == "abstain"
        else row["detection"]["statistic"]
        for row in exp["calibration_rows"]
    ]
    assert all(math.isfinite(value) for value in actual)
    calibrator = exp["calibrator"]
    assert sorted(actual) == calibrator["global_statistics"]
    assert len(actual) == calibrator["sample_count"]
    for definition in calibrator["word_bins"]:
        label = definition["label"]
        expected = sorted(
            value for row, value in zip(exp["calibration_rows"], actual)
            if bucket(calibrator["word_bins"], row["word_count"]) == label
        )
        assert expected == calibrator["bucket_statistics"][label]


def assert_selected_path_arithmetic(exp):
    configuration = exp["runtime"]["configuration"]
    all_rows = exp["calibration_rows"] + exp["development_rows"]
    for row in all_rows:
        detection = row["detection"]
        assert detection["selected_invalid_edges"] == 0
        if detection["status"] == "abstain":
            assert detection["statistic"] is None
            continue
        assert detection["status"] == "ok"
        units = detection["units"]
        assert len(units) == detection["unit_count"]
        for unit in units:
            penalty = 0.18 if unit["boundary_hops"] >= 2 else 0.0
            expected = unit["match"]["score"] - configuration["score_center"] - penalty
            assert math.isclose(expected, unit["utility"], rel_tol=1e-12, abs_tol=1e-12)
        raw = sum(unit["utility"] for unit in units)
        assert math.isclose(raw, detection["raw_utility"], rel_tol=1e-12, abs_tol=1e-12)
        expected = raw / math.sqrt(len(units)) if units else 0.0
        assert math.isclose(expected, detection["statistic"], rel_tol=1e-12, abs_tol=1e-12)


def audit_decisions(exp):
    mismatches = []
    saved_hits = p_replay_hits = independent_hits = 0
    for row in exp["development_rows"]:
        saved = row["calibration"]
        assert type(saved["is_watermarked"]) is bool
        assert math.isfinite(saved["alpha"]) and 0 < saved["alpha"] < 1
        computed = probability(exp["calibrator"], row)
        saved_hits += saved["is_watermarked"]
        replay = saved["p_value"] <= saved["alpha"] and row["detection"]["status"] != "abstain"
        independent = computed["p_value"] <= saved["alpha"] and row["detection"]["status"] != "abstain"
        p_replay_hits += replay
        independent_hits += independent
        differs = [
            key for key in ("bucket", "scope", "null_count", "statistic", "p_value")
            if computed[key] != saved[key]
        ]
        if differs or replay != saved["is_watermarked"] or independent != saved["is_watermarked"]:
            mismatches.append({"row_id": row["row_id"], "fields": differs})
    return {
        "rows": len(exp["development_rows"]),
        "saved_boolean_hits": saved_hits,
        "saved_p_value_replay_hits": p_replay_hits,
        "independent_empirical_p_hits": independent_hits,
        "mismatches": mismatches,
    }


def stats_at_alpha(calibrator, positive_rows, clean_rows, stress_rows, alpha):
    positive_flags = {
        (row["group_id"], row["variant_id"]):
        probability(calibrator, row)["p_value"] <= alpha
        for row in positive_rows
    }
    baseline_rows = [row for row in positive_rows if row["cohort"] == BASELINE_COHORT]
    detected = [row["group_id"] for row in baseline_rows
                if positive_flags[(row["group_id"], row["variant_id"])]]
    all_five = sum(all(positive_flags.get((group, variant), False)
                       for variant in VARIANTS) for group in detected)
    stress_hits = [row for row in stress_rows if probability(calibrator, row)["p_value"] <= alpha]
    all_one_hop_hits = [
        row for row in stress_hits
        if all(unit["boundary_hops"] == 1 for unit in row["detection"]["units"])
    ]
    clean_hits = sum(probability(calibrator, row)["p_value"] <= alpha for row in clean_rows)
    baseline = ratio(len(detected), len(baseline_rows))
    survival = ratio(all_five, len(detected))
    stress = ratio(len(stress_hits), len(stress_rows))
    gates = {
        "baseline_at_least_80pct": baseline["rate"] is not None and baseline["rate"] >= .8,
        "all_five_conditional_at_least_80pct": survival["rate"] is not None and survival["rate"] >= .8,
        "stress_row_fp_at_most_1pct": stress["rate"] is not None and stress["rate"] <= .01,
    }
    values = calibrator["global_statistics"]
    tail_count = math.floor(alpha * (len(values) + 1) - 1)
    threshold = values[len(values) - tail_count - 1] if 0 <= tail_count < len(values) else None
    return {
        "alpha": alpha,
        "global_score_threshold_exclusive": threshold,
        "baseline": baseline, "all_five_conditional": survival,
        "clean_null": ratio(clean_hits, len(clean_rows)),
        "stress_rows": stress,
        "stress_document_any_variant": ratio(
            len({row["group_id"] for row in stress_hits}),
            len({row["group_id"] for row in stress_rows}),
        ),
        "larger_hop_penalty_fp_floor_on_observed_assets": {
            "rows": len(all_one_hop_hits),
            "documents": len({row["group_id"] for row in all_one_hop_hits}),
            "conditions": "same graph, bank, parser, alpha, calibration identities; only increase p2/p3",
        },
        "gates_diagnostic_only": gates,
        "all_diagnostic_gates": all(gates.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "experiments/swan_elastic_v4_p2_audit1")
    args = parser.parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    p1 = load_experiment(ROOT / "experiments/swan_elastic_v4_development1")
    p2 = load_experiment(ROOT / "experiments/swan_elastic_v4_p2_formal")
    for exp in (p1, p2):
        assert_source_bindings(exp)
        assert_calibration_distribution(exp)
        assert_selected_path_arithmetic(exp)
    decisions = {exp["path"].name: audit_decisions(exp) for exp in (p1, p2)}
    assert all(not item["mismatches"] for item in decisions.values())
    stress_rows = [row for row in p2["development_rows"] if row["cohort"] == NULL_COHORT]
    assert len(stress_rows) == len(p2["development_rows"])
    saved = decisions[p2["path"].name]
    summary_fp = p2["result"]["summary"][NULL_COHORT]
    assert saved["independent_empirical_p_hits"] == summary_fp["successes"]
    assert saved["rows"] == summary_fp["trials"]
    calibration_ids = {row["allocation_id"] for row in p2["calibration_inputs"]}
    calibration_hashes = {row["window_sha256"] for row in p2["calibration_inputs"]}
    stress_inputs = p2["development_inputs"]
    source_hashes = {row["metadata"]["source_window_sha256"] for row in stress_inputs}
    grouped = defaultdict(list)
    for row in stress_rows:
        grouped[row["group_id"]].append(row)
    assert all(Counter(row["variant_id"] for row in group) == Counter(VARIANTS)
               for group in grouped.values())
    p1_ids = {row["allocation_id"] for row in p1["calibration_inputs"]}
    overlap = {
        "calibration_documents": len(calibration_ids),
        "stress_rows": len(stress_rows), "stress_source_documents": len(grouped),
        "calibration_stress_group_overlap": len(calibration_ids & set(grouped)),
        "calibration_stress_source_hash_overlap": len(calibration_hashes & source_hashes),
        "p1_calibration_p2_original_document_overlap": len(p1_ids & calibration_ids),
        "distinct_stress_text_sha256": len({row["text_sha256"] for row in stress_inputs}),
        "variants_per_document": 5,
    }
    false_positives = [row for row in stress_rows if row["calibration"]["is_watermarked"]]
    hits_per_group = Counter(row["group_id"] for row in false_positives)
    clean_map = {row["allocation_id"]: row for row in p2["calibration_rows"]}
    clean_positive_ids = {
        identity for identity, row in clean_map.items()
        if probability(p2["calibrator"], row)["p_value"] <= .01
    }
    per_variant = {
        variant: ratio(
            sum(row["variant_id"] == variant for row in false_positives),
            sum(row["variant_id"] == variant for row in stress_rows),
        ) for variant in VARIANTS
    }
    false_positive_info = {
        "row_metric": ratio(len(false_positives), len(stress_rows)),
        "document_any_variant_metric": ratio(len(hits_per_group), len(grouped)),
        "per_variant": per_variant,
        "false_positive_groups_by_number_of_positive_variants": {
            str(k): v for k, v in sorted(Counter(hits_per_group.values()).items())
        },
        "clean_self_calibration_positive_documents": len(clean_positive_ids),
        "false_positive_groups_already_positive_when_clean": len(set(hits_per_group) & clean_positive_ids),
        "false_positive_groups_created_by_boundary_variants": len(set(hits_per_group) - clean_positive_ids),
        "false_positive_rows_with_invalid_candidates": sum(
            row["detection"]["invalid_edge_count"] > 0 for row in false_positives),
        "selected_invalid_edges": sum(row["detection"]["selected_invalid_edges"]
                                      for row in false_positives),
        "unique_false_positive_texts": len({row["text_sha256"] for row in false_positives}),
        "non_finite_statistics": 0,
        "all_one_hop_false_positive_rows": sum(
            all(unit["boundary_hops"] == 1 for unit in row["detection"]["units"])
            for row in false_positives),
        "all_one_hop_false_positive_documents": len({
            row["group_id"] for row in false_positives
            if all(unit["boundary_hops"] == 1 for unit in row["detection"]["units"])
        }),
    }
    fields = (
        "row_id", "group_id", "variant_id", "text_sha256", "word_count",
        "statistic", "p_value", "clean_statistic", "clean_self_p_value",
        "score_change_from_clean", "invalid_candidates", "unit_count",
        "one_hop_units", "two_hop_units", "three_plus_hop_units",
    )
    with (out / "FALSE_POSITIVES.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in false_positives:
            det = row["detection"]
            clean = clean_map[row["group_id"]]
            hops = [unit["boundary_hops"] for unit in det["units"]]
            writer.writerow({
                "row_id": row["row_id"], "group_id": row["group_id"],
                "variant_id": row["variant_id"], "text_sha256": row["text_sha256"],
                "word_count": row["word_count"], "statistic": det["statistic"],
                "p_value": row["calibration"]["p_value"],
                "clean_statistic": clean["detection"]["statistic"],
                "clean_self_p_value": probability(p2["calibrator"], clean)["p_value"],
                "score_change_from_clean": det["statistic"] - clean["detection"]["statistic"],
                "invalid_candidates": det["invalid_edge_count"],
                "unit_count": len(hops), "one_hop_units": hops.count(1),
                "two_hop_units": hops.count(2),
                "three_plus_hop_units": sum(hop >= 3 for hop in hops),
            })
    grids = {}
    requested = [.001, .002, .003, .004, .005, .006, .007, .0075, .008, .009, .01]
    for label, exp in (("p1b_original_calibration", p1), ("p2_overlapping_calibration", p2)):
        calibrator = exp["calibrator"]
        args_for_stats = (calibrator, p1["development_rows"], p2["calibration_rows"], stress_rows)
        grid = [stats_at_alpha(*args_for_stats, alpha) for alpha in requested]
        breakpoints = sorted({probability(calibrator, row)["p_value"]
                              for row in p1["development_rows"] + p2["calibration_rows"] + stress_rows
                              if probability(calibrator, row)["p_value"] <= .01} | {0.0, .01})
        frontier = [stats_at_alpha(*args_for_stats, alpha) for alpha in breakpoints]
        grids[label] = {
            "calibration_sha256": sha(exp["path"] / "results/CALIBRATION.json"),
            "scope": "observed assets, exploratory only; not independent confirmation",
            "requested_grid": grid, "all_decision_breakpoints_upto_0_01": frontier,
            "feasible_alpha_count_upto_0_01": sum(item["all_diagnostic_gates"] for item in frontier),
            "at_alpha_0_01_null_exceeds_1pct": grid[-1]["stress_rows"]["rate"] > .01,
            "higher_alpha_cannot_reduce_false_positives": True,
        }
    default_p1 = grids["p1b_original_calibration"]["requested_grid"][-1]
    assert default_p1["baseline"] == p1["result"]["summary"]["baseline_detection"]
    assert default_p1["all_five_conditional"] == p1["result"]["summary"]["survival_all_five_variants"]
    result = {
        "version": "swan-elastic-v4-p2-audit:v1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete", "decision": "formal_claim_not_supported",
        "observation": "The original 51/5000 is reproduced; it is not a counting correction.",
        "verification": {
            "manifest_and_result_hash_bindings": True,
            "calibration_distribution_rebuilt_exactly": True,
            "selected_path_arithmetic_verified": True,
            "full_denominators_and_input_identities_verified": True,
            "finite_statistics_only": True,
            "p_values_and_booleans_reproduced": True,
        },
        "decision_replays": decisions,
        "source_independence": overlap,
        "false_positives": false_positive_info,
        "original_result_sha256": sha(p2["path"] / "RESULT.json"),
        "original_report_failure": read_json(p2["path"] / "FAILURE.json"),
        "limitations": [
            "Only selected-path arithmetic was replayed; full candidate DP and parser were not rerun.",
            "The 5000 variants share 1000 originals used in P2 calibration.",
            "All alpha comparisons use observed assets and cannot establish formal FPR or TPR.",
            "P2 contains no new positives; P1b's 30 positives are development assets.",
            "Changing hop penalties requires rerunning DP and recalibrating.",
            "One-hop false-positive paths provide a floor only when graph, scorer, parser, "
            "alpha and calibration identities remain fixed and p2/p3 only increase.",
        ],
        "artifacts": {"false_positives_csv_sha256": sha(out / "FALSE_POSITIVES.csv")},
    }
    json_output(out / "ALPHA_EXPLORATION.json", grids)
    result["artifacts"]["alpha_exploration_sha256"] = sha(out / "ALPHA_EXPLORATION.json")
    json_output(out / "AUDIT_RESULT.json", result)
    print(json.dumps({
        "original_recount": saved,
        "source_independence": overlap,
        "false_positives": false_positive_info,
        "alpha_feasible_counts": {
            label: item["feasible_alpha_count_upto_0_01"] for label, item in grids.items()
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
