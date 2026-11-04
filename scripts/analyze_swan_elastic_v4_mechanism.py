#!/usr/bin/env python3
"""Describe cached selected paths; never rerun or replace historical detection."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter
import csv
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audit_swan_elastic_v4_p2 import (
    BASELINE_COHORT, VARIANTS, assert_source_bindings, json_output,
    load_experiment, read_json, sha,
)
from src.core.bank_scorer import template_features

OUT = ROOT / "experiments/swan_elastic_v4_mechanism1"
AUDIT = ROOT / "experiments/swan_elastic_v4_p2_audit1"
BANK = ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_100.json"
COMPONENTS = ("score", "concept_score", "relation_score", "path_score")


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        return None
    return {
        "min": ordered[0], "median": statistics.median(ordered),
        "mean": statistics.mean(ordered), "max": ordered[-1],
    }


def auc(positives, negatives):
    """Descriptive empirical AUROC; half credit for ties, no significance test."""
    ordered = sorted(negatives)
    wins = sum(
        bisect_left(ordered, value)
        + .5 * (bisect_right(ordered, value) - bisect_left(ordered, value))
        for value in positives
    )
    return wins / (len(positives) * len(negatives))


def template_registry():
    bank = read_json(BANK)
    registry, groups = [], {}
    for index, row in enumerate(bank):
        features = template_features(row["normalized_template_amr"])
        signature = tuple(tuple(sorted(counter.items())) for counter in (
            features.concepts, features.relations, features.paths,
        ))
        groups.setdefault(signature, []).append(index)
        registry.append({
            "template_index": index,
            "concept_count": sum(features.concepts.values()),
            "relation_count": sum(features.relations.values()),
            "path_count": sum(features.paths.values()),
            "path_features": [
                {"feature": list(key), "count": count}
                for key, count in sorted(features.paths.items())
            ],
        })
    for indices in groups.values():
        for index in indices:
            registry[index]["signature_equivalent_template_indices"] = indices
    return registry, [indices for indices in groups.values() if len(indices) > 1]


def feature_row(row, registry, stage):
    detection = row["detection"]
    assert detection["status"] == "ok"
    units = detection["units"]
    assert len(units) == detection["unit_count"] and units
    empty_full = 0
    for unit in units:
        match = unit["match"]
        assert 0 <= match["template_index"] < len(registry)
        assert math.isclose(
            match["score"],
            .25 * match["concept_score"] + .5 * match["relation_score"]
            + .25 * match["path_score"], rel_tol=1e-12, abs_tol=1e-12,
        )
        if registry[match["template_index"]]["path_count"] == 0:
            assert match["path_score"] in (0.0, 1.0)
            empty_full += match["path_score"] == 1.0
    templates = Counter(unit["match"]["template_index"] for unit in units)
    one_hop = sum(unit["boundary_hops"] == 1 for unit in units)
    full_path = sum(unit["match"]["path_score"] == 1.0 for unit in units)
    bonus = .25 * empty_full / math.sqrt(len(units))
    return {
        "row_id": row.get("row_id", row["allocation_id"] if "allocation_id" in row else ""),
        "group_id": row.get("group_id", row.get("allocation_id")),
        "stage": stage, "cohort": row["cohort"],
        "variant_id": row.get("variant_id", "clean"),
        "text_sha256": row.get("text_sha256", row.get("window_sha256")),
        "word_count": row["word_count"], "unit_count": len(units),
        "candidate_edges": row["candidate_edges"],
        "total_boundary_hops": sum(unit["boundary_hops"] for unit in units),
        "one_hop_count": one_hop, "one_hop_fraction": one_hop / len(units),
        "all_one_hop": one_hop == len(units),
        "statistic": detection["statistic"],
        **{
            f"mean_{key}": statistics.mean(unit["match"][key] for unit in units)
            for key in COMPONENTS
        },
        "full_path_count": full_path, "full_path_fraction": full_path / len(units),
        "empty_template_full_path_count": empty_full,
        "empty_template_full_path_fraction": empty_full / len(units),
        "nonempty_template_full_path_count": full_path - empty_full,
        "empty_path_bonus_statistic_reference": bonus,
        "fixed_path_without_empty_bonus_reference": detection["statistic"] - bonus,
        "green_fraction": sum(unit["match"]["score"] >= .75 for unit in units) / len(units),
        "unique_templates": len(templates),
        "max_template_share": max(templates.values()) / len(units),
        "template_87_or_95_fraction": (templates[87] + templates[95]) / len(units),
    }


def describe(source_rows, feature_rows, registry):
    assert source_rows and len(source_rows) == len(feature_rows)
    units = [unit for row in source_rows for unit in row["detection"]["units"]]
    templates = Counter(unit["match"]["template_index"] for unit in units)
    empty_full = sum(row["empty_template_full_path_count"] for row in feature_rows)
    full_path = sum(row["full_path_count"] for row in feature_rows)
    return {
        "documents_or_variant_rows": len(source_rows), "selected_units": len(units),
        "all_one_hop_rows": sum(row["all_one_hop"] for row in feature_rows),
        "hops": dict(sorted(Counter(unit["boundary_hops"] for unit in units).items())),
        "unit_count_histogram": dict(sorted(Counter(row["unit_count"] for row in feature_rows).items())),
        "unit_weighted_component_means": {
            key: statistics.mean(unit["match"][key] for unit in units) for key in COMPONENTS
        },
        "full_path_score_units": full_path,
        "empty_template_full_path_score_units": empty_full,
        "empty_template_full_path_share_of_full_path_scores": empty_full / full_path if full_path else None,
        "rows_with_empty_template_full_path": sum(
            row["empty_template_full_path_count"] > 0 for row in feature_rows),
        "top_templates": [
            {"template_index": index, "units": count,
             "share": count / len(units), "reference_path_count": registry[index]["path_count"]}
            for index, count in templates.most_common(15)
        ],
        "document_feature_distributions": {
            key: distribution([row[key] for row in feature_rows])
            for key in feature_rows[0] if isinstance(feature_rows[0][key], (int, float))
            and not isinstance(feature_rows[0][key], bool)
        },
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for relative, expected in read_json(AUDIT / "AUDIT_MANIFEST.json")["files"].items():
        assert sha(ROOT / relative) == expected, relative
    p1 = load_experiment(ROOT / "experiments/swan_elastic_v4_development1")
    p2 = load_experiment(ROOT / "experiments/swan_elastic_v4_p2_formal")
    for exp in (p1, p2):
        assert_source_bindings(exp)
        assert sha(BANK) == exp["runtime"]["fingerprints"]["bank_sha256"]
    registry, equivalents = template_registry()
    baseline = [row for row in p1["development_rows"] if row["cohort"] == BASELINE_COHORT]
    variants = [row for row in p1["development_rows"] if row["variant_id"] in VARIANTS]
    stress = p2["development_rows"]
    false_positives = [row for row in stress if row["calibration"]["is_watermarked"]]
    fp_ids = {row["group_id"] for row in false_positives}
    clean = p2["calibration_rows"]
    groups = {
        "positive_baseline": baseline, "positive_five_variants": variants,
        **{
            f"positive_variant_{variant}": [
                row for row in p1["development_rows"] if row["variant_id"] == variant]
            for variant in ("boundary_only_candidate_0",) + VARIANTS
        },
        "clean_null": clean, "stress_null_variant_rows": stress,
        "clean_originals_of_fp_groups": [row for row in clean if row["allocation_id"] in fp_ids],
        "false_positive_variant_rows": false_positives,
        "false_positive_all_one_hop_rows": [
            row for row in false_positives if all(
                unit["boundary_hops"] == 1 for unit in row["detection"]["units"])],
    }
    positive_features = [
        feature_row(row, registry, "p1b_positive") for row in p1["development_rows"]]
    baseline_features = [row for row in positive_features if row["variant_id"] == "baseline"]
    clean_features = [feature_row(row, registry, "p2_clean") for row in clean]
    stress_features = [feature_row(row, registry, "p2_stress") for row in stress]
    features_by_id = {
        (row["stage"], row["row_id"]): row
        for row in positive_features + clean_features + stress_features
    }
    summaries = {}
    for name, source_rows in groups.items():
        stage = "p1b_positive" if name.startswith("positive") else (
            "p2_clean" if name.startswith("clean") else "p2_stress")
        feature_rows = [
            features_by_id[(stage, row.get("row_id", row.get("allocation_id")))]
            for row in source_rows
        ]
        summaries[name] = describe(source_rows, feature_rows, registry)
    worst_stress = {}
    for row in stress_features:
        previous = worst_stress.get(row["group_id"])
        if previous is None or row["statistic"] > previous["statistic"]:
            worst_stress[row["group_id"]] = row
    auc_fields = (
        "statistic", "mean_concept_score", "mean_relation_score", "mean_path_score",
        "full_path_fraction", "empty_template_full_path_fraction",
        "fixed_path_without_empty_bonus_reference",
    )
    aucs = {
        key: {
            "baseline_vs_clean_null": auc(
                [row[key] for row in baseline_features], [row[key] for row in clean_features]),
            "baseline_vs_stress_row_selected_by_max_original_statistic_per_source": auc(
                [row[key] for row in baseline_features], [row[key] for row in worst_stress.values()]),
        } for key in auc_fields
    }
    unchanged_fp_rows = [
        features_by_id[("p2_stress", row["row_id"])] for row in false_positives
        if features_by_id[("p2_stress", row["row_id"])]["empty_template_full_path_count"] == 0
    ]
    result = {
        "version": "swan-elastic-v4-selected-path-mechanism:v1",
        "scope": "post-hoc descriptive diagnostics; no new detection or formal FPR/TPR",
        "input_hash_bindings_verified": True,
        "all_selected_scores_match_025_050_025_components": True,
        "bank_sha256": sha(BANK),
        "reference_feature_code_sha256": sha(ROOT / "src/core/bank_scorer.py"),
        "reference_scorer_code_sha_matches_p1_runtime": (
            sha(ROOT / "src/core/bank_scorer.py") == p1["runtime"]["code_sha256"]["bank_scorer.py"]),
        "reference_decoder_code_sha_matches_p1_runtime": (
            sha(ROOT / "src/core/amr_extractor.py") == p1["runtime"]["code_sha256"]["amr_extractor.py"]),
        "reference_code_matches_p2_runtime_not_established": True,
        "reference_template_registry": registry,
        "reference_empty_path_template_indices": [
            row["template_index"] for row in registry if row["path_count"] == 0],
        "reference_signature_equivalent_groups": equivalents,
        "summaries": summaries,
        "empirical_auc_descriptive_only": aucs,
        "candidate": {
            "name": "no_empty_empty_path_reward",
            "change": "set path F1(empty, empty) to 0 only; retain all other rules",
            "state": "hypothesis_only_not_implemented_or_evaluated",
            "justification": "absence of a two-hop path receives 0.25 under the current reference scorer",
            "decision": "one_small_new_source_development_comparison_is_worthwhile; no_full_P2_yet",
            "unchanged_selected_path_fp_rows": len(unchanged_fp_rows),
            "unchanged_selected_path_fp_source_documents": len({
                row["group_id"] for row in unchanged_fp_rows}),
            "conditional_fp_floor": (
                "The unchanged FP paths remain legal and score-identical, while all calibration scores "
                "can only decrease. They remain positive at fixed alpha if reference semantics, "
                "graph, parser, other scoring rules and calibration identities are held fixed."),
            "warning": (
                "Positive space variants have high exposure; fixed-path proxy AUROC decreases. "
                "This is a principled diagnostic candidate, not evidence of improved detection."),
        },
        "limitations": [
            "Only selected winning template/path components are cached here.",
            "Reference Bank feature extraction is from current local code, not a bound P2 execution snapshot.",
            "Changing a match component can change the winning template and DP path.",
            "Fixed-path subtraction is an algebraic diagnostic, not the modified detector score.",
            "All observed assets, including the same-source P2, remain exploratory.",
            "Variant rows and units are dependent; descriptive AUROC has no confirmatory confidence interval.",
            "No candidate threshold, formal FPR, TPR, or survival rate is inferred from these features.",
        ],
    }
    json_output(OUT / "MECHANISM_RESULT.json", result)
    all_features = positive_features + clean_features + stress_features
    with (OUT / "DOCUMENT_FEATURES.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_features[0]))
        writer.writeheader()
        writer.writerows(all_features)
    json_output(OUT / "FEATURE_MANIFEST.json", {
        "version": "swan-v4-mechanism-feature-package:v1",
        "source_files": {
            str(path.relative_to(ROOT)): sha(path)
            for path in [
                Path(__file__).resolve(), BANK, ROOT / "src/core/bank_scorer.py",
                ROOT / "src/core/amr_extractor.py", ROOT / "src/core/sqlite_cache.py",
                ROOT / "src/core/models.py",
                AUDIT / "AUDIT_MANIFEST.json",
            ] + [
                exp["path"] / name for exp in (p1, p2) for name in (
                    "INPUT_MANIFEST.json", "RUNTIME_MANIFEST.json", "RESULT.json",
                    "results/CALIBRATION.json", "results/calibration_rows.jsonl",
                    "results/development_rows.jsonl",
                )
            ]
        },
        "environment": {"python": platform.python_version(), "penman": version("penman")},
        "artifacts": {
            name: sha(OUT / name) for name in ("MECHANISM_RESULT.json", "DOCUMENT_FEATURES.csv")
        },
        "row_count": len(all_features), "csv_contains_source_text": False,
        "original_p2_result_sha256": sha(p2["path"] / "RESULT.json"),
    })
    package_files = (
        "REPORT.md", "DEVELOPMENT_PROTOCOL.md", "MECHANISM_RESULT.json",
        "DOCUMENT_FEATURES.csv", "FEATURE_MANIFEST.json",
    )
    if all((OUT / name).is_file() for name in package_files):
        json_output(OUT / "PACKAGE_MANIFEST.json", {
            "version": "swan-v4-mechanism-package:v1",
            "scope": "selected-path diagnostics; candidate not implemented or evaluated",
            "files": {name: sha(OUT / name) for name in package_files},
            "reproduction_script_sha256": sha(Path(__file__).resolve()),
            "original_audit_manifest_sha256": sha(AUDIT / "AUDIT_MANIFEST.json"),
            "original_p2_result_sha256": sha(p2["path"] / "RESULT.json"),
            "aligned_document_sha256": {
                name: sha(ROOT / name) for name in (
                    "SWAN_弹性重构防御_总体规划.md", "SWAN_弹性重构防御_部署运行手册.md",
                    "任务清单.md", "experiments/README.md",
                )
            },
        })
    print(json.dumps({
        "feature_rows": len(all_features),
        "empty_path_templates_reference": result["reference_empty_path_template_indices"],
        "cohorts": {
            name: {
                "rows": value["documents_or_variant_rows"],
                "units": value["selected_units"],
                "empty_full_path_units": value["empty_template_full_path_score_units"],
                "exposed_rows": value["rows_with_empty_template_full_path"],
                "full_path_units": value["full_path_score_units"],
                "median_reference_bonus_statistic":
                    value["document_feature_distributions"]["empty_path_bonus_statistic_reference"]["median"],
            } for name, value in summaries.items()
        },
        "empirical_auc_descriptive_only": aucs,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
