"""Audit and summarize the confirmation sentence-boundary mechanism study."""

from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from analyze_sentence_count_decomposition import decompose
from analyze_segmentation_intervention import paired_success, rate_summary
from prepare_confirmation_experiment import ARMS
from run_confirmation_mechanism import lexical_tokens
from run_segmentation_intervention import ROOT, compatible_runtime, sha, text_sha, write_json


FIXED_THRESHOLD = 2.33


def verify_variant(variant, expected_segments):
    details = variant["details"]
    if [row["text"] for row in details] != expected_segments or not details:
        raise ValueError("Detected segments differ from frozen manifest")
    green = 0
    for row in details:
        scores = row["bank_scores"]
        if len(scores) != 50 or not all(math.isfinite(v) and 0 <= v <= 1 for v in scores):
            raise ValueError("Invalid S2Match vector")
        if row["score"] != max(scores):
            raise ValueError("Saved maximum does not match template scores")
        green += row["score"] >= .65
    n = len(details)
    z = (green - .25 * n) / math.sqrt(n * .25 * .75)
    if variant["n"] != n or variant["green"] != green or not math.isclose(variant["z"], z, abs_tol=1e-10):
        raise ValueError("Invalid aggregate detection score")


def summarize_mask(mask):
    return rate_summary(sum(mask), len(mask))


def transitions(before, after):
    if len(before) != len(after):
        raise ValueError("Transition vectors are not aligned")
    return {
        "both_escape": sum(a and b for a, b in zip(before, after)),
        "before_only_escape": sum(a and not b for a, b in zip(before, after)),
        "after_only_escape": sum(b and not a for a, b in zip(before, after)),
        "neither_escape": sum(not a and not b for a, b in zip(before, after)),
        "mcnemar_exact_two_sided_p": paired_success(before, after)["mcnemar_exact_two_sided_p"],
    }


def decomposition_summary(rows):
    if not rows:
        return None
    categories = {
        "both_individually_sufficient": 0,
        "count_only_sufficient": 0,
        "green_rate_only_sufficient": 0,
        "joint_change_required": 0,
        "no_actual_escape": 0,
    }
    for row in rows:
        count_low = row["count_only"] <= FIXED_THRESHOLD
        rate_low = row["rate_only"] <= FIXED_THRESHOLD
        if row["observed"] > FIXED_THRESHOLD:
            category = "no_actual_escape"
        elif count_low and rate_low:
            category = "both_individually_sufficient"
        elif count_low:
            category = "count_only_sufficient"
        elif rate_low:
            category = "green_rate_only_sufficient"
        else:
            category = "joint_change_required"
        categories[category] += 1
    return {
        "n": len(rows),
        "mean_observed_drop": statistics.mean(r["baseline"] - r["observed"] for r in rows),
        "mean_count_drop": statistics.mean(r["count_drop"] for r in rows),
        "mean_green_rate_drop": statistics.mean(r["rate_drop"] for r in rows),
        "count_only_escape": sum(r["count_only"] <= FIXED_THRESHOLD for r in rows),
        "green_rate_only_escape": sum(r["rate_only"] <= FIXED_THRESHOLD for r in rows),
        "actual_escape": sum(r["observed"] <= FIXED_THRESHOLD for r in rows),
        "sufficiency_categories": categories,
    }


def analyze(folder):
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {row["id"]: row for row in manifest["records"]}
    if len(expected) != 134:
        raise ValueError("Expected 134 unique mechanism records")
    source = ROOT / "experiments/confirmation_three_arm30"
    dependency_paths = {
        "confirmation_final_analysis": source / "final_analysis.json",
        "confirmation_runtime": source / "runtime.json",
        "segmentation_summary": ROOT / "experiments/segmentation_intervention/summary.json",
        "segmentation_manifest": ROOT / "experiments/segmentation_intervention/manifest.json",
        "prepare_script": ROOT / "scripts/run_confirmation_mechanism.py",
    }
    for name, path in dependency_paths.items():
        if sha(path) != manifest["dependencies"][name]:
            raise ValueError(f"Frozen dependency changed: {name}")
    inputs = {
        "manifest.json": sha(manifest_path),
        "analysis_script": sha(Path(__file__)),
    }
    detected = {}
    runtimes = set()
    for path in sorted((folder / "detected").glob("*.json")):
        saved = json.loads(path.read_text())
        record = saved["record"]
        if record["id"] not in expected or record != expected[record["id"]]:
            raise ValueError("Detected record differs from manifest")
        if record["id"] in detected:
            raise ValueError("Duplicate detected identity")
        if saved["runtime"]["manifest_sha256"] != inputs["manifest.json"]:
            raise ValueError("Detection used a different manifest")
        runtimes.add(text_sha(json.dumps(compatible_runtime(saved), sort_keys=True)))
        if set(saved["variants"]) != set(record["segments"]):
            raise ValueError("Variant set differs")
        for name, variant in saved["variants"].items():
            verify_variant(variant, record["segments"][name])
        detected[record["id"]] = saved
        inputs[str(path.relative_to(folder))] = sha(path)
    if set(detected) != set(expected) or len(runtimes) != 1:
        raise ValueError("Mechanism detection is incomplete or mixes runtimes")

    thresholds = manifest["protocol"]["calibrated_thresholds"]
    punctuation = [detected[rid] for rid in sorted(detected)
                   if detected[rid]["record"]["kind"] == "punctuation_control"]
    punctuation_rows = []
    for saved in punctuation:
        record, variants = saved["record"], saved["variants"]
        if lexical_tokens(record["text"]) != lexical_tokens(" ".join(record["segments"]["original"])):
            raise ValueError("Punctuation control changed lexical tokens")
        if (variants["original"]["n"], variants["merged_sentence"]["n"],
                variants["merged_exact_clause"]["n"]) != (5, 1, 5):
            raise ValueError("Punctuation intervention did not realize exact 5 -> 1 -> 5 units")
        punctuation_rows.append({
            "prompt_id": record["prompt_id"],
            **{f"{mode}_{key}": variants[mode][key]
               for mode in ("original", "merged_sentence", "merged_rule_clause",
                            "merged_exact_clause")
               for key in ("n", "green", "z")},
        })
    punctuation_summary = {}
    for threshold_name, mode_thresholds in (
        ("fixed_2_33", {"original": FIXED_THRESHOLD, "merged_sentence": FIXED_THRESHOLD,
                        "merged_rule_clause": FIXED_THRESHOLD,
                        "merged_exact_clause": FIXED_THRESHOLD}),
        ("calibrated", {"original": thresholds["sentence"],
                        "merged_sentence": thresholds["sentence"],
                        "merged_rule_clause": thresholds["clause"],
                        "merged_exact_clause": thresholds["clause"]}),
    ):
        masks = {
            mode: [row[f"{mode}_z"] <= mode_thresholds[mode] for row in punctuation_rows]
            for mode in ("original", "merged_sentence", "merged_rule_clause",
                         "merged_exact_clause")
        }
        punctuation_summary[threshold_name] = {
            "thresholds": mode_thresholds,
            "escape_rates": {mode: summarize_mask(mask) for mode, mask in masks.items()},
            "original_to_merged": transitions(masks["original"], masks["merged_sentence"]),
            "merged_to_rule_resegmented": transitions(
                masks["merged_sentence"], masks["merged_rule_clause"]),
            "merged_to_exact_resegmented": transitions(
                masks["merged_sentence"], masks["merged_exact_clause"]),
            "recovered_detections_after_rule_clause_split": sum(
                before and not after
                for before, after in zip(masks["merged_sentence"], masks["merged_rule_clause"])
            ),
            "recovered_detections_after_exact_clause_split": sum(
                before and not after
                for before, after in zip(masks["merged_sentence"], masks["merged_exact_clause"])
            ),
        }

    selected_lookup = {
        (row["record"]["prompt_id"], row["record"]["arm"], row["record"]["candidate"]): row
        for row in detected.values() if row["record"]["kind"] == "selected_attack"
    }
    selected_summary = {}
    decompositions = defaultdict(list)
    all_candidate_strata = {}
    for arm in ARMS:
        groups = [json.loads(path.read_text()) for path in
                  sorted((source / "evaluated").glob(f"*-{arm}.json"))]
        for path in sorted((source / "evaluated").glob(f"*-{arm}.json")):
            inputs[str(path.relative_to(ROOT))] = sha(path)
        arm_roles = {}
        for role in ("feedback", "no_feedback"):
            historical, fresh_sentence, fresh_clause = [], [], []
            calibrated_sentence, calibrated_clause = [], []
            changed, rerun_mismatch = 0, 0
            for group in groups:
                candidate = group[f"{role}_candidate"]
                historical.append(group[f"{role}_success"])
                if candidate is None:
                    fresh_sentence.append(False)
                    fresh_clause.append(False)
                    calibrated_sentence.append(False)
                    calibrated_clause.append(False)
                    continue
                saved = selected_lookup[(group["prompt_id"], arm, candidate)]
                variants = saved["variants"]
                fresh_sentence.append(variants["sentence"]["z"] <= FIXED_THRESHOLD)
                fresh_clause.append(variants["clause"]["z"] <= FIXED_THRESHOLD)
                calibrated_sentence.append(variants["sentence"]["z"] <= thresholds["sentence"])
                calibrated_clause.append(variants["clause"]["z"] <= thresholds["clause"])
                changed += variants["sentence"]["n"] != variants["clause"]["n"]
                rerun_mismatch += not math.isclose(
                    saved["record"]["historical_detection"]["z"],
                    variants["sentence"]["z"], abs_tol=1e-10)
                baseline = json.loads(
                    (source / "baselines" / f'{group["prompt_id"]}.json').read_text())
                b, a = baseline["detection"], group["rows"][candidate]["detection"]
                decompositions[(arm, role)].append(
                    {"prompt_id": group["prompt_id"], **decompose(
                        b["n"], b["green"], a["n"], a["green"])}
                )
            arm_roles[role] = {
                "historical_fixed_escape": summarize_mask(historical),
                "fresh_sentence_fixed_escape": summarize_mask(fresh_sentence),
                "fresh_clause_fixed_escape": summarize_mask(fresh_clause),
                "fresh_sentence_calibrated_escape": summarize_mask(calibrated_sentence),
                "fresh_clause_calibrated_escape": summarize_mask(calibrated_clause),
                "fresh_sentence_to_clause_fixed": transitions(fresh_sentence, fresh_clause),
                "fresh_sentence_to_clause_calibrated": transitions(
                    calibrated_sentence, calibrated_clause),
                "clause_recovered_fixed_detections": sum(
                    a and not b for a, b in zip(fresh_sentence, fresh_clause)),
                "segmentation_changed": changed,
                "fresh_sentence_z_differs_from_historical": rerun_mismatch,
                "decomposition": decomposition_summary(decompositions[(arm, role)]),
            }
        selected_summary[arm] = arm_roles
        strata = defaultdict(lambda: {"quality_candidates": 0, "escapes": 0})
        for group in groups:
            for row in group["rows"]:
                if row["quality_valid"]:
                    n = str(row["detection"]["n"])
                    strata[n]["quality_candidates"] += 1
                    strata[n]["escapes"] += row["detection"]["z"] <= FIXED_THRESHOLD
        all_candidate_strata[arm] = {
            n: {**values, "escape_rate": values["escapes"] / values["quality_candidates"]}
            for n, values in sorted(strata.items(), key=lambda item: int(item[0]))
        }

    return {
        "status": "complete_verified",
        "protocol": {
            "fixed_threshold": FIXED_THRESHOLD,
            "calibrated_thresholds": thresholds,
            "punctuation_control": "Lexically identical 5 -> 1 -> 5 segmentation intervention",
            "decomposition": "Two-feature algebraic Shapley decomposition; not causal attribution",
            "selected_candidates": "Frozen feedback/no-feedback identities; missing quality selections remain failures",
        },
        "coverage": {
            "records": len(detected), "punctuation_controls": len(punctuation),
            "selected_candidates": len(selected_lookup),
        },
        "punctuation_control": punctuation_summary,
        "selected_resegmentation": selected_summary,
        "quality_candidate_sentence_strata": all_candidate_strata,
        "limitations": [
            "Clause splitter is rule-based and does not reconstruct gold discourse units.",
            "Fresh S2Match reruns are stochastic; historical mismatches are reported.",
            "Existing human calibration thresholds are reused, not re-estimated.",
            "Punctuation-only merging tests detector mechanics, not natural rewriting quality preferences.",
            "Algebraic decomposition permits fractional virtual green counts and is not a text intervention.",
        ],
        "rows": {"punctuation": punctuation_rows},
        "input_sha256": inputs,
    }


if __name__ == "__main__":
    folder = ROOT / "experiments/confirmation_mechanism30"
    result = analyze(folder)
    write_json(folder / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("rows", "input_sha256")}, indent=2))
