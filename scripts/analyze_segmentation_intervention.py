"""Summarize paired detection; calibrate only after all frozen cohorts finish."""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys

from analyze_e5_transfer import exact_ci
from run_segmentation_intervention import ROOT, compatible_runtime, sha, text_sha, write_json


def calibrate(scores, alpha=.01):
    if not scores or not 0 < alpha < 1 or not all(math.isfinite(s) for s in scores):
        raise ValueError("Calibration requires finite scores and 0 < alpha < 1")
    # Strict z > threshold. Ties reduce, rather than inflate, empirical FPR.
    return sorted(scores, reverse=True)[math.floor(alpha * len(scores))]


def rate_summary(count, total):
    if total < 1 or not 0 <= count <= total:
        raise ValueError("Invalid binomial counts")
    return {"count": count, "n": total, "rate": count / total,
            "clopper_pearson_95": exact_ci(count, total)}


def paired_success(first, second):
    if not first or len(first) != len(second):
        raise ValueError("Paired outcomes must be nonempty and aligned")
    lost = sum(a and not b for a, b in zip(first, second))
    gained = sum(b and not a for a, b in zip(first, second))
    discordant = lost + gained
    p_value = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(lost, gained) + 1))
                  / 2 ** discordant) if discordant else 1.
    return {"n": len(first), "both_success": sum(a and b for a, b in zip(first, second)),
            "sentence_only_success": lost, "clause_only_success": gained,
            "neither_success": sum(not a and not b for a, b in zip(first, second)),
            "valid_asr_delta_clause_minus_sentence": (gained - lost) / len(first),
            "mcnemar_exact_two_sided_p": p_value,
            "note": "Paired exploratory subset; does not include parser/matcher or calibration uncertainty."}


def summarize(folder):
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {r["id"]: r for r in manifest["records"]}
    by_cohort = {name: [] for name in ("attack", "human_calibration", "human_test")}
    hashes = {"manifest.json": sha(manifest_path)}
    signatures = set()
    pairs = []
    for path in sorted((folder / "detected").glob("*.json")):
        row = json.loads(path.read_text())
        record = row["record"]
        if record["id"] not in expected or record != expected[record["id"]]:
            raise ValueError(f"Result does not match manifest: {path}")
        if row["runtime"]["manifest_sha256"] != hashes["manifest.json"]:
            raise ValueError("Result used a different manifest")
        signatures.add(text_sha(json.dumps(compatible_runtime(row), sort_keys=True)))
        hashes[str(path.relative_to(folder))] = sha(path)
        variants = row["variants"]
        for name in ("sentence", "clause"):
            variant = variants[name]
            n = len(variant["details"])
            green = sum(r["score"] >= .65 for r in variant["details"])
            if n == 0 or n != variant["n"] or green != variant["green"]:
                raise ValueError("Invalid parsed/matched counts")
            if [r["text"] for r in variant["details"]] != record["segments"][name]:
                raise ValueError("Scored segments do not match manifest")
            expected_z = (green - .25 * n) / math.sqrt(n * .25 * .75)
            if not math.isclose(expected_z, variant["z"], abs_tol=1e-10):
                raise ValueError("Stored score differs from sentence matches")
        by_cohort[record["cohort"]].append(row)
        if record["cohort"] == "attack":
            pairs.append({
                "id": record["id"], "quality_valid": record["quality_valid"],
                "historical_z": record["historical_z"],
                **{f"{mode}_{key}": variants[mode][key]
                   for mode in ("sentence", "clause") for key in ("n", "green", "green_rate", "z")},
            })
    if len(signatures) > 1:
        raise ValueError("Mixed runtimes: do not aggregate")
    completed = [r["record"]["id"] for group in by_cohort.values() for r in group]
    if len(completed) != len(set(completed)):
        raise ValueError("Duplicate result identity")
    coverage = {
        name: {"completed": len(rows), "planned": sum(r["cohort"] == name for r in expected.values())}
        for name, rows in by_cohort.items()
    }
    all_done = all(v["completed"] == v["planned"] for v in coverage.values())
    ready = all_done and all(coverage[name]["completed"] >= 100 for name in ("human_calibration", "human_test"))
    paired = {
        "n": len(pairs),
        "changed_segmentation": sum(p["sentence_n"] != p["clause_n"] for p in pairs),
        "green_rate_increased": sum(p["clause_green_rate"] > p["sentence_green_rate"] for p in pairs),
        "green_rate_decreased": sum(p["clause_green_rate"] < p["sentence_green_rate"] for p in pairs),
        "z_increased": sum(p["clause_z"] > p["sentence_z"] for p in pairs),
        "z_decreased": sum(p["clause_z"] < p["sentence_z"] for p in pairs),
        "historical_z_mismatches": sum(not math.isclose(p["historical_z"], p["sentence_z"], abs_tol=1e-10) for p in pairs),
        "mean_delta_z": statistics.mean(p["clause_z"] - p["sentence_z"] for p in pairs) if pairs else None,
        "fixed_2_33_crossings_up": sum(p["sentence_z"] <= 2.33 < p["clause_z"] for p in pairs),
        "fixed_2_33_crossings_down": sum(p["clause_z"] <= 2.33 < p["sentence_z"] for p in pairs),
        "note": "Fixed-threshold crossings are descriptive, not calibrated defense recovery rates.",
    }
    calibrated = None
    calibrated_pair = None
    if ready:
        calibrated = {}
        success_masks = {}
        for mode in ("sentence", "clause"):
            calibration = [r["variants"][mode]["z"] for r in by_cohort["human_calibration"]]
            test = [r["variants"][mode]["z"] for r in by_cohort["human_test"]]
            attack = by_cohort["attack"]
            threshold = calibrate(calibration)
            success_masks[mode] = [r["record"]["quality_valid"] and r["variants"][mode]["z"] <= threshold for r in attack]
            successes = sum(success_masks[mode])
            false_positives = sum(s > threshold for s in test)
            calibrated[mode] = {
                "threshold": threshold, "empirical_calibration_fpr": sum(s > threshold for s in calibration) / len(calibration),
                "test_fpr": false_positives / len(test),
                "test_false_positives": rate_summary(false_positives, len(test)),
                "test_fpr_point_estimate_within_1pct": false_positives / len(test) <= .01,
                "valid_successes": successes, "attack_n": len(attack),
                "valid_asr": successes / len(attack),
                "valid_success_interval": rate_summary(successes, len(attack)),
                "threshold_policy": "Calibrated once on calibration cohort; never tuned on test cohort",
            }
        calibrated_pair = paired_success(success_masks["sentence"], success_masks["clause"])
    return {"status": "complete_calibrated" if ready else "paired_pilot_uncalibrated",
            "coverage": coverage, "paired_attack": paired, "calibrated": calibrated,
            "calibrated_paired_attack": calibrated_pair,
            "pending_ids": sorted(set(expected) - set(completed)),
            "pairs": pairs, "input_sha256": hashes}


def repeat_mismatches(folder, repeats):
    os.environ.setdefault("NLTK_DATA", str(ROOT / ".nltk_data"))
    os.environ.setdefault("GLOVE_VECTORS_PATH", str(ROOT / "vectors/glove.6B.100d.txt"))
    sys.path.insert(0, str(ROOT / "third_party/SWAN"))
    from utils.amr_utils import load_amr_bank, compute_s2match_score

    base = ROOT / "experiments/planE_e2_joint250"
    old_path = base / "parsed.json"
    old_graphs = json.loads(old_path.read_text())
    bank_path = ROOT / "third_party/SWAN/amr_bank/banks/amr_bank_50.json"
    bank = load_amr_bank(str(bank_path))
    rows = []
    inputs = {"historical_parsed_sha256": sha(old_path), "bank_sha256": sha(bank_path),
              "vectors_sha256": sha(Path(os.environ["GLOVE_VECTORS_PATH"]))}
    for path in sorted((folder / "detected").glob("attack-*.json")):
        saved = json.loads(path.read_text())
        record = saved["record"]
        variant = saved["variants"]["sentence"]
        if math.isclose(variant["z"], record["historical_z"], abs_tol=1e-10):
            continue
        inputs[str(path.relative_to(folder))] = sha(path)
        graphs = [d["amr"] for d in variant["details"]]
        trials = []
        for _ in range(repeats):
            scores = [max(float(compute_s2match_score(g, template)) for template in bank) for g in graphs]
            n = len(scores)
            green = sum(s >= .65 for s in scores)
            trials.append({"best_scores": scores, "green": green,
                           "z": (green - .25 * n) / math.sqrt(n * .25 * .75)})
        rows.append({"id": record["id"], "same_amrs_as_historical": graphs == old_graphs[record["position"]],
                     "historical_z": record["historical_z"], "rerun_z": variant["z"], "trials": trials})
        print(json.dumps({"id": record["id"], "same_amrs": rows[-1]["same_amrs_as_historical"],
                          "z_range": [min(t["z"] for t in trials), max(t["z"] for t in trials)]}), flush=True)
    result = {"repeats": repeats, "selection": "All completed attack records whose original-segmentation z differs from history",
              "purpose": "Stochastic matching diagnostic; does not replace primary paired results",
              "rows": rows, "input_sha256": inputs}
    write_json(folder / "matching_repeatability.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=ROOT / "experiments/segmentation_intervention")
    parser.add_argument("--repeat-mismatches", type=int, default=0)
    args = parser.parse_args()
    if args.repeat_mismatches < 0:
        parser.error("Repeat count must be nonnegative")
    result = summarize(args.folder)
    write_json(args.folder / "summary.json", result)
    print(json.dumps({k: result[k] for k in ("status", "coverage", "paired_attack", "calibrated")}, indent=2))
    if args.repeat_mismatches:
        repeat_mismatches(args.folder, args.repeat_mismatches)


if __name__ == "__main__":
    main()
