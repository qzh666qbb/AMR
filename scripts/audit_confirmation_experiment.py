"""Verify three-arm checkpoints and report only complete paired-cohort rates."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from analyze_segmentation_intervention import paired_success, rate_summary
from evaluate_adaptive_retry import gates
from prepare_confirmation_experiment import ARMS
from run_confirmation_experiment import digest
from run_segmentation_intervention import ROOT, sha, text_sha, write_json


def verify_score(scored):
    if scored is None:
        return
    details = scored["details"]
    if not details:
        raise ValueError("Empty scored document")
    for row in details:
        scores = row["bank_scores"]
        if len(scores) != 50 or not all(math.isfinite(s) and 0 <= s <= 1 for s in scores):
            raise ValueError("Invalid template scores")
        if row["score"] != max(scores):
            raise ValueError("Incorrect maximum template score")
    n = len(details)
    g = sum(row["score"] >= .65 for row in details)
    z = (g - .25 * n) / math.sqrt(n * .25 * .75)
    if scored["n"] != n or scored["green"] != g or not math.isclose(scored["z"], z, abs_tol=1e-10):
        raise ValueError("Detection score mismatch")


def audit(folder):
    inputs = {}

    def load(path):
        inputs[str(path.relative_to(folder))] = sha(path)
        return json.loads(path.read_text())

    preparation = load(folder / "preparation.json")
    runtime = load(folder / "runtime.json")
    prompts = {r["prompt_id"]: r for r in preparation["prompts"]}
    if len(prompts) != len(preparation["prompts"]):
        raise ValueError("Duplicate prompt identity")
    baseline_map = {}
    for path in sorted((folder / "baselines").glob("*.json")):
        row = load(path)
        pid = row["prompt_id"]
        if pid not in prompts or pid in baseline_map or row["runtime_sha256"] != digest(runtime):
            raise ValueError("Unexpected baseline identity or runtime")
        if row["prompt_sha256"] != prompts[pid]["generation_prompt_sha256"]:
            raise ValueError("Baseline prompt identity changed")
        if text_sha(row["text"]) != row["text_sha256"]:
            raise ValueError("Baseline text changed")
        if len(row["sentences"]) != 5 or any(s["attempts"] > 50 for s in row["sentences"]):
            raise ValueError("Injection sentence budget exceeded")
        verify_score(row["detection"])
        baseline_map[pid] = row
    evaluated = {arm: {} for arm in ARMS}
    for path in sorted((folder / "evaluated").glob("*.json")):
        item = load(path)
        pid, arm = item["prompt_id"], item["arm"]
        if pid not in baseline_map or arm not in ARMS or pid in evaluated[arm]:
            raise ValueError("Unpaired or duplicate evaluated identity")
        quality = load(folder / "quality" / f"{pid}-{arm}.json")
        if digest(quality) != item["quality_sha256"]:
            raise ValueError("Quality checkpoint hash mismatch")
        rows = item["rows"]
        if len(rows) != 4 or [r["candidate"] for r in rows] != list(range(4)):
            raise ValueError("Candidate budget or order differs")
        generated = []
        for row, label in zip(rows, quality["rows"]):
            candidate = load(folder / "attacks" / f'{pid}-{arm}-{row["candidate"]}.json')
            generated.append(candidate)
            if any(row[k] != v for k, v in candidate.items()):
                raise ValueError("Evaluated candidate differs from generated text")
            if row["source"] != baseline_map[pid]["text"] or row["source_sha256"] != baseline_map[pid]["text_sha256"]:
                raise ValueError("Wrong watermark source")
            if {k: v for k, v in row.items() if k != "detection"} != label:
                raise ValueError("Quality labels changed after detection")
            surface = gates(row, [d["amr"] for d in row["details"]])
            review = row["review"]
            valid = all(surface.values()) and review["score"] >= 4 and not review["major_error"] and row["finish_reason"] == "stop"
            if row["surface"] != surface or row["quality_valid"] != valid:
                raise ValueError("Quality gate reconstruction failed")
            verify_score(row["detection"])
            if row["detection"] and [{k: v for k, v in d.items() if k not in ("score", "bank_scores")}
                                     for d in row["detection"]["details"]] != row["details"]:
                raise ValueError("Detection did not use quality-reviewed AMRs")
        if digest(generated) != quality["group_sha256"]:
            raise ValueError("Generation group hash mismatch")
        eligible = [r for r in rows if r["quality_valid"] and r["detection"]]
        first = eligible[0] if eligible else None
        best = min(eligible, key=lambda r: (r["detection"]["z"], r["candidate"])) if eligible else None
        for name, selected in (("no_feedback", first), ("feedback", best)):
            if item[f"{name}_candidate"] != (selected["candidate"] if selected else None):
                raise ValueError("Selection rule mismatch")
            if item[f"{name}_success"] != bool(selected and selected["detection"]["z"] <= 2.33):
                raise ValueError("Success label mismatch")
        if quality["no_feedback_candidate"] != item["no_feedback_candidate"]:
            raise ValueError("No-feedback selection changed")
        evaluated[arm][pid] = item

    ids = sorted(prompts)
    ready = set(baseline_map) == set(prompts) and all(set(group) == set(prompts) for group in evaluated.values())
    detected = {pid for pid, r in baseline_map.items() if r["detection"] and r["detection"]["z"] > 2.33}
    arms = {}
    for arm, group in evaluated.items():
        rows = [r for item in group.values() for r in item["rows"]]
        arms[arm] = {
            "completed_paragraphs": len(group), "quality_candidates": sum(r["quality_valid"] for r in rows),
            "quality_paragraphs": sum(any(r["quality_valid"] for r in item["rows"]) for item in group.values()),
            "feedback_successes": sum(item["feedback_success"] for item in group.values()),
            "no_feedback_successes": sum(item["no_feedback_success"] for item in group.values()),
            "new_successes_on_initially_detected": sum(item["feedback_success"] for pid, item in group.items() if pid in detected),
            "rates": None,
        }
        if ready:
            a = arms[arm]
            a["rates"] = {
                "valid_asr": rate_summary(a["feedback_successes"], len(ids)),
                "no_feedback_valid_asr": rate_summary(a["no_feedback_successes"], len(ids)),
                "conditional_valid_asr": rate_summary(a["new_successes_on_initially_detected"], len(detected)) if detected else None,
            }
    comparisons = {}
    if ready:
        for comparator in ("plain_paraphrase", "boundary_only"):
            comparison = paired_success(
                [evaluated[comparator][i]["feedback_success"] for i in ids],
                [evaluated["full_e2"][i]["feedback_success"] for i in ids])
            comparisons[f"{comparator}_to_full_e2"] = {
                "comparator": comparator, "n": comparison["n"],
                "both_success": comparison["both_success"],
                "comparator_only_success": comparison["sentence_only_success"],
                "full_e2_only_success": comparison["clause_only_success"],
                "neither_success": comparison["neither_success"],
                "valid_asr_delta_full_minus_comparator": comparison["valid_asr_delta_clause_minus_sentence"],
                "mcnemar_exact_two_sided_p": comparison["mcnemar_exact_two_sided_p"],
                "note": "Unadjusted paired test; calibration and generation randomness not integrated.",
            }
    calls = [load(p) for p in sorted((folder / "calls").glob("*.json"))]
    costs, returned = {}, Counter()
    for row in calls:
        if digest(row["request"]) != row["request_sha256"]:
            raise ValueError("API request hash mismatch")
        parts = row["scope"].split("-")
        stage = parts[0]
        name = parts[4] if stage in ("attack", "review") else "shared"
        key = f"{stage}:{name}"
        cost = costs.setdefault(key, {"attempts": 0, "complete": 0, "unknown_usage_calls": 0,
                                     "prompt_tokens": 0, "completion_tokens": 0})
        cost["attempts"] += 1
        if row["status"] == "complete":
            cost["complete"] += 1
            response = row["response"]
            returned[response["model"]] += 1
            if response["model"] != runtime["model_returned"]:
                raise ValueError("API returned a different model")
            for k in ("prompt_tokens", "completion_tokens"):
                cost[k] += response["usage"].get(k, 0)
        else:
            cost["unknown_usage_calls"] += 1
    return {"status": "complete_verified" if ready else "partial_verified_no_final_rates",
            "planned_paragraphs": len(ids), "baselines_completed": len(baseline_map),
            "initially_detected_baselines": len(detected),
            "all_five_sentences_accepted": sum(r["all_injection_accepted"] for r in baseline_map.values()),
            "arms": arms, "paired_comparisons": comparisons,
            "api_costs_by_stage_arm": costs, "returned_models": dict(returned),
            "notes": ["Token totals exclude any call with unknown usage.",
                      "Paired comparisons reuse the same baselines; conditional denominator excludes original misses.",
                      "No final rates until the entire frozen cohort is complete."],
            "input_sha256": inputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=ROOT / "experiments/confirmation_three_arm30")
    args = parser.parse_args()
    result = audit(args.folder)
    write_json(args.folder / "audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "input_sha256"}, indent=2))


if __name__ == "__main__":
    main()
