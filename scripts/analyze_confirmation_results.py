"""Finalize the complete cohort without new generation or detector queries."""

from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import statistics

from scipy.stats import binomtest

from audit_confirmation_experiment import audit
from prepare_confirmation_experiment import ARMS
from run_confirmation_experiment import decode_reviews, digest
from run_segmentation_intervention import ROOT, sha, write_json


def distribution(values):
    if not values:
        return None
    return {"n": len(values), "min": min(values), "median": statistics.median(values),
            "mean": statistics.mean(values), "max": max(values)}


def holm_adjust(values):
    ordered = sorted(values, key=values.get)
    result, previous = {}, 0.
    for rank, key in enumerate(ordered):
        previous = max(previous, min(1., (len(ordered) - rank) * values[key]))
        result[key] = previous
    return result


def analyze(folder):
    verified = audit(folder)
    if verified["status"] != "complete_verified":
        raise ValueError("Final analysis requires the complete frozen cohort")
    inputs = dict(verified["input_sha256"])

    def load(path):
        inputs[str(path.relative_to(folder))] = sha(path)
        return json.loads(path.read_text())

    expected = {"baselines": 30, "injection": 150, "attacks": 360,
                "quality": 90, "evaluated": 90}
    for name, count in expected.items():
        if len(list((folder / name).glob("*.json"))) != count:
            raise ValueError(f"Unexpected artifact count: {name}")
    baselines = {r["prompt_id"]: r for r in
                 (load(p) for p in sorted((folder / "baselines").glob("*.json")))}
    calls = {p.name: load(p) for p in sorted((folder / "calls").glob("*.json"))}
    for pid, baseline in baselines.items():
        for slot, row in enumerate(baseline["sentences"]):
            if row != load(folder / "injection" / f"{pid}-{slot}.json"):
                raise ValueError("Baseline sentence differs from injection checkpoint")
            prefix = f"inject-{pid}-{slot}"
            spent = sum(r["scope"] == prefix or r["scope"].startswith(prefix + "-recovery")
                        for r in calls.values())
            if spent != row["attempts"] or not 1 <= spent <= 50:
                raise ValueError("Transmitted sentence requests differ from saved budget")

    arms = {}
    initially_detected = {pid for pid, b in baselines.items() if b["detection"]["z"] > 2.33}
    for arm in ARMS:
        groups = [load(folder / "evaluated" / f"{pid}-{arm}.json") for pid in sorted(baselines)]
        rows = []
        for group in groups:
            pid = group["prompt_id"]
            review = calls[f"review-{pid}-{arm}-000.json"]
            decoded = decode_reviews(review["response"]["content"])
            for row in group["rows"]:
                k = row["candidate"]
                call = calls[f"attack-{pid}-{arm}-{k}-000.json"]
                if call["status"] != "complete" or row["attack"] != call["response"]["content"].strip():
                    raise ValueError("Candidate differs from original API response")
                if row["review"] != decoded[k]:
                    raise ValueError("Review differs from original API response")
                rows.append(row)
        selected = [next((r for r in g["rows"] if r["candidate"] == g["feedback_candidate"]), None)
                    for g in groups]
        selected = [r for r in selected if r is not None]
        arms[arm] = {
            **verified["arms"][arm],
            "candidate_sentence_counts": dict(Counter(r["detection"]["n"] for r in rows)),
            "candidate_words": distribution([len(r["attack"].split()) for r in rows]),
            "selected_sentence_counts": dict(Counter(r["detection"]["n"] for r in selected)),
            "selected_green_counts": dict(Counter(r["detection"]["green"] for r in selected)),
            "conditional_no_feedback_successes": sum(
                g["no_feedback_success"] for g in groups if g["prompt_id"] in initially_detected),
            "quality_pass_fraction": sum(r["quality_valid"] for r in rows) / len(rows),
        }
        for rate in arms[arm]["rates"].values():
            if rate is None:
                continue
            ci = binomtest(rate["count"], rate["n"]).proportion_ci(method="exact")
            if not all(math.isclose(a, b, abs_tol=1e-9) for a, b in
                       zip(rate["clopper_pearson_95"], (ci.low, ci.high))):
                raise ValueError("Independent confidence interval verification failed")

    comparisons = verified["paired_comparisons"]
    p_values = {}
    for key, pair in comparisons.items():
        a, b = pair["comparator_only_success"], pair["full_e2_only_success"]
        p = binomtest(a, a + b, .5).pvalue if a + b else 1.
        if not math.isclose(p, pair["mcnemar_exact_two_sided_p"], abs_tol=1e-12):
            raise ValueError("Independent paired test verification failed")
        p_values[key] = p
    adjusted = holm_adjust(p_values)
    comparisons = {k: {**v, "holm_adjusted_p_two_comparisons": adjusted[k]}
                   for k, v in comparisons.items()}
    unknown = []
    for filename, call in calls.items():
        if call["status"] != "complete":
            unknown.append({"file": filename, "status": call["status"],
                            "error_type": call.get("error_type")})
        elif not all(type(call["response"]["usage"].get(k)) is int
                     and call["response"]["usage"][k] >= 0
                     for k in ("prompt_tokens", "completion_tokens")):
            raise ValueError("Completed call has missing or invalid token usage")

    final_run = load(folder / "run_logs/status.json")
    if final_run["status"] != "runner_succeeded" or final_run["runner_returncode"] != 0:
        raise ValueError("Expected successful supervised exit")
    heartbeat_path = folder / "run_logs" / f'{final_run["run_id"]}.jsonl'
    inputs[str(heartbeat_path.relative_to(folder))] = sha(heartbeat_path)
    heartbeats = [json.loads(line) for line in heartbeat_path.read_text().splitlines()]
    times = [datetime.fromisoformat(r["updated_at"]).timestamp() for r in heartbeats]
    return {
        "status": "complete_independently_verified", "artifact_counts": expected,
        "baseline_words": distribution([len(r["text"].split()) for r in baselines.values()]),
        "baseline_sentence_counts": dict(Counter(r["detection"]["n"] for r in baselines.values())),
        "initially_undetected_ids": sorted(set(baselines) - initially_detected),
        "all_five_sentences_accepted": verified["all_five_sentences_accepted"],
        "injection_sentence_failures": sum(not s["accepted"] for b in baselines.values() for s in b["sentences"]),
        "arms": arms, "paired_comparisons": comparisons,
        "api_costs_by_stage_arm": verified["api_costs_by_stage_arm"],
        "unknown_usage_calls": unknown, "call_status_counts": dict(Counter(r["status"] for r in calls.values())),
        "final_supervised_run": {
            "run_id": final_run["run_id"], "wall_seconds": times[-1] - times[0],
            "heartbeat_count": len(times),
            "max_heartbeat_gap_seconds": max(b - a for a, b in zip(times, times[1:])),
            "new_calls": sum(datetime.fromisoformat(r["started_at"]).timestamp() >= times[0]
                             for r in calls.values()),
            "new_uncertain_calls": sum(
                r["status"] != "complete" and datetime.fromisoformat(r["started_at"]).timestamp() >= times[0]
                for r in calls.values()),
        },
        "limitations": [
            "Prompt-assigned arms, not a verified vocabulary-preserving causal intervention.",
            "Fresh generated baselines are not interchangeable with historical 250-paragraph sources.",
            "Fixed z=2.33; no new same-FPR calibration or AUROC experiment.",
            "Confidence intervals condition on this run; parser, generation and reviewer uncertainty excluded.",
            "Holm correction covers only the two recorded feedback comparisons.",
            "Token totals are not currency charges; uncertain calls have unknown usage.",
        ],
        "analysis_sha256": sha(Path(__file__)), "audit_result_sha256": digest(verified),
        "input_sha256": inputs,
    }


if __name__ == "__main__":
    folder = ROOT / "experiments/confirmation_three_arm30"
    result = analyze(folder)
    write_json(folder / "final_analysis.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "input_sha256"}, indent=2))
