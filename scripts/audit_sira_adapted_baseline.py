"""Independently verify the complete SIRA-adapted comparison experiment."""

from collections import Counter
import json
import math
from pathlib import Path

from analyze_segmentation_intervention import paired_success, rate_summary
from audit_confirmation_experiment import verify_score
from evaluate_adaptive_retry import gates
from prepare_confirmation_experiment import ARMS
from run_confirmation_experiment import (
    REVIEW, decode_reviews, digest, request_body,
)
from run_segmentation_intervention import ROOT, sha, text_sha, write_json
from run_sira_adapted_baseline import (
    ATTACK_PROMPT, K, MASK_PERCENTILE, REFERENCE_PROMPT,
    SelfInformationModel, mask_tokens,
)


def comparison(left_name, left, right_name, right):
    result = paired_success(left, right)
    return {
        "left": left_name, "right": right_name, "n": result["n"],
        "both_success": result["both_success"],
        "left_only_success": result["sentence_only_success"],
        "right_only_success": result["clause_only_success"],
        "neither_success": result["neither_success"],
        "valid_asr_delta_right_minus_left": result["valid_asr_delta_clause_minus_sentence"],
        "mcnemar_exact_two_sided_p": result["mcnemar_exact_two_sided_p"],
    }


def audit(folder, recompute_self_information=True):
    inputs = {}

    def load(path):
        inputs[str(path.relative_to(folder))] = sha(path)
        return json.loads(path.read_text())

    manifest = load(folder / "manifest.json")
    runtime = load(folder / "runtime.json")
    inputs["audit_script"] = sha(Path(__file__))
    records = {row["prompt_id"]: row for row in manifest["records"]}
    if len(records) != 30 or len(manifest["records"]) != 30:
        raise ValueError("Expected 30 unique frozen records")
    for key, expected in manifest["dependencies"].items():
        path = ROOT / key
        if not path.exists() or sha(path) != expected:
            raise ValueError(f"Frozen manifest dependency changed: {key}")
    for key, expected in runtime["dependencies"].items():
        path = ROOT / key
        if not path.exists() or sha(path) != expected:
            raise ValueError(f"Frozen runtime dependency changed: {key}")
    runtime_hash = digest(runtime)

    calls = {}
    returned_models = Counter()
    costs = {}
    for path in sorted((folder / "calls").glob("*.json")):
        row = load(path)
        if row["scope"] in calls or digest(row["request"]) != row["request_sha256"]:
            raise ValueError("Duplicate scope or invalid API request hash")
        if row["status"] != "complete":
            raise ValueError("Incomplete API call in final cohort")
        if row["response"]["model"] != runtime["model_returned"]:
            raise ValueError("Returned API model changed")
        calls[row["scope"]] = row
        returned_models[row["response"]["model"]] += 1
        stage = row["scope"].split("-")[0]
        cost = costs.setdefault(stage, {
            "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "unknown_usage_calls": 0,
        })
        cost["calls"] += 1
        usage = row["response"].get("usage", {})
        if not all(name in usage for name in ("prompt_tokens", "completion_tokens")):
            cost["unknown_usage_calls"] += 1
        for name in ("prompt_tokens", "completion_tokens"):
            cost[name] += usage.get(name, 0)
    if len(calls) != 181:
        raise ValueError("Expected exactly 181 journaled calls")
    probe = calls.get("probe")
    if not probe or probe["request"] != request_body("Reply exactly OK.", 0, 16):
        raise ValueError("Probe request changed")

    calculator = SelfInformationModel() if recompute_self_information else None
    evaluated = {}
    mask_fractions = []
    mask_value_mismatches = 0
    for pid, record in records.items():
        baseline_path = ROOT / record["baseline_path"]
        baseline = json.loads(baseline_path.read_text())
        if (sha(baseline_path) != record["baseline_sha256"]
                or baseline["text"] != record["source"]
                or baseline["text_sha256"] != record["source_sha256"]):
            raise ValueError("Frozen baseline changed")

        reference = load(folder / "references" / f"{pid}.json")
        reference_call = calls.get(f"reference-{pid}")
        expected_reference = request_body(
            REFERENCE_PROMPT + record["source"], 0, 256,
            system="You are a helpful rewriter.")
        if (reference["runtime_sha256"] != runtime_hash
                or reference["source_sha256"] != record["source_sha256"]
                or reference["reference_sha256"] != text_sha(reference["reference"])
                or reference_call is None or reference_call["request"] != expected_reference
                or reference_call["response"]["content"].strip() != reference["reference"]):
            raise ValueError("Reference generation mismatch")

        saved_mask = load(folder / "masks" / f"{pid}.json")
        if (saved_mask["runtime_sha256"] != runtime_hash
                or saved_mask["source_sha256"] != record["source_sha256"]
                or saved_mask["token_count"] != len(saved_mask["tokens"])
                or saved_mask["token_count"] != len(saved_mask["token_ids"])
                or saved_mask["token_count"] != len(saved_mask["self_information"])):
            raise ValueError("Mask identity or vector length mismatch")
        rebuilt = mask_tokens(
            saved_mask["tokens"], saved_mask["self_information"], MASK_PERCENTILE)
        for key in ("blank_text", "percentile", "mask", "masked_tokens", "token_count"):
            if saved_mask[key] != rebuilt[key]:
                raise ValueError("Saved mask does not follow frozen transform")
        if not math.isclose(saved_mask["cutoff"], rebuilt["cutoff"], abs_tol=1e-10):
            raise ValueError("Mask percentile cutoff mismatch")
        mask_fractions.append(saved_mask["masked_tokens"] / saved_mask["token_count"])
        if calculator:
            fresh = calculator.calculate(record["source"])
            if (fresh["token_ids"] != saved_mask["token_ids"]
                    or fresh["tokens"] != saved_mask["tokens"]
                    or fresh["mask"] != saved_mask["mask"]
                    or fresh["blank_text"] != saved_mask["blank_text"]):
                raise ValueError("Fresh self-information reconstruction changed mask")
            mask_value_mismatches += sum(
                not math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5)
                for a, b in zip(fresh["self_information"], saved_mask["self_information"])
            )

        attacks = []
        for candidate in range(K):
            attack = load(folder / "attacks" / f"{pid}-{candidate}.json")
            prompt = (
                ATTACK_PROMPT + f"reference: {reference['reference']}\n"
                f"incomplete pragraph: {saved_mask['blank_text']}\n"
                f"Candidate variant: {candidate}"
            )
            call = calls.get(f"attack-{pid}-{candidate}")
            if (attack["prompt_id"] != pid or attack["candidate"] != candidate
                    or attack["runtime_sha256"] != runtime_hash
                    or attack["source"] != record["source"]
                    or attack["source_sha256"] != record["source_sha256"]
                    or attack["reference_sha256"] != reference["reference_sha256"]
                    or attack["mask_sha256"] != digest(saved_mask)
                    or attack["attack_sha256"] != text_sha(attack["attack"])
                    or call is None or call["request"] != request_body(prompt, .7, 256)
                    or call["response"]["content"].strip() != attack["attack"]):
                raise ValueError("Attack generation mismatch")
            attacks.append(attack)

        quality = load(folder / "quality" / f"{pid}.json")
        if (quality["group_sha256"] != digest(attacks)
                or quality["runtime_sha256"] != runtime_hash):
            raise ValueError("Quality group mismatch")
        review_prompt = REVIEW + "SOURCE:\n" + record["source"] + "\n\n" + "\n\n".join(
            f'CANDIDATE {row["candidate"]}:\n{row["attack"]}' for row in attacks)
        review_call = calls.get(f"review-{pid}")
        if review_call is None or review_call["request"] != request_body(review_prompt, 0, 700):
            raise ValueError("Quality review request mismatch")
        reviews = decode_reviews(review_call["response"]["content"])
        if len(quality["rows"]) != K:
            raise ValueError("Quality candidate budget mismatch")
        for attack, row in zip(attacks, quality["rows"]):
            if any(row[key] != value for key, value in attack.items()):
                raise ValueError("Quality row changed generated candidate")
            surface = gates(row, [detail["amr"] for detail in row["details"]])
            valid = (all(surface.values()) and row["review"]["score"] >= 4
                     and not row["review"]["major_error"]
                     and row["finish_reason"] == "stop")
            if (row["surface"] != surface or row["review"] != reviews[row["candidate"]]
                    or row["quality_valid"] != valid):
                raise ValueError("Quality label reconstruction failed")
        first_quality = next(
            (row["candidate"] for row in quality["rows"] if row["quality_valid"]), None)
        if quality["no_feedback_candidate"] != first_quality:
            raise ValueError("Quality-only selection rule mismatch")

        result = load(folder / "evaluated" / f"{pid}.json")
        if (result["quality_sha256"] != digest(quality)
                or result["runtime_sha256"] != runtime_hash
                or len(result["rows"]) != K):
            raise ValueError("Evaluated group mismatch")
        for scored, labeled in zip(result["rows"], quality["rows"]):
            if {key: value for key, value in scored.items() if key != "detection"} != labeled:
                raise ValueError("Quality labels changed after detection")
            verify_score(scored["detection"])
            if [{key: value for key, value in detail.items()
                 if key not in ("score", "bank_scores")}
                    for detail in scored["detection"]["details"]] != scored["details"]:
                raise ValueError("Detector did not reuse quality-stage AMR")
        eligible = [
            row for row in result["rows"] if row["quality_valid"] and row["detection"]]
        best = min(
            eligible, key=lambda row: (row["detection"]["z"], row["candidate"])
        ) if eligible else None
        first = next(
            (row for row in eligible if row["candidate"] == first_quality), None)
        if (result["feedback_candidate"] != (best["candidate"] if best else None)
                or result["feedback_success"] != bool(
                    best and best["detection"]["z"] <= 2.33)
                or result["no_feedback_candidate"] != first_quality
                or result["no_feedback_success"] != bool(
                    first and first["detection"]["z"] <= 2.33)):
            raise ValueError("Final selection or success label mismatch")
        evaluated[pid] = result

    ids = sorted(records)
    if set(evaluated) != set(records):
        raise ValueError("Incomplete final cohort")
    if set(calls) != ({"probe"}
                      | {f"reference-{pid}" for pid in ids}
                      | {f"attack-{pid}-{candidate}" for pid in ids for candidate in range(K)}
                      | {f"review-{pid}" for pid in ids}):
        raise ValueError("Unexpected API scope in journal")

    rows = [row for result in evaluated.values() for row in result["rows"]]
    quality_candidates = [row for row in rows if row["quality_valid"]]
    feedback = [evaluated[pid]["feedback_success"] for pid in ids]
    no_feedback = [evaluated[pid]["no_feedback_success"] for pid in ids]
    detected_ids = {
        pid for pid, record in records.items()
        if record["baseline_detection"]["z"] > 2.33
    }
    conditional_feedback = [
        evaluated[pid]["feedback_success"] for pid in ids if pid in detected_ids]
    quality_paragraphs = sum(
        any(row["quality_valid"] for row in evaluated[pid]["rows"]) for pid in ids)
    comparisons = {}
    confirmation = ROOT / "experiments/confirmation_three_arm30/evaluated"
    for arm in ARMS:
        other = []
        for pid in ids:
            path = confirmation / f"{pid}-{arm}.json"
            inputs[f"comparison/{path.name}"] = sha(path)
            other.append(json.loads(path.read_text())["feedback_success"])
        comparisons[f"{arm}_vs_sira_adapted"] = comparison(
            arm, other, "sira_adapted", feedback)

    return {
        "status": "complete_independently_verified",
        "method": "SIRA-adapted; not official model-equivalent reproduction",
        "coverage": {
            "paragraphs": len(ids), "candidates": len(rows),
            "quality_candidates": len(quality_candidates),
            "quality_paragraphs": quality_paragraphs,
            "api_calls": len(calls),
        },
        "candidate_rates": {
            "quality_pass": rate_summary(len(quality_candidates), len(rows)),
            "all_candidate_low_z": rate_summary(
                sum(row["detection"]["z"] <= 2.33 for row in rows), len(rows)),
            "quality_and_low_z": rate_summary(
                sum(row["quality_valid"] and row["detection"]["z"] <= 2.33 for row in rows),
                len(rows)),
        },
        "paragraph_rates": {
            "quality_coverage": rate_summary(quality_paragraphs, len(ids)),
            "feedback_valid_asr": rate_summary(sum(feedback), len(ids)),
            "no_feedback_valid_asr": rate_summary(sum(no_feedback), len(ids)),
            "conditional_feedback_valid_asr": rate_summary(
                sum(conditional_feedback), len(conditional_feedback)),
        },
        "mask_statistics": {
            "percentile": MASK_PERCENTILE,
            "mean_mask_fraction": sum(mask_fractions) / len(mask_fractions),
            "min_mask_fraction": min(mask_fractions),
            "max_mask_fraction": max(mask_fractions),
            "fresh_value_mismatches_at_1e_5": mask_value_mismatches,
            "fresh_masks_recomputed": bool(calculator),
        },
        "paired_comparisons": comparisons,
        "api_costs": costs,
        "returned_models": dict(returned_models),
        "notes": [
            "SIRA-adapted uses Qwen2.5-0.5B for self-information and DeepSeek for rewriting.",
            "Official SIRA Llama-3-8B results are not reproduced or claimed.",
            "Paired p-values are exact two-sided McNemar tests without multiplicity correction.",
            "Conditional denominator contains only baselines with original z > 2.33.",
        ],
        "input_sha256": inputs,
    }


if __name__ == "__main__":
    folder = ROOT / "experiments/sira_adapted30"
    result = audit(folder)
    write_json(folder / "audit.json", result)
    print(json.dumps({
        key: value for key, value in result.items() if key != "input_sha256"
    }, indent=2))
