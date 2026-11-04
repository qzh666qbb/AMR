"""Independently audit the completed KGW/SynthID transfer experiment."""

import argparse
import json
import math
from pathlib import Path

from analyze_e5_transfer import exact_ci
from evaluate_adaptive_retry import gates
from run_confirmation_experiment import digest
from run_cross_watermark_transfer import (
    K,
    ROOT,
    WATERMARKS,
    WatermarkRuntime,
    calibrated_threshold,
)
from run_segmentation_intervention import sha, text_sha, write_json


def rate(count, total):
    return {
        "count": count,
        "n": total,
        "rate": count / total,
        "clopper_pearson_95": exact_ci(count, total),
    }


def paired_test(left, right):
    if set(left) != set(right) or not left:
        raise ValueError("Paired outcomes are missing or misaligned")
    left_only = sum(left[key] and not right[key] for key in left)
    right_only = sum(right[key] and not left[key] for key in left)
    discordant = left_only + right_only
    p_value = (
        min(
            1.0,
            2
            * sum(
                math.comb(discordant, index)
                for index in range(min(left_only, right_only) + 1)
            )
            / (2 ** discordant),
        )
        if discordant
        else 1.0
    )
    return {
        "n": len(left),
        "both_success": sum(left[key] and right[key] for key in left),
        "kgw_only_success": left_only,
        "synthid_only_success": right_only,
        "neither_success": sum(
            not left[key] and not right[key] for key in left
        ),
        "mcnemar_exact_two_sided_p": p_value,
    }


def audit(folder):
    manifest_path = folder / "manifest.json"
    thresholds_path = folder / "thresholds.json"
    summary_path = folder / "execution_summary.json"
    manifest = json.loads(manifest_path.read_text())
    thresholds = json.loads(thresholds_path.read_text())
    summary = json.loads(summary_path.read_text())
    if summary["status"] != "complete":
        raise ValueError("Cross-watermark execution is not complete")
    if len(manifest["prompts"]) != 30:
        raise ValueError("Expected 30 frozen prompts")

    hashes = {
        str(path.relative_to(ROOT)): sha(path)
        for path in (manifest_path, thresholds_path, summary_path, Path(__file__))
    }
    runtime = WatermarkRuntime(load_model=False)
    outcomes = {}
    audited = {}

    for watermark in WATERMARKS:
        calibration_paths = sorted(
            (folder / "calibration" / watermark).glob("*.json")
        )
        calibration_rows = [
            json.loads(path.read_text()) for path in calibration_paths
        ]
        if len(calibration_rows) != 500:
            raise ValueError(f"Expected 500 {watermark} human scores")
        calibration = [
            row["detection"]["score"]
            for row in calibration_rows
            if row["cohort"] == "human_calibration"
        ]
        test = [
            row["detection"]["score"]
            for row in calibration_rows
            if row["cohort"] == "human_test"
        ]
        threshold = calibrated_threshold(calibration)
        if threshold != thresholds[watermark]["threshold"]:
            raise ValueError("Stored threshold differs from calibration")
        if sum(score > threshold for score in calibration) != (
            thresholds[watermark]["calibration_false_positives"]
        ):
            raise ValueError("Calibration FPR mismatch")
        if sum(score > threshold for score in test) != (
            thresholds[watermark]["test_false_positives"]
        ):
            raise ValueError("Test FPR mismatch")

        baseline_paths = sorted(
            (folder / "baselines" / watermark).glob("*.json")
        )
        quality_paths = sorted(
            (folder / "quality" / watermark).glob("*.json")
        )
        evaluated_paths = sorted(
            (folder / "evaluated" / watermark).glob("*.json")
        )
        if not (
            len(baseline_paths)
            == len(quality_paths)
            == len(evaluated_paths)
            == 30
        ):
            raise ValueError(f"Incomplete {watermark} experiment artifacts")
        baseline_by_id = {
            row["prompt_id"]: row
            for row in (json.loads(path.read_text()) for path in baseline_paths)
        }
        quality_by_id = {
            row["prompt_id"]: row
            for row in (json.loads(path.read_text()) for path in quality_paths)
        }
        evaluated_by_id = {
            row["prompt_id"]: row
            for row in (
                json.loads(path.read_text()) for path in evaluated_paths
            )
        }
        prompt_ids = [row["prompt_id"] for row in manifest["prompts"]]
        if (
            set(baseline_by_id)
            != set(quality_by_id)
            or set(quality_by_id) != set(evaluated_by_id)
            or set(evaluated_by_id) != set(prompt_ids)
        ):
            raise ValueError(f"Misaligned {watermark} prompt identities")

        baseline_score_mismatches = 0
        attack_score_mismatches = 0
        quality_mismatches = 0
        candidate_count = 0
        quality_candidates = 0
        low_quality_candidates = 0
        successes = {}
        baseline_detected = 0
        quality_coverage = 0

        for prompt_id in prompt_ids:
            baseline = baseline_by_id[prompt_id]
            if text_sha(baseline["text"]) != baseline["text_sha256"]:
                raise ValueError("Baseline text hash mismatch")
            fresh_baseline = runtime.score(watermark, baseline["text"])
            if not math.isclose(
                fresh_baseline["score"],
                baseline["detection"]["score"],
                abs_tol=1e-12,
            ):
                baseline_score_mismatches += 1
            detected = baseline["detection"]["score"] > threshold
            if detected != baseline["detected"]:
                raise ValueError("Baseline detection label mismatch")
            baseline_detected += detected

            quality = quality_by_id[prompt_id]
            evaluated = evaluated_by_id[prompt_id]
            if quality.get("target_scores_read") is not False:
                raise ValueError("Quality artifact is not explicitly blind")
            if any(
                "target_detection" in row or "detection" in row
                for row in quality["rows"]
            ):
                raise ValueError("Target score leaked into quality selection")
            if evaluated["quality_sha256"] != digest(quality):
                raise ValueError("Evaluated artifact does not bind quality file")
            expected_selected = next(
                (
                    row["candidate"]
                    for row in quality["rows"]
                    if row["quality_valid"]
                ),
                None,
            )
            if expected_selected != quality["no_feedback_candidate"]:
                raise ValueError("No-feedback selection is not first valid")
            quality_coverage += expected_selected is not None

            evaluated_rows = {
                row["candidate"]: row for row in evaluated["rows"]
            }
            if set(evaluated_rows) != set(range(K)):
                raise ValueError("Candidate identities are incomplete")
            for row in quality["rows"]:
                candidate_count += 1
                surface = gates(
                    row, [detail["amr"] for detail in row["details"]]
                )
                expected_valid = (
                    all(surface.values())
                    and row["review"]["score"] >= 4
                    and not row["review"]["major_error"]
                    and row["finish_reason"] == "stop"
                )
                if (
                    surface != row["surface"]
                    or expected_valid != row["quality_valid"]
                ):
                    quality_mismatches += 1
                quality_candidates += expected_valid
                scored = evaluated_rows[row["candidate"]]
                fresh = runtime.score(watermark, row["attack"])
                if not math.isclose(
                    fresh["score"],
                    scored["target_detection"]["score"],
                    abs_tol=1e-12,
                ):
                    attack_score_mismatches += 1
                low_quality_candidates += (
                    expected_valid
                    and fresh["score"] <= threshold
                )
            selected = (
                evaluated_rows[expected_selected]
                if expected_selected is not None else None
            )
            expected_success = bool(
                selected
                and selected["quality_valid"]
                and selected["target_detection"]["score"] <= threshold
            )
            if expected_success != evaluated["no_feedback_success"]:
                raise ValueError("No-feedback success label mismatch")
            successes[prompt_id] = expected_success

        if (
            baseline_score_mismatches
            or attack_score_mismatches
            or quality_mismatches
        ):
            raise ValueError(f"Independent {watermark} recomputation mismatch")
        outcomes[watermark] = successes
        audited[watermark] = {
            "threshold": threshold,
            "calibration_fpr": rate(
                sum(score > threshold for score in calibration),
                len(calibration),
            ),
            "test_fpr": rate(
                sum(score > threshold for score in test), len(test)
            ),
            "baseline_detection": rate(baseline_detected, 30),
            "quality_coverage": rate(quality_coverage, 30),
            "quality_candidates": rate(quality_candidates, candidate_count),
            "quality_valid_candidate_low_score": rate(
                low_quality_candidates, quality_candidates
            ),
            "no_feedback_valid_asr": rate(sum(successes.values()), 30),
            "recomputation": {
                "baseline_score_mismatches": baseline_score_mismatches,
                "attack_score_mismatches": attack_score_mismatches,
                "quality_mismatches": quality_mismatches,
            },
        }
        for path in (
            calibration_paths + baseline_paths + quality_paths + evaluated_paths
        ):
            hashes[str(path.relative_to(ROOT))] = sha(path)

    call_paths = sorted((folder / "calls").glob("*.json"))
    calls = [json.loads(path.read_text()) for path in call_paths]
    if len(calls) != 30 * len(WATERMARKS) * (K + 1):
        raise ValueError("Expected exactly 300 attack and review calls")
    if any(row["status"] != "complete" for row in calls):
        raise ValueError("Incomplete or uncertain API call")
    if any(row["request_sha256"] != digest(row["request"]) for row in calls):
        raise ValueError("API request hash mismatch")
    if len({row["scope"] for row in calls}) != len(calls):
        raise ValueError("Duplicate API call scope")
    for path in call_paths:
        hashes[str(path.relative_to(ROOT))] = sha(path)

    result = {
        "status": "complete_independently_verified",
        "protocol": (
            "No-feedback candidate identity fixed by quality before target "
            "scores; target thresholds calibrated on independent human text"
        ),
        "watermarks": audited,
        "paired_success": paired_test(
            outcomes["kgw"], outcomes["synthid"]
        ),
        "api": {
            "calls": len(calls),
            "all_complete": True,
            "all_request_hashes_verified": True,
            "unique_scopes": len({row["scope"] for row in calls}),
        },
        "input_sha256": hashes,
    }
    write_json(folder / "audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folder",
        type=Path,
        default=ROOT / "experiments/cross_watermark_transfer30",
    )
    args = parser.parse_args()
    print(json.dumps(audit(args.folder), indent=2))


if __name__ == "__main__":
    main()
