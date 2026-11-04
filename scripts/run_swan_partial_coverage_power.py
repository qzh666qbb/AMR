"""Offline detector-power audit for partial SWAN slot coverage."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from run_confirmation_experiment import freeze
from run_segmentation_intervention import ROOT, sha, write_json


OUTPUT = ROOT / "experiments/swan_partial_coverage_power1"
QUALITY = ROOT / "experiments/swan_quality_first_windows4"
ORACLE = ROOT / "experiments/swan_oracle_composition1"
CALIBRATION = ROOT / "experiments/swan_bank100_null_calibration1"
DENOMINATOR = 12
TARGET_PROBABILITIES = (0.8, 0.9, 147 / 150, 1.0)
MIN_DOCUMENTS = 3


def z_score(green, units, null_rate):
    return (
        (green - null_rate * units)
        / math.sqrt(units * null_rate * (1 - null_rate))
    )


def minimum_green(units, null_rate, threshold):
    for green in range(units + 1):
        if z_score(green, units, null_rate) > threshold:
            return green
    return None


def binomial_pmf(k, n, probability):
    if k < 0 or k > n:
        return 0.0
    return (
        math.comb(n, k)
        * probability ** k
        * (1 - probability) ** (n - k)
    )


def convolution_tail(
    target_units,
    background_units,
    target_probability,
    background_probability,
    minimum_total_green,
):
    total = 0.0
    for target_green in range(target_units + 1):
        for background_green in range(background_units + 1):
            if target_green + background_green < minimum_total_green:
                continue
            total += (
                binomial_pmf(
                    target_green,
                    target_units,
                    target_probability,
                )
                * binomial_pmf(
                    background_green,
                    background_units,
                    background_probability,
                )
            )
    return total


def dependency_paths():
    paths = {
        Path(__file__),
        ROOT / "scripts/audit_swan_partial_coverage_power.py",
        OUTPUT / "protocol.md",
    }
    for folder in (QUALITY, ORACLE, CALIBRATION):
        paths.update(
            {
                folder / "protocol.md",
                folder / "manifest.json",
                folder / "summary.json",
                folder / "audit.json",
                folder / "recovery_verification.json",
            }
        )
    return sorted(paths)


def dependencies():
    return {
        str(path.relative_to(ROOT)): sha(path)
        for path in dependency_paths()
    }


def input_snapshot():
    quality = json.loads((QUALITY / "summary.json").read_text())
    oracle = json.loads((ORACLE / "summary.json").read_text())
    calibration = json.loads((CALIBRATION / "summary.json").read_text())
    documents = [
        {
            "prompt_id": row["prompt_id"],
            "short_slots": row["short_slots"],
        }
        for row in quality["documents"]
        if row["quality_pass"]
    ]
    original_units = sum(
        row["conditions"]["original"]["n"]
        for row in oracle["groups"]
    )
    original_green = sum(
        row["conditions"]["original"]["green"]
        for row in oracle["groups"]
    )
    return {
        "quality_documents": documents,
        "quality_document_count": len(documents),
        "quality_manifest_sha256": quality["manifest_sha256"],
        "oracle_original_units": original_units,
        "oracle_original_green": original_green,
        "oracle_target_green_rate": original_green / original_units,
        "calibrated_lambda": calibration["calibration"]["lambda_hat"],
        "calibrated_threshold": calibration["calibration"][
            "document_z_threshold"
        ],
        "calibration_human_test_fpr": calibration["calibration"][
            "calibrated"
        ]["human_test"]["rate"],
        "calibration_machine_quality_documents": calibration[
            "machine_quality_pass"
        ],
        "calibration_machine_quality_gate": 18,
    }


def prepare(folder=OUTPUT):
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        return check(folder)
    inputs = input_snapshot()
    if (
        inputs["quality_document_count"] != 14
        or inputs["oracle_original_units"] != 150
        or inputs["oracle_original_green"] != 147
        or inputs["calibration_machine_quality_documents"] >= inputs[
            "calibration_machine_quality_gate"
        ]
    ):
        raise ValueError("Frozen input assumptions changed")
    manifest = {
        "status": "frozen_before_power_computation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "workflow": "swan_partial_coverage_detector_power_audit",
            "denominator": DENOMINATOR,
            "target_probabilities": list(TARGET_PROBABILITIES),
            "minimum_documents": MIN_DOCUMENTS,
            "fixed_primary": {
                "bank": "bank50 historical reference",
                "lambda": 0.25,
                "threshold": 2.33,
                "strict_greater_than": True,
            },
            "calibrated_sensitivity": {
                "bank": "bank100",
                "lambda": inputs["calibrated_lambda"],
                "threshold": inputs["calibrated_threshold"],
                "strict_greater_than": True,
                "cannot_authorize_injection": True,
            },
            "no_api_parser_s2match_injection_detection_or_attack": True,
        },
        "inputs": inputs,
        "dependencies": dependencies(),
    }
    freeze(manifest_path, manifest)
    return manifest


def check(folder=OUTPUT):
    manifest = json.loads((folder / "manifest.json").read_text())
    if (
        manifest["protocol"]["workflow"]
        != "swan_partial_coverage_detector_power_audit"
        or manifest["protocol"]["denominator"] != DENOMINATOR
        or manifest["inputs"] != input_snapshot()
    ):
        raise ValueError("Frozen manifest inputs changed")
    for relative, expected in manifest["dependencies"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Frozen dependency changed: {relative}")
    return manifest


def probability_key(value):
    if math.isclose(value, 147 / 150):
        return "historical_bank50_147_of_150"
    return f"target_p_{value:.2f}"


def analyze_rule(name, rule, quality_documents):
    null_rate = rule["lambda"]
    threshold = rule["threshold"]
    required = minimum_green(DENOMINATOR, null_rate, threshold)
    null_tail = convolution_tail(
        0,
        DENOMINATOR,
        0.0,
        null_rate,
        required,
    )
    slot_rows = []
    for slots in range(7):
        probabilities = {
            probability_key(probability): convolution_tail(
                slots,
                DENOMINATOR - slots,
                probability,
                null_rate,
                required,
            )
            for probability in TARGET_PROBABILITIES
        }
        slot_rows.append(
            {
                "candidate_slots": slots,
                "required_total_green": required,
                "required_background_green_if_all_targets_green": max(
                    0,
                    required - slots,
                ),
                "all_targets_green_zero_background_detected": slots >= required,
                "modeled_detection_probability": probabilities,
            }
        )
    rows_by_slots = {
        row["candidate_slots"]: row for row in slot_rows
    }
    documents = []
    for document in quality_documents:
        slots = document["short_slots"]
        values = rows_by_slots[slots]
        documents.append(
            {
                **document,
                "all_targets_green_zero_background_detected": values[
                    "all_targets_green_zero_background_detected"
                ],
                "modeled_detection_probability": values[
                    "modeled_detection_probability"
                ],
            }
        )
    guaranteed = sum(
        row["all_targets_green_zero_background_detected"]
        for row in documents
    )
    expected = {
        probability_key(probability): sum(
            row["modeled_detection_probability"][
                probability_key(probability)
            ]
            for row in documents
        )
        for probability in TARGET_PROBABILITIES
    }
    return {
        "name": name,
        "lambda": null_rate,
        "threshold": threshold,
        "denominator": DENOMINATOR,
        "minimum_total_green": required,
        "null_binomial_tail": null_tail,
        "slot_sensitivity": slot_rows,
        "quality_cohort": {
            "documents": documents,
            "guaranteed_detectable_documents": guaranteed,
            "minimum_required_documents": MIN_DOCUMENTS,
            "guaranteed_gate_pass": guaranteed >= MIN_DOCUMENTS,
            "modeled_expected_detected_documents": expected,
        },
    }


def run(folder=OUTPUT):
    manifest = check(folder)
    fixed = analyze_rule(
        "fixed_bank50_primary",
        manifest["protocol"]["fixed_primary"],
        manifest["inputs"]["quality_documents"],
    )
    calibrated = analyze_rule(
        "bank100_calibrated_sensitivity",
        manifest["protocol"]["calibrated_sensitivity"],
        manifest["inputs"]["quality_documents"],
    )
    fixed_decision = (
        "supports_fixed_partial_coverage_injection_preflight"
        if fixed["quality_cohort"]["guaranteed_gate_pass"]
        else "stop_fixed_partial_coverage_route"
    )
    calibrated_decision = (
        "requires_new_bank100_null_validation"
        if calibrated["quality_cohort"]["guaranteed_gate_pass"]
        else "bank100_partial_coverage_not_algebraically_supported"
    )
    summary = {
        "status": "complete_offline_power_audit",
        "inputs": manifest["inputs"],
        "rules": {
            "fixed_primary": fixed,
            "calibrated_sensitivity": calibrated,
        },
        "decisions": {
            "fixed_primary": fixed_decision,
            "calibrated_sensitivity": calibrated_decision,
            "overall": (
                f"{fixed_decision};{calibrated_decision}"
            ),
        },
        "resource_counts": {
            "new_api_calls": 0,
            "new_neural_parser_calls": 0,
            "new_s2match_calls": 0,
            "new_injection_detection_or_attack_calls": 0,
        },
        "limitations": [
            "Exact binomial sensitivities assume independent sentence units.",
            "The 147/150 target rate comes from low-quality bank50 assets.",
            "No bank100 targeted-green rate is available.",
            "The calibrated bank100 machine-quality continuation gate failed.",
            "No modeled probability is an empirical TPR estimate.",
        ],
        "manifest_sha256": sha(folder / "manifest.json"),
    }
    write_json(folder / "summary.json", summary)
    write_report(folder, summary)
    return summary


def write_report(folder, summary):
    fixed = summary["rules"]["fixed_primary"]
    calibrated = summary["rules"]["calibrated_sensitivity"]
    lines = [
        "# SWAN partial-coverage detector-power audit",
        "",
        "| Rule | Total green required in 12 | Quality documents guaranteed | "
        "Decision |",
        "|---|---:|---:|---|",
        f"| Fixed bank50 reference | {fixed['minimum_total_green']} | "
        f"{fixed['quality_cohort']['guaranteed_detectable_documents']}/14 | "
        f"`{summary['decisions']['fixed_primary']}` |",
        f"| Calibrated bank100 sensitivity | "
        f"{calibrated['minimum_total_green']} | "
        f"{calibrated['quality_cohort']['guaranteed_detectable_documents']}/14 "
        f"| `{summary['decisions']['calibrated_sensitivity']}` |",
        "",
        "All calculations retain the complete 12-sentence detector denominator.",
        "The bank100 row is sensitivity analysis and cannot authorize injection.",
        "No API, parser, S2Match, detector rerun, injection, or attack was run.",
        "",
    ]
    (folder / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = prepare(args.output) if args.stage == "prepare" else run(args.output)
    if args.stage == "prepare":
        result = {
            "status": "prepared",
            "manifest_sha256": sha(args.output / "manifest.json"),
            "quality_documents": result["inputs"]["quality_document_count"],
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
