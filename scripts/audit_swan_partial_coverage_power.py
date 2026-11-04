"""Independent audit and recovery guard for partial-coverage power results."""

import argparse
import json
import math
from pathlib import Path

from run_confirmation_experiment import digest
from run_segmentation_intervention import sha, write_json
import run_swan_partial_coverage_power as experiment


def independent_z(green, units, null_rate):
    variance = units * null_rate * (1.0 - null_rate)
    return (green - units * null_rate) / math.sqrt(variance)


def independent_minimum(units, null_rate, threshold):
    passing = [
        green
        for green in range(units + 1)
        if independent_z(green, units, null_rate) > threshold
    ]
    return min(passing) if passing else None


def independent_probability(n, k, probability):
    return (
        math.factorial(n)
        / (math.factorial(k) * math.factorial(n - k))
        * probability ** k
        * (1.0 - probability) ** (n - k)
    )


def independent_tail(slots, target_probability, background_probability, required):
    probability = 0.0
    for target_green in range(slots + 1):
        target_mass = independent_probability(
            slots,
            target_green,
            target_probability,
        )
        for background_green in range(experiment.DENOMINATOR - slots + 1):
            if target_green + background_green >= required:
                probability += target_mass * independent_probability(
                    experiment.DENOMINATOR - slots,
                    background_green,
                    background_probability,
                )
    return probability


def close(left, right):
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def audit_rule(rule, documents):
    required = independent_minimum(
        experiment.DENOMINATOR,
        rule["lambda"],
        rule["threshold"],
    )
    if required != rule["minimum_total_green"]:
        raise ValueError("Minimum total green mismatch")
    expected_null = independent_tail(
        0,
        0.0,
        rule["lambda"],
        required,
    )
    if not close(expected_null, rule["null_binomial_tail"]):
        raise ValueError("Null tail mismatch")
    by_slots = {
        row["candidate_slots"]: row
        for row in rule["slot_sensitivity"]
    }
    if set(by_slots) != set(range(7)):
        raise ValueError("Slot sensitivity range mismatch")
    for slots, row in by_slots.items():
        if (
            row["required_total_green"] != required
            or row["required_background_green_if_all_targets_green"]
            != max(0, required - slots)
            or row["all_targets_green_zero_background_detected"]
            != (slots >= required)
        ):
            raise ValueError("Deterministic slot field mismatch")
        for probability in experiment.TARGET_PROBABILITIES:
            key = experiment.probability_key(probability)
            expected = independent_tail(
                slots,
                probability,
                rule["lambda"],
                required,
            )
            if not close(
                expected,
                row["modeled_detection_probability"][key],
            ):
                raise ValueError("Modeled detection probability mismatch")
    saved_documents = rule["quality_cohort"]["documents"]
    if [
        {
            "prompt_id": row["prompt_id"],
            "short_slots": row["short_slots"],
        }
        for row in saved_documents
    ] != documents:
        raise ValueError("Quality cohort identity mismatch")
    guaranteed = 0
    expected_totals = {
        experiment.probability_key(probability): 0.0
        for probability in experiment.TARGET_PROBABILITIES
    }
    for document in saved_documents:
        slot_row = by_slots[document["short_slots"]]
        if (
            document["all_targets_green_zero_background_detected"]
            != slot_row["all_targets_green_zero_background_detected"]
            or document["modeled_detection_probability"]
            != slot_row["modeled_detection_probability"]
        ):
            raise ValueError("Document power mapping mismatch")
        guaranteed += document[
            "all_targets_green_zero_background_detected"
        ]
        for key, value in document[
            "modeled_detection_probability"
        ].items():
            expected_totals[key] += value
    if (
        guaranteed
        != rule["quality_cohort"]["guaranteed_detectable_documents"]
        or rule["quality_cohort"]["guaranteed_gate_pass"]
        != (guaranteed >= experiment.MIN_DOCUMENTS)
    ):
        raise ValueError("Guaranteed cohort gate mismatch")
    for key, expected in expected_totals.items():
        if not close(
            expected,
            rule["quality_cohort"][
                "modeled_expected_detected_documents"
            ][key],
        ):
            raise ValueError("Expected cohort detection mismatch")
    return len(by_slots) * len(experiment.TARGET_PROBABILITIES)


def audit(folder=experiment.OUTPUT):
    manifest = experiment.check(folder)
    summary = json.loads((folder / "summary.json").read_text())
    documents = manifest["inputs"]["quality_documents"]
    probabilities = 0
    for key in ("fixed_primary", "calibrated_sensitivity"):
        probabilities += audit_rule(summary["rules"][key], documents)
    fixed_gate = summary["rules"]["fixed_primary"][
        "quality_cohort"
    ]["guaranteed_gate_pass"]
    calibrated_gate = summary["rules"]["calibrated_sensitivity"][
        "quality_cohort"
    ]["guaranteed_gate_pass"]
    expected_fixed = (
        "supports_fixed_partial_coverage_injection_preflight"
        if fixed_gate
        else "stop_fixed_partial_coverage_route"
    )
    expected_calibrated = (
        "requires_new_bank100_null_validation"
        if calibrated_gate
        else "bank100_partial_coverage_not_algebraically_supported"
    )
    if (
        summary["decisions"]["fixed_primary"] != expected_fixed
        or summary["decisions"]["calibrated_sensitivity"]
        != expected_calibrated
        or any(summary["resource_counts"].values())
    ):
        raise ValueError("Decision or resource accounting mismatch")
    result = {
        "status": "passed",
        "mismatches": 0,
        "quality_documents_reconstructed": len(documents),
        "slot_levels_reconstructed": 14,
        "modeled_probabilities_reconstructed": probabilities,
        "integer_thresholds_reconstructed": 2,
        "new_api_calls": 0,
        "new_neural_parser_calls": 0,
        "new_s2match_calls": 0,
        "summary_sha256": sha(folder / "summary.json"),
    }
    write_json(folder / "audit.json", result)
    return result


def stable_files(folder):
    return [
        folder / "manifest.json",
        folder / "summary.json",
        folder / "report.md",
        folder / "audit.json",
    ]


def recovery(folder=experiment.OUTPUT):
    experiment.check(folder)
    before = {
        str(path.relative_to(folder)): sha(path)
        for path in stable_files(folder)
    }
    aggregate = digest(before)
    import socket

    original_connection = socket.create_connection

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Recovery attempted network access")

    socket.create_connection = forbidden
    try:
        result = experiment.run(folder)
    finally:
        socket.create_connection = original_connection
    after = {
        str(path.relative_to(folder)): sha(path)
        for path in stable_files(folder)
    }
    if before != after or aggregate != digest(after):
        raise ValueError("Recovery changed stable artifacts")
    verification = {
        "status": "complete_no_compute_recovery",
        "decision": result["decisions"]["overall"],
        "files_checked": len(before),
        "stable_aggregate_sha256": aggregate,
        "network_forbidden": True,
        "api_not_imported": True,
        "neural_parser_not_imported": True,
        "s2match_not_imported": True,
        "new_api_calls": 0,
        "new_neural_parser_calls": 0,
        "new_s2match_calls": 0,
    }
    write_json(folder / "recovery_verification.json", verification)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "recovery"))
    parser.add_argument("--output", type=Path, default=experiment.OUTPUT)
    args = parser.parse_args()
    result = audit(args.output) if args.stage == "audit" else recovery(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
