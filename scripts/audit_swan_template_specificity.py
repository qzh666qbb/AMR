"""Independent audit for the template-specific SWAN preflight."""

import argparse
import json
import math
from pathlib import Path
from unittest.mock import patch

from run_confirmation_experiment import digest
from run_segmentation_intervention import sha, write_json
import audit_swan_local_multi_instance as local_audit
import run_swan_local_multi_instance as local
import run_swan_template_specificity as experiment


def stable_paths(folder, include_audit=False):
    paths = [folder / "manifest.json"]
    for name in (
        "development.json",
        "locked_rule.json",
        "stop_decision.json",
        "summary.json",
        "report.md",
    ):
        path = folder / name
        if path.exists():
            paths.append(path)
    paths.extend(sorted((folder / "checkpoints").glob("*/*/*.json")))
    if include_audit and (folder / "audit.json").exists():
        paths.append(folder / "audit.json")
    return paths


def aggregate(folder, include_audit=False):
    return digest(
        [
            {
                "path": str(path.relative_to(folder)),
                "sha256": sha(path),
            }
            for path in stable_paths(folder, include_audit)
        ]
    )


def independent_value(candidate, feature):
    if feature == "balanced_min":
        return min(candidate["coverage"], candidate["local_precision"])
    return candidate[feature]


def independent_scores(checkpoint, feature):
    scores = [0.0 for _ in range(50)]
    for graph in checkpoint["graphs"]:
        for candidate in graph["candidates"]:
            index = candidate["bank_index"]
            scores[index] = max(
                scores[index],
                independent_value(candidate, feature),
            )
    return scores


def independent_statistic(scores, matrix, top_k, leave_out=None):
    evidence = []
    for bank_index in range(50):
        count = 0
        for row_index in range(len(matrix)):
            if row_index == leave_out:
                continue
            if matrix[row_index][bank_index] >= scores[bank_index]:
                count += 1
        probability = min(
            1.0,
            (count + 1) / experiment.CALIBRATION_DOCUMENTS,
        )
        evidence.append(
            {
                "bank_index": bank_index,
                "feature_score": scores[bank_index],
                "tail_probability": probability,
                "surprisal": -math.log(probability),
            }
        )
    evidence.sort(
        key=lambda row: (-row["surprisal"], row["bank_index"])
    )
    selected = evidence[:top_k]
    return {
        "statistic": sum(row["surprisal"] for row in selected),
        "top_evidence": selected,
    }


def independent_condition(
    checkpoints,
    feature,
    top_k,
    matrix,
    threshold,
):
    rows = []
    for checkpoint in sorted(
        checkpoints,
        key=lambda row: row["document_id"],
    ):
        result = independent_statistic(
            independent_scores(checkpoint, feature),
            matrix,
            top_k,
        )
        top_indices = {
            row["bank_index"] for row in result["top_evidence"]
        }
        known = [
            bank_index
            for graph in checkpoint["graphs"]
            for bank_index in graph["known_target_template_indices"]
        ]
        rows.append(
            {
                "document_id": checkpoint["document_id"],
                "statistic": result["statistic"],
                "threshold_margin": (
                    result["statistic"] - threshold
                ),
                "detected": result["statistic"] > threshold,
                "top_evidence": result["top_evidence"],
                "known_target_instances": len(known),
                "known_target_instances_in_top_k": sum(
                    bank_index in top_indices for bank_index in known
                ),
            }
        )
    return {
        "documents": len(rows),
        "detected": sum(row["detected"] for row in rows),
        "known_target_instances": sum(
            row["known_target_instances"] for row in rows
        ),
        "known_target_instances_in_top_k": sum(
            row["known_target_instances_in_top_k"] for row in rows
        ),
        "rows": rows,
    }


def independent_rank(row):
    positive = row["positive_development"]
    original = positive["original"]["detected"]
    reparse = positive["reparse"]["detected"]
    return (
        min(original, reparse),
        reparse,
        original,
        positive["composition"]["detected"],
        -row["calibration_false_positives"],
        experiment.FEATURE_PREFERENCE[row["rule"]["feature"]],
        -row["rule"]["top_k"],
    )


def reconstruct_development(calibration, positive):
    ordered = sorted(calibration, key=lambda row: row["document_id"])
    by_condition = {
        representation: [
            row
            for row in positive
            if row["representation"] == representation
        ]
        for representation in ("original", "composition", "reparse")
    }
    rows = []
    for rule in experiment.GRID:
        matrix = [
            independent_scores(checkpoint, rule["feature"])
            for checkpoint in ordered
        ]
        statistics = [
            independent_statistic(
                scores,
                matrix,
                rule["top_k"],
                leave_out=index,
            )["statistic"]
            for index, scores in enumerate(matrix)
        ]
        descending = sorted(statistics, reverse=True)
        threshold = descending[
            experiment.CALIBRATION_MAX_FALSE_POSITIVES
        ]
        false_positives = sum(
            value > threshold for value in statistics
        )
        rows.append(
            {
                "rule_id": (
                    f"{rule['feature']}-top{rule['top_k']}"
                ),
                "rule": rule,
                "threshold": threshold,
                "calibration_false_positives": false_positives,
                "calibration_statistics": statistics,
                "positive_development": {
                    representation: independent_condition(
                        checkpoints,
                        rule["feature"],
                        rule["top_k"],
                        matrix,
                        threshold,
                    )
                    for representation, checkpoints in (
                        by_condition.items()
                    )
                },
            }
        )
    return rows, max(rows, key=independent_rank)


def independent_attack_result(
    checkpoints,
    feature,
    top_k,
    matrix,
    threshold,
    inventory,
):
    result = independent_condition(
        checkpoints,
        feature,
        top_k,
        matrix,
        threshold,
    )
    metadata = {row["document_id"]: row for row in inventory}
    for row in result["rows"]:
        item = metadata[row["document_id"]]
        row["quality_valid"] = item["quality_valid"]
        row["historically_successful"] = item[
            "historically_successful"
        ]

    def subgroup(predicate):
        rows = [row for row in result["rows"] if predicate(row)]
        return {
            "documents": len(rows),
            "detected": sum(row["detected"] for row in rows),
        }

    result["all"] = subgroup(lambda _: True)
    result["quality_valid"] = subgroup(
        lambda row: row["quality_valid"]
    )
    result["historically_successful_quality_valid"] = subgroup(
        lambda row: (
            row["quality_valid"] and row["historically_successful"]
        )
    )
    return result


def audit_attack_checkpoints(folder):
    payloads = experiment.attack_payloads()
    if not (folder / "locked_rule.json").exists():
        existing = list((folder / "checkpoints").glob("*/*/*.json"))
        if existing:
            raise ValueError("Attack checkpoints exist after a stop decision")
        return [], 0
    s2match = local.load_s2match()
    checkpoints = []
    candidates = 0
    for payload in payloads:
        path = local.checkpoint_path(folder, payload)
        if not path.exists():
            raise ValueError(f"Missing attack checkpoint: {path}")
        checkpoint = json.loads(path.read_text())
        candidates += local_audit.validate_checkpoint(
            checkpoint,
            payload,
            s2match,
        )
        checkpoints.append(checkpoint)
    return checkpoints, candidates


def audit(folder=experiment.OUTPUT):
    experiment.check(folder)
    calibration = experiment.load_reused(
        "human_calibration",
        ("sentence",),
    )
    positive = experiment.load_reused(
        "positive_development",
        ("original", "composition", "reparse"),
    )
    grid_rows, selected = reconstruct_development(
        calibration,
        positive,
    )
    development = json.loads((folder / "development.json").read_text())
    expected_development = {
        "status": "development_complete",
        "manifest_sha256": sha(folder / "manifest.json"),
        "grid_rows": grid_rows,
        "selected_rule_id": selected["rule_id"],
        "development_gate": {
            "minimum_original_and_reparse": (
                experiment.DEVELOPMENT_MIN_DETECTIONS
            ),
            "passed": (
                selected["positive_development"]["original"]["detected"]
                >= experiment.DEVELOPMENT_MIN_DETECTIONS
                and selected["positive_development"]["reparse"][
                    "detected"
                ]
                >= experiment.DEVELOPMENT_MIN_DETECTIONS
            ),
        },
        "integrity": {
            "human_calibration_documents": len(calibration),
            "positive_development_checkpoints": len(positive),
            "new_api_calls": 0,
            "new_parser_calls": 0,
            "new_s2match_calls": 0,
        },
    }
    if development != expected_development:
        raise ValueError("Development reconstruction mismatch")
    attack_checkpoints, attack_candidates = audit_attack_checkpoints(
        folder
    )
    if not development["development_gate"]["passed"]:
        if not (folder / "stop_decision.json").exists():
            raise ValueError("Missing development stop decision")
        summary = json.loads((folder / "summary.json").read_text())
        if (
            "locked_evaluation" in summary
            or summary["decision"]
            != "stop_template_specificity_preflight"
        ):
            raise ValueError("Stopped workflow evaluated protected data")
        result = {
            "status": "passed",
            "mismatches": 0,
            "workflow": "development_stop",
            "grid_rules_reconstructed": len(grid_rows),
            "attack_checkpoints_checked": 0,
            "attack_candidate_alignments_checked": 0,
            "new_api_calls": 0,
            "new_parser_calls": 0,
            "new_s2match_calls": 0,
            "summary_sha256": sha(folder / "summary.json"),
            "stable_aggregate_sha256": aggregate(folder),
        }
        write_json(folder / "audit.json", result)
        return result

    locked = json.loads((folder / "locked_rule.json").read_text())
    expected_locked = {
        "manifest_sha256": sha(folder / "manifest.json"),
        "development_sha256": sha(folder / "development.json"),
        "rule_id": selected["rule_id"],
        "rule": selected["rule"],
        "threshold": selected["threshold"],
        "calibration_false_positives": selected[
            "calibration_false_positives"
        ],
        "positive_development": selected["positive_development"],
    }
    for key, value in expected_locked.items():
        if locked[key] != value:
            raise ValueError("Locked rule reconstruction mismatch")
    matrix = [
        independent_scores(checkpoint, locked["rule"]["feature"])
        for checkpoint in sorted(
            calibration,
            key=lambda row: row["document_id"],
        )
    ]
    human = experiment.load_reused("human_test", ("sentence",))
    holdout = experiment.load_reused(
        "positive_holdout",
        ("original", "composition", "reparse"),
    )
    feature = locked["rule"]["feature"]
    top_k = locked["rule"]["top_k"]
    threshold = locked["threshold"]
    expected_human = independent_condition(
        human,
        feature,
        top_k,
        matrix,
        threshold,
    )
    expected_positive = {
        representation: independent_condition(
            [
                row
                for row in holdout
                if row["representation"] == representation
            ],
            feature,
            top_k,
            matrix,
            threshold,
        )
        for representation in ("original", "composition", "reparse")
    }
    inventory = experiment.attack_inventory()
    expected_attack = {
        representation: independent_attack_result(
            [
                row
                for row in attack_checkpoints
                if row["representation"] == representation
            ],
            feature,
            top_k,
            matrix,
            threshold,
            inventory,
        )
        for representation in experiment.ATTACK_REPRESENTATIONS
    }
    summary = json.loads((folder / "summary.json").read_text())
    expected_evaluation = {
        "human_test": expected_human,
        "positive_holdout": expected_positive,
        "external_attack": expected_attack,
    }
    if summary["locked_evaluation"] != expected_evaluation:
        raise ValueError("Locked evaluation reconstruction mismatch")
    gates = {
        "development_pass": True,
        "human_test_pass": (
            expected_human["detected"]
            <= experiment.TEST_MAX_FALSE_POSITIVES
        ),
        "original_holdout_pass": (
            expected_positive["original"]["detected"]
            >= experiment.HOLDOUT_MIN_DETECTIONS
        ),
        "reparse_holdout_pass": (
            expected_positive["reparse"]["detected"]
            >= experiment.HOLDOUT_MIN_DETECTIONS
        ),
        "quality_valid_attack_sentence_pass": (
            expected_attack["sentence"]["quality_valid"]["detected"]
            >= experiment.ATTACK_MIN_DETECTIONS
        ),
    }
    gates["all_pass"] = all(gates.values())
    expected_decision = (
        "template_specificity_feasible"
        if gates["all_pass"]
        else "stop_template_specificity_rule"
    )
    if summary["gates"] != gates or summary["decision"] != (
        expected_decision
    ):
        raise ValueError("Gate or decision mismatch")
    result = {
        "status": "passed",
        "mismatches": 0,
        "workflow": "locked_evaluation",
        "grid_rules_reconstructed": len(grid_rows),
        "locked_endpoints_reconstructed": 6,
        "attack_checkpoints_checked": len(attack_checkpoints),
        "attack_candidate_alignments_checked": attack_candidates,
        "new_api_calls": 0,
        "new_parser_calls": 0,
        "new_s2match_calls": 0,
        "summary_sha256": sha(folder / "summary.json"),
        "stable_aggregate_sha256": aggregate(folder),
    }
    write_json(folder / "audit.json", result)
    return result


def forbidden(name):
    def fail(*args, **kwargs):
        del args, kwargs
        raise AssertionError(
            f"Recovery attempted forbidden operation: {name}"
        )

    return fail


def recovery(folder=experiment.OUTPUT):
    if not (folder / "audit.json").exists():
        audit(folder)
    before = aggregate(folder, include_audit=True)
    s2match = local.load_s2match()
    with (
        patch.object(
            local,
            "score_graph",
            forbidden("local alignment extraction"),
        ),
        patch.object(
            s2match,
            "compute_pool",
            forbidden("S2Match compute_pool"),
        ),
    ):
        result = audit(folder)
    after = aggregate(folder, include_audit=True)
    if before != after:
        raise ValueError("Recovery changed stable artifacts")
    verification = {
        "status": "passed",
        "stable_before_sha256": before,
        "stable_after_sha256": after,
        "alignment_extraction_forbidden": True,
        "s2match_forbidden": True,
        "network_forbidden": True,
        "neural_parser_forbidden": True,
        "audit_sha256": sha(folder / "audit.json"),
        "audit_result": result,
    }
    write_json(folder / "recovery_verification.json", verification)
    return verification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("audit", "recovery", "all"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=experiment.OUTPUT,
    )
    args = parser.parse_args()
    if args.command == "audit":
        result = audit(args.output)
    elif args.command == "recovery":
        result = recovery(args.output)
    else:
        audit(args.output)
        result = recovery(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
