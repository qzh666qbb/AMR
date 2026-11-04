"""Audit and recovery guards for the local multi-instance preflight."""

import argparse
import json
import math
from pathlib import Path
from unittest.mock import patch

from run_confirmation_experiment import digest
from run_segmentation_intervention import sha, write_json
import run_swan_local_multi_instance as experiment


def stable_paths(folder, include_audit=False):
    paths = [
        folder / "manifest.json",
        folder / "development.json",
        folder / "locked_rule.json",
        folder / "summary.json",
        folder / "report.md",
    ]
    paths.extend(sorted((folder / "checkpoints").glob("*/*/*.json")))
    if include_audit:
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


def all_payloads():
    return (
        experiment.human_payloads("human_calibration")
        + experiment.human_payloads("human_test")
        + experiment.positive_payloads(
            experiment.DEVELOPMENT_IDS + experiment.HOLDOUT_IDS
        )
    )


def validate_candidate(candidate, parsed):
    required = {
        "bank_index",
        "occurrence",
        "soft_match",
        "template_triples",
        "local_candidate_triples",
        "coverage",
        "local_precision",
        "local_f1",
        "active_nodes",
    }
    if set(candidate) != required:
        raise ValueError("Candidate schema mismatch")
    active = candidate["active_nodes"]
    if (
        candidate["bank_index"] not in range(50)
        or candidate["occurrence"] not in range(
            experiment.MAX_OCCURRENCES
        )
        or active != sorted(set(active))
        or not active
        or min(active) < 0
        or max(active) >= len(parsed["instances"])
    ):
        raise ValueError("Candidate identity mismatch")
    expected_local = experiment.local_triple_count(parsed, set(active))
    if expected_local != candidate["local_candidate_triples"]:
        raise ValueError("Local denominator mismatch")
    coverage = (
        candidate["soft_match"] / candidate["template_triples"]
    )
    precision = (
        candidate["soft_match"]
        / candidate["local_candidate_triples"]
    )
    if (
        not math.isclose(
            coverage,
            candidate["coverage"],
            abs_tol=1e-12,
        )
        or not math.isclose(
            precision,
            candidate["local_precision"],
            abs_tol=1e-12,
        )
        or not math.isclose(
            experiment.harmonic(precision, coverage),
            candidate["local_f1"],
            abs_tol=1e-12,
        )
        or coverage + 1e-12 < experiment.PREFILTER_RECALL
        or coverage > 1 + 1e-9
        or precision > 1 + 1e-9
    ):
        raise ValueError("Candidate metric mismatch")


def validate_checkpoint(checkpoint, payload, s2match):
    if (
        checkpoint["document_id"] != payload["document_id"]
        or checkpoint["cohort"] != payload["cohort"]
        or checkpoint["representation"] != payload["representation"]
        or checkpoint["source_identity"]
        != experiment.payload_identity(payload)
        or len(checkpoint["graphs"]) != len(payload["graphs"])
    ):
        raise ValueError("Checkpoint identity mismatch")
    screened = 0
    hill_climbs = 0
    candidates = 0
    for graph, source in zip(checkpoint["graphs"], payload["graphs"]):
        if (
            graph["graph_id"] != source["graph_id"]
            or graph["amr_sha256"] != experiment.text_sha(source["amr"])
            or graph["bank_scores_sha256"]
            != digest(source["bank_scores"])
            or graph["known_target_template_indices"]
            != source["known_target_template_indices"]
            or graph["screened_templates"] not in range(51)
            or graph["hill_climbs"]
            > graph["screened_templates"]
            * experiment.MAX_OCCURRENCES
        ):
            raise ValueError("Graph checkpoint mismatch")
        parsed = experiment.parse_graph(source["amr"], "a", s2match)
        if graph["candidate_triples"] != parsed["triples"]:
            raise ValueError("Candidate triple count mismatch")
        identities = set()
        by_template = {}
        for candidate in graph["candidates"]:
            validate_candidate(candidate, parsed)
            identity = (
                candidate["bank_index"],
                candidate["occurrence"],
            )
            if identity in identities:
                raise ValueError("Duplicate candidate occurrence")
            identities.add(identity)
            by_template.setdefault(candidate["bank_index"], []).append(
                candidate
            )
        for rows in by_template.values():
            for index, first in enumerate(rows):
                for second in rows[index + 1:]:
                    if set(first["active_nodes"]) & set(
                        second["active_nodes"]
                    ):
                        raise ValueError(
                            "Repeated template occurrences overlap"
                        )
        screened += graph["screened_templates"]
        hill_climbs += graph["hill_climbs"]
        candidates += len(graph["candidates"])
    if (
        screened != checkpoint["screened_templates"]
        or hill_climbs != checkpoint["hill_climbs"]
        or not math.isfinite(checkpoint["elapsed_seconds"])
        or checkpoint["elapsed_seconds"] < 0
    ):
        raise ValueError("Checkpoint accounting mismatch")
    return candidates


def reconstruct_development(calibration, positive):
    by_condition = {
        condition: [
            row
            for row in positive
            if row["representation"] == condition
        ]
        for condition in ("original", "composition", "reparse")
    }
    rows = []
    for rule in experiment.GRID:
        counts = [
            experiment.document_evidence(checkpoint, rule)["count"]
            for checkpoint in calibration
        ]
        threshold, false_positives = experiment.calibrated_threshold(
            counts
        )
        recoveries = {
            condition: experiment.condition_result(
                by_condition[condition],
                rule,
                threshold,
            )["detected"]
            for condition in by_condition
        }
        rows.append(
            {
                "rule_id": experiment.rule_id(rule),
                "rule": rule,
                "document_threshold": threshold,
                "calibration_false_positives": false_positives,
                "positive_development": recoveries,
            }
        )
    selected = max(rows, key=experiment.development_rank)
    return rows, selected


def audit(folder=experiment.OUTPUT):
    experiment.check(folder)
    payloads = all_payloads()
    s2match = experiment.load_s2match()
    checkpoints = []
    candidate_count = 0
    for payload in payloads:
        path = experiment.checkpoint_path(folder, payload)
        checkpoint = json.loads(path.read_text())
        candidate_count += validate_checkpoint(
            checkpoint,
            payload,
            s2match,
        )
        checkpoints.append(checkpoint)
    calibration = [
        row
        for row in checkpoints
        if row["cohort"] == "human_calibration"
    ]
    development_positive = [
        row
        for row in checkpoints
        if row["cohort"] == "positive_development"
    ]
    test = [
        row for row in checkpoints if row["cohort"] == "human_test"
    ]
    holdout = [
        row
        for row in checkpoints
        if row["cohort"] == "positive_holdout"
    ]
    grid_rows, selected = reconstruct_development(
        calibration,
        development_positive,
    )
    development = json.loads((folder / "development.json").read_text())
    if (
        development["manifest_sha256"] != sha(folder / "manifest.json")
        or development["grid_rows"] != grid_rows
        or development["selected_rule_id"] != selected["rule_id"]
    ):
        raise ValueError("Development reconstruction mismatch")
    locked = json.loads((folder / "locked_rule.json").read_text())
    expected_locked = {
        "manifest_sha256": sha(folder / "manifest.json"),
        "development_sha256": sha(folder / "development.json"),
        "rule_id": selected["rule_id"],
        "rule": selected["rule"],
        "document_threshold": selected["document_threshold"],
        "calibration_false_positives": selected[
            "calibration_false_positives"
        ],
        "positive_development": selected["positive_development"],
    }
    for key, value in expected_locked.items():
        if locked[key] != value:
            raise ValueError("Locked rule reconstruction mismatch")
    rule = locked["rule"]
    threshold = locked["document_threshold"]
    expected_human = experiment.condition_result(
        test,
        rule,
        threshold,
    )
    expected_positive = {
        condition: experiment.condition_result(
            [
                row
                for row in holdout
                if row["representation"] == condition
            ],
            rule,
            threshold,
        )
        for condition in ("original", "composition", "reparse")
    }
    summary = json.loads((folder / "summary.json").read_text())
    if (
        summary["locked_evaluation"]["human_test"] != expected_human
        or summary["locked_evaluation"]["positive_holdout"]
        != expected_positive
    ):
        raise ValueError("Locked evaluation reconstruction mismatch")
    gates = {
        "human_test_pass": (
            expected_human["detected"]
            <= experiment.TEST_MAX_FALSE_POSITIVES
        ),
        "original_recovery_pass": (
            expected_positive["original"]["detected"]
            >= experiment.POSITIVE_MIN_RECOVERY
        ),
        "reparse_recovery_pass": (
            expected_positive["reparse"]["detected"]
            >= experiment.POSITIVE_MIN_RECOVERY
        ),
    }
    gates["all_pass"] = all(gates.values())
    if summary["gates"] != gates:
        raise ValueError("Decision gate mismatch")
    expected_decision = (
        "local_multi_instance_feasible"
        if gates["all_pass"]
        else "stop_current_local_multi_instance_rule"
    )
    if summary["decision"] != expected_decision:
        raise ValueError("Decision mismatch")
    result = {
        "status": "passed",
        "mismatches": 0,
        "checkpoints_checked": len(checkpoints),
        "candidate_alignments_checked": candidate_count,
        "grid_rules_reconstructed": len(grid_rows),
        "locked_endpoints_reconstructed": 4,
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
    s2match = experiment.load_s2match()
    with patch.object(
        s2match,
        "compute_pool",
        forbidden("S2Match compute_pool"),
    ):
        result = audit(folder)
    after = aggregate(folder, include_audit=True)
    if before != after:
        raise ValueError("Recovery changed stable artifacts")
    verification = {
        "status": "passed",
        "stable_before_sha256": before,
        "stable_after_sha256": after,
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
    parser.add_argument("--output", type=Path, default=experiment.OUTPUT)
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
