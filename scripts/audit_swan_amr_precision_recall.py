"""Audit and recovery guard for S2Match precision/recall decomposition."""

import argparse
import json
import math
from pathlib import Path
import statistics
from unittest.mock import patch

from run_confirmation_experiment import digest
from run_segmentation_intervention import sha, write_json
from run_swan_known_positive_audit import load_utils
import run_swan_amr_precision_recall as experiment


def independent_sign_test(differences):
    losses = sum(value < 0 for value in differences)
    gains = sum(value > 0 for value in differences)
    ties = len(differences) - losses - gains
    n = losses + gains
    probability = (
        sum(math.comb(n, value) for value in range(losses, n + 1))
        / 2**n
        if n
        else 1.0
    )
    median = statistics.median(differences)
    return {
        "differences": differences,
        "median_difference": median,
        "losses": losses,
        "ties": ties,
        "gains": gains,
        "non_ties": n,
        "one_sided_sign_p": probability,
        "confirmed_negative": bool(
            median < 0 and probability <= experiment.SIGN_ALPHA
        ),
    }


def stable_paths(folder, include_audit=False):
    paths = [
        folder / "manifest.json",
        folder / "summary.json",
        folder / "report.md",
    ]
    paths.extend(sorted((folder / "scores").glob("*.json")))
    if include_audit:
        paths.append(folder / "audit.json")
    return paths


def aggregate(folder, include_audit=False):
    return digest([
        {
            "path": str(path.relative_to(folder)),
            "sha256": sha(path),
        }
        for path in stable_paths(folder, include_audit)
    ])


def audit(folder=experiment.OUTPUT):
    manifest = experiment.check(folder)
    observations = 0
    checkpoints = []
    for target in manifest["targets"]:
        checkpoint = json.loads(
            experiment.checkpoint_path(folder, target).read_text()
        )
        for key in (
            "prompt_id",
            "pair_index",
            "slot",
            "target_template_sha256",
            "original_amr_sha256",
            "composition_amr_sha256",
            "reparse_amr_sha256",
        ):
            if checkpoint[key] != target[key]:
                raise ValueError("Target checkpoint identity mismatch")
        if checkpoint["manifest_sha256"] != sha(folder / "manifest.json"):
            raise ValueError("Checkpoint manifest mismatch")
        for condition in ("original", "composition", "reparse"):
            value = checkpoint["conditions"][condition]
            rows = value["observations"]
            if len(rows) != experiment.REPEATS:
                raise ValueError("Observation count mismatch")
            for row in rows:
                precision = row["precision"]
                recall = row["recall"]
                f1 = row["f1"]
                if not all(
                    math.isfinite(item) and 0 <= item <= 1
                    for item in (precision, recall, f1)
                ):
                    raise ValueError("Invalid saved metric")
                expected_f1 = (
                    2 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0.0
                )
                if not math.isclose(f1, expected_f1, abs_tol=1e-12):
                    raise ValueError("Saved F1 identity mismatch")
            expected_median = {
                metric: statistics.median(row[metric] for row in rows)
                for metric in ("precision", "recall", "f1")
            }
            if value["median"] != expected_median:
                raise ValueError("Metric median mismatch")
            observations += len(rows)
        if (
            checkpoint["new_s2match_observations"] != 15
            or not math.isfinite(checkpoint["elapsed_seconds"])
            or checkpoint["elapsed_seconds"] < 0
        ):
            raise ValueError("Checkpoint accounting mismatch")
        checkpoints.append(checkpoint)
    summary = json.loads((folder / "summary.json").read_text())
    if (
        len(checkpoints) != experiment.EXPECTED_TARGETS
        or observations != experiment.EXPECTED_S2MATCH
        or summary["integrity"]["new_s2match_observations"] != observations
    ):
        raise ValueError("Experiment inventory mismatch")
    precision = independent_sign_test([
        row["composition_precision_change"]
        for row in summary["documents"]
    ])
    specific = independent_sign_test([
        row["precision_minus_recall_change"]
        for row in summary["documents"]
    ])
    if (
        precision != summary["primary"]["composition_precision_loss"]
        or specific != summary["primary"]["precision_specific_dilution"]
    ):
        raise ValueError("Primary sign test mismatch")
    rebuilt = experiment.summarize(folder)
    if rebuilt != summary:
        raise ValueError("Summary does not reconstruct")
    result = {
        "status": "passed",
        "mismatches": 0,
        "target_checkpoints_checked": len(checkpoints),
        "precision_recall_f1_observations_checked": observations,
        "f1_identities_checked": observations,
        "document_sign_tests_reconstructed": 2,
        "new_api_calls": 0,
        "new_neural_parser_calls": 0,
        "new_s2match_calls": 0,
        "summary_sha256": sha(folder / "summary.json"),
        "stable_aggregate_sha256": aggregate(folder),
    }
    write_json(folder / "audit.json", result)
    return result


def forbidden(name):
    def fail(*args, **kwargs):
        del args, kwargs
        raise AssertionError(f"Recovery attempted forbidden operation: {name}")

    return fail


def recovery(folder=experiment.OUTPUT):
    if not (folder / "audit.json").exists():
        audit(folder)
    before = aggregate(folder, include_audit=True)
    utils = load_utils()
    import amrlib
    import requests

    with (
        patch.object(
            utils,
            "compute_s2match_from_strings",
            forbidden("compute_s2match_from_strings"),
        ),
        patch.object(
            amrlib,
            "load_stog_model",
            forbidden("amrlib.load_stog_model"),
        ),
        patch.object(requests, "post", forbidden("requests.post")),
        patch.object(requests, "get", forbidden("requests.get")),
    ):
        result = experiment.run(folder)
    after = aggregate(folder, include_audit=True)
    if before != after:
        raise ValueError("Recovery changed stable artifacts")
    verification = {
        "status": "complete_no_compute_recovery",
        "decision": result["primary"]["decision"],
        "stable_aggregate_sha256": after,
        "files_checked": len(stable_paths(folder, include_audit=True)),
        "network_forbidden": True,
        "parser_loading_forbidden": True,
        "s2match_forbidden": True,
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
