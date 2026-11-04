"""Offline audit and recovery guard for AMR composition decomposition."""

import argparse
import json
import math
from pathlib import Path
import statistics
from unittest.mock import patch

from run_confirmation_experiment import digest
from run_segmentation_intervention import sha, text_sha, write_json
from run_swan_known_positive_audit import load_utils
import run_swan_amr_composition_decomposition as experiment


def validate_vector(values, expected):
    if (
        len(values) != expected
        or not all(
            isinstance(value, (int, float))
            and math.isfinite(value)
            and 0 <= value <= 1
            for value in values
        )
    ):
        raise ValueError("Invalid saved S2Match vector")


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
        "confirmed_loss": bool(
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
    checkpoints = []
    score_observations = 0
    for source in manifest["sources"]:
        path = experiment.checkpoint_path(folder, source)
        checkpoint = json.loads(path.read_text())
        loaded = experiment.load_graphs(source)
        details, parsed, scored = loaded[1], loaded[3], loaded[4]
        composed_raw = experiment.compose_raw_amrs(
            details[0]["raw_amr"],
            details[1]["raw_amr"],
        )
        utils = load_utils()
        composed_amr = experiment.transform_amr(composed_raw, utils)
        if (
            checkpoint["prompt_id"] != source["prompt_id"]
            or checkpoint["pair_index"] != source["pair_index"]
            or checkpoint["source_slots"] != source["source_slots"]
            or checkpoint["manifest_sha256"] != sha(folder / "manifest.json")
            or checkpoint["composed_raw_amr"] != composed_raw
            or checkpoint["composed_amr"] != composed_amr
            or checkpoint["reparsed_amr"] != parsed["detail"]["amr"]
            or text_sha(composed_raw) != source["composed_raw_sha256"]
            or text_sha(composed_amr) != source["composed_amr_sha256"]
        ):
            raise ValueError("Composition checkpoint provenance mismatch")
        bank = checkpoint["bank"]
        validate_vector(bank["composition_scores"], 50)
        if (
            not math.isclose(
                bank["composition_max"],
                max(bank["composition_scores"]),
                abs_tol=1e-12,
            )
            or bank["composition_green"]
            != (bank["composition_max"] >= experiment.GREEN_THRESHOLD)
            or bank["saved_original_max"]
            != [detail["score"] for detail in details]
            or bank["saved_reparse_max"] != scored["detail"]["score"]
            or bank["saved_reparse_green"] != scored["green"]
        ):
            raise ValueError("Bank score derivation mismatch")
        expected_targets = []
        for component, saved in zip(
            source["components"],
            checkpoint["targets"],
        ):
            if {
                key: saved[key]
                for key in component
            } != component:
                raise ValueError("Target provenance mismatch")
            for condition in ("original", "composition", "reparse"):
                values = saved["scores"][condition]["observations"]
                validate_vector(values, experiment.REPEATS)
                if not math.isclose(
                    saved["scores"][condition]["median"],
                    statistics.median(values),
                    abs_tol=1e-12,
                ):
                    raise ValueError("Target median mismatch")
            expected_targets.append(saved)
        if len(expected_targets) != 2:
            raise ValueError("Target count mismatch")
        similarity = checkpoint["composition_reparse_similarity"]
        validate_vector(similarity["observations"], experiment.REPEATS)
        if not math.isclose(
            similarity["median"],
            statistics.median(similarity["observations"]),
            abs_tol=1e-12,
        ):
            raise ValueError("Graph similarity median mismatch")
        expected_triples = {
            "original_sum": sum(
                experiment.graph_triples(detail["amr"])
                for detail in details
            ),
            "composition": experiment.graph_triples(composed_amr),
            "reparse": experiment.graph_triples(parsed["detail"]["amr"]),
        }
        if (
            checkpoint["triple_counts"] != expected_triples
            or checkpoint["new_s2match_observations"] != 85
            or not math.isfinite(checkpoint["elapsed_seconds"])
            or checkpoint["elapsed_seconds"] < 0
        ):
            raise ValueError("Checkpoint accounting mismatch")
        score_observations += checkpoint["new_s2match_observations"]
        checkpoints.append(checkpoint)
    saved_summary = json.loads((folder / "summary.json").read_text())
    if (
        len(checkpoints) != 60
        or score_observations != experiment.EXPECTED_S2MATCH
        or saved_summary["integrity"]["new_s2match_observations"]
        != score_observations
    ):
        raise ValueError("Experiment inventory mismatch")
    composition_differences = [
        row["composition_minus_original"]
        for row in saved_summary["documents"]
    ]
    reparse_differences = [
        row["reparse_minus_composition"]
        for row in saved_summary["documents"]
    ]
    if (
        independent_sign_test(composition_differences)
        != saved_summary["primary"]["composition_matcher_dilution"]
        or independent_sign_test(reparse_differences)
        != saved_summary["primary"]["additional_reparse_loss"]
    ):
        raise ValueError("Primary sign test mismatch")
    rebuilt = experiment.summarize(folder)
    if rebuilt != saved_summary:
        raise ValueError("Summary does not reconstruct")
    result = {
        "status": "passed",
        "mismatches": 0,
        "composition_graphs_reconstructed": len(checkpoints),
        "target_score_vectors_checked": len(checkpoints) * 2 * 3,
        "graph_similarity_vectors_checked": len(checkpoints),
        "saved_s2match_observations_checked": score_observations,
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
            "compute_s2match_score",
            forbidden("compute_s2match_score"),
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
