"""Offline audit and recovery guard for the SWAN oracle diagnostic."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from run_confirmation_experiment import digest
from run_segmentation_intervention import amr_body, sha, write_json
from run_swan_known_positive_audit import load_utils
from run_swan_oracle_composition import (
    OUTPUT,
    build_summary,
    check,
    parser_path,
    report_text,
    run,
    runtime_payload,
    score_path,
    validate_detail,
)


def transformed_amr(raw_amr, utils):
    import penman

    graph = amr_body(raw_amr)
    penman.decode(graph)
    graph = utils.normalize_amr_variables(
        utils.generate_template_amr(utils.normalize_amr_variables(graph))
    )
    penman.decode(graph)
    return graph


def stable_paths(folder, include_audit=False):
    paths = [
        folder / "manifest.json",
        folder / "runtime.json",
        folder / "summary.json",
        folder / "report.md",
    ]
    paths.extend(sorted((folder / "pair_parser").glob("*.json")))
    paths.extend(sorted((folder / "pair_scores").glob("*.json")))
    if include_audit:
        paths.append(folder / "audit.json")
    return paths


def aggregate_sha256(folder, include_audit=False):
    rows = [
        {
            "path": str(path.relative_to(folder)),
            "sha256": sha(path),
        }
        for path in stable_paths(folder, include_audit=include_audit)
    ]
    return digest(rows)


def audit(folder=OUTPUT):
    manifest = check(folder)
    runtime = json.loads((folder / "runtime.json").read_text())
    if runtime != runtime_payload(folder):
        raise ValueError("Runtime differs from the frozen environment signature")
    utils = load_utils()
    inputs = {
        str(path.relative_to(folder)): sha(path)
        for path in stable_paths(folder)
    }
    parser_checked = 0
    score_checked = 0
    for source in manifest["sources"]:
        for unit in source["pair_units"]:
            prompt_id = source["prompt_id"]
            pair_index = unit["pair_index"]
            parsed_path = parser_path(folder, prompt_id, pair_index)
            scored_path = score_path(folder, prompt_id, pair_index)
            parsed = json.loads(parsed_path.read_text())
            scored = json.loads(scored_path.read_text())
            if (
                parsed["prompt_id"] != prompt_id
                or parsed["pair_index"] != pair_index
                or parsed["source_slots"] != unit["source_slots"]
                or parsed["text"] != unit["text"]
                or parsed["text_sha256"] != unit["text_sha256"]
            ):
                raise ValueError("Pair parser provenance mismatch")
            detail = parsed["detail"]
            if detail["text"] != unit["text"]:
                raise ValueError("Pair parser text mismatch")
            rebuilt = transformed_amr(detail["raw_amr"], utils)
            if rebuilt != detail["amr"]:
                raise ValueError("Saved parser graph transformation mismatch")
            if (
                scored["prompt_id"] != prompt_id
                or scored["pair_index"] != pair_index
                or scored["text_sha256"] != unit["text_sha256"]
                or scored["parser_sha256"] != sha(parsed_path)
                or {
                    key: scored["detail"][key]
                    for key in ("text", "raw_amr", "amr")
                }
                != detail
            ):
                raise ValueError("Pair score provenance mismatch")
            green = validate_detail(scored["detail"])
            if scored["green"] != green:
                raise ValueError("Saved pair green label is inconsistent")
            parser_checked += 1
            score_checked += 1
    expected_summary = build_summary(folder)
    saved_summary = json.loads((folder / "summary.json").read_text())
    if saved_summary != expected_summary:
        raise ValueError("Summary does not reconstruct from frozen checkpoints")
    if (folder / "report.md").read_text() != report_text(expected_summary):
        raise ValueError("Markdown report does not reconstruct")
    if not saved_summary["integrity_pass"]:
        raise ValueError("Completed diagnostic has a failed integrity gate")
    payload = {
        "status": "complete_offline_audit",
        "parser_checkpoints_reconstructed": parser_checked,
        "score_checkpoints_verified": score_checked,
        "saved_s2match_observations_verified": score_checked * 50,
        "api_calls": 0,
        "new_parser_calls": 0,
        "new_s2match_calls": 0,
        "summary_exact": True,
        "report_exact": True,
        "decision": saved_summary["primary"]["decision"],
        "input_sha256": inputs,
        "stable_aggregate_sha256": aggregate_sha256(folder),
    }
    write_json(folder / "audit.json", payload)
    return payload


def forbidden(name):
    def fail(*_args, **_kwargs):
        raise AssertionError(f"Recovery attempted forbidden operation: {name}")

    return fail


def verify_recovery(folder=OUTPUT):
    audit_path = folder / "audit.json"
    if not audit_path.exists():
        audit(folder)
    before = aggregate_sha256(folder, include_audit=True)
    import amrlib
    import requests

    utils = load_utils()
    with (
        patch.object(requests, "post", forbidden("requests.post")),
        patch.object(amrlib, "load_stog_model", forbidden("amrlib.load_stog_model")),
        patch.object(
            utils,
            "compute_s2match_score",
            forbidden("compute_s2match_score"),
        ),
    ):
        summary = run(folder)
    after = aggregate_sha256(folder, include_audit=True)
    if before != after:
        raise ValueError("Recovery changed stable experiment artifacts")
    if summary != json.loads((folder / "summary.json").read_text()):
        raise ValueError("Recovery summary differs")
    payload = {
        "status": "complete_no_compute_recovery",
        "stable_aggregate_sha256": after,
        "network_calls": 0,
        "parser_loads": 0,
        "s2match_calls": 0,
        "decision": summary["primary"]["decision"],
    }
    write_json(folder / "recovery_verification.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify-recovery", action="store_true")
    args = parser.parse_args()
    result = (
        verify_recovery(args.output)
        if args.verify_recovery
        else audit(args.output)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
