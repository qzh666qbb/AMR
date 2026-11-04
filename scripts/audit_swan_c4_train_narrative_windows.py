"""Offline audit and recovery guard for C4-train narrative windows."""

import argparse
import json
from pathlib import Path

from audit_swan_natural_mixed_windows import (
    audit_documents,
    audit_parser,
    audit_reviews,
)
from run_confirmation_experiment import digest
from run_segmentation_intervention import sha, write_json
import run_swan_c4_train_narrative_windows as experiment


def audit_acquisition(folder, manifest):
    journal = json.loads((folder / "source_acquisition.json").read_text())
    attempts = journal.get("attempts", [])
    if (
        not attempts
        or len(attempts) > experiment.MAX_ACQUISITION_ATTEMPTS
        or [row.get("ordinal") for row in attempts]
        != list(range(1, len(attempts) + 1))
        or any(
            row.get("status") not in {"failed", "complete"}
            for row in attempts
        )
        or attempts[-1].get("status") != "complete"
        or attempts[-1].get("scanned_through")
        != manifest["dataset"]["scanned_through"]
        or attempts[-1].get("eligible_rows_scanned")
        != manifest["screening"]["eligible_rows_scanned"]
        or attempts[-1].get("selected_documents")
        != len(manifest["sources"])
        or sum(row["shards_opened"] for row in attempts)
        != manifest["dataset"]["dataset_stream_http_requests"]
    ):
        raise ValueError("Source acquisition journal mismatch")
    return len(attempts)


def audit(folder=experiment.OUTPUT):
    manifest = experiment.check(folder)
    acquisition_attempts = audit_acquisition(folder, manifest)
    sentences, short_slots = audit_documents(folder, manifest)
    line_mappings = sum(
        len({row["line_index"] for row in source["window"]["sentences"]})
        for source in manifest["sources"]
    )
    if len(manifest["sources"]) < experiment.DOCUMENTS:
        if (
            list((folder / "calls").glob("*.json"))
            or list((folder / "reviews").glob("*.json"))
            or list((folder / "parser").glob("*.json"))
        ):
            raise ValueError("Corpus-gate failure used downstream computation")
        summary = experiment.summarize(folder)
        if summary["decision"] != "stop_c4_train_narrative_window_route":
            raise ValueError("Corpus-gate decision mismatch")
        result = {
            "status": "passed_corpus_gate_audit",
            "mismatches": 0,
            "source_acquisition_attempts_checked": acquisition_attempts,
            "selected_documents_reconstructed": len(manifest["sources"]),
            "window_sentences_reconstructed": sentences,
            "source_line_mappings_reconstructed": line_mappings,
            "short_slots_reconstructed": short_slots,
            "new_api_calls": 0,
            "new_neural_parser_calls": 0,
            "new_s2match_calls": 0,
            "summary_sha256": sha(folder / "summary.json"),
        }
        write_json(folder / "audit.json", result)
        return result

    runtime = json.loads(experiment.mixed.runtime_path(folder).read_text())
    if runtime["manifest_sha256"] != sha(folder / "manifest.json"):
        raise ValueError("Runtime manifest identity mismatch")
    api = audit_reviews(folder, manifest, runtime)
    parser_sentences = audit_parser(folder, manifest, runtime)
    summary = experiment.summarize(folder)
    if (
        summary["status"] != "complete"
        or api.sent != 0
        or sum(api.counters.values()) != experiment.API_CAP
        or summary["api_attempts"] != experiment.API_CAP
        or summary["neural_parser_sentences"] != parser_sentences
    ):
        raise ValueError("Summary accounting mismatch")
    expected_decision = (
        "c4_train_narrative_windows_support_partial_injection_preflight"
        if summary["joint_pass"] >= experiment.mixed.MIN_JOINT_DOCUMENTS
        else "stop_c4_train_narrative_window_route"
    )
    if (
        summary["decision"] != expected_decision
        or summary["continuation_gate"]
        != (
            summary["joint_pass"]
            >= experiment.mixed.MIN_JOINT_DOCUMENTS
        )
    ):
        raise ValueError("Continuation decision mismatch")
    result = {
        "status": "passed",
        "mismatches": 0,
        "source_acquisition_attempts_checked": acquisition_attempts,
        "selected_documents_reconstructed": experiment.DOCUMENTS,
        "window_sentences_reconstructed": sentences,
        "source_line_mappings_reconstructed": line_mappings,
        "short_slots_reconstructed": short_slots,
        "api_calls_replayed": experiment.API_CAP,
        "parser_sentences_reconstructed": parser_sentences,
        "new_api_calls": 0,
        "new_neural_parser_calls": 0,
        "new_s2match_calls": 0,
        "summary_sha256": sha(folder / "summary.json"),
    }
    write_json(folder / "audit.json", result)
    return result


def stable_files(folder):
    paths = [
        folder / "source_acquisition.json",
        folder / "manifest.json",
        folder / "summary.json",
        folder / "report.md",
        folder / "audit.json",
        experiment.mixed.runtime_path(folder),
    ]
    for subfolder in ("calls", "generated", "reviews", "parser"):
        paths.extend(sorted((folder / subfolder).glob("*.json")))
    return [path for path in paths if path.exists()]


def recovery(folder=experiment.OUTPUT):
    manifest = experiment.check(folder)
    before = {
        str(path.relative_to(folder)): sha(path)
        for path in stable_files(folder)
    }
    aggregate = digest(before)
    import amrlib
    import requests
    import spacy

    original_check = experiment.check
    original_post = requests.post
    original_get = requests.get
    original_spacy = spacy.load
    original_parser = amrlib.load_stog_model
    original_utils = experiment.mixed.load_utils

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Recovery attempted forbidden computation")

    experiment.check = lambda _folder: manifest
    requests.post = forbidden
    requests.get = forbidden
    spacy.load = forbidden
    amrlib.load_stog_model = forbidden
    experiment.mixed.load_utils = forbidden
    try:
        result = experiment.run(folder, "")
    finally:
        experiment.check = original_check
        requests.post = original_post
        requests.get = original_get
        spacy.load = original_spacy
        amrlib.load_stog_model = original_parser
        experiment.mixed.load_utils = original_utils
    after = {
        str(path.relative_to(folder)): sha(path)
        for path in stable_files(folder)
    }
    if before != after or aggregate != digest(after):
        raise ValueError("Recovery changed stable artifacts")
    verification = {
        "status": "complete_no_compute_recovery",
        "decision": result["decision"],
        "files_checked": len(before),
        "stable_aggregate_sha256": aggregate,
        "network_forbidden": True,
        "spacy_loading_forbidden": True,
        "neural_amr_parser_forbidden": True,
        "graph_utility_loading_forbidden": True,
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
