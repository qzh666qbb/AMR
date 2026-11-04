"""Offline audit and recovery guard for quality-first natural windows."""

import argparse
import json
import math
from pathlib import Path

from run_confirmation_experiment import (
    CallJournal,
    digest,
    request_body,
)
from run_segmentation_intervention import sha, text_sha, write_json
import run_swan_quality_first_windows as experiment


class Replay(CallJournal):
    def __init__(self, folder):
        super().__init__(folder, "", experiment.API_CAP)

    def request(self, scope, body):
        ordinal = self.counters.get(scope, 0)
        path = self.folder / "calls" / f"{scope}-{ordinal:03d}.json"
        if not path.exists():
            raise ValueError("Audit replay would require an API call")
        return super().request(scope, body)


def audit_acquisition(folder, manifest):
    journal = json.loads((folder / "source_acquisition.json").read_text())
    attempts = journal.get("attempts", [])
    if (
        not attempts
        or len(attempts) > experiment.MAX_ACQUISITION_ATTEMPTS
        or [row.get("ordinal") for row in attempts]
        != list(range(1, len(attempts) + 1))
        or attempts[-1].get("status") != "complete"
        or attempts[-1].get("scanned_through")
        != manifest["dataset"]["scanned_through"]
        or attempts[-1].get("eligible_rows_scanned")
        != manifest["screening"]["eligible_rows_scanned"]
        or attempts[-1].get("selected_documents")
        != len(manifest["sources"])
        or attempts[-1].get("short_slot_predicate_evaluations") != 0
        or sum(row["shards_opened"] for row in attempts)
        != manifest["dataset"]["dataset_stream_http_requests"]
    ):
        raise ValueError("Source acquisition journal mismatch")
    return len(attempts)


def audit_documents(folder, manifest):
    sentences = 0
    line_mappings = 0
    for source in manifest["sources"]:
        expected = experiment.build_quality_document(source)
        saved = json.loads(
            (folder / "generated" / f"{source['prompt_id']}.json").read_text()
        )
        if saved != expected or any(
            "short" in key for key in saved
        ):
            raise ValueError("Slot-blind document reconstruction mismatch")
        sentences += len(expected["sentences"])
        line_mappings += len({
            row["line_index"] for row in source["window"]["sentences"]
        })
    return sentences, line_mappings


def audit_reviews(folder, manifest, runtime):
    api = Replay(folder)
    probe = api.request(
        "probe",
        request_body("Reply exactly OK.", 0, 16),
    )
    if (
        probe["model"] != runtime["model_returned"]
        or probe["id"] != runtime["probe_id"]
        or probe["content"].strip() != "OK"
    ):
        raise ValueError("Connectivity probe mismatch")
    api.model_returned = runtime["model_returned"]
    for source in manifest["sources"]:
        generated = json.loads(
            (folder / "generated" / f"{source['prompt_id']}.json").read_text()
        )
        body = experiment.mixed.style_body(generated["passage"])
        body_text = json.dumps(body, sort_keys=True)
        if (
            "short_slots" in body_text
            or "coverage" in body_text
            or "sentence_words" in body_text
        ):
            raise ValueError("Quality review exposed coverage metadata")
        response = api.request(f"style-{source['prompt_id']}", body)
        errors = {}
        try:
            style = experiment.mixed.decode_baseline_review(
                response["content"]
            )
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            style = None
            errors["style"] = type(exc).__name__
        gates = experiment.mixed.style_gate(generated, style)
        expected = {
            "prompt_id": source["prompt_id"],
            "source_sha256": source["source_sha256"],
            "passage_sha256": generated["passage_sha256"],
            "runtime_sha256": digest(runtime),
            "request_sha256": digest(body),
            "request_exposes_coverage": False,
            "response": response,
            "style": style,
            "decode_errors": errors,
            "gates": gates,
            "quality_pass": all(gates.values()),
        }
        saved = json.loads(
            (folder / "reviews" / f"{source['prompt_id']}.json").read_text()
        )
        if saved != expected:
            raise ValueError("Quality review reconstruction mismatch")
    return api


def audit_coverage(folder, manifest):
    slots = 0
    for source in manifest["sources"]:
        generated = json.loads(
            (folder / "generated" / f"{source['prompt_id']}.json").read_text()
        )
        expected = experiment.build_coverage(generated)
        saved = json.loads(
            (folder / "coverage" / f"{source['prompt_id']}.json").read_text()
        )
        if saved != expected:
            raise ValueError("Coverage reveal reconstruction mismatch")
        slots += saved["slot_count"]
    return slots


def audit_parser(folder, manifest, runtime):
    eligible = experiment.joint_quality_coverage_sources(folder, manifest)
    parser_paths = list((folder / "parser").glob("*.json"))
    if len(eligible) < experiment.MIN_JOINT_COVERAGE_DOCUMENTS:
        if parser_paths:
            raise ValueError("AMR ran below joint quality-coverage gate")
        return 0
    utils = experiment.mixed.load_utils()
    bank = json.loads(experiment.mixed.BANK_PATH.read_text())
    bank_counts = [
        experiment.mixed.s2_counts(row["normalized_template_amr"])
        for row in bank
    ]
    checked = 0
    eligible_ids = {source["prompt_id"] for source in eligible}
    if {
        path.stem for path in parser_paths
    } != eligible_ids:
        raise ValueError("Parser document set mismatch")
    for source in eligible:
        coverage = json.loads(
            (folder / "coverage" / f"{source['prompt_id']}.json").read_text()
        )
        generated = json.loads(
            (folder / "generated" / f"{source['prompt_id']}.json").read_text()
        )
        parsed = json.loads(
            (folder / "parser" / f"{source['prompt_id']}.json").read_text()
        )
        rebuilt = []
        for index, (sentence, saved) in enumerate(
            zip(coverage["slots"], parsed["sentences"])
        ):
            identity = {
                "sentence_index": index,
                "source_sentence_index": coverage[
                    "slot_source_indices"
                ][index],
                "text": sentence,
                "text_sha256": text_sha(sentence),
                "words": len(sentence.split()),
            }
            row = experiment.mixed.analyze_parser_output(
                identity,
                saved["raw_parser_output"],
                utils,
                bank_counts,
            )
            row["input_tokens"] = saved["input_tokens"]
            rebuilt.append(row)
        expected = {
            "prompt_id": source["prompt_id"],
            "source_sha256": source["source_sha256"],
            "passage_sha256": generated["passage_sha256"],
            "coverage_sha256": digest(coverage),
            "runtime_sha256": digest(runtime),
            "sentences": rebuilt,
            "elapsed_seconds": parsed["elapsed_seconds"],
        }
        if (
            parsed != expected
            or len(rebuilt) != coverage["slot_count"]
            or not math.isfinite(parsed["elapsed_seconds"])
            or parsed["elapsed_seconds"] < 0
            or experiment.mixed.graph_document(parsed) is None
        ):
            raise ValueError("Parser reconstruction mismatch")
        checked += len(rebuilt)
    return checked


def audit(folder=experiment.OUTPUT):
    manifest = experiment.check(folder)
    acquisition_attempts = audit_acquisition(folder, manifest)
    sentences, line_mappings = audit_documents(folder, manifest)
    if len(manifest["sources"]) < experiment.DOCUMENTS:
        if (
            list((folder / "calls").glob("*.json"))
            or list((folder / "reviews").glob("*.json"))
            or list((folder / "coverage").glob("*.json"))
            or list((folder / "parser").glob("*.json"))
        ):
            raise ValueError("Corpus-gate failure used downstream computation")
        summary = experiment.summarize(folder)
        result = {
            "status": "passed_corpus_gate_audit",
            "mismatches": 0,
            "source_acquisition_attempts_checked": acquisition_attempts,
            "selected_documents_reconstructed": len(manifest["sources"]),
            "window_sentences_reconstructed": sentences,
            "source_line_mappings_reconstructed": line_mappings,
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
    slots = audit_coverage(folder, manifest)
    parser_sentences = audit_parser(folder, manifest, runtime)
    summary = experiment.summarize(folder)
    if (
        summary["status"] != "complete"
        or api.sent != 0
        or sum(api.counters.values()) != experiment.API_CAP
        or summary["api_attempts"] != experiment.API_CAP
        or summary["neural_parser_sentences"] != parser_sentences
        or summary["decision"]
        not in {
            "stop_quality_first_window_route",
            "quality_first_windows_support_partial_injection_preflight",
        }
    ):
        raise ValueError("Summary accounting mismatch")
    result = {
        "status": "passed",
        "mismatches": 0,
        "source_acquisition_attempts_checked": acquisition_attempts,
        "selected_documents_reconstructed": experiment.DOCUMENTS,
        "window_sentences_reconstructed": sentences,
        "source_line_mappings_reconstructed": line_mappings,
        "short_slots_reconstructed": slots,
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
    for subfolder in ("calls", "generated", "reviews", "coverage", "parser"):
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
