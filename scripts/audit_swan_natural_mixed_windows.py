"""Offline audit and recovery guard for natural mixed-window preflight."""

import argparse
import json
import math
from pathlib import Path

from run_confirmation_experiment import CallJournal, digest, request_body
from run_segmentation_intervention import sha, text_sha, write_json
from run_swan_atomic_fact_preflight import analyze_parser_output, s2_counts
from run_swan_known_positive_audit import load_utils
from run_swan_natural_mixed_windows import (
    API_CAP,
    DOCUMENTS,
    OUTPUT,
    BANK_PATH,
    build_document,
    check,
    decode_baseline_review,
    freeze_documents,
    graph_document,
    runtime_path,
    style_body,
    style_gate,
    summarize,
    run,
)


class Replay(CallJournal):
    def __init__(self, folder):
        super().__init__(folder, "", API_CAP)

    def request(self, scope, body):
        ordinal = self.counters.get(scope, 0)
        path = self.folder / "calls" / f"{scope}-{ordinal:03d}.json"
        if not path.exists():
            raise ValueError("Audit replay would require an API call")
        return super().request(scope, body)


def audit_documents(folder, manifest):
    freeze_documents(folder, manifest)
    sentences = 0
    short_slots = 0
    for source in manifest["sources"]:
        expected = build_document(source)
        saved = json.loads(
            (folder / "generated" / f"{source['prompt_id']}.json").read_text()
        )
        if saved != expected:
            raise ValueError("Natural window reconstruction mismatch")
        sentences += len(expected["sentences"])
        short_slots += len(expected["facts"])
    return sentences, short_slots


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
        body = style_body(generated["passage"])
        response = api.request(f"style-{source['prompt_id']}", body)
        errors = {}
        try:
            style = decode_baseline_review(response["content"])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            style = None
            errors["style"] = type(exc).__name__
        gates = style_gate(generated, style)
        expected = {
            "prompt_id": source["prompt_id"],
            "source_sha256": source["source_sha256"],
            "passage_sha256": generated["passage_sha256"],
            "runtime_sha256": digest(runtime),
            "request_sha256": digest(body),
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


def audit_parser(folder, manifest, runtime):
    utils = load_utils()
    bank = json.loads(BANK_PATH.read_text())
    bank_counts = [
        s2_counts(row["normalized_template_amr"])
        for row in bank
    ]
    checked = 0
    for source in manifest["sources"]:
        review = json.loads(
            (folder / "reviews" / f"{source['prompt_id']}.json").read_text()
        )
        path = folder / "parser" / f"{source['prompt_id']}.json"
        if not review["quality_pass"]:
            if path.exists():
                raise ValueError("Parser exists for quality-failed window")
            continue
        parsed = json.loads(path.read_text())
        generated = json.loads(
            (folder / "generated" / f"{source['prompt_id']}.json").read_text()
        )
        rebuilt = []
        for index, (sentence, saved) in enumerate(
            zip(generated["facts"], parsed["sentences"])
        ):
            identity = {
                "sentence_index": index,
                "source_sentence_index": generated[
                    "fact_source_indices"
                ][index],
                "text": sentence,
                "text_sha256": text_sha(sentence),
                "words": len(sentence.split()),
            }
            row = analyze_parser_output(
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
            "runtime_sha256": digest(runtime),
            "sentences": rebuilt,
            "elapsed_seconds": parsed["elapsed_seconds"],
        }
        if (
            parsed != expected
            or len(rebuilt) != len(generated["facts"])
            or not math.isfinite(parsed["elapsed_seconds"])
            or parsed["elapsed_seconds"] < 0
            or graph_document(parsed) is None
        ):
            raise ValueError("Parser reconstruction mismatch")
        checked += len(rebuilt)
    return checked


def audit(folder=OUTPUT):
    manifest = check(folder)
    sentences, short_slots = audit_documents(folder, manifest)
    if len(manifest["sources"]) < DOCUMENTS:
        if (
            list((folder / "calls").glob("*.json"))
            or list((folder / "reviews").glob("*.json"))
            or list((folder / "parser").glob("*.json"))
        ):
            raise ValueError("Corpus-gate failure used downstream computation")
        summary = summarize(folder)
        if summary["decision"] != "stop_natural_mixed_window_route":
            raise ValueError("Corpus-gate decision mismatch")
        result = {
            "status": "passed_corpus_gate_audit",
            "mismatches": 0,
            "selected_documents_reconstructed": len(manifest["sources"]),
            "window_sentences_reconstructed": sentences,
            "short_slots_reconstructed": short_slots,
            "new_api_calls": 0,
            "new_neural_parser_calls": 0,
            "new_s2match_calls": 0,
            "summary_sha256": sha(folder / "summary.json"),
        }
        write_json(folder / "audit.json", result)
        return result
    runtime = json.loads(runtime_path(folder).read_text())
    api = audit_reviews(folder, manifest, runtime)
    parser_sentences = audit_parser(folder, manifest, runtime)
    summary = summarize(folder)
    if (
        summary["status"] != "complete"
        or api.sent != 0
        or sum(api.counters.values()) != API_CAP
        or summary["neural_parser_sentences"] != parser_sentences
    ):
        raise ValueError("Summary accounting mismatch")
    result = {
        "status": "passed",
        "mismatches": 0,
        "selected_documents_reconstructed": DOCUMENTS,
        "window_sentences_reconstructed": sentences,
        "short_slots_reconstructed": short_slots,
        "api_calls_replayed": API_CAP,
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
        runtime_path(folder),
    ]
    for subfolder in ("calls", "generated", "reviews", "parser"):
        paths.extend(sorted((folder / subfolder).glob("*.json")))
    return [path for path in paths if path.exists()]


def recovery(folder=OUTPUT):
    before = {
        str(path.relative_to(folder)): sha(path)
        for path in stable_files(folder)
    }
    aggregate = digest(before)
    import amrlib
    import requests

    original_post = requests.post
    original_get = requests.get
    original_parser = amrlib.load_stog_model

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Recovery attempted forbidden computation")

    requests.post = forbidden
    requests.get = forbidden
    amrlib.load_stog_model = forbidden
    try:
        result = run(folder, "")
    finally:
        requests.post = original_post
        requests.get = original_get
        amrlib.load_stog_model = original_parser
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
        "neural_amr_parser_forbidden": True,
        "new_api_calls": 0,
        "new_neural_parser_calls": 0,
        "new_s2match_calls": 0,
    }
    write_json(folder / "recovery_verification.json", verification)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "recovery"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = audit(args.output) if args.stage == "audit" else recovery(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
