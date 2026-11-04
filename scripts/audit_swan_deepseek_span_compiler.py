"""Offline audit and recovery guard for the DeepSeek span compiler."""

import argparse
import json
import math
from pathlib import Path

from run_confirmation_experiment import CallJournal, digest, request_body
from run_segmentation_intervention import sha, text_sha, write_json
from run_swan_atomic_fact_preflight import analyze_parser_output, s2_counts
from run_swan_bank100_microfact_null import BANK_PATH
from run_swan_deepseek_span_compiler import (
    API_CAP,
    BATCHES,
    DOCUMENTS,
    FACTS,
    OUTPUT,
    build_document,
    check,
    decode_baseline_review,
    decode_grounding_review,
    decode_plan_response,
    generated_path,
    grounding_body,
    plan_body,
    plan_path,
    process_batches,
    quality_gates,
    run,
    runtime_path,
    style_body,
    summarize,
)
from run_swan_known_positive_audit import load_utils


class Replay(CallJournal):
    def __init__(self, folder):
        super().__init__(folder, "", API_CAP)

    def request(self, scope, body):
        ordinal = self.counters.get(scope, 0)
        path = self.folder / "calls" / f"{scope}-{ordinal:03d}.json"
        if not path.exists():
            raise ValueError("Audit replay would require an API call")
        return super().request(scope, body)


def audit_plans(folder, manifest, api, runtime):
    import nltk

    checked_batches = 0
    checked_plans = 0
    for source in manifest["sources"]:
        selected = []
        batch_rows = []
        for batch in range(BATCHES):
            body = plan_body(source["source"], batch, selected)
            response = api.request(
                f"plan-{source['prompt_id']}-{batch:02d}",
                body,
            )
            plans, error = decode_plan_response(response)
            expected = {
                "prompt_id": source["prompt_id"],
                "batch": batch,
                "source_sha256": source["source_sha256"],
                "avoid_claims": selected,
                "avoid_claims_sha256": digest(selected),
                "request_sha256": digest(body),
                "runtime_sha256": digest(runtime),
                "response": response,
                "plans": plans,
                "decode_error": error,
            }
            saved = json.loads(
                plan_path(folder, source["prompt_id"], batch).read_text()
            )
            if saved != expected:
                raise ValueError("Planning checkpoint mismatch")
            batch_rows.append(saved)
            selected = process_batches(batch_rows, source)["selected"]
            checked_batches += 1
            checked_plans += len(plans or [])
        expected_document = build_document(
            folder,
            source,
            nltk.sent_tokenize,
        )
        saved_document = json.loads(
            generated_path(folder, source["prompt_id"]).read_text()
        )
        if saved_document != expected_document:
            raise ValueError("Compiled document mismatch")
    return checked_batches, checked_plans


def audit_reviews(folder, manifest, api, runtime):
    for source in manifest["sources"]:
        generated = json.loads(
            generated_path(folder, source["prompt_id"]).read_text()
        )
        style_request = style_body(generated["passage"])
        source_request = grounding_body(
            source["source"],
            generated["facts"],
        )
        style_response = api.request(
            f"style-{source['prompt_id']}",
            style_request,
        )
        source_response = api.request(
            f"facts-{source['prompt_id']}",
            source_request,
        )
        errors = {}
        try:
            style = decode_baseline_review(style_response["content"])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            style = None
            errors["style"] = type(exc).__name__
        try:
            grounding = decode_grounding_review(source_response["content"])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            grounding = None
            errors["grounding"] = type(exc).__name__
        gates = quality_gates(generated, style, grounding)
        expected = {
            "prompt_id": source["prompt_id"],
            "source_sha256": source["source_sha256"],
            "passage_sha256": generated["passage_sha256"],
            "runtime_sha256": digest(runtime),
            "style_request_sha256": digest(style_request),
            "grounding_request_sha256": digest(source_request),
            "style_response": style_response,
            "grounding_response": source_response,
            "style": style,
            "grounding": grounding,
            "decode_errors": errors,
            "gates": gates,
            "quality_pass": all(gates.values()),
        }
        saved = json.loads(
            (folder / "reviews" / f"{source['prompt_id']}.json").read_text()
        )
        if saved != expected:
            raise ValueError("Quality review checkpoint mismatch")


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
                raise ValueError("Parser exists for quality-failed document")
            continue
        parsed = json.loads(path.read_text())
        generated = json.loads(
            generated_path(folder, source["prompt_id"]).read_text()
        )
        rebuilt = []
        for index, (sentence, saved) in enumerate(
            zip(generated["facts"], parsed["sentences"])
        ):
            identity = {
                "sentence_index": index,
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
            or len(rebuilt) != FACTS
            or not math.isfinite(parsed["elapsed_seconds"])
            or parsed["elapsed_seconds"] < 0
        ):
            raise ValueError("Parser reconstruction mismatch")
        checked += len(rebuilt)
    return checked


def audit(folder=OUTPUT):
    manifest = check(folder)
    runtime = json.loads(runtime_path(folder).read_text())
    calls = sorted((folder / "calls").glob("*.json"))
    if len(calls) != API_CAP:
        raise ValueError("Unexpected API journal count")
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
    batches, plans = audit_plans(folder, manifest, api, runtime)
    audit_reviews(folder, manifest, api, runtime)
    parser_sentences = audit_parser(folder, manifest, runtime)
    summary = summarize(folder)
    if (
        summary["status"] != "complete"
        or summary["planning_batches"] != batches
        or summary["decoded_plans"] != plans
        or summary["api_attempts"] != len(calls)
        or summary["neural_parser_sentences"] != parser_sentences
        or sum(api.counters.values()) != len(calls)
        or api.sent != 0
        or summary["new_s2match_calls"] != 0
    ):
        raise ValueError("Summary accounting mismatch")
    result = {
        "status": "passed",
        "mismatches": 0,
        "api_calls_replayed": len(calls),
        "planning_batches_checked": batches,
        "decoded_plans_recompiled": plans,
        "documents_reconstructed": DOCUMENTS,
        "parser_sentences_reconstructed": parser_sentences,
        "new_api_calls": api.sent,
        "new_neural_parser_calls": 0,
        "new_s2match_calls": 0,
        "summary_sha256": sha(folder / "summary.json"),
    }
    write_json(folder / "audit.json", result)
    return result


def stable_files(folder):
    paths = [
        folder / "manifest.json",
        runtime_path(folder),
        folder / "summary.json",
        folder / "report.md",
        folder / "audit.json",
    ]
    for subfolder in ("calls", "plans", "generated", "reviews", "parser"):
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
    original_parser = amrlib.load_stog_model

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Recovery attempted forbidden computation")

    requests.post = forbidden
    amrlib.load_stog_model = forbidden
    try:
        result = run(folder, "")
    finally:
        requests.post = original_post
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
        "neural_parser_forbidden": True,
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
