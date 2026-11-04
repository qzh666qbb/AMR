"""Terminalize and audit a dependency-pruner run after an uncertain API call."""

import argparse
import json
from pathlib import Path

from audit_swan_dependency_pruner import Replay, audit_construction
from run_confirmation_experiment import digest, request_body
from run_segmentation_intervention import sha, write_json
from run_swan_dependency_pruner import (
    API_CAP,
    DOCUMENTS,
    OUTPUT,
    check,
    decode_baseline_review,
    decode_grounding_review,
    generated_path,
    grounding_body,
    quality_gates,
    runtime_path,
    style_body,
)


FAILED_SCOPE = "style-c4-validation-1646"
REVIEWED_IDS = (
    "c4-validation-1638",
    "c4-validation-1643",
    "c4-validation-1644",
)
UNREVIEWED_IDS = (
    "c4-validation-1646",
    "c4-validation-1651",
)


def call_inventory(folder):
    calls = [
        json.loads(path.read_text())
        for path in sorted((folder / "calls").glob("*.json"))
    ]
    complete = [row for row in calls if row["status"] == "complete"]
    uncertain = [
        row for row in calls if row["status"] == "failed_or_uncertain"
    ]
    if (
        len(calls) != 8
        or len(complete) != 7
        or len(uncertain) != 1
        or uncertain[0]["scope"] != FAILED_SCOPE
        or uncertain[0]["transport_attempts"] != 1
    ):
        raise ValueError("Unexpected terminal API journal inventory")
    return calls, complete, uncertain[0]


def build_terminal_summary(folder):
    manifest = check(folder)
    generated = [
        json.loads(
            generated_path(folder, source["prompt_id"]).read_text()
        )
        for source in manifest["sources"]
    ]
    construction_pass = sum(
        row["selection"]["pass"] for row in generated
    )
    if construction_pass:
        raise ValueError("Cannot terminalize: a construction passed")
    reviews = sorted((folder / "reviews").glob("*.json"))
    reviewed_ids = tuple(
        json.loads(path.read_text())["prompt_id"] for path in reviews
    )
    if reviewed_ids != REVIEWED_IDS:
        raise ValueError("Unexpected completed review cohort")
    if any(
        (folder / "reviews" / f"{prompt_id}.json").exists()
        for prompt_id in UNREVIEWED_IDS
    ):
        raise ValueError("Unexpected review after uncertain transport")
    calls, complete_calls, uncertain = call_inventory(folder)
    parser_files = sorted((folder / "parser").glob("*.json"))
    if parser_files:
        raise ValueError("Parser must not run after construction failure")
    result = {
        "status": "terminated_after_uncertain_review_call",
        "allocated_documents": DOCUMENTS,
        "construction_gate_complete": True,
        "construction_pass": construction_pass,
        "input_plans": sum(row["input_plans"] for row in generated),
        "prunable_plans": sum(row["prunable_plans"] for row in generated),
        "documents": [
            {
                "prompt_id": row["prompt_id"],
                "input_plans": row["input_plans"],
                "prunable_plans": row["prunable_plans"],
                "selected_facts": len(row["facts"]),
                "words": row["structure"]["words"],
                "construction_pass": row["selection"]["pass"],
                "quality_review": (
                    "complete"
                    if row["prompt_id"] in REVIEWED_IDS
                    else "not_completed"
                ),
            }
            for row in generated
        ],
        "quality_review": {
            "status": "incomplete_not_used_for_decision",
            "completed_documents": len(REVIEWED_IDS),
            "allocated_documents": DOCUMENTS,
            "completed_prompt_ids": list(REVIEWED_IDS),
            "unreviewed_prompt_ids": list(UNREVIEWED_IDS),
        },
        "api": {
            "cap": API_CAP,
            "attempted": len(calls),
            "complete": len(complete_calls),
            "failed_or_uncertain": 1,
            "not_sent": API_CAP - len(calls),
            "uncertain_scope": uncertain["scope"],
            "uncertain_error_type": uncertain["error_type"],
            "uncertain_elapsed_seconds": uncertain["elapsed_seconds"],
            "automatic_retry": False,
        },
        "continuation_gate": False,
        "decision": "stop_dependency_pruner_route",
        "decision_basis": (
            "All 5 allocated documents failed the deterministic 15-fact "
            "construction prerequisite before quality review or AMR parsing"
        ),
        "neural_parser_sentences": 0,
        "new_semantic_planning_calls": 0,
        "new_s2match_calls": 0,
        "new_watermark_injection_detection_or_attack_calls": 0,
        "manifest_sha256": sha(folder / "manifest.json"),
    }
    return result


def write_report(folder, result):
    lines = [
        "# SWAN dependency-path pruner terminal report",
        "",
        "The deterministic construction gate completed before the review "
        "transport interruption.",
        "",
        "| Document | Input plans | Prunable | Selected | Words | Construction | "
        "Review |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["documents"]:
        lines.append(
            f"| {row['prompt_id']} | {row['input_plans']} | "
            f"{row['prunable_plans']} | {row['selected_facts']} | "
            f"{row['words']} | "
            f"{'yes' if row['construction_pass'] else 'no'} | "
            f"{row['quality_review']} |"
        )
    lines.extend(
        [
            "",
            f"- Construction pass: **{result['construction_pass']}/{DOCUMENTS}**.",
            f"- Reviews completed: "
            f"**{result['quality_review']['completed_documents']}/{DOCUMENTS}**; "
            "the incomplete quality review is not reported as a 0/5 quality rate.",
            f"- API transport: {result['api']['complete']} complete, "
            f"{result['api']['failed_or_uncertain']} failed/uncertain, "
            f"{result['api']['not_sent']} not sent; no automatic retry.",
            "- Local AMR parser, S2Match, injection, detection, and attack calls: 0.",
            f"- Decision: `{result['decision']}` based only on the completed "
            "deterministic construction prerequisite.",
            "",
        ]
    )
    (folder / "terminal_report.md").write_text("\n".join(lines))


def finalize(folder=OUTPUT):
    result = build_terminal_summary(folder)
    write_json(folder / "terminal_summary.json", result)
    write_report(folder, result)
    return result


def audit_completed_reviews(folder, manifest, runtime):
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
    sources = {
        source["prompt_id"]: source
        for source in manifest["sources"]
    }
    for prompt_id in REVIEWED_IDS:
        source = sources[prompt_id]
        generated = json.loads(
            generated_path(folder, prompt_id).read_text()
        )
        style_request = style_body(generated["passage"])
        source_request = grounding_body(
            source["source"],
            generated["facts"],
        )
        style_response = api.request(
            f"style-{prompt_id}",
            style_request,
        )
        source_response = api.request(
            f"facts-{prompt_id}",
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
            "prompt_id": prompt_id,
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
            (folder / "reviews" / f"{prompt_id}.json").read_text()
        )
        if saved != expected:
            raise ValueError("Completed review reconstruction mismatch")
    return api


def audit(folder=OUTPUT):
    manifest = check(folder)
    runtime = json.loads(runtime_path(folder).read_text())
    input_plans, candidates, selected = audit_construction(folder, manifest)
    api = audit_completed_reviews(folder, manifest, runtime)
    _, _, uncertain = call_inventory(folder)
    prompt_id = "c4-validation-1646"
    generated = json.loads(generated_path(folder, prompt_id).read_text())
    expected_body = style_body(generated["passage"])
    if (
        uncertain["request_sha256"] != digest(expected_body)
        or uncertain["request"] != expected_body
        or uncertain["error_type"] != "TypeError"
        or "response" in uncertain
    ):
        raise ValueError("Uncertain request journal mismatch")
    if any(
        (folder / "calls" / filename).exists()
        for filename in (
            "facts-c4-validation-1646-000.json",
            "style-c4-validation-1651-000.json",
            "facts-c4-validation-1651-000.json",
        )
    ):
        raise ValueError("A request was sent after the uncertain call")
    summary = finalize(folder)
    if (
        summary["input_plans"] != input_plans
        or summary["construction_pass"] != 0
        or api.sent != 0
        or sum(api.counters.values()) != 7
    ):
        raise ValueError("Terminal summary accounting mismatch")
    result = {
        "status": "passed_terminal_audit",
        "mismatches": 0,
        "input_plans_checked": input_plans,
        "pruning_candidates_reconstructed": candidates,
        "selected_facts_reconstructed": selected,
        "complete_api_calls_replayed": 7,
        "uncertain_api_calls_verified": 1,
        "unsent_api_calls_verified": 3,
        "documents_reconstructed": DOCUMENTS,
        "new_api_calls": 0,
        "new_semantic_planning_calls": 0,
        "new_neural_parser_calls": 0,
        "new_s2match_calls": 0,
        "terminal_summary_sha256": sha(folder / "terminal_summary.json"),
    }
    write_json(folder / "terminal_audit.json", result)
    return result


def stable_files(folder):
    paths = [
        folder / "manifest.json",
        runtime_path(folder),
        folder / "summary.json",
        folder / "terminal_summary.json",
        folder / "terminal_report.md",
        folder / "terminal_audit.json",
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
    import spacy

    original_post = requests.post
    original_parser = amrlib.load_stog_model
    original_spacy = spacy.load

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Terminal recovery attempted forbidden computation")

    requests.post = forbidden
    amrlib.load_stog_model = forbidden
    spacy.load = forbidden
    try:
        result = finalize(folder)
    finally:
        requests.post = original_post
        amrlib.load_stog_model = original_parser
        spacy.load = original_spacy
    after = {
        str(path.relative_to(folder)): sha(path)
        for path in stable_files(folder)
    }
    if before != after or aggregate != digest(after):
        raise ValueError("Terminal recovery changed stable artifacts")
    verification = {
        "status": "complete_no_compute_terminal_recovery",
        "decision": result["decision"],
        "files_checked": len(before),
        "stable_aggregate_sha256": aggregate,
        "network_forbidden": True,
        "dependency_parser_forbidden": True,
        "neural_amr_parser_forbidden": True,
        "new_api_calls": 0,
        "new_semantic_planning_calls": 0,
        "new_neural_parser_calls": 0,
        "new_s2match_calls": 0,
    }
    write_json(folder / "terminal_recovery_verification.json", verification)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("finalize", "audit", "recovery"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.stage == "finalize":
        result = finalize(args.output)
    elif args.stage == "audit":
        result = audit(args.output)
    else:
        result = recovery(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
