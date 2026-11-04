"""Run one corrected source-anchored target-47 SWAN preflight slot."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

import run_swan_coherent_injection_pilot as base
import run_swan_injection_only_pilot as injection
import run_swan_normal_length_coherent_pilot as historical
from swan_anchor1_sources import (
    REVISION,
    history_snapshot,
    identity,
    screening,
    stream_rows,
)
from swan_feasibility2_common import (
    ROOT,
    digest as canonical_digest,
    freeze,
    load,
    now,
    sha256,
    single_writer,
)
from swan_final1_budget import GlobalStop, MAX_RESERVE, Pause, transport
from swan_final1_contract import environment_lock
from swan_final1_runtime import official_runtime
from swan_final1_transform import (
    FrozenSyntax,
    LEAK_RE,
    punctuation,
    sentence_spans,
    text_hash,
    unit_texts,
    unsupported,
    word_tokens,
)

OUT = ROOT / "experiments/swan_source_anchored_single_preflight2"
SOURCE_RANGE = ("train", 1_649_000, 1_650_000)
SOURCE_PATH = OUT / "source.json"
SOURCE_MANIFEST = OUT / "source_manifest.json"
SNAPSHOT_PATH = OUT / "prior_identity_snapshot.json"
SCREENING_PATH = OUT / "source_screening.json"
GLOVE = ROOT / "vectors/glove.6B.100d.txt"
EXPECTED_GLOVE = "95dde4dfd627ab26608d33e76d1195ec059734bd29089ea52cadb08d07c64544"
V1_MANIFEST = (
    ROOT / "experiments/swan_source_anchored_splittable1/manifest.json"
)
V1_RESULT = ROOT / "experiments/swan_source_anchored_splittable1/result.json"
NAMESPACE = "swan-source-anchored-single-preflight2"
TEMPLATE_SCHEDULE = (47,)
SLOT_PURPOSES = (
    "Express one explicit fact from frozen source anchor 1.",
)
FIXED_TOPIC = "One frozen source fact."
THEME_TOKENS = frozenset()
MAX_SLOTS = 1
MAX_ATTEMPTS = 30
MIN_WORDS = 5
MAX_WORDS = 12
REVIEW_COUNT = 0
MAX_CALLS = 1 + MAX_SLOTS * MAX_ATTEMPTS
MAX_COST_NANODOLLARS = MAX_CALLS * MAX_RESERVE
PILOT_CAP = (MAX_CALLS, MAX_COST_NANODOLLARS)
SOURCE_ANCHOR_INDEX = 0
COMMAND_RE = re.compile(
    r"\b(?:you|your|yours|yourself|yourselves|please|must|should|"
    r"ensure|remember)\b|(?:^|\s)(?:do not|don't|make sure|let us|let's)\b",
    re.I,
)
_SYNTAX = None


def syntax_model():
    global _SYNTAX
    if _SYNTAX is None:
        _SYNTAX = FrozenSyntax()
    return _SYNTAX


def compact_screen(row):
    return {
        key: row[key]
        for key in (
            "locator",
            "split",
            "row",
            "url_sha256",
            "normalized_source_sha256",
            "normalized_window_sha256",
            "source_word_count",
            "checks",
            "selected",
        )
    }


def check_source():
    manifest = load(SOURCE_MANIFEST)
    source = load(SOURCE_PATH)
    if manifest["dataset_revision"] != REVISION \
            or manifest["source_range"] != list(SOURCE_RANGE) \
            or manifest["replacement_allowed"] \
            or manifest["generation_parser_scoring_calls"] != 0 \
            or manifest["credential_file_access"] \
            or manifest["final1_sealed_content_access"] \
            or manifest["selected_anchor_index"] != SOURCE_ANCHOR_INDEX:
        raise ValueError("Frozen source allocation contract changed")
    if manifest["source_sha256"] != sha256(SOURCE_PATH) \
            or manifest["snapshot_sha256"] != sha256(SNAPSHOT_PATH) \
            or manifest["screening_sha256"] != sha256(SCREENING_PATH):
        raise ValueError("Frozen source allocation artifact changed")
    snapshot = load(SNAPSHOT_PATH)
    historical_text = set(snapshot["normalized_text_sha256"])
    historical_url = set(snapshot["url_sha256"])
    source_ids = {
        source["normalized_source_sha256"],
        source["normalized_window_sha256"],
        *(identity(anchor) for anchor in source["source_anchors"]),
    }
    if source["split"] != SOURCE_RANGE[0] \
            or not SOURCE_RANGE[1] <= source["row"] < SOURCE_RANGE[2] \
            or not source["selected"] \
            or not all(source["checks"].values()) \
            or len(source["source_anchors"]) != 10 \
            or not 100 <= source["source_word_count"] <= 220 \
            or source_ids & historical_text \
            or source["url_sha256"] in historical_url:
        raise ValueError("Frozen source no longer passes allocation gates")
    return {
        "state": "source_hashes_verified",
        "source_sha256": sha256(SOURCE_PATH),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "locator": source["locator"],
        "selected_anchor_index": SOURCE_ANCHOR_INDEX,
        "selected_anchor_sha256": text_hash(
            source["source_anchors"][SOURCE_ANCHOR_INDEX]
        ),
    }


def allocate_source():
    if SOURCE_MANIFEST.exists():
        return check_source()
    snapshot = load(SNAPSHOT_PATH) if SNAPSHOT_PATH.exists() else freeze(
        SNAPSHOT_PATH,
        history_snapshot(excluded=OUT),
    )
    seen_text = set(snapshot["normalized_text_sha256"])
    seen_url = set(snapshot["url_sha256"])
    screened = []
    selected = None
    for split, index, row in stream_rows(*SOURCE_RANGE):
        result = screening(row, split, index, seen_text, seen_url)
        screened.append(compact_screen(result))
        if result["selected"]:
            selected = {
                **result,
                "allocation_id": "source-anchored-single-preflight2-0000",
                "dataset_revision": REVISION,
                "selected_anchor_index": SOURCE_ANCHOR_INDEX,
            }
            break
    if selected is None:
        raise RuntimeError("Frozen source scan exhausted without replacement")
    freeze(SOURCE_PATH, selected)
    freeze(SCREENING_PATH, screened)
    freeze(SOURCE_MANIFEST, {
        "created_at": now(),
        "dataset_revision": REVISION,
        "dataset_endpoint": "https://datasets-server.huggingface.co/rows",
        "source_range": list(SOURCE_RANGE),
        "selection_rule": (
            "first unseen first-qualifying contiguous ten-sentence window"
        ),
        "window_rule": (
            "10 sentences; each 8-35 words; total 100-220 words"
        ),
        "selected_anchor_rule": "anchor index 0 fixed before generation",
        "selected_anchor_index": SOURCE_ANCHOR_INDEX,
        "replacement_allowed": False,
        "rows_screened": len(screened),
        "selected_locator": selected["locator"],
        "snapshot_sha256": sha256(SNAPSHOT_PATH),
        "screening_sha256": sha256(SCREENING_PATH),
        "source_sha256": sha256(SOURCE_PATH),
        "generation_parser_scoring_calls": 0,
        "credential_file_access": False,
        "final1_sealed_content_access": False,
    })
    return check_source()


def source_record():
    check_source()
    return load(SOURCE_PATH)


def pilot_digest(value):
    if isinstance(value, list) and value \
            and value[0] == "swan_coherent_injection_pilot1":
        value = [NAMESPACE, *value[1:]]
    return canonical_digest(value)


def historical_hashes():
    check_source()
    return set(load(SNAPSHOT_PATH)["normalized_text_sha256"])


def context_for(slot, previous):
    if slot != 0 or previous:
        raise ValueError("Single preflight accepts only empty slot-0 context")
    source = source_record()
    anchor = source["source_anchors"][SOURCE_ANCHOR_INDEX]
    return (
        "Write one source-grounded factual sentence.\n"
        f"FROZEN SOURCE WINDOW:\n{source['source_window']}\n"
        f"ASSIGNED SOURCE ANCHOR 1/10:\n{anchor}\n"
        "The target AMR is mandatory. State one explicit fact from the assigned "
        "anchor without inference. Copy names and numbers exactly. Use one "
        "complete simple declarative sentence of 5 to 12 words. Begin with "
        "ordinary uppercase orthography and end in one period. Use no comma, "
        "colon, semicolon, question mark, or exclamation mark."
    )


def orthographic_initial_capital(text):
    match = re.search(r"[A-Za-z]", text)
    return bool(match and match.group(0).isupper())


def surface_result_with_syntax(
    text,
    previous,
    forbidden_hashes,
    source,
    syntax,
):
    text = text.strip()
    if previous:
        raise ValueError("Single preflight has no prior selected sentence")
    anchor = source["source_anchors"][SOURCE_ANCHOR_INDEX]
    spans = sentence_spans(text) if text else []
    candidate_info = syntax(text) if text else {
        "complete": False,
        "content_lemmas": [],
    }
    anchor_info = syntax(anchor)
    overlap = sorted(
        set(candidate_info["content_lemmas"])
        & set(anchor_info["content_lemmas"])
    )
    marks = punctuation(text)
    checks = {
        "nonempty": bool(text),
        "one_punkt_unit": len(spans) == 1
        and unit_texts(text, spans) == [text],
        "terminal_single_period": text.endswith(".")
        and marks.get(".", 0) == 1,
        "word_count_5_12": MIN_WORDS <= len(word_tokens(text)) <= MAX_WORDS,
        "complete_subject_finite_predicate": candidate_info["complete"] is True,
        "orthographic_initial_capital": orthographic_initial_capital(text),
        "simple_punctuation": not any(mark in text for mark in ",;:!?"),
        "supported_punctuation": unsupported(text) is None,
        "no_second_person_or_command": not bool(COMMAND_RE.search(text)),
        "no_amr_or_placeholder_leak": not bool(LEAK_RE.search(text)),
        "at_least_two_anchor_content_lemmas": len(overlap) >= 2,
        "not_previously_observed_text": identity(text) not in forbidden_hashes,
    }
    return {
        **checks,
        "pass": all(checks.values()),
        "word_count": len(word_tokens(text)),
        "source_anchor_index": SOURCE_ANCHOR_INDEX,
        "source_anchor_sha256": text_hash(anchor),
        "anchor_content_lemma_overlap": overlap,
        "candidate_syntax": candidate_info,
        "anchor_syntax": anchor_info,
    }


def surface_result(text, previous, forbidden_hashes):
    return surface_result_with_syntax(
        text,
        previous,
        forbidden_hashes,
        source_record(),
        syntax_model(),
    )


def budget():
    return {
        "currency": "USD",
        "accounting_unit": "nanodollar",
        "model": base.MODEL,
        "pricing_basis": "peak cache-miss upper bound",
        "input_nanodollars_per_token": 300,
        "output_nanodollars_per_token": 1200,
        "max_request_bytes": 8192,
        "input_overhead_tokens": 1024,
        "max_output_tokens": 50,
        "max_call_reserve_nanodollars": MAX_RESERVE,
        "max_slots": MAX_SLOTS,
        "max_attempts_per_slot": MAX_ATTEMPTS,
        "model_identity_probe_calls": 1,
        "auxiliary_review_calls": 0,
        "max_calls": MAX_CALLS,
        "max_cost_nanodollars": MAX_COST_NANODOLLARS,
        "max_cost_usd": MAX_COST_NANODOLLARS / 1e9,
        "transport_retries": 0,
        "uncertain_calls_retain_full_reserve": True,
    }


def dependencies():
    paths = [
        Path(__file__),
        OUT / "protocol.md",
        OUT / "current_price_evidence.md",
        SOURCE_PATH,
        SOURCE_MANIFEST,
        SNAPSHOT_PATH,
        SCREENING_PATH,
        V1_MANIFEST,
        V1_RESULT,
        base.BANK,
        base.EXAMPLES,
        GLOVE,
        ROOT / "scripts/run_swan_coherent_injection_pilot.py",
        ROOT / "scripts/run_swan_normal_length_coherent_pilot.py",
        ROOT / "scripts/run_swan_injection_only_pilot.py",
        ROOT / "scripts/run_swan_positive_collection.py",
        ROOT / "scripts/swan_anchor1_sources.py",
        ROOT / "scripts/swan_feasibility2_common.py",
        ROOT / "scripts/swan_final1_budget.py",
        ROOT / "scripts/swan_final1_contract.py",
        ROOT / "scripts/swan_final1_runtime.py",
        ROOT / "scripts/swan_final1_transform.py",
        ROOT / "scripts/test_swan_source_anchored_single_preflight.py",
    ]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def frozen_design():
    source = source_record()
    anchor = source["source_anchors"][SOURCE_ANCHOR_INDEX]
    return {
        "purpose": "corrected one-slot prerequisite after V1 gate mismatch",
        "source_sha256": sha256(SOURCE_PATH),
        "source_window_sha256": text_hash(source["source_window"]),
        "selected_anchor_index": SOURCE_ANCHOR_INDEX,
        "selected_anchor_sha256": text_hash(anchor),
        "template_schedule": list(TEMPLATE_SCHEDULE),
        "target_selection_limitation": "single favorable target 47",
        "generation": {
            "model": base.MODEL,
            "thinking": "disabled",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_output_tokens": 50,
        },
        "surface": {
            "min_words": MIN_WORDS,
            "max_words": MAX_WORDS,
            "minimum_anchor_content_lemma_overlap": 2,
            "initial_capitalization": (
                "first ASCII alphabetic character is uppercase"
            ),
            "proper_noun_first_token_required": False,
            "historically_observed_text_forbidden": True,
        },
        "fallback_allowed": False,
        "document_generation": False,
        "quality_review": False,
        "detection": False,
        "attack": False,
    }


def prepare():
    if (OUT / "manifest.json").exists():
        return check()
    check_source()
    if sha256(base.BANK) != base.EXPECTED_BANK:
        raise ValueError("Bank50 changed")
    if sha256(GLOVE) != EXPECTED_GLOVE:
        raise ValueError("GloVe vector file changed")
    if load(V1_RESULT)["status"] != "slot_exhausted":
        raise ValueError("V1 terminal status changed")
    historical_hashes()
    rotation_hash = sha256(base.ROTATION_ATTESTATION)
    freeze(OUT / "budget_estimate.json", budget())
    freeze(OUT / "environment_lock.json", environment_lock())
    freeze(OUT / "design.json", frozen_design())
    manifest = freeze(OUT / "manifest.json", {
        "schema_version": 1,
        "experiment": NAMESPACE,
        "frozen_at": now(),
        "status": "frozen_preexecution",
        "development_preflight": True,
        "implementation_repair_exception": True,
        "v1_manifest_sha256": sha256(V1_MANIFEST),
        "v1_result_sha256": sha256(V1_RESULT),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "source_sha256": sha256(SOURCE_PATH),
        "slots": MAX_SLOTS,
        "max_attempts_per_slot": MAX_ATTEMPTS,
        "match_threshold": base.MATCH_THRESHOLD,
        "bank_sha256": base.EXPECTED_BANK,
        "glove_sha256": EXPECTED_GLOVE,
        "budget_sha256": sha256(OUT / "budget_estimate.json"),
        "environment_sha256": sha256(OUT / "environment_lock.json"),
        "design_sha256": sha256(OUT / "design.json"),
        "rotation_attestation_path": str(
            base.ROTATION_ATTESTATION.relative_to(ROOT)
        ),
        "rotation_attestation_sha256": rotation_hash,
        "dependencies": dependencies(),
        "new_output_observations_at_freeze": 0,
        "fallback_allowed": False,
        "credential_file_access": False,
        "final1_sealed_content_access": False,
        "source_anchoring_protocol": True,
        "formal_source_anchoring_claim": False,
        "document_claim": False,
        "human_quality_claim": False,
        "empirical_calibration_claim": False,
        "detection_authorized": False,
        "attack_authorized": False,
        "automatic_expansion_authorized": False,
    })
    freeze(OUT / "authorization_template.json", {
        "phase": "pilot",
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "budget_sha256": sha256(OUT / "budget_estimate.json"),
        "max_calls": MAX_CALLS,
        "max_cost_nanodollars": MAX_COST_NANODOLLARS,
        "authorized_by": None,
        "authorization_text": None,
        "key_rotation_attestation_source_sha256": rotation_hash,
        "authorized_at": None,
    })
    return {
        "state": manifest["status"],
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "source_sha256": sha256(SOURCE_PATH),
        "max_calls": MAX_CALLS,
        "max_cost_usd": MAX_COST_NANODOLLARS / 1e9,
        "design_sha256": sha256(OUT / "design.json"),
    }


def check(full_environment=True):
    check_source()
    manifest = load(OUT / "manifest.json")
    valid = (
        manifest["experiment"] == NAMESPACE
        and manifest["development_preflight"]
        and manifest["implementation_repair_exception"]
        and manifest["v1_manifest_sha256"] == sha256(V1_MANIFEST)
        and manifest["v1_result_sha256"] == sha256(V1_RESULT)
        and manifest["source_manifest_sha256"] == sha256(SOURCE_MANIFEST)
        and manifest["source_sha256"] == sha256(SOURCE_PATH)
        and manifest["slots"] == MAX_SLOTS
        and manifest["max_attempts_per_slot"] == MAX_ATTEMPTS
        and manifest["match_threshold"] == base.MATCH_THRESHOLD
        and not manifest["fallback_allowed"]
        and not manifest["credential_file_access"]
        and not manifest["final1_sealed_content_access"]
        and manifest["source_anchoring_protocol"]
        and not manifest["formal_source_anchoring_claim"]
        and not manifest["document_claim"]
        and not manifest["human_quality_claim"]
        and not manifest["empirical_calibration_claim"]
        and not manifest["detection_authorized"]
        and not manifest["attack_authorized"]
        and not manifest["automatic_expansion_authorized"]
        and manifest["new_output_observations_at_freeze"] == 0
    )
    if not valid:
        raise ValueError("Preflight identity or claim boundary changed")
    if sha256(base.BANK) != base.EXPECTED_BANK \
            or sha256(GLOVE) != EXPECTED_GLOVE \
            or manifest["budget_sha256"] != sha256(OUT / "budget_estimate.json") \
            or manifest["environment_sha256"] != sha256(OUT / "environment_lock.json") \
            or manifest["design_sha256"] != sha256(OUT / "design.json") \
            or manifest["rotation_attestation_sha256"] \
            != sha256(base.ROTATION_ATTESTATION):
        raise ValueError("Frozen preflight artifact changed")
    if load(OUT / "budget_estimate.json") != budget() \
            or load(OUT / "design.json") != frozen_design():
        raise ValueError("Frozen budget or design changed")
    for relative, expected in manifest["dependencies"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Dependency changed: {relative}")
    historical_hashes()
    if full_environment and environment_lock() != load(
        OUT / "environment_lock.json"
    ):
        raise ValueError("Execution environment changed")
    journal_path = OUT / "private/pilot/journal.jsonl"
    accounting = None
    if journal_path.exists():
        accounting = base.PilotJournal(
            1, None, sha256(OUT / "manifest.json"), allow_halted=True
        ).totals()
    return {
        "state": "frozen_hashes_verified",
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "environment_verified": full_environment,
        "accounting": accounting,
    }


def result_from_slots(slots, reviews, journal):
    if reviews:
        raise GlobalStop("Single preflight cannot contain review calls")
    accepted = [row for row in slots if row["accepted"]]
    if len(slots) != 1 or len(accepted) not in (0, 1):
        raise GlobalStop("Single preflight slot cardinality changed")
    status = "preflight_passed" if accepted else "slot_exhausted"
    selected = [{
        "slot": row["slot"],
        "text": row["text"],
        "text_sha256": row["text_sha256"],
        "target_template_index": row["target_template_index"],
        "score": row["score"],
        "context_sha256": row["context_sha256"],
        "source_anchor_index": SOURCE_ANCHOR_INDEX,
        "source_anchor_sha256": text_hash(
            source_record()["source_anchors"][SOURCE_ANCHOR_INDEX]
        ),
        "slot_sha256": sha256(base.slot_path(row["slot"])),
    } for row in accepted]
    attempts = [
        load(base.attempt_path(0, attempt))
        for attempt in range(slots[0]["attempts"])
    ]
    result = {
        "experiment": NAMESPACE,
        "completed_at": now(),
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "source_sha256": sha256(SOURCE_PATH),
        "status": status,
        "slots_started": 1,
        "accepted_slots": len(accepted),
        "exhausted_slots": int(not accepted),
        "target_accepted": 1,
        "fallback_slots": 0,
        "attempts": slots[0]["attempts"],
        "surface_eligible_attempts": sum(
            row["surface"]["pass"] for row in attempts
        ),
        "parser_s2match_calls": sum(
            row["score_computed"] for row in attempts
        ),
        "selected": selected,
        "slot_sha256": {"0": sha256(base.slot_path(0))},
        "document_generated": False,
        "quality_review_calls": 0,
        "detection_calls": 0,
        "attack_run": False,
        "human_quality_reviewed": False,
        "accounting": journal.totals(),
        "prerequisite_pass": bool(accepted),
        "automatic_expansion_authorized": False,
    }
    freeze(OUT / "result.json", result)
    return result


def diagnostics_empty(row):
    diagnostics = row.get("candidate_diagnostics") or {}
    return (
        not diagnostics.get("logged_warnings")
        and not diagnostics.get("python_warnings")
        and not diagnostics.get("stderr")
        and row.get("parser_or_scorer_error_type") is None
    )


def run(max_new_calls):
    check()
    base.require_authorization()
    base.require_price()
    if (OUT / "result.json").exists():
        return load(OUT / "result.json")
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise PermissionError("DEEPSEEK_API_KEY is not set")
    if type(max_new_calls) is not int or max_new_calls < 1:
        raise ValueError("Explicit positive --max-new-calls required")

    journal = base.PilotJournal(
        max_new_calls,
        lambda body: transport(body, key),
        sha256(OUT / "manifest.json"),
    )
    probe_id = "single-preflight-model-probe"
    probe = base.response_from_journal(journal, probe_id)
    if probe is None:
        try:
            journal.request(probe_id, "Reply OK.", None, 1, 0.7, 0.9)
        except Pause:
            return {"state": "paused_before_probe", **journal.totals()}
        except Exception as exc:
            base.halt_once(
                journal, f"model_probe_failure:{type(exc).__name__}"
            )
            raise GlobalStop("Model identity probe failed") from exc

    bank = [row["normalized_template_amr"] for row in load(base.BANK)]
    examples = load(base.EXAMPLES)[:3]
    previous = []
    with official_runtime(OUT, error_folder=OUT / "errors") as module:
        if base.slot_path(0).exists():
            slot = base.verify_slot(0, previous)
        else:
            traces = []
            for attempt in range(MAX_ATTEMPTS):
                try:
                    row = base.run_attempt(
                        module,
                        journal,
                        bank,
                        examples,
                        0,
                        attempt,
                        previous,
                    )
                except Pause:
                    return {
                        "state": "paused_before_call",
                        "slot": 0,
                        "attempt": attempt,
                        "accepted_slots": 0,
                        **journal.totals(),
                    }
                traces.append(row)
                if row["selected"]:
                    break
            slot = base.finalize_slot(0, traces, previous)
    return result_from_slots([slot], [], journal)


def audit():
    check()
    result = load(OUT / "result.json")
    slot = load(base.slot_path(0))
    traces = [
        load(base.attempt_path(0, attempt))
        for attempt in range(slot["attempts"])
    ]
    failures = []
    if result["fallback_slots"] != 0 or slot["fallback"]:
        failures.append("fallback_present")
    if result["accounting"]["uncertain_or_invalid_calls"] != 0:
        failures.append("uncertain_or_invalid_call")
    if result["accounting"]["calls"] > MAX_CALLS:
        failures.append("call_cap_exceeded")
    if result["quality_review_calls"] or result["detection_calls"] \
            or result["attack_run"] or result["document_generated"]:
        failures.append("out_of_scope_stage_ran")
    for attempt, trace in enumerate(traces):
        if trace["manifest_sha256"] != sha256(OUT / "manifest.json") \
                or trace["slot"] != 0 \
                or trace["attempt"] != attempt \
                or trace["template_index"] != 47 \
                or trace["surface"]["source_anchor_index"] \
                != SOURCE_ANCHOR_INDEX:
            failures.append(f"attempt_{attempt:02d}_identity_invalid")
        if trace["score_computed"] and not diagnostics_empty(trace):
            failures.append(f"attempt_{attempt:02d}_scorer_diagnostic")
    strict_positive_found = result["status"] == "preflight_passed"
    if strict_positive_found:
        saved = result["selected"][0]
        trace = traces[slot["terminal_attempt"]]
        valid = (
            slot["accepted"] is True
            and slot["fallback"] is False
            and slot["target_template_index"] == 47
            and slot["score"] >= base.MATCH_THRESHOLD
            and saved["score"] == slot["score"] == trace["score"]
            and saved["text"] == slot["text"] == trace["candidate"]
            and trace["status"] == "accepted"
            and trace["selected"] is True
            and trace["official_score_accepted"] is True
            and trace["surface"]["pass"] is True
            and trace["surface"]["orthographic_initial_capital"] is True
            and len(
                trace["surface"]["anchor_content_lemma_overlap"]
            ) >= 2
            and diagnostics_empty(trace)
        )
        if not valid:
            failures.append("selected_strict_trace_invalid")
    elif slot["accepted"] or slot["attempts"] != MAX_ATTEMPTS:
        failures.append("negative_stop_rule_invalid")
    execution_integrity_pass = not failures
    return freeze(OUT / "audit.json", {
        "completed_at": now(),
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "result_sha256": sha256(OUT / "result.json"),
        "source_sha256": sha256(SOURCE_PATH),
        "attempt_trace_count": len(traces),
        "surface_eligible_attempts": sum(
            row["surface"]["pass"] for row in traces
        ),
        "parser_s2match_calls": sum(
            row["score_computed"] for row in traces
        ),
        "all_target_47": all(row["template_index"] == 47 for row in traces),
        "strict_positive_found": strict_positive_found,
        "selected_score": (
            result["selected"][0]["score"] if strict_positive_found else None
        ),
        "fallback_slots": result["fallback_slots"],
        "uncertain_or_invalid_calls": (
            result["accounting"]["uncertain_or_invalid_calls"]
        ),
        "out_of_scope_calls": 0,
        "failures": failures,
        "execution_integrity_pass": execution_integrity_pass,
        "preflight_pass": execution_integrity_pass and strict_positive_found,
    })


def status():
    row = {
        "experiment": NAMESPACE,
        "source_present": SOURCE_PATH.exists(),
        "manifest_present": (OUT / "manifest.json").exists(),
        "authorization_present": (OUT / "authorization.json").exists(),
        "price_check_present": (OUT / "price_check.json").exists(),
        "result_present": (OUT / "result.json").exists(),
        "audit_present": (OUT / "audit.json").exists(),
    }
    for name in ("source_manifest", "manifest", "result", "audit"):
        path = OUT / f"{name}.json"
        if path.exists():
            row[f"{name}_sha256"] = sha256(path)
    if (OUT / "result.json").exists():
        result = load(OUT / "result.json")
        row["result_status"] = result["status"]
        row["accepted_slots"] = result["accepted_slots"]
        row["fallback_slots"] = result["fallback_slots"]
        row["accounting"] = result["accounting"]
    return row


def configure_base():
    base.OUT = OUT
    base.TEMPLATE_SCHEDULE = TEMPLATE_SCHEDULE
    base.SLOT_PURPOSES = SLOT_PURPOSES
    base.FIXED_TOPIC = FIXED_TOPIC
    base.THEME_TOKENS = THEME_TOKENS
    base.MAX_SLOTS = MAX_SLOTS
    base.MAX_ATTEMPTS = MAX_ATTEMPTS
    base.MIN_WORDS = MIN_WORDS
    base.MAX_WORDS = MAX_WORDS
    base.REVIEW_COUNT = REVIEW_COUNT
    base.REVIEW_KEYS = ()
    base.MAX_CALLS = MAX_CALLS
    base.MAX_COST_NANODOLLARS = MAX_COST_NANODOLLARS
    base.PILOT_CAP = PILOT_CAP
    base.digest = pilot_digest
    base.historical_hashes = historical_hashes
    base.context_for = context_for
    base.surface_result = surface_result
    base.dependencies = dependencies
    base.frozen_design = frozen_design
    base.prepare = prepare
    base.check = check
    base.result_from_slots = result_from_slots
    base.status = status


def main():
    configure_base()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "allocate-source",
        "prepare",
        "check",
        "authorize",
        "record-price",
        "run",
        "audit",
        "status",
    ))
    parser.add_argument("--max-new-calls", type=int)
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--max-cost-usd")
    parser.add_argument("--authorized-by")
    parser.add_argument("--authorization-text")
    args = parser.parse_args()
    with single_writer(OUT):
        if args.command == "allocate-source":
            result = allocate_source()
        elif args.command == "prepare":
            result = prepare()
        elif args.command == "check":
            result = check()
        elif args.command == "authorize":
            result = base.authorize(
                args.authorized_by,
                args.authorization_text,
                args.max_calls,
                args.max_cost_usd,
            )
        elif args.command == "record-price":
            result = base.record_price()
        elif args.command == "run":
            result = run(args.max_new_calls)
        elif args.command == "audit":
            result = audit()
        else:
            result = status()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        reason = str(exc) if isinstance(
            exc,
            (GlobalStop, PermissionError, ValueError, FileNotFoundError),
        ) else "See saved local provenance; no automatic retry"
        print(json.dumps({
            "state": "stopped",
            "error_type": type(exc).__name__,
            "reason": reason,
        }), file=sys.stderr)
        raise
