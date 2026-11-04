"""Collect ten genuine SWAN-positive sentences without prose constraints."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr
from datetime import datetime, timezone
from decimal import Decimal
import io
import json
import logging
import math
import os
from pathlib import Path
import sys
import warnings

import run_swan_positive_collection as swan
from swan_feasibility2_common import (
    ROOT,
    digest,
    freeze,
    load,
    now,
    sha256,
    single_writer,
)
from swan_final1_budget import (
    GlobalStop,
    Journal as BaseJournal,
    MAX_RESERVE,
    MODEL,
    Pause,
    transport,
)
from swan_final1_contract import environment_lock
from swan_final1_quality import parse_utc
from swan_final1_runtime import official_runtime
from swan_final1_transform import LEAK_RE, sentence_spans, text_hash, unit_texts

OUT = ROOT / "experiments/swan_injection_only_pilot1"
BANK = ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_50.json"
EXAMPLES = BANK.parent.parent / "artifacts/normalized_amr_examples.json"
ROTATION_ATTESTATION = (
    ROOT / "experiments/swan_normal_length_feasibility2/authorization_pilot.json"
)
EXPECTED_BANK = "f0a88c1af73d12078586d10ef68804c12ed5b79cabc07c9f6c2a34bb084ef5a4"
TEMPLATE_SCHEDULE = (27, 11, 47, 14, 43, 25, 37, 22, 45, 13)
MAX_SLOTS = 20
MAX_ATTEMPTS = 30
TARGET_ACCEPTED = 10
MAX_CALLS = 1 + MAX_SLOTS * MAX_ATTEMPTS
MAX_COST_NANODOLLARS = MAX_CALLS * MAX_RESERVE
PILOT_CAP = (MAX_CALLS, MAX_COST_NANODOLLARS)
MATCH_THRESHOLD = 0.75
PARSER_LOGGERS = (
    "amrlib.models.parse_xfm.penman_serializer",
    "utils.amr_utils",
)


def target_for(slot):
    if type(slot) is not int or not 0 <= slot < MAX_SLOTS:
        raise ValueError("Slot outside frozen schedule")
    return TEMPLATE_SCHEDULE[slot % len(TEMPLATE_SCHEDULE)]


def surface_result(text):
    text = text.strip()
    spans = sentence_spans(text)
    checks = {
        "nonempty": bool(text),
        "one_punkt_unit": len(spans) == 1 and unit_texts(text, spans) == [text],
        "no_amr_or_placeholder_leak": not bool(LEAK_RE.search(text)),
    }
    checks["pass"] = all(checks.values())
    return checks


class DiagnosticHandler(logging.Handler):
    def __init__(self, records):
        super().__init__(level=logging.WARNING)
        self.records = records

    def emit(self, record):
        self.records.append({
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        })


@contextmanager
def capture_candidate_diagnostics():
    diagnostics = {
        "logged_warnings": [],
        "python_warnings": [],
        "stderr": "",
    }
    handler = DiagnosticHandler(diagnostics["logged_warnings"])
    loggers = [logging.getLogger(name) for name in PARSER_LOGGERS]
    stderr = io.StringIO()
    for logger in loggers:
        logger.addHandler(handler)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                with redirect_stderr(stderr):
                    yield diagnostics
            finally:
                diagnostics["python_warnings"] = [{
                    "category": warning.category.__name__,
                    "message": str(warning.message),
                    "filename": warning.filename,
                    "lineno": warning.lineno,
                } for warning in caught]
    finally:
        diagnostics["stderr"] = stderr.getvalue()
        for logger in loggers:
            logger.removeHandler(handler)


class PilotJournal(BaseJournal):
    def __init__(self, max_new_calls, send, contract_hash, allow_halted=False):
        super().__init__(
            OUT,
            "pilot",
            max_new_calls,
            send,
            contract_hash,
            allow_halted=allow_halted,
        )

    def capacity(self, calls=1):
        totals = self.totals()
        if self.new_calls + calls > self.limit:
            return "invocation_cap"
        if totals["calls"] + calls > MAX_CALLS:
            return "phase_call_cap"
        if totals["exposure_nanodollars"] + calls * MAX_RESERVE \
                > MAX_COST_NANODOLLARS:
            return "phase_money_cap"
        return None


def dependencies():
    paths = [
        Path(__file__),
        OUT / "protocol.md",
        OUT / "current_price_evidence.md",
        BANK,
        EXAMPLES,
        ROOT / "scripts/run_swan_positive_collection.py",
        ROOT / "scripts/swan_feasibility2_common.py",
        ROOT / "scripts/swan_final1_budget.py",
        ROOT / "scripts/swan_final1_contract.py",
        ROOT / "scripts/swan_final1_quality.py",
        ROOT / "scripts/swan_final1_runtime.py",
        ROOT / "scripts/swan_final1_transform.py",
        ROOT / "scripts/test_swan_injection_only_pilot.py",
    ]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def budget():
    return {
        "currency": "USD",
        "accounting_unit": "nanodollar",
        "model": MODEL,
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
        "max_calls": MAX_CALLS,
        "max_cost_nanodollars": MAX_COST_NANODOLLARS,
        "max_cost_usd": MAX_COST_NANODOLLARS / 1e9,
        "transport_retries": 0,
        "uncertain_calls_retain_full_reserve": True,
    }


def frozen_design():
    return {
        "optional_context": "",
        "template_schedule": list(TEMPLATE_SCHEDULE),
        "target_by_slot": {
            str(slot): target_for(slot) for slot in range(MAX_SLOTS)
        },
        "examples": "first_three_frozen_normalized_examples",
        "generation": {
            "model": MODEL,
            "thinking": "disabled",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_output_tokens": 50,
        },
    }


def prepare():
    if (OUT / "manifest.json").exists():
        return check()
    if sha256(BANK) != EXPECTED_BANK:
        raise ValueError("Bank50 changed")
    rotation_hash = sha256(ROTATION_ATTESTATION)
    freeze(OUT / "budget_estimate.json", budget())
    freeze(OUT / "environment_lock.json", environment_lock())
    freeze(OUT / "design.json", frozen_design())
    manifest = freeze(OUT / "manifest.json", {
        "schema_version": 1,
        "experiment": "swan_injection_only_pilot1",
        "frozen_at": now(),
        "status": "frozen_preexecution",
        "development_pilot": True,
        "target_accepted": TARGET_ACCEPTED,
        "max_slots": MAX_SLOTS,
        "max_attempts_per_slot": MAX_ATTEMPTS,
        "match_threshold": MATCH_THRESHOLD,
        "bank_sha256": EXPECTED_BANK,
        "budget_sha256": sha256(OUT / "budget_estimate.json"),
        "environment_sha256": sha256(OUT / "environment_lock.json"),
        "design_sha256": sha256(OUT / "design.json"),
        "rotation_attestation_path": str(ROTATION_ATTESTATION.relative_to(ROOT)),
        "rotation_attestation_sha256": rotation_hash,
        "dependencies": dependencies(),
        "new_output_observations_at_freeze": 0,
        "fallback_allowed": False,
        "credential_file_access": False,
        "sealed_source_access": False,
        "coherence_constraint": False,
        "source_anchoring_constraint": False,
        "split_constraint": False,
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
        "max_calls": MAX_CALLS,
        "max_cost_usd": budget()["max_cost_usd"],
        "design_sha256": sha256(OUT / "design.json"),
    }


def check(full_environment=True):
    manifest = load(OUT / "manifest.json")
    identity_ok = (
        manifest["experiment"] == "swan_injection_only_pilot1"
        and manifest["development_pilot"]
        and manifest["target_accepted"] == TARGET_ACCEPTED
        and manifest["max_slots"] == MAX_SLOTS
        and manifest["max_attempts_per_slot"] == MAX_ATTEMPTS
        and manifest["match_threshold"] == MATCH_THRESHOLD
        and not manifest["fallback_allowed"]
        and not manifest["credential_file_access"]
        and not manifest["sealed_source_access"]
        and not manifest["coherence_constraint"]
        and not manifest["source_anchoring_constraint"]
        and not manifest["split_constraint"]
        and manifest["new_output_observations_at_freeze"] == 0
    )
    if not identity_ok:
        raise ValueError("Pilot identity or isolation contract changed")
    if sha256(BANK) != EXPECTED_BANK \
            or manifest["budget_sha256"] != sha256(OUT / "budget_estimate.json") \
            or manifest["environment_sha256"] != sha256(OUT / "environment_lock.json") \
            or manifest["design_sha256"] != sha256(OUT / "design.json") \
            or manifest["rotation_attestation_sha256"] != sha256(ROTATION_ATTESTATION):
        raise ValueError("Frozen pilot artifact changed")
    if load(OUT / "budget_estimate.json") != budget():
        raise ValueError("Budget calculation changed")
    if load(OUT / "design.json") != frozen_design():
        raise ValueError("Prompt or target schedule changed")
    for relative, expected in manifest["dependencies"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Dependency changed: {relative}")
    if full_environment and environment_lock() != load(OUT / "environment_lock.json"):
        raise ValueError("Execution environment changed")
    journal_path = OUT / "private/pilot/journal.jsonl"
    accounting = None
    if journal_path.exists():
        accounting = PilotJournal(
            1, None, sha256(OUT / "manifest.json"), allow_halted=True
        ).totals()
    return {
        "state": "frozen_hashes_verified",
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "environment_verified": full_environment,
        "accounting": accounting,
    }


def rotation_attestation():
    row = load(ROTATION_ATTESTATION)
    if row.get("key_rotation_attested") is not True \
            or row.get("status") != "explicit_pilot_authorization" \
            or row.get("max_calls", 0) < MAX_CALLS \
            or row.get("max_cost_nanodollars", 0) < MAX_COST_NANODOLLARS:
        raise PermissionError("Prior key-rotation attestation is insufficient")
    return row


def authorize(who, text, calls, dollars):
    check()
    if not who or not who.strip() or not text or not text.strip():
        raise PermissionError("Explicit authorizer identity and consent text required")
    if calls != MAX_CALLS \
            or Decimal(dollars or "0") * 1_000_000_000 != MAX_COST_NANODOLLARS:
        raise PermissionError("Authorization must exactly match the frozen pilot cap")
    prior = rotation_attestation()
    return freeze(OUT / "authorization.json", {
        "phase": "pilot",
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "budget_sha256": sha256(OUT / "budget_estimate.json"),
        "max_calls": MAX_CALLS,
        "max_cost_nanodollars": MAX_COST_NANODOLLARS,
        "authorized_by": who.strip(),
        "authorization_text": text.strip(),
        "key_rotation_attested": True,
        "key_rotation_attestation_source": str(ROTATION_ATTESTATION.relative_to(ROOT)),
        "key_rotation_attestation_source_sha256": sha256(ROTATION_ATTESTATION),
        "key_rotation_attested_at": prior["authorized_at"],
        "authorized_at": now(),
        "status": "explicit_pilot_authorization",
    })


def require_authorization():
    path = OUT / "authorization.json"
    if not path.is_file():
        raise PermissionError("No immutable pilot budget authorization")
    row = load(path)
    manifest = load(OUT / "manifest.json")
    valid = (
        row.get("manifest_sha256") == sha256(OUT / "manifest.json")
        and row.get("budget_sha256") == sha256(OUT / "budget_estimate.json")
        and row.get("status") == "explicit_pilot_authorization"
        and (row.get("max_calls"), row.get("max_cost_nanodollars")) == PILOT_CAP
        and row.get("key_rotation_attested") is True
        and row.get("key_rotation_attestation_source_sha256")
        == sha256(ROTATION_ATTESTATION)
        and bool(row.get("authorized_by", "").strip())
        and bool(row.get("authorization_text", "").strip())
    )
    if not valid:
        raise PermissionError("Authorization does not match frozen contract")
    if parse_utc(row["authorized_at"]) < parse_utc(manifest["frozen_at"]):
        raise PermissionError("Historical consent cannot authorize this contract")
    rotation_attestation()
    return row


def record_price():
    check(full_environment=False)
    raw = (OUT / "current_price_evidence.md").read_text(encoding="utf-8")
    required = (
        "https://api-docs.deepseek.com/quick_start/pricing",
        "Model: deepseek-flash",
        "USD 0.30",
        "USD 1.20",
    )
    if any(value not in raw for value in required):
        raise ValueError("Official price evidence is incomplete")
    return freeze(OUT / "price_check.json", {
        "checked_at": now(),
        "source_url": required[0],
        "model": MODEL,
        "input_price_usd_per_million": "0.30",
        "output_price_usd_per_million": "1.20",
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "evidence_sha256": sha256(OUT / "current_price_evidence.md"),
    })


def require_price():
    path = OUT / "price_check.json"
    if not path.is_file():
        raise PermissionError("Current official pricing has not been recorded")
    row = load(path)
    age = (datetime.now(timezone.utc) - parse_utc(row["checked_at"])).total_seconds()
    valid = (
        0 <= age <= 24 * 3600
        and row["manifest_sha256"] == sha256(OUT / "manifest.json")
        and row["evidence_sha256"] == sha256(OUT / "current_price_evidence.md")
        and Decimal(row["input_price_usd_per_million"]) == Decimal(".30")
        and Decimal(row["output_price_usd_per_million"]) == Decimal("1.20")
    )
    if not valid:
        raise PermissionError("Official price evidence is stale or changed")
    return row


def response_from_journal(journal, call_id):
    call = journal.calls.get(call_id)
    if call is None:
        return None
    terminal = call["terminal"]
    if terminal is None or terminal["kind"] != "complete":
        raise GlobalStop("Existing call is not safely reusable")
    path = journal.folder / "responses" / f"{call_id}.json"
    parsed = journal._parse(load(path), call["reserve"]["reserve"])
    return {**parsed, "call_sha256": sha256(path), "replayed": True}


def score_candidate(module, candidate, target, score_seed):
    raw = module.parse_amr(candidate)
    normalized = module.normalize_amr_variables(raw)
    templated = module.generate_template_amr(normalized)
    final = module.normalize_amr_variables(templated)
    with swan.fixed_score_seed(score_seed):
        accepted, score = module.decide_s2match_match(
            final, target, MATCH_THRESHOLD
        )
    score = float(score)
    if not math.isfinite(score) or not 0 <= score <= 1 \
            or bool(accepted) != (score >= MATCH_THRESHOLD):
        raise GlobalStop("Invalid official S2Match decision")
    return {
        "raw_amr": raw,
        "raw_amr_sha256": text_hash(raw),
        "normalized_amr": normalized,
        "normalized_amr_sha256": text_hash(normalized),
        "candidate_template_before_final_normalization": templated,
        "candidate_template_before_final_normalization_sha256": text_hash(templated),
        "candidate_template_amr": final,
        "candidate_template_amr_sha256": text_hash(final),
        "score_seed": score_seed,
        "threshold": MATCH_THRESHOLD,
        "score": score,
        "official_score_accepted": bool(accepted),
    }


def score_with_diagnostics(module, candidate, target, score_seed):
    diagnostics = None
    scored = None
    error_type = None
    error_message = None
    try:
        with capture_candidate_diagnostics() as diagnostics:
            scored = score_candidate(module, candidate, target, score_seed)
    except GlobalStop as exc:
        exc.candidate_diagnostics = diagnostics
        raise
    except Exception as exc:
        error_type = type(exc).__name__
        error_message = str(exc)
    diagnostics_present = bool(
        diagnostics["logged_warnings"]
        or diagnostics["python_warnings"]
        or diagnostics["stderr"]
    )
    return {
        "scored": scored,
        "diagnostics": diagnostics,
        "diagnostics_present": diagnostics_present,
        "error_type": error_type,
        "error_message": error_message,
        "clean": scored is not None and error_type is None and not diagnostics_present,
    }


def attempt_path(slot, attempt):
    return OUT / "private/pilot/generation_attempts" / (
        f"injection-only-slot-{slot:02d}-attempt-{attempt:02d}.json"
    )


def slot_path(slot):
    return OUT / "private/pilot/generation_slots" / (
        f"injection-only-slot-{slot:02d}.json"
    )


def halt_once(journal, reason):
    if not any(event["kind"] == "halt" for event in journal.events):
        journal.event("halt", reason=reason)


def run_attempt(module, journal, bank, examples, slot, attempt, selected_hashes):
    target_index = target_for(slot)
    target = bank[target_index]
    context = ""
    user_prompt = module.build_user_prompt(target, examples, context)
    system_prompt = module.build_system_prompt()
    call_id = f"injection-only-slot-{slot:02d}-attempt-{attempt:02d}"
    path = attempt_path(slot, attempt)
    base = {
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "slot": slot,
        "attempt": attempt,
        "call_id": call_id,
        "context": context,
        "context_sha256": text_hash(context),
        "template_index": target_index,
        "template": target,
        "template_sha256": text_hash(target),
        "user_prompt_sha256": text_hash(user_prompt),
        "system_prompt_sha256": text_hash(system_prompt),
    }
    if path.exists():
        row = load(path)
        if any(row.get(key) != value for key, value in base.items()):
            raise GlobalStop("Saved attempt identity changed")
        if row.get("selected") and row.get("candidate_sha256") in selected_hashes:
            raise GlobalStop("Saved accepted attempt is no longer unique")
        return row
    try:
        response = response_from_journal(journal, call_id)
        if response is None:
            response = journal.request(
                call_id, user_prompt, system_prompt, 50, 0.7, 0.9
            )
    except Pause:
        raise
    except Exception as exc:
        call = journal.calls.get(call_id)
        if call is not None and call["terminal"] is not None \
                and call["terminal"]["kind"] == "complete":
            response = response_from_journal(journal, call_id)
        else:
            freeze(path, {
                **base,
                "status": "transport_failed_or_uncertain",
                "error_type": type(exc).__name__,
                "selected": False,
            })
            halt_once(journal, "transport_failed_or_uncertain")
            raise GlobalStop("Uncertain request permanently halted the pilot") from exc

    candidate = response["text"].strip()
    surface = surface_result(candidate)
    row = {
        **base,
        "call_sha256": response["call_sha256"],
        "candidate": candidate,
        "candidate_sha256": text_hash(candidate),
        "stop_reason": response["stop_reason"],
        "usage": response["usage"],
        "response_metadata": response["response_metadata"],
        "surface": surface,
        "score_computed": False,
        "selected": False,
    }
    if response["stop_reason"] != "stop" or not surface["pass"]:
        row["status"] = "surface_or_completion_rejected"
        return freeze(path, row)

    score_seed = int(digest([
        "swan_injection_only_pilot1",
        slot,
        attempt,
        target_index,
    ])[:8], 16)
    try:
        assessment = score_with_diagnostics(
            module, candidate, target, score_seed
        )
    except GlobalStop as exc:
        freeze(path, {
            **row,
            "status": "scorer_integrity_halt",
            "candidate_diagnostics": getattr(
                exc, "candidate_diagnostics", None
            ),
            "error_type": type(exc).__name__,
        })
        halt_once(journal, f"scorer_integrity_halt:{type(exc).__name__}")
        raise

    row["candidate_diagnostics"] = assessment["diagnostics"]
    row["parser_or_scorer_error_type"] = assessment["error_type"]
    row["parser_or_scorer_error_message"] = assessment["error_message"]
    if assessment["scored"] is not None:
        row.update(assessment["scored"], score_computed=True)
    if not assessment["clean"]:
        row["status"] = "parser_diagnostic_or_exception_rejected"
        return freeze(path, row)

    unique = row["candidate_sha256"] not in selected_hashes
    selected = row["official_score_accepted"] and unique
    row.update(
        unique_selected_text=unique,
        selected=selected,
        status="accepted" if selected else "score_or_identity_rejected",
    )
    return freeze(path, row)


def verify_slot(slot):
    path = slot_path(slot)
    row = load(path)
    if row["manifest_sha256"] != sha256(OUT / "manifest.json") \
            or row["slot"] != slot \
            or row["target_template_index"] != target_for(slot) \
            or row["status"] not in ("accepted", "exhausted") \
            or row["fallback"]:
        raise GlobalStop("Saved slot identity or status changed")
    traces = [attempt_path(slot, attempt) for attempt in range(row["attempts"])]
    if row["attempt_trace_sha256"] != [sha256(path) for path in traces]:
        raise GlobalStop("Saved slot attempt evidence changed")
    final = load(traces[-1])
    if row["accepted"]:
        if not final["selected"] or row["text"] != final["candidate"] \
                or row["text_sha256"] != final["candidate_sha256"] \
                or row["score"] != final["score"] \
                or row["score"] < MATCH_THRESHOLD:
            raise GlobalStop("Accepted slot does not match its terminal trace")
    elif row["attempts"] != MAX_ATTEMPTS or row["text"]:
        raise GlobalStop("Exhausted slot contains fallback or early stop")
    return row


def finalize_slot(slot, traces):
    selected = traces[-1] if traces and traces[-1]["selected"] else None
    target_index = target_for(slot)
    row = {
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "slot": slot,
        "status": "accepted" if selected else "exhausted",
        "attempts": len(traces),
        "attempt_trace_sha256": [
            sha256(attempt_path(slot, attempt))
            for attempt in range(len(traces))
        ],
        "accepted": selected is not None,
        "fallback": False,
        "text": selected["candidate"] if selected else "",
        "text_sha256": selected["candidate_sha256"] if selected else None,
        "target_template_index": target_index,
        "target_template_sha256": text_hash(
            traces[0]["template"] if traces else ""
        ),
        "score": selected["score"] if selected else None,
        "terminal_attempt": selected["attempt"] if selected else MAX_ATTEMPTS - 1,
    }
    freeze(slot_path(slot), row)
    return verify_slot(slot)


def result_from_slots(slots, journal):
    accepted = [row for row in slots if row["accepted"]]
    hashes = [row["text_sha256"] for row in accepted]
    if len(hashes) != len(set(hashes)):
        raise GlobalStop("Accepted slot texts are not unique")
    result = {
        "experiment": "swan_injection_only_pilot1",
        "completed_at": now(),
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "status": "target_reached" if len(accepted) >= TARGET_ACCEPTED
        else "slot_cap_exhausted",
        "slots_started": len(slots),
        "accepted_slots": len(accepted),
        "exhausted_slots": sum(not row["accepted"] for row in slots),
        "target_accepted": TARGET_ACCEPTED,
        "fallback_slots": 0,
        "selected": [{
            "slot": row["slot"],
            "text": row["text"],
            "text_sha256": row["text_sha256"],
            "target_template_index": row["target_template_index"],
            "score": row["score"],
            "slot_sha256": sha256(slot_path(row["slot"])),
        } for row in accepted],
        "slot_sha256": {
            str(row["slot"]): sha256(slot_path(row["slot"])) for row in slots
        },
        "accounting": journal.totals(),
        "accepted_only_composition_ready": len(accepted) >= TARGET_ACCEPTED,
    }
    freeze(OUT / "result.json", result)
    return result


def run(max_new_calls):
    check()
    require_authorization()
    require_price()
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise PermissionError("DEEPSEEK_API_KEY is not set")
    if type(max_new_calls) is not int or max_new_calls < 1:
        raise ValueError("Explicit positive --max-new-calls required")
    if (OUT / "result.json").exists():
        return load(OUT / "result.json")

    journal = PilotJournal(
        max_new_calls,
        lambda body: transport(body, key),
        sha256(OUT / "manifest.json"),
    )
    probe_id = "injection-only-model-probe"
    probe = response_from_journal(journal, probe_id)
    if probe is None:
        try:
            journal.request(probe_id, "Reply OK.", None, 1, 0.7, 0.9)
        except Pause:
            return {"state": "paused_before_probe", **journal.totals()}
        except Exception as exc:
            halt_once(journal, f"model_probe_failure:{type(exc).__name__}")
            raise GlobalStop("Model identity probe failed") from exc

    bank = [row["normalized_template_amr"] for row in load(BANK)]
    examples = load(EXAMPLES)[:3]
    slots = []
    selected_hashes = set()
    with official_runtime(OUT, error_folder=OUT / "errors") as module:
        for slot in range(MAX_SLOTS):
            if len(selected_hashes) >= TARGET_ACCEPTED:
                break
            if slot_path(slot).exists():
                saved = verify_slot(slot)
                slots.append(saved)
                if saved["accepted"]:
                    if saved["text_sha256"] in selected_hashes:
                        raise GlobalStop("Accepted slot text duplicated")
                    selected_hashes.add(saved["text_sha256"])
                continue
            traces = []
            for attempt in range(MAX_ATTEMPTS):
                try:
                    row = run_attempt(
                        module,
                        journal,
                        bank,
                        examples,
                        slot,
                        attempt,
                        selected_hashes,
                    )
                except Pause:
                    return {
                        "state": "paused_before_call",
                        "slot": slot,
                        "attempt": attempt,
                        "accepted_slots": len(selected_hashes),
                        **journal.totals(),
                    }
                traces.append(row)
                if row["selected"]:
                    break
            saved = finalize_slot(slot, traces)
            slots.append(saved)
            if saved["accepted"]:
                selected_hashes.add(saved["text_sha256"])
    return result_from_slots(slots, journal)


def status():
    manifest = OUT / "manifest.json"
    result_path = OUT / "result.json"
    result = {
        "experiment": "swan_injection_only_pilot1",
        "manifest_present": manifest.exists(),
        "authorization_present": (OUT / "authorization.json").exists(),
        "price_check_present": (OUT / "price_check.json").exists(),
        "result_present": result_path.exists(),
    }
    if manifest.exists():
        result["manifest_sha256"] = sha256(manifest)
        journal_path = OUT / "private/pilot/journal.jsonl"
        if journal_path.exists():
            result["accounting"] = PilotJournal(
                1, None, sha256(manifest), allow_halted=True
            ).totals()
    if result_path.exists():
        result["result"] = load(result_path)
        result["result_sha256"] = sha256(result_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "check", "authorize", "record-price", "run", "status"),
    )
    parser.add_argument("--max-new-calls", type=int)
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--max-cost-usd")
    parser.add_argument("--authorized-by")
    parser.add_argument("--authorization-text")
    args = parser.parse_args()
    with single_writer(OUT):
        if args.command == "prepare":
            result = prepare()
        elif args.command == "check":
            result = check()
        elif args.command == "authorize":
            result = authorize(
                args.authorized_by,
                args.authorization_text,
                args.max_calls,
                args.max_cost_usd,
            )
        elif args.command == "record-price":
            result = record_price()
        elif args.command == "run":
            result = run(args.max_new_calls)
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
