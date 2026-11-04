"""Collect two SWAN-accepted sentences with a natural semicolon split."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
import logging
import math
import os
from pathlib import Path
import sys

from run_swan_accepted_baseline_preflight import FailOnWarning
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
from swan_final1_transform import (
    LEAK_RE,
    FrozenSyntax,
    _candidates,
    sentence_spans,
    text_hash,
    unit_texts,
    unsupported,
    word_tokens,
)

OUT = ROOT / "experiments/swan_splittable_positive_pilot1"
BANK = ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_50.json"
EXAMPLES = (
    BANK.parent.parent / "artifacts/normalized_amr_examples.json"
)
EXPECTED_BANK = "f0a88c1af73d12078586d10ef68804c12ed5b79cabc07c9f6c2a34bb084ef5a4"
TEMPLATE_SCHEDULE = (45, 16, 15, 1, 22, 35, 9, 48, 36, 49)
SUBJECTS = ("Aster Labs", "Meridian Press", "Harbor Council", "Northstar Institute")
MAX_SLOTS = 4
MAX_ATTEMPTS = 50
TARGET_ACCEPTED = 2
MAX_CALLS = 1 + MAX_SLOTS * MAX_ATTEMPTS
MAX_COST_NANODOLLARS = MAX_CALLS * MAX_RESERVE
PILOT_CAP = (MAX_CALLS, MAX_COST_NANODOLLARS)
MATCH_THRESHOLD = 0.75


def context_for(slot):
    subject = SUBJECTS[slot]
    return (
        "This is a strict formatting task and the target AMR remains the "
        "primary semantic constraint. Write 10 to 22 words as exactly one "
        "orthographic sentence ending in one period. Use exactly one "
        "semicolon and no comma. Put a complete declarative clause on each "
        f"side. Start both clauses exactly with the proper name '{subject}'. "
        "The second clause must briefly confirm or restate the first without "
        "adding an unrelated fact."
    )


def schedule_for(slot):
    offset = slot % len(TEMPLATE_SCHEDULE)
    rotated = TEMPLATE_SCHEDULE[offset:] + TEMPLATE_SCHEDULE[:offset]
    return [index for index in rotated for _ in range(5)]


def surface_result(text, subject, syntax):
    text = text.strip()
    spans = sentence_spans(text)
    _, splits = _candidates(text, spans, syntax)
    left, separator, right = text.partition(";")
    checks = {
        "word_count": len(word_tokens(text)),
        "word_range_pass": 10 <= len(word_tokens(text)) <= 22,
        "one_punkt_unit": len(spans) == 1 and unit_texts(text, spans) == [text],
        "terminal_period": text.endswith(".") and not text.endswith(".."),
        "exactly_one_semicolon": text.count(";") == 1,
        "no_comma": "," not in text,
        "supported_punctuation": unsupported(text) is None,
        "no_amr_leak": not bool(LEAK_RE.search(text)),
        "left_subject_exact": left.strip().startswith(subject + " "),
        "right_subject_exact": bool(separator)
        and right.strip().startswith(subject + " "),
        "recognized_natural_split": len(splits) == 1
        and splits[0]["original_punctuation"] == ";",
    }
    checks["pass"] = all(value for key, value in checks.items() if key != "word_count")
    return {**checks, "split_candidates": splits}


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
        ROOT / "scripts/run_swan_accepted_baseline_preflight.py",
        ROOT / "scripts/swan_feasibility2_common.py",
        ROOT / "scripts/swan_final1_budget.py",
        ROOT / "scripts/swan_final1_contract.py",
        ROOT / "scripts/swan_final1_runtime.py",
        ROOT / "scripts/swan_final1_transform.py",
        ROOT / "scripts/test_swan_splittable_positive_pilot.py",
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


def prepare():
    if (OUT / "manifest.json").exists():
        return check()
    if sha256(BANK) != EXPECTED_BANK:
        raise ValueError("Bank50 changed")
    freeze(OUT / "budget_estimate.json", budget())
    freeze(OUT / "environment_lock.json", environment_lock())
    design = freeze(OUT / "design.json", {
        "subjects": list(SUBJECTS),
        "template_schedule": list(TEMPLATE_SCHEDULE),
        "expanded_schedule_by_slot": {
            str(slot): schedule_for(slot) for slot in range(MAX_SLOTS)
        },
        "contexts": {
            str(slot): context_for(slot) for slot in range(MAX_SLOTS)
        },
    })
    manifest = freeze(OUT / "manifest.json", {
        "schema_version": 1,
        "experiment": "swan_splittable_positive_pilot1",
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
        "dependencies": dependencies(),
        "new_output_observations_at_freeze": 0,
        "fallback_allowed": False,
        "credential_file_access": False,
        "sealed_source_access": False,
    })
    freeze(OUT / "authorization_template.json", {
        "phase": "pilot",
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "budget_sha256": sha256(OUT / "budget_estimate.json"),
        "max_calls": MAX_CALLS,
        "max_cost_nanodollars": MAX_COST_NANODOLLARS,
        "authorized_by": None,
        "authorization_text": None,
        "key_rotation_attested": None,
        "authorized_at": None,
    })
    return {
        "state": manifest["status"],
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "max_calls": MAX_CALLS,
        "max_cost_usd": budget()["max_cost_usd"],
        "design_sha256": digest(design),
    }


def check(full_environment=True):
    manifest = load(OUT / "manifest.json")
    if manifest["experiment"] != "swan_splittable_positive_pilot1" \
            or not manifest["development_pilot"] \
            or manifest["target_accepted"] != TARGET_ACCEPTED \
            or manifest["max_slots"] != MAX_SLOTS \
            or manifest["max_attempts_per_slot"] != MAX_ATTEMPTS \
            or manifest["match_threshold"] != MATCH_THRESHOLD \
            or manifest["fallback_allowed"] \
            or manifest["credential_file_access"] \
            or manifest["sealed_source_access"] \
            or manifest["new_output_observations_at_freeze"] != 0:
        raise ValueError("Pilot identity or isolation contract changed")
    if sha256(BANK) != EXPECTED_BANK \
            or manifest["budget_sha256"] != sha256(OUT / "budget_estimate.json") \
            or manifest["environment_sha256"] != sha256(OUT / "environment_lock.json") \
            or manifest["design_sha256"] != sha256(OUT / "design.json"):
        raise ValueError("Frozen pilot artifact changed")
    if load(OUT / "budget_estimate.json") != budget():
        raise ValueError("Budget calculation changed")
    expected_design = {
        "subjects": list(SUBJECTS),
        "template_schedule": list(TEMPLATE_SCHEDULE),
        "expanded_schedule_by_slot": {
            str(slot): schedule_for(slot) for slot in range(MAX_SLOTS)
        },
        "contexts": {
            str(slot): context_for(slot) for slot in range(MAX_SLOTS)
        },
    }
    if load(OUT / "design.json") != expected_design:
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


def authorize(who, text, calls, dollars, key_rotation_attested):
    check()
    if not who or not who.strip() or not text or not text.strip():
        raise PermissionError("Explicit authorizer identity and consent text required")
    if key_rotation_attested is not True:
        raise PermissionError("Explicit API key rotation attestation required")
    if calls != MAX_CALLS \
            or Decimal(dollars or "0") * 1_000_000_000 != MAX_COST_NANODOLLARS:
        raise PermissionError("Authorization must exactly match the frozen pilot cap")
    return freeze(OUT / "authorization.json", {
        "phase": "pilot",
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "budget_sha256": sha256(OUT / "budget_estimate.json"),
        "max_calls": MAX_CALLS,
        "max_cost_nanodollars": MAX_COST_NANODOLLARS,
        "authorized_by": who.strip(),
        "authorization_text": text.strip(),
        "key_rotation_attested": True,
        "authorized_at": now(),
        "status": "explicit_pilot_authorization",
    })


def require_authorization():
    path = OUT / "authorization.json"
    if not path.is_file():
        raise PermissionError("No immutable pilot budget authorization")
    row = load(path)
    if row.get("manifest_sha256") != sha256(OUT / "manifest.json") \
            or row.get("budget_sha256") != sha256(OUT / "budget_estimate.json") \
            or row.get("status") != "explicit_pilot_authorization" \
            or (row.get("max_calls"), row.get("max_cost_nanodollars")) != PILOT_CAP \
            or row.get("key_rotation_attested") is not True \
            or not row.get("authorized_by", "").strip() \
            or not row.get("authorization_text", "").strip():
        raise PermissionError("Authorization does not match frozen contract")
    if parse_utc(row["authorized_at"]) < parse_utc(load(OUT / "manifest.json")["frozen_at"]):
        raise PermissionError("Historical consent cannot authorize this contract")
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
    if not 0 <= age <= 24 * 3600 \
            or row["manifest_sha256"] != sha256(OUT / "manifest.json") \
            or row["evidence_sha256"] != sha256(OUT / "current_price_evidence.md") \
            or Decimal(row["input_price_usd_per_million"]) != Decimal(".30") \
            or Decimal(row["output_price_usd_per_million"]) != Decimal("1.20"):
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


def attempt_path(slot, attempt):
    return OUT / "private/pilot/generation_attempts" / (
        f"splittable-pilot-slot-{slot:02d}-attempt-{attempt:02d}.json"
    )


def slot_path(slot):
    return OUT / "private/pilot/generation_slots" / (
        f"splittable-pilot-slot-{slot:02d}.json"
    )


def run_attempt(module, journal, syntax, bank, examples, slot, attempt, selected_hashes):
    path = attempt_path(slot, attempt)
    if path.exists():
        row = load(path)
        if row["manifest_sha256"] != sha256(OUT / "manifest.json") \
                or row["slot"] != slot or row["attempt"] != attempt:
            raise GlobalStop("Saved attempt identity changed")
        return row
    target_index = schedule_for(slot)[attempt]
    target = bank[target_index]
    context = context_for(slot)
    user_prompt = module.build_user_prompt(target, examples, context)
    system_prompt = module.build_system_prompt()
    call_id = f"splittable-pilot-slot-{slot:02d}-attempt-{attempt:02d}"
    base = {
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "slot": slot,
        "attempt": attempt,
        "call_id": call_id,
        "subject": SUBJECTS[slot],
        "context_sha256": text_hash(context),
        "template_index": target_index,
        "template": target,
        "template_sha256": text_hash(target),
        "user_prompt_sha256": text_hash(user_prompt),
        "system_prompt_sha256": text_hash(system_prompt),
    }
    try:
        response = response_from_journal(journal, call_id)
        if response is None:
            response = journal.request(
                call_id, user_prompt, system_prompt, 50, 0.7, 0.9
            )
    except Pause:
        raise
    except Exception as exc:
        freeze(path, {
            **base,
            "status": "transport_failed_or_uncertain",
            "error_type": type(exc).__name__,
        })
        if not any(event["kind"] == "halt" for event in journal.events):
            journal.event("halt", reason="transport_failed_or_uncertain")
        raise GlobalStop("Uncertain request permanently halted the pilot") from exc

    candidate = response["text"].strip()
    surface = surface_result(candidate, SUBJECTS[slot], syntax)
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
    if response["stop_reason"] == "length" or not surface["pass"]:
        row["status"] = "surface_rejected"
        return freeze(path, row)
    score_seed = int(digest([
        "swan_splittable_positive_pilot1",
        slot,
        attempt,
        target_index,
    ])[:8], 16)
    try:
        scored = score_candidate(module, candidate, target, score_seed)
    except Exception as exc:
        freeze(path, {
            **row,
            "status": "parser_or_scorer_integrity_failure",
            "error_type": type(exc).__name__,
        })
        if not any(event["kind"] == "halt" for event in journal.events):
            journal.event(
                "halt",
                reason=f"parser_or_scorer_integrity_failure:{type(exc).__name__}",
            )
        raise GlobalStop("Parser or scorer integrity failure") from exc
    unique = row["candidate_sha256"] not in selected_hashes
    row.update(
        scored,
        score_computed=True,
        unique_selected_text=unique,
        selected=scored["official_score_accepted"] and unique,
        status="accepted" if scored["official_score_accepted"] and unique
        else "score_or_identity_rejected",
    )
    return freeze(path, row)


def verify_slot(slot):
    path = slot_path(slot)
    row = load(path)
    if row["manifest_sha256"] != sha256(OUT / "manifest.json") \
            or row["slot"] != slot \
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
                or row["score"] != final["score"]:
            raise GlobalStop("Accepted slot does not match its terminal trace")
    elif row["attempts"] != MAX_ATTEMPTS or row["text"]:
        raise GlobalStop("Exhausted slot contains fallback or early stop")
    return row


def finalize_slot(slot, traces):
    selected = traces[-1] if traces and traces[-1]["selected"] else None
    row = {
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "slot": slot,
        "subject": SUBJECTS[slot],
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
        "target_template_index": selected["template_index"] if selected else None,
        "target_template": selected["template"] if selected else None,
        "target_template_sha256": selected["template_sha256"] if selected else None,
        "score": selected["score"] if selected else None,
        "terminal_attempt": selected["attempt"] if selected else MAX_ATTEMPTS - 1,
    }
    freeze(slot_path(slot), row)
    return verify_slot(slot)


def result_from_slots(slots, journal):
    accepted = [row for row in slots if row["accepted"]]
    result = {
        "experiment": "swan_splittable_positive_pilot1",
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
            "target_template_sha256": row["target_template_sha256"],
            "score": row["score"],
            "slot_sha256": sha256(slot_path(row["slot"])),
        } for row in accepted],
        "slot_sha256": {
            str(row["slot"]): sha256(slot_path(row["slot"])) for row in slots
        },
        "accounting": journal.totals(),
        "downstream_standard_C2_ready": len(accepted) >= TARGET_ACCEPTED,
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
    if not max_new_calls or max_new_calls < 1:
        raise ValueError("Explicit positive --max-new-calls required")
    if (OUT / "result.json").exists():
        return load(OUT / "result.json")
    journal = PilotJournal(
        max_new_calls,
        lambda body: transport(body, key),
        sha256(OUT / "manifest.json"),
    )
    probe_id = "splittable-pilot-model-probe"
    probe = response_from_journal(journal, probe_id)
    if probe is None:
        journal.request(probe_id, "Reply OK.", None, 1, 0.7, 0.9)

    bank = [row["normalized_template_amr"] for row in load(BANK)]
    examples = load(EXAMPLES)[:3]
    syntax = FrozenSyntax()
    slots = []
    selected_hashes = set()
    for slot in range(MAX_SLOTS):
        if len(selected_hashes) >= TARGET_ACCEPTED:
            break
        if slot_path(slot).exists():
            saved = verify_slot(slot)
            slots.append(saved)
            if saved["accepted"]:
                selected_hashes.add(saved["text_sha256"])
            continue
        traces = []
        warning_handler = FailOnWarning(level=logging.WARNING)
        parser_logger = logging.getLogger(
            "amrlib.models.parse_xfm.penman_serializer"
        )
        parser_logger.addHandler(warning_handler)
        try:
            with official_runtime(OUT, error_folder=OUT / "errors") as module:
                for attempt in range(MAX_ATTEMPTS):
                    try:
                        row = run_attempt(
                            module,
                            journal,
                            syntax,
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
                            **journal.totals(),
                        }
                    traces.append(row)
                    if row["selected"]:
                        break
        finally:
            parser_logger.removeHandler(warning_handler)
        saved = finalize_slot(slot, traces)
        slots.append(saved)
        if saved["accepted"]:
            selected_hashes.add(saved["text_sha256"])
    return result_from_slots(slots, journal)


def status():
    manifest = OUT / "manifest.json"
    result_path = OUT / "result.json"
    result = {
        "experiment": "swan_splittable_positive_pilot1",
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
    parser.add_argument("--key-rotation-attested", action="store_true")
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
                args.key_rotation_attested,
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
