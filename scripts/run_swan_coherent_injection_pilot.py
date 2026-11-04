"""Generate one strictly accepted, topic-coherent ten-sentence SWAN document."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import sys

import run_swan_injection_only_pilot as injection
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
    sentence_spans,
    text_hash,
    unit_texts,
    word_tokens,
)

OUT = ROOT / "experiments/swan_coherent_injection_pilot1"
BANK = ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_50.json"
EXAMPLES = BANK.parent.parent / "artifacts/normalized_amr_examples.json"
ROTATION_ATTESTATION = (
    ROOT / "experiments/swan_normal_length_feasibility2/authorization_pilot.json"
)
HISTORICAL_RESULT = ROOT / "experiments/swan_injection_only_pilot1/result.json"
EXPECTED_BANK = "f0a88c1af73d12078586d10ef68804c12ed5b79cabc07c9f6c2a34bb084ef5a4"
EXPECTED_HISTORICAL_RESULT = (
    "cb481c602074756928abcd7e84ad2bdb815b95852e18b354096a14cd56563f2a"
)

TEMPLATE_SCHEDULE = (47, 47, 27, 11, 14, 43, 25, 37, 27, 11)
SLOT_PURPOSES = (
    "Establish that United Nations released a report.",
    "Establish that Alice Morgan received or organized the report.",
    "Tell the team to keep the briefing concise.",
    "Ask for a summary about New York for Alice.",
    "Ask for an explanation of the policy process or what changed.",
    "Ask for a summary of the United Nations report for Alice.",
    "Ask for the main policy details.",
    "Ask for the report or article topic to be explained to Alice.",
    "Tell the team to keep the recommendations practical or clear.",
    "Ask for a final summary about New York for Alice.",
)
FIXED_TOPIC = (
    "Alice Morgan is preparing a concise internal briefing about a United "
    "Nations policy report concerning New York."
)
THEME_TOKENS = frozenset({
    "alice",
    "article",
    "brief",
    "briefing",
    "concise",
    "details",
    "explain",
    "final",
    "findings",
    "morgan",
    "nations",
    "new",
    "policy",
    "practical",
    "process",
    "recommendations",
    "report",
    "summary",
    "united",
    "york",
})
MAX_SLOTS = 10
MAX_ATTEMPTS = 50
REVIEW_COUNT = 2
MAX_CALLS = 1 + MAX_SLOTS * MAX_ATTEMPTS + REVIEW_COUNT
MAX_COST_NANODOLLARS = MAX_CALLS * MAX_RESERVE
PILOT_CAP = (MAX_CALLS, MAX_COST_NANODOLLARS)
MATCH_THRESHOLD = 0.75
MIN_WORDS = 3
MAX_WORDS = 18
MAX_PRIOR_CONTEXT = 3
REVIEW_KEYS = (
    "coherence",
    "naturalness",
    "nonredundancy",
    "task_consistency",
)
TERMINAL_RE = re.compile(r"""[.!?]["')\]]*$""")


def target_for(slot):
    if type(slot) is not int or not 0 <= slot < MAX_SLOTS:
        raise ValueError("Slot outside frozen schedule")
    return TEMPLATE_SCHEDULE[slot]


def historical_hashes():
    if sha256(HISTORICAL_RESULT) != EXPECTED_HISTORICAL_RESULT:
        raise ValueError("Historical strict-acceptance result changed")
    result = load(HISTORICAL_RESULT)
    if result["status"] != "target_reached" or result["fallback_slots"] != 0:
        raise ValueError("Historical strict-acceptance source is invalid")
    return {row["text_sha256"] for row in result["selected"]}


def context_for(slot, previous):
    if len(previous) != slot:
        raise ValueError("Context history does not match slot")
    recent = previous[-MAX_PRIOR_CONTEXT:]
    history = "\n".join(f"- {text}" for text in recent) if recent else "(none)"
    return (
        "Write the next sentence of a cohesive ten-sentence editorial task brief.\n"
        f"FIXED TOPIC: {FIXED_TOPIC}\n"
        f"SLOT {slot + 1}/10 PURPOSE: {SLOT_PURPOSES[slot]}\n"
        f"RECENT ACCEPTED SENTENCES:\n{history}\n"
        "The target AMR is mandatory. Use the fixed topic and slot purpose only "
        "where compatible with that graph. Continue the same assignment, avoid "
        "repetition, use 3 to 18 words, and output exactly one complete sentence."
    )


def token_set(text):
    return {token.casefold() for token in word_tokens(text)}


def token_jaccard(left, right):
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def surface_result(text, previous, forbidden_hashes):
    text = text.strip()
    base = injection.surface_result(text)
    words = word_tokens(text)
    overlap = sorted(token_set(text) & THEME_TOKENS)
    similarities = [token_jaccard(text, prior) for prior in previous]
    checks = {
        **{key: value for key, value in base.items() if key != "pass"},
        "terminal_punctuation": bool(TERMINAL_RE.search(text)),
        "word_count_3_18": MIN_WORDS <= len(words) <= MAX_WORDS,
        "theme_vocabulary_overlap": bool(overlap),
        "prior_token_jaccard_at_most_point8": all(
            value <= 0.8 for value in similarities
        ),
        "not_historical_strict_acceptance": (
            text_hash(text) not in forbidden_hashes
        ),
    }
    checks["pass"] = all(checks.values())
    return {
        **checks,
        "word_count": len(words),
        "theme_overlap": overlap,
        "prior_token_jaccard": similarities,
    }


def review_prompt(label, document):
    return (
        f"Blind review pass {label}. Evaluate only the passage below as one "
        "editorial task brief. Return exactly one JSON object with integer "
        "scores from 1 to 5 for coherence, naturalness, nonredundancy, and "
        "task_consistency, plus boolean severe_problem. Coherence requires the "
        "sentences to belong to one interpretable assignment; task_consistency "
        "requires a stable topic and recipient. Use no markdown and no extra "
        f"keys.\n\nPASSAGE:\n{document}"
    )


def parse_review(text):
    try:
        row = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    expected = {*REVIEW_KEYS, "severe_problem"}
    if not isinstance(row, dict) or set(row) != expected:
        return None
    if any(type(row[key]) is not int or not 1 <= row[key] <= 5 for key in REVIEW_KEYS):
        return None
    if type(row["severe_problem"]) is not bool:
        return None
    return row


def review_pass(row):
    return (
        row is not None
        and not row["severe_problem"]
        and all(row[key] >= 4 for key in REVIEW_KEYS)
    )


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
        HISTORICAL_RESULT,
        ROOT / "scripts/run_swan_injection_only_pilot.py",
        ROOT / "scripts/run_swan_positive_collection.py",
        ROOT / "scripts/swan_feasibility2_common.py",
        ROOT / "scripts/swan_final1_budget.py",
        ROOT / "scripts/swan_final1_contract.py",
        ROOT / "scripts/swan_final1_quality.py",
        ROOT / "scripts/swan_final1_runtime.py",
        ROOT / "scripts/swan_final1_transform.py",
        ROOT / "scripts/test_swan_coherent_injection_pilot.py",
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
        "auxiliary_review_calls": REVIEW_COUNT,
        "max_calls": MAX_CALLS,
        "max_cost_nanodollars": MAX_COST_NANODOLLARS,
        "max_cost_usd": MAX_COST_NANODOLLARS / 1e9,
        "transport_retries": 0,
        "uncertain_calls_retain_full_reserve": True,
    }


def frozen_design():
    return {
        "fixed_topic": FIXED_TOPIC,
        "slot_purposes": list(SLOT_PURPOSES),
        "template_schedule": list(TEMPLATE_SCHEDULE),
        "target_by_slot": {
            str(slot): target_for(slot) for slot in range(MAX_SLOTS)
        },
        "theme_tokens": sorted(THEME_TOKENS),
        "max_prior_context": MAX_PRIOR_CONTEXT,
        "examples": "first_three_frozen_normalized_examples",
        "generation": {
            "model": MODEL,
            "thinking": "disabled",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_output_tokens": 50,
        },
        "surface": {
            "min_words": MIN_WORDS,
            "max_words": MAX_WORDS,
            "max_prior_token_jaccard": 0.8,
            "historical_exact_collision_forbidden": True,
        },
        "auxiliary_reviews": {
            "count": REVIEW_COUNT,
            "same_backend_as_generation": True,
            "human_review": False,
            "minimum_each_score": 4,
            "severe_problem_required": False,
        },
    }


def prepare():
    if (OUT / "manifest.json").exists():
        return check()
    if sha256(BANK) != EXPECTED_BANK:
        raise ValueError("Bank50 changed")
    historical_hashes()
    rotation_hash = sha256(ROTATION_ATTESTATION)
    freeze(OUT / "budget_estimate.json", budget())
    freeze(OUT / "environment_lock.json", environment_lock())
    freeze(OUT / "design.json", frozen_design())
    manifest = freeze(OUT / "manifest.json", {
        "schema_version": 1,
        "experiment": "swan_coherent_injection_pilot1",
        "frozen_at": now(),
        "status": "frozen_preexecution",
        "development_pilot": True,
        "slots": MAX_SLOTS,
        "max_attempts_per_slot": MAX_ATTEMPTS,
        "match_threshold": MATCH_THRESHOLD,
        "bank_sha256": EXPECTED_BANK,
        "historical_result_sha256": EXPECTED_HISTORICAL_RESULT,
        "budget_sha256": sha256(OUT / "budget_estimate.json"),
        "environment_sha256": sha256(OUT / "environment_lock.json"),
        "design_sha256": sha256(OUT / "design.json"),
        "rotation_attestation_path": str(
            ROTATION_ATTESTATION.relative_to(ROOT)
        ),
        "rotation_attestation_sha256": rotation_hash,
        "dependencies": dependencies(),
        "new_output_observations_at_freeze": 0,
        "fallback_allowed": False,
        "credential_file_access": False,
        "sealed_source_access": False,
        "source_anchoring_constraint": False,
        "normal_length_claim": False,
        "human_quality_claim": False,
        "attack_authorized": False,
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
        manifest["experiment"] == "swan_coherent_injection_pilot1"
        and manifest["development_pilot"]
        and manifest["slots"] == MAX_SLOTS
        and manifest["max_attempts_per_slot"] == MAX_ATTEMPTS
        and manifest["match_threshold"] == MATCH_THRESHOLD
        and not manifest["fallback_allowed"]
        and not manifest["credential_file_access"]
        and not manifest["sealed_source_access"]
        and not manifest["source_anchoring_constraint"]
        and not manifest["normal_length_claim"]
        and not manifest["human_quality_claim"]
        and not manifest["attack_authorized"]
        and manifest["new_output_observations_at_freeze"] == 0
    )
    if not identity_ok:
        raise ValueError("Pilot identity or claim boundary changed")
    if sha256(BANK) != EXPECTED_BANK \
            or sha256(HISTORICAL_RESULT) != EXPECTED_HISTORICAL_RESULT \
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
    historical_hashes()
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
        "key_rotation_attestation_source": str(
            ROTATION_ATTESTATION.relative_to(ROOT)
        ),
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


def attempt_path(slot, attempt):
    return OUT / "private/pilot/generation_attempts" / (
        f"coherent-slot-{slot:02d}-attempt-{attempt:02d}.json"
    )


def slot_path(slot):
    return OUT / "private/pilot/generation_slots" / (
        f"coherent-slot-{slot:02d}.json"
    )


def review_path(label):
    return OUT / "private/pilot/quality_reviews" / f"review-{label.lower()}.json"


def halt_once(journal, reason):
    if not any(event["kind"] == "halt" for event in journal.events):
        journal.event("halt", reason=reason)


def run_attempt(module, journal, bank, examples, slot, attempt, previous):
    target_index = target_for(slot)
    target = bank[target_index]
    context = context_for(slot, previous)
    user_prompt = module.build_user_prompt(target, examples, context)
    system_prompt = module.build_system_prompt()
    call_id = f"coherent-slot-{slot:02d}-attempt-{attempt:02d}"
    path = attempt_path(slot, attempt)
    selected_hashes = {text_hash(text) for text in previous}
    forbidden = historical_hashes() | selected_hashes
    base = {
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "slot": slot,
        "attempt": attempt,
        "call_id": call_id,
        "context": context,
        "context_sha256": text_hash(context),
        "prior_selected_sha256": [text_hash(text) for text in previous],
        "slot_purpose": SLOT_PURPOSES[slot],
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
    surface = surface_result(candidate, previous, forbidden)
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
        "swan_coherent_injection_pilot1",
        slot,
        attempt,
        target_index,
    ])[:8], 16)
    try:
        assessment = injection.score_with_diagnostics(
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

    row.update(
        selected=row["official_score_accepted"],
        status="accepted" if row["official_score_accepted"]
        else "score_rejected",
    )
    return freeze(path, row)


def verify_slot(slot, previous):
    path = slot_path(slot)
    row = load(path)
    expected_context = context_for(slot, previous)
    if row["manifest_sha256"] != sha256(OUT / "manifest.json") \
            or row["slot"] != slot \
            or row["target_template_index"] != target_for(slot) \
            or row["context_sha256"] != text_hash(expected_context) \
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
                or row["score"] < MATCH_THRESHOLD \
                or not final["surface"]["pass"]:
            raise GlobalStop("Accepted slot does not match its terminal trace")
    elif row["attempts"] != MAX_ATTEMPTS or row["text"]:
        raise GlobalStop("Exhausted slot contains fallback or early stop")
    return row


def finalize_slot(slot, traces, previous):
    selected = traces[-1] if traces and traces[-1]["selected"] else None
    target_index = target_for(slot)
    context = context_for(slot, previous)
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
        "context_sha256": text_hash(context),
        "slot_purpose": SLOT_PURPOSES[slot],
        "target_template_index": target_index,
        "target_template_sha256": text_hash(
            traces[0]["template"] if traces else ""
        ),
        "score": selected["score"] if selected else None,
        "terminal_attempt": selected["attempt"] if selected else MAX_ATTEMPTS - 1,
    }
    freeze(slot_path(slot), row)
    return verify_slot(slot, previous)


def document_from_slots(slots):
    if len(slots) != MAX_SLOTS or not all(row["accepted"] for row in slots):
        raise ValueError("A coherent document requires ten accepted slots")
    text = " ".join(row["text"] for row in slots)
    spans = sentence_spans(text)
    texts = unit_texts(text, spans)
    if len(spans) != MAX_SLOTS or texts != [row["text"] for row in slots]:
        raise GlobalStop("Accepted composition did not preserve ten exact units")
    return {
        "document_id": "coherent-injection-pilot-00",
        "text": text,
        "text_sha256": text_hash(text),
        "word_count": len(word_tokens(text)),
        "segmentation_spans": spans,
        "sentence_text_sha256": [row["text_sha256"] for row in slots],
    }


def run_review(journal, label, document):
    label = label.upper()
    path = review_path(label)
    prompt = review_prompt(label, document["text"])
    system_prompt = (
        "You are a strict prose-quality evaluator. Follow the requested JSON "
        "schema exactly and evaluate only the supplied passage."
    )
    call_id = f"coherent-review-{label.lower()}"
    base = {
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "label": label,
        "call_id": call_id,
        "document_text_sha256": document["text_sha256"],
        "prompt_sha256": text_hash(prompt),
        "system_prompt_sha256": text_hash(system_prompt),
        "blind_to_templates_scores_and_detection": True,
        "same_backend_as_generation": True,
        "human_review": False,
    }
    if path.exists():
        row = load(path)
        if any(row.get(key) != value for key, value in base.items()):
            raise GlobalStop("Saved quality review identity changed")
        return row
    try:
        response = response_from_journal(journal, call_id)
        if response is None:
            response = journal.request(
                call_id, prompt, system_prompt, 50, 0.7, 0.9
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
                "pass": False,
            })
            halt_once(journal, "review_transport_failed_or_uncertain")
            raise GlobalStop("Uncertain review request halted the pilot") from exc
    parsed = parse_review(response["text"]) if response["stop_reason"] == "stop" else None
    row = {
        **base,
        "call_sha256": response["call_sha256"],
        "raw_response": response["text"],
        "raw_response_sha256": text_hash(response["text"]),
        "stop_reason": response["stop_reason"],
        "usage": response["usage"],
        "response_metadata": response["response_metadata"],
        "parsed": parsed,
        "valid_schema": parsed is not None,
        "pass": review_pass(parsed),
        "status": "complete",
    }
    return freeze(path, row)


def result_from_slots(slots, reviews, journal):
    accepted = [row for row in slots if row["accepted"]]
    hashes = [row["text_sha256"] for row in accepted]
    if len(hashes) != len(set(hashes)):
        raise GlobalStop("Accepted slot texts are not unique")
    complete = len(accepted) == MAX_SLOTS
    document = document_from_slots(slots) if complete else None
    auxiliary_pass = (
        complete
        and len(reviews) == REVIEW_COUNT
        and all(row["pass"] for row in reviews)
    )
    if not complete:
        status = "slot_exhausted"
    elif auxiliary_pass:
        status = "coherent_baseline_ready"
    else:
        status = "auxiliary_coherence_gate_failed"
    result = {
        "experiment": "swan_coherent_injection_pilot1",
        "completed_at": now(),
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "status": status,
        "slots_started": len(slots),
        "accepted_slots": len(accepted),
        "exhausted_slots": sum(not row["accepted"] for row in slots),
        "target_accepted": MAX_SLOTS,
        "fallback_slots": 0,
        "selected": [{
            "slot": row["slot"],
            "text": row["text"],
            "text_sha256": row["text_sha256"],
            "target_template_index": row["target_template_index"],
            "score": row["score"],
            "context_sha256": row["context_sha256"],
            "slot_sha256": sha256(slot_path(row["slot"])),
        } for row in accepted],
        "slot_sha256": {
            str(row["slot"]): sha256(slot_path(row["slot"])) for row in slots
        },
        "document": document,
        "auxiliary_reviews": [{
            "label": row["label"],
            "path": str(review_path(row["label"]).relative_to(ROOT)),
            "sha256": sha256(review_path(row["label"])),
            "parsed": row["parsed"],
            "pass": row["pass"],
        } for row in reviews],
        "auxiliary_coherence_gate_pass": auxiliary_pass,
        "human_quality_reviewed": False,
        "accounting": journal.totals(),
        "coherent_baseline_ready": auxiliary_pass,
        "attack_authorized": False,
    }
    freeze(OUT / "result.json", result)
    return result


def run(max_new_calls):
    check()
    require_authorization()
    require_price()
    if (OUT / "result.json").exists():
        return load(OUT / "result.json")
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise PermissionError("DEEPSEEK_API_KEY is not set")
    if type(max_new_calls) is not int or max_new_calls < 1:
        raise ValueError("Explicit positive --max-new-calls required")

    journal = PilotJournal(
        max_new_calls,
        lambda body: transport(body, key),
        sha256(OUT / "manifest.json"),
    )
    probe_id = "coherent-model-probe"
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
    previous = []
    with official_runtime(OUT, error_folder=OUT / "errors") as module:
        for slot in range(MAX_SLOTS):
            if slot_path(slot).exists():
                saved = verify_slot(slot, previous)
                slots.append(saved)
                if saved["accepted"]:
                    previous.append(saved["text"])
                    continue
                break
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
                        previous,
                    )
                except Pause:
                    return {
                        "state": "paused_before_call",
                        "slot": slot,
                        "attempt": attempt,
                        "accepted_slots": len(previous),
                        **journal.totals(),
                    }
                traces.append(row)
                if row["selected"]:
                    break
            saved = finalize_slot(slot, traces, previous)
            slots.append(saved)
            if not saved["accepted"]:
                break
            previous.append(saved["text"])

    if len(slots) != MAX_SLOTS or not all(row["accepted"] for row in slots):
        return result_from_slots(slots, [], journal)

    document = document_from_slots(slots)
    reviews = []
    for label in ("A", "B"):
        try:
            reviews.append(run_review(journal, label, document))
        except Pause:
            return {
                "state": "paused_before_review",
                "review": label,
                "accepted_slots": len(previous),
                **journal.totals(),
            }
    return result_from_slots(slots, reviews, journal)


def status():
    manifest = OUT / "manifest.json"
    result_path = OUT / "result.json"
    result = {
        "experiment": "swan_coherent_injection_pilot1",
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
