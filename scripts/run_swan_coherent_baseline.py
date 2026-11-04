"""Qualify and detect the strict ten-sentence coherent SWAN baseline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
import math
import os
from pathlib import Path
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
from swan_final1_runtime import official_runtime, score_triple
from swan_final1_transform import sentence_spans, text_hash, unit_texts, word_tokens

OUT = ROOT / "experiments/swan_coherent_baseline1"
SOURCE = ROOT / "experiments/swan_coherent_injection_pilot2"
BANK = ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_50.json"
ROTATION_ATTESTATION = (
    ROOT / "experiments/swan_normal_length_feasibility2/authorization_pilot.json"
)
EXPECTED_BANK = "f0a88c1af73d12078586d10ef68804c12ed5b79cabc07c9f6c2a34bb084ef5a4"
EXPECTED_SOURCE_MANIFEST = (
    "7c590f52670cc205bfe462f9e2befe7ffaa6d768d8606ca0cb3ba8ae17906f8a"
)
EXPECTED_SOURCE_RESULT = (
    "b7d699f39b8521c1a16f7afe181b7881a6cfa0a88b25cbf0f420c14a1d41bde0"
)
COUNT = 10
MATCH_THRESHOLD = 0.75
GREEN_THRESHOLD = 0.65
NULL_RATE = 0.25
REVIEW_LABELS = ("A", "B", "C")
REVIEW_KEYS = (
    "coherence",
    "naturalness",
    "nonredundancy",
    "task_consistency",
)
MAX_CALLS = len(REVIEW_LABELS)
MAX_COST_NANODOLLARS = MAX_CALLS * MAX_RESERVE
PILOT_CAP = (MAX_CALLS, MAX_COST_NANODOLLARS)
EXPECTED_ACTORS = ("United Nations", "Maria Lopez", "David Chen")
EXPECTED_PREDICATES = (
    "released",
    "received",
    "reviewed",
    "examined",
    "proposed",
    "explained",
    "requested",
    "revised",
    "prepared",
    "approved",
)


def diagnostics_empty(row):
    diagnostics = row.get("candidate_diagnostics") or {}
    return (
        not diagnostics.get("logged_warnings")
        and not diagnostics.get("python_warnings")
        and not diagnostics.get("stderr")
        and row.get("parser_or_scorer_error_type") is None
    )


def source_inputs():
    manifest_path = SOURCE / "manifest.json"
    result_path = SOURCE / "result.json"
    if sha256(manifest_path) != EXPECTED_SOURCE_MANIFEST \
            or sha256(result_path) != EXPECTED_SOURCE_RESULT:
        raise ValueError("Frozen coherent source changed")
    result = load(result_path)
    if result["status"] != "auxiliary_coherence_gate_failed" \
            or result["accepted_slots"] != COUNT \
            or result["fallback_slots"] != 0 \
            or len(result["selected"]) != COUNT:
        raise ValueError("Source is not the frozen 10/10 strict-injection result")

    selected = []
    seen = set()
    for source in result["selected"]:
        slot = source["slot"]
        slot_path = SOURCE / "private/pilot/generation_slots" / (
            f"coherent-slot-{slot:02d}.json"
        )
        if sha256(slot_path) != source["slot_sha256"] \
                or sha256(slot_path) != result["slot_sha256"][str(slot)]:
            raise ValueError("Source slot hash changed")
        saved = load(slot_path)
        trace_path = SOURCE / "private/pilot/generation_attempts" / (
            f"coherent-slot-{slot:02d}-attempt-{saved['terminal_attempt']:02d}.json"
        )
        trace = load(trace_path)
        if saved["attempt_trace_sha256"][-1] != sha256(trace_path):
            raise ValueError("Terminal source trace hash changed")
        valid = (
            saved["accepted"] is True
            and saved["fallback"] is False
            and saved["text"] == source["text"] == trace["candidate"]
            and saved["text_sha256"] == source["text_sha256"]
            == trace["candidate_sha256"] == text_hash(source["text"])
            and saved["score"] == source["score"] == trace["score"]
            and math.isfinite(saved["score"])
            and saved["score"] >= MATCH_THRESHOLD
            and trace["selected"] is True
            and trace["status"] == "accepted"
            and trace["stop_reason"] == "stop"
            and trace["surface"]["pass"] is True
            and trace["score_computed"] is True
            and trace["official_score_accepted"] is True
            and diagnostics_empty(trace)
        )
        if not valid or saved["text_sha256"] in seen:
            raise ValueError("Source slot is not a unique strict acceptance")
        seen.add(saved["text_sha256"])
        selected.append({
            "slot": slot,
            "text": saved["text"],
            "text_sha256": saved["text_sha256"],
            "target_template_index": saved["target_template_index"],
            "target_score": saved["score"],
            "slot_path": str(slot_path.relative_to(ROOT)),
            "slot_sha256": sha256(slot_path),
            "trace_path": str(trace_path.relative_to(ROOT)),
            "trace_sha256": sha256(trace_path),
        })

    if [row["slot"] for row in selected] != list(range(COUNT)):
        raise ValueError("Source selection is not the exact ten-slot sequence")
    text = " ".join(row["text"] for row in selected)
    spans = sentence_spans(text)
    texts = unit_texts(text, spans)
    if len(spans) != COUNT or texts != [row["text"] for row in selected]:
        raise ValueError("Concatenation does not preserve ten exact units")
    if text_hash(text) != result["document"]["text_sha256"] \
            or text != result["document"]["text"]:
        raise ValueError("Source document changed")

    actor_counts = {actor: text.count(actor) for actor in EXPECTED_ACTORS}
    predicate_hits = [
        predicate for predicate, sentence in zip(EXPECTED_PREDICATES, texts)
        if predicate in sentence.casefold()
    ]
    structural = {
        "sentence_count": len(texts),
        "unique_sentence_hashes": len(seen),
        "actor_counts": actor_counts,
        "expected_predicate_hits": predicate_hits,
        "pass": (
            len(texts) == COUNT
            and len(seen) == COUNT
            and all(actor_counts.values())
            and len(predicate_hits) == COUNT
        ),
    }
    if not structural["pass"]:
        raise ValueError("Source does not satisfy event-log structural checks")

    prior_reviews = []
    for source_review in result["auxiliary_reviews"]:
        path = ROOT / source_review["path"]
        if sha256(path) != source_review["sha256"]:
            raise ValueError("Prior auxiliary review changed")
        prior_reviews.append({
            "path": source_review["path"],
            "sha256": source_review["sha256"],
            "parsed": source_review["parsed"],
            "pass": source_review["pass"],
        })
    return {
        "source_manifest_path": str(manifest_path.relative_to(ROOT)),
        "source_manifest_sha256": sha256(manifest_path),
        "source_result_path": str(result_path.relative_to(ROOT)),
        "source_result_sha256": sha256(result_path),
        "source_prior_auxiliary_gate_pass": (
            result["auxiliary_coherence_gate_pass"]
        ),
        "source_prior_reviews": prior_reviews,
        "posthoc_genre_adjudication": True,
        "selected": selected,
        "structural_coherence_checks": structural,
        "document": {
            "document_id": "coherent-baseline-00",
            "text": text,
            "text_sha256": text_hash(text),
            "word_count": len(word_tokens(text)),
            "segmentation_spans": spans,
            "sentence_text_sha256": [row["text_sha256"] for row in selected],
        },
    }


def review_prompt(label, text):
    return (
        f"Blind adjudication {label}. Evaluate only the passage below as a "
        "concise project event log, not as literary prose or instructions. "
        "Short event-by-event sentences and repeated actor names are acceptable. "
        "Return exactly one JSON object with integer scores from 1 to 5 for "
        "coherence, naturalness, nonredundancy, and task_consistency, plus "
        "boolean severe_problem. Coherence requires a comprehensible sequence "
        "from receiving a report through revision and approval. Nonredundancy "
        "requires distinct events. Task consistency requires stable actors and "
        "one report workflow. Use no markdown and no extra keys.\n\n"
        f"PASSAGE:\n{text}"
    )


def parse_review(text):
    try:
        row = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(row, dict) or set(row) != {*REVIEW_KEYS, "severe_problem"}:
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
        and row["coherence"] >= 4
        and row["naturalness"] >= 3
        and row["nonredundancy"] >= 4
        and row["task_consistency"] >= 4
    )


class ReviewJournal(BaseJournal):
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
        SOURCE / "manifest.json",
        SOURCE / "result.json",
        ROOT / "scripts/run_swan_injection_only_pilot.py",
        ROOT / "scripts/run_swan_positive_collection.py",
        ROOT / "scripts/swan_feasibility2_common.py",
        ROOT / "scripts/swan_final1_budget.py",
        ROOT / "scripts/swan_final1_contract.py",
        ROOT / "scripts/swan_final1_quality.py",
        ROOT / "scripts/swan_final1_runtime.py",
        ROOT / "scripts/swan_final1_transform.py",
        ROOT / "scripts/test_swan_coherent_baseline.py",
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
        "review_calls": MAX_CALLS,
        "max_call_reserve_nanodollars": MAX_RESERVE,
        "max_calls": MAX_CALLS,
        "max_cost_nanodollars": MAX_COST_NANODOLLARS,
        "max_cost_usd": MAX_COST_NANODOLLARS / 1e9,
        "transport_retries": 0,
    }


def prepare():
    if (OUT / "manifest.json").exists():
        return check()
    if sha256(BANK) != EXPECTED_BANK:
        raise ValueError("Bank50 changed")
    freeze(OUT / "inputs.json", source_inputs())
    freeze(OUT / "budget_estimate.json", budget())
    freeze(OUT / "environment_lock.json", environment_lock())
    rotation_hash = sha256(ROTATION_ATTESTATION)
    manifest = freeze(OUT / "manifest.json", {
        "schema_version": 1,
        "experiment": "swan_coherent_baseline1",
        "frozen_at": now(),
        "status": "frozen_before_genre_adjudication_and_detection",
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST,
        "source_result_sha256": EXPECTED_SOURCE_RESULT,
        "inputs_sha256": sha256(OUT / "inputs.json"),
        "budget_sha256": sha256(OUT / "budget_estimate.json"),
        "environment_sha256": sha256(OUT / "environment_lock.json"),
        "bank_sha256": EXPECTED_BANK,
        "rotation_attestation_sha256": rotation_hash,
        "dependencies": dependencies(),
        "review_count": MAX_CALLS,
        "review_passes_required": 2,
        "green_threshold": GREEN_THRESHOLD,
        "null_rate": NULL_RATE,
        "posthoc_genre_adjudication": True,
        "human_quality_claim": False,
        "normal_length_claim": False,
        "empirical_calibration_claim": False,
        "attack_authorized": False,
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
        "key_rotation_attestation_source_sha256": rotation_hash,
        "authorized_at": None,
    })
    return {
        "state": manifest["status"],
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "inputs_sha256": sha256(OUT / "inputs.json"),
        "max_calls": MAX_CALLS,
        "max_cost_usd": budget()["max_cost_usd"],
    }


def check(full_environment=True):
    manifest = load(OUT / "manifest.json")
    valid = (
        manifest["experiment"] == "swan_coherent_baseline1"
        and manifest["source_manifest_sha256"] == EXPECTED_SOURCE_MANIFEST
        and manifest["source_result_sha256"] == EXPECTED_SOURCE_RESULT
        and manifest["review_count"] == MAX_CALLS
        and manifest["review_passes_required"] == 2
        and manifest["green_threshold"] == GREEN_THRESHOLD
        and manifest["null_rate"] == NULL_RATE
        and manifest["posthoc_genre_adjudication"]
        and not manifest["human_quality_claim"]
        and not manifest["normal_length_claim"]
        and not manifest["empirical_calibration_claim"]
        and not manifest["attack_authorized"]
        and not manifest["sealed_source_access"]
    )
    if not valid:
        raise ValueError("Baseline identity or claim boundary changed")
    if sha256(BANK) != EXPECTED_BANK \
            or sha256(SOURCE / "manifest.json") != EXPECTED_SOURCE_MANIFEST \
            or sha256(SOURCE / "result.json") != EXPECTED_SOURCE_RESULT \
            or manifest["inputs_sha256"] != sha256(OUT / "inputs.json") \
            or manifest["budget_sha256"] != sha256(OUT / "budget_estimate.json") \
            or manifest["environment_sha256"] != sha256(OUT / "environment_lock.json") \
            or manifest["rotation_attestation_sha256"] != sha256(ROTATION_ATTESTATION):
        raise ValueError("Frozen baseline artifact changed")
    if load(OUT / "inputs.json") != source_inputs():
        raise ValueError("Frozen baseline inputs changed")
    if load(OUT / "budget_estimate.json") != budget():
        raise ValueError("Frozen review budget changed")
    for relative, expected in manifest["dependencies"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Dependency changed: {relative}")
    if full_environment and environment_lock() != load(OUT / "environment_lock.json"):
        raise ValueError("Execution environment changed")
    journal_path = OUT / "private/pilot/journal.jsonl"
    accounting = None
    if journal_path.exists():
        accounting = ReviewJournal(
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
        raise PermissionError("Authorization must exactly match the review cap")
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
        raise PermissionError("No immutable review authorization")
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


def review_path(label):
    return OUT / "reviews" / f"review-{label.lower()}.json"


def run_review(journal, label, document):
    path = review_path(label)
    prompt = review_prompt(label, document["text"])
    system_prompt = (
        "You are a strict text-quality adjudicator. Follow the JSON schema "
        "exactly and evaluate only the supplied passage."
    )
    call_id = f"event-log-review-{label.lower()}"
    base = {
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "inputs_sha256": sha256(OUT / "inputs.json"),
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
            raise GlobalStop("Saved adjudication identity changed")
        return row
    response = response_from_journal(journal, call_id)
    if response is None:
        response = journal.request(
            call_id, prompt, system_prompt, 50, 0.7, 0.9
        )
    parsed = parse_review(response["text"]) if response["stop_reason"] == "stop" else None
    return freeze(path, {
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
    })


def quality(max_new_calls):
    check()
    require_authorization()
    require_price()
    result_path = OUT / "quality_result.json"
    if result_path.exists():
        return load(result_path)
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise PermissionError("DEEPSEEK_API_KEY is not set")
    if type(max_new_calls) is not int or max_new_calls < 1:
        raise ValueError("Explicit positive --max-new-calls required")
    journal = ReviewJournal(
        max_new_calls,
        lambda body: transport(body, key),
        sha256(OUT / "manifest.json"),
    )
    document = load(OUT / "inputs.json")["document"]
    reviews = []
    for label in REVIEW_LABELS:
        try:
            reviews.append(run_review(journal, label, document))
        except Pause:
            return {
                "state": "paused_before_review",
                "review": label,
                **journal.totals(),
            }
        except Exception as exc:
            if not any(event["kind"] == "halt" for event in journal.events):
                journal.event(
                    "halt",
                    reason=f"review_failure:{type(exc).__name__}",
                )
            raise
    passes = sum(row["pass"] for row in reviews)
    result = {
        "experiment": "swan_coherent_baseline1",
        "completed_at": now(),
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "inputs_sha256": sha256(OUT / "inputs.json"),
        "status": "qualified_for_detection" if passes >= 2
        else "quality_gate_failed",
        "review_passes": passes,
        "review_passes_required": 2,
        "qualified_for_detection": passes >= 2,
        "human_quality_reviewed": False,
        "posthoc_genre_adjudication": True,
        "reviews": [{
            "label": row["label"],
            "path": str(review_path(row["label"]).relative_to(ROOT)),
            "sha256": sha256(review_path(row["label"])),
            "parsed": row["parsed"],
            "pass": row["pass"],
        } for row in reviews],
        "accounting": journal.totals(),
    }
    return freeze(result_path, result)


def seed(*parts):
    return int(digest(["swan_coherent_baseline1", *parts])[:8], 16)


def score_unit(module, bank, document_id, index, source):
    diagnostics = None
    try:
        with injection.capture_candidate_diagnostics() as diagnostics:
            raw = module.parse_amr(source["text"])
            normalized = module.normalize_amr_variables(raw)
            templated = module.generate_template_amr(normalized)
            candidate = module.normalize_amr_variables(templated)
            scores = []
            for bank_index, graph in enumerate(bank):
                precision, recall, f1 = [
                    float(value)
                    for value in score_triple(
                        candidate,
                        graph,
                        seed("detect", document_id, index, bank_index),
                    )
                ]
                if any(
                    not math.isfinite(value) or not 0 <= value <= 1
                    for value in (precision, recall, f1)
                ):
                    raise ValueError("Invalid S2Match score")
                scores.append({
                    "bank_index": bank_index,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                })
    except Exception as exc:
        failure = {
            "document_id": document_id,
            "unit_index": index,
            "text_sha256": source["text_sha256"],
            "diagnostics": diagnostics,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        freeze(OUT / f"detection_failure_unit_{index:02d}.json", failure)
        raise
    if diagnostics["logged_warnings"] \
            or diagnostics["python_warnings"] \
            or diagnostics["stderr"]:
        failure = {
            "document_id": document_id,
            "unit_index": index,
            "text_sha256": source["text_sha256"],
            "diagnostics": diagnostics,
            "error_type": "DetectionDiagnostic",
            "error_message": "Detection emitted a prohibited diagnostic",
        }
        freeze(OUT / f"detection_failure_unit_{index:02d}.json", failure)
        raise RuntimeError(f"Detection diagnostic in unit {index}")
    maximum = max(row["f1"] for row in scores)
    return {
        "unit_index": index,
        "text": source["text"],
        "text_sha256": source["text_sha256"],
        "source_slot": source["slot"],
        "source_target_template_index": source["target_template_index"],
        "source_target_score": source["target_score"],
        "raw_amr": raw,
        "raw_amr_sha256": text_hash(raw),
        "candidate_template_amr": candidate,
        "candidate_template_amr_sha256": text_hash(candidate),
        "diagnostics": diagnostics,
        "scores": scores,
        "max_score": maximum,
        "max_bank_indices": [
            row["bank_index"] for row in scores if row["f1"] == maximum
        ],
        "green": maximum >= GREEN_THRESHOLD,
    }


def detect():
    check()
    quality_path = OUT / "quality_result.json"
    if not quality_path.is_file():
        raise PermissionError("Quality adjudication has not run")
    quality_result = load(quality_path)
    if quality_result["manifest_sha256"] != sha256(OUT / "manifest.json") \
            or quality_result["inputs_sha256"] != sha256(OUT / "inputs.json") \
            or not quality_result["qualified_for_detection"]:
        raise PermissionError("Frozen quality gate did not authorize detection")
    result_path = OUT / "detection.json"
    if result_path.exists():
        return load(result_path)
    inputs = load(OUT / "inputs.json")
    document = inputs["document"]
    bank = [row["normalized_template_amr"] for row in load(BANK)]
    units = []
    with official_runtime(OUT, error_folder=OUT / "errors") as module:
        for index, source in enumerate(inputs["selected"]):
            units.append(score_unit(
                module,
                bank,
                document["document_id"],
                index,
                source,
            ))
    green = sum(row["green"] for row in units)
    n = len(units)
    z = (green - NULL_RATE * n) / math.sqrt(
        n * NULL_RATE * (1 - NULL_RATE)
    )
    result = {
        "completed_at": now(),
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "inputs_sha256": sha256(OUT / "inputs.json"),
        "quality_result_sha256": sha256(quality_path),
        "document_id": document["document_id"],
        "document_text_sha256": document["text_sha256"],
        "N": n,
        "green": green,
        "green_rate": green / n,
        "z": z,
        "detected_at_1pct_theoretical": z > 2.33,
        "detected_at_5pct_theoretical": z > 1.645,
        "independent_empirical_calibration_pending": True,
        "new_generation_calls": 0,
        "provider_calls_during_detection": 0,
        "human_quality_reviewed": False,
        "posthoc_genre_adjudication": True,
        "units": units,
    }
    return freeze(result_path, result)


def status():
    result = {
        "experiment": "swan_coherent_baseline1",
        "manifest_present": (OUT / "manifest.json").exists(),
        "authorization_present": (OUT / "authorization.json").exists(),
        "quality_present": (OUT / "quality_result.json").exists(),
        "detection_present": (OUT / "detection.json").exists(),
    }
    if (OUT / "manifest.json").exists():
        result["manifest_sha256"] = sha256(OUT / "manifest.json")
        journal_path = OUT / "private/pilot/journal.jsonl"
        if journal_path.exists():
            result["accounting"] = ReviewJournal(
                1, None, sha256(OUT / "manifest.json"), allow_halted=True
            ).totals()
    if (OUT / "quality_result.json").exists():
        result["quality"] = load(OUT / "quality_result.json")
        result["quality_sha256"] = sha256(OUT / "quality_result.json")
    if (OUT / "detection.json").exists():
        detection = load(OUT / "detection.json")
        result["detection_sha256"] = sha256(OUT / "detection.json")
        result["detection"] = {
            key: detection[key]
            for key in (
                "N",
                "green",
                "green_rate",
                "z",
                "detected_at_1pct_theoretical",
                "detected_at_5pct_theoretical",
                "independent_empirical_calibration_pending",
            )
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "check",
            "authorize",
            "record-price",
            "quality",
            "detect",
            "status",
        ),
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
        elif args.command == "quality":
            result = quality(args.max_new_calls)
        elif args.command == "detect":
            result = detect()
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
