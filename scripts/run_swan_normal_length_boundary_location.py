"""Run a detector-blind count-neutral boundary-location pilot."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import run_swan_injection_only_pilot as injection
from swan_cached_s2match import CachedS2Matcher
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
from swan_final1_transform import (
    FrozenSyntax,
    _candidates,
    apply_edits,
    punctuation,
    sentence_spans,
    text_hash,
    unit_texts,
    word_tokens,
)

OUT = ROOT / "experiments/swan_normal_length_boundary_location1"
SOURCE_PILOT = ROOT / "experiments/swan_normal_length_coherent_pilot1"
SOURCE_BASELINE = ROOT / "experiments/swan_normal_length_coherent_baseline1"
BANK = ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_50.json"
GLOVE = ROOT / "vectors/glove.6B.100d.txt"
ROTATION_ATTESTATION = (
    ROOT / "experiments/swan_normal_length_feasibility2/authorization_pilot.json"
)
EXPECTED_PILOT_MANIFEST = (
    "bd02fce51992f9f7fcecf4586505de26ce1c030eca9c97345fb61bee1624cb3f"
)
EXPECTED_PILOT_RESULT = (
    "d71fe9a9946c1c4feff85ef49122267b5a1f1c724294c1b35b1a45b30d042587"
)
EXPECTED_BASELINE_MANIFEST = (
    "413e0d03dbda7cbeb5a57f7fc5732d4cd39d51381d249f9253fc34e730832ba0"
)
EXPECTED_BASELINE_INPUTS = (
    "0902cf00eaba0454152902a8f7aa964ef85c0e9e29c7f339cbf798cfe9bb5261"
)
EXPECTED_BASELINE_DETECTION = (
    "589efc3cd93c809ee150163b18592c71e7e7b78738811dcbfc2842e2e051c69d"
)
EXPECTED_BANK = "f0a88c1af73d12078586d10ef68804c12ed5b79cabc07c9f6c2a34bb084ef5a4"
GREEN_THRESHOLD = 0.65
NULL_RATE = 0.25
REVIEW_LABELS = ("A", "B")
MAX_CALLS = len(REVIEW_LABELS)
MAX_COST_NANODOLLARS = MAX_CALLS * MAX_RESERVE
PILOT_CAP = (MAX_CALLS, MAX_COST_NANODOLLARS)
NAMESPACE = "swan-normal-length-boundary-location1"


def record(text):
    spans = sentence_spans(text)
    return {
        "text": text,
        "text_sha256": text_hash(text),
        "word_tokens": word_tokens(text),
        "word_tokens_sha256": digest(word_tokens(text)),
        "character_length": len(text),
        "punctuation": dict(sorted(punctuation(text).items())),
        "segmentation_spans": spans,
        "num_detection_units": len(spans),
        "unit_texts": unit_texts(text, spans),
    }


def merged_condition(text, merge):
    edit = {
        "position": merge["position"],
        "before": ".",
        "after": ";",
        "kind": "merge",
    }
    changed = apply_edits(text, [edit])
    row = {
        **record(changed),
        "merge": merge,
        "punctuation_edits": [edit],
    }
    original = record(text)
    if row["word_tokens"] != original["word_tokens"] \
            or row["character_length"] != original["character_length"]:
        raise ValueError("Boundary merge changed words, order, or length")
    if row["num_detection_units"] != original["num_detection_units"] - 1:
        raise ValueError("Boundary merge did not remove exactly one unit")
    return row


def select_pair(text):
    syntax = FrozenSyntax()
    c0 = record(text)
    merges, splits = _candidates(text, c0["segmentation_spans"], syntax)
    pairs = [
        (left, right)
        for index, left in enumerate(merges)
        for right in merges[index + 1:]
        if abs(left["unit_index"] - right["unit_index"]) > 1
    ]
    pairs.sort(key=lambda pair: (
        pair[0]["unit_index"],
        pair[1]["unit_index"],
    ))
    if not pairs:
        raise ValueError("No nonadjacent boundary-location pair")
    selector = int(hashlib.sha256(
        f"{NAMESPACE}:{c0['text_sha256']}".encode()
    ).hexdigest(), 16)
    pair = list(pairs[selector % len(pairs)])
    if (selector // len(pairs)) % 2:
        pair.reverse()
    return c0, merges, splits, pairs, pair


def build_inputs():
    paths = {
        "pilot_manifest": SOURCE_PILOT / "manifest.json",
        "pilot_result": SOURCE_PILOT / "result.json",
        "baseline_manifest": SOURCE_BASELINE / "manifest.json",
        "baseline_inputs": SOURCE_BASELINE / "inputs.json",
        "baseline_detection": SOURCE_BASELINE / "detection.json",
    }
    expected = {
        "pilot_manifest": EXPECTED_PILOT_MANIFEST,
        "pilot_result": EXPECTED_PILOT_RESULT,
        "baseline_manifest": EXPECTED_BASELINE_MANIFEST,
        "baseline_inputs": EXPECTED_BASELINE_INPUTS,
        "baseline_detection": EXPECTED_BASELINE_DETECTION,
    }
    for key, path in paths.items():
        if sha256(path) != expected[key]:
            raise ValueError(f"Frozen upstream changed: {key}")
    pilot = load(paths["pilot_result"])
    baseline_inputs = load(paths["baseline_inputs"])
    detection = load(paths["baseline_detection"])
    if pilot["status"] != "normal_length_baseline_ready" \
            or pilot["fallback_slots"] != 0 \
            or not pilot["auxiliary_quality_gate_pass"] \
            or detection["N"] != 20 \
            or detection["green"] != 20 \
            or detection["document_text_sha256"] != pilot["document"]["text_sha256"]:
        raise ValueError("Upstream is not the frozen detector-positive baseline")
    text = pilot["document"]["text"]
    c0, merges, splits, pairs, pair = select_pair(text)
    if splits:
        raise ValueError("Standard C2 status changed: internal split now exists")
    r0 = merged_condition(text, pair[0])
    r1 = merged_condition(text, pair[1])
    if r0["num_detection_units"] != r1["num_detection_units"] \
            or r0["num_detection_units"] != 19:
        raise ValueError("R0/R1 detector-unit count differs")
    if r0["word_tokens"] != r1["word_tokens"] \
            or r0["punctuation"] != r1["punctuation"] \
            or r0["character_length"] != r1["character_length"]:
        raise ValueError("R0/R1 count-neutral invariants failed")
    if pair[0]["unit_index"] == pair[1]["unit_index"] \
            or abs(pair[0]["unit_index"] - pair[1]["unit_index"]) <= 1:
        raise ValueError("Selected boundaries are not nonadjacent")
    restored = apply_edits(
        r0["text"], r0["punctuation_edits"], reverse=True
    )
    if restored != text:
        raise ValueError("R4 exact restoration failed")
    blind_swap = int(text_hash(text), 16) % 2 == 1
    blind_order = ["R1", "R0"] if blind_swap else ["R0", "R1"]
    return {
        "upstream": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in paths.values()
        },
        "source_document": {
            "document_id": pilot["document"]["document_id"],
            "text_sha256": pilot["document"]["text_sha256"],
            "word_count": pilot["document"]["word_count"],
            "model_assisted_quality_gate_pass": True,
            "human_quality_reviewed": False,
        },
        "construction": {
            "algorithm": "frozen_boundary_location_v1",
            "detector_feedback": False,
            "merge_candidate_count": len(merges),
            "split_candidate_count": len(splits),
            "standard_C2_constructed": False,
            "standard_C2_failure": "no_eligible_internal_split",
            "nonadjacent_pair_count": len(pairs),
            "selected_boundaries": [row["unit_index"] for row in pair],
            "selected_merges": pair,
            "blind_review_order": blind_order,
        },
        "conditions": {
            "C0": c0,
            "R0": r0,
            "R1": r1,
            "R4": record(restored),
        },
        "source_selected": baseline_inputs["selected"],
        "source_detection_sha256": EXPECTED_BASELINE_DETECTION,
    }


def dependencies():
    paths = [
        Path(__file__),
        OUT / "protocol.md",
        OUT / "current_price_evidence.md",
        BANK,
        GLOVE,
        SOURCE_PILOT / "manifest.json",
        SOURCE_PILOT / "result.json",
        SOURCE_BASELINE / "manifest.json",
        SOURCE_BASELINE / "inputs.json",
        SOURCE_BASELINE / "detection.json",
        ROOT / "scripts/swan_cached_s2match.py",
        ROOT / "scripts/swan_final1_transform.py",
        ROOT / "scripts/run_swan_injection_only_pilot.py",
        ROOT / "scripts/swan_final1_budget.py",
        ROOT / "scripts/swan_final1_contract.py",
        ROOT / "scripts/swan_final1_quality.py",
        ROOT / "scripts/swan_final1_runtime.py",
        ROOT / "scripts/test_swan_normal_length_boundary_location.py",
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
    freeze(OUT / "inputs.json", build_inputs())
    freeze(OUT / "budget_estimate.json", budget())
    freeze(OUT / "environment_lock.json", environment_lock())
    rotation_hash = sha256(ROTATION_ATTESTATION)
    inputs = load(OUT / "inputs.json")
    manifest = freeze(OUT / "manifest.json", {
        "schema_version": 1,
        "experiment": "swan_normal_length_boundary_location1",
        "frozen_at": now(),
        "status": "frozen_before_quality_and_detection",
        "inputs_sha256": sha256(OUT / "inputs.json"),
        "budget_sha256": sha256(OUT / "budget_estimate.json"),
        "environment_sha256": sha256(OUT / "environment_lock.json"),
        "rotation_attestation_sha256": rotation_hash,
        "dependencies": dependencies(),
        "standard_C2_constructed": False,
        "standard_C2_failure": "no_eligible_internal_split",
        "relocation_conditions": ["R0", "R1"],
        "selected_boundaries": inputs["construction"]["selected_boundaries"],
        "review_count": MAX_CALLS,
        "review_passes_required": MAX_CALLS,
        "green_threshold": GREEN_THRESHOLD,
        "null_rate": NULL_RATE,
        "provider_calls_during_detection": 0,
        "human_quality_claim": False,
        "calibrated_fpr_claim": False,
        "population_attack_claim": False,
        "attack_authorized": True,
        "attack_scope": "exploratory_boundary_location_only",
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
        "selected_boundaries": manifest["selected_boundaries"],
        "standard_C2_constructed": False,
        "max_calls": MAX_CALLS,
        "max_cost_usd": budget()["max_cost_usd"],
    }


def check(full_environment=True):
    manifest = load(OUT / "manifest.json")
    valid = (
        manifest["experiment"] == "swan_normal_length_boundary_location1"
        and not manifest["standard_C2_constructed"]
        and manifest["standard_C2_failure"] == "no_eligible_internal_split"
        and manifest["relocation_conditions"] == ["R0", "R1"]
        and manifest["review_count"] == MAX_CALLS
        and manifest["review_passes_required"] == MAX_CALLS
        and manifest["green_threshold"] == GREEN_THRESHOLD
        and manifest["null_rate"] == NULL_RATE
        and manifest["provider_calls_during_detection"] == 0
        and not manifest["human_quality_claim"]
        and not manifest["calibrated_fpr_claim"]
        and not manifest["population_attack_claim"]
        and manifest["attack_authorized"]
        and manifest["attack_scope"] == "exploratory_boundary_location_only"
    )
    if not valid:
        raise ValueError("Boundary-location identity or claim boundary changed")
    if manifest["inputs_sha256"] != sha256(OUT / "inputs.json") \
            or manifest["budget_sha256"] != sha256(OUT / "budget_estimate.json") \
            or manifest["environment_sha256"] != sha256(OUT / "environment_lock.json") \
            or manifest["rotation_attestation_sha256"] != sha256(ROTATION_ATTESTATION):
        raise ValueError("Frozen boundary-location artifact changed")
    if load(OUT / "inputs.json") != build_inputs():
        raise ValueError("Detector-blind construction replay changed")
    if load(OUT / "budget_estimate.json") != budget():
        raise ValueError("Review budget changed")
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


def review_prompt(label, left, right):
    return (
        f"Blind pair review {label}. Compare only Text Left and Text Right. "
        "They are two punctuation variants of a concise project event log. "
        "Return exactly one JSON object with booleans meaning_preserved, "
        "same_events_and_roles, and severe_problem, plus integer scores from "
        "1 to 5 for left_naturalness and right_naturalness. Meaning is "
        "preserved only if facts, actors, roles, event order, polarity, and "
        "modality agree. Use no markdown and no extra keys.\n\n"
        f"TEXT LEFT:\n{left}\n\nTEXT RIGHT:\n{right}"
    )


def parse_review(text):
    try:
        row = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    keys = {
        "meaning_preserved",
        "same_events_and_roles",
        "left_naturalness",
        "right_naturalness",
        "severe_problem",
    }
    if not isinstance(row, dict) or set(row) != keys:
        return None
    if type(row["meaning_preserved"]) is not bool \
            or type(row["same_events_and_roles"]) is not bool \
            or type(row["severe_problem"]) is not bool:
        return None
    for key in ("left_naturalness", "right_naturalness"):
        if type(row[key]) is not int or not 1 <= row[key] <= 5:
            return None
    return row


def review_pass(row):
    return (
        row is not None
        and row["meaning_preserved"]
        and row["same_events_and_roles"]
        and not row["severe_problem"]
        and row["left_naturalness"] >= 3
        and row["right_naturalness"] >= 3
    )


def review_path(label):
    return OUT / "quality_reviews" / f"review-{label.lower()}.json"


def run_review(journal, label, inputs):
    order = inputs["construction"]["blind_review_order"]
    left = inputs["conditions"][order[0]]["text"]
    right = inputs["conditions"][order[1]]["text"]
    prompt = review_prompt(label, left, right)
    system_prompt = (
        "You are a strict blind semantic-preservation reviewer. Follow the "
        "requested JSON schema exactly."
    )
    call_id = f"boundary-quality-{label.lower()}"
    path = review_path(label)
    identity = {
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "inputs_sha256": sha256(OUT / "inputs.json"),
        "label": label,
        "call_id": call_id,
        "blind_order": order,
        "left_text_sha256": text_hash(left),
        "right_text_sha256": text_hash(right),
        "prompt_sha256": text_hash(prompt),
        "system_prompt_sha256": text_hash(system_prompt),
        "blind_to_condition_names_and_detection": True,
        "human_review": False,
    }
    if path.exists():
        row = load(path)
        if any(row.get(key) != value for key, value in identity.items()):
            raise GlobalStop("Saved boundary review identity changed")
        return row
    response = response_from_journal(journal, call_id)
    if response is None:
        response = journal.request(
            call_id, prompt, system_prompt, 50, 0.7, 0.9
        )
    parsed = parse_review(response["text"]) if response["stop_reason"] == "stop" else None
    return freeze(path, {
        **identity,
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
    path = OUT / "quality_result.json"
    if path.exists():
        return load(path)
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
    inputs = load(OUT / "inputs.json")
    reviews = []
    for label in REVIEW_LABELS:
        try:
            reviews.append(run_review(journal, label, inputs))
        except Pause:
            return {
                "state": "paused_before_review",
                "review": label,
                **journal.totals(),
            }
    passed = all(row["pass"] for row in reviews)
    return freeze(path, {
        "experiment": "swan_normal_length_boundary_location1",
        "completed_at": now(),
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "inputs_sha256": sha256(OUT / "inputs.json"),
        "status": "qualified_for_detection" if passed else "quality_gate_failed",
        "qualified_for_detection": passed,
        "human_quality_reviewed": False,
        "reviews": [{
            "label": row["label"],
            "path": str(review_path(row["label"]).relative_to(ROOT)),
            "sha256": sha256(review_path(row["label"])),
            "parsed": row["parsed"],
            "pass": row["pass"],
        } for row in reviews],
        "accounting": journal.totals(),
    })


def score_seed(*parts):
    return int(digest([NAMESPACE, *parts])[:8], 16)


def score_merged(module, cached, bank, condition, index, text):
    diagnostics = None
    try:
        with injection.capture_candidate_diagnostics() as diagnostics:
            raw = module.parse_amr(text)
            normalized = module.normalize_amr_variables(raw)
            templated = module.generate_template_amr(normalized)
            candidate = module.normalize_amr_variables(templated)
            scores = []
            for bank_index, graph in enumerate(bank):
                precision, recall, f1 = [
                    float(value)
                    for value in cached.score(
                        candidate,
                        graph,
                        score_seed("detect", condition, text_hash(text), bank_index),
                    )
                ]
                if any(
                    not math.isfinite(value) or not 0 <= value <= 1
                    for value in (precision, recall, f1)
                ):
                    raise ValueError("Invalid merged-unit score")
                scores.append({
                    "bank_index": bank_index,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                })
    except Exception as exc:
        freeze(OUT / f"detection_failure_{condition}_{index:02d}.json", {
            "condition": condition,
            "unit_index": index,
            "text_sha256": text_hash(text),
            "diagnostics": diagnostics,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        })
        raise
    if diagnostics["logged_warnings"] \
            or diagnostics["python_warnings"] \
            or diagnostics["stderr"]:
        raise RuntimeError(f"Detection diagnostic in {condition} unit {index}")
    maximum = max(row["f1"] for row in scores)
    return {
        "unit_index": index,
        "text": text,
        "text_sha256": text_hash(text),
        "reused_source_unit": False,
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
        "target_47_score": scores[47]["f1"],
        "green": maximum >= GREEN_THRESHOLD,
    }


def validate_equivalence(cached, bank, module, inputs):
    merged_text = next(
        text for text in inputs["conditions"]["R0"]["unit_texts"]
        if text not in {row["text"] for row in inputs["source_selected"]}
    )
    with injection.capture_candidate_diagnostics() as diagnostics:
        raw = module.parse_amr(merged_text)
        normalized = module.normalize_amr_variables(raw)
        templated = module.generate_template_amr(normalized)
        candidate = module.normalize_amr_variables(templated)
        seed = score_seed("equivalence", text_hash(merged_text), 47)
        official = tuple(float(value) for value in score_triple(
            candidate, bank[47], seed
        ))
        cached_value = tuple(float(value) for value in cached.score(
            candidate, bank[47], seed
        ))
    differences = [
        abs(left - right) for left, right in zip(official, cached_value)
    ]
    result = {
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "inputs_sha256": sha256(OUT / "inputs.json"),
        "merged_text_sha256": text_hash(merged_text),
        "bank_index": 47,
        "score_seed": seed,
        "official": list(official),
        "cached": list(cached_value),
        "absolute_differences": differences,
        "diagnostics": diagnostics,
        "pass": (
            not diagnostics["logged_warnings"]
            and not diagnostics["python_warnings"]
            and not diagnostics["stderr"]
            and all(value <= 1e-12 for value in differences)
        ),
    }
    freeze(OUT / "equivalence_check.json", result)
    if not result["pass"]:
        raise RuntimeError("Cached merged-unit equivalence check failed")
    return result


def condition_result(condition, condition_input, source_detection, module, cached, bank):
    source_by_text = {
        row["text"]: row for row in source_detection["units"]
    }
    units = []
    merged_count = 0
    for index, text in enumerate(condition_input["unit_texts"]):
        source = source_by_text.get(text)
        if source is not None:
            units.append({
                **source,
                "unit_index": index,
                "reused_source_unit": True,
                "source_unit_index": source["unit_index"],
            })
        else:
            units.append(score_merged(
                module, cached, bank, condition, index, text
            ))
            merged_count += 1
    expected_merged = 0 if condition in {"C0", "R4"} else 1
    if merged_count != expected_merged:
        raise ValueError(f"{condition} merged-unit count changed")
    green = sum(row["green"] for row in units)
    n = len(units)
    z = (green - NULL_RATE * n) / math.sqrt(
        n * NULL_RATE * (1 - NULL_RATE)
    )
    return {
        "condition": condition,
        "text_sha256": condition_input["text_sha256"],
        "word_tokens_sha256": condition_input["word_tokens_sha256"],
        "punctuation": condition_input["punctuation"],
        "N": n,
        "green": green,
        "green_rate": green / n,
        "z": z,
        "detected_at_1pct_theoretical": z > 2.33,
        "detected_at_5pct_theoretical": z > 1.645,
        "merged_units_scored": merged_count,
        "source_units_reused": n - merged_count,
        "units": units,
    }


def detect():
    check()
    quality_path = OUT / "quality_result.json"
    if not quality_path.is_file() or not load(quality_path)["qualified_for_detection"]:
        raise PermissionError("Boundary pair quality gate did not pass")
    result_path = OUT / "detection.json"
    if result_path.exists():
        return load(result_path)
    inputs = load(OUT / "inputs.json")
    source_detection = load(SOURCE_BASELINE / "detection.json")
    bank = [row["normalized_template_amr"] for row in load(BANK)]
    with official_runtime(OUT, error_folder=OUT / "errors") as module:
        cached = CachedS2Matcher(GLOVE)
        equivalence = validate_equivalence(cached, bank, module, inputs)
        conditions = {
            condition: condition_result(
                condition,
                inputs["conditions"][condition],
                source_detection,
                module,
                cached,
                bank,
            )
            for condition in ("C0", "R0", "R1", "R4")
        }
    r0, r1 = conditions["R0"], conditions["R1"]
    result = {
        "completed_at": now(),
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "inputs_sha256": sha256(OUT / "inputs.json"),
        "quality_result_sha256": sha256(quality_path),
        "equivalence_check_sha256": sha256(OUT / "equivalence_check.json"),
        "equivalence_pass": equivalence["pass"],
        "standard_C2_constructed": False,
        "standard_C2_failure": "no_eligible_internal_split",
        "primary_comparison": "R0_vs_R1_boundary_location_at_constant_N",
        "conditions": conditions,
        "paired_difference_R1_minus_R0": {
            "N": r1["N"] - r0["N"],
            "green": r1["green"] - r0["green"],
            "green_rate": r1["green_rate"] - r0["green_rate"],
            "z": r1["z"] - r0["z"],
            "detection_transition_1pct": (
                f"{int(r0['detected_at_1pct_theoretical'])}->"
                f"{int(r1['detected_at_1pct_theoretical'])}"
            ),
        },
        "word_sequence_identical": (
            inputs["conditions"]["R0"]["word_tokens"]
            == inputs["conditions"]["R1"]["word_tokens"]
        ),
        "punctuation_multiset_identical": (
            inputs["conditions"]["R0"]["punctuation"]
            == inputs["conditions"]["R1"]["punctuation"]
        ),
        "exact_restoration": (
            inputs["conditions"]["R4"]["text"]
            == inputs["conditions"]["C0"]["text"]
        ),
        "provider_calls_during_detection": 0,
        "human_quality_reviewed": False,
        "empirical_calibration_pending": True,
        "population_attack_claim": False,
    }
    return freeze(result_path, result)


def status():
    result = {
        "experiment": "swan_normal_length_boundary_location1",
        "manifest_present": (OUT / "manifest.json").exists(),
        "quality_present": (OUT / "quality_result.json").exists(),
        "equivalence_present": (OUT / "equivalence_check.json").exists(),
        "detection_present": (OUT / "detection.json").exists(),
    }
    if (OUT / "manifest.json").exists():
        result["manifest_sha256"] = sha256(OUT / "manifest.json")
    if (OUT / "quality_result.json").exists():
        result["quality"] = load(OUT / "quality_result.json")
        result["quality_sha256"] = sha256(OUT / "quality_result.json")
    if (OUT / "detection.json").exists():
        detection = load(OUT / "detection.json")
        result["detection_sha256"] = sha256(OUT / "detection.json")
        result["summary"] = {
            "standard_C2_constructed": detection["standard_C2_constructed"],
            "primary_comparison": detection["primary_comparison"],
            "conditions": {
                key: {
                    metric: row[metric]
                    for metric in ("N", "green", "green_rate", "z",
                                   "detected_at_1pct_theoretical")
                }
                for key, row in detection["conditions"].items()
            },
            "paired_difference_R1_minus_R0": (
                detection["paired_difference_R1_minus_R0"]
            ),
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
        ) else "See saved provenance; no automatic retry"
        print(json.dumps({
            "state": "stopped",
            "error_type": type(exc).__name__,
            "reason": reason,
        }), file=sys.stderr)
        raise
