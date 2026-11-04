"""Run the discourse-linked follow-up for a coherent SWAN-positive baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import run_swan_coherent_injection_pilot as base
from swan_feasibility2_common import (
    ROOT,
    digest as canonical_digest,
    freeze,
    load,
    now,
    sha256,
    single_writer,
)
from swan_final1_budget import GlobalStop, MAX_RESERVE
from swan_final1_contract import environment_lock
from swan_final1_transform import sentence_spans, text_hash, unit_texts, word_tokens

OUT = ROOT / "experiments/swan_coherent_injection_pilot3"
PILOT1_RESULT = ROOT / "experiments/swan_coherent_injection_pilot1/result.json"
PILOT2_RESULT = ROOT / "experiments/swan_coherent_injection_pilot2/result.json"
EXPECTED_PILOT1_RESULT = (
    "1cafd7b20e811c91c9de5d49cfbe0058d2928935dea4a193b832fe3807dad329"
)
EXPECTED_PILOT2_RESULT = (
    "b7d699f39b8521c1a16f7afe181b7881a6cfa0a88b25cbf0f420c14a1d41bde0"
)
TEMPLATE_SCHEDULE = (47,) * 10
PREFERRED_REALIZATIONS = (
    "Initially, the United Nations released a Boston housing report.",
    "Maria Lopez then received the report.",
    "David Chen subsequently reviewed the findings.",
    "Maria Lopez then examined the policy.",
    "The United Nations consequently proposed changes.",
    "David Chen then explained the process.",
    "Maria Lopez subsequently requested further details.",
    "The United Nations then revised the report.",
    "David Chen consequently prepared the final briefing.",
    "Finally, Maria Lopez approved the recommendations.",
)
SLOT_PURPOSES = tuple(
    f"Use this preferred realization: {text}"
    for text in PREFERRED_REALIZATIONS
)
FIXED_TOPIC = (
    "The United Nations released a Boston housing policy report. Analyst Maria "
    "Lopez and editor David Chen process it into a concise chronological briefing."
)
THEME_TOKENS = frozenset({
    "approved",
    "boston",
    "briefing",
    "changes",
    "chen",
    "consequently",
    "david",
    "details",
    "explained",
    "finally",
    "findings",
    "housing",
    "initially",
    "lopez",
    "maria",
    "nations",
    "policy",
    "prepared",
    "process",
    "recommendations",
    "released",
    "report",
    "requested",
    "reviewed",
    "revised",
    "subsequently",
    "then",
    "united",
})
MAX_ATTEMPTS = 20
MAX_CALLS = 1 + len(TEMPLATE_SCHEDULE) * MAX_ATTEMPTS + base.REVIEW_COUNT
MAX_COST_NANODOLLARS = MAX_CALLS * MAX_RESERVE
PILOT_CAP = (MAX_CALLS, MAX_COST_NANODOLLARS)
BASE_FROZEN_DESIGN = base.frozen_design


def pilot_digest(value):
    if isinstance(value, list) and value \
            and value[0] == "swan_coherent_injection_pilot1":
        value = ["swan_coherent_injection_pilot3", *value[1:]]
    return canonical_digest(value)


def historical_hashes():
    sources = (
        (base.HISTORICAL_RESULT, base.EXPECTED_HISTORICAL_RESULT),
        (PILOT1_RESULT, EXPECTED_PILOT1_RESULT),
        (PILOT2_RESULT, EXPECTED_PILOT2_RESULT),
    )
    hashes = set()
    for path, expected in sources:
        if sha256(path) != expected:
            raise ValueError(f"Historical result changed: {path.name}")
        row = load(path)
        if row["fallback_slots"] != 0:
            raise ValueError(f"Historical result contains fallback: {path.name}")
        hashes.update(item["text_sha256"] for item in row["selected"])
    if len(hashes) != 24:
        raise ValueError("Unexpected historical strict-acceptance identity count")
    return hashes


def context_for(slot, previous):
    if len(previous) != slot:
        raise ValueError("Context history does not match slot")
    recent = previous[-base.MAX_PRIOR_CONTEXT:]
    history = "\n".join(f"- {text}" for text in recent) if recent else "(none)"
    return (
        "Write the next sentence of a cohesive chronological workflow summary.\n"
        f"FIXED TOPIC: {FIXED_TOPIC}\n"
        f"PREFERRED REALIZATION {slot + 1}/10: {PREFERRED_REALIZATIONS[slot]}\n"
        f"RECENT ACCEPTED SENTENCES:\n{history}\n"
        "The target AMR is mandatory. Use the preferred realization exactly if "
        "it is compatible with that graph; otherwise make the smallest necessary "
        "change. Keep the discourse link, use 3 to 18 words, and output exactly "
        "one complete sentence."
    )


def review_prompt(label, document):
    return (
        f"Blind review pass {label}. Evaluate only the passage below as a concise "
        "chronological workflow summary. Return exactly one JSON object with "
        "integer scores from 1 to 5 for coherence, naturalness, nonredundancy, "
        "and task_consistency, plus boolean severe_problem. Coherence requires "
        "a clear event progression; task_consistency requires stable actors, "
        "topic, and outcome. Parallel sentence structure is acceptable when "
        "events are distinct. Use no markdown and no extra keys.\n\n"
        f"PASSAGE:\n{document}"
    )


def dependencies():
    paths = [
        Path(__file__),
        OUT / "protocol.md",
        OUT / "current_price_evidence.md",
        base.BANK,
        base.EXAMPLES,
        base.HISTORICAL_RESULT,
        PILOT1_RESULT,
        PILOT2_RESULT,
        ROOT / "scripts/run_swan_coherent_injection_pilot.py",
        ROOT / "scripts/run_swan_injection_only_pilot.py",
        ROOT / "scripts/run_swan_positive_collection.py",
        ROOT / "scripts/swan_feasibility2_common.py",
        ROOT / "scripts/swan_final1_budget.py",
        ROOT / "scripts/swan_final1_contract.py",
        ROOT / "scripts/swan_final1_quality.py",
        ROOT / "scripts/swan_final1_runtime.py",
        ROOT / "scripts/swan_final1_transform.py",
        ROOT / "scripts/test_swan_coherent_injection_pilot3.py",
    ]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def frozen_design():
    result = BASE_FROZEN_DESIGN()
    result.update({
        "adaptive_followup": True,
        "predecessor_experiment": "swan_coherent_injection_pilot2",
        "predecessor_result_sha256": EXPECTED_PILOT2_RESULT,
        "preferred_realizations": list(PREFERRED_REALIZATIONS),
        "target_selection_limitation": (
            "All slots use favorable target 47 and fixed wording guidance"
        ),
        "review_genre": "concise chronological workflow summary",
    })
    return result


def prepare():
    if (OUT / "manifest.json").exists():
        return check()
    if sha256(base.BANK) != base.EXPECTED_BANK:
        raise ValueError("Bank50 changed")
    historical_hashes()
    rotation_hash = sha256(base.ROTATION_ATTESTATION)
    freeze(OUT / "budget_estimate.json", base.budget())
    freeze(OUT / "environment_lock.json", environment_lock())
    freeze(OUT / "design.json", frozen_design())
    manifest = freeze(OUT / "manifest.json", {
        "schema_version": 1,
        "experiment": "swan_coherent_injection_pilot3",
        "frozen_at": now(),
        "status": "frozen_preexecution",
        "development_pilot": True,
        "adaptive_followup": True,
        "predecessor_result_sha256": EXPECTED_PILOT2_RESULT,
        "slots": base.MAX_SLOTS,
        "max_attempts_per_slot": MAX_ATTEMPTS,
        "match_threshold": base.MATCH_THRESHOLD,
        "bank_sha256": base.EXPECTED_BANK,
        "historical_result_sha256": base.EXPECTED_HISTORICAL_RESULT,
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
        "max_cost_usd": base.budget()["max_cost_usd"],
        "design_sha256": sha256(OUT / "design.json"),
    }


def check(full_environment=True):
    manifest = load(OUT / "manifest.json")
    identity_ok = (
        manifest["experiment"] == "swan_coherent_injection_pilot3"
        and manifest["development_pilot"]
        and manifest["adaptive_followup"]
        and manifest["predecessor_result_sha256"] == EXPECTED_PILOT2_RESULT
        and manifest["slots"] == base.MAX_SLOTS
        and manifest["max_attempts_per_slot"] == MAX_ATTEMPTS
        and manifest["match_threshold"] == base.MATCH_THRESHOLD
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
    if sha256(base.BANK) != base.EXPECTED_BANK \
            or sha256(base.HISTORICAL_RESULT) != base.EXPECTED_HISTORICAL_RESULT \
            or sha256(PILOT1_RESULT) != EXPECTED_PILOT1_RESULT \
            or sha256(PILOT2_RESULT) != EXPECTED_PILOT2_RESULT \
            or manifest["budget_sha256"] != sha256(OUT / "budget_estimate.json") \
            or manifest["environment_sha256"] != sha256(OUT / "environment_lock.json") \
            or manifest["design_sha256"] != sha256(OUT / "design.json") \
            or manifest["rotation_attestation_sha256"] \
            != sha256(base.ROTATION_ATTESTATION):
        raise ValueError("Frozen Pilot artifact changed")
    if load(OUT / "budget_estimate.json") != base.budget():
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
        accounting = base.PilotJournal(
            1, None, sha256(OUT / "manifest.json"), allow_halted=True
        ).totals()
    return {
        "state": "frozen_hashes_verified",
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "environment_verified": full_environment,
        "accounting": accounting,
    }


def document_from_slots(slots):
    if len(slots) != base.MAX_SLOTS or not all(row["accepted"] for row in slots):
        raise ValueError("A coherent document requires ten accepted slots")
    text = " ".join(row["text"] for row in slots)
    spans = sentence_spans(text)
    texts = unit_texts(text, spans)
    if len(spans) != base.MAX_SLOTS or texts != [row["text"] for row in slots]:
        raise GlobalStop("Accepted composition did not preserve ten exact units")
    return {
        "document_id": "coherent-injection-pilot3-00",
        "text": text,
        "text_sha256": text_hash(text),
        "word_count": len(word_tokens(text)),
        "segmentation_spans": spans,
        "sentence_text_sha256": [row["text_sha256"] for row in slots],
    }


def result_from_slots(slots, reviews, journal):
    accepted = [row for row in slots if row["accepted"]]
    hashes = [row["text_sha256"] for row in accepted]
    if len(hashes) != len(set(hashes)):
        raise GlobalStop("Accepted slot texts are not unique")
    complete = len(accepted) == base.MAX_SLOTS
    document = document_from_slots(slots) if complete else None
    auxiliary_pass = (
        complete
        and len(reviews) == base.REVIEW_COUNT
        and all(row["pass"] for row in reviews)
    )
    status = (
        "slot_exhausted" if not complete
        else "coherent_baseline_ready" if auxiliary_pass
        else "auxiliary_coherence_gate_failed"
    )
    result = {
        "experiment": "swan_coherent_injection_pilot3",
        "completed_at": now(),
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "status": status,
        "adaptive_followup": True,
        "predecessor_result_sha256": EXPECTED_PILOT2_RESULT,
        "slots_started": len(slots),
        "accepted_slots": len(accepted),
        "exhausted_slots": sum(not row["accepted"] for row in slots),
        "target_accepted": base.MAX_SLOTS,
        "fallback_slots": 0,
        "selected": [{
            "slot": row["slot"],
            "text": row["text"],
            "text_sha256": row["text_sha256"],
            "target_template_index": row["target_template_index"],
            "score": row["score"],
            "context_sha256": row["context_sha256"],
            "slot_sha256": sha256(base.slot_path(row["slot"])),
        } for row in accepted],
        "slot_sha256": {
            str(row["slot"]): sha256(base.slot_path(row["slot"])) for row in slots
        },
        "document": document,
        "auxiliary_reviews": [{
            "label": row["label"],
            "path": str(base.review_path(row["label"]).relative_to(ROOT)),
            "sha256": sha256(base.review_path(row["label"])),
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


def status():
    manifest = OUT / "manifest.json"
    result_path = OUT / "result.json"
    result = {
        "experiment": "swan_coherent_injection_pilot3",
        "manifest_present": manifest.exists(),
        "authorization_present": (OUT / "authorization.json").exists(),
        "price_check_present": (OUT / "price_check.json").exists(),
        "result_present": result_path.exists(),
    }
    if manifest.exists():
        result["manifest_sha256"] = sha256(manifest)
        journal_path = OUT / "private/pilot/journal.jsonl"
        if journal_path.exists():
            result["accounting"] = base.PilotJournal(
                1, None, sha256(manifest), allow_halted=True
            ).totals()
    if result_path.exists():
        result["result"] = load(result_path)
        result["result_sha256"] = sha256(result_path)
    return result


def configure_base():
    base.OUT = OUT
    base.TEMPLATE_SCHEDULE = TEMPLATE_SCHEDULE
    base.SLOT_PURPOSES = SLOT_PURPOSES
    base.FIXED_TOPIC = FIXED_TOPIC
    base.THEME_TOKENS = THEME_TOKENS
    base.MAX_ATTEMPTS = MAX_ATTEMPTS
    base.MAX_CALLS = MAX_CALLS
    base.MAX_COST_NANODOLLARS = MAX_COST_NANODOLLARS
    base.PILOT_CAP = PILOT_CAP
    base.digest = pilot_digest
    base.historical_hashes = historical_hashes
    base.context_for = context_for
    base.review_prompt = review_prompt
    base.dependencies = dependencies
    base.frozen_design = frozen_design
    base.prepare = prepare
    base.check = check
    base.document_from_slots = document_from_slots
    base.result_from_slots = result_from_slots
    base.status = status


def main():
    configure_base()
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
            result = base.authorize(
                args.authorized_by,
                args.authorization_text,
                args.max_calls,
                args.max_cost_usd,
            )
        elif args.command == "record-price":
            result = base.record_price()
        elif args.command == "run":
            result = base.run(args.max_new_calls)
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
