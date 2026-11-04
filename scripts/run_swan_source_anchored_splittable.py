"""Run one source-anchored, splittable, normal-length SWAN pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from swan_cached_s2match import CachedS2Matcher
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
from swan_final1_runtime import official_runtime, score_triple
from swan_final1_transform import (
    FrozenSyntax,
    LEAK_RE,
    _candidates,
    apply_edits,
    punctuation,
    sentence_spans,
    text_hash,
    unit_texts,
    unsupported,
    word_tokens,
)

OUT = ROOT / "experiments/swan_source_anchored_splittable1"
SOURCE_RANGE = ("train", 1_650_000, 1_651_000)
SOURCE_PATH = OUT / "source.json"
SOURCE_MANIFEST = OUT / "source_manifest.json"
SNAPSHOT_PATH = OUT / "prior_identity_snapshot.json"
SCREENING_PATH = OUT / "source_screening.json"
GLOVE = ROOT / "vectors/glove.6B.100d.txt"
EXPECTED_GLOVE = "95dde4dfd627ab26608d33e76d1195ec059734bd29089ea52cadb08d07c64544"
NAMESPACE = "swan-source-anchored-splittable1"
TEMPLATE_SCHEDULE = (47,) * 21
SLOT_PURPOSES = tuple(
    f"Express the frozen source detail assigned to slot {slot + 1}."
    for slot in range(21)
)
FIXED_TOPIC = "A concise factual digest of the frozen source window."
THEME_TOKENS = frozenset()
REVIEW_KEYS = (
    "coherence",
    "naturalness",
    "nonredundancy",
    "task_consistency",
    "source_consistency",
)
MAX_SLOTS = 21
MAX_ATTEMPTS = 30
MIN_WORDS = 5
MAX_WORDS = 12
MIN_DOCUMENT_WORDS = 100
MAX_DOCUMENT_WORDS = 220
REVIEW_COUNT = 2
MAX_CALLS = 1 + MAX_SLOTS * MAX_ATTEMPTS + REVIEW_COUNT
MAX_COST_NANODOLLARS = MAX_CALLS * MAX_RESERVE
PILOT_CAP = (MAX_CALLS, MAX_COST_NANODOLLARS)
GREEN_THRESHOLD = 0.65
NULL_RATE = 0.25
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
            or manifest["final1_sealed_content_access"]:
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
                "allocation_id": "source-anchored-splittable-pilot-0000",
                "dataset_revision": REVISION,
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


def anchor_schedule(source=None):
    source = source or source_record()
    counts = source["anchor_word_counts"]
    longest = max(range(len(counts)), key=lambda index: (counts[index], -index))
    schedule = []
    for index in range(10):
        schedule.extend((index, index))
        if index == longest:
            schedule.append(index)
    if len(schedule) != MAX_SLOTS or schedule[:2] != [0, 0]:
        raise ValueError("Frozen source anchor schedule is invalid")
    return tuple(schedule)


def variant_number(slot, schedule=None):
    schedule = schedule or anchor_schedule()
    anchor = schedule[slot]
    return sum(value == anchor for value in schedule[:slot + 1])


def pilot_digest(value):
    if isinstance(value, list) and value \
            and value[0] == "swan_coherent_injection_pilot1":
        value = [NAMESPACE, *value[1:]]
    return canonical_digest(value)


def historical_hashes():
    return historical.historical_hashes()


def context_for(slot, previous):
    if len(previous) != slot:
        raise ValueError("Context history does not match slot")
    source = source_record()
    schedule = anchor_schedule(source)
    anchor_index = schedule[slot]
    variant = variant_number(slot, schedule)
    recent = previous[-base.MAX_PRIOR_CONTEXT:]
    recent_text = "\n".join(f"- {text}" for text in recent) if recent else "(none)"
    return (
        "Write one sentence for a factual digest using only the frozen source.\n"
        f"SOURCE WINDOW:\n{source['source_window']}\n"
        f"ASSIGNED SOURCE ANCHOR {anchor_index + 1}/10:\n"
        f"{source['source_anchors'][anchor_index]}\n"
        f"DETAIL VARIANT FOR THIS ANCHOR: {variant}\n"
        f"RECENT ACCEPTED SENTENCES:\n{recent_text}\n"
        "The target AMR is mandatory. State one explicit fact or detail from "
        "the assigned anchor without inference. Copy names and numbers exactly. "
        "Use one complete simple declarative sentence of 5 to 12 words ending "
        "in one period. Use no comma, colon, semicolon, question mark, or "
        "exclamation mark. Keep this variant distinct from prior sentences."
    )


def token_jaccard(left, right):
    left_tokens = {token.casefold() for token in word_tokens(left)}
    right_tokens = {token.casefold() for token in word_tokens(right)}
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def surface_result_with_syntax(
    text,
    previous,
    forbidden_hashes,
    source,
    syntax,
):
    text = text.strip()
    slot = len(previous)
    schedule = anchor_schedule(source)
    anchor_index = schedule[slot]
    anchor = source["source_anchors"][anchor_index]
    spans = sentence_spans(text) if text else []
    candidate_info = syntax(text) if text else {
        "complete": False,
        "capitalization_valid_mid_sentence": False,
        "content_lemmas": [],
    }
    anchor_info = syntax(anchor)
    overlap = sorted(
        set(candidate_info["content_lemmas"])
        & set(anchor_info["content_lemmas"])
    )
    similarities = [token_jaccard(text, prior) for prior in previous]
    marks = punctuation(text)
    checks = {
        "nonempty": bool(text),
        "one_punkt_unit": len(spans) == 1
        and unit_texts(text, spans) == [text],
        "terminal_single_period": text.endswith(".")
        and marks.get(".", 0) == 1,
        "word_count_5_12": MIN_WORDS <= len(word_tokens(text)) <= MAX_WORDS,
        "complete_subject_finite_predicate": candidate_info["complete"] is True,
        "capitalized_clause": (
            candidate_info["capitalization_valid_mid_sentence"] is True
        ),
        "simple_punctuation": not any(mark in text for mark in ",;:!?"),
        "supported_punctuation": unsupported(text) is None,
        "no_second_person_or_command": not bool(COMMAND_RE.search(text)),
        "no_amr_or_placeholder_leak": not bool(LEAK_RE.search(text)),
        "at_least_two_anchor_content_lemmas": len(overlap) >= 2,
        "prior_token_jaccard_at_most_point8": all(
            value <= 0.8 for value in similarities
        ),
        "not_historical_strict_acceptance": (
            text_hash(text) not in forbidden_hashes
        ),
    }
    return {
        **checks,
        "pass": all(checks.values()),
        "word_count": len(word_tokens(text)),
        "source_anchor_index": anchor_index,
        "source_anchor_sha256": text_hash(anchor),
        "anchor_content_lemma_overlap": overlap,
        "prior_token_jaccard": similarities,
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


def review_prompt(label, document):
    source = source_record()
    return (
        f"Blind review pass {label}. Compare the source and passage below. "
        "Return exactly one JSON object with integer scores from 1 to 5 for "
        "coherence, naturalness, nonredundancy, task_consistency, and "
        "source_consistency, plus boolean severe_problem. Source consistency "
        "requires every passage claim to be explicitly supported by the source "
        "with no invented fact. Evaluate the passage as a concise factual "
        "digest; short sentences are acceptable. Use no markdown and no extra "
        f"keys.\n\nSOURCE:\n{source['source_window']}\n\n"
        f"PASSAGE:\n{document}"
    )


def review_pass(row):
    return (
        row is not None
        and not row["severe_problem"]
        and row["coherence"] >= 4
        and row["naturalness"] >= 3
        and row["nonredundancy"] >= 3
        and row["task_consistency"] >= 4
        and row["source_consistency"] >= 5
    )


def condition_record(text):
    spans = sentence_spans(text)
    return {
        "text": text,
        "text_sha256": text_hash(text),
        "word_tokens": word_tokens(text),
        "word_tokens_sha256": canonical_digest(word_tokens(text)),
        "character_length": len(text),
        "punctuation": dict(sorted(punctuation(text).items())),
        "segmentation_spans": spans,
        "num_detection_units": len(spans),
        "unit_texts": unit_texts(text, spans),
    }


def construct_conditions(text, syntax=None):
    syntax = syntax or syntax_model()
    c0 = condition_record(text)
    merges, splits = _candidates(text, c0["segmentation_spans"], syntax)
    expected_split = text.index(";")
    eligible_splits = [
        row for row in splits
        if row["position"] == expected_split
        and row["unit_index"] == 0
        and row["original_punctuation"] == ";"
    ]
    eligible_merges = [
        row for row in merges
        if row["unit_index"] >= 2
    ]
    if len(eligible_splits) != 1 or not eligible_merges:
        return {
            "constructed": False,
            "failure": "missing_fixed_split_or_nonadjacent_merge",
            "merge_candidate_count": len(merges),
            "split_candidate_count": len(splits),
            "eligible_merge_count": len(eligible_merges),
            "conditions": {"C0": c0},
            "detector_feedback": False,
        }
    selector = int(hashlib.sha256(
        f"{NAMESPACE}:{c0['text_sha256']}".encode()
    ).hexdigest(), 16)
    merge = sorted(
        eligible_merges,
        key=lambda row: (row["unit_index"], row["position"]),
    )[selector % len(eligible_merges)]
    split = eligible_splits[0]
    edits = sorted([
        {
            "position": merge["position"],
            "before": ".",
            "after": ";",
            "kind": "merge",
        },
        {
            "position": split["position"],
            "before": ";",
            "after": ".",
            "kind": "split",
        },
    ], key=lambda row: row["position"])
    c2 = {
        **condition_record(apply_edits(text, edits)),
        "punctuation_edits": edits,
    }
    restored = apply_edits(c2["text"], edits, reverse=True)
    r4 = condition_record(restored)
    valid = (
        c0["num_detection_units"] == c2["num_detection_units"] == 20
        and c0["word_tokens"] == c2["word_tokens"]
        and c0["punctuation"] == c2["punctuation"]
        and c0["character_length"] == c2["character_length"]
        and restored == text
        and r4 == c0
    )
    if not valid:
        raise GlobalStop("Frozen C2 count-neutral invariants failed")
    return {
        "constructed": True,
        "failure": None,
        "algorithm": "fixed_compound_hash_merge_v1",
        "namespace": NAMESPACE,
        "merge_candidate_count": len(merges),
        "split_candidate_count": len(splits),
        "eligible_merge_count": len(eligible_merges),
        "selected_split": split,
        "selected_merge": merge,
        "punctuation_edits": edits,
        "detector_feedback": False,
        "conditions": {"C0": c0, "C2": c2, "R4": r4},
    }


def document_from_slots(slots):
    if len(slots) != MAX_SLOTS or not all(row["accepted"] for row in slots):
        raise ValueError("The source-anchored document requires 21 acceptances")
    texts = [row["text"] for row in slots]
    if not texts[0].endswith("."):
        raise GlobalStop("Fixed compound left clause lacks terminal period")
    compound = texts[0][:-1] + "; " + texts[1]
    detection_units = [compound, *texts[2:]]
    text = " ".join(detection_units)
    words = len(word_tokens(text))
    spans = sentence_spans(text)
    if unit_texts(text, spans) != detection_units \
            or len(spans) != 20 \
            or not MIN_DOCUMENT_WORDS <= words <= MAX_DOCUMENT_WORDS:
        raise GlobalStop("Fixed C0 composition failed structural gates")
    construction = construct_conditions(text)
    return {
        "document_id": "source-anchored-splittable1-00",
        "text": text,
        "text_sha256": text_hash(text),
        "word_count": words,
        "segmentation_spans": spans,
        "generated_sentence_sha256": [row["text_sha256"] for row in slots],
        "source_anchor_schedule": list(anchor_schedule()),
        "fixed_compound_slots": [0, 1],
        "construction": construction,
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
        base.BANK,
        base.EXAMPLES,
        GLOVE,
        base.HISTORICAL_RESULT,
        *[path for path, _ in historical.PILOT_RESULTS],
        ROOT / "scripts/run_swan_coherent_injection_pilot.py",
        ROOT / "scripts/run_swan_normal_length_coherent_pilot.py",
        ROOT / "scripts/run_swan_injection_only_pilot.py",
        ROOT / "scripts/run_swan_positive_collection.py",
        ROOT / "scripts/swan_anchor1_sources.py",
        ROOT / "scripts/swan_cached_s2match.py",
        ROOT / "scripts/swan_feasibility2_common.py",
        ROOT / "scripts/swan_final1_budget.py",
        ROOT / "scripts/swan_final1_contract.py",
        ROOT / "scripts/swan_final1_quality.py",
        ROOT / "scripts/swan_final1_runtime.py",
        ROOT / "scripts/swan_final1_transform.py",
        ROOT / "scripts/test_swan_source_anchored_splittable.py",
    ]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def frozen_design():
    source = source_record()
    return {
        "source_sha256": sha256(SOURCE_PATH),
        "source_window_sha256": text_hash(source["source_window"]),
        "source_anchor_sha256": [
            text_hash(anchor) for anchor in source["source_anchors"]
        ],
        "source_anchor_schedule": list(anchor_schedule(source)),
        "extra_anchor_rule": "longest word count; first index wins ties",
        "fixed_compound_slots": [0, 1],
        "template_schedule": list(TEMPLATE_SCHEDULE),
        "target_selection_limitation": "all 21 slots use favorable target 47",
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
            "max_prior_token_jaccard": 0.8,
            "historical_exact_collision_forbidden": True,
        },
        "document_word_count": {
            "minimum": MIN_DOCUMENT_WORDS,
            "maximum": MAX_DOCUMENT_WORDS,
        },
        "construction": {
            "algorithm": "fixed_compound_hash_merge_v1",
            "hash_namespace": NAMESPACE,
            "detector_feedback": False,
            "required_detection_units": 20,
        },
        "auxiliary_reviews": {
            "count": REVIEW_COUNT,
            "same_backend_as_generation": True,
            "human_review": False,
            "minimum_scores": {
                "coherence": 4,
                "naturalness": 3,
                "nonredundancy": 3,
                "task_consistency": 4,
                "source_consistency": 5,
            },
        },
    }


def prepare():
    if (OUT / "manifest.json").exists():
        return check()
    check_source()
    if sha256(base.BANK) != base.EXPECTED_BANK:
        raise ValueError("Bank50 changed")
    if sha256(GLOVE) != EXPECTED_GLOVE:
        raise ValueError("GloVe vector file changed")
    historical_hashes()
    rotation_hash = sha256(base.ROTATION_ATTESTATION)
    freeze(OUT / "budget_estimate.json", base.budget())
    freeze(OUT / "environment_lock.json", environment_lock())
    freeze(OUT / "design.json", frozen_design())
    manifest = freeze(OUT / "manifest.json", {
        "schema_version": 1,
        "experiment": NAMESPACE,
        "frozen_at": now(),
        "status": "frozen_preexecution",
        "development_pilot": True,
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "source_sha256": sha256(SOURCE_PATH),
        "slots": MAX_SLOTS,
        "max_attempts_per_slot": MAX_ATTEMPTS,
        "match_threshold": base.MATCH_THRESHOLD,
        "minimum_document_words": MIN_DOCUMENT_WORDS,
        "maximum_document_words": MAX_DOCUMENT_WORDS,
        "required_detection_units": 20,
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
        "human_quality_claim": False,
        "empirical_calibration_claim": False,
        "detector_feedback_for_construction": False,
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
        and manifest["development_pilot"]
        and manifest["source_manifest_sha256"] == sha256(SOURCE_MANIFEST)
        and manifest["source_sha256"] == sha256(SOURCE_PATH)
        and manifest["slots"] == MAX_SLOTS
        and manifest["max_attempts_per_slot"] == MAX_ATTEMPTS
        and manifest["match_threshold"] == base.MATCH_THRESHOLD
        and manifest["minimum_document_words"] == MIN_DOCUMENT_WORDS
        and manifest["maximum_document_words"] == MAX_DOCUMENT_WORDS
        and manifest["required_detection_units"] == 20
        and not manifest["fallback_allowed"]
        and not manifest["credential_file_access"]
        and not manifest["final1_sealed_content_access"]
        and manifest["source_anchoring_protocol"]
        and not manifest["formal_source_anchoring_claim"]
        and not manifest["human_quality_claim"]
        and not manifest["empirical_calibration_claim"]
        and not manifest["detector_feedback_for_construction"]
        and manifest["new_output_observations_at_freeze"] == 0
    )
    if not valid:
        raise ValueError("Pilot identity or claim boundary changed")
    if sha256(base.BANK) != base.EXPECTED_BANK \
            or sha256(GLOVE) != EXPECTED_GLOVE \
            or manifest["budget_sha256"] != sha256(OUT / "budget_estimate.json") \
            or manifest["environment_sha256"] != sha256(OUT / "environment_lock.json") \
            or manifest["design_sha256"] != sha256(OUT / "design.json") \
            or manifest["rotation_attestation_sha256"] \
            != sha256(base.ROTATION_ATTESTATION):
        raise ValueError("Frozen pilot artifact changed")
    if load(OUT / "budget_estimate.json") != base.budget() \
            or load(OUT / "design.json") != frozen_design():
        raise ValueError("Frozen budget or design changed")
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


def result_from_slots(slots, reviews, journal):
    accepted = [row for row in slots if row["accepted"]]
    hashes = [row["text_sha256"] for row in accepted]
    if len(hashes) != len(set(hashes)):
        raise GlobalStop("Accepted slot texts are not unique")
    complete = len(accepted) == MAX_SLOTS
    document = document_from_slots(slots) if complete else None
    construction_pass = bool(
        document and document["construction"]["constructed"]
    )
    auxiliary_pass = (
        complete
        and construction_pass
        and len(reviews) == REVIEW_COUNT
        and all(row["pass"] for row in reviews)
    )
    if not complete:
        status = "slot_exhausted"
    elif not construction_pass:
        status = "standard_c2_construction_failed"
    elif not auxiliary_pass:
        status = "auxiliary_source_quality_gate_failed"
    else:
        status = "source_anchored_baseline_ready"
    schedule = anchor_schedule()
    result = {
        "experiment": NAMESPACE,
        "completed_at": now(),
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "source_sha256": sha256(SOURCE_PATH),
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
            "source_anchor_index": schedule[row["slot"]],
            "source_anchor_sha256": text_hash(
                source_record()["source_anchors"][schedule[row["slot"]]]
            ),
            "slot_sha256": sha256(base.slot_path(row["slot"])),
        } for row in accepted],
        "slot_sha256": {
            str(row["slot"]): sha256(base.slot_path(row["slot"]))
            for row in slots
        },
        "document": document,
        "standard_c2_constructed": construction_pass,
        "auxiliary_reviews": [{
            "label": row["label"],
            "path": str(base.review_path(row["label"]).relative_to(ROOT)),
            "sha256": sha256(base.review_path(row["label"])),
            "parsed": row["parsed"],
            "pass": row["pass"],
        } for row in reviews],
        "auxiliary_source_quality_gate_pass": auxiliary_pass,
        "human_quality_reviewed": False,
        "independent_empirical_calibration_pending": True,
        "accounting": journal.totals(),
        "source_anchored_baseline_ready": auxiliary_pass,
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


def audit():
    check()
    result = load(OUT / "result.json")
    source = source_record()
    failures = []
    schedule = anchor_schedule(source)
    for selected in result["selected"]:
        slot = selected["slot"]
        saved = load(base.slot_path(slot))
        trace_path = base.attempt_path(slot, saved["terminal_attempt"])
        trace = load(trace_path)
        valid = (
            saved["accepted"] is True
            and saved["fallback"] is False
            and saved["target_template_index"] == 47
            and saved["score"] >= base.MATCH_THRESHOLD
            and selected["score"] == saved["score"] == trace["score"]
            and selected["text"] == saved["text"] == trace["candidate"]
            and trace["status"] == "accepted"
            and trace["selected"] is True
            and trace["official_score_accepted"] is True
            and trace["surface"]["pass"] is True
            and trace["surface"]["source_anchor_index"] == schedule[slot]
            and len(trace["surface"]["anchor_content_lemma_overlap"]) >= 2
            and diagnostics_empty(trace)
            and saved["attempt_trace_sha256"][-1] == sha256(trace_path)
        )
        if not valid:
            failures.append({"slot": slot, "reason": "strict_trace_invalid"})
    document = result.get("document")
    construction = document.get("construction") if document else None
    passed = (
        result["fallback_slots"] == 0
        and result["accepted_slots"] == MAX_SLOTS
        and len(result["selected"]) == MAX_SLOTS
        and not failures
        and construction is not None
        and construction["constructed"]
        and result["auxiliary_source_quality_gate_pass"]
        and not result["human_quality_reviewed"]
    )
    return freeze(OUT / "audit.json", {
        "completed_at": now(),
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "result_sha256": sha256(OUT / "result.json"),
        "source_sha256": sha256(SOURCE_PATH),
        "strict_trace_failures": failures,
        "strict_trace_count": len(result["selected"]),
        "all_target_47": all(
            row["target_template_index"] == 47 for row in result["selected"]
        ),
        "all_scores_at_least_point75": all(
            row["score"] >= 0.75 for row in result["selected"]
        ),
        "fallback_slots": result["fallback_slots"],
        "standard_c2_constructed": bool(
            construction and construction["constructed"]
        ),
        "auxiliary_source_quality_gate_pass": (
            result["auxiliary_source_quality_gate_pass"]
        ),
        "human_quality_reviewed": False,
        "empirical_calibration_pending": True,
        "pass": passed,
    })


def detection_seed(*parts):
    return int(canonical_digest([NAMESPACE, *parts])[:8], 16)


def score_unit(module, cached, bank, condition, index, text):
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
                    float(value) for value in cached.score(
                        candidate,
                        graph,
                        detection_seed(
                            "detect", condition, text_hash(text), bank_index
                        ),
                    )
                ]
                if any(
                    not math.isfinite(value) or not 0 <= value <= 1
                    for value in (precision, recall, f1)
                ):
                    raise ValueError("Invalid Bank50 score")
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


def validate_equivalence(cached, bank):
    result = load(OUT / "result.json")
    first = result["selected"][0]
    saved = load(base.slot_path(first["slot"]))
    trace = load(base.attempt_path(first["slot"], saved["terminal_attempt"]))
    candidate = trace["candidate_template_amr"]
    seed = detection_seed("equivalence", 0, 47)
    with injection.capture_candidate_diagnostics() as diagnostics:
        official = tuple(float(value) for value in score_triple(
            candidate, bank[47], seed
        ))
        cached_value = tuple(float(value) for value in cached.score(
            candidate, bank[47], seed
        ))
    differences = [
        abs(left - right) for left, right in zip(official, cached_value)
    ]
    row = {
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "result_sha256": sha256(OUT / "result.json"),
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
    freeze(OUT / "equivalence_check.json", row)
    if not row["pass"]:
        raise RuntimeError("Cached S2Match equivalence check failed")
    return row


def detection_summary(condition, record, units):
    n = len(units)
    green = sum(row["green"] for row in units)
    z = (green - NULL_RATE * n) / math.sqrt(
        n * NULL_RATE * (1 - NULL_RATE)
    )
    return {
        "condition": condition,
        "text_sha256": record["text_sha256"],
        "word_tokens_sha256": record["word_tokens_sha256"],
        "punctuation": record["punctuation"],
        "N": n,
        "green": green,
        "green_rate": green / n,
        "z": z,
        "detected_at_1pct_theoretical": z > 2.33,
        "detected_at_5pct_theoretical": z > 1.645,
        "units": units,
    }


def detect_c0():
    check()
    result = load(OUT / "result.json")
    audit_row = load(OUT / "audit.json")
    if result["status"] != "source_anchored_baseline_ready" \
            or not audit_row["pass"]:
        raise PermissionError("Source and auxiliary quality gates did not pass")
    path = OUT / "detection_c0.json"
    if path.exists():
        return load(path)
    c0 = result["document"]["construction"]["conditions"]["C0"]
    bank = [row["normalized_template_amr"] for row in load(base.BANK)]
    with official_runtime(OUT, error_folder=OUT / "errors") as module:
        cached = CachedS2Matcher(GLOVE)
        equivalence = validate_equivalence(cached, bank)
        units = [
            score_unit(module, cached, bank, "C0", index, text)
            for index, text in enumerate(c0["unit_texts"])
        ]
    summary = detection_summary("C0", c0, units)
    return freeze(path, {
        "completed_at": now(),
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "result_sha256": sha256(OUT / "result.json"),
        "audit_sha256": sha256(OUT / "audit.json"),
        "equivalence_check_sha256": sha256(OUT / "equivalence_check.json"),
        "equivalence_pass": equivalence["pass"],
        **summary,
        "baseline_all_green": summary["green"] == summary["N"] == 20,
        "provider_calls_during_detection": 0,
        "human_quality_reviewed": False,
        "empirical_calibration_pending": True,
    })


def detect_attack():
    check()
    c0_detection = load(OUT / "detection_c0.json")
    if not c0_detection["baseline_all_green"]:
        raise PermissionError("C0 is not an all-green Bank50 baseline")
    path = OUT / "detection.json"
    if path.exists():
        return load(path)
    result = load(OUT / "result.json")
    construction = result["document"]["construction"]
    if not construction["constructed"]:
        raise PermissionError("Standard C2 was not constructed")
    c2 = construction["conditions"]["C2"]
    r4 = construction["conditions"]["R4"]
    bank = [row["normalized_template_amr"] for row in load(base.BANK)]
    with official_runtime(OUT, error_folder=OUT / "errors") as module:
        cached = CachedS2Matcher(GLOVE)
        c2_units = [
            score_unit(module, cached, bank, "C2", index, text)
            for index, text in enumerate(c2["unit_texts"])
        ]
    c2_summary = detection_summary("C2", c2, c2_units)
    r4_summary = {
        **{
            key: value for key, value in c0_detection.items()
            if key in (
                "text_sha256",
                "word_tokens_sha256",
                "punctuation",
                "N",
                "green",
                "green_rate",
                "z",
                "detected_at_1pct_theoretical",
                "detected_at_5pct_theoretical",
                "units",
            )
        },
        "condition": "R4",
        "exact_c0_reuse": True,
    }
    if r4_summary["text_sha256"] != r4["text_sha256"]:
        raise GlobalStop("R4 is not exact C0 restoration")
    return freeze(path, {
        "completed_at": now(),
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "result_sha256": sha256(OUT / "result.json"),
        "detection_c0_sha256": sha256(OUT / "detection_c0.json"),
        "standard_c2_constructed": True,
        "construction_algorithm": construction["algorithm"],
        "selected_split": construction["selected_split"],
        "selected_merge": construction["selected_merge"],
        "conditions": {
            "C0": {
                key: c0_detection[key]
                for key in (
                    "condition",
                    "text_sha256",
                    "word_tokens_sha256",
                    "punctuation",
                    "N",
                    "green",
                    "green_rate",
                    "z",
                    "detected_at_1pct_theoretical",
                    "detected_at_5pct_theoretical",
                    "units",
                )
            },
            "C2": c2_summary,
            "R4": r4_summary,
        },
        "paired_difference_C2_minus_C0": {
            "N": c2_summary["N"] - c0_detection["N"],
            "green": c2_summary["green"] - c0_detection["green"],
            "green_rate": (
                c2_summary["green_rate"] - c0_detection["green_rate"]
            ),
            "z": c2_summary["z"] - c0_detection["z"],
            "detection_transition_1pct": (
                f"{int(c0_detection['detected_at_1pct_theoretical'])}->"
                f"{int(c2_summary['detected_at_1pct_theoretical'])}"
            ),
        },
        "word_sequence_identical": (
            construction["conditions"]["C0"]["word_tokens"]
            == construction["conditions"]["C2"]["word_tokens"]
        ),
        "punctuation_multiset_identical": (
            construction["conditions"]["C0"]["punctuation"]
            == construction["conditions"]["C2"]["punctuation"]
        ),
        "exact_restoration": (
            construction["conditions"]["C0"]["text"]
            == construction["conditions"]["R4"]["text"]
        ),
        "provider_calls_during_detection": 0,
        "human_quality_reviewed": False,
        "empirical_calibration_pending": True,
        "population_attack_claim": False,
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
        "c0_detection_present": (OUT / "detection_c0.json").exists(),
        "attack_detection_present": (OUT / "detection.json").exists(),
    }
    for name in (
        "source_manifest",
        "manifest",
        "result",
        "audit",
        "detection_c0",
        "detection",
    ):
        path = OUT / f"{name}.json"
        if path.exists():
            row[f"{name}_sha256"] = sha256(path)
    if (OUT / "result.json").exists():
        result = load(OUT / "result.json")
        row["result_status"] = result["status"]
        row["accepted_slots"] = result["accepted_slots"]
        row["fallback_slots"] = result["fallback_slots"]
        row["auxiliary_source_quality_gate_pass"] = (
            result["auxiliary_source_quality_gate_pass"]
        )
    if (OUT / "detection.json").exists():
        detection = load(OUT / "detection.json")
        row["detection_summary"] = {
            condition: {
                key: value for key, value in metrics.items()
                if key in ("N", "green", "green_rate", "z")
            }
            for condition, metrics in detection["conditions"].items()
        }
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
    base.REVIEW_KEYS = REVIEW_KEYS
    base.MAX_CALLS = MAX_CALLS
    base.MAX_COST_NANODOLLARS = MAX_COST_NANODOLLARS
    base.PILOT_CAP = PILOT_CAP
    base.digest = pilot_digest
    base.historical_hashes = historical_hashes
    base.context_for = context_for
    base.surface_result = surface_result
    base.review_prompt = review_prompt
    base.review_pass = review_pass
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
    parser.add_argument("command", choices=(
        "allocate-source",
        "prepare",
        "check",
        "authorize",
        "record-price",
        "run",
        "audit",
        "detect-c0",
        "detect-attack",
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
            result = base.run(args.max_new_calls)
        elif args.command == "audit":
            result = audit()
        elif args.command == "detect-c0":
            result = detect_c0()
        elif args.command == "detect-attack":
            result = detect_attack()
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
