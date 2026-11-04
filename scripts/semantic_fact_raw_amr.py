"""Integrity checks for literal raw-AMR records; no semantic facts inferred here."""
from __future__ import annotations

import hashlib


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_inputs(data):
    if data.get("intended_symbols_included") is not False or data.get("reference_facts_included") is not False:
        raise ValueError("Parser input must exclude target and reference labels")
    rows = data["rows"]
    if len(rows) != 192 or len({r["case_id"] for r in rows}) != 192:
        raise ValueError("Exactly 192 distinct case ids required")
    for row in rows:
        if set(row) != {"case_id", "input_text", "input_text_sha256", "window"}:
            raise ValueError("Unexpected parser-input field")
        if text_sha(row["input_text"]) != row["input_text_sha256"]:
            raise ValueError("Literal input SHA mismatch")
        if not row["input_text"].strip():
            raise ValueError("Empty parser input")
    return rows


def audit_raw(record, row, freeze_sha):
    if record["case_id"] != row["case_id"] or record["text_sha256"] != row["input_text_sha256"]:
        raise ValueError("Parser result identity mismatch")
    if record["freeze_sha256"] != freeze_sha:
        raise ValueError("Parser result freeze mismatch")
    if record["amr_sha256"] != (text_sha(record["amr"]) if record["amr"] else None):
        raise ValueError("Raw AMR SHA mismatch")
    accepted = [a for a in record["attempts"] if a["accepted"]]
    selected = accepted[0]["beam_index"] if len(accepted) == 1 else None
    reasons = []
    if not record["amr"]:
        reasons.append("no_raw_output")
    if not record["valid"]:
        reasons.append("invalid_roundtrip_or_parser_output")
    if selected != 0:
        reasons.append("nonzero_or_missing_selected_beam")
    if record["events"] or record["decode_events"]:
        reasons.append("bound_parser_or_roundtrip_diagnostic")
    if record["clipped"]:
        reasons.append("native_output_limit")
    return {
        "raw_penman_present": bool(record["amr"]),
        "selected_beam": selected,
        "structurally_eligible": not reasons,
        "exclusion_reasons": reasons,
        "parse_status": "VALID_AUDITED_PENMAN" if not reasons else "EXCLUDED_RAW_OUTPUT",
    }


def replay_signature(record):
    """Exclude elapsed time and CUDA memory measurements from determinism checks."""
    return {k: record[k] for k in (
        "case_id", "text_sha256", "amr", "amr_sha256", "valid", "events",
        "attempts", "decode_events", "beam_lengths", "clipped", "token_length",
        "freeze_sha256")}
