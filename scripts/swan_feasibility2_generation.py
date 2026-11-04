"""Ten-slot official SWAN generation for the normal-length feasibility pilot."""
from __future__ import annotations

from pathlib import Path

from swan_anchor1_sources import identity
from swan_feasibility2_budget import DocumentFailure, GlobalStop
from swan_feasibility2_common import digest, freeze, load, now, sha256
from swan_final1_transform import document_structure, sentence_spans, unit_texts, word_tokens

SLOTS = 10
MAX_ATTEMPTS = 50


def stage_folder(folder):
    return Path(folder) / "private" / "pilot"


def slot_path(folder, allocation, slot):
    return stage_folder(folder) / "generation_slots" / (
        f"{allocation}-slot-{slot:02d}.json"
    )


def document_path(folder, allocation):
    return Path(folder) / "documents" / "pilot" / f"{allocation}.json"


def topic_context(source, prior):
    recent = prior[-3:]
    continuation = " ".join(recent) if recent else "(opening sentence)"
    return (
        "Write one sentence for a cohesive, neutral news-style passage. "
        "The target AMR is the primary semantic constraint. Use the source sentence "
        "only as broad topical inspiration; do not force incompatible source facts. "
        "Use 10 to 22 words, exactly one complete declarative sentence, and no lists. "
        f"SOURCE TOPIC: {source['source_anchors'][0]} "
        f"RECENT PASSAGE: {continuation}"
    )


def load_slots(folder, allocation):
    private = stage_folder(folder)
    rows = []
    for slot in range(SLOTS):
        start = private / "slot_starts" / f"{allocation}-slot-{slot:02d}.json"
        path = slot_path(folder, allocation, slot)
        if not start.exists():
            if path.exists():
                raise GlobalStop("Slot exists without a durable start")
            break
        if not path.exists():
            return rows, "interrupted_slot_no_retry"
        row = load(path)
        if row["allocation_id"] != allocation or row["slot"] != slot:
            raise GlobalStop("Slot sequence changed")
        for attempt, expected in enumerate(row["attempt_trace_sha256"]):
            trace = private / "generation_attempts" / (
                f"{allocation}-slot-{slot:02d}-attempt-{attempt:02d}.json"
            )
            if sha256(trace) != expected:
                raise GlobalStop("Attempt trace changed")
        rows.append(row)
    return rows, None


def finish_document(folder, source, slots, reason, historical, seen):
    texts = [row["text"] for row in slots]
    structure = document_structure(texts)
    normalized = identity(structure["text"])
    independent = normalized not in historical and normalized not in seen
    selected_truncated = False
    for row in slots:
        trace = load(
            stage_folder(folder) / "generation_attempts" /
            f"{source['allocation_id']}-slot-{row['slot']:02d}-"
            f"attempt-{row['attempts'] - 1:02d}.json"
        )
        selected_truncated |= trace["stop_reason"] == "length"
    exact_slots = reason == "ten_slots_complete" and len(slots) == SLOTS
    slot_surface = [
        len(sentence_spans(text)) == 1
        and unit_texts(text, sentence_spans(text)) == [text]
        and 10 <= len(word_tokens(text)) <= 22
        for text in texts
    ]
    accepted = sum(bool(row["accepted"]) for row in slots)
    result = {
        "allocation_id": source["allocation_id"],
        "phase": "pilot",
        "ordinal": source["ordinal"],
        "source_locator": source["locator"],
        "domain": source["domain"],
        "source_excerpt": source["source_window"],
        "source_sha256": digest(source),
        "manifest_sha256": sha256(Path(folder) / "manifest.json"),
        "finished_at": now(),
        "termination": reason,
        "generation_complete": exact_slots,
        **structure,
        "exact_ten_slots": exact_slots,
        "slot_surface_pass": slot_surface,
        "all_slots_surface_pass": len(slot_surface) == SLOTS and all(slot_surface),
        "accepted_slots": accepted,
        "fallback_slots": len(slots) - accepted,
        "slot_acceptance_rate": accepted / SLOTS,
        "all_slots_s2match_accepted": accepted == SLOTS,
        "selected_provider_truncation": selected_truncated,
        "normalized_text_sha256": normalized,
        "identity_pass": independent,
        "prose_structure_pass": exact_slots
        and structure["structural_pass"]
        and all(slot_surface)
        and independent
        and not selected_truncated,
        "slot_sha256": [
            sha256(slot_path(folder, source["allocation_id"], row["slot"]))
            for row in slots
        ],
    }
    freeze(document_path(folder, source["allocation_id"]), result)
    if structure["text"]:
        seen.add(normalized)
    return result


def run_documents(folder, sources, journal, module, run_slot, bank, examples_path):
    folder = Path(folder)
    historical = set(
        load(folder / "prior_identity_snapshot.json")["normalized_text_sha256"]
    )
    seen = set()
    counts = {
        "terminal": 0,
        "prose_structure_pass": 0,
        "accepted_slots": 0,
        "completed_slots": 0,
    }
    for source in sources:
        allocation = source["allocation_id"]
        path = document_path(folder, allocation)
        slots, interrupted = load_slots(folder, allocation)
        if path.exists():
            row = load(path)
            if row["source_sha256"] != digest(source):
                raise GlobalStop("Saved document source changed")
            counts["terminal"] += 1
            counts["prose_structure_pass"] += row["prose_structure_pass"]
            counts["accepted_slots"] += row["accepted_slots"]
            counts["completed_slots"] += len(slots)
            if row["text"]:
                seen.add(row["normalized_text_sha256"])
            continue
        reason = interrupted
        while reason is None and len(slots) < SLOTS:
            capacity = journal.capacity(MAX_ATTEMPTS)
            if capacity == "invocation_cap":
                return {**counts, "state": "paused_before_slot", **journal.totals()}
            if capacity:
                reason = capacity
                break
            slot = len(slots)
            context = topic_context(source, [row["text"] for row in slots])
            start = stage_folder(folder) / "slot_starts" / (
                f"{allocation}-slot-{slot:02d}.json"
            )
            freeze(start, {
                "allocation_id": allocation,
                "slot": slot,
                "started_at": now(),
                "context_sha256": digest(context),
                "manifest_sha256": journal.contract_hash,
            })
            try:
                row = run_slot(
                    module, stage_folder(folder), source, slot, context,
                    bank, examples_path, journal,
                )
                if not row["text"]:
                    raise DocumentFailure("empty_official_output")
                slots.append(row)
            except DocumentFailure as exc:
                reason = str(exc)
            except GlobalStop:
                raise
            except Exception as exc:
                journal.event("halt", reason=f"unexpected_slot_error:{type(exc).__name__}")
                raise GlobalStop("Unexpected slot failure; continuation prohibited") from exc
        if reason is None:
            reason = "ten_slots_complete"
        row = finish_document(folder, source, slots, reason, historical, seen)
        counts["terminal"] += 1
        counts["prose_structure_pass"] += row["prose_structure_pass"]
        counts["accepted_slots"] += row["accepted_slots"]
        counts["completed_slots"] += len(slots)
    return {**counts, "state": "phase_terminal", **journal.totals()}


def verify_documents(folder, sources):
    folder = Path(folder)
    historical = set(
        load(folder / "prior_identity_snapshot.json")["normalized_text_sha256"]
    )
    seen, documents = set(), []
    for source in sources:
        path = document_path(folder, source["allocation_id"])
        if not path.is_file():
            raise ValueError("Pilot still has pending allocations")
        saved = load(path)
        slots, interrupted = load_slots(folder, source["allocation_id"])
        if saved["source_sha256"] != digest(source) \
                or saved["manifest_sha256"] != sha256(folder / "manifest.json"):
            raise GlobalStop("Document source or contract changed")
        if saved["text"] != " ".join(row["text"] for row in slots):
            raise GlobalStop("Document text differs from frozen slots")
        if saved["accepted_slots"] != sum(row["accepted"] for row in slots) \
                or saved["fallback_slots"] != sum(row["fallback"] for row in slots):
            raise GlobalStop("Slot acceptance summary changed")
        expected_reason = interrupted or (
            "ten_slots_complete" if len(slots) == SLOTS else saved["termination"]
        )
        if saved["termination"] != expected_reason:
            raise GlobalStop("Document termination changed")
        normalized = identity(saved["text"])
        independent = normalized not in historical and normalized not in seen
        if saved["normalized_text_sha256"] != normalized \
                or saved["identity_pass"] != independent:
            raise GlobalStop("Document identity changed")
        if saved["text"]:
            seen.add(normalized)
        documents.append(saved)
    return documents
