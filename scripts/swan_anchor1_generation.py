"""Source-anchored SWAN generation with a detector-blind surface gate."""
from __future__ import annotations

import math
from pathlib import Path
import random
import re

import run_swan_positive_collection as swan
from swan_anchor1_budget import DocumentFailure, GlobalStop, Journal, body_for
from swan_anchor1_common import digest, freeze, load, now, seed, sha256
from swan_anchor1_sources import identity
from swan_final1_transform import (
    LEAK_RE,
    document_structure,
    sentence_spans,
    unit_texts,
    word_tokens,
)

SLOTS = 10
TEMPLATES_PER_SLOT = 10
ATTEMPTS_PER_TEMPLATE = 3
MAX_ATTEMPTS = TEMPLATES_PER_SLOT * ATTEMPTS_PER_TEMPLATE
MATCH_THRESHOLD = .75
COMMAND_RE = re.compile(
    r"\b(?:you|your|yours|yourself|yourselves|please|must|should|"
    r"ensure|remember)\b|(?:^|\s)(?:do not|don't|make sure|let us|let's)\b",
    re.I,
)


def source_context(source, slot, generated):
    recent = generated[-2:]
    recent_text = "\n".join(f"- {text}" for text in recent) if recent else "(none)"
    return (
        "Use the following source only as factual grounding.\n"
        f"SOURCE WINDOW:\n{source['source_window']}\n"
        f"CURRENT SOURCE ANCHOR {slot + 1}/10:\n{source['source_anchors'][slot]}\n"
        f"RECENT GENERATED SENTENCES:\n{recent_text}\n"
        "Write one factual declarative sentence grounded in the current source anchor. "
        "Use 10 to 22 words, include an explicit subject and finite predicate, avoid "
        "second-person or imperative language, and do not repeat a prior sentence."
    )


def token_jaccard(left, right):
    a = {token.casefold() for token in word_tokens(left)}
    b = {token.casefold() for token in word_tokens(right)}
    return len(a & b) / len(a | b) if a or b else 1.0


def surface_screen(candidate, anchor, previous, syntax):
    text = candidate.strip()
    spans = sentence_spans(text) if text else []
    candidate_info = syntax(text) if text else {
        "complete": False, "content_lemmas": [], "capitalization_valid_mid_sentence": False
    }
    anchor_info = syntax(anchor)
    anchor_lemmas = set(anchor_info["content_lemmas"])
    candidate_lemmas = set(candidate_info["content_lemmas"])
    overlap = sorted(anchor_lemmas & candidate_lemmas)
    similarities = [token_jaccard(text, prior) for prior in previous]
    checks = {
        "nonempty": bool(text),
        "one_sentence": len(spans) == 1 and unit_texts(text, spans) == [text],
        "terminal_punctuation": bool(re.search(r'[.!?]["\')\]]*$', text)),
        "word_count_10_22": 10 <= len(word_tokens(text)) <= 22,
        "complete_subject_finite_predicate": candidate_info["complete"] is True,
        "no_second_person_or_command": not COMMAND_RE.search(text),
        "no_amr_or_placeholder_leak": not LEAK_RE.search(text),
        "anchor_content_lemma_overlap": bool(overlap),
        "prior_token_jaccard_at_most_point8": all(value <= .8 for value in similarities),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "word_count": len(word_tokens(text)),
        "anchor_content_lemma_overlap": overlap,
        "prior_token_jaccard": similarities,
        "anchor_syntax": anchor_info,
        "candidate_syntax": candidate_info,
    }


def template_schedule(bank, allocation_id, slot):
    rng = random.Random(seed("template-schedule", allocation_id, slot))
    return [rng.randrange(len(bank)) for _ in range(TEMPLATES_PER_SLOT)]


def stage_folder(folder, phase):
    return Path(folder) / "private" / phase


def slot_path(folder, phase, allocation, slot):
    return stage_folder(folder, phase) / "generation_slots" / (
        f"{allocation}-slot-{slot:02d}.json"
    )


def doc_path(folder, phase, allocation):
    return Path(folder) / "documents" / phase / f"{allocation}.json"


def run_slot(folder, phase, source, slot, generated, journal, module, bank, examples, syntax):
    folder = Path(folder)
    private = stage_folder(folder, phase)
    allocation = source["allocation_id"]
    context = source_context(source, slot, generated)
    schedule = template_schedule(bank, allocation, slot)
    identity_fields = {
        "allocation_id": allocation,
        "slot": slot,
        "source_sha256": digest(source),
        "context_sha256": digest(context),
        "manifest_sha256": sha256(folder / "manifest.json"),
        "source_manifest_sha256": sha256(folder / "source_manifest.json"),
        "template_indices": schedule,
    }
    path = slot_path(folder, phase, allocation, slot)
    start_path = private / "slot_starts" / f"{allocation}-slot-{slot:02d}.json"
    if path.exists():
        saved = load(path)
        if any(saved.get(key) != value for key, value in identity_fields.items()):
            raise GlobalStop("Frozen anchor1 slot identity changed")
        return saved
    if start_path.exists():
        raise DocumentFailure("interrupted_slot_no_retry")
    freeze(start_path, {**identity_fields, "started_at": now()})

    traces = []
    selected = None
    system_prompt = module.build_system_prompt()
    for attempt in range(MAX_ATTEMPTS):
        template_index = schedule[attempt // ATTEMPTS_PER_TEMPLATE]
        template = bank[template_index]
        user_prompt = module.build_user_prompt(template, examples, context=context)
        call_id = f"{allocation}-slot-{slot:02d}-attempt-{attempt:02d}"
        trace = {
            **identity_fields,
            "attempt": attempt,
            "call_id": call_id,
            "template_index": template_index,
            "template_sha256": swan.text_sha256(template),
            "user_prompt_sha256": swan.text_sha256(user_prompt),
            "system_prompt_sha256": swan.text_sha256(system_prompt),
            "status": "requesting",
        }
        trace_path = private / "generation_attempts" / f"{call_id}.json"
        try:
            response = journal.request(call_id, user_prompt, system_prompt, 50, .7, .9)
        except Exception as exc:
            trace.update(status="transport_failed_or_uncertain",
                         error_type=type(exc).__name__)
            freeze(trace_path, trace)
            raise
        candidate = response["text"].strip()
        trace.update(
            status="surface_pending",
            candidate=candidate,
            candidate_sha256=swan.text_sha256(candidate),
            call_sha256=response["call_sha256"],
            stop_reason=response["stop_reason"],
            usage=response["usage"],
            response_metadata=response["response_metadata"],
        )
        screen = surface_screen(candidate, source["source_anchors"][slot], generated, syntax)
        if response["stop_reason"] == "length":
            screen["checks"]["provider_not_truncated"] = False
            screen["pass"] = False
        else:
            screen["checks"]["provider_not_truncated"] = True
        trace["surface"] = screen
        if screen["pass"]:
            try:
                raw = module.parse_amr(candidate)
                normalized = module.normalize_amr_variables(raw)
                templated = module.generate_template_amr(normalized)
                candidate_template = module.normalize_amr_variables(templated)
                score_seed = seed("injection-score", allocation, slot, attempt)
                with swan.fixed_score_seed(score_seed):
                    accepted, score = module.decide_s2match_match(
                        candidate_template, template, threshold=MATCH_THRESHOLD
                    )
                score = float(score)
                if not math.isfinite(score) or not 0 <= score <= 1 \
                        or bool(accepted) != (score >= MATCH_THRESHOLD):
                    raise GlobalStop("Invalid S2Match decision")
                trace.update(
                    raw_amr=raw,
                    raw_amr_sha256=swan.text_sha256(raw),
                    normalized_amr=normalized,
                    normalized_amr_sha256=swan.text_sha256(normalized),
                    candidate_template_before_final_normalization=templated,
                    candidate_template_amr=candidate_template,
                    candidate_template_amr_sha256=swan.text_sha256(candidate_template),
                    score_seed=score_seed,
                    threshold=MATCH_THRESHOLD,
                    score=score,
                    accepted=bool(accepted),
                    status="complete",
                )
            except GlobalStop:
                raise
            except Exception as exc:
                trace.update(status="local_inference_failed",
                             error_type=type(exc).__name__)
                freeze(trace_path, trace)
                raise DocumentFailure("local_parser_or_s2match_failed") from exc
        else:
            trace.update(score=None, accepted=False, status="surface_rejected")
        freeze(trace_path, trace)
        traces.append(trace)
        if trace["accepted"]:
            selected = trace
            break

    if selected is None:
        eligible = [trace for trace in traces if trace["surface"]["pass"]]
        if eligible:
            selected = min(eligible, key=lambda row: (-row["score"], row["attempt"]))
    if selected is None:
        failure = {
            **identity_fields,
            "status": "no_surface_eligible_candidate",
            "reason": "no_surface_eligible_candidate",
            "attempts": len(traces),
            "attempt_trace_sha256": [sha256(
                private / "generation_attempts" / f"{trace['call_id']}.json"
            ) for trace in traces],
        }
        freeze(
            private / "generation_failures" / f"{allocation}-slot-{slot:02d}.json",
            failure,
        )
        raise DocumentFailure("no_surface_eligible_candidate")

    result = {
        **identity_fields,
        "status": "accepted" if selected["accepted"] else "fallback",
        "attempts": len(traces),
        "selected_attempt": selected["attempt"],
        "attempt_trace_sha256": [sha256(
            private / "generation_attempts" / f"{trace['call_id']}.json"
        ) for trace in traces],
        "source_anchor": source["source_anchors"][slot],
        "source_anchor_sha256": swan.text_sha256(source["source_anchors"][slot]),
        "text": selected["candidate"],
        "text_sha256": selected["candidate_sha256"],
        "chosen_template_index": selected["template_index"],
        "chosen_template": bank[selected["template_index"]],
        "chosen_template_sha256": selected["template_sha256"],
        "score": selected["score"],
        "accepted": selected["accepted"],
        "fallback": not selected["accepted"],
        "surface": selected["surface"],
    }
    return freeze(path, result)


def load_slots(folder, phase, allocation):
    private = stage_folder(folder, phase)
    slots = []
    for slot in range(SLOTS):
        start = private / "slot_starts" / f"{allocation}-slot-{slot:02d}.json"
        path = slot_path(folder, phase, allocation, slot)
        if not start.exists():
            if path.exists():
                raise GlobalStop("Slot artifact exists without durable start")
            break
        if not path.exists():
            failure = private / "generation_failures" / (
                f"{allocation}-slot-{slot:02d}.json"
            )
            return slots, (
                load(failure).get("reason", "interrupted_slot_no_retry")
                if failure.exists() else "interrupted_slot_no_retry"
            )
        value = load(path)
        if value["allocation_id"] != allocation or value["slot"] != slot:
            raise GlobalStop("Slot sequence changed")
        for attempt, expected in enumerate(value["attempt_trace_sha256"]):
            trace = private / "generation_attempts" / (
                f"{allocation}-slot-{slot:02d}-attempt-{attempt:02d}.json"
            )
            if sha256(trace) != expected:
                raise GlobalStop("Attempt trace changed")
        slots.append(value)
    return slots, None


def finish_document(folder, phase, source, slots, reason, historical, seen):
    structure = document_structure([slot["text"] for slot in slots])
    normalized = identity(structure["text"])
    independent = normalized not in historical and normalized not in seen
    complete = reason == "ten_slots_complete"
    result = {
        "allocation_id": source["allocation_id"],
        "phase": phase,
        "ordinal": source["ordinal"],
        "source_locator": source["locator"],
        "domain": source["domain"],
        "source_excerpt": source["source_window"],
        "source_sha256": digest(source),
        "manifest_sha256": sha256(Path(folder) / "manifest.json"),
        "finished_at": now(),
        "termination": reason,
        "generation_complete": complete,
        **structure,
        "exact_ten_units": len(slots) == SLOTS
        and structure["num_detection_units"] == SLOTS,
        "all_selected_surface_pass": bool(slots)
        and all(slot["surface"]["pass"] for slot in slots),
        "normalized_text_sha256": normalized,
        "identity_pass": independent,
        "structural_pass": complete
        and structure["structural_pass"]
        and len(slots) == SLOTS
        and structure["num_detection_units"] == SLOTS
        and all(slot["surface"]["pass"] for slot in slots)
        and independent,
        "slot_sha256": [sha256(slot_path(
            folder, phase, source["allocation_id"], slot["slot"]
        )) for slot in slots],
    }
    freeze(doc_path(folder, phase, source["allocation_id"]), result)
    if structure["text"]:
        seen.add(normalized)
    return result


def run_documents(folder, phase, sources, journal, module, bank, examples, syntax):
    folder = Path(folder)
    historical = set(load(folder / "prior_identity_snapshot.json")["normalized_text_sha256"])
    seen = set()
    if phase == "main":
        for path in sorted((folder / "documents/pilot").glob("*.json")):
            row = load(path)
            if row["text"]:
                seen.add(row["normalized_text_sha256"])
    counts = {"terminal": 0, "structural_pass": 0}
    for source in sources:
        allocation = source["allocation_id"]
        document_path = doc_path(folder, phase, allocation)
        slots, interrupted = load_slots(folder, phase, allocation)
        if document_path.exists():
            row = load(document_path)
            counts["terminal"] += 1
            counts["structural_pass"] += row["structural_pass"]
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
            try:
                slot = run_slot(
                    folder, phase, source, len(slots), [row["text"] for row in slots],
                    journal, module, bank, examples, syntax,
                )
                slots.append(slot)
            except DocumentFailure as exc:
                reason = str(exc)
                failure = stage_folder(folder, phase) / "generation_failures" / (
                    f"{allocation}-slot-{len(slots):02d}.json"
                )
                if not failure.exists():
                    freeze(failure, {
                        "allocation_id": allocation,
                        "slot": len(slots),
                        "source_sha256": digest(source),
                        "manifest_sha256": sha256(folder / "manifest.json"),
                        "reason": reason,
                        "status": "terminal_document_failure",
                    })
            except GlobalStop:
                raise
            except Exception as exc:
                journal.event("halt", reason=f"unexpected_slot_error:{type(exc).__name__}")
                raise GlobalStop("Unexpected slot failure; continuation prohibited") from exc
        if reason is None and len(slots) == SLOTS:
            reason = "ten_slots_complete"
        row = finish_document(folder, phase, source, slots, reason, historical, seen)
        counts["terminal"] += 1
        counts["structural_pass"] += row["structural_pass"]
    return {**counts, "state": "phase_terminal", **journal.totals()}


def verify_generation(folder, phase, sources):
    """Replay source/context/trace selection without parser or scorer calls."""
    folder = Path(folder)
    contract_hash = sha256(folder / "manifest.json")
    journal = Journal(folder, phase, 1, None, contract_hash)
    from swan_anchor1_contract import BANK
    bank = [row["normalized_template_amr"] for row in load(BANK)]
    examples = load(BANK.parent.parent / "artifacts/normalized_amr_examples.json")[:3]
    official = swan.official_function_module_for_test({})
    historical = set(load(folder / "prior_identity_snapshot.json")["normalized_text_sha256"])
    if phase == "main":
        historical.update(
            load(path)["normalized_text_sha256"]
            for path in sorted((folder / "documents/pilot").glob("*.json"))
            if load(path)["text"]
        )
    documents = []
    for source in sources:
        allocation = source["allocation_id"]
        path = doc_path(folder, phase, allocation)
        if not path.exists():
            raise ValueError(f"{phase} still has pending allocations")
        saved = load(path)
        slots, interrupted = load_slots(folder, phase, allocation)
        if saved["source_sha256"] != digest(source) \
                or saved["manifest_sha256"] != contract_hash:
            raise GlobalStop("Document source or contract changed")
        for slot, row in enumerate(slots):
            context = source_context(source, slot, [value["text"] for value in slots[:slot]])
            schedule = template_schedule(
                [None] * 50, allocation, slot
            )
            if row["context_sha256"] != digest(context) \
                    or row["template_indices"] != schedule \
                    or row["source_anchor"] != source["source_anchors"][slot] \
                    or not row["surface"]["pass"]:
                raise GlobalStop("Slot context, anchor, schedule or surface decision changed")
            traces = [
                load(stage_folder(folder, phase) / "generation_attempts" /
                     f"{allocation}-slot-{slot:02d}-attempt-{attempt:02d}.json")
                for attempt in range(row["attempts"])
            ]
            if not 1 <= len(traces) <= MAX_ATTEMPTS:
                raise GlobalStop("Attempt count is outside the frozen slot budget")
            system_prompt = official.build_system_prompt()
            for trace in traces:
                template = bank[trace["template_index"]]
                user_prompt = official.build_user_prompt(
                    template, examples, context=context
                )
                request = body_for(user_prompt, system_prompt)
                call = journal.calls.get(trace["call_id"])
                if not call or not call["terminal"] \
                        or call["terminal"]["kind"] != "complete" \
                        or trace["call_sha256"] != call["terminal"]["response_sha256"] \
                        or call["reserve"]["request"] != request \
                        or trace["template_index"] != \
                        schedule[trace["attempt"] // ATTEMPTS_PER_TEMPLATE] \
                        or trace["template_sha256"] != swan.text_sha256(template):
                    raise GlobalStop("Attempt lacks its frozen request/template provenance")
                response = journal._parse(
                    load(journal.folder / "responses" / f"{trace['call_id']}.json"),
                    call["reserve"]["reserve"],
                )
                if trace["candidate"] != response["text"] \
                        or trace["usage"] != response["usage"] \
                        or trace["stop_reason"] != response["stop_reason"]:
                    raise GlobalStop("Attempt differs from the provider response")
                if trace["surface"]["pass"]:
                    if trace["threshold"] != MATCH_THRESHOLD \
                            or not isinstance(trace["score"], (int, float)) \
                            or not 0 <= trace["score"] <= 1 \
                            or trace["accepted"] != \
                            (trace["score"] >= MATCH_THRESHOLD):
                        raise GlobalStop("S2Match decision changed")
                elif trace["score"] is not None or trace["accepted"]:
                    raise GlobalStop("Surface-rejected candidate was scored or accepted")
            eligible = [trace for trace in traces if trace["surface"]["pass"]]
            accepted = [trace for trace in traces if trace["accepted"]]
            if accepted and accepted[0] is not traces[-1]:
                raise GlobalStop("Slot continued after its first accepted candidate")
            expected = accepted[0] if accepted else (
                min(eligible, key=lambda trace: (-trace["score"], trace["attempt"]))
                if eligible else None
            )
            if expected is None or row["selected_attempt"] != expected["attempt"] \
                    or row["text"] != expected["candidate"] \
                    or row["text_sha256"] != expected["candidate_sha256"] \
                    or row["surface"] != expected["surface"] \
                    or row["accepted"] != expected["accepted"] \
                    or row["fallback"] == row["accepted"]:
                raise GlobalStop("Accepted/fallback selection changed")
        expected_reason = interrupted or (
            "ten_slots_complete" if len(slots) == SLOTS else saved["termination"]
        )
        if saved["termination"] != expected_reason \
                or saved["text"] != " ".join(slot["text"] for slot in slots):
            raise GlobalStop("Document termination or text changed")
        structure = document_structure([slot["text"] for slot in slots])
        expected_pass = (
            expected_reason == "ten_slots_complete"
            and structure["structural_pass"]
            and len(slots) == SLOTS
            and structure["num_detection_units"] == SLOTS
            and all(slot["surface"]["pass"] for slot in slots)
        )
        normalized = identity(saved["text"])
        independent = normalized not in historical
        if saved["normalized_text_sha256"] != normalized \
                or saved["identity_pass"] != independent \
                or saved["structural_pass"] != (expected_pass and independent):
            raise GlobalStop("Document structural decision changed")
        if saved["text"]:
            historical.add(normalized)
        documents.append(saved)
    return documents
