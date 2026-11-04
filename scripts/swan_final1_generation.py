"""Final1 document state machine; complete slots are reusable, partial slots are not."""
from __future__ import annotations

from pathlib import Path
import random

from swan_final1_budget import DocumentFailure, GlobalStop, Journal, body_for
from swan_final1_common import digest, freeze, load, now, seed, sha256
from swan_final1_sources import identity
from swan_final1_transform import document_structure, stop_document, text_hash


def stage_folder(folder, phase):
    return Path(folder) / "private" / phase


def doc_path(folder, phase, allocation):
    return Path(folder) / "documents" / phase / f"{allocation}.json"


def load_slots(folder, phase, allocation):
    private = stage_folder(folder, phase)
    slots = []
    for slot in range(24):
        start = private / "slot_starts" / f"{allocation}-slot-{slot:02d}.json"
        path = private / "generation_slots" / f"{allocation}-slot-{slot:02d}.json"
        if not start.exists():
            if path.exists():
                raise GlobalStop("Slot artifact exists without durable start")
            break
        if not path.exists():
            return slots, "interrupted_slot_no_retry"
        value = load(path)
        if value["allocation_id"] != allocation or value["slot"] != slot:
            raise GlobalStop("Slot identity changed")
        for attempt, expected in enumerate(value["attempt_trace_sha256"]):
            trace = private / "generation_attempts" / f"{allocation}-slot-{slot:02d}-attempt-{attempt:02d}.json"
            if sha256(trace) != expected:
                raise GlobalStop("Injection attempt provenance changed")
        slots.append(value)
        if stop_document([r["text"] for r in slots]):
            break
    return slots, None


def verify_slot_contexts(folder, phase, source, slots, interrupted):
    private = stage_folder(folder, phase)
    contract = sha256(Path(folder) / "manifest.json")
    source_manifest = sha256(Path(folder) / "source_manifest.json")
    for index in range(len(slots) + bool(interrupted)):
        context = " ".join([source["generation_prompt"]] + [r["text"] for r in slots[:index]])
        start = load(private / "slot_starts" / f"{source['allocation_id']}-slot-{index:02d}.json")
        expected = {"allocation_id": source["allocation_id"], "slot": index,
                    "context_sha256": digest(context), "manifest_sha256": contract}
        if any(start.get(k) != v for k, v in expected.items()):
            raise GlobalStop("Durable slot start differs from source/context/contract")
        if index < len(slots):
            row = slots[index]
            if row["context_sha256"] != text_hash(context) \
                    or row["source_manifest_sha256"] != source_manifest \
                    or row["execution_contract_sha256"] != contract:
                raise GlobalStop("Official slot provenance differs from context/contract")


def finish_document(folder, phase, source, slots, reason, historical, seen):
    texts = [s["text"] for s in slots]
    structure = document_structure(texts)
    normalized = identity(structure["text"])
    independent = normalized not in historical and normalized not in seen
    success = reason in ("first_100_words", "maximum_slots")
    # A provider length termination is a frozen truncation indicator.
    truncated = False
    private = stage_folder(folder, phase)
    for slot in slots:
        last = slot["attempts"] - 1
        trace = private / "generation_attempts" / (
            f"{source['allocation_id']}-slot-{slot['slot']:02d}-attempt-{last:02d}.json")
        truncated |= load(trace)["stop_reason"] == "length"
    result = {
        "allocation_id": source["allocation_id"], "phase": phase, "ordinal": source["ordinal"],
        "source_locator": source["locator"], "domain": source["domain"],
        "source_excerpt": source["generation_prompt"], "source_sha256": digest(source),
        "manifest_sha256": sha256(Path(folder) / "manifest.json"), "finished_at": now(),
        "termination": reason, "generation_complete": success,
        **structure, "normalized_text_sha256": normalized,
        "identity_pass": independent, "provider_truncation": truncated,
        "structural_pass": success and structure["structural_pass"] and independent and not truncated,
        "slot_sha256": [sha256(private / "generation_slots" / (
            f"{source['allocation_id']}-slot-{r['slot']:02d}.json")) for r in slots],
    }
    freeze(doc_path(folder, phase, source["allocation_id"]), result)
    if structure["text"]:
        seen.add(normalized)
    return result


def run_documents(folder, phase, sources, journal, module, run_slot, bank, examples):
    folder = Path(folder)
    history = set(load(folder / "prior_identity_snapshot.json")["normalized_text_sha256"])
    seen = set()
    if phase == "main":
        for path in sorted((folder / "documents" / "pilot").glob("*.json")):
            row = load(path)
            if row["text"]:
                seen.add(row["normalized_text_sha256"])
    counts = {"terminal": 0, "structural_pass": 0}
    for source in sources:
        allocation = source["allocation_id"]
        path = doc_path(folder, phase, allocation)
        slots, interrupted = load_slots(folder, phase, allocation)
        verify_slot_contexts(folder, phase, source, slots, interrupted)
        if path.exists():
            saved = load(path)
            if saved["source_sha256"] != digest(source) or saved["slot_sha256"] != [
                    sha256(stage_folder(folder, phase) / "generation_slots" /
                           f"{allocation}-slot-{r['slot']:02d}.json") for r in slots]:
                raise GlobalStop("Document provenance changed")
            if saved["text"]:
                seen.add(saved["normalized_text_sha256"])
            counts["terminal"] += 1
            counts["structural_pass"] += saved["structural_pass"]
            continue
        reason = interrupted or stop_document([r["text"] for r in slots])
        while reason is None:
            capacity = journal.capacity(50)
            if capacity == "invocation_cap":
                return {**counts, "state": "paused_before_slot", **journal.totals()}
            if capacity:
                # All remaining allocations retain explicit terminal attrition rows.
                reason = capacity
                break
            slot_index = len(slots)
            private = stage_folder(folder, phase)
            context = " ".join([source["generation_prompt"]] + [r["text"] for r in slots])
            freeze(private / "slot_starts" / f"{allocation}-slot-{slot_index:02d}.json",
                   {"allocation_id": allocation, "slot": slot_index, "at": now(),
                    "context_sha256": digest(context), "manifest_sha256": journal.contract_hash})
            try:
                slot = run_slot(module, private, source, slot_index, context, bank, examples, journal)
                if not slot["text"]:
                    raise DocumentFailure("empty_official_output")
                slots.append(slot)
                reason = stop_document([r["text"] for r in slots])
            except DocumentFailure as exc:
                reason = str(exc)
            except GlobalStop:
                raise
            except Exception as exc:
                # Unexpected implementation/provenance errors require investigation.
                journal.event("halt", reason=f"unexpected_slot_error:{type(exc).__name__}")
                raise GlobalStop("Unexpected slot failure; automatic continuation prohibited") from exc
        result = finish_document(folder, phase, source, slots, reason, history, seen)
        counts["terminal"] += 1
        counts["structural_pass"] += result["structural_pass"]
    return {**counts, "state": "phase_terminal", **journal.totals()}


def injection_assets():
    from swan_final1_contract import BANK
    return ([r["normalized_template_amr"] for r in load(BANK)],
            load(BANK.parent.parent / "artifacts/normalized_amr_examples.json")[:3])


def verify_generation(folder, phase, sources):
    """No inference: bind documents to sources, saved slots and complete calls."""
    folder = Path(folder)
    contract_hash = sha256(folder / "manifest.json")
    journal = Journal(folder, phase, 1, None, contract_hash)
    import run_swan_positive_collection as swan
    official = swan.official_function_module_for_test({})
    bank, examples = injection_assets()
    documents = []
    for source in sources:
        path = doc_path(folder, phase, source["allocation_id"])
        if not path.exists():
            raise ValueError(f"{phase} still has pending allocations")
        row = load(path)
        slots, interrupted = load_slots(folder, phase, source["allocation_id"])
        verify_slot_contexts(folder, phase, source, slots, interrupted)
        if interrupted:
            slot = len(slots)
            failure_path = (stage_folder(folder, phase) / "generation_failures" /
                            f"{source['allocation_id']}-slot-{slot:02d}.json")
            if failure_path.exists():
                failure = load(failure_path)
                if failure["allocation_id"] != source["allocation_id"] or failure["slot"] != slot \
                        or failure["source_manifest_sha256"] != sha256(folder / "source_manifest.json") \
                        or failure["execution_contract_sha256"] != contract_hash \
                        or failure["attempts_recorded"] != len(failure["attempt_trace_sha256"]):
                    raise GlobalStop("Failed slot provenance differs")
                for attempt, expected in enumerate(failure["attempt_trace_sha256"]):
                    trace_path = (stage_folder(folder, phase) / "generation_attempts" /
                                  f"{source['allocation_id']}-slot-{slot:02d}-attempt-{attempt:02d}.json")
                    if sha256(trace_path) != expected:
                        raise GlobalStop("Failed attempt trace differs")
                    trace = load(trace_path)
                    call = journal.calls.get(trace["call_id"])
                    terminal = call and call["terminal"]
                    if trace["status"] == "transport_failed_or_uncertain":
                        if not terminal or terminal["kind"] != "uncertain":
                            raise GlobalStop("Uncertain failed attempt journal differs")
                    elif not terminal or terminal["kind"] != "complete":
                        raise GlobalStop("Local failed attempt lacks complete provider response")
            elif row["termination"] != "interrupted_slot_no_retry":
                raise GlobalStop("Interrupted slot lacks failure provenance")
        if row["manifest_sha256"] != contract_hash or row["source_sha256"] != digest(source):
            raise GlobalStop("Document contract or source differs")
        if row["text"] != " ".join(s["text"] for s in slots):
            raise GlobalStop("Document differs from official outputs")
        if row["slot_sha256"] != [
                sha256(stage_folder(folder, phase) / "generation_slots" /
                       f"{source['allocation_id']}-slot-{s['slot']:02d}.json") for s in slots]:
            raise GlobalStop("Document slot hashes differ")
        if row["generation_complete"] and (interrupted or row["termination"] != stop_document(
                [s["text"] for s in slots])):
            raise GlobalStop("Document stop rule differs")
        structure = document_structure([s["text"] for s in slots])
        for k, v in structure.items():
            if k != "structural_pass" and row[k] != v:
                raise GlobalStop("Document structure changed")
        truncated = False
        for slot in slots:
            if slot["attempts"] != len(slot["attempt_trace_sha256"]) or not 1 <= slot["attempts"] <= 50:
                raise GlobalStop("Official slot attempt count differs")
            context = " ".join([source["generation_prompt"]] +
                               [r["text"] for r in slots[:slot["slot"]]])
            template_seed = seed("injection-template", source["allocation_id"], slot["slot"])
            rng = random.Random(template_seed)
            templates = [rng.choice(bank) for _ in range((slot["attempts"]+4)//5)]
            chosen = [{"template_index": bank.index(t), "template_sha256": text_hash(t)}
                      for t in templates]
            if slot["template_seed"] != template_seed or slot["templates_chosen"] != chosen:
                raise GlobalStop("Official seeded template schedule differs")
            for attempt in range(slot["attempts"]):
                call_id = f"{source['allocation_id']}-slot-{slot['slot']:02d}-attempt-{attempt:02d}"
                call = journal.calls.get(call_id)
                trace = load(stage_folder(folder, phase) / "generation_attempts" / f"{call_id}.json")
                template = templates[attempt//5]
                request = body_for(official.build_user_prompt(template, examples, context),
                                   official.build_system_prompt())
                if not call or not call["terminal"] or call["terminal"]["kind"] != "complete" \
                        or trace["call_sha256"] != call["terminal"]["response_sha256"]:
                    raise GlobalStop("Official attempt lacks complete request provenance")
                if call["reserve"]["request"] != request or trace["template"] != template \
                        or trace["template_index"] != bank.index(template) \
                        or trace["score_seed"] != seed("injection-score", source["allocation_id"],
                                                     slot["slot"], attempt):
                    raise GlobalStop("Official prompt/template/score seed differs")
                response = journal._parse(
                    load(journal.folder / "responses" / f"{call_id}.json"),
                    call["reserve"]["reserve"])
                if trace["candidate"] != response["text"] or trace["usage"] != response["usage"] \
                        or trace["stop_reason"] != response["stop_reason"]:
                    raise GlobalStop("Trace differs from saved provider response")
                if trace["threshold"] != .75 or trace["accepted"] != (trace["score"] >= .75) \
                        or not 0 <= trace["score"] <= 1:
                    raise GlobalStop("Official acceptance decision differs")
                if len(trace.get("normalized_amrs", [])) != 2 \
                        or len(trace.get("normalized_amr_sha256", [])) != 2 \
                        or any(text_hash(value) != expected for value, expected in zip(
                            trace["normalized_amrs"], trace["normalized_amr_sha256"])) \
                        or trace["candidate_template_amr"] != trace["normalized_amrs"][-1] \
                        or text_hash(trace["candidate_template_before_final_normalization"]) != \
                        trace["candidate_template_before_final_normalization_sha256"]:
                    raise GlobalStop("Candidate R(s)/T(R(s)) provenance differs")
                if attempt < slot["attempts"] - 1 and trace["accepted"]:
                    raise GlobalStop("Official slot continued after acceptance")
                if attempt == slot["attempts"] - 1:
                    truncated |= trace["stop_reason"] == "length"
                    if trace["candidate"].strip() != slot["text"] or trace["accepted"] != slot["accepted"]:
                        raise GlobalStop("Official selected candidate differs")
                    if slot["chosen_template"] != template or slot["score"] != trace["score"] \
                            or slot["fallback"] != (not trace["accepted"] and bool(slot["text"])):
                        raise GlobalStop("Official target/score/fallback provenance differs")
                    if not slot["accepted"] and (slot["attempts"] != 50 or not slot["fallback"]):
                        raise GlobalStop("Official fallback budget differs")
        if row["provider_truncation"] != truncated:
            raise GlobalStop("Provider truncation flag changed")
        documents.append(row)
    historical = set(load(folder / "prior_identity_snapshot.json")["normalized_text_sha256"])
    if phase == "main":
        historical.update(load(p)["normalized_text_sha256"]
                          for p in (folder / "documents" / "pilot").glob("*.json") if load(p)["text"])
    for row in documents:
        normalized = identity(row["text"])
        independent = normalized not in historical
        expected = row["generation_complete"] and document_structure(
            [s["text"] for s in load_slots(folder, phase, row["allocation_id"])[0]])["structural_pass"] \
            and independent and not row["provider_truncation"]
        if row["normalized_text_sha256"] != normalized or row["identity_pass"] != independent \
                or row["structural_pass"] != expected:
            raise GlobalStop("Identity/structural decision cannot be reproduced")
        if row["text"]:
            historical.add(normalized)
    return documents
