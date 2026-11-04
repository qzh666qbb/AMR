"""Frozen W05 null continuation; missing responses fail sources, never replay."""
import json
from pathlib import Path
import time

import run_swan_normal_completion1 as normal
import swan_final1_budget as budget
from swan_final1_common import freeze, load, sha256, single_writer


def source_of(call_id):
    source, separator, suffix = call_id.rpartition("-null-slot-")
    if not separator or not source or len(suffix) != 13 or suffix[2:11] != "-attempt-":
        raise budget.GlobalStop("Unknown null call identity")
    if not suffix[:2].isdigit() or not suffix[-2:].isdigit():
        raise budget.GlobalStop("Malformed null call identity")
    return source


def unavailable_sources(journal):
    result = set()
    for cid, call in journal.calls.items():
        terminal = call["terminal"]
        if terminal is None or terminal["kind"] == "uncertain":
            result.add(source_of(cid))
        elif terminal["kind"] != "complete":
            raise budget.GlobalStop("Invalid provider response requires new investigation")
    return result


class Journal(budget.Journal):
    def __init__(self, out, parent, send):
        budget.CAPS = dict(normal.CAPS)
        self.previous = budget.Journal(
            parent, "llm_null", normal.CAPS["llm_null"][0], None,
            sha256(parent / "FREEZE.json"), allow_halted=True)
        # Reject invalid/schema/accounting parents; only explicitly bound missing responses qualify.
        unavailable_sources(self.previous)
        super().__init__(out, "llm_null", normal.CAPS["llm_null"][0], send,
                         sha256(out / "FREEZE.json"))
        if set(self.previous.calls) & set(self.calls):
            raise budget.GlobalStop("Duplicate request across generations")
        unavailable_sources(self)

    def totals(self):
        current = super().totals()
        previous = self.previous.totals()
        for key in ("calls", "actual_peak_nanodollars", "exposure_nanodollars",
                    "uncertain_or_invalid_calls"):
            current[key] += previous[key]
        current["inherited_calls"] = previous["calls"]
        current["inherited_exposure_nanodollars"] = previous["exposure_nanodollars"]
        return current

    def request(self, call_id, user_text, system_text, max_tokens, temperature, top_p):
        if call_id in self.previous.calls:
            raise budget.GlobalStop("Original requests are permanently non-replayable")
        if call_id in self.calls:
            call = self.calls[call_id]
            if call["terminal"] is None or call["terminal"]["kind"] != "complete":
                raise budget.DocumentFailure("request_uncertain_no_retry")
            if call["reserve"]["request"] != budget.body_for(
                    user_text, system_text, max_tokens, temperature, top_p):
                raise budget.GlobalStop("Changed request on checkpoint replay")
            value = self._parse(load(self.folder / "responses" / (call_id + ".json")),
                                call["reserve"]["reserve"])
            return {**value, "call_sha256": call["terminal"]["response_sha256"]}
        try:
            return super().request(call_id, user_text, system_text,
                                   max_tokens, temperature, top_p)
        except budget.DocumentFailure as exc:
            if str(exc) != "empty_model_output":
                raise
            # Completed empty responses are paid candidates rejected by the unchanged length gate.
            return self.request(call_id, user_text, system_text, max_tokens, temperature, top_p)


def sources(out):
    design = load(out / "private/ALIGNMENT_DESIGN.json")
    ids = set(design["llm_calibration"] + design["llm_test"])
    result = [r for phase in ("calibration", "test")
              for r in load(out / "private/sources" / (phase + ".json"))
              if r["allocation_id"] in ids]
    if len(result) != 800 or len({r["allocation_id"] for r in result}) != 800:
        raise budget.GlobalStop("Frozen 800-source cohort changed")
    return result


def check_record(out, source, path, reference):
    relative = str(path.relative_to(out))
    if relative in reference["inherited_null_documents"]:
        if sha256(path) != reference["inherited_null_documents"][relative]:
            raise budget.GlobalStop("Inherited null document changed")
    else:
        record = load(path)
        if record["freeze_sha256"] != sha256(out / "FREEZE.json"):
            raise budget.GlobalStop("Successor document freeze mismatch")
    record = load(path)
    payload = {k: v for k, v in record.items() if k != "record_sha256"}
    if (record["record_sha256"] != normal.rt.digest(json.dumps(payload, sort_keys=True))
            or record["source_id"] != source["allocation_id"]):
        raise budget.GlobalStop("Null document checkpoint mismatch")
    return record


def execute(out, parent, send, verify_sources):
    """Transport is injected for deterministic contract tests and real execution."""
    with single_writer(out):
        started = time.monotonic()
        try:
            if (out / "HALT.json").exists():
                raise budget.GlobalStop("Permanent successor halt")
            normal.verify(out)
            reference = verify_sources()
            journal = Journal(out, parent, send)
            planned = sources(out)
            inherited = {Path(p).stem for p in reference["inherited_null_documents"]}
            previous_sources = {source_of(cid) for cid in journal.previous.calls}
            if previous_sources != inherited | unavailable_sources(journal.previous):
                raise budget.GlobalStop("Unaccounted partial parent sources")
            if (out / "TERMINAL.json").exists():
                terminal = load(out / "TERMINAL.json")
                for key, name in (("result_sha256", "RESULT.json"), ("audit_sha256", "AUDIT.json")):
                    if terminal[key] != sha256(out / name):
                        raise budget.GlobalStop("Null terminal changed")
                normal.verify_seal(out, out / "AUDIT.json")
                normal.verify_seal(out, out / "LLM_NULL_GENERATION_SEAL.json")
                return
            tokenize = normal.tokenizer()
            records = []
            for index, source in enumerate(planned):
                path = out / "private/null_documents" / (source["allocation_id"] + ".json")
                if path.exists():
                    records.append(check_record(out, source, path, reference))
                    continue
                if normal.shutil.disk_usage(out).free < 1024**3:
                    raise normal.rt.ResourcePause("Null generation disk headroom")
                sentences, failure = [], None
                blocked = unavailable_sources(journal.previous) | unavailable_sources(journal)
                if source["allocation_id"] in blocked:
                    failure = "missing_response_source_unavailable"
                else:
                    for slot in range(16):
                        context = source["generation_prompt"] + " " + " ".join(sentences)
                        try:
                            result = normal.null_slot(
                                out, source, slot, context, journal, tokenize)
                        except budget.DocumentFailure as exc:
                            if str(exc) not in ("request_uncertain_no_retry",
                                                "request_body_exceeds_8192_bytes"):
                                raise
                            failure = ("missing_response_source_unavailable"
                                       if str(exc) == "request_uncertain_no_retry" else str(exc))
                            break
                        sentences.append(result["text"])
                        if not result["accepted"]:
                            failure = "slot_exhausted"
                            break
                text = " ".join(sentences)
                eligible = (failure is None and len(sentences) == 16
                            and tokenize(text) == sentences
                            and 100 <= len(normal.WORD.findall(text)) <= 220)
                record = {"source_id": source["allocation_id"], "text": text,
                          "sentences": sentences, "eligible": eligible, "failure": failure,
                          "words": len(normal.WORD.findall(text)),
                          "freeze_sha256": sha256(out / "FREEZE.json")}
                freeze(path, {**record, "record_sha256": normal.rt.digest(
                    json.dumps(record, sort_keys=True))})
                records.append(record)
                normal.rt.atomic(out / "HEARTBEAT.json", {
                    "phase": "llm_null_generation", "completed": index + 1, "total": 800,
                    "eligible": sum(r["eligible"] for r in records),
                    "budget": journal.totals(), "at": normal.rt.now()})
            if len(records) != 800:
                raise budget.GlobalStop("Incomplete null source denominator")
            verify_sources()
            totals = {k: v for k, v in journal.totals().items() if k != "new_calls"}
            freeze(out / "LLM_NULL_BUDGET.json", totals)
            missing = unavailable_sources(journal.previous) | unavailable_sources(journal)
            freeze(out / "MISSING_RESPONSE_DISPOSITION.json", {
                "source_count": len(missing), "source_ids": sorted(missing),
                "request_replay": False, "reserves_retained_once": True,
                "provider_charge_reconciled": False})
            seal_files = list((out / "private/null_documents").glob("*.json"))
            seal_files += [out / "LLM_NULL_BUDGET.json", out / "MISSING_RESPONSE_DISPOSITION.json"]
            normal.file_seal(out, out / "LLM_NULL_GENERATION_SEAL.json", seal_files)
            design = load(out / "private/ALIGNMENT_DESIGN.json")
            byid = {r["source_id"]: r for r in records}
            result = {
                "status": "NULL_GENERATION_COMPLETE_MISSING_RETAINED",
                "completed_sources": 800, "eligible": sum(r["eligible"] for r in records),
                "missing_response_sources": len(missing), "budget": totals,
                "coverage": {phase: {
                    "n": len(design[phase]),
                    "eligible": sum(byid[s]["eligible"] for s in design[phase])
                } for phase in ("llm_calibration", "llm_test")},
                "formal_FPR": None, "formal_TPR": None, "human_reviews": 0,
            }
            freeze(out / "RESULT.json", result)
            accounting = list((out / "private/llm_null").glob("journal.jsonl"))
            accounting += list((out / "private/llm_null/responses").glob("*.json"))
            normal.file_seal(out, out / "AUDIT.json", seal_files + accounting + [
                out / "SOURCE_REFERENCE.json", out / "LLM_NULL_GENERATION_SEAL.json"])
            freeze(out / "TERMINAL.json", {
                "status": result["status"], "result_sha256": sha256(out / "RESULT.json"),
                "audit_sha256": sha256(out / "AUDIT.json")})
        except (budget.Pause, normal.rt.ResourcePause) as exc:
            normal.rt.atomic(out / "PAUSED.json", {"reason_type": type(exc).__name__,
                                                  "reason": str(exc)})
            raise SystemExit(76 if isinstance(exc, budget.Pause) else 75)
        except Exception as exc:
            normal.rt.atomic(out / "HALT.json", {"reason_type": type(exc).__name__,
                                                "automatic_restart_allowed": False})
            raise
        finally:
            normal.rt.atomic(out / "private/sessions" / (str(time.time_ns()) + ".json"),
                             {"stage": "null_generation", "seconds": time.monotonic() - started})
