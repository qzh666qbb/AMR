"""W04 continuation with immutable parent evidence and source-level missingness."""
import argparse
import json
from pathlib import Path
import re
import sys

import run_swan_v4_confirmation1 as run
from swan_final1_common import freeze, load, sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/swan_v4_independent_confirmation2"
PARENT = ROOT.parent / "v4_confirmation1_runtime/experiments/swan_v4_independent_confirmation1"
ORIGINAL_VALIDATE = run.validate_document
BUDGET = run.BUDGET


def source_of(call_id):
    match = re.fullmatch(r"(.+)-slot-(\d{2})-attempt-(\d{2})", call_id)
    if not match:
        raise BUDGET.GlobalStop("Unmapped generation call identity")
    return match[1]


def unavailable(journal):
    result = set()
    for cid, call in journal.calls.items():
        terminal = call["terminal"]
        if terminal is None or terminal["kind"] == "uncertain":
            result.add(source_of(cid))
        elif terminal["kind"] != "complete":
            raise BUDGET.GlobalStop("Invalid provider response requires investigation")
    return result


class Journal(BUDGET.Journal):
    def __init__(self, out, parent, send=None):
        BUDGET.CAPS = {"pilot": (run.CALL_CAP, run.COST_CAP)}
        self.previous = BUDGET.Journal(parent, "pilot", run.CALL_CAP, None,
                                      sha256(parent / "FREEZE.json"), allow_halted=True)
        unavailable(self.previous)
        super().__init__(out, "pilot", run.CALL_CAP, send, sha256(out / "FREEZE.json"))
        if set(self.calls) & set(self.previous.calls):
            raise BUDGET.GlobalStop("Duplicate lineage request")
        unavailable(self)

    def totals(self):
        current, old = super().totals(), self.previous.totals()
        for key in ("calls", "actual_peak_nanodollars", "exposure_nanodollars",
                    "uncertain_or_invalid_calls"):
            current[key] += old[key]
        current["inherited_calls"] = old["calls"]
        current["inherited_exposure_nanodollars"] = old["exposure_nanodollars"]
        return current

    def request(self, call_id, user_text, system_text, max_tokens, temperature, top_p):
        source_of(call_id)
        if call_id in self.previous.calls:
            raise BUDGET.GlobalStop("Original requests permanently non-replayable")
        if call_id in self.calls:
            call = self.calls[call_id]
            if call["terminal"] is None or call["terminal"]["kind"] != "complete":
                raise BUDGET.DocumentFailure("request_uncertain_no_retry")
            if call["reserve"]["request"] != BUDGET.body_for(
                    user_text, system_text, max_tokens, temperature, top_p):
                raise BUDGET.GlobalStop("Checkpoint request changed")
            response = self._parse(load(self.folder / "responses" / (call_id + ".json")),
                                   call["reserve"]["reserve"])
            return {**response, "call_sha256": call["terminal"]["response_sha256"]}
        try:
            return super().request(call_id, user_text, system_text, max_tokens, temperature, top_p)
        except BUDGET.DocumentFailure as exc:
            if str(exc) != "empty_model_output":
                raise
            return self.request(call_id, user_text, system_text, max_tokens, temperature, top_p)


def journal():
    return Journal(OUT, PARENT)


def validate_document(source, path):
    reference = load(OUT / "SOURCE_REFERENCE.json")
    relative = str(path.relative_to(OUT))
    if relative in reference["inherited_documents"]:
        if sha256(path) != reference["inherited_documents"][relative]:
            raise run.IntegrityHalt("Inherited document changed")
        # The parent contract and all nested slot/attempt bindings remain original.
        saved = run.OUT
        try:
            run.OUT = PARENT
            return ORIGINAL_VALIDATE(source, PARENT / relative)
        finally:
            run.OUT = saved
    return ORIGINAL_VALIDATE(source, path)


def verify():
    plan = load(OUT / "FREEZE.json")
    for name, expected in plan["dependencies"].items():
        if sha256(ROOT / name) != expected:
            raise run.IntegrityHalt("Successor dependency changed:" + name)
    reference = load(OUT / "SOURCE_REFERENCE.json")
    if reference["parent"] != str(PARENT):
        raise run.IntegrityHalt("Parent reference changed")
    for name, expected in reference["files"].items():
        if sha256(PARENT / name) != expected:
            raise run.IntegrityHalt("Parent evidence changed")
    for name, expected in reference["parent_dependencies"].items():
        original, current = PARENT.parents[1] / name, ROOT / name
        if original.resolve() != current.resolve() and sha256(original) != expected:
            raise run.IntegrityHalt("Parent dependency changed")
    # Source selection replay uses its unmodified original root and allocation names.
    run.sources.ROOT, run.sources.OUT = PARENT.parents[1], PARENT
    run.sources.audit()
    if plan["positive_sources"] != 250 or plan["generation_max_calls"] != 40550:
        raise run.IntegrityHalt("Frozen cohort or request cap changed")
    return plan


def configure():
    run.OUT = OUT
    run.bind()
    run.verify, run.journal = verify, journal
    run.validate_document, run.generate = validate_document, generate


def gpu_guard():
    import torch
    if not torch.cuda.is_available():
        raise run.ResourcePause("CUDA_required")
    free, total = torch.cuda.mem_get_info()
    if (total - free) / total >= .8:
        raise run.ResourcePause("GPU_80pct_headroom")


def seal_generation():
    run.verify_generation()
    missing = unavailable(journal().previous) | unavailable(journal())
    freeze(OUT / "MISSING_RESPONSE_DISPOSITION.json", {
        "sources": sorted(missing), "source_count": len(missing),
        "reserves_retained_once": True, "request_replay": False,
        "provider_charge_reconciled": False})
    records = [
        p for pattern in ("private/documents/*.json", "private/slots/*.json",
                          "private/attempts/*.json", "private/pilot/journal.jsonl",
                          "private/pilot/responses/*.json", "private/local_errors/*.json")
        for p in OUT.glob(pattern)]
    records += [OUT / n for n in ("GENERATION_COMPLETE.json", "SOURCE_REFERENCE.json",
                                  "MISSING_RESPONSE_DISPOSITION.json", "GENERATION_READINESS.json")]
    generation = OUT / "generation"
    complete = load(OUT / "GENERATION_COMPLETE.json")
    freeze(generation / "RESULT.json", {
        "status": "GENERATION_COMPLETE_MISSING_RETAINED", "sources": 250,
        "automatic_qualified": complete["automatic_qualified"],
        "budget": complete["budget"], "missing_sources": len(missing),
        "independent_human_quality": None})
    freeze(generation / "AUDIT.json", {
        "freeze_sha256": sha256(OUT / "FREEZE.json"),
        "files": {str(p.relative_to(OUT)): sha256(p) for p in records}})
    freeze(generation / "TERMINAL.json", {
        "status": "GENERATION_COMPLETE_MISSING_RETAINED",
        "result_sha256": sha256(generation / "RESULT.json"),
        "audit_sha256": sha256(generation / "AUDIT.json")})


def generate(key_file):
    verify()
    if (OUT / "GENERATION_COMPLETE.json").exists():
        run.verify_generation()
        run.review_packet()
        seal_generation()
        return
    records, bank = run.read_sources("positive_source"), run.shared.read_bank()
    if run.shared.assignment(records, bank) != load(OUT / "GENERATION_ASSIGNMENT.json"):
        raise run.IntegrityHalt("Original target assignment changed")
    log = journal()
    inherited = {Path(p).stem for p in load(OUT / "SOURCE_REFERENCE.json")["inherited_documents"]}
    if {source_of(cid) for cid in log.previous.calls} != inherited | unavailable(log.previous):
        raise run.IntegrityHalt("Unaccounted parent partial sources")
    key = run.GENERATOR.pilot.credential.read_credential(key_file)
    log.send = lambda body: BUDGET.transport(body, key)
    tokenize = run.shared.tokenizer()
    gpu_guard()
    with run.shared.runtime(OUT) as module:
        # Every new invocation checks real FP32 parser and matching replay before requests.
        readiness = run.shared.readiness(module, bank)
        gpu_guard()
        if not (OUT / "GENERATION_READINESS.json").exists():
            freeze(OUT / "GENERATION_READINESS.json", readiness)
        original = module.evaluate
        for index, source in enumerate(records):
            run.shared.space_check()
            path = OUT / "private/documents" / (source["allocation_id"] + ".json")
            if path.exists():
                validate_document(source, path)
                continue
            count, lo, hi = run.shared.BIN_CONTRACTS[source["length_bin"]]
            run.GENERATOR.pilot.LENGTH = run.shared.prompt_addition(source["length_bin"])

            def evaluate(text, target, candidate_id):
                gpu_guard()
                result = original(text, target, candidate_id)
                if not lo <= len(run.shared.WORD.findall(text)) <= hi:
                    result.update(accepted=False, rejection="frozen_bin_length_gate")
                return result

            module.evaluate = evaluate
            sentences, hashes, failure = [], {}, None
            if source["allocation_id"] in unavailable(log.previous) | unavailable(log):
                failure = "missing_response_source_unavailable"
            else:
                for slot in range(count):
                    try:
                        record = run.GENERATOR.run_slot(
                            module, source, slot,
                            source["generation_prompt"] + " " + " ".join(sentences), bank, log)
                    except BUDGET.DocumentFailure as exc:
                        if str(exc) not in ("request_uncertain_no_retry", "request_body_exceeds_8192_bytes"):
                            raise
                        failure = ("missing_response_source_unavailable"
                                   if str(exc) == "request_uncertain_no_retry" else str(exc))
                        break
                    slot_path = OUT / "private/slots" / (f"{source['allocation_id']}-slot-{slot:02d}.json")
                    hashes[str(slot_path.relative_to(OUT))] = sha256(slot_path)
                    sentences.append(record["text"])
                    if not record["accepted"]:
                        failure = "slot_attempts_exhausted"
                        break
            text = " ".join(sentences)
            lower, upper = run.sources.BINS[source["length_bin"]]
            if failure is None and (tokenize(text) != sentences or not lower <= len(text.split()) <= upper):
                failure = "whole_document_sentence_or_word_gate"
            freeze(path, {
                "allocation_id": source["allocation_id"], "source_window_sha256": source["window_sha256"],
                "contract_sha256": sha256(OUT / "FREEZE.json"), "length_bin": source["length_bin"],
                "text": text, "sentences": sentences, "automatic_qualified": failure is None,
                "failure": failure, "slot_sha256": hashes, "human_qualified": None,
                "detector_feedback_used": False})
            run.heartbeat("generating", completed_documents=index + 1, total=250, budget=log.totals())
    documents = {f"private/documents/{s['allocation_id']}.json":
                 sha256(OUT / "private/documents" / (s["allocation_id"] + ".json")) for s in records}
    verify()
    freeze(OUT / "GENERATION_COMPLETE.json", {
        "contract_sha256": sha256(OUT / "FREEZE.json"), "documents": documents, "budget": log.totals(),
        "automatic_qualified": sum(load(OUT / p)["automatic_qualified"] for p in documents),
        "all_attempts_before_detector": True, "independent_human_submissions": 0})
    run.review_packet()
    seal_generation()
    run.heartbeat("generation_complete", completed_documents=250, total=250)


def verify_final():
    """Validate every scoring row and nested candidate evidence before queue success."""
    verify()
    run.verify_generation()
    audit = load(OUT / "AUDIT.json")
    if audit["row_count"] != 8500 or len(audit["rows"]) != 8500:
        raise run.IntegrityHalt("Incomplete score denominator")
    for name, expected in audit["rows"].items():
        if sha256(OUT / name) != expected:
            raise run.IntegrityHalt("Scoring row changed")
        row = load(OUT / name)
        if row.get("evidence_path") and sha256(OUT / row["evidence_path"]) != row["evidence_sha256"]:
            raise run.IntegrityHalt("Candidate evidence changed")
    for name, expected in load(OUT / "generation/AUDIT.json")["files"].items():
        if sha256(OUT / name) != expected:
            raise run.IntegrityHalt("Generation evidence changed")
    freeze(OUT / "FINAL_VERIFICATION.json", {
        "status": "PASS", "rows": 8500, "independent_human_quality": False,
        "result_sha256": sha256(OUT / "RESULT.json"), "audit_sha256": sha256(OUT / "AUDIT.json")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("verify", "generate", "score", "collect-human"))
    ap.add_argument("--key-file", type=Path)
    args = ap.parse_args()
    configure()
    code = run.execute(args.action, args.key_file)
    if args.action == "score" and code == 0:
        verify_final()
    return code


if __name__ == "__main__":
    sys.exit(main())
