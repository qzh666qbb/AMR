"""Frozen independent v4 confirmation; generation precedes every detector row."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_swan_path0_completion1 as shared
import swan_v4_confirmation1_sources as sources
import swan_v4_confirmation1_scoring as scoring
from swan_final1_common import digest, freeze, load, now, sha256, single_writer
from swan_supplement1_runtime import IntegrityHalt
from swan_supplement4_runtime import ResourcePause

OUT = ROOT / "experiments/swan_v4_independent_confirmation1"
GENERATOR = shared.generation
BUDGET = shared.budget
POSITIVE_N = 250
CALL_CAP = 40550
COST_CAP = 25_000_000_000


def bind():
    shared.OUT = OUT
    GENERATOR.OUT = OUT
    GENERATOR.pilot.OUT = OUT
    BUDGET.CAPS = {"pilot": (CALL_CAP, COST_CAP)}


def atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="status-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def heartbeat(stage, **fields):
    atomic(OUT / "HEARTBEAT.json", {"at": now(), "stage": stage, **fields})


def verify():
    plan = load(OUT / "FREEZE.json")
    for relative, expected in plan["dependencies"].items():
        if sha256(ROOT / relative) != expected:
            raise IntegrityHalt("dependency_SHA_changed:" + relative)
    sources.audit()
    if plan["positive_sources"] != POSITIVE_N or plan["generation_max_calls"] != CALL_CAP:
        raise IntegrityHalt("generation_contract_changed")
    return plan


class Journal(GENERATOR.ReplayJournal):
    def request(self, *args, **kwargs):
        try:
            return super().request(*args, **kwargs)
        except BUDGET.DocumentFailure as exc:
            if str(exc) != "empty_model_output" or self.uncertain:
                raise
            # This completed paid response is already journaled. Replaying it
            # returns an empty candidate to the official candidate rejection.
            return super().request(*args, **kwargs)


def journal():
    bind()
    result = Journal(OUT, "pilot", CALL_CAP, None, sha256(OUT / "FREEZE.json"), parents=[])
    if result.uncertain:
        raise IntegrityHalt("uncertain_call_blocks_resume")
    return result


def read_sources(stage):
    return [json.loads(line) for line in (OUT / "sealed" / (stage + ".jsonl")).read_text().splitlines()]


def validate_document(source, path):
    document = load(path)
    if (document["contract_sha256"] != sha256(OUT / "FREEZE.json")
            or document["allocation_id"] != source["allocation_id"]
            or document["source_window_sha256"] != source["window_sha256"]
            or document["length_bin"] != source["length_bin"]
            or document["detector_feedback_used"] is not False):
        raise IntegrityHalt("generated_document_identity_changed")
    sentences = []
    for index, (relative, expected) in enumerate(document["slot_sha256"].items()):
        slot_path = OUT / relative
        if (relative != f"private/slots/{source['allocation_id']}-slot-{index:02d}.json"
                or sha256(slot_path) != expected):
            raise IntegrityHalt("generated_slot_identity_changed")
        record = load(slot_path)
        if (record["allocation_id"] != source["allocation_id"] or record["slot"] != index
                or record["contract_sha256"] != sha256(OUT / "FREEZE.json")
                or record["context_sha256"] != digest(source["generation_prompt"] + " " + " ".join(sentences))):
            raise IntegrityHalt("generated_slot_binding_changed")
        for call_id, attempt_sha in record["attempt_file_hashes"].items():
            if sha256(OUT / "private/attempts" / (call_id + ".json")) != attempt_sha:
                raise IntegrityHalt("generated_attempt_changed")
        sentences.append(record["text"])
    if document["text"] != " ".join(sentences) or document["sentences"] != sentences:
        raise IntegrityHalt("generated_text_changed")
    if document["automatic_qualified"] and (
            len(sentences) != shared.BIN_CONTRACTS[source["length_bin"]][0] or document["failure"] is not None):
        raise IntegrityHalt("generated_qualification_changed")
    return document


def review_packet():
    records = []
    for source in read_sources("positive_source"):
        path = OUT / "private/documents" / (source["allocation_id"] + ".json")
        records.append((source, validate_document(source, path)))
    records.sort(key=lambda pair: digest(["v4-review-order", pair[0]["allocation_id"]]))
    mapping, items = {}, []
    for source, document in records:
        item_id = digest(["v4-blind", source["allocation_id"]])[:20]
        mapping[item_id] = source["allocation_id"]
        items.append({"item_id": item_id, "context": source["generation_prompt"], "text": document["text"]})
    folder = OUT / "private/human_review"
    freeze(folder / "PACKET.json", {"items": items, "reviewers_required": 2,
                                   "scores_and_targets_hidden": True})
    freeze(OUT / "private/HUMAN_LINKAGE.json", mapping)
    for reviewer in ("reviewer1", "reviewer2"):
        path = folder / (reviewer + ".csv")
        if not path.exists():
            with path.open("x", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "item_id", "grammar_pass", "coherence_pass", "no_AMR_leak",
                    "reviewer_id", "is_human", "notes"])
                writer.writeheader()
                writer.writerows({"item_id": x["item_id"]} for x in items)


def verify_generation():
    complete = load(OUT / "GENERATION_COMPLETE.json")
    expected_paths = {
        f"private/documents/{s['allocation_id']}.json": s for s in read_sources("positive_source")}
    if (complete["contract_sha256"] != sha256(OUT / "FREEZE.json")
            or set(complete["documents"]) != set(expected_paths) or len(expected_paths) != POSITIVE_N):
        raise IntegrityHalt("generation_denominator_or_contract_changed")
    documents = []
    for relative, source in expected_paths.items():
        if sha256(OUT / relative) != complete["documents"][relative]:
            raise IntegrityHalt("generated_document_changed")
        documents.append(validate_document(source, OUT / relative))
    totals = journal().totals()
    for key in ("calls", "actual_peak_nanodollars", "exposure_nanodollars", "uncertain_or_invalid_calls"):
        if totals[key] != complete["budget"][key]:
            raise IntegrityHalt("generation_accounting_changed")
    if complete["automatic_qualified"] != sum(d["automatic_qualified"] for d in documents):
        raise IntegrityHalt("generation_qualified_count_changed")
    return complete


def generate(key_file):
    verify()
    bind()
    if (OUT / "GENERATION_COMPLETE.json").exists():
        verify_generation()
        review_packet()
        return
    bank = shared.read_bank()
    records = read_sources("positive_source")
    if shared.assignment(records, bank) != load(OUT / "GENERATION_ASSIGNMENT.json"):
        raise IntegrityHalt("target_assignment_changed")
    log = journal()
    key = GENERATOR.pilot.credential.read_credential(key_file)
    log.send = lambda body: BUDGET.transport(body, key)
    tokenize = shared.tokenizer()
    with shared.runtime(OUT) as module:
        if not (OUT / "GENERATION_READINESS.json").exists():
            freeze(OUT / "GENERATION_READINESS.json", shared.readiness(module, bank))
        original_evaluate = module.evaluate
        for index, source in enumerate(records):
            shared.space_check()
            path = OUT / "private/documents" / (source["allocation_id"] + ".json")
            if path.exists():
                validate_document(source, path)
                continue
            count, lo, hi = shared.BIN_CONTRACTS[source["length_bin"]]
            GENERATOR.pilot.LENGTH = shared.prompt_addition(source["length_bin"])

            def evaluate(text, target, candidate_id):
                result = original_evaluate(text, target, candidate_id)
                if not lo <= len(shared.WORD.findall(text)) <= hi:
                    result.update(accepted=False, rejection="frozen_bin_length_gate")
                return result

            module.evaluate = evaluate
            sentences, hashes, failure = [], {}, None
            for slot in range(count):
                try:
                    record = GENERATOR.run_slot(
                        module, source, slot, source["generation_prompt"] + " " + " ".join(sentences),
                        bank, log)
                except BUDGET.DocumentFailure as exc:
                    if log.uncertain:
                        raise IntegrityHalt("uncertain_call_blocks_resume") from exc
                    if str(exc) != "request_body_exceeds_8192_bytes":
                        raise
                    failure = "request_body_exceeds_8192_bytes"
                    break
                slot_path = OUT / "private/slots" / (f"{source['allocation_id']}-slot-{slot:02d}.json")
                hashes[str(slot_path.relative_to(OUT))] = sha256(slot_path)
                sentences.append(record["text"])
                if not record["accepted"]:
                    failure = "slot_attempts_exhausted"
                    break
            text = " ".join(sentences)
            lower, upper = sources.BINS[source["length_bin"]]
            if failure is None and (tokenize(text) != sentences or not lower <= len(text.split()) <= upper):
                failure = "whole_document_sentence_or_word_gate"
            freeze(path, {
                "allocation_id": source["allocation_id"], "source_window_sha256": source["window_sha256"],
                "contract_sha256": sha256(OUT / "FREEZE.json"), "length_bin": source["length_bin"],
                "text": text, "sentences": sentences, "automatic_qualified": failure is None,
                "failure": failure, "slot_sha256": hashes, "human_qualified": None,
                "detector_feedback_used": False,
            })
            heartbeat("generating", completed_documents=index + 1, total=POSITIVE_N, budget=log.totals())
    documents = {f"private/documents/{s['allocation_id']}.json":
                 sha256(OUT / "private/documents" / (s["allocation_id"] + ".json")) for s in records}
    freeze(OUT / "GENERATION_COMPLETE.json", {
        "contract_sha256": sha256(OUT / "FREEZE.json"), "documents": documents, "budget": log.totals(),
        "automatic_qualified": sum(load(OUT / p)["automatic_qualified"] for p in documents),
        "all_attempts_before_detector": True, "independent_human_submissions": 0,
    })
    verify_generation()
    review_packet()
    heartbeat("generation_complete", completed_documents=POSITIVE_N, total=POSITIVE_N)


def score():
    import torch
    dev = scoring.core()
    verify()
    generated = verify_generation()
    review_packet()
    if not torch.cuda.is_available():
        raise ResourcePause("CUDA_unavailable")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    if torch.cuda.mem_get_info()[0] < 3072 * 1024**2:
        raise ResourcePause("parser_admission_requires_3072MiB_free")
    args = shared.detector_args()
    shared.space_check()
    freeze(OUT / "P0.json", dev.run_p0(args.bank))
    detector, recovery = scoring.build_control(args)
    model = recovery.backend._load()
    if any(p.dtype != torch.float32 for p in model.model.parameters()):
        raise IntegrityHalt("detector_parser_not_FP32")
    payload = dev.runtime_payload(args, {"control": detector})
    payload.update(version="v4-independent-confirmation-completion1",
                   completion_contract_sha256=sha256(OUT / "FREEZE.json"),
                   generation_sha256=sha256(OUT / "GENERATION_COMPLETE.json"))
    payload["calibration"]["curve_alphas"] = [scoring.ALPHA]
    freeze(OUT / "RUNTIME_MANIFEST.json", json.loads(json.dumps(payload)))
    runtime_sha = sha256(OUT / "RUNTIME_MANIFEST.json")
    all_rows, all_sources, calibrator = {}, {}, None
    for stage in ("calibration", "null", "positive_source"):
        records, rows = read_sources(stage), []
        total = len(records) * (1 if stage == "calibration" else 6)
        for source in records:
            document = None if stage != "positive_source" else validate_document(
                source, OUT / "private/documents" / (source["allocation_id"] + ".json"))
            failed = document is not None and not document["automatic_qualified"]
            texts = ({"baseline": source["text"]} if stage == "calibration" else
                     {v: "" for v in scoring.VARIANTS} if failed else
                     dev.variants(document["text"] if document else source["text"]))
            for variant, text in texts.items():
                shared.space_check()
                binding = {
                    "runtime_sha256": runtime_sha, "input_manifest_sha256": sha256(OUT / "INPUT_MANIFEST.json"),
                    "allocation_id": source["allocation_id"], "source_window_sha256": source["window_sha256"],
                    "variant": variant, "text_sha256": dev.sha(text.encode()),
                }
                path = OUT / "results/rows" / (source["allocation_id"] + "__" + variant + ".json")
                if failed:
                    row = freeze(path, scoring.miss_row(source, variant, binding, document["failure"]))
                else:
                    row = dev.checkpoint(path, binding)
                    if row is None:
                        heartbeat(stage, completed=len(rows), total=total, active_row=path.stem)
                        row = scoring.score_document(args, detector, recovery, source, variant, text, binding)
                        freeze(path, row)
                rows.append(row)
                heartbeat(stage, completed=len(rows), total=total, parser_unique_spans=len(recovery.records))
        all_rows[stage], all_sources[stage] = rows, records
        if stage == "calibration":
            calibrator = scoring.fit(args, detector, rows, records, runtime_sha)
    summary = scoring.summarize(
        all_rows["null"], all_rows["positive_source"], calibrator,
        [s["allocation_id"] for s in all_sources["null"]],
        [s["allocation_id"] for s in all_sources["positive_source"]])
    row_hashes = {str(p.relative_to(OUT)): sha256(p) for p in sorted((OUT / "results/rows").glob("*.json"))}
    if len(row_hashes) != 8500:
        raise IntegrityHalt("complete_8500_row_denominator_required")
    invalid = sum(r["arms"]["control"]["selected_invalid_edges"] for rows in all_rows.values() for r in rows)
    if invalid:
        raise IntegrityHalt("invalid_selected_edges")
    result = {
        "status": "computational_complete_pending_human_quality",
        "formal_decision": "pending_human_quality" if summary["computational_gates_pass"] else "computational_gates_failed",
        "binding_runtime_sha256": runtime_sha, "calibration_rows": 1000, "null_rows": 6000, "positive_rows": 1500,
        "positive_sources": POSITIVE_N, "automatic_qualified": generated["automatic_qualified"],
        "generation_budget": generated["budget"], "independent_human_quality": None,
        "parser_recovery_summary": recovery.summary(), **summary,
        "limitations": ["short_text_17_to_48_words_only", "independent_human_quality_pending",
                        "does_not_rewrite_prior_development_failure"],
    }
    freeze(OUT / "RESULT.json", result)
    freeze(OUT / "AUDIT.json", {
        "status": "computational_complete", "formal_closure": False,
        "result_sha256": sha256(OUT / "RESULT.json"), "generation_sha256": sha256(OUT / "GENERATION_COMPLETE.json"),
        "calibration_sha256": sha256(OUT / "results/CALIBRATION_control.json"),
        "rows": row_hashes, "row_count": 8500, "selected_invalid_edges": invalid, "source_replacement": False,
    })
    freeze(OUT / "TERMINAL.json", {
        "status": result["status"], "result_sha256": sha256(OUT / "RESULT.json"),
        "audit_sha256": sha256(OUT / "AUDIT.json"), "formal_closure": False})
    heartbeat(result["status"], completed_rows=8500, total_rows=8500)


def collect_reviews():
    verify()
    verify_generation()
    review_packet()
    packet_path = OUT / "private/human_review/PACKET.json"
    packet = load(packet_path)
    ids = {item["item_id"] for item in packet["items"]}
    if len(ids) != POSITIVE_N:
        raise IntegrityHalt("human_packet_denominator")
    accepted, identities, hashes = [], [], {}
    for name in ("reviewer1", "reviewer2"):
        path = packet_path.parent / (name + ".csv")
        hashes[name] = sha256(path)
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != POSITIVE_N or {r["item_id"] for r in rows} != ids:
            raise ValueError("Review universe missing or duplicated")
        valid, authors = {}, set()
        for row in rows:
            answers = [row[k].strip() for k in ("grammar_pass", "coherence_pass", "no_AMR_leak")]
            if all(x in ("0", "1") for x in answers) and row["reviewer_id"].strip() and row["is_human"].strip().lower() == "true":
                valid[row["item_id"]] = all(x == "1" for x in answers)
                authors.add(row["reviewer_id"].strip().casefold())
        if len(authors) > 1:
            raise ValueError("One independent reviewer identity required per sheet")
        identities.append(authors)
        accepted.append(valid)
    if identities[0] & identities[1]:
        raise ValueError("Two distinct independent human reviewers required")
    both = set(accepted[0]) & set(accepted[1])
    qualified = {i for i in both if accepted[0][i] and accepted[1][i]}
    result = {
        "status": "complete" if len(both) == POSITIVE_N else "pending",
        "total_sources": POSITIVE_N, "valid_submissions": [len(a) for a in accepted],
        "complete_sources": len(both), "qualified_sources": len(qualified),
        "missing_full_denominator": POSITIVE_N - len(both),
        "quality_pass": len(both) == POSITIVE_N and len(qualified) == POSITIVE_N,
        "criteria": "both_independent_reviewers_pass_all_three_items_for_every_source",
        "review_files_sha256": hashes, "packet_sha256": sha256(packet_path),
        "reviewer_independence": "distinct_declared_human_ids; identity_not_externally_authenticated",
    }
    evidence = OUT / "private/human_collections" / (digest(result) + ".json")
    freeze(evidence, result)
    atomic(OUT / "HUMAN_STATUS.json", {**result, "evidence_sha256": sha256(evidence), "at": now()})
    if len(both) == POSITIVE_N and (OUT / "TERMINAL.json").exists():
        computation = load(OUT / "RESULT.json")
        closure = {
            "status": "formal_assessment_complete",
            "decision": "Go" if result["quality_pass"] and computation["computational_gates_pass"] else "No-Go",
            "result_sha256": sha256(OUT / "RESULT.json"), "human_evidence_sha256": sha256(evidence),
            "scope": "fixed_v4_short_text_independent_confirmation",
        }
        freeze(OUT / "formal_assessments" / (digest(closure) + ".json"), closure)
        atomic(OUT / "FORMAL_STATUS.json", closure)
    return result


def execute(command, key_file=None):
    bind()
    try:
        with single_writer(OUT):
            if (OUT / "HALT.json").exists() and command != "verify":
                raise IntegrityHalt("permanent_halt_exists")
            if command == "verify":
                verify()
                return 0
            if command == "generate":
                generate(key_file)
            elif command == "score":
                # A budget pause must not turn a dependent scoring job into an integrity failure.
                if not (OUT / "GENERATION_COMPLETE.json").exists():
                    heartbeat("waiting_for_generation", scientific_result=None)
                    return 76
                score()
            else:
                print(json.dumps(collect_reviews()))
    except BUDGET.Pause as exc:
        heartbeat("budget_pause", reason=str(exc), budget=journal().totals(), scientific_result=None)
        return 76  # Release this queue so other independent jobs can proceed.
    except BaseException as exc:
        if isinstance(exc, ResourcePause) or "OutOfMemoryError" in type(exc).__name__ or "CUDA out of memory" in str(exc) or (isinstance(exc, OSError) and exc.errno == 28):
            heartbeat("resource_pause", reason=type(exc).__name__, scientific_result=None)
            return 75
        if command == "collect-human":
            raise  # Invalid review submission is fixable; preserve computation.
        heartbeat("integrity_halt", error_type=type(exc).__name__)
        if not (OUT / "HALT.json").exists():
            freeze(OUT / "HALT.json", {"at": now(), "error_type": type(exc).__name__, "command": command})
        raise
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "generate", "score", "collect-human"))
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    sys.exit(execute(args.command, args.key_file))
