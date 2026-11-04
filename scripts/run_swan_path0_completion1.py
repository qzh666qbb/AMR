"""Same-source path0 completion: score-blind generation, fresh shared AMRs."""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from swan_final1_common import digest, freeze, load, now, sha256, single_writer
import swan_final1_budget as budget
import run_swan_supplement4 as generation
from swan_supplement4_runtime import ResourcePause, runtime
from swan_supplement1_runtime import IntegrityHalt, content_seed, readiness
from swan_supplement1_sources import WORD, tokenizer

OUT = ROOT / "experiments/swan_path0_completion1"
BIN_CONTRACTS = ((2, 9, 12), (3, 9, 10), (4, 9, 10), (4, 11, 12))
VARIANTS = ("baseline", "semicolon_all", "semicolon_pairs", "comma_all",
            "newline_all", "space_all")
CALL_CAP = 8050
COST_CAP = 25_000_000_000


def bind():
    generation.OUT = OUT
    generation.pilot.OUT = OUT
    budget.CAPS = {"pilot": (CALL_CAP, COST_CAP)}


def heartbeat(stage, **fields):
    from src.core.elastic_v4_path0_inputs import write
    write(OUT / "HEARTBEAT.json", {"at": now(), "stage": stage, **fields})


def verify():
    from src.core.elastic_v4_path0_inputs import verify as verify_inputs
    plan = load(OUT / "FREEZE.json")
    for path, expected in plan["dependencies"].items():
        if sha256(ROOT / path) != expected:
            raise IntegrityHalt("dependency_SHA_changed:" + path)
    verify_inputs(OUT)
    return plan


def read_sources(stage):
    return [json.loads(line) for line in
            (OUT / f"sealed/{stage}.jsonl").read_text().splitlines()]


def read_bank():
    path = generation.pilot.swan.OFFICIAL_ROOT / "amr_bank/banks/amr_bank_100.json"
    records = load(path)
    bank = [item["normalized_template_amr"] for item in records]
    if len(bank) != 100 or any(not isinstance(item, str) or not item for item in bank):
        raise IntegrityHalt("Bank100_schema_changed")
    return bank


def prompt_addition(bucket):
    _, lo, hi = BIN_CONTRACTS[bucket]
    return (
        f"\n\nFixed additional constraints: Write {lo} to {hi} lexical words in "
        "one complete sentence ending with a period. Continue the general topic "
        "of the context, using plausible fictional details when needed. Respect "
        "the AMR structure. Do not output graph notation, placeholders, headings "
        "or explanations."
    )


def assignment(sources, bank):
    return {
        source["allocation_id"]: {
            "length_bin": source["length_bin"],
            "sentences": BIN_CONTRACTS[source["length_bin"]][0],
            "prompt_addition": prompt_addition(source["length_bin"]),
            "slots": [
                {"slot": slot, "template_indices": template_indices(
                    source["allocation_id"], slot, len(bank))}
                for slot in range(BIN_CONTRACTS[source["length_bin"]][0])
            ],
        } for source in sources
    }


def template_indices(allocation_id, slot, count):
    rng = random.Random(content_seed("template", allocation_id, slot))
    return [rng.randrange(count) for _ in range(10)]


def space_check():
    if shutil.disk_usage(OUT).free < 1024**3:
        raise ResourcePause("disk_headroom_below_1GiB")


def generate(key_file):
    verify()
    bind()
    if (OUT / "GENERATION_COMPLETE.json").exists():
        review_packet()
        return verify_generation()
    bank = read_bank()
    sources = read_sources("positive_source")
    frozen_assignment = load(OUT / "GENERATION_ASSIGNMENT.json")
    if assignment(sources, bank) != frozen_assignment:
        raise IntegrityHalt("target_assignment_changed")
    journal = generation.ReplayJournal(
        OUT, "pilot", CALL_CAP, None, sha256(OUT / "FREEZE.json"), parents=[])
    if journal.uncertain:
        raise IntegrityHalt("uncertain_call_blocks_resume")
    key = generation.pilot.credential.read_credential(key_file)
    journal.send = lambda body: budget.transport(body, key)
    token = tokenizer()
    with runtime(OUT) as module:
        if not (OUT / "GENERATION_READINESS.json").exists():
            freeze(OUT / "GENERATION_READINESS.json", readiness(module, bank))
        original_evaluate = module.evaluate
        for index, source in enumerate(sources):
            space_check()
            path = OUT / "private/documents" / (source["allocation_id"] + ".json")
            if path.exists():
                continue
            count, lo, hi = BIN_CONTRACTS[source["length_bin"]]
            generation.pilot.LENGTH = prompt_addition(source["length_bin"])

            def evaluate(text, target, candidate_id):
                result = original_evaluate(text, target, candidate_id)
                if not lo <= len(WORD.findall(text)) <= hi:
                    result.update(accepted=False, rejection="frozen_bin_length_gate")
                return result

            module.evaluate = evaluate
            sentences, slot_hashes, failure = [], {}, None
            for slot in range(count):
                try:
                    record = generation.run_slot(
                        module, source, slot,
                        source["generation_prompt"] + " " + " ".join(sentences),
                        bank, journal)
                except budget.DocumentFailure:
                    # Unknown provider usage is never retried or treated as a rejection.
                    if journal.uncertain:
                        raise IntegrityHalt("uncertain_call_blocks_resume")
                    raise
                slot_path = OUT / "private/slots" / (
                    f"{source['allocation_id']}-slot-{slot:02d}.json")
                slot_hashes[str(slot_path.relative_to(OUT))] = sha256(slot_path)
                sentences.append(record["text"])
                if not record["accepted"]:
                    failure = "slot_attempts_exhausted"
                    break
            text = " ".join(sentences)
            lower, upper = ((17, 24), (25, 32), (33, 40), (41, 48))[source["length_bin"]]
            if failure is None and (token(text) != sentences or
                                    not lower <= len(text.split()) <= upper):
                failure = "whole_document_sentence_or_word_gate"
            freeze(path, {
                "allocation_id": source["allocation_id"],
                "source_window_sha256": source["window_sha256"],
                "contract_sha256": sha256(OUT / "FREEZE.json"),
                "length_bin": source["length_bin"], "text": text,
                "sentences": sentences, "automatic_qualified": failure is None,
                "failure": failure, "slot_sha256": slot_hashes,
                "human_qualified": None, "detector_feedback_used": False,
            })
            heartbeat("generating", completed_documents=index + 1, total=50,
                      budget=journal.totals())
    documents = {
        str(path.relative_to(OUT)): sha256(path)
        for path in sorted((OUT / "private/documents").glob("*.json"))
    }
    if len(documents) != 50:
        raise IntegrityHalt("generation_denominator_incomplete")
    freeze(OUT / "GENERATION_COMPLETE.json", {
        "contract_sha256": sha256(OUT / "FREEZE.json"),
        "documents": documents, "budget": journal.totals(),
        "automatic_qualified": sum(load(OUT / path)["automatic_qualified"] for path in documents),
        "all_attempts_before_detector": True, "independent_human_submissions": 0,
    })
    review_packet()
    heartbeat("generation_complete", completed_documents=50, total=50)
    return verify_generation()


def verify_generation():
    complete = load(OUT / "GENERATION_COMPLETE.json")
    if complete["contract_sha256"] != sha256(OUT / "FREEZE.json"):
        raise IntegrityHalt("generation_contract_changed")
    if len(complete["documents"]) != 50:
        raise IntegrityHalt("generation_denominator_changed")
    bind()
    journal = generation.ReplayJournal(
        OUT, "pilot", CALL_CAP, None, sha256(OUT / "FREEZE.json"), parents=[])
    if journal.uncertain:
        raise IntegrityHalt("uncertain_generation")
    for path, expected in complete["documents"].items():
        if sha256(OUT / path) != expected:
            raise IntegrityHalt("generated_document_changed")
        for slot_path, slot_sha in load(OUT / path)["slot_sha256"].items():
            if sha256(OUT / slot_path) != slot_sha:
                raise IntegrityHalt("generated_slot_changed")
            for call_id, attempt_sha in load(OUT / slot_path)["attempt_file_hashes"].items():
                if sha256(OUT / "private/attempts" / (call_id + ".json")) != attempt_sha:
                    raise IntegrityHalt("generated_attempt_changed")
    return complete


def review_packet():
    folder = OUT / "private/human_review"
    sources = {s["allocation_id"]: s for s in read_sources("positive_source")}
    documents = [load(p) for p in sorted((OUT / "private/documents").glob("*.json"))]
    order = sorted(documents, key=lambda d: digest(["review-order", d["allocation_id"]]))
    mapping = {}
    items = []
    for document in order:
        item_id = digest(["path0-blind", document["allocation_id"]])[:20]
        mapping[item_id] = document["allocation_id"]
        items.append({"item_id": item_id,
                      "context": sources[document["allocation_id"]]["generation_prompt"],
                      "text": document["text"]})
    freeze(folder / "PACKET.json", {"items": items, "reviewers_required": 2,
                                   "scores_and_targets_hidden": True})
    freeze(OUT / "private/HUMAN_LINKAGE.json", mapping)
    for reviewer in ("reviewer1", "reviewer2"):
        path = folder / f"{reviewer}.csv"
        if not path.exists():
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "item_id", "grammar_pass", "coherence_pass", "no_AMR_leak",
                    "reviewer_id", "is_human", "notes"])
                writer.writeheader()
                writer.writerows({"item_id": item["item_id"]} for item in items)


def detector_args():
    from src.core.bank_detector import add_runtime_arguments
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    args = parser.parse_args([])
    args.output_root = OUT
    args.cache_db = str(OUT / "private/fresh_detector.db")
    args.capitalization_gap_words, args.long_gap_stride_words = 12, 40
    args.model_batch_size, args.request_batch_size = 16, 128
    return args


def miss_row(source, variant, binding, reason):
    return {
        "binding": binding, "allocation_id": source["allocation_id"],
        "row_id": source["allocation_id"] + "__" + variant, "variant": variant,
        "stage": "positive_source", "word_count": 0, "length_bin": source["length_bin"],
        "generation_failure": reason,
        "arms": {name: {"status": "abstain", "statistic": None,
                        "selected_invalid_edges": 0, "abstain_reason": reason}
                 for name in ("control", "path0")},
    }


def positive_summary(rows, calibrators):
    from src.core.elastic_v4_path0_development import ALPHAS, rate
    if len(rows) != 300 or len({r["allocation_id"] for r in rows}) != 50:
        raise IntegrityHalt("positive_scoring_denominator")
    output = {}
    for arm, calibrator in calibrators.items():
        curves = {}
        for alpha in ALPHAS:
            hits = {variant: set() for variant in VARIANTS}
            abstains = {variant: 0 for variant in VARIANTS}
            for row in rows:
                result = row["arms"][arm]
                if result["status"] == "abstain":
                    abstains[row["variant"]] += 1
                elif calibrator.calibrate(result["statistic"], row["word_count"], alpha).is_watermarked:
                    hits[row["variant"]].add(row["allocation_id"])
            baseline = hits["baseline"]
            survival = {variant: rate(len(baseline & hits[variant]), len(baseline))
                        for variant in VARIANTS[1:]}
            joint = set.intersection(*(hits[variant] for variant in VARIANTS))
            curves[str(alpha)] = {
                "full_denominator_TPR": {variant: rate(len(hits[variant]), 50)
                                         for variant in VARIANTS},
                "conditional_survival": survival,
                "worst_condition_survival": min(
                    item["rate"] for item in survival.values()) if baseline else None,
                "all_five_survival": rate(len(joint), len(baseline)),
                "abstain_full_denominator": abstains,
            }
        output[arm] = {"primary": curves["0.01"], "alpha_curve": curves}
    return output


def score():
    from src.core import elastic_v4_path0_development as dev
    from src.core.elastic_v4_path0_inputs import variants, write
    import torch
    verify()
    generated = verify_generation()
    review_packet()
    if not torch.cuda.is_available():
        raise IntegrityHalt("CUDA_required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false")
    if torch.cuda.mem_get_info()[0] < 3072 * 1024**2:
        raise ResourcePause("parser_admission_requires_3072MiB_free")
    args = detector_args()
    space_check()
    freeze(OUT / "P0.json", dev.run_p0(args.bank))
    arms, recovery = dev.build_arms(args)
    model = recovery.backend._load()
    if any(p.dtype != torch.float32 for p in model.model.parameters()):
        raise IntegrityHalt("detector_parser_not_FP32")
    payload = dev.runtime_payload(args, arms)
    payload["completion_contract_sha256"] = sha256(OUT / "FREEZE.json")
    payload["generation_sha256"] = sha256(OUT / "GENERATION_COMPLETE.json")
    payload = json.loads(json.dumps(payload))
    freeze(OUT / "RUNTIME_MANIFEST.json", payload)
    runtime_sha = sha256(OUT / "RUNTIME_MANIFEST.json")
    all_rows = {}
    calibrators = None
    for stage in ("calibration", "null", "positive_source"):
        sources = read_sources(stage)
        rows = []
        for source in sources:
            document = None if stage != "positive_source" else load(
                OUT / "private/documents" / (source["allocation_id"] + ".json"))
            if stage == "calibration":
                texts = {"baseline": source["text"]}
            elif document is not None and not document["automatic_qualified"]:
                texts = {variant: "" for variant in VARIANTS}
            else:
                texts = variants(document["text"] if document else source["text"])
            for variant, text in texts.items():
                space_check()
                binding = {
                    "runtime_sha256": runtime_sha,
                    "input_manifest_sha256": sha256(OUT / "INPUT_MANIFEST.json"),
                    "allocation_id": source["allocation_id"],
                    "source_window_sha256": source["window_sha256"],
                    "variant": variant, "text_sha256": dev.sha(text.encode()),
                }
                path = OUT / "results/rows" / (source["allocation_id"] + "__" + variant + ".json")
                if document is not None and not document["automatic_qualified"]:
                    row = miss_row(source, variant, binding, document["failure"])
                    freeze(path, row)
                else:
                    row = dev.checkpoint(path, binding)
                    if row is None:
                        heartbeat(stage, completed=len(rows), total=len(sources) * len(texts),
                                  active_row=path.stem)
                        row = dev.score_document(args, arms, recovery, source, variant, text, binding)
                        write(path, row)
                rows.append(row)
                heartbeat(stage, completed=len(rows), total=len(sources) * len(texts),
                          parser_unique_spans=len(recovery.records))
        all_rows[stage] = rows
        if stage == "calibration":
            calibrators = {name: dev.fit(args, name, detector, rows, sources, runtime_sha)
                           for name, detector in arms.items()}
    if len(all_rows["calibration"]) != 1000 or len(all_rows["null"]) != 1200:
        raise IntegrityHalt("null_scoring_denominator")
    null = dev.summarize_null(all_rows["null"], calibrators)
    # Replace inherited pending-generation metadata with the actual completion state.
    null.update(status="null_complete", positive_generated=50,
                decision="see_combined_development_result")
    null["limitations"] = ["development_only", "independent_human_quality_pending",
                           "normal_length_unconfirmed"]
    freeze(OUT / "NULL_RESULT.json", null)
    positive = positive_summary(all_rows["positive_source"], calibrators)
    result = {
        "status": "computational_complete_pending_human_quality",
        "decision": ("insufficient_automatic_positives" if generated["automatic_qualified"] < 40
                     else "pending_independent_human_quality"),
        "automatic_qualified": generated["automatic_qualified"], "positive_slots": 50,
        "generation_budget": generated["budget"], "human_qualified": None,
        "calibration_rows": 1000, "null_rows": 1200, "positive_rows": 300,
        "null": null, "positive": positive, "binding_runtime_sha256": runtime_sha,
        "parser_recovery_summary": recovery.summary(),
        "limitations": ["development_only", "automatic_quality_not_human_truth",
                        "normal_length_unconfirmed"],
    }
    freeze(OUT / "RESULT.json", result)
    row_hashes = {str(p.relative_to(OUT)): sha256(p)
                  for p in sorted((OUT / "results/rows").glob("*.json"))}
    freeze(OUT / "AUDIT.json", {
        "status": "computational_complete", "formal_quality_closure": False,
        "generation_sha256": sha256(OUT / "GENERATION_COMPLETE.json"),
        "result_sha256": sha256(OUT / "RESULT.json"), "rows": row_hashes,
        "row_count": len(row_hashes), "source_replacement": False,
        "selected_invalid_edges": sum(r["arms"][a]["selected_invalid_edges"]
                                      for rows in all_rows.values() for r in rows for a in arms),
    })
    lines = ["# path0 两臂开发补全", "",
             "计算阶段完成；独立真人质量尚未回收，不能据此给正式 Go。",
             f"自动资格 {generated['automatic_qualified']}/50；失败保留在 TPR 的50篇分母。",
             "", "|条件|对照 Null|path0 Null|对照检出|path0检出|", "|---|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        fp = [null["arms"][a]["primary"]["conditions"][variant]["successes"] for a in arms]
        tp = [positive[a]["primary"]["full_denominator_TPR"][variant]["successes"] for a in arms]
        lines.append(f"|{variant}|{fp[0]}/200|{fp[1]}/200|{tp[0]}/50|{tp[1]}/50|")
    lines += ["", "RESULT.json 包含全部 alpha 曲线、置信区间、条件存活和五条件联合存活。",
              "本次为短文本同源开发后继，不能充作独立确认或正常长度证据。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    freeze(OUT / "TERMINAL.json", {"status": result["status"],
                                 "result_sha256": sha256(OUT / "RESULT.json"),
                                 "audit_sha256": sha256(OUT / "AUDIT.json")})
    heartbeat(result["status"], completed_rows=2500, total_rows=2500)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "generate", "score"))
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    bind()
    try:
        with single_writer(OUT):
            if args.command != "verify" and (OUT / "HALT.json").exists():
                raise IntegrityHalt("permanent_halt_exists")
            if args.command == "verify":
                verify()
                print(json.dumps({"status": "frozen_inputs_verified", "generation_calls": 0}))
            elif args.command == "generate":
                generate(args.key_file)
            else:
                score()
    except BaseException as exc:
        if isinstance(exc, ResourcePause) or "OutOfMemoryError" in type(exc).__name__ \
                or "CUDA out of memory" in str(exc) or (isinstance(exc, OSError) and exc.errno == 28):
            heartbeat("resource_pause", reason=type(exc).__name__, scientific_result=None)
            return 75
        heartbeat("integrity_halt", error_type=type(exc).__name__)
        if not (OUT / "HALT.json").exists():
            freeze(OUT / "HALT.json", {"at": now(), "error_type": type(exc).__name__,
                                     "command": args.command})
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
