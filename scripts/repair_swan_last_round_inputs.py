"""Versioned, bounded surface repair; never select by detector output."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_swan_blind_eval import save_json
from scripts.prepare_swan_newline_negatives import load_key
from scripts.swan_last_round_inputs import identity, prior_inventory, sha
from src.core.swan_blind_protocol import digest

OLD = ROOT / "experiments/swan_multiview_development1/final_completion1"
OUT = ROOT / "experiments/swan_multiview_development1/final_completion2"
PRIVATE = OUT / "private"
SETTINGS = {"model": "deepseek-flash", "temperature": .9, "max_tokens": 1200,
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"}}
SYSTEM = "Write natural English prose. Return only the requested JSON object."
FRAMEWORK = """Repair one sentence in a short, coherent English account.
Background article: {title}
Background paragraph: {background}
Already accepted sentences: {context}
Original sentence requiring repair: {original}
Surface problem on the previous attempt: {problem}
Write one complete, natural sentence. Aim for exactly 8 whitespace-separated
words; the permitted range is 5 to 12. Prefer a simple subject, verb and object.
Describe a specific concrete event without repeating any accepted sentence.
Use no headings, quotations, numbering, abbreviated names or AMR placeholders.
Use exactly one period, at the very end, and no other . ! ? or newline.
AMR represents meaning; NE means a named entity, N a noun and X an unspecified
concept. Replace placeholders with actual words. Short AMR realization examples:
{examples}
Additional semantic constraint:
{constraint}
Return precisely this JSON shape: {{"sentence":"Eight word sentence ending here."}}
Check the word count before answering. Do not explain your answer."""


def now():
    return datetime.now(timezone.utc).isoformat()


def budget():
    control = json.loads((OUT / "CONTROL.json").read_text())
    return datetime.now(timezone.utc) < datetime.fromisoformat(control["deadline_at"]) \
        and not any((OUT / name).exists() for name in ("HALT.json", "DEADLINE_STOP.json"))


def problems(slot, accepted):
    text = slot.get("text", "")
    reasons = []
    if slot.get("error"):
        reasons.append("call_failure")
    if slot.get("finish_reason") != "stop":
        reasons.append("finish_reason")
    if not 5 <= len(re.findall(r"\S+", text)) <= 12:
        reasons.append("word_count")
    if len(re.findall(r"[.!?]", text)) != 1 or not text.endswith("."):
        reasons.append("terminal_period")
    if "\n" in text or "\r" in text:
        reasons.append("newline")
    if identity(text) in accepted:
        reasons.append("duplicate_sentence")
    return reasons


def prepare():
    if (OUT / "GENERATION_FREEZE.json").exists():
        raise RuntimeError("Repair already frozen")
    PRIVATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name in ("CONTROL.json", "SOURCE_FREEZE.json", "CANDIDATE_SELECTION.json"):
        shutil.copyfile(OLD / name, OUT / name)
    for name in ("HUMAN_DOCUMENTS.json", "GENERATION_ASSIGNMENTS.json", "PROMPT_RESOURCES.json"):
        shutil.copyfile(OLD / "private" / name, PRIVATE / name)
    assignments = json.loads((PRIVATE / "GENERATION_ASSIGNMENTS.json").read_text())["documents"]
    originals = []
    for d in assignments:
        source = OLD / "private/generated" / (d["document_id"] + ".json")
        dest = PRIVATE / "original" / source.name
        dest.parent.mkdir(exist_ok=True, mode=0o700)
        shutil.copyfile(source, dest)
        originals.append({"document_id": d["document_id"], "sha256": sha(dest)})
    prior_text, _, inventory = prior_inventory()
    save_json(PRIVATE / "PRIOR_IDENTITIES.json", {"identities": sorted(prior_text),
                                                 "inventory": inventory})
    save_json(OUT / "GENERATION_FREEZE.json", {
        "schema": "swan-last-round-repair/v2", "created_at": now(),
        "settings": SETTINGS, "system_prompt": SYSTEM, "prompt_framework": FRAMEWORK,
        "implementation_sha256": sha(__file__),
        "protocol_sha256": sha(OUT / "REPAIR_PROTOCOL.md"),
        "input_files_sha256": {name: sha(PRIVATE / name) for name in (
            "HUMAN_DOCUMENTS.json", "GENERATION_ASSIGNMENTS.json", "PROMPT_RESOURCES.json",
            "PRIOR_IDENTITIES.json")},
        "original_documents": originals,
        "source_freeze_sha256": sha(OUT / "SOURCE_FREEZE.json"),
        "candidate_selection_sha256": sha(OUT / "CANDIDATE_SELECTION.json"),
        "predecessor_generation_sha256": sha(OLD / "GENERATION_FREEZE.json"),
        "deadline_at": json.loads((OUT / "CONTROL.json").read_text())["deadline_at"],
        "dispatch_cap": 4000, "attempts_per_failed_slot": 3, "workers": 4,
        "documents": 250, "sentences_per_document": 16,
        "retention": "valid-original-slots-first; later duplicates repaired",
        "selection": "call/surface/duplicate-only; never detector/target-score",
        "negative_constraint": "No additional graph constraint; continue the account naturally.",
        "positive_constraint": "Realize the following AMR structure using actual words:\n{template}",
        "label_policy": "all original generation intents including injection failures",
        "dispatch_journal": "fsync exclusive dispatch before network; no attempt redispatch",
        "old_halt_immutable": True,
    })
    print(json.dumps({"repair_frozen": True, "documents": len(assignments),
                      "new_dispatch_cap": 4000}), flush=True)


def verify():
    freeze = json.loads((OUT / "GENERATION_FREEZE.json").read_text())
    if sha(__file__) != freeze["implementation_sha256"] \
            or sha(OUT / "REPAIR_PROTOCOL.md") != freeze["protocol_sha256"]:
        raise RuntimeError("Frozen repair code/protocol changed")
    for name, expected in freeze["input_files_sha256"].items():
        if sha(PRIVATE / name) != expected:
            raise RuntimeError("Frozen repair input changed")
    for original in freeze["original_documents"]:
        if sha(PRIVATE / "original" / (original["document_id"] + ".json")) != original["sha256"]:
            raise RuntimeError("Frozen original changed")
    return freeze


def generate():
    freeze = verify()
    lock = (PRIVATE / "repair.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    calls = PRIVATE / "calls"
    calls.mkdir(exist_ok=True, mode=0o700)
    resources = json.loads((PRIVATE / "PROMPT_RESOURCES.json").read_text())
    human = json.loads((PRIVATE / "HUMAN_DOCUMENTS.json").read_text())["documents"]
    assignments = json.loads((PRIVATE / "GENERATION_ASSIGNMENTS.json").read_text())["documents"]
    key, mutex = load_key(), threading.Lock()
    dispatched = len(list(calls.glob("*.dispatch.json")))
    committed, retained, repaired = 0, 0, 0

    def one_doc(d):
        nonlocal dispatched, committed, retained, repaired
        dest = PRIVATE / "generated" / (d["document_id"] + ".json")
        if dest.exists():
            with mutex:
                committed += 1
            return
        original = json.loads((PRIVATE / "original" / dest.name).read_text())
        h, accepted, context, slots = human[d["source_index"]], set(), [], []
        for index, old in enumerate(original["slots"]):
            slot = {**old, "provenance": "original", "repair_attempts": 0}
            issues = problems(slot, accepted)
            for attempt in range(3) if issues else ():
                if not budget():
                    slot = {**slot, "error": "deadline", "provenance": "unresolved"}
                    break
                callid = f"{d['document_id']}-{index:02d}-r{attempt+1}"
                dispatch = calls / (callid + ".dispatch.json")
                receipt = calls / (callid + ".receipt.json")
                template = d.get("template_ids", [None]*16)[index]
                constraint = freeze["negative_constraint"] if template is None else \
                    freeze["positive_constraint"].format(template=resources["templates"][template])
                prompt = FRAMEWORK.format(
                    title=h["article_title"], background=h["text"], context=" ".join(context),
                    original=old["text"], problem=", ".join(issues),
                    examples=resources["examples"], constraint=constraint)
                payload = {**SETTINGS, "messages": [
                    {"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
                if receipt.exists():
                    response = json.loads(receipt.read_text())
                elif dispatch.exists():
                    response = {"text": "", "error": "uncertain_prior_dispatch"}
                    save_json(receipt, response)
                else:
                    with mutex:
                        if dispatched >= freeze["dispatch_cap"]:
                            response = {"text": "", "error": "repair_dispatch_cap"}
                            break
                        fd = os.open(dispatch, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                        with os.fdopen(fd, "w") as stream:
                            json.dump({"call_id": callid, "at": now(),
                                       "request_sha256": digest(payload)}, stream)
                            stream.flush()
                            os.fsync(stream.fileno())
                        dispatched += 1
                    request = urllib.request.Request(
                        "https://api.deepseek.com/chat/completions",
                        data=json.dumps(payload).encode(), headers={
                            "Content-Type": "application/json", "Authorization": "Bearer "+key})
                    try:
                        with urllib.request.urlopen(request, timeout=55) as reply:
                            data = json.load(reply)
                        choice = data["choices"][0]
                        content = json.loads(choice["message"]["content"])
                        text = content["sentence"]
                        if not isinstance(text, str):
                            raise ValueError("sentence not string")
                        response = {"text": text.strip(), "finish_reason": choice["finish_reason"],
                                    "response_model": data.get("model"),
                                    "usage": data.get("usage", {}), "at": now()}
                    except Exception as exc:
                        response = {"text": "", "error": type(exc).__name__, "at": now()}
                    save_json(receipt, response)
                slot = {**old, **response, "error": response.get("error"),
                        "finish_reason": response.get("finish_reason"),
                        "provenance": "repair", "repair_attempts": attempt+1}
                issues = problems(slot, accepted)
                if not issues:
                    break
            slot["surface_failure_reasons"] = problems(slot, accepted)
            slots.append(slot)
            if not slot["surface_failure_reasons"]:
                accepted.add(identity(slot["text"]))
                context.append(slot["text"])
        save_json(dest, {**d, "slots": slots})
        with mutex:
            committed += 1
            retained += sum(s["provenance"] == "original" for s in slots)
            repaired += sum(s["provenance"] == "repair" and not s["surface_failure_reasons"]
                            for s in slots)
            save_json(OUT / "GENERATION_HEARTBEAT.json", {
                "at": now(), "state": "running", "documents_committed": committed,
                "generation_dispatches": dispatched, "retained_original_slots": retained,
                "accepted_repaired_slots": repaired, "cap": freeze["dispatch_cap"]})

    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(one_doc, d) for d in assignments]):
            future.result()
    audit()
    save_json(OUT / "GENERATION_HEARTBEAT.json", {
        "at": now(), "state": "complete", "documents_committed": committed,
        "generation_dispatches": dispatched, "cap": freeze["dispatch_cap"]})


def audit():
    freeze = verify()
    assignments = json.loads((PRIVATE / "GENERATION_ASSIGNMENTS.json").read_text())["documents"]
    human = json.loads((PRIVATE / "HUMAN_DOCUMENTS.json").read_text())["documents"]
    prior = set(json.loads((PRIVATE / "PRIOR_IDENTITIES.json").read_text())["identities"])
    docs, seen, counts, slot_counts, words = [], set(), Counter(), Counter(), []
    for d in assignments:
        path = PRIVATE / "generated" / (d["document_id"] + ".json")
        data = json.loads(path.read_text()) if path.exists() else {
            "slots": [{"text": "", "error": "document_not_completed"} for _ in range(16)]}
        accepted, reasons = set(), []
        for slot in data["slots"]:
            issues = problems(slot, accepted)
            slot_counts.update(issues)
            if issues:
                reasons.append("generation_surface_or_call_failure")
            accepted.add(identity(slot["text"]))
            words.append(len(re.findall(r"\S+", slot["text"])))
        if len(data["slots"]) != 16:
            raise RuntimeError("Frozen slot count changed")
        sentences = [s["text"] for s in data["slots"]]
        text = " ".join(sentences)
        tid = identity(text)
        if tid in seen or tid in prior:
            reasons.append("duplicate_or_prior_document")
        seen.add(tid)
        reasons = sorted(set(reasons))
        counts.update(reasons)
        docs.append({**d, "text": text, "sentences": sentences, "eligible": not reasons,
                     "failure_reasons": reasons, "normalized_text_sha256": tid,
                     "target_template_ids": d.get("template_ids")})
    cal = {d["source_article_sha256"] for d in docs+human if d["split"] == "calibration"}
    test = {d["source_article_sha256"] for d in docs+human if d["split"] == "test"}
    if cal & test:
        raise RuntimeError("Source-family leakage")
    save_json(PRIVATE / "DOCUMENTS.json", {"documents": human+docs})
    dispatched = len(list((PRIVATE / "calls").glob("*.dispatch.json")))
    eligible = Counter(f"{d['arm']}:{d['split']}" for d in docs if d["eligible"])
    result = {
        "created_at": now(), "source_family_overlap": 0,
        "generation_dispatches": dispatched, "dispatch_cap_pass": dispatched <= 4000,
        "all_frozen_documents_retained": len(human) == 300 and len(docs) == 250,
        "generated_eligible": sum(d["eligible"] for d in docs),
        "eligible_by_arm_split": dict(eligible), "human_documents": len(human),
        "input_failure_counts": dict(counts), "slot_failure_counts": dict(slot_counts),
        "input_manifest_sha256": sha(PRIVATE / "DOCUMENTS.json"),
        "generation_freeze_sha256": sha(OUT / "GENERATION_FREEZE.json"),
        "new_evaluation_ready": all(eligible[k] >= 19 for k in (
            "negative:calibration", "negative:test", "positive:test")),
        "predecessor_generated_eligible": 0, "predecessor_halt_preserved": True,
        "injection_failure_exclusion": False, "score_screening": False,
        "mean_sentence_words": sum(words)/len(words),
    }
    save_json(OUT / "INPUT_AUDIT.json", result)
    lines = ["# 修复版输入审计", "",
             f"保留300篇人类参考与250个生成意图；生成合格{result['generated_eligible']}/250。",
             f"新增请求{dispatched}/4000。仅按调用、格式、重复修复，未读取检测分数。",
             "", "| 分组 | 合格数 |", "|---|---|"]
    lines.extend(f"| {k} | {v} |" for k, v in sorted(eligible.items()))
    lines += ["", "调用或格式失败继续保留分母；语义目标是否实现不参与排除。",
              "原失败版不改写；校准和测试来源文章交集为0。"]
    (OUT / "INPUT_REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k: result[k] for k in (
        "generated_eligible", "generation_dispatches", "new_evaluation_ready")}), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--phase", choices=("prepare", "generate", "audit"), required=True)
    args = cli.parse_args()
    {"prepare": prepare, "generate": generate, "audit": audit}[args.phase]()
