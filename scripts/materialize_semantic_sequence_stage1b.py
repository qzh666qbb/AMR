"""Deterministic variants, blind inputs, duplicate audit, and human-review packets."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from run_semantic_sequence_stage1b import (
    BANK, PARAMETERS, ROOT, RUN, SEED, atomic_bytes, atomic_json, carrier_messages,
    digest_bytes, encoded, jsonl, normalize, now, paraphrase_messages, read_jsonl, sha,
)

CONDITIONS = ("clean", "paraphrase", "merge", "split", "delete-2", "insert-2", "swap-1")


def rng_for(doc_id, condition):
    seed = int(digest_bytes(f"{SEED}|{doc_id}|{condition}".encode()), 16)
    return random.Random(seed)


def assemble(items):
    """Keep empty planned slots and construction spans; these are not semantic gold."""
    text = ""
    spans = []
    for origin, slot, piece in items:
        if spans:
            text += " "
        piece = piece.strip()
        start = len(text)
        text += piece
        spans.append({"origin": origin, "slot": slot, "start": start, "end": len(text)})
    return text, spans


def variant(doc, arm, condition):
    source = doc["arms"][arm]
    rows = source["slots"]
    assert [r["slot"] for r in rows] == list(range(8)), "Never compact failed slots"
    items = [(arm, r["slot"], r["text"]) for r in rows]
    operations = {}
    failures = []
    randomizer = rng_for(doc["doc_id"], condition)
    if condition == "paraphrase":
        return source["paraphrase"], [], {
            "semantic_mapping": "PENDING_INDEPENDENT_HUMAN_REVIEW",
        }, failures
    if condition == "merge":
        for index in range(0, 8, 2):
            origin, slot, text = items[index]
            changed, n = re.subn(r"[.!?]+(?=[\"')\]]*\s*$)", ",", text.strip())
            if n != 1:
                failures.append({"slot": slot, "reason": "no_terminal_boundary_to_replace"})
            items[index] = (origin, slot, changed)
        operations["paired_slots"] = [[i, i + 1] for i in range(0, 8, 2)]
    elif condition == "split":
        for index, (origin, slot, text) in enumerate(items):
            tokens = list(re.finditer(r"\S+", text))
            if len(tokens) < 2:
                failures.append({"slot": slot, "reason": "fewer_than_two_whitespace_words"})
                continue
            end = tokens[len(tokens) // 2 - 1].end()
            if text[:end].endswith((".", "!", "?")):
                failures.append({"slot": slot, "reason": "midpoint_already_has_sentence_boundary"})
            items[index] = (origin, slot, text[:end] + "." + text[end:])
    elif condition == "delete-2":
        removed = sorted(randomizer.sample(range(8), 2))
        items = [row for row in items if row[1] not in removed]
        operations["deleted_slots"] = removed
    elif condition == "insert-2":
        negatives = doc["arms"]["unwatermarked"]["slots"]
        picked = randomizer.sample(range(8), 2)
        gaps = sorted(randomizer.sample(range(9), 2))
        additions = dict(zip(gaps, picked))
        mixed = []
        for gap in range(9):
            if gap in additions:
                nslot = additions[gap]
                mixed.append(("paired_unwatermarked", nslot, negatives[nslot]["text"]))
            if gap < 8:
                mixed.append(items[gap])
        items = mixed
        operations["insertions"] = [{"original_gap": gap, "negative_slot": additions[gap]}
                                   for gap in gaps]
    elif condition == "swap-1":
        index = randomizer.randrange(7)
        items[index], items[index + 1] = items[index + 1], items[index]
        operations["swapped_slots"] = [index, index + 1]
    elif condition != "clean":
        raise ValueError(condition)
    text, spans = assemble(items)
    return text, spans, operations, failures


def duplicate_groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        norm = normalize(row["text"])
        if norm:
            grouped[digest_bytes(norm.encode())].append({k: v for k, v in row.items() if k != "text"})
    return [{"normalized_sha256": digest, "occurrences": group}
            for digest, group in sorted(grouped.items()) if len(group) > 1]


def is_cross_split(group):
    return len({r["split"] for r in group["occurrences"] if r.get("split")}) > 1


def duplicate_audit(documents):
    carriers, paragraphs = [], []
    for doc in documents:
        for arm, source in doc["arms"].items():
            base = {"doc_id": doc["doc_id"], "split": doc["split"], "arm": arm}
            carriers.extend([dict(base, slot=r["slot"], text=r["text"]) for r in source["slots"]])
            paragraphs.extend([dict(base, condition="clean", text=source["text"]),
                               dict(base, condition="paraphrase", text=source["paraphrase"])])
    old_carriers, old_paragraphs = set(), set()
    for part in ("dataset5", "dataset5_paired_neg"):
        path = ROOT / f"experiments/swan_multiview_development1/{part}/documents.json"
        for doc in json.loads(path.read_text())["documents"]:
            old_paragraphs.add(normalize(doc["text"]))
            old_carriers.update(normalize(s["text"]) for s in doc["slots"])
    carrier_groups = duplicate_groups(carriers)
    paragraph_groups = duplicate_groups(paragraphs)
    old_hits = []
    for level, rows, old in (("carrier", carriers, old_carriers),
                             ("paragraph", paragraphs, old_paragraphs)):
        for row in rows:
            if normalize(row["text"]) and normalize(row["text"]) in old:
                old_hits.append(dict({k: v for k, v in row.items() if k != "text"}, level=level,
                                     normalized_sha256=digest_bytes(normalize(row["text"]).encode())))
    cross = [dict(group, level=level)
             for level, groups in (("carrier", carrier_groups), ("paragraph", paragraph_groups))
             for group in groups if is_cross_split(group)]
    return {
        "normalization": "case preserved; whitespace collapsed and stripped",
        "carrier_duplicate_groups": carrier_groups,
        "paragraph_duplicate_groups": paragraph_groups,
        "cross_development_readout_groups": cross, "matches_old_corpus": old_hits,
        "unique_nonempty_carrier_texts": len({normalize(r["text"]) for r in carriers if normalize(r["text"])}),
        "empty_carriers": sum(not normalize(r["text"]) for r in carriers),
        "independent_readout_text_eligibility": "INVALID_DUPLICATION" if cross else "NO_EXACT_CROSS_SPLIT_DUPLICATION",
        "qualification": "Exact-text audit only; not evidence of semantic independence or human quality.",
    }


ANNOTATION_GUIDE = """# 独立双人核验说明

本目录包含新语料，保持私有。两位核验者分别完成自己的文件；不得互看答案。
当前所有答案均为空，机器没有代替人工判定。

## 第一步：只看文本

先向核验者提供各自 `rater_*/stage1_text_only.jsonl`。每行是一篇源文本及其复述，
不含模板、符号、模型预测或方法名称，也不含开发/读出划分。

在 `source_propositions` 与 `paraphrase_propositions` 中标出所有可核验命题：
`event_id`、`proposition`、`anchor_spans`（Python Unicode 字符的左闭右开区间）。
不要假设恰好八个事件。一个载体可能未实现、含多个事件或没有明确事件。
在 `mappings` 中填源事件到复述事件的一对多/多对一关系及 preserved/omitted/changed/
uncertain；在 `added_events` 记录新增事件，填写顺序变化和置信说明。

每人先完成固定文件顺序的前 12 个事件核验，记录 `timing_pilot` 的起止时间与事件数，
单独估算人工工时。无匹配不能通过移动 span 或猜测目标符号补齐。

## 第二步：核对目标结构

协调者保存两人第一步答案的 SHA256 后，才发放
`coordinator_only/stage2_templates_sealed.jsonl` 的独立副本。
每行对应一个计划载体，含目标图及构造片段位置，共 192 行、384 个 clean/paraphrase
核验项/人。构造片段位置不等于真实事件锚点；复述映射必须来自第一步。

填写 clean/paraphrase 的 `realized`（yes/no/uncertain）、`event_anchor_spans`、
`mapped_proposition_ids`、遗漏/新增/命题改变/顺序交换和原因。
若文本为空、未实现或不能判断，保留计划位置，按规则计失败。
这里只判定图角色与关系是否实现，不参考分类器、S2Match 或实验表现。

## 仲裁与审计

两人完成后先计算分歧率及 uncertain 比例，再逐项仲裁，保留两份原始答案和仲裁理由。
`coordinator_only/adjudication_empty.jsonl` 是空仲裁模板。协调者须维护第一、二阶段和
仲裁文件的 SHA；不得只保留一致子集。盲预测封存前不得把目标标签送入提取器。

48 篇原文及其复述已经生成，不代表 G1 可控性通过；未完成双人核验前不填 Go。
"""


def annotation_packets(run, documents, assignments):
    base = run / "private" / "annotation"
    bank = json.loads(BANK.read_text())
    first, second, keys = [], [], []
    order = list(documents)
    random.Random(SEED + 111).shuffle(order)
    for doc in order:
        doc_id = doc["doc_id"]
        anon = "review-" + digest_bytes(f"{SEED}|review|{doc_id}".encode())[:16]
        pos = doc["arms"]["positive"]
        clean, spans = assemble([("positive", s["slot"], s["text"]) for s in pos["slots"]])
        assert clean == pos["text"]
        keys.append({"review_id": anon, "doc_id": doc_id})
        first.append({
            "review_id": anon, "source_text": clean, "paraphrase_text": pos["paraphrase"],
            "source_sha256": pos["text_sha256"], "paraphrase_sha256": pos["paraphrase_sha256"],
            "source_propositions": [], "paraphrase_propositions": [], "mappings": [],
            "added_events": [], "order_changes": [], "reviewer_id": None, "completed": False,
            "timing_pilot": None,
        })
        for slot in assignments[doc_id]["slots"]:
            i = slot["slot"]
            empty = {"realized": None, "event_anchor_spans": [], "mapped_proposition_ids": [],
                     "omitted": None, "added": None, "proposition_changed": None,
                     "order_changed": None, "reason": None}
            second.append({
                "review_id": anon, "carrier_id": f"{anon}-c{i}",
                "source_text": clean, "paraphrase_text": pos["paraphrase"],
                "source_carrier_span_construction_only": [spans[i]["start"], spans[i]["end"]],
                "target_amr": bank[slot["intended_template"]]["normalized_template_amr"],
                "clean": dict(empty), "paraphrase": dict(empty),
                "reviewer_id": None, "stage1_answers_sha256": None, "completed": False,
            })
    for rater in ("rater_1", "rater_2"):
        jsonl(base / rater / "stage1_text_only.jsonl", first)
    jsonl(base / "coordinator_only" / "stage2_templates_sealed.jsonl", second)
    jsonl(base / "coordinator_only" / "review_identity_map.jsonl", keys)
    jsonl(base / "coordinator_only" / "adjudication_empty.jsonl", [
        {"carrier_id": row["carrier_id"], "rater_1_sha256": None, "rater_2_sha256": None,
         "clean_final": None, "paraphrase_final": None, "disagreements": [],
         "adjudicator_id": None, "reason": None, "completed": False} for row in second])
    atomic_bytes(base / "README.md", ANNOTATION_GUIDE.encode())
    return {"text_pairs_per_reviewer": len(first), "planned_carriers_per_reviewer": len(second),
            "clean_plus_paraphrase_judgments_per_reviewer": len(second) * 2,
            "reviewers_required": 2, "completed_human_judgments": 0}


def materialize(run):
    run = Path(run)
    if not (run / "GENERATION_COMPLETE.json").exists() or (run / "GENERATION_HALT.json").exists():
        raise RuntimeError("Generation is incomplete or halted")
    if (run / "DATA_READY.json").exists():
        raise RuntimeError("Artifacts already sealed; verify rather than overwrite")
    freeze = json.loads((run / "FREEZE_1B.json").read_text())
    for relative, expected in freeze["frozen_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError("Frozen input SHA mismatch")
    records = sorted((run / "private/calls").glob("*.response.json"))
    assert len(records) == 432
    responses = [json.loads(p.read_text()) for p in records]
    assert all(r["completion_received"] for r in responses)
    assert len({r["response_id"] for r in responses if r["response_id"]}) == 432
    assert Counter(r["kind"] for r in responses) == Counter(carrier=384, paraphrase=48)
    for path, response in zip(records, responses):
        prefix = path.name.removesuffix(".response.json")
        provider = path.with_name(prefix + ".provider.json")
        assert sha(provider) == response["raw_sha256"], "Raw provider payload changed"
        payload = json.loads(provider.read_text())
        content = payload["choices"][0]["message"].get("content")
        assert response["text"] == (content if isinstance(content, str) else "")
        intent = json.loads(path.with_name(prefix + ".intent.json").read_text())
        assert digest_bytes(encoded(intent["body"])) == response["request_sha256"]
        assert intent["dispatched_at"] > freeze["frozen_at"]
    documents = [json.loads((run / f"private/documents/d{i:02d}.json").read_text()) for i in range(24)]
    assignments = read_jsonl(run / "private/ASSIGNMENTS.jsonl")
    prompts = read_jsonl(run / "private/PROMPTS.jsonl")
    bank = json.loads(BANK.read_text())
    by_call = {r["call_id"]: r for r in responses}
    for doc in documents:
        d = doc["doc_id"]
        assert doc["split"] == prompts[d]["split"] == assignments[d]["split"]
        for arm, source in doc["arms"].items():
            prior = []
            for slot in source["slots"]:
                response = by_call[f"d{d:02d}-{arm}-s{slot['slot']}"]
                assert slot["text"] == response["text"]
                target = assignments[d]["slots"][slot["slot"]]["intended_template"]
                template = bank[target]["normalized_template_amr"] if arm == "positive" else None
                messages = carrier_messages(prompts[d]["prompt"], " ".join(prior), slot["slot"], template)
                assert digest_bytes(encoded(dict(PARAMETERS, messages=messages))) == response["request_sha256"]
                prior.append(slot["text"].strip())
            assert source["text"] == " ".join(prior)
            assert source["text_sha256"] == digest_bytes(source["text"].encode())
            response = by_call[f"d{d:02d}-{arm}-paraphrase"]
            assert source["paraphrase"] == response["text"]
            assert source["paraphrase_sha256"] == digest_bytes(source["paraphrase"].encode())
            body = dict(PARAMETERS, messages=paraphrase_messages(source["text"]))
            assert digest_bytes(encoded(body)) == response["request_sha256"]
    blind, evaluator, word_counts, operation_failures = [], [], [], []
    for doc in documents:
        for arm in ("positive", "unwatermarked"):
            for condition in (CONDITIONS if arm == "positive" else CONDITIONS[:2]):
                text, spans, operations, failures = variant(doc, arm, condition)
                anon = "input-" + digest_bytes(f"{SEED}|{doc['doc_id']}|{arm}|{condition}".encode())[:24]
                text_sha = digest_bytes(text.encode())
                blind.append({"anonymous_id": anon, "text": text, "text_sha256": text_sha})
                evaluator.append({
                    "anonymous_id": anon, "text_sha256": text_sha, "doc_id": doc["doc_id"],
                    "split": doc["split"], "domain": doc["domain"], "arm": arm, "condition": condition,
                    "source_sha256": doc["arms"][arm]["text_sha256"],
                    "construction_spans_not_semantic_gold": spans, "operations": operations,
                    "operation_failures": failures,
                    "planned_sequence": assignments[doc["doc_id"]]["slots"] if arm == "positive" else None,
                    "planned_event_denominator": (6 if condition == "delete-2" else 8) if arm == "positive" else None,
                    "semantic_gold_status": "PENDING_HUMAN_REVIEW",
                })
                if failures:
                    operation_failures.append({"anonymous_id": anon, "failures": failures})
            word_counts.append({"doc_id": doc["doc_id"], "arm": arm,
                                "clean_words": len(doc["arms"][arm]["text"].split()),
                                "paraphrase_words": len(doc["arms"][arm]["paraphrase"].split())})
    assert len(blind) == 216 and len({r["anonymous_id"] for r in blind}) == 216
    blind.sort(key=lambda r: r["anonymous_id"])
    jsonl(run / "private/BLIND_INPUTS.jsonl", blind)
    jsonl(run / "private/EVALUATOR_MANIFEST.jsonl", evaluator)
    duplication = duplicate_audit(documents)
    atomic_json(run / "DUPLICATE_AUDIT.json", duplication)
    annotations = annotation_packets(run, documents, assignments)
    carrier_responses = [r for r in responses if r["kind"] == "carrier"]
    audit = {
        "created_at": now(), "generation_status": "COMPLETE_NOT_SCORE_FILTERED",
        "document_pairs": 24, "original_documents": 48, "paraphrases": 48,
        "planned_carriers": 384, "retained_carrier_completions": len(carrier_responses),
        "total_completions": len(responses), "automatic_retries": 0,
        "unique_provider_response_ids": len({r["response_id"] for r in responses}),
        "request_chain_raw_response_and_document_sha_audit": "PASS",
        "variants": 216, "positive_variants": 168, "unwatermarked_variants": 48,
        "finish_reason_counts": dict(Counter(r["finish_reason"] for r in responses)),
        "surface_failures_by_arm": {
            arm: {key: sum(not r["surface_audit"][key] for r in carrier_responses if f"-{arm}-" in r["call_id"])
                  for key in ("nonempty", "target_length_met", "single_sentence_heuristic", "terminal_punctuation")}
            for arm in ("positive", "unwatermarked")},
        "surface_checks_are_not_independent_semantic_quality": True,
        "operation_failures": operation_failures, "word_counts": word_counts,
        "annotation": annotations, "realization_rate": None, "paraphrase_fidelity_rate": None,
        "stage1b_go_no_go": "PENDING_HUMAN_LABELS_AND_BLIND_INFERENCE",
        "text_eligibility": duplication["independent_readout_text_eligibility"],
        "amr_parser_calls": 0, "embedding_calls": 0, "s2match_calls": 0,
    }
    atomic_json(run / "QUALITY_AUDIT.json", audit)
    static_files = [run / "FREEZE_1B.json", run / "GENERATION_COMPLETE.json",
                    run / "DUPLICATE_AUDIT.json", run / "QUALITY_AUDIT.json", Path(__file__)]
    static_files += sorted(p for p in (run / "private").rglob("*")
                           if p.is_file() and p.suffix in (".json", ".jsonl", ".md")
                           and p.name not in ("process.json",))
    manifest = "".join(f"{sha(p)}  {p.relative_to(ROOT)}\n" for p in sorted(static_files))
    atomic_bytes(run / "MANIFEST_GENERATION.sha256", manifest.encode())
    atomic_json(run / "DATA_READY.json", {
        "created_at": now(), "status": "TEXTS_AND_ANNOTATION_PACKETS_READY",
        "variants": 216, "generation_manifest_sha256": sha(run / "MANIFEST_GENERATION.sha256"),
        "blind_input_sha256": sha(run / "private/BLIND_INPUTS.jsonl"),
        "duplicate_audit_sha256": sha(run / "DUPLICATE_AUDIT.json"),
        "human_labels_complete": False, "blind_inference_complete": False,
    })
    heartbeat = json.loads((run / "heartbeat.json").read_text())
    heartbeat.update({
        "status": "GENERATION_COMPLETE", "updated_at": now(), "variants_ready": 216,
        "status_reconciliation": "Derived from sealed completion record and verified 432-response ledger; periodic heartbeat can race with terminal status.",
    })
    atomic_json(run / "heartbeat.json", heartbeat)
    print(json.dumps({"status": "TEXTS_AND_ANNOTATION_PACKETS_READY",
                      "completions": 432, "variants": 216,
                      "cross_split_duplicate_groups": len(duplication["cross_development_readout_groups"]),
                      "old_corpus_matches": len(duplication["matches_old_corpus"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args()
    os.umask(0o077)
    materialize(args.run)
