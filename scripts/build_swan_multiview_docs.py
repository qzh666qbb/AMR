"""构造多视图水印数据集：长文 + 短片段载体（自带标签）。

设计依据见 generation_probe1/FINDINGS.md：
- 载体只对 ≤12 词短片段有效，因此长文由短句槽位组成；
- 模板可二分，只用预筛通过的模板；
- 载体位置与模板 ID 在注入时已知，属于自带标签，不需人工从零标注。

本脚本产出正例文本；攻击变体与负例另行构造。不写入任何既有实验目录。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party/swan_s2match_runtime_711a231d"))

from probe_swan_multitemplate_yield import (  # noqa: E402
    BANK, EXAMPLES, GLOVE, THRESHOLD, MODEL, call_api, build_prompt, system_prompt,
    surface_gate, strip_snt, normalize_to_template,
)

DEFAULT_OUT = ROOT / "experiments/swan_multiview_development1/dataset1"
MIN_DOC_WORDS, MAX_DOC_WORDS = 100, 180

VAGUE_WORDS = frozenset({
    "device", "thing", "something", "object", "entity", "term", "item",
    "concept", "stuff", "person", "idea", "reason", "way", "kind", "type",
})
CONCRETE_INSTRUCTION = (
    "\n\nAdditional constraints for natural, substantive text:\n"
    "- Use specific, concrete vocabulary suited to {domain} (real names, places, technical terms).\n"
    "- Avoid vague filler words: device, thing, something, object, entity, term, item, concept.\n"
    "- Write a sentence with real content; do not merely ask what a word means.\n"
    "- Do not reuse a sentence already present in this document."
)


def quality_metrics(text: str) -> dict:
    import re
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'\-]*", text)]
    vague = sum(1 for w in words if w in VAGUE_WORDS)
    return {"words": len(words), "vague_words": vague,
            "vague_ratio": round(vague / max(1, len(words)), 4)}


def usable_templates(screen_path: Path) -> list:
    """返回通过的模板索引，按首次出现顺序去重（抽样可能重复命中同一模板）。"""
    data = json.loads(screen_path.read_text())
    seen, ordered = set(), []
    for row in data["records"]:
        index = row["template_index"]
        if row["accepted_attempt"] and index not in seen:
            seen.add(index)
            ordered.append(index)
    return ordered


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--documents", type=int, default=3)
    ap.add_argument("--doc-offset", type=int, default=0,
                    help="文档编号起始偏移，用于多进程分片且不撞号")
    ap.add_argument("--slots", type=int, default=20)
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--context-mode", choices=("prev", "topic", "none"), default="prev",
                    help="prev=前几句(连贯但压制模板) topic=固定主题 none=无上下文")
    ap.add_argument("--topic", default="a short project log entry")
    ap.add_argument("--style", choices=("default", "concrete"), default="default",
                    help="concrete=追加具体词汇/反泛化词约束，用于提升文本质量")
    ap.add_argument("--domain", default="a general interest article")
    ap.add_argument("--dedup", action="store_true", help="拒绝与本文档已有句子重复的候选")
    ap.add_argument("--device", default="cpu", help="AMR 解析设备 cpu/cuda")
    ap.add_argument("--rotate", action="store_true",
                    help="按文档轮转模板起点，避免所有文档使用同一模板顺序（默认开启以提升多样性）")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--screen", type=Path,
                    default=ROOT / "experiments/swan_multiview_development1/"
                                     "generation_probe1/multitemplate_yield_t24_a4.json")
    args = ap.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY 未设置")

    pool = usable_templates(args.screen)
    if not pool:
        raise SystemExit("可用模板池为空")
    bank = json.loads(BANK.read_text())
    examples = json.loads(EXAMPLES.read_text())

    from src.core.amr_extractor import AmrlibBackend
    import amr_py3
    from s2match import compute_s2match_from_strings

    if not hasattr(amr_py3, "log_helper"):
        class _SilentLog:
            def debug(self, *a):
                pass

            def error(self, *a):
                pass

        amr_py3.log_helper = _SilentLog()
    backend = AmrlibBackend(str(ROOT / "models/model_parse_xfm_bart_large-v0_1_0"),
                            str(ROOT / "models/bart-large-tokenizer"), device=args.device)
    backend._load()
    sys_prompt = system_prompt()

    args.output.mkdir(parents=True, exist_ok=True)
    docs, calls, in_tok, out_tok = [], 0, 0, 0
    t_start = time.monotonic()

    for doc_idx in range(args.documents):
        doc_id = args.doc_offset + doc_idx
        slots, contexts, attempts_log, seen_norm = [], [], [], set()
        for slot in range(args.slots):
            # 按文档轮转模板起点：避免所有文档使用同一模板顺序（会造成跨文档近似重复）
            start = doc_id if args.rotate else 0
            template_index = pool[(start + slot) % len(pool)]
            template = bank[template_index]["normalized_template_amr"]
            if args.context_mode == "prev":
                context = " ".join(contexts[-3:])
            elif args.context_mode == "topic":
                context = args.topic
            else:
                context = ""
            prompt = build_prompt(template, examples, context=context)
            if args.style == "concrete":
                prompt += CONCRETE_INSTRUCTION.format(domain=args.domain)
            accepted = None
            for n in range(1, args.max_attempts + 1):
                r = call_api(prompt, key, 200, system=sys_prompt)
                calls += 1
                in_tok += r["usage"]["prompt_tokens"]
                out_tok += r["usage"]["completion_tokens"]
                text = " ".join(r["text"].split())
                duplicate = bool(args.dedup and text.lower() in seen_norm)
                gate = surface_gate(text)
                score = None
                if gate["pass"] and not duplicate:
                    try:
                        norm = normalize_to_template(
                            strip_snt(backend.parse_batch([text])[0]))
                        _, _, score = compute_s2match_from_strings(
                            norm, template, vectors_file=str(GLOVE))
                        score = round(float(score), 4)
                    except Exception as exc:
                        gate["error"] = f"{type(exc).__name__}: {exc}"
                attempts_log.append({"slot": slot, "attempt": n, "text": text,
                                     "words": gate["words"], "s2match": score,
                                     "duplicate": duplicate})
                if score is not None and score >= THRESHOLD:
                    accepted = {"slot": slot, "template_index": template_index,
                                "text": text, "s2match": score}
                    seen_norm.add(text.lower())
                    break
            if accepted is None:
                print(f"  doc{doc_id} slot{slot} t{template_index} FAILED "
                      f"({args.max_attempts} 次)", flush=True)
                continue
            slots.append(accepted)
            contexts.append(accepted["text"])
            print(f"  doc{doc_id} slot{slot} t{template_index} ok "
                  f"{accepted['s2match']:.3f} | {accepted['text'][:60]}", flush=True)

        doc_text = " ".join(s["text"] for s in slots)
        docs.append({
            "document_id": f"mv-doc-{doc_id:04d}",
            "text": doc_text,
            "word_count": len(doc_text.split()),
            "slots": slots,
            "accepted_slots": len(slots),
            "attempts": attempts_log,
            "template_indices": [s["template_index"] for s in slots],
            "label_status": "carrier_positions_known_by_construction",
            "is_watermarked": True,
        })
        print(f"[doc {doc_idx}] {len(slots)}/{args.slots} slots, "
              f"{len(doc_text.split())} words", flush=True)

    payload = {
        "status": "carrier_dataset_positive_only_engineering",
        "threshold": THRESHOLD, "template_pool": pool, "pool_size": len(pool),
        "documents": len(docs), "target_slots_per_document": args.slots,
        "accepted_slots_total": sum(d["accepted_slots"] for d in docs),
        "paid_calls": calls, "input_tokens": in_tok, "output_tokens": out_tok,
        "peak_price_cost_usd_upper": round((in_tok * 0.30 + out_tok * 1.20) / 1e6, 6),
        "wall_seconds": round(time.monotonic() - t_start, 1),
        "style": args.style, "context_mode": args.context_mode,
        "domain": args.domain, "dedup": bool(args.dedup),
        "rotate": bool(args.rotate), "doc_offset": args.doc_offset,
        "device": args.device,
        "quality_per_document": [quality_metrics(d["text"]) for d in docs],
        "note": "负例与攻击变体尚未构造；本文件只含正例",
    }
    (args.output / "documents.json").write_text(
        json.dumps({"summary": payload, "documents": docs}, ensure_ascii=False, indent=2))
    (args.output / "documents.jsonl").write_text(
        "".join(json.dumps(d, ensure_ascii=False, sort_keys=True) + "\n" for d in docs))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    try:
        shown = (args.output / "documents.json").relative_to(ROOT)
    except ValueError:
        shown = args.output / "documents.json"
    print(f"\nwritten: {shown}")


if __name__ == "__main__":
    main()
