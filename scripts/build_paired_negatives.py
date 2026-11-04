"""构造配对负例：同生成器、同主题/风格/长度，但不同命题、且不按模板约束。

目的：隔离“水印”与“生成痕迹/领域/风格”两个变量。
正例：dataset5 中按模板生成的载体句。
负例：同一个模型、同一文档上下文、同长度，但描述不同事件，不要求匹配任何模板。

输出目录结构与 dataset5 一致，可直接送入 run_dataset4_fusion_recovery.py 打分。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_swan_multitemplate_yield import call_api, system_prompt  # noqa: E402

DEFAULT_DOCS = ROOT / "experiments/swan_multiview_development1/dataset5/documents.json"
DEFAULT_OUT = ROOT / "experiments/swan_multiview_development1/dataset5_paired_neg"

PROMPT = """Write ONE natural English sentence of about {n} words that continues the passage below.

Passage so far:
{context}

Reference sentence to AVOID (do not paraphrase it; do not reuse its event, entities, or structure):
{reference}

Requirements:
- Same topic, tone, and domain as the passage.
- Describe a DIFFERENT event from the reference sentence.
- Do NOT try to match any semantic template or abstract meaning representation.
- Write a plain, specific sentence; avoid vague filler words.
- Return only the sentence, no quotes or explanation."""

VAGUE = frozenset({
    "device", "thing", "something", "object", "entity", "term", "item",
    "concept", "stuff", "person", "idea", "reason", "way", "kind", "type",
})


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z'\-]*", text.lower())


def jaccard(a: str, b: str) -> float:
    sa, sb = set(words(a)), set(words(b))
    if not sa or not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def single_sentence(text: str) -> bool:
    return len(re.findall(r"[.!?]", text)) <= 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--documents", type=Path, default=DEFAULT_DOCS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--doc-offset", type=int, default=0)
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--context-turns", type=int, default=3)
    ap.add_argument("--min-words", type=int, default=5)
    ap.add_argument("--max-words", type=int, default=30)
    ap.add_argument("--max-jaccard", type=float, default=0.55)
    args = ap.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY 未设置")

    payload = json.loads(args.documents.read_text())
    docs = payload["documents"]
    if args.doc_offset or args.limit_docs:
        end = args.doc_offset + args.limit_docs if args.limit_docs else None
        docs = docs[args.doc_offset:end]

    out_docs, pairs, calls, in_tok, out_tok = [], [], 0, 0, 0
    t0 = time.monotonic()
    sys_prompt = system_prompt()

    for d in docs:
        contexts, neg_slots, seen = [], [], set()
        for slot in d["slots"]:
            ref = slot["text"]
            n_words = max(args.min_words, min(args.max_words, len(words(ref))))
            best = None
            for attempt in range(1, args.max_attempts + 1):
                prompt = PROMPT.format(
                    n=n_words,
                    context=" ".join(contexts[-args.context_turns:]) or "(start of the text)",
                    reference=ref,
                )
                r = call_api(prompt, key, 160, system=sys_prompt)
                calls += 1
                in_tok += r["usage"]["prompt_tokens"]
                out_tok += r["usage"]["completion_tokens"]
                text = " ".join(r["text"].split()).strip('" ')
                wc = len(words(text))
                jac = jaccard(text, ref)
                ok = (args.min_words <= wc <= args.max_words
                      and single_sentence(text) and jac < args.max_jaccard
                      and text.lower() not in seen)
                if ok:
                    best = {"text": text, "words": wc, "jaccard": round(jac, 4),
                            "attempt": attempt, "valid": True}
                    break
                if best is None or jac < best["jaccard"]:
                    best = {"text": text, "words": wc, "jaccard": round(jac, 4),
                            "attempt": attempt, "valid": False}
            if not best:
                raise SystemExit(f"doc {d['document_id']} slot {slot['slot']} 无候选")
            neg_slots.append({
                "slot": slot["slot"],
                "text": best["text"],
                "template_index": slot["template_index"],
                "reference_text": ref,
                "jaccard": best["jaccard"],
                "words": best["words"],
                "valid": best["valid"],
            })
            seen.add(best["text"].lower())
            contexts.append(best["text"])
            if not best["valid"]:
                print(f"  WARN doc {d['document_id']} slot {slot['slot']} "
                      f"未通过校验 (jac={best['jaccard']})", flush=True)

        doc_text = " ".join(s["text"] for s in neg_slots)
        out_docs.append({
            "document_id": d["document_id"],
            "text": doc_text,
            "word_count": len(doc_text.split()),
            "slots": neg_slots,
            "accepted_slots": len(neg_slots),
            "template_indices": [s["template_index"] for s in neg_slots],
            "pairing": "same_doc_slot_same_generator_different_event",
            "is_watermarked": False,
        })
        for s in neg_slots:
            pairs.append({
                "doc": d["document_id"], "slot": s["slot"],
                "positive_text": s["reference_text"], "negative_text": s["text"],
                "template_index": s["template_index"], "jaccard": s["jaccard"],
            })
        print(f"[{d['document_id']}] {len(neg_slots)} 槽, {len(doc_text.split())} 词", flush=True)

    summary = {
        "status": "paired_negative_by_same_generator_different_event",
        "documents": len(out_docs),
        "slots": sum(len(d["slots"]) for d in out_docs),
        "invalid_slots": sum(1 for d in out_docs for s in d["slots"] if not s["valid"]),
        "paid_calls": calls, "input_tokens": in_tok, "output_tokens": out_tok,
        "peak_price_cost_usd_upper": round((in_tok * 0.30 + out_tok * 1.20) / 1e6, 6),
        "wall_seconds": round(time.monotonic() - t0, 1),
        "doc_offset": args.doc_offset,
        "constraints": {"min_words": args.min_words, "max_words": args.max_words,
                        "max_jaccard": args.max_jaccard},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "documents.json").write_text(
        json.dumps({"summary": summary, "documents": out_docs}, ensure_ascii=False, indent=2))
    (args.output / "pairs.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
