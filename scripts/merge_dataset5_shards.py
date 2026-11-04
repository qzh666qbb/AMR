"""合并 dataset5 的生成分片为单一正例数据集，并做质量审计。

审计项：
- 槽位产率、每篇词数
- 篇内重复 / **跨篇重复**（模板轮转若失效会暴露）
- 模板覆盖度（24 模板池里每个模板被用了几次）
- 泛化词比例
- 总花费与调用数
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARDS = ROOT / "experiments/swan_multiview_development1/dataset5_shards"
OUT = ROOT / "experiments/swan_multiview_development1/dataset5"
VAGUE = frozenset({
    "device", "thing", "something", "object", "entity", "term", "item",
    "concept", "stuff", "person", "idea", "reason", "way", "kind", "type",
})


def quality(text: str) -> dict:
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'\-]*", text)]
    vague = sum(1 for w in words if w in VAGUE)
    return {"words": len(words), "vague_words": vague,
            "vague_ratio": round(vague / max(1, len(words)), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, default=SHARDS)
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()

    files = sorted(args.shards.glob("shard*/documents.json"))
    if not files:
        raise SystemExit("未找到分片产物")
    docs, calls, in_tok, out_tok, pool = [], 0, 0, 0, None
    for f in files:
        d = json.loads(f.read_text())
        s = d["summary"]
        calls += s["paid_calls"]
        in_tok += s["input_tokens"]
        out_tok += s["output_tokens"]
        pool = s["template_pool"]
        docs.extend(d["documents"])
        print(f"  {f.parent.name}: {len(d['documents'])} 篇, "
              f"{s['accepted_slots_total']} 槽, ${s['peak_price_cost_usd_upper']:.4f}")

    docs.sort(key=lambda x: x["document_id"])
    # 跨篇重复审计
    seen = {}
    dup_cross = 0
    for doc in docs:
        for s in doc["slots"]:
            key = s["text"].strip().lower()
            if key in seen and seen[key] != doc["document_id"]:
                dup_cross += 1
            seen[key] = doc["document_id"]
    dup_within = sum(
        len([1 for t, c in Counter(s["text"].strip().lower()
                                   for s in d["slots"]).items() if c > 1])
        for d in docs)
    tpl_counter = Counter(t for d in docs for t in d["template_indices"])

    total_slots = sum(len(d["slots"]) for d in docs)
    summary = {
        "status": "carrier_dataset_positive_only_scaled",
        "documents": len(docs), "target_slots_per_document": 20,
        "accepted_slots_total": total_slots,
        "accept_rate": round(total_slots / (20 * len(docs)), 4),
        "template_pool": pool, "pool_size": len(pool) if pool else 0,
        "template_usage": dict(sorted(tpl_counter.items())),
        "templates_unused": [t for t in (pool or []) if t not in tpl_counter],
        "duplicates_within_docs": dup_within,
        "duplicates_across_docs": dup_cross,
        "paid_calls": calls, "input_tokens": in_tok, "output_tokens": out_tok,
        "peak_price_cost_usd_upper": round((in_tok * 0.30 + out_tok * 1.20) / 1e6, 6),
        "quality": [quality(d["text"]) for d in docs],
        "words": [d["word_count"] for d in docs],
        "note": "正例；载体位置为注入时已知的设计参数，非人工标注",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "documents.json").write_text(json.dumps(
        {"summary": summary, "documents": docs}, ensure_ascii=False, indent=2))
    (args.output / "documents.jsonl").write_text("".join(
        json.dumps(d, ensure_ascii=False, sort_keys=True) + "\n" for d in docs))

    print(f"\n合并：{len(docs)} 篇，{total_slots} 槽，"
          f"接受率 {summary['accept_rate']:.1%}")
    w = summary["words"]
    print(f"词数：min={min(w)} 中位={sorted(w)[len(w)//2]} max={max(w)}")
    vr = [q["vague_ratio"] for q in summary["quality"]]
    print(f"泛化词比例：中位 {sorted(vr)[len(vr)//2]:.3f}，max {max(vr):.3f}")
    print(f"篇内重复 {dup_within}，**跨篇重复 {dup_cross}**")
    print(f"模板覆盖 {len(tpl_counter)}/{len(pool)}，未用模板 {summary['templates_unused']}")
    print(f"总调用 {calls}，成本 ${summary['peak_price_cost_usd_upper']:.4f}")
    print(f"written: {args.output / 'documents.json'}")


if __name__ == "__main__":
    main()
