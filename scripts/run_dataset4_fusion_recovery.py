"""dataset4 融合检测实验：单视图 vs 作用域局部 vs 多视图融合。

核心问题：把打分从"整图比对"改成"作用域局部比对"（候选 AMR 拆成事件子图），
能否把被句界破坏的载体救回来？

为什么用排名指标：我们没有负例集，无法校准判定阈值。排名（真实载体模板在
100 个 Bank 模板中的位次）是**免校准**的，且在三个检测器之间是共同尺度。

三个检测器（都在同一批 224 个 carrier 实例上）：
  D1  single_view    : 整图 S2Match，取重叠候选最大值（= 现役 SWAN 行为）
  D2  scope_local    : 候选拆成事件子图，各子图分别 S2Match，取最大值
  D3  fusion         : D2 的 AMR 子图分 + 语义向量视图 + 实体视图，对齐后融合

只用缓存里的 AMR；D3 需要编码器（本地 MiniLM），不调用 parser、不调用 API。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party/swan_s2match_runtime_711a231d"))

import penman  # noqa: E402

from src.core.amr_extractor import AmrlibBackend, AMRExtractor  # noqa: E402
from src.core.multiview_candidates import WindowConfig, candidate_windows, select_budget  # noqa: E402
from src.core.sqlite_cache import AMRCache, normalize_text  # noqa: E402

from probe_swan_multitemplate_yield import normalize_to_template, strip_snt  # noqa: E402
import amr_py3  # noqa: E402
import s2match as s2match_mod  # noqa: E402
from s2match import compute_s2match_from_strings  # noqa: E402

if not hasattr(amr_py3, "log_helper"):
    class _SilentLog:
        def debug(self, *a): pass
        def error(self, *a): pass
    amr_py3.log_helper = _SilentLog()

GLOVE = ROOT / "vectors/glove.6B.100d.txt"
_S2MATCH_VECS = s2match_mod.load_vecs(str(GLOVE))
s2match_mod.load_vecs = lambda fp: _S2MATCH_VECS

BANK = ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_100.json"
DATASET4 = ROOT / "experiments/swan_multiview_development1/dataset4/documents.json"
DEFAULT_OUT = ROOT / "experiments/swan_multiview_development1/dataset4_fusion1"
CACHE = Path("/tmp/amr_multiview_20261020/dataset4_test.db")
WINDOW = WindowConfig(lengths=(12,), stride=6, max_words=24, max_candidates=128)
THRESHOLD = 0.75

from run_dataset4_attack_recovery import ATTACKS, _annotate_merge_spans  # noqa: E402

_EVENT_RE = re.compile(r".+-\d\d$")


# ───────────────────── 作用域局部：事件子图分解 ─────────────────────

def event_subgraphs(amr_str: str, max_depth: int = 3, max_nodes: int = 12) -> list:
    """把一张 AMR 拆成若干以事件节点为根的子图（PENMAN 字符串）。

    这是"作用域局部"的实现：SWAN 的载体是**单句级**结构，整图比对时
    其它句子的内容会稀释分数；按事件节点切开后，每个子图重新变成
    与单个注入单元尺度可比的图。
    """
    try:
        graph = penman.decode(amr_str)
    except Exception:
        return []
    concepts, inst_triples = {}, []
    for t in graph.instances():
        v, c = str(t.source), str(t.target)
        concepts[v] = c
        inst_triples.append((v, ":instance", c))
    adj = {}
    for s, r, t in graph.triples:
        s, r, t = str(s), str(r), str(t)
        if r == ":instance":
            continue
        adj.setdefault(s, []).append((r, t))

    subs = []
    for node, concept in concepts.items():
        if not _EVENT_RE.match(concept):
            continue
        keep_nodes, keep_edges = {node}, []
        seen, frontier = {node}, [(node, 0)]
        while frontier:
            cur, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for role, tgt in adj.get(cur, []):
                keep_edges.append((cur, role, tgt))
                if tgt in concepts and tgt not in seen and len(keep_nodes) < max_nodes:
                    seen.add(tgt)
                    keep_nodes.add(tgt)
                    frontier.append((tgt, depth + 1))
        triples = [(n, ":instance", concepts[n]) for n in keep_nodes] + keep_edges
        # 去重且保序
        dedup, out = set(), []
        for tr in triples:
            if tr not in dedup:
                dedup.add(tr)
                out.append(tr)
        try:
            subs.append(penman.encode(penman.Graph(out, top=node)))
        except Exception:
            continue
    return subs


# ───────────────────── 评分器 ─────────────────────

class S2Scorer:
    """S2Match 打分，带规范化与结果缓存。"""

    def __init__(self, bank_amr: list):
        self.bank = bank_amr
        self._norm_cache = {}
        self._score_cache = {}
        self.calls = 0

    def norm(self, amr: str) -> str | None:
        if amr not in self._norm_cache:
            try:
                self._norm_cache[amr] = normalize_to_template(strip_snt(amr))
            except Exception:
                self._norm_cache[amr] = None
        return self._norm_cache[amr]

    def scores(self, amr: str) -> list:
        """返回对全部 Bank 模板的 S2Match 分数列表（None 表示失败）。"""
        if amr in self._score_cache:
            return self._score_cache[amr]
        norm = self.norm(amr)
        if norm is None:
            self._score_cache[amr] = [None] * len(self.bank)
            return self._score_cache[amr]
        out = []
        for tpl in self.bank:
            try:
                _, _, sc = compute_s2match_from_strings(
                    norm, tpl, vectors_file=str(GLOVE))
                out.append(float(sc))
            except Exception:
                out.append(None)
            self.calls += 1
        self._score_cache[amr] = out
        return out


def rank_of(values: list, target: int) -> int | None:
    """target 在 values 中的排名（1 = 最高），同分按索引升序打破。"""
    valid = [(v, i) for i, v in enumerate(values) if v is not None]
    if not valid:
        return None
    tv = values[target]
    if tv is None:
        return None
    return 1 + sum(1 for v, _ in valid if v > tv)


def rank_vec(values: list) -> list:
    """每个索引的排名（1 = 最高）；无分位置为 None。"""
    valid = sorted(((v, i) for i, v in enumerate(values) if v is not None),
                   key=lambda x: (-x[0], x[1]))
    ranks = [None] * len(values)
    for pos, (_, i) in enumerate(valid, start=1):
        ranks[i] = pos
    return ranks


def rrf_fuse(a: list, b: list, k: int = 10) -> list:
    """秩级融合（Reciprocal Rank Fusion）。

    不同作用域的 S2Match 分数尺度不可比（子图天然偏高），直接取 max 会
    把干扰模板抬得比正例更多。RRF 只在**秩**上合并，回避尺度问题。
    """
    ra, rb = rank_vec(a), rank_vec(b)
    out = []
    for i in range(len(a)):
        parts = []
        if ra[i] is not None:
            parts.append(1.0 / (k + ra[i]))
        if rb[i] is not None:
            parts.append(1.0 / (k + rb[i]))
        out.append(sum(parts) if parts else None)
    return out


def elementwise_max(a: list, b: list) -> list:
    out = []
    for x, y in zip(a, b):
        if x is None:
            out.append(y)
        elif y is None:
            out.append(x)
        else:
            out.append(max(x, y))
    return out


# ───────────────────── 主流程 ─────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dataset", type=Path, default=DATASET4,
                    help="正例数据集 documents.json（默认 dataset4）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--doc-offset", type=int, default=0,
                    help="从第几篇开始（配合 --limit-docs 做多进程分片）")
    ap.add_argument("--cache", type=Path, default=CACHE,
                    help="AMR 缓存库路径（多进程分片须各自独立，避免写竞争）")
    ap.add_argument("--window-lengths", nargs="+", type=int, default=[12],
                    help="候选窗口词数（默认 12；短窗基线用 6）")
    ap.add_argument("--window-stride", type=int, default=6)
    ap.add_argument("--views", choices=("d1", "dual"), default="dual",
                    help="d1=只算整图视图（短窗基线用），dual=同时算子图视图")
    ap.add_argument("--unit-mode", choices=("window", "sentence"), default="window",
                    help="sentence=每个槽位整句解析一次（P0 官方协议用）")
    args = ap.parse_args()

    window = WindowConfig(lengths=tuple(args.window_lengths),
                          stride=args.window_stride,
                          max_words=max(24, 2 * max(args.window_lengths)),
                          max_candidates=128)

    docs = json.loads(args.dataset.read_text())["documents"]
    if args.doc_offset:
        docs = docs[args.doc_offset:]
    if args.limit_docs:
        docs = docs[: args.limit_docs]
    bank_amr = [r["normalized_template_amr"]
                for r in json.loads(BANK.read_text())]
    backend = AmrlibBackend(str(ROOT / "models/model_parse_xfm_bart_large-v0_1_0"),
                            str(ROOT / "models/bart-large-tokenizer"),
                            device=args.device, batch_size=args.batch_size)
    fp = backend.fingerprint
    cache = AMRCache(str(args.cache))
    extractor = AMRExtractor(backend, cache, request_batch_size=args.batch_size)
    scorer = S2Scorer(bank_amr)

    records = {}          # (doc, attack, slot) -> dict
    sub_counts = []

    for doc in docs:
        slots = doc["slots"]
        for attack_name, fn in ATTACKS.items():
            if attack_name == "merge":
                text, spans = fn(slots)
                _annotate_merge_spans(spans)
            else:
                text, spans = fn(slots)
            candidates = select_budget(candidate_windows(text, window)["candidates"], 0) \
                if args.unit_mode == "window" else []
            # 先确保所有候选的 AMR 已解析入缓存（否则下面 cache.get 全部落空）
            if candidates:
                texts = list(dict.fromkeys(normalize_text(c["text"]) for c in candidates))
                extractor.extract_many(texts)

            for si, slot in enumerate(slots):
                tpl = slot["template_index"]
                if attack_name == "merge":
                    span = None
                    for sp in spans:
                        if sp.get("_slot_a_idx") == si:
                            span = sp["slot_a"]; break
                        if sp.get("_slot_b_idx") == si:
                            span = sp["slot_b"]; break
                else:
                    span = spans[si] if si < len(spans) else None
                if span is None:
                    continue
                if args.unit_mode == "sentence":
                    unit_text = text[span[0]:span[1]]
                    extractor.extract_many([normalize_text(unit_text)])
                    candidates = [{"text": unit_text,
                                   "start_char": span[0], "end_char": span[1]}]

                d1 = [None] * len(bank_amr)
                d2 = [None] * len(bank_amr)
                cand_vecs = []
                n_cand = 0
                n_sub = 0
                for cand in candidates:
                    if not (span[0] < cand["end_char"] and cand["start_char"] < span[1]):
                        continue
                    rec = cache.get(normalize_text(cand["text"]), fp)
                    if rec is None:
                        continue
                    n_cand += 1
                    # 视图 A：整图
                    s1 = scorer.scores(rec.amr)
                    d1 = elementwise_max(d1, s1)
                    # 视图 B：事件子图（作用域局部）
                    if args.views == "dual":
                        subs = event_subgraphs(rec.amr, max_depth=args.max_depth)
                        n_sub += len(subs)
                        s2 = [None] * len(bank_amr)
                        for sub in subs:
                            s2 = elementwise_max(s2, scorer.scores(sub))
                    else:
                        s2 = [None] * len(bank_amr)
                    d2 = elementwise_max(d2, s2)
                    cand_vecs.append((s1, s2))
                if n_cand == 0:
                    continue
                # D3：朴素 max 融合（已知会抬高干扰项，作为负对照保留）
                d3 = elementwise_max(d1, d2)
                # D4：秩级融合（RRF），回避跨作用域尺度不可比的问题
                d4 = [None] * len(bank_amr)
                if args.views == "dual":
                    for s1, s2 in cand_vecs:
                        d4 = elementwise_max(d4, rrf_fuse(s1, s2))
                else:
                    d4 = list(d1)
                # D5：门控融合。整图若已看到强证据（>= 注入阈值）就沿用整图，
                # 否则退回多作用域秩融合。门控只用可观测分数，不用标签。
                best_whole = max((v for v in d1 if v is not None), default=None)
                d5 = d1 if (best_whole is not None and best_whole >= THRESHOLD) else d4
                records[f"{doc['document_id']}|{attack_name}|{si}"] = {
                    "doc": doc["document_id"], "attack": attack_name, "slot": si,
                    "template_index": tpl,
                    "n_candidates": n_cand, "n_subgraphs": n_sub,
                    "best_whole_score": best_whole,
                    "gated_to_fusion": d5 is d4,
                    "rank_d1": rank_of(d1, tpl),
                    "rank_d2": rank_of(d2, tpl),
                    "rank_d3": rank_of(d3, tpl),
                    "rank_d4": rank_of(d4, tpl),
                    "rank_d5": rank_of(d5, tpl),
                    # 目标条件化分数（真实载体模板自身）
                    "score_d1_true": d1[tpl], "score_d2_true": d2[tpl],
                    "score_d3_true": d3[tpl], "score_d4_true": d4[tpl],
                    "score_d5_true": d5[tpl],
                    # 全库最高分（与负例侧 max-over-bank 口径对称）
                    "max_d1": max((v for v in d1 if v is not None), default=None),
                    "max_d2": max((v for v in d2 if v is not None), default=None),
                    "max_d3": max((v for v in d3 if v is not None), default=None),
                    "max_d4": max((v for v in d4 if v is not None), default=None),
                    "max_d5": max((v for v in d5 if v is not None), default=None),
                    # 完整向量（scope 级 z 归一化需要按模板统计）
                    "vec_d1": [None if v is None else round(v, 4) for v in d1],
                    "vec_d2": [None if v is None else round(v, 4) for v in d2],
                }
                sub_counts.append(n_sub)
            print(f"  {doc['document_id']} {attack_name} done "
                  f"(s2match calls={scorer.calls})", flush=True)

    # ── 汇总 ──
    def agg(rows, key):
        ranks = [r[key] for r in rows if r[key] is not None]
        if not ranks:
            return {"n": 0, "top1": 0.0, "top5": 0.0, "top10": 0.0,
                    "median_rank": None}
        ranks_sorted = sorted(ranks)
        mid = len(ranks_sorted) // 2
        return {
            "n": len(ranks),
            "top1": round(sum(r == 1 for r in ranks) / len(ranks), 4),
            "top5": round(sum(r <= 5 for r in ranks) / len(ranks), 4),
            "top10": round(sum(r <= 10 for r in ranks) / len(ranks), 4),
            "median_rank": ranks_sorted[mid],
        }

    rows = list(records.values())
    by_attack = {}
    for r in rows:
        by_attack.setdefault(r["attack"], []).append(r)

    summary = {
        "note": ("排名指标免校准：真实载体模板在100个Bank模板中的位次。"
                 "1=最优。三个检测器同尺度可比。"),
        "carrier_instances": len(rows),
        "overall": {
            "D1_single_view": agg(rows, "rank_d1"),
            "D2_scope_local": agg(rows, "rank_d2"),
            "D3_naive_max_fusion": agg(rows, "rank_d3"),
            "D4_rank_fusion_rrf": agg(rows, "rank_d4"),
            "D5_gated_fusion": agg(rows, "rank_d5"),
        },
        "by_attack": {
            a: {
                "n": len(v),
                "D1_single_view": agg(v, "rank_d1"),
                "D2_scope_local": agg(v, "rank_d2"),
                "D3_naive_max_fusion": agg(v, "rank_d3"),
                "D4_rank_fusion_rrf": agg(v, "rank_d4"),
                "D5_gated_fusion": agg(v, "rank_d5"),
            } for a, v in sorted(by_attack.items())
        },
        "gate_stats": {
            "gated_to_fusion": sum(1 for r in rows if r["gated_to_fusion"]),
            "kept_whole_graph": sum(1 for r in rows if not r["gated_to_fusion"]),
        },
        "s2match_calls": scorer.calls,
        "mean_subgraphs_per_candidate": (
            round(sum(sub_counts) / len(sub_counts), 2) if sub_counts else 0),
        "per_instance": rows,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "FUSION_RECOVERY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n" + "=" * 132)
    print("排名指标对比（真实载体模板在 100 个 Bank 模板中的位次；免校准，1=最优，越小越好）")
    print("=" * 132)
    print(f"{'attack':<11} {'n':>4} | "
          f"{'D1整图':>14} {'中位':>5} | {'D2局部':>14} {'中位':>5} | "
          f"{'D3朴max':>14} {'中位':>5} | {'D4秩融合':>14} {'中位':>5} | "
          f"{'D5门控':>14} {'中位':>5}")
    print("-" * 132)
    keys = [("D1_single_view", "D1整图"), ("D2_scope_local", "D2局部"),
            ("D3_naive_max_fusion", "D3朴max"), ("D4_rank_fusion_rrf", "D4秩融合"),
            ("D5_gated_fusion", "D5门控")]
    for a, v in sorted(summary["by_attack"].items()):
        cells = f"{a:<11} {v['n']:>4} | "
        for k, _ in keys:
            s = v[k]
            med = s['median_rank']
            cells += (f"{s['top5']:>14.1%} "
                      f"{(str(med) if med is not None else '-'):>5} | ")
        print(cells)
    print("-" * 132)
    cells = f"{'ALL':<11} {summary['carrier_instances']:>4} | "
    for k, _ in keys:
        s = summary["overall"][k]
        med = s['median_rank']
        cells += (f"{s['top5']:>14.1%} "
                  f"{(str(med) if med is not None else '-'):>5} | ")
    print(cells)
    print()
    print(" top1  " + "  ".join(
        f"{label}={summary['overall'][k]['top1']:.1%}" for k, label in keys))
    g = summary["gate_stats"]
    print(f"门控：退回融合 {g['gated_to_fusion']}/{len(rows)}，"
          f"沿用整图 {g['kept_whole_graph']}/{len(rows)}")
    print(f"S2Match 调用: {scorer.calls}，平均每候选 {summary['mean_subgraphs_per_candidate']} 个子图")
    print(f"written: {args.output / 'FUSION_RECOVERY.json'}")


if __name__ == "__main__":
    main()
