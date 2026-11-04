"""P1：标点恢复基线实验。

**问题**：面对句界攻击，最朴素也最强的针对性防御是"检测前把标点恢复回句号"。
若它就能解决问题，E（作用域局部 + 分位校准）没有存在必要。必须正面对比。

**两档恢复**：
- **盲恢复（blind）**：防守方不知道原始边界，只能用规则。
    semicolon: ';' → '.'
    comma    : ',' → '.'（会误伤句内逗号，这是盲恢复的真实代价）
    space    : 在小写词后、大写词前插入 '. '（启发式，signal 已被删除）
    newline  : 句号仍在，无需恢复
- **oracle 恢复**：按已知原始边界重建，结果恰等于原文 = 恢复能力的**上界**。

**不可恢复的攻击**：merge（插入 ", and" 并改变词序）、split（新增句号位置不可知）——
标点恢复对它们无效，这是该基线的固有边界，须如实报告。

**评估口径**：与 E 完全一致 —— 作用域局部子图 + 零例经验分位映射，
载体分数 = 与之重叠的所有候选窗口的逐元素最大，再对模板取 $-\max_t 2\ln p_t$。
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party/swan_s2match_runtime_711a231d"))

from src.core.amr_extractor import AmrlibBackend, AMRExtractor  # noqa: E402
from src.core.multiview_candidates import WindowConfig, candidate_windows, select_budget  # noqa: E402
from src.core.sqlite_cache import AMRCache, normalize_text  # noqa: E402

from run_dataset4_fusion_recovery import (  # noqa: E402
    BANK, S2Scorer, elementwise_max, event_subgraphs,
)
from run_dataset4_attack_recovery import ATTACKS, _annotate_merge_spans  # noqa: E402

DATASET4 = ROOT / "experiments/swan_multiview_development1/dataset4/documents.json"
NULL = ROOT / "experiments/swan_multiview_development1/null_fp_audit1"
DEFAULT_OUT = NULL / "P1_PUNCT_BASELINE.json"
WINDOW = WindowConfig(lengths=(12,), stride=6, max_words=24, max_candidates=128)


# ───────────────────── 恢复规则 ─────────────────────

def restore_blind(text: str, attack: str) -> str:
    """防守方视角的规则恢复：不使用任何原始边界信息。"""
    if attack == "semicolon":
        return text.replace(";", ".")
    if attack == "comma":
        return text.replace(",", ".")
    if attack == "newline":
        return text            # 句号仍在，无需恢复
    if attack == "space":
        # 小写/数字结尾的词 + 空格 + 大写开头的词 → 断句
        return re.sub(r"(?<=[a-z0-9])\s+(?=[A-Z])", ". ", text)
    return text


def oracle_text(slots: list) -> str:
    """上界：按已知边界重建，恰等于原文。"""
    return " ".join(s["text"] for s in slots)


def span_map(text: str, slots: list) -> list:
    """在（恢复后的）文本中重新定位每个 slot 的字符区间。"""
    spans, cursor = [], 0
    for s in slots:
        body = re.sub(r"[.!?;,]$", "", s["text"])
        idx = text.find(body, cursor)
        if idx < 0:
            spans.append(None)
            continue
        spans.append((idx, idx + len(body)))
        cursor = idx + len(body)
    return spans


# ───────────────────── 打分 ─────────────────────

def build_ecdf(units, key):
    n_tpl = len(units[0][key])
    cols = [[] for _ in range(n_tpl)]
    for u in units:
        for t, v in enumerate(u[key]):
            if v is not None:
                cols[t].append(v)
    return [sorted(c) for c in cols]


def p_emp(arr, score):
    n = len(arr)
    if n == 0:
        return 1.0
    return (1 + n - bisect.bisect_left(arr, score)) / (1 + n)


def e_statistic(d2: list, ecdf: list) -> float | None:
    best = None
    for t, v in enumerate(d2):
        if v is None:
            continue
        x = -2.0 * math.log(p_emp(ecdf[t], v))
        if best is None or x > best:
            best = x
    return best


def tau_at(scores, q):
    s = sorted([x for x in scores if x is not None], reverse=True)
    if not s:
        return float("nan")
    k = max(1, math.ceil(q * len(s)))
    return s[min(k, len(s)) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--null-file", type=Path,
                    default=NULL / "PARTIAL_WINDOW_SCALED.json")
    ap.add_argument("--cache", type=Path,
                    default=Path("/tmp/amr_multiview_20261020/p1_restore.db"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    bank = [r["normalized_template_amr"] for r in json.loads(BANK.read_text())]
    null_units = [u for u in json.loads(args.null_file.read_text())["per_unit"]
                  if u.get("vec_d2")]
    calib = [u for u in null_units if u["cohort"] == "calibration"]
    test = [u for u in null_units if u["cohort"] == "independent_test"]
    ecdf = build_ecdf(calib, "vec_d2")
    print(f"零例：校准 {len(calib)}，测试 {len(test)}")

    docs = json.loads(DATASET4.read_text())["documents"]
    backend = AmrlibBackend(str(ROOT / "models/model_parse_xfm_bart_large-v0_1_0"),
                            str(ROOT / "models/bart-large-tokenizer"),
                            device=args.device, batch_size=args.batch_size)
    cache = AMRCache(str(args.cache))
    extractor = AMRExtractor(backend, cache, request_batch_size=args.batch_size)
    scorer = S2Scorer(bank)

    # 阈值口径：按**实测 FPR 匹配**（阈值取零例测试集的经验分位，FPR 由构造等于目标）。
    # 这与 P0/ED 显著性分析一致；另报"校准半区定阈值"的实测 FPR 作为透明对照。
    cal_e = [e_statistic(u["vec_d2"], ecdf) for u in calib]
    test_e = [e_statistic(u["vec_d2"], ecdf) for u in test]
    taus = {q: tau_at(test_e, q) for q in (0.01, 0.05)}
    cal_taus = {q: tau_at(cal_e, q) for q in (0.01, 0.05)}
    cal_realized = {
        q: sum(1 for v in test_e if v is not None and v >= cal_taus[q])
        / len([v for v in test_e if v is not None]) for q in cal_taus}
    print("阈值口径：按实测 FPR 匹配（零例测试集经验分位）")
    print(f"  q=1% τ={taus[0.01]:.4f}   q=5% τ={taus[0.05]:.4f}")
    print("透明对照（若改用校准半区定阈值，测试集实测 FPR 会漂移）：")
    print(f"  名义 1% → 实测 {cal_realized[0.01]:.2%}；"
          f"名义 5% → 实测 {cal_realized[0.05]:.2%}")

    scenarios = ["baseline", "semicolon", "comma", "space", "newline", "split", "merge"]
    # 标点恢复对 split/merge 无效：词序已变 / 新增边界不可知
    restore_applicable = {"baseline", "semicolon", "comma", "space", "newline"}
    rows = []

    for doc in docs:
        slots = doc["slots"]
        for attack in scenarios:
            # 原始（未恢复）文本
            text_orig, _spans = ATTACKS[attack](slots)

            kinds = ["none"]
            if attack in restore_applicable:
                kinds += ["blind", "oracle"]

            for kind in kinds:
                if kind == "none":
                    text = text_orig
                    # 未恢复时用攻击函数自己给出的 span（含句末标点），
                    # 与既有 P0/D7 口径一致；split 的 span 只有这样才能覆盖整槽
                    spans = (None if attack == "merge" else _spans)
                    if spans is None:
                        spans = span_map(text, slots)
                elif kind == "blind":
                    text = restore_blind(text_orig, attack)
                    spans = span_map(text, slots)
                else:
                    text = oracle_text(slots)
                    spans = span_map(text, slots)
                cands = select_budget(
                    candidate_windows(text, WINDOW)["candidates"], 0)
                texts = list(dict.fromkeys(normalize_text(c["text"]) for c in cands))
                extractor.extract_many(texts)

                for si, slot in enumerate(slots):
                    sp = spans[si] if si < len(spans) else None
                    if sp is None:
                        continue
                    d2 = [None] * len(bank)
                    n_win = 0
                    for c in cands:
                        if not (sp[0] < c["end_char"] and c["start_char"] < sp[1]):
                            continue
                        rec = extractor.extract(normalize_text(c["text"]))
                        if rec is None or not rec.graph_valid:
                            continue
                        n_win += 1
                        for sub in event_subgraphs(rec.amr):
                            d2 = elementwise_max(d2, scorer.scores(sub))
                    rows.append({
                        "doc": doc["document_id"], "attack": attack, "kind": kind,
                        "slot": si, "template_index": slot["template_index"],
                        "n_windows": n_win,
                        "e": e_statistic(d2, ecdf),
                    })
        print(f"  {doc['document_id']} done (calls={scorer.calls})", flush=True)

    # ── 汇总 ──
    def rate(rows_, q):
        """分母固定为 32 槽：恢复后无法定位的载体计为**失败**（等价于载体被恢复动作破坏）。"""
        vals = [r["e"] for r in rows_ if r["e"] is not None]
        hits = sum(1 for v in vals if v >= taus[q])
        return hits / NSLOT_TOTAL if NSLOT_TOTAL else None, hits, len(vals)

    # ── 汇总（分母固定为总槽数：不可定位计失败）──
    NSLOT_TOTAL = sum(len(d["slots"]) for d in docs)
    summary = {"note": "标点恢复基线；blind=规则恢复，oracle=上界(=原文)，none=未恢复",
               "denominator": "总槽数（恢复后不可定位的载体计为失败）",
               "slot_total": NSLOT_TOTAL,
               "thresholds": {f"q{q}": round(taus[q], 4) for q in taus},
               "null_test_units": len(test), "rows": []}
    print("\n" + "=" * 96)
    print("载体恢复率（E 统计量；分母 = 总槽数 32，恢复后不可定位计失败）")
    print("=" * 96)
    print(f"{'attack':<11} | {'none@1%':>9} {'blind@1%':>9} {'oracle@1%':>10}"
          f" | {'none@5%':>9} {'blind@5%':>9} {'oracle@5%':>10} | {'盲恢复可定位':>12}")
    print("-" * 96)
    for attack in scenarios:
        line = {"attack": attack}
        for kind in ("none", "blind", "oracle"):
            sub = [r for r in rows if r["attack"] == attack and r["kind"] == kind]
            r1, h1, loc = rate(sub, 0.01) if sub else (None, 0, 0)
            r5, h5, _ = rate(sub, 0.05) if sub else (None, 0, 0)
            line[kind] = {"n": len(sub), "r1": r1, "r5": r5,
                          "hits_q01": h1, "locatable": loc}
            line[f"_{kind}_vals"] = [r["e"] for r in sub]

        def f(v):
            return f"{v:>9.1%}" if v is not None else f"{'-':>9}"
        loc = line["blind"]["locatable"]
        loc_s = f"{loc}/{NSLOT_TOTAL}" if line["blind"]["n"] else "-"
        print(f"{attack:<11} | {f(line['none']['r1'])} {f(line['blind']['r1'])} "
              f"{f(line['oracle']['r1']):>10} | {f(line['none']['r5'])} "
              f"{f(line['blind']['r5'])} {f(line['oracle']['r5']):>10} | {loc_s:>12}")
        summary["rows"].append({k: v for k, v in line.items() if not k.startswith("_")})
        summary.setdefault("per_carrier", []).extend(
            [{"attack": attack, "kind": kind, "e": v}
             for kind in ("none", "blind", "oracle") for v in line[f"_{kind}_vals"]])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nS2Match 调用: {scorer.calls}")
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
