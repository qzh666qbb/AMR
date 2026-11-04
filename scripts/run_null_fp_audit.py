"""负例（零假设）审计：多作用域融合 D5 是否抬高空文分数、破坏 FPR。

问题：D5 在正例上把被破坏句界的恢复率提升了 1.8–2.4 倍。但如果 D5 把**所有**文本的
分数都抬高了，那提升就是假的。本脚本用冻结的 C4 人类文本零假设集回答这个问题。

判据用**官方 SWAN 规则**（见 detection/detect_from_parsed_amrs.py）：
  单元 green  ⇔ max_over_bank S2Match >= 0.65
  z = (G - 0.25*N) / sqrt(N * 0.25 * 0.75)

比较 D1（整图，= 现役行为）与 D5（门控多作用域融合）在空文上的：
  1. 单元级虚假 green 率
  2. 文档级 z 分布
  3. 校准集上定阈值 → 独立测试集上的真实 FPR
"""
from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party/swan_s2match_runtime_711a231d"))

from src.core.amr_extractor import AmrlibBackend, AMRExtractor  # noqa: E402
from src.core.multiview_candidates import WindowConfig, candidate_windows  # noqa: E402
from src.core.sqlite_cache import AMRCache, normalize_text  # noqa: E402

from run_dataset4_fusion_recovery import (  # noqa: E402
    BANK, S2Scorer, elementwise_max, event_subgraphs, rrf_fuse,
)

NULL_DIR = ROOT / "experiments/swan_elastic_formal_null1/inputs"
DEFAULT_OUT = ROOT / "experiments/swan_multiview_development1/null_fp_audit1"
UNIT_CFG = WindowConfig(lengths=(12,), stride=6, max_words=200, max_candidates=256)
GREEN_THRESHOLD = 0.65     # 官方 detection 默认阈值
GATE_THRESHOLD = 0.75      # D5 门控阈值（注入阈值）
LAMBDA = 0.25              # 官方 λ


def load_windows(path: Path, limit: int) -> list:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(html.unescape(row["window"]))
        if limit and len(rows) >= limit:
            break
    return rows


def units_of(text: str) -> list:
    """单元 = 句界候选（与官方 SWAN 的句级单元一致）。"""
    cands = candidate_windows(text, UNIT_CFG)["candidates"]
    seen, out = set(), []
    for c in cands:
        if "surface_boundary" not in c["kinds"]:
            continue
        key = (c["start_char"], c["end_char"])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def zscore(green: int, n: int) -> float:
    denom = math.sqrt(n * LAMBDA * (1 - LAMBDA))
    if denom < 1e-9:
        return float("nan")
    return (green - LAMBDA * n) / denom


def score_unit(amr: str, scorer: S2Scorer, max_depth: int) -> dict:
    """返回一个单元在四个视图/融合下的全 Bank 分数向量。"""
    d1 = scorer.scores(amr)
    subs = event_subgraphs(amr, max_depth=max_depth)
    d2 = [None] * len(scorer.bank)
    for sub in subs:
        d2 = elementwise_max(d2, scorer.scores(sub))
    d3 = elementwise_max(d1, d2)
    d4 = rrf_fuse(d1, d2)
    best_whole = max((v for v in d1 if v is not None), default=None)
    d5 = d1 if (best_whole is not None and best_whole >= GATE_THRESHOLD) else d4
    return {"d1": d1, "d2": d2, "d3": d3, "d4": d4, "d5": d5,
            "n_subgraphs": len(subs), "best_whole": best_whole}


def top(vals: list) -> float:
    return max((v for v in vals if v is not None), default=0.0)


def describe(xs: list) -> dict:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    def pct(p):
        idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
        return round(s[idx], 4)
    return {
        "n": len(s), "mean": round(sum(s) / len(s), 4),
        "median": pct(0.5), "p90": pct(0.90), "p95": pct(0.95),
        "p99": pct(0.99), "max": round(s[-1], 4),
    }


def threshold_at_fpr(null_scores: list, target: float) -> float:
    """在零假设分数上取经验分位，使 FPR ≈ target。"""
    s = sorted(null_scores, reverse=True)
    k = max(1, int(math.ceil(target * len(s))))
    return s[min(k, len(s)) - 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache", type=Path,
                    default=Path("/tmp/amr_multiview_20261020/null_audit.db"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--calibration-docs", type=int, default=60)
    ap.add_argument("--test-docs", type=int, default=60)
    ap.add_argument("--vectors", type=Path, default=None,
                    help="若指定，额外保存每个单元的全套分数向量（z 归一化需要）")
    args = ap.parse_args()

    bank_amr = [r["normalized_template_amr"]
                for r in json.loads(BANK.read_text())]
    calib = load_windows(NULL_DIR / "calibration_windows.jsonl", args.calibration_docs)
    test = load_windows(NULL_DIR / "independent_test_windows.jsonl", args.test_docs)
    print(f"零假设文档：校准 {len(calib)}，独立测试 {len(test)}", flush=True)

    backend = AmrlibBackend(
        str(ROOT / "models/model_parse_xfm_bart_large-v0_1_0"),
        str(ROOT / "models/bart-large-tokenizer"),
        device=args.device, batch_size=args.batch_size)
    cache = AMRCache(str(args.cache))
    extractor = AMRExtractor(backend, cache, request_batch_size=args.batch_size)
    scorer = S2Scorer(bank_amr)

    all_units = []       # 单元级记录
    doc_rows = []        # 文档级记录

    args.output.mkdir(parents=True, exist_ok=True)
    ckpt = args.output / "PARTIAL.json"

    def checkpoint():
        ckpt.write_text(json.dumps(
            {"documents_done": len(doc_rows), "units": len(all_units),
             "s2match_calls": scorer.calls,
             "per_document": doc_rows, "per_unit": all_units},
            ensure_ascii=False, indent=2))

    for cohort, windows in (("calibration", calib), ("independent_test", test)):
        for di, text in enumerate(windows):
            units = units_of(text)
            if not units:
                continue
            texts = list(dict.fromkeys(normalize_text(u["text"]) for u in units))
            records = extractor.extract_many(texts)
            g = {k: 0 for k in ("d1", "d2", "d3", "d4", "d5")}
            per_unit = []
            for u in units:
                rec = records.get(normalize_text(u["text"]))
                if rec is None or not rec.graph_valid:
                    continue
                sc = score_unit(rec.amr, scorer, args.max_depth)
                greens = {k: top(sc[k]) >= GREEN_THRESHOLD for k in g}
                for k in g:
                    g[k] += int(greens[k])
                entry = {
                    "cohort": cohort, "doc": di,
                    "unit": u["text"][:120], "word_count": u["word_count"],
                    **{f"top_{k}": round(top(sc[k]), 4) for k in g},
                    "n_subgraphs": sc["n_subgraphs"], "best_whole": sc["best_whole"],
                    "green": greens,
                }
                if args.vectors is not None:
                    # z 归一化需要按模板的完整分布，故保留整套向量
                    entry["vec_d1"] = [None if v is None else round(v, 4) for v in sc["d1"]]
                    entry["vec_d2"] = [None if v is None else round(v, 4) for v in sc["d2"]]
                per_unit.append(entry)
                all_units.append(per_unit[-1])
            n = len(per_unit)
            if n == 0:
                continue
            doc_rows.append({
                "cohort": cohort, "doc": di, "n_units": n,
                "green_d1": g["d1"], "green_d2": g["d2"], "green_d3": g["d3"],
                "green_d4": g["d4"], "green_d5": g["d5"],
                **{f"z_{k}": round(zscore(g[k], n), 4) for k in g},
            })
            if (di + 1) % 5 == 0:
                print(f"  [{cohort}] {di+1}/{len(windows)} "
                      f"(s2match calls={scorer.calls})", flush=True)
                checkpoint()

    cal = [r for r in doc_rows if r["cohort"] == "calibration"]
    tst = [r for r in doc_rows if r["cohort"] == "independent_test"]

    summary = {
        "note": ("零假设审计：只建立 FPR 侧证据。正例侧 TPR 需要更多正例文档，本脚本不报 TPR。"),
        "unit_definition": "句界单元（surface_boundary）",
        "green_rule": f"max_over_bank S2Match >= {GREEN_THRESHOLD}（官方 detection 默认）",
        "z_formula": "z = (G - 0.25N) / sqrt(N*0.25*0.75)",
        "documents": {"calibration": len(cal), "independent_test": len(tst)},
        "units_total": len(all_units),
        "unit_level": {
            "n": len(all_units),
            **{f"spurious_green_rate_{k}": round(
                sum(1 for u in all_units if u["green"][k]) / len(all_units), 4)
               for k in ("d1", "d2", "d3", "d4", "d5")},
            "top_score_d1": describe([u["top_d1"] for u in all_units]),
            "top_score_d5": describe([u["top_d5"] for u in all_units]),
            "mean_subgraphs": round(
                sum(u["n_subgraphs"] for u in all_units) / len(all_units), 2),
        },
        "doc_level_z": {
            "calibration": {k: describe([r[f"z_{k}"] for r in cal])
                            for k in ("d1", "d2", "d3", "d4", "d5")},
            "independent_test": {k: describe([r[f"z_{k}"] for r in tst])
                                 for k in ("d1", "d2", "d3", "d4", "d5")},
        },
        "fixed_fpr_thresholds": {},
        "s2match_calls": scorer.calls,
        "per_document": doc_rows,
    }

    # 校准集定阈值 → 独立测试集真实 FPR
    for target in (0.01, 0.05, 0.10):
        entry = {}
        for k in ("d1", "d2", "d3", "d4", "d5"):
            cal_z = [r[f"z_{k}"] for r in cal]
            if not cal_z:
                continue
            tau = threshold_at_fpr(cal_z, target)
            realized = (sum(1 for r in tst if r[f"z_{k}"] >= tau) / len(tst)
                        if tst else None)
            entry[k] = {"tau": round(tau, 4),
                        "realized_test_fpr": round(realized, 4) if realized is not None else None}
        summary["fixed_fpr_thresholds"][f"target_{target}"] = entry

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "NULL_FP_AUDIT.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))

    # ── 打印 ──
    ul = summary["unit_level"]
    print("\n" + "=" * 84)
    print("单元级：空文上的虚假 green 率（官方规则 max>=0.65）")
    print("=" * 84)
    print(f"  单元总数 {ul['n']}，平均每单元 {ul['mean_subgraphs']} 个子图")
    for k in ("d1", "d2", "d3", "d4", "d5"):
        print(f"  {k.upper():<3} 虚假 green 率 = {ul[f'spurious_green_rate_{k}']:.2%}")
    print(f"\n  单元最高分（D1）: {ul['top_score_d1']}")
    print(f"  单元最高分（D5）: {ul['top_score_d5']}")

    print("\n" + "=" * 84)
    print("文档级 z 分布（D1 = 现役行为）")
    print("=" * 84)
    for cohort, rows in (("校准", summary['doc_level_z']['calibration']),
                         ("独立测试", summary['doc_level_z']['independent_test'])):
        print(f"  [{cohort}]")
        for k in ("d1", "d2", "d3", "d4", "d5"):
            d = rows[k]
            print(f"    {k.upper():<3} n={d['n']:<4} 中位={d['median']:>7} "
                  f"p95={d['p95']:>7} p99={d['p99']:>7} max={d['max']:>8}")

    print("\n" + "=" * 84)
    print("校准集定阈值 → 独立测试集真实 FPR")
    print("=" * 84)
    for target, entry in summary["fixed_fpr_thresholds"].items():
        print(f"  目标 {target}")
        for k, v in entry.items():
            print(f"    {k.upper():<3} τ={v['tau']:>8}  实测FPR={v['realized_test_fpr']:.2%}")
    print(f"\nS2Match 调用: {scorer.calls}")
    print(f"written: {args.output / 'NULL_FP_AUDIT.json'}")


if __name__ == "__main__":
    main()
