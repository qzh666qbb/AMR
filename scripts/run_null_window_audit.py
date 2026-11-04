"""对称口径版零假设审计：零例侧改用与正例完全相同的候选窗口枚举。

**为什么要做这个**：旧零例侧把"句界单元"本身当作一个窗口（每单元 1 个机会），
而正例侧对每个载体取了"所有重叠候选窗口的逐元素最大"（每载体约 3.6 个机会）。
机会数不同 → 正例侧统计量系统性偏高 → TPR 被高估。本脚本消除该不对称。

**对称口径定义（两侧逐字相同）**：
  给定一个查询 span（正例 = 载体 char span；零例 = 句界单元 span）
  → 枚举所有与之字符重叠的候选窗口
  → 每个窗口算 d1/d2 分数向量
  → 逐元素取最大，得到该查询 span 的聚合向量

这样两侧的"机会数分布"由同一套候选枚举规则决定，不再有系统性偏向。
"""
from __future__ import annotations

import argparse
import html
import json
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
    BANK, S2Scorer, elementwise_max, event_subgraphs,
)
from run_null_fp_audit import GREEN_THRESHOLD, UNIT_CFG, load_windows, zscore  # noqa: E402

NULL_DIR = ROOT / "experiments/swan_elastic_formal_null1/inputs"
DEFAULT_OUT = ROOT / "experiments/swan_multiview_development1/null_fp_audit1"
# 与正例侧 WINDOW 配置逐字一致
WINDOW = WindowConfig(lengths=(12,), stride=6, max_words=24, max_candidates=128)
GATE_THRESHOLD = 0.75


def load_range(path: Path, offset: int, limit: int) -> list:
    """从 offset 开始取 limit 条（用于多进程分片）。"""
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        if i < offset:
            continue
        if limit and len(rows) >= limit:
            break
        rows.append(html.unescape(json.loads(line)["window"]))
    return rows


def vec_of(amr: str, scorer: S2Scorer, max_depth: int):
    """返回该 AMR 的 d1 / d2 分数向量与子图数。"""
    d1 = scorer.scores(amr)
    subs = event_subgraphs(amr, max_depth=max_depth)
    d2 = [None] * len(scorer.bank)
    for sub in subs:
        d2 = elementwise_max(d2, scorer.scores(sub))
    return d1, d2, len(subs)


def top(vals):
    return max((v for v in vals if v is not None), default=None)


def overlap(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache", type=Path,
                    default=Path("/tmp/amr_multiview_20261020/null_window_audit.db"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--calibration-docs", type=int, default=40)
    ap.add_argument("--test-docs", type=int, default=40)
    ap.add_argument("--cal-offset", type=int, default=0)
    ap.add_argument("--test-offset", type=int, default=0)
    ap.add_argument("--no-calibration", action="store_true",
                    help="只跑测试半区（补片用，避免重复解析校准文档）")
    ap.add_argument("--window-lengths", nargs="+", type=int, default=[12],
                    help="候选窗口词数（默认 12；短窗基线用 6）")
    ap.add_argument("--window-stride", type=int, default=6)
    args = ap.parse_args()

    window = WindowConfig(lengths=tuple(args.window_lengths),
                          stride=args.window_stride,
                          max_words=max(24, 2 * max(args.window_lengths)),
                          max_candidates=128)

    bank_amr = [r["normalized_template_amr"]
                for r in json.loads(BANK.read_text())]
    calib = [] if args.no_calibration else load_range(
        NULL_DIR / "calibration_windows.jsonl", args.cal_offset, args.calibration_docs)
    test = load_range(NULL_DIR / "independent_test_windows.jsonl",
                      args.test_offset, args.test_docs)
    print(f"对称口径零假设：校准 {len(calib)} (offset {args.cal_offset})，"
          f"测试 {len(test)} (offset {args.test_offset})", flush=True)

    backend = AmrlibBackend(
        str(ROOT / "models/model_parse_xfm_bart_large-v0_1_0"),
        str(ROOT / "models/bart-large-tokenizer"),
        device=args.device, batch_size=args.batch_size)
    cache = AMRCache(str(args.cache))
    extractor = AMRExtractor(backend, cache, request_batch_size=args.batch_size)
    scorer = S2Scorer(bank_amr)

    units = []
    out_path = args.output
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt = out_path / "PARTIAL_WINDOW.json"

    def checkpoint():
        ckpt.write_text(json.dumps({
            "unit_definition": "句界单元 span；分数 = 与之重叠的所有候选窗口的逐元素最大",
            "window_config": {"lengths": list(WINDOW.lengths), "stride": WINDOW.stride,
                              "max_words": WINDOW.max_words},
            "units": len(units), "s2match_calls": scorer.calls,
            "per_unit": units,
        }, ensure_ascii=False, indent=2))

    for cohort, docs in (("calibration", calib), ("independent_test", test)):
        base = args.cal_offset if cohort == "calibration" else args.test_offset
        for di, text in enumerate(docs):
            doc_id = base + di          # 全局唯一编号，避免多分片撞号
            cands = candidate_windows(text, window)["candidates"]
            if not cands:
                continue
            # 先解析所有候选窗口
            texts = list(dict.fromkeys(normalize_text(c["text"]) for c in cands))
            records = extractor.extract_many(texts)
            # 预先算好每个候选窗口的向量
            vecs = {}
            for c in cands:
                rec = records.get(normalize_text(c["text"]))
                if rec is None or not rec.graph_valid:
                    continue
                if c["text"] not in vecs:
                    vecs[c["text"]] = vec_of(rec.amr, scorer, args.max_depth)

            # 单元 = 句界候选的 span；聚合其所有重叠候选窗口
            seen = set()
            for unit_c in cands:
                if "surface_boundary" not in unit_c["kinds"]:
                    continue
                key = (unit_c["start_char"], unit_c["end_char"])
                if key in seen:
                    continue
                seen.add(key)
                d1 = [None] * len(bank_amr)
                d2 = [None] * len(bank_amr)
                n_win = 0
                n_sub = 0
                for c in cands:
                    if not overlap(key, (c["start_char"], c["end_char"])):
                        continue
                    v = vecs.get(c["text"])
                    if v is None:
                        continue
                    n_win += 1
                    n_sub += v[2]
                    d1 = elementwise_max(d1, v[0])
                    d2 = elementwise_max(d2, v[1])
                if n_win == 0:
                    continue
                units.append({
                    "cohort": cohort, "doc": doc_id,
                    "unit": unit_c["text"][:120],
                    "word_count": unit_c["word_count"],
                    "n_windows": n_win, "n_subgraphs": n_sub,
                    "top_d1": None if top(d1) is None else round(top(d1), 4),
                    "top_d2": None if top(d2) is None else round(top(d2), 4),
                    "vec_d1": [None if v is None else round(v, 4) for v in d1],
                    "vec_d2": [None if v is None else round(v, 4) for v in d2],
                })
            if (di + 1) % 5 == 0:
                print(f"  [{cohort}] {di+1}/{len(docs)} units={len(units)} "
                      f"calls={scorer.calls}", flush=True)
                checkpoint()

    checkpoint()
    n_win = [u["n_windows"] for u in units]
    summary = {
        "note": "对称口径零假设：单元分数 = 与之重叠的所有候选窗口的逐元素最大",
        "unit_definition": "句界单元 span + 窗口聚合",
        "window_config": {"lengths": list(WINDOW.lengths), "stride": WINDOW.stride,
                          "max_words": WINDOW.max_words},
        "cohorts": {c: sum(1 for u in units if u["cohort"] == c)
                    for c in ("calibration", "independent_test")},
        "units": len(units),
        "windows_per_unit_mean": round(sum(n_win) / len(n_win), 2) if n_win else 0,
        "s2match_calls": scorer.calls,
    }
    (out_path / "WINDOW_AUDIT_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n" + "=" * 70)
    print(f"单元数 {len(units)}，平均每单元重叠窗口 "
          f"{summary['windows_per_unit_mean']} 个")
    print(f"（正例侧每载体重叠窗口数需与之一致，才能算对称）")
    print(f"written: {ckpt}")
    print(f"written: {out_path / 'WINDOW_AUDIT_SUMMARY.json'}")


if __name__ == "__main__":
    main()
