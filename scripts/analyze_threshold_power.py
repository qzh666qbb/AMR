"""阈值的功效诊断：固定 FPR 下的比较到底需要多少零例样本。

**动机**：D5/D6/D7/D8 四种融合方案都在 AUC 上有改善，但在固定 FPR 下拿不到
显著增益。在把原因归给"方法不行"之前，必须先排除**统计功效不足**：
1% FPR 在 285 个零例单元上就是第 3 高的分数，阶统计量的方差极大。

本脚本量化：
  1. 阈值本身的抽样标准差随零例规模 n 如何变化；
  2. 由此导致的 TPR@固定FPR 的置信区间宽度；
  3. 要让阈值 SD 降到给定水平，需要多少零例单元。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NULL = ROOT / "experiments/swan_multiview_development1/null_fp_audit1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--null-file", type=Path,
                    default=NULL / "PARTIAL_WINDOW_MERGED.json")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--output", type=Path, default=NULL / "THRESHOLD_POWER.json")
    args = ap.parse_args()

    units = [u for u in json.loads(args.null_file.read_text())["per_unit"]
             if u.get("vec_d1") and u.get("vec_d2")]
    # 用 D1 分数（现役口径）做阈值稳定性分析
    scores = np.array([max(v for v in u["vec_d1"] if v is not None) for u in units])
    print(f"零例池：{len(scores)} 个单元（校准+测试合并，仅用于稳定性诊断）")
    print(f"分数范围 [{scores.min():.4f}, {scores.max():.4f}]，"
          f"中位 {np.median(scores):.4f}")

    rng = np.random.default_rng(args.seed)
    sizes = [100, 200, 285, 400, 589]
    qs = [0.01, 0.05]
    rows = {}
    print(f"\n{'n':>6} | " + " | ".join(
        f"{'q=' + format(q, '.0%') + ' τ的SD':>18} {'τ 5-95%区间':>22}" for q in qs))
    print("-" * 96)
    for n in sizes:
        if n > len(scores):
            continue
        rec = {}
        for q in qs:
            taus = []
            for _ in range(args.draws):
                idx = rng.integers(0, len(scores), n)
                s = np.sort(scores[idx])[::-1]
                k = max(1, math.ceil(q * n))
                taus.append(s[min(k, n) - 1])
            taus = np.array(taus)
            rec[f"q{q}"] = {
                "tau_mean": round(float(taus.mean()), 4),
                "tau_sd": round(float(taus.std(ddof=1)), 4),
                "tau_p05": round(float(np.percentile(taus, 5)), 4),
                "tau_p95": round(float(np.percentile(taus, 95)), 4),
                "order_stat": max(1, math.ceil(q * n)),
            }
        rows[n] = rec
        cells = f"{n:>6} | "
        for q in qs:
            r = rec[f"q{q}"]
            cells += (f"{r['tau_sd']:>18.4f} "
                      f"[{r['tau_p05']:>8.4f},{r['tau_p95']:>8.4f}]  ")
        print(cells)

    # 外推：阈值 SD ∝ 1/sqrt(n)（阶统计量的标准标度）
    base_n = 285
    if base_n in rows:
        sd1 = rows[base_n]["q0.01"]["tau_sd"]
        targets = [0.02, 0.01, 0.005]
        print(f"\n外推：若要让 1% FPR 阈值的 SD 从 {sd1:.4f} 降到——")
        needed = {}
        for t in targets:
            n_req = int(math.ceil(base_n * (sd1 / t) ** 2))
            print(f"  {t:.3f}（降 {sd1/t:.1f}×）→ 需零例单元 ≈ {n_req}")
            needed[str(t)] = n_req
    else:
        needed = {}

    payload = {
        "note": "阈值抽样稳定性诊断；用 D1 分数（max over bank）",
        "null_pool": int(len(scores)),
        "draws": args.draws,
        "by_size": rows,
        "required_n_extrapolation": needed,
        "order_stat_note": "1% FPR 在 n=285 时阈值是第 3 高分数，阶统计量方差极大",
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
