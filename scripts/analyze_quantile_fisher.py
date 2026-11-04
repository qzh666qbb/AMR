"""D7：分位映射 + Fisher 组合。

动机：z 分数无界，其"对全库取极值"的零分布尾部重，导致严格 FPR 下不可分。
分位映射把分数变成 **p 值（有界于 [0,1]）**，尾部被压住；再用 Fisher 组合两个作用域。

  p_s(u, t) = (1 + #{校准集中 S_s(·,t) >= S_s(u,t)}) / (1 + n_calib)     经验上尾分位
  X(u, t)   = -2 * ( ln p_whole(u,t) + ln p_sub(u,t) )                   Fisher 组合
  score(u)  = max_t X(u, t)

对照：
  D1q  单作用域分位映射          score = max_t (-2 ln p_whole)
  D7a  经验分位 + Fisher
  D7b  正态拟合分位 + Fisher（B0 实测零分布近正态：偏度 −0.035、超额峰度 0.117）
  D6b  z 归一化 + 求和（上一轮，作为对照）

同时上报**饱和率**：经验分位在 n_calib 下的分辨率下限为 1/(1+n_calib)，
若大量单元触底，统计量会失去分辨力。这是经验分位映射的已知风险。
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NULL = ROOT / "experiments/swan_multiview_development1/null_fp_audit1"
POS = ROOT / "experiments/swan_multiview_development1/dataset4_fusion1"
SIGMA_FLOOR = 1e-6


# ───────────────────── 经验分位 ─────────────────────

def build_ecdf(units: list, key: str) -> list:
    """按模板构建升序校准分数数组。"""
    n_tpl = len(units[0][key])
    cols = [[] for _ in range(n_tpl)]
    for u in units:
        for t, v in enumerate(u[key]):
            if v is not None:
                cols[t].append(v)
    return [sorted(c) for c in cols]


def p_empirical(sorted_asc: list, score: float) -> float:
    """经验上尾 p 值（保守 +1 修正，避免 p=0）。"""
    n = len(sorted_asc)
    if n == 0:
        return 1.0
    # 严格小于 score 的个数
    below = bisect.bisect_left(sorted_asc, score)
    ge = n - below
    return (1 + ge) / (1 + n)


def build_normal(units: list, key: str) -> tuple:
    n_tpl = len(units[0][key])
    mu, sigma = [], []
    for t in range(n_tpl):
        vals = [u[key][t] for u in units if u[key][t] is not None]
        if len(vals) < 2:
            mu.append(0.0); sigma.append(1.0); continue
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
        mu.append(m); sigma.append(max(math.sqrt(var), SIGMA_FLOOR))
    return mu, sigma


def p_normal(mu: float, sigma: float, score: float) -> float:
    """正态上尾 p 值。"""
    z = (score - mu) / sigma
    return 0.5 * math.erfc(z / math.sqrt(2.0))


# ───────────────────── 打分器 ─────────────────────

class QuantileScorer:
    def __init__(self, calib: list, mode: str):
        self.mode = mode
        if mode == "empirical":
            self.ecdf = {s: build_ecdf(calib, f"vec_{s}") for s in ("d1", "d2")}
        else:
            self.norm = {s: build_normal(calib, f"vec_{s}") for s in ("d1", "d2")}
        self.saturated = 0      # 触底的 (单元, 模板) 对
        self.pairs = 0

    def p(self, scope: str, t: int, score: float) -> float:
        self.pairs += 1
        if self.mode == "empirical":
            arr = self.ecdf[scope][t]
            n = len(arr)
            if n == 0:
                return 1.0
            p = (1 + n - bisect.bisect_left(arr, score)) / (1 + n)
            if p <= 1.0 / (1 + n):
                self.saturated += 1
            return p
        mu, sigma = self.norm[scope]
        p = p_normal(mu[t], sigma[t], score)
        return max(p, 1e-300)

    def statistic(self, unit: dict, use: str) -> float | None:
        """use ∈ {'whole','fisher'}"""
        v1 = unit.get("vec_d1")
        v2 = unit.get("vec_d2")
        if v1 is None:
            return None
        best = None
        for t in range(len(v1)):
            if v1[t] is None:
                continue
            p1 = self.p("d1", t, v1[t])
            x = -2.0 * math.log(p1)
            if use == "fisher":
                if v2 is None or v2[t] is None:
                    continue
                p2 = self.p("d2", t, v2[t])
                x += -2.0 * math.log(p2)
            if best is None or x > best:
                best = x
        return best


def zsum_statistic(unit: dict, fit_norm: dict) -> float | None:
    """D6b：z 归一化后两作用域求和（上一轮方案，作对照）。"""
    v1, v2 = unit.get("vec_d1"), unit.get("vec_d2")
    if v1 is None or v2 is None:
        return None
    best = None
    for t in range(len(v1)):
        if v1[t] is None or v2[t] is None:
            continue
        mu1, s1 = fit_norm["d1"]; mu2, s2 = fit_norm["d2"]
        z = (v1[t] - mu1[t]) / s1[t] + (v2[t] - mu2[t]) / s2[t]
        if best is None or z > best:
            best = z
    return best


# ───────────────────── 评估 ─────────────────────

def auc(pos: list, neg: list) -> float:
    if not pos or not neg:
        return float("nan")
    combined = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks, i = {}, 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[combined[k]] = avg
        i = j + 1
    s = sum(ranks.get((v, 1), 0.0) for v in pos)
    n1, n0 = len(pos), len(neg)
    return (s - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def curve(pos: list, neg: list, targets: list) -> list:
    neg_desc = sorted(neg, reverse=True)
    out = []
    for q in targets:
        k = max(1, math.ceil(q * len(neg_desc)))
        tau = neg_desc[min(k, len(neg_desc)) - 1]
        out.append({"target": q, "tau": round(tau, 4),
                    "fpr": round(sum(1 for x in neg if x >= tau) / len(neg), 4),
                    "tpr": round(sum(1 for x in pos if x >= tau) / len(pos), 4)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=NULL / "QUANTILE_FISHER.json")
    ap.add_argument("--null-file", type=Path, default=NULL / "PARTIAL.json",
                    help="零假设文件；对称口径用 PARTIAL_WINDOW_MERGED.json")
    ap.add_argument("--positive-mode", choices=["max", "target"], default="max",
                    help="正例口径：max=全库最高分；target=真实注入模板")
    args = ap.parse_args()

    null_units = [u for u in json.loads(args.null_file.read_text())["per_unit"]
                  if u.get("vec_d1") and u.get("vec_d2")]
    pos_rows = [r for r in json.loads((POS / "FUSION_RECOVERY.json").read_text())["per_instance"]
                if r.get("vec_d1") and r.get("vec_d2")]
    calib = [u for u in null_units if u["cohort"] == "calibration"]
    test = [u for u in null_units if u["cohort"] == "independent_test"]
    print(f"零假设：校准 {len(calib)}，测试 {len(test)}；正例 {len(pos_rows)}；"
          f"正例口径={args.positive_mode}")

    fit_norm = {s: build_normal(calib, f"vec_{s}") for s in ("d1", "d2")}

    sc_emp = QuantileScorer(calib, "empirical")
    sc_nor = QuantileScorer(calib, "normal")

    variants = {
        "D1q 单作用域分位": lambda u: sc_emp.statistic(u, "whole"),
        "D7a 经验分位+Fisher": lambda u: sc_emp.statistic(u, "fisher"),
        "D7b 正态分位+Fisher": lambda u: sc_nor.statistic(u, "fisher"),
        "D6b z求和(上轮)": lambda u: zsum_statistic(u, fit_norm),
    }

    targets = [0.01, 0.02, 0.05, 0.10, 0.20]
    payload = {"note": "分位映射把分数变为有界 p 值后再 Fisher 组合；阈值在零假设上校准",
               "positive_mode": args.positive_mode,
               "null_calibration": len(calib), "null_test": len(test),
               "positive": len(pos_rows), "variants": {}}

    for name, fn in variants.items():
        neg = [v for v in (fn(u) for u in test) if v is not None]
        pos = [v for v in (fn(r) for r in pos_rows) if v is not None]
        a = auc(pos, neg)
        cv = curve(pos, neg, targets)
        payload["variants"][name] = {"auc": round(a, 4), "curve": cv}
        print(f"\n{'=' * 70}\n{name}   AUC = {a:.4f}\n{'=' * 70}")
        print(f"{'目标FPR':>8} {'阈值':>10} {'实测FPR':>9} {'TPR':>9}")
        for c in cv:
            print(f"{c['target']:>8.0%} {c['tau']:>10.3f} {c['fpr']:>9.2%} {c['tpr']:>9.2%}")

    payload["saturation"] = {
        "empirical_floor": round(1.0 / (1 + len(calib)), 6),
        "pairs_total": sc_emp.pairs,
        "pairs_at_floor": sc_emp.saturated,
        "floor_rate": round(sc_emp.saturated / max(1, sc_emp.pairs), 4),
    }
    print(f"\n饱和诊断（经验分位）：分辨率下限 = 1/(1+{len(calib)}) = "
          f"{1.0/(1+len(calib)):.5f}；触底 (单元,模板) 对 "
          f"{sc_emp.saturated}/{sc_emp.pairs} = "
          f"{sc_emp.saturated/max(1,sc_emp.pairs):.1%}")

    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
