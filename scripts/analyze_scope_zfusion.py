"""D6：作用域级 z 归一化融合，以及完整 FPR-TPR 曲线对比。

动机（上一轮的根因）：子图 S2Match 天然高于整图 S2Match，两者不可比。
RRF 用排名抹平了尺度但丢掉了幅度，导致融合分数无法对接固定阈值。

本方案：**按模板**用零假设分布把每个作用域的分数 z 归一化，再在同一尺度上融合。

  z_s(unit, t) = (S_s(unit, t) - μ_{s,t}) / σ_{s,t}      μ,σ 取自零假设校准集
  Z(unit, t)   = max( z_whole, z_sub )  或  w1*z_whole + w2*z_sub
  score(unit)  = max_t Z(unit, t)

阈值在零假设**测试集**上评定 FPR，在正例载体上评定 TPR。
由于归一化后所有作用域同尺度，阈值可以统一使用（这是上一轮做不到的）。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NULL = ROOT / "experiments/swan_multiview_development1/null_fp_audit1"
POS = ROOT / "experiments/swan_multiview_development1/dataset4_fusion1"
SIGMA_FLOOR = 1e-6


def fit_scope(units: list, key: str) -> tuple:
    """按模板统计 (μ, σ)。units 为校准集零假设单元。"""
    n_tpl = len(units[0][key])
    mu, sigma = [], []
    for t in range(n_tpl):
        vals = [u[key][t] for u in units if u[key][t] is not None]
        if len(vals) < 2:
            mu.append(0.0)
            sigma.append(1.0)
            continue
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
        mu.append(m)
        sigma.append(max(math.sqrt(var), SIGMA_FLOOR))
    return mu, sigma


def score_unit(unit: dict, fit: dict, mode: str, w: tuple) -> float | None:
    """返回该单元的融合分数（对全库取最大）。"""
    z_best = None
    for scope in ("d1", "d2"):
        vec = unit.get(f"vec_{scope}")
        if vec is None:
            return None
        mu, sigma = fit[scope]
        zs = [None if v is None else (v - mu[i]) / sigma[i]
              for i, v in enumerate(vec)]
        if mode == "max":
            if z_best is None:
                z_best = zs
            else:
                z_best = [a if b is None else (b if a is None else max(a, b))
                          for a, b in zip(z_best, zs)]
        else:  # weighted
            if z_best is None:
                z_best = [None if z is None else w[0] * z for z in zs]
            else:
                z_best = [a if b is None else a + w[1] * b
                          for a, b in zip(z_best, zs)]
    vals = [z for z in z_best if z is not None]
    return max(vals) if vals else None


def raw_whole(unit: dict) -> float | None:
    """D1 基准：整图 S2Match 对全库取最大。"""
    vec = unit.get("vec_d1")
    if vec is None:
        return None
    vals = [v for v in vec if v is not None]
    return max(vals) if vals else None


def auc(pos: list, neg: list) -> float:
    """Mann-Whitney AUC（无并列处理，用秩和）。"""
    if not pos or not neg:
        return float("nan")
    combined = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks = {}
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[combined[k]] = avg
        i = j + 1
    # 用值到秩的映射（注意重复值共享平均秩）
    pos_rank_sum = 0.0
    for v in pos:
        pos_rank_sum += ranks.get((v, 1), 0.0)
    n1, n0 = len(pos), len(neg)
    return (pos_rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def curve(pos: list, neg: list, targets: list) -> list:
    """对给定目标 FPR，返回 (实测FPR, 阈值, TPR)。"""
    neg_sorted = sorted(neg, reverse=True)
    out = []
    for q in targets:
        k = max(1, math.ceil(q * len(neg_sorted)))
        tau = neg_sorted[min(k, len(neg_sorted)) - 1]
        fpr = sum(1 for x in neg if x >= tau) / len(neg)
        tpr = sum(1 for x in pos if x >= tau) / len(pos)
        out.append({"target": q, "tau": round(tau, 4),
                    "fpr": round(fpr, 4), "tpr": round(tpr, 4)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=NULL / "SCOPE_ZFUSION.json")
    ap.add_argument("--w1", type=float, default=1.0)
    ap.add_argument("--w2", type=float, default=1.0)
    args = ap.parse_args()

    null_units = json.loads((NULL / "PARTIAL.json").read_text())["per_unit"]
    pos_rows = json.loads((POS / "FUSION_RECOVERY.json").read_text())["per_instance"]
    null_units = [u for u in null_units if u.get("vec_d1") and u.get("vec_d2")]
    pos_rows = [r for r in pos_rows if r.get("vec_d1") and r.get("vec_d2")]
    calib = [u for u in null_units if u["cohort"] == "calibration"]
    test = [u for u in null_units if u["cohort"] == "independent_test"]
    print(f"零假设：校准 {len(calib)}，测试 {len(test)}；正例 {len(pos_rows)}")

    fit = {s: fit_scope(calib, f"vec_{s}") for s in ("d1", "d2")}

    variants = {
        "D1 整图(raw)": lambda u: raw_whole(u),
        "D6a scope-z max": lambda u: score_unit(u, fit, "max", (1.0, 1.0)),
        "D6b scope-z sum": lambda u: score_unit(u, fit, "weighted", (args.w1, args.w2)),
    }

    targets = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
    payload = {"note": "阈值在零假设分布上校准；归一化后各作用域同尺度",
               "null_calibration": len(calib), "null_test": len(test),
               "positive": len(pos_rows), "variants": {}}

    for name, fn in variants.items():
        neg = [fn(u) for u in test]
        pos = [fn(r) for r in pos_rows]
        neg = [v for v in neg if v is not None]
        pos = [v for v in pos if v is not None]
        a = auc(pos, neg)
        cv = curve(pos, neg, targets)
        payload["variants"][name] = {"auc": round(a, 4), "curve": cv}
        print("\n" + "=" * 74)
        print(f"{name}   AUC = {a:.4f}")
        print("=" * 74)
        print(f"{'目标FPR':>8} {'阈值':>9} {'实测FPR':>9} {'TPR':>9}")
        for c in cv:
            print(f"{c['target']:>8.0%} {c['tau']:>9.3f} {c['fpr']:>9.2%} {c['tpr']:>9.2%}")

    # 逐攻击：D1 vs D6a 在同 FPR 下的 TPR
    print("\n" + "=" * 74)
    print("逐攻击 TPR（正例；阈值取自同一零假设测试集分位）")
    print("=" * 74)
    by_attack = {}
    for r in pos_rows:
        by_attack.setdefault(r["attack"], []).append(r)
    detail = {}
    for q in (0.01, 0.05, 0.10):
        taus = {}
        for name, fn in variants.items():
            neg = sorted([x for x in (fn(u) for u in test) if x is not None], reverse=True)
            k = max(1, math.ceil(q * len(neg)))
            taus[name] = neg[min(k, len(neg)) - 1]
        print(f"\n  q = {q:.0%}：" + "  ".join(
            f"{n}=τ{taus[n]:.3f}" for n in variants))
        print(f"  {'attack':<11} {'n':>4} | " + " | ".join(
            f"{n:>14}" for n in variants))
        for attack in sorted(by_attack):
            rows = by_attack[attack]
            cells = f"  {attack:<11} {len(rows):>4} | "
            for name, fn in variants.items():
                vals = [fn(r) for r in rows]
                rate = sum(1 for v in vals if v is not None and v >= taus[name]) / len(rows)
                cells += f"{rate:>14.1%} | "
            print(cells)
            detail.setdefault(attack, {})[f"q{q}"] = {
                name: round(sum(1 for v in (fn(r) for r in rows)
                                if v is not None and v >= taus[name]) / len(rows), 4)
                for name, fn in variants.items()}
    payload["per_attack"] = detail

    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
