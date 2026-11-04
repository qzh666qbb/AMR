"""P0 审计：文档级 FPR + 增益归因四臂消融。

回答两个必须先答的问题：

Q1（安全性）：把载体分数聚合成文档级 z 统计量后，D7a 是否比现役 D1 产生更多假阳？
    这是"方案是否可用"的硬门槛。若 D7a 在 63 个真实 C4 文档上误报更多，其余问题不必再问。

Q2（归因）：D7a 的增益来自"作用域局部打分"还是"分位校准"？
    A = 整图 + 原始分数        （现役 D1）
    B = 整图 + 分位映射        （只加校准）
    C = 作用域局部 + 原始分数  （只加局部）
    D = 双作用域 + 分位+Fisher （D7a）
    若 B ≈ D，则贡献全在校准，"多视图"叙事不成立。
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NULL = ROOT / "experiments/swan_multiview_development1/null_fp_audit1"
POS = ROOT / "experiments/swan_multiview_development1/dataset4_fusion1"
LAMBDA = 0.25


def build_ecdf(units, key):
    n_tpl = len(units[0][key])
    cols = [[] for _ in range(n_tpl)]
    for u in units:
        for t, v in enumerate(u[key]):
            if v is not None:
                cols[t].append(v)
    return [sorted(c) for c in cols]


def make_detectors(calib):
    ecdf = {s: build_ecdf(calib, f"vec_{s}") for s in ("d1", "d2")}

    def p_emp(scope, t, score):
        arr = ecdf[scope][t]
        n = len(arr)
        return (1 + n - bisect.bisect_left(arr, score)) / (1 + n)

    def arm_a(u):                      # 整图 raw
        v = u.get("vec_d1")
        if v is None:
            return None
        vals = [x for x in v if x is not None]
        return max(vals) if vals else None

    def arm_b(u):                      # 整图 分位（单作用域 -2ln p）
        v = u.get("vec_d1")
        if v is None:
            return None
        best = None
        for t in range(len(v)):
            if v[t] is None:
                continue
            x = -2.0 * math.log(p_emp("d1", t, v[t]))
            if best is None or x > best:
                best = x
        return best

    def arm_c(u):                      # 局部 raw
        v = u.get("vec_d2")
        if v is None:
            return None
        vals = [x for x in v if x is not None]
        return max(vals) if vals else None

    def arm_e(u):                      # 局部 分位（单作用域）
        v = u.get("vec_d2")
        if v is None:
            return None
        best = None
        for t in range(len(v)):
            if v[t] is None:
                continue
            x = -2.0 * math.log(p_emp("d2", t, v[t]))
            if best is None or x > best:
                best = x
        return best

    def arm_f(u):                      # 双作用域 raw 取最大（= 朴素融合 D3）
        a, c = arm_a(u), arm_c(u)
        if a is None:
            return c
        if c is None:
            return a
        return max(a, c)

    def arm_d(u):                      # 双作用域 分位 Fisher
        v1, v2 = u.get("vec_d1"), u.get("vec_d2")
        if v1 is None or v2 is None:
            return None
        best = None
        for t in range(len(v1)):
            if v1[t] is None or v2[t] is None:
                continue
            x = -2.0 * (math.log(p_emp("d1", t, v1[t]))
                        + math.log(p_emp("d2", t, v2[t])))
            if best is None or x > best:
                best = x
        return best

    return {"A 整图+原始": arm_a, "B 整图+分位": arm_b,
            "C 局部+原始": arm_c, "E 局部+分位": arm_e,
            "F 双域+原始max": arm_f, "D 双域+分位Fisher": arm_d}


def auc(pos, neg):
    pos = [v for v in pos if v is not None]
    neg = [v for v in neg if v is not None]
    if not pos or not neg:
        return float("nan")
    comb = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks, i = {}, 0
    while i < len(comb):
        j = i
        while j + 1 < len(comb) and comb[j + 1][0] == comb[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[comb[k]] = avg
        i = j + 1
    s = sum(ranks.get((v, 1), 0.0) for v in pos)
    n1, n0 = len(pos), len(neg)
    return (s - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def tpr(pos, neg, q):
    """在零例 q 分位上取阈值，返回正例通过率。"""
    nd = sorted([v for v in neg if v is not None], reverse=True)
    pv = [v for v in pos if v is not None]
    if not nd or not pv:
        return float("nan")
    k = max(1, math.ceil(q * len(nd)))
    tau = nd[min(k, len(nd)) - 1]
    return sum(1 for v in pv if v >= tau) / len(pv)


def zscore(g, n):
    denom = math.sqrt(n * LAMBDA * (1 - LAMBDA))
    if denom < 1e-9:
        return float("nan")
    return (g - LAMBDA * n) / denom


def tau_at(scores, q):
    s = sorted([x for x in scores if x is not None], reverse=True)
    if not s:
        return float("nan")
    k = max(1, math.ceil(q * len(s)))
    return s[min(k, len(s)) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--null-file", type=Path,
                    default=NULL / "PARTIAL_WINDOW_SCALED.json",
                    help="零例合并文件；扩样后用 PARTIAL_WINDOW_200.json")
    ap.add_argument("--output", type=Path, default=NULL / "P0_AUDIT.json")
    ap.add_argument("--pos-file", type=Path, default=POS / "FUSION_RECOVERY.json",
                    help="正例 per-instance 文件（dataset4 或 dataset5）")
    args = ap.parse_args()

    null_units = [u for u in json.loads(args.null_file.read_text())["per_unit"]
                  if u.get("vec_d1") and u.get("vec_d2")]
    rows = [r for r in json.loads(args.pos_file.read_text())["per_instance"]
            if r.get("vec_d1") and r.get("vec_d2")]
    calib = [u for u in null_units if u["cohort"] == "calibration"]
    test = [u for u in null_units if u["cohort"] == "independent_test"]
    det = make_detectors(calib)
    names = list(det)

    # ── 按文档聚合 ──
    null_docs = {}
    for u in test:
        null_docs.setdefault((u["cohort"], u["doc"]), []).append(u)
    pos_docs = {}
    for r in rows:
        # 文档级单位必须是「一份文献」，即 (doc, attack)；不能把 7 种攻击混成一个文档
        pos_docs.setdefault((r["doc"], r["attack"]), []).append(r)

    print("=" * 84)
    print("Q1 文档级 FPR 审计（零例无攻击；测试集文档与单元）")
    print("=" * 84)
    print(f"零例测试文档 {len(null_docs)}，单元 {len(test)}，"
          f"平均 {len(test)/len(null_docs):.1f} 单元/文档")

    # 单元级阈值在「校准半区」上定，文档聚合只在「测试半区」上评
    payload = {"note": "单元阈值取自校准半区；文档级统计量在测试半区上评估",
               "null_test_docs": len(null_docs), "null_test_units": len(test),
               "doc_green": {}, "doc_z": {}, "unit_auc": {}}

    print(f"\n{'检测器':<20} {'单元阈值q=5%':>14} {'文档G均值':>10} {'z均值':>8}"
          f" {'z_p95':>8} {'z最大':>8} {'z>=2.33比例':>12}")
    print("-" * 84)
    for name in names:
        f = det[name]
        tau = tau_at([f(u) for u in calib], 0.05)
        gs, zs = [], []
        for d, us in null_docs.items():
            n = len(us)
            g = sum(1 for u in us if (f(u) is not None and f(u) >= tau))
            gs.append(g)
            zs.append(zscore(g, n))
        zs_sorted = sorted(zs)
        payload["doc_green"][name] = {"mean": round(float(np.mean(gs)), 3),
                                      "tau_q05": round(tau, 4)}
        payload["doc_z"][name] = {
            "mean": round(float(np.mean(zs)), 3),
            "p95": round(float(np.percentile(zs, 95)), 3),
            "max": round(max(zs), 3),
            "frac_ge_2.33": round(sum(1 for z in zs if z >= 2.33) / len(zs), 4),
        }
        print(f"{name:<20} {tau:>14.4f} {np.mean(gs):>10.2f} {np.mean(zs):>8.2f}"
              f" {np.percentile(zs,95):>8.2f} {max(zs):>8.2f}"
              f" {sum(1 for z in zs if z >= 2.33)/len(zs):>12.1%}")

    # ── 载体级 AUC 四臂对比（Q2）──
    print("\n" + "=" * 84)
    print("Q2 四臂消融（载体级 AUC；测试集零例 vs 全部正例）")
    print("=" * 84)
    print(f"{'臂':<22} {'AUC':>9} {'TPR@1%':>9} {'TPR@5%':>9} {'相对A':>9}")
    print("-" * 84)
    base = None
    for name in names:
        f = det[name]
        neg = [f(u) for u in test]
        pos = [f(r) for r in rows]
        a = auc(pos, neg)
        t1, t5 = tpr(pos, neg, 0.01), tpr(pos, neg, 0.05)
        if base is None:
            base = a
        payload["unit_auc"][name] = {"auc": round(a, 4), "tpr1": round(t1, 4),
                                     "tpr5": round(t5, 4)}
        print(f"{name:<22} {a:>9.4f} {t1:>9.1%} {t5:>9.1%} {a-base:>+9.4f}")

    # ── 正例文档级 TPR（只有 2 篇，仅作方向参考）──
    print("\n" + "=" * 84)
    print("正例文档级（每篇 × 每攻击 = 一份文献，N=16；仅 2 篇，方向参考）")
    print("=" * 84)
    for name in names:
        f = det[name]
        tau = tau_at([f(u) for u in calib], 0.05)
        cells = []
        for (d, atk), rs in sorted(pos_docs.items()):
            n = len(rs)
            g = sum(1 for r in rs if f(r) is not None and f(r) >= tau)
            cells.append(f"{atk}:{zscore(g,n):.1f}")
        print(f"  {name:<22} " + " ".join(cells))

    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
