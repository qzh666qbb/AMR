"""文档级 TPR：把载体级分数聚合成文档级 z 统计量，补齐最后一个统计口径。

**为什么必须做**：此前所有比较都是**载体级恢复率**。真实水印检测的端点是
"这份文档是否被判为含水印"，需要 $G$（命中单元数）与
$z=(G-0.25N)/\sqrt{N\cdot0.25\cdot0.75}$。

**口径**：
1. 单元级阈值 $\tau_u$：取零例测试集在目标单元 FPR 处的经验分位（实测 FPR 匹配）
2. 文档级 $G$ = 该文档中分数 ≥ $\tau_u$ 的单元数，$N$ = 该文档单元数
3. 文档级阈值 $\tau_z$：取零例文档 z 分布在目标文档 FPR 处的分位
4. TPR = 正例文档中 $z \ge \tau_z$ 的比例

**正例文档单位**：dataset5 的 30 篇 × 7 攻击 = 210 份文献（每份 N=15~20 单元）
**零例文档单位**：63 篇 C4 文档（平均 15.3 单元）
两侧单元数分布接近，可比。
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
EV = ROOT / "experiments/swan_multiview_development1/dataset5_eval"
LAMBDA = 0.25


def build_ecdf(units, key):
    n = len(units[0][key])
    cols = [[] for _ in range(n)]
    for u in units:
        for t, v in enumerate(u[key]):
            if v is not None:
                cols[t].append(v)
    return [sorted(c) for c in cols]


ECDF = None


def p_emp(arr, s):
    n = len(arr)
    return 1.0 if n == 0 else (1 + n - bisect.bisect_left(arr, s)) / (1 + n)


def A(u):
    v = u.get("vec_d1")
    if v is None:
        return None
    vals = [x for x in v if x is not None]
    return max(vals) if vals else None


def E(u):
    v = u.get("vec_d2")
    if v is None:
        return None
    best = None
    for t, x in enumerate(v):
        if x is None:
            continue
        val = -2.0 * math.log(p_emp(ECDF[t], x))
        if best is None or val > best:
            best = val
    return best


def D(u):
    v1, v2 = u.get("vec_d1"), u.get("vec_d2")
    if v1 is None or v2 is None:
        return None
    best = None
    for t in range(len(v1)):
        if v1[t] is None or v2[t] is None:
            continue
        val = -2.0 * (math.log(p_emp(ECDF[t], v1[t]))
                      + math.log(p_emp(ECDF[t], v2[t])))
        if best is None or val > best:
            best = val
    return best


def tau_at(scores, q):
    s = sorted([x for x in scores if x is not None], reverse=True)
    if not s:
        return float("nan")
    k = max(1, math.ceil(q * len(s)))
    return s[min(k, len(s)) - 1]


def zscore(g, n):
    d = math.sqrt(n * LAMBDA * (1 - LAMBDA))
    return float("nan") if d < 1e-9 else (g - LAMBDA * n) / d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=EV / "DOC_LEVEL_TPR.json")
    ap.add_argument("--null", type=Path, default=NULL / "PARTIAL_WINDOW_SCALED.json",
                    help="零例合并文件（默认旧的 63 篇；扩样后用 PARTIAL_WINDOW_200.json）")
    ap.add_argument("--unit-fpr", nargs="+", type=float, default=[0.01, 0.05])
    ap.add_argument("--doc-fpr", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    ap.add_argument("--match-n", type=int, default=0,
                    help="N 对齐：正例文档每次随机抽 match-n 个单元（零例文档天然只有约 6 个）")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261021)
    args = ap.parse_args()

    null = [u for u in json.loads(args.null.read_text())["per_unit"]
            if u.get("vec_d1") and u.get("vec_d2")]
    calib = [u for u in null if u["cohort"] == "calibration"]
    test = [u for u in null if u["cohort"] == "independent_test"]
    rows = [r for r in json.loads((EV / "FUSION_RECOVERY.json").read_text())["per_instance"]
            if r.get("vec_d1") and r.get("vec_d2")]

    global ECDF
    ECDF = build_ecdf(calib, "vec_d2")
    det = {"A 整图+原始": A, "E 局部+分位": E, "D 双域+分位Fisher": D}

    # 零例文档分组：cohort + 全局 doc 编号
    null_docs = {}
    for u in test:
        null_docs.setdefault((u["cohort"], u["doc"]), []).append(u)
    # 正例文档分组：(doc, attack)
    pos_docs = {}
    for r in rows:
        pos_docs.setdefault((r["doc"], r["attack"]), []).append(r)

    ns_null = [len(v) for v in null_docs.values()]
    ns_pos = [len(v) for v in pos_docs.values()]
    print(f"零例文档 {len(null_docs)}，单元数 min/中位/max = "
          f"{min(ns_null)}/{int(np.median(ns_null))}/{max(ns_null)}")
    print(f"正例文档 {len(pos_docs)}（30 篇 × 7 攻击），单元数 min/中位/max = "
          f"{min(ns_pos)}/{int(np.median(ns_pos))}/{max(ns_pos)}")

    payload = {"note": "文档级 z 统计量；单元阈值与文档阈值均按实测 FPR 匹配",
               "null_docs": len(null_docs), "positive_docs": len(pos_docs),
               "null_units_per_doc": {"min": min(ns_null),
                                      "median": float(np.median(ns_null)),
                                      "max": max(ns_null)},
               "results": {}, "per_attack": {}}

    for q_u in args.unit_fpr:
        print("\n" + "=" * 96)
        print(f"单元级 FPR 匹配点 q_u = {q_u:.0%}")
        print("=" * 96)
        for name, f in det.items():
            tau_u = tau_at([f(u) for u in test], q_u)
            null_z = []
            for d, us in null_docs.items():
                g = sum(1 for u in us if (f(u) is not None and f(u) >= tau_u))
                null_z.append(zscore(g, len(us)))
            pos_z = []
            for k, rs in pos_docs.items():
                g = sum(1 for r in rs if (f(r) is not None and f(r) >= tau_u))
                pos_z.append((k, zscore(g, len(rs))))
            pz = [z for _, z in pos_z]

            # N 对齐：零例文档平均只有约 6 个单元，正例有 15~20 个；
            # 不做对齐会系统性高估正例的文档级检验力。这里对正例做多次等 N 抽样。
            draws_z = None
            if args.match_n:
                rng = np.random.default_rng(args.seed)
                draws_z = []
                for _ in range(args.draws):
                    row = {}
                    for k, rs in pos_docs.items():
                        n = len(rs)
                        if n > args.match_n:
                            idx = rng.choice(n, size=args.match_n, replace=False)
                            sel = [rs[i] for i in idx]
                        else:
                            sel = rs
                        g = sum(1 for r in sel if (f(r) is not None and f(r) >= tau_u))
                        row[k] = zscore(g, len(sel))
                    draws_z.append(row)

            def tpr_draws(tau):
                """在等 N 抽样下估计文档 TPR：先算每轮 TPR，再对轮次取平均。"""
                if draws_z is None:
                    return sum(1 for z in pz if z >= tau) / len(pz)
                return float(np.mean([
                    sum(1 for z in row.values() if z >= tau) / len(row)
                    for row in draws_z]))

            entry = {"tau_unit": round(tau_u, 4),
                     "null_z_mean": round(float(np.mean(null_z)), 3),
                     "null_z_p95": round(float(np.percentile(null_z, 95)), 3),
                     "null_z_max": round(max(null_z), 3),
                     "pos_z_mean": round(float(np.mean(pz)), 3),
                     "pos_z_median": round(float(np.median(pz)), 3),
                     "doc_tpr": {}}
            print(f"\n  {name}")
            print(f"    τ_unit={tau_u:.4f}  零例 z: 均值 {np.mean(null_z):.2f} "
                  f"p95 {np.percentile(null_z,95):.2f} max {max(null_z):.2f}")
            print(f"    正例 z: 均值 {np.mean(pz):.2f} 中位 {np.median(pz):.2f}")
            print(f"    {'目标文档FPR':>10} {'τ_z':>7} {'实测零例':>9} {'文档TPR':>9}")
            for q_d in args.doc_fpr:
                tau_z = tau_at(null_z, q_d)
                realized = sum(1 for z in null_z if z >= tau_z) / len(null_z)
                tpr = tpr_draws(tau_z)
                print(f"    {q_d:>10.0%} {tau_z:>7.2f} {realized:>9.1%} {tpr:>9.1%}")
                entry["doc_tpr"][f"docfpr_{q_d}"] = {
                    "tau_z": round(tau_z, 3), "realized_null": round(realized, 4),
                    "tpr": round(tpr, 4)}
            # 固定 2.33（SWAN 惯例阈值）下的表现
            tpr_233 = tpr_draws(2.33)
            null_233 = sum(1 for z in null_z if z >= 2.33) / len(null_z)
            entry["z233"] = {"tpr": round(tpr_233, 4), "null_rate": round(null_233, 4)}
            print(f"    固定 z≥2.33：零例 {null_233:.1%}，文档 TPR {tpr_233:.1%}")
            payload["results"][f"q_u{q_u}::{name}"] = entry

            # 逐攻击
            if q_u == args.unit_fpr[0]:
                for atk in ("baseline", "semicolon", "newline", "split",
                            "merge", "comma", "space"):
                    if draws_z is None:
                        zs = [z for (d, a), z in pos_z if a == atk]
                        frac = (sum(1 for z in zs if z >= 2.33) / len(zs)) if zs else None
                        mean_z = float(np.mean(zs)) if zs else None
                        n_docs = len(zs)
                    else:
                        per_draw = [[row[k] for k in row if k[1] == atk]
                                    for row in draws_z]
                        per_draw = [d for d in per_draw if d]
                        frac = float(np.mean([
                            sum(1 for z in d if z >= 2.33) / len(d) for d in per_draw]))
                        mean_z = float(np.mean([np.mean(d) for d in per_draw]))
                        n_docs = len(per_draw[0])
                    if frac is None:
                        continue
                    payload["per_attack"].setdefault(name, {})[atk] = {
                        "n_docs": n_docs,
                        "z_mean": round(mean_z, 3),
                        "frac_ge_233": round(frac, 4),
                    }

    print("\n" + "=" * 96)
    print(f"逐攻击文档级通过率（固定 z ≥ 2.33，q_u = {args.unit_fpr[0]:.0%}）")
    print("=" * 96)
    print(f"{'attack':<11} " + " ".join(f"{n[:12]:>13}" for n in det))
    for atk in ("baseline", "semicolon", "newline", "split", "merge", "comma", "space"):
        cells = f"{atk:<11} "
        for name in det:
            v = payload["per_attack"].get(name, {}).get(atk)
            cells += f"{(v['frac_ge_233'] if v else float('nan')):>13.1%}"
        print(cells)

    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
