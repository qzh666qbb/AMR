"""聚类 bootstrap：修正"224 实例 = 32 载体 × 7 攻击"带来的 CI 低估。

**问题**：`dataset4_fusion1` 的 224 条记录里，唯一 (document, slot) 载体只有 32 个。
同一载体在 7 种攻击变体下的分数高度相关（同一句话的两种标点写法）。
之前的 bootstrap 在**实例层**重采样，把 7 倍相关样本当独立 → CI 被人为压窄。

**修正**：以 (document, slot) 为**聚类**重采样，同一聚类的 7 条记录整体进出。
这给出诚实的 CI。若显著性在聚类 bootstrap 下消失，则原结论不成立。
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
SIGMA_FLOOR = 1e-6


def build_ecdf(units, key):
    n_tpl = len(units[0][key])
    cols = [[] for _ in range(n_tpl)]
    for u in units:
        for t, v in enumerate(u[key]):
            if v is not None:
                cols[t].append(v)
    return [sorted(c) for c in cols]


def make_stats(calib):
    ecdf = {s: build_ecdf(calib, f"vec_{s}") for s in ("d1", "d2")}

    def p_emp(scope, t, score):
        arr = ecdf[scope][t]
        n = len(arr)
        return (1 + n - bisect.bisect_left(arr, score)) / (1 + n)

    def d7a(u):                       # 分位映射 + Fisher（极值聚合）
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

    def d1(u):                        # 现役：整图 max over bank
        v = u.get("vec_d1")
        if v is None:
            return None
        vals = [x for x in v if x is not None]
        return max(vals) if vals else None

    return {"D7a": d7a, "D1": d1}


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


def tpr_at(pos_vals, neg_vals, q):
    nd = sorted([v for v in neg_vals if v is not None], reverse=True)
    pv = [v for v in pos_vals if v is not None]
    if not nd or not pv:
        return float("nan")
    k = max(1, math.ceil(q * len(nd)))
    tau = nd[min(k, len(nd)) - 1]
    return sum(1 for v in pv if v >= tau) / len(pv)


def pct(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--null-file", type=Path,
                    default=NULL / "PARTIAL_WINDOW_SCALED.json")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--output", type=Path,
                    default=NULL / "CLUSTER_BOOTSTRAP.json")
    args = ap.parse_args()

    null_units = [u for u in json.loads(args.null_file.read_text())["per_unit"]
                  if u.get("vec_d1") and u.get("vec_d2")]
    rows = [r for r in json.loads((POS / "FUSION_RECOVERY.json").read_text())["per_instance"]
            if r.get("vec_d1") and r.get("vec_d2")]
    calib = [u for u in null_units if u["cohort"] == "calibration"]
    test = [u for u in null_units if u["cohort"] == "independent_test"]
    stats = make_stats(calib)

    # 零例也按文档聚类（同一文档内的单元相关）
    neg_by_doc = {}
    for u in test:
        neg_by_doc.setdefault(u["doc"], []).append(u)
    # 正例按 (doc, slot) 聚类
    pos_by_carrier = {}
    for r in rows:
        pos_by_carrier.setdefault((r["doc"], r["slot"]), []).append(r)

    print(f"零例测试单元 {len(test)}（{len(neg_by_doc)} 个文档聚类）")
    print(f"正例实例 {len(rows)}（{len(pos_by_carrier)} 个载体聚类 × 攻击）")

    # 预先算分
    neg_sc = {k: {d: [f(u) for u in us] for d, us in neg_by_doc.items()}
              for k, f in stats.items()}
    pos_sc = {k: {c: [f(r) for r in rs] for c, rs in pos_by_carrier.items()}
              for k, f in stats.items()}

    neg_docs = list(neg_by_doc)
    pos_clusters = list(pos_by_carrier)

    print("\n观测值（全量，未重采样）：")
    obs = {}
    for k in stats:
        neg_all = [v for d in neg_docs for v in neg_sc[k][d]]
        pos_all = [v for c in pos_clusters for v in pos_sc[k][c]]
        a = auc(pos_all, neg_all)
        obs[k] = {"auc": a,
                  "tpr1": tpr_at(pos_all, neg_all, 0.01),
                  "tpr5": tpr_at(pos_all, neg_all, 0.05)}
        print(f"  {k}: AUC={a:.4f}  TPR@1%={obs[k]['tpr1']:.1%}  "
              f"TPR@5%={obs[k]['tpr5']:.1%}")

    rng = np.random.default_rng(args.seed)
    d_auc, d_t1, d_t5 = [], [], []
    for _ in range(args.draws):
        nd_idx = rng.integers(0, len(neg_docs), len(neg_docs))
        pc_idx = rng.integers(0, len(pos_clusters), len(pos_clusters))
        vals = {}
        for k in stats:
            neg_all = [v for i in nd_idx for v in neg_sc[k][neg_docs[i]]]
            pos_all = [v for i in pc_idx for v in pos_sc[k][pos_clusters[i]]]
            vals[k] = (pos_all, neg_all)
        for k in stats:
            if k == "D7a":
                continue
            pa, na = vals["D7a"]
            pb, nb = vals[k]
            d_auc.append(auc(pa, na) - auc(pb, nb))
            d_t1.append(tpr_at(pa, na, 0.01) - tpr_at(pb, nb, 0.01))
            d_t5.append(tpr_at(pa, na, 0.05) - tpr_at(pb, nb, 0.05))

    payload = {"note": "聚类 bootstrap：(document, slot) 为载体聚类，document 为零例聚类",
               "null_test_units": len(test), "null_doc_clusters": len(neg_docs),
               "positive_instances": len(rows),
               "carrier_clusters": len(pos_clusters),
               "observed": {k: {kk: round(vv, 4) for kk, vv in vv_.items()}
                            for k, vv_ in obs.items()},
               "cluster_delta": {}}

    print(f"\n聚类 bootstrap（{args.draws} 次，载体聚类 {len(pos_clusters)} 个）：")
    print(f"{'对比':<12} {'ΔAUC':>8} {'95%CI':>22} {'P(Δ>0)':>8}")
    print("-" * 56)
    o = obs["D7a"]["auc"] - obs["D1"]["auc"]
    lo, hi = pct(d_auc, 0.025), pct(d_auc, 0.975)
    pg = sum(1 for x in d_auc if x > 0) / len(d_auc)
    print(f"{'D7a − D1':<12} {o:>8.4f} [{lo:>8.4f},{hi:>8.4f}] {pg:>8.3f}")
    payload["cluster_delta"]["auc"] = {"delta": round(o, 4),
                                       "ci95": [round(lo, 4), round(hi, 4)]}

    for label, arr, key_q in (("TPR@1%", d_t1, "tpr1"), ("TPR@5%", d_t5, "tpr5")):
        oo = obs["D7a"][key_q] - obs["D1"][key_q]
        lo, hi = pct(arr, 0.025), pct(arr, 0.975)
        pg = sum(1 for x in arr if x > 0) / len(arr)
        print(f"{label:<12} {oo:>8.4f} [{lo:>8.4f},{hi:>8.4f}] {pg:>8.3f}")
        payload["cluster_delta"][key_q] = {
            "delta": round(oo, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "prob_greater": round(pg, 4)}

    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
