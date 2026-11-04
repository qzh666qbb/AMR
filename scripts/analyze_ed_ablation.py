"""比较 E（局部+分位，单作用域）与 D（双域分位Fisher）与 A（现役）的聚类显著性。

P0 消融显示 E 的 AUC (0.9355) 略高于 D (0.9275)。若差异稳健，
则方法定义应改为"作用域局部 + 分位校准"（单视图），而不是"多视图融合"。
这直接决定论文怎么定位。
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NULL = ROOT / "experiments/swan_multiview_development1/null_fp_audit1"
POS = ROOT / "experiments/swan_multiview_development1/dataset4_fusion1"


def build_ecdf(units, key):
    n_tpl = len(units[0][key])
    cols = [[] for _ in range(n_tpl)]
    for u in units:
        for t, v in enumerate(u[key]):
            if v is not None:
                cols[t].append(v)
    return [sorted(c) for c in cols]


def make(calib):
    ecdf = {s: build_ecdf(calib, f"vec_{s}") for s in ("d1", "d2")}

    def p_emp(scope, t, sc):
        arr = ecdf[scope][t]
        n = len(arr)
        return (1 + n - bisect.bisect_left(arr, sc)) / (1 + n)

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
        for t in range(len(v)):
            if v[t] is None:
                continue
            x = -2.0 * math.log(p_emp("d2", t, v[t]))
            if best is None or x > best:
                best = x
        return best

    def D(u):
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

    return {"A 整图+原始": A, "E 局部+分位": E, "D 双域+分位Fisher": D}


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
    nd = sorted([v for v in neg if v is not None], reverse=True)
    pv = [v for v in pos if v is not None]
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
    ap.add_argument("--pos-file", type=Path,
                    default=POS / "FUSION_RECOVERY.json",
                    help="正例 per-instance 文件")
    ap.add_argument("--dataset", type=Path, default=None,
                    help="正例数据集 documents.json，用于按 (doc,slot) 回查载体文本做去重")
    ap.add_argument("--dedup-text", action="store_true",
                    help="按载体文本去重（保留首次出现）。跨文档重复文本完全相关，不去重会虚高显著性")
    ap.add_argument("--output", type=Path, default=NULL / "ED_ABLATION_CLUSTER.json")
    args = ap.parse_args()

    null_units = [u for u in json.loads(args.null_file.read_text())["per_unit"]
                  if u.get("vec_d1") and u.get("vec_d2")]
    rows = [r for r in json.loads(args.pos_file.read_text())["per_instance"]
            if r.get("vec_d1") and r.get("vec_d2")]
    calib = [u for u in null_units if u["cohort"] == "calibration"]
    test = [u for u in null_units if u["cohort"] == "independent_test"]
    det = make(calib)
    names = list(det)

    neg_by_doc, pos_by_carrier = {}, {}
    for u in test:
        neg_by_doc.setdefault(u["doc"], []).append(u)
    if args.dedup_text:
        # 跨文档重复文本完全相关：只保留首次出现
        slot_text = {}
        if args.dataset:
            for d in json.loads(args.dataset.read_text())["documents"]:
                for s in d["slots"]:
                    slot_text[(d["document_id"], s["slot"])] = \
                        re.sub(r"[.!?;,]$", "", s["text"]).strip().lower()
        seen_txt, kept = set(), []
        for r in rows:
            text = slot_text.get((r["doc"], r["slot"]))
            key = ((text if text is not None else (r["doc"], r["slot"])),
                   r["attack"])          # 必须连 attack 一起，否则每个载体只留 1 个攻击
            if key in seen_txt:
                continue
            seen_txt.add(key)
            kept.append(r)
        print(f"按 (载体文本, 攻击) 去重：{len(rows)} → {len(kept)} 条实例，"
              f"唯一载体 {len({k[0] for k in seen_txt})}")
        rows = kept
    for r in rows:
        pos_by_carrier.setdefault((r["doc"], r["slot"]), []).append(r)
    nd_keys, pc_keys = list(neg_by_doc), list(pos_by_carrier)

    neg_sc = {k: {d: [f(u) for u in us] for d, us in neg_by_doc.items()}
              for k, f in det.items()}
    pos_sc = {k: {c: [f(r) for r in rs] for c, rs in pos_by_carrier.items()}
              for k, f in det.items()}

    print(f"零例测试单元 {len(test)}（{len(nd_keys)} 文档）；"
          f"正例 {len(rows)}（{len(pc_keys)} 载体聚类）")
    print("\n观测：")
    obs = {}
    for k in names:
        na = [v for d in nd_keys for v in neg_sc[k][d]]
        pa = [v for c in pc_keys for v in pos_sc[k][c]]
        obs[k] = {"auc": auc(pa, na), "t1": tpr(pa, na, 0.01), "t5": tpr(pa, na, 0.05)}
        print(f"  {k:<20} AUC={obs[k]['auc']:.4f} "
              f"TPR@1%={obs[k]['t1']:.1%} TPR@5%={obs[k]['t5']:.1%}")

    rng = np.random.default_rng(args.seed)
    pairs = [("E 局部+分位", "A 整图+原始"), ("E 局部+分位", "D 双域+分位Fisher")]
    res = {f"{a} − {b}": {"auc": [], "t1": [], "t5": []} for a, b in pairs}
    for _ in range(args.draws):
        ni = rng.integers(0, len(nd_keys), len(nd_keys))
        pi = rng.integers(0, len(pc_keys), len(pc_keys))
        vals = {}
        for k in names:
            vals[k] = ([v for i in pi for v in pos_sc[k][pc_keys[i]]],
                       [v for i in ni for v in neg_sc[k][nd_keys[i]]])
        for a, b in pairs:
            key = f"{a} − {b}"
            res[key]["auc"].append(auc(*vals[a]) - auc(*vals[b]))
            res[key]["t1"].append(tpr(*vals[a], 0.01) - tpr(*vals[b], 0.01))
            res[key]["t5"].append(tpr(*vals[a], 0.05) - tpr(*vals[b], 0.05))

    payload = {"note": "聚类 bootstrap：(doc, slot) 为载体聚类", "observed": {}, "deltas": {}}
    print(f"\n聚类 bootstrap（{args.draws} 次）：")
    for a, b in pairs:
        key = f"{a} − {b}"
        print(f"\n  {key}")
        entry = {}
        for label, arr, ok in (("ΔAUC", res[key]["auc"], "auc"),
                               ("ΔTPR@1%", res[key]["t1"], "t1"),
                               ("ΔTPR@5%", res[key]["t5"], "t5")):
            o = obs[a][ok] - obs[b][ok]
            lo, hi = pct(arr, 0.025), pct(arr, 0.975)
            print(f"    {label:<9} {o:>8.4f} [{lo:>8.4f},{hi:>8.4f}]  "
                  f"P(Δ>0)={sum(1 for x in arr if x > 0)/len(arr):.3f}")
            entry[ok] = {"delta": round(o, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        payload["deltas"][key] = entry
    payload["observed"] = {k: {kk: round(vv, 4) for kk, vv in v.items()}
                           for k, v in obs.items()}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
