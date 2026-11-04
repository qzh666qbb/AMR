"""难负例（配对负例）下的 A 与 E 对照。

- 负例：同生成器、同文档位置、不同命题、无模板约束（dataset5_paired_neg）。
- 校准：负例文档按编号对半分为校准/测试；阈值只来自校准半区。
- A：整图 max 原始分数。
- E：事件子图 + 分位校准（phi = max_t -2 ln p_t）。
- B1（可选）：6 词/步长 1 + 分位校准，使用同一批负例。

输出：每个攻击下正例对测试负例的 AUC、TPR@1%、TPR@5%，以及配对差值。
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_ed_ablation import auc, tpr  # noqa: E402

POS = ROOT / "experiments/swan_multiview_development1/dataset5_eval/FUSION_RECOVERY.json"
NEG = ROOT / "experiments/swan_multiview_development1/dataset5_paired_neg_eval/FUSION_RECOVERY.json"
OUT = ROOT / "experiments/swan_multiview_development1/dataset5_eval/PAIRED_E_VS_A.json"
ATTACKS = ("baseline", "semicolon", "newline", "split", "merge", "comma", "space")


def max_valid(vec):
    vals = [v for v in (vec or []) if v is not None]
    return max(vals) if vals else None


def build_ecdf(rows, view):
    """每个模板一列排序分数，返回 (sorted_scores_per_template)。"""
    n_t = len(rows[0][view]) if rows and rows[0].get(view) else 100
    cols = [[] for _ in range(n_t)]
    for r in rows:
        vec = r.get(view)
        if not vec:
            continue
        for t, v in enumerate(vec):
            if v is not None:
                cols[t].append(v)
    return [sorted(c) for c in cols]


def phi_of(vec, cols):
    if not vec:
        return None
    best = None
    for t, v in enumerate(vec):
        if v is None or not cols[t]:
            continue
        c = cols[t]
        n = len(c)
        p = (1 + (n - bisect.bisect_left(c, v))) / (1 + n)
        z = -2.0 * math.log(p)
        if best is None or z > best:
            best = z
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=Path, default=POS)
    ap.add_argument("--neg", type=Path, default=NEG)
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--boot", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=20261022)
    args = ap.parse_args()

    pos = json.loads(args.pos.read_text())["per_instance"]
    neg = json.loads(args.neg.read_text())["per_instance"]
    docs = sorted({r["doc"] for r in neg})
    half = len(docs) // 2
    cal_docs, test_docs = set(docs[:half]), set(docs[half:])
    cal_neg = [r for r in neg if r["doc"] in cal_docs]
    test_neg = [r for r in neg if r["doc"] in test_docs]
    print(f"负例文档：校准 {len(cal_docs)} / 测试 {len(test_docs)}；"
          f"单元：校准 {len(cal_neg)} / 测试 {len(test_neg)}")

    cols_d1 = build_ecdf(cal_neg, "vec_d1")
    cols_d2 = build_ecdf(cal_neg, "vec_d2")

    def enrich(rows):
        for r in rows:
            r["_A"] = max_valid(r.get("vec_d1"))
            r["_E"] = phi_of(r.get("vec_d2"), cols_d2)
            r["_Ed1"] = phi_of(r.get("vec_d1"), cols_d1)
        return rows

    enrich(pos)
    enrich(test_neg)
    P = {(r["doc"], r["slot"], r["attack"]): r for r in pos}
    N = {(r["doc"], r["slot"], r["attack"]): r for r in test_neg}
    keys = sorted(set(P) & set(N))

    rng = np.random.default_rng(args.seed)
    by_doc = defaultdict(list)
    for k in keys:
        by_doc[k[0]].append(k)
    kdocs = sorted(by_doc)

    def boot_ci(fn, draws):
        vals = []
        for _ in range(draws):
            sample = []
            for i in rng.integers(0, len(kdocs), len(kdocs)):
                sample.extend(by_doc[kdocs[i]])
            v = fn(sample)
            if v is not None and np.isfinite(v):
                vals.append(v)
        if not vals:
            return [None, None]
        return [round(float(np.percentile(vals, 2.5)), 4),
                round(float(np.percentile(vals, 97.5)), 4)]

    result = {"calibration_docs": len(cal_docs), "test_docs": len(test_docs),
              "paired": len(keys), "per_attack": {}}
    print(f"\n{'attack':10} {'A AUC':>7} {'E AUC':>7} {'A@1%':>7} {'E@1%':>7} "
          f"{'A@5%':>7} {'E@5%':>7}")
    for a in ATTACKS:
        ks = [k for k in keys if k[2] == a]
        pa = [P[k]["_A"] for k in ks if P[k]["_A"] is not None]
        na = [N[k]["_A"] for k in ks if N[k]["_A"] is not None]
        pe = [P[k]["_E"] for k in ks if P[k]["_E"] is not None]
        ne = [N[k]["_E"] for k in ks if N[k]["_E"] is not None]
        row = {
            "n": len(ks),
            "A_auc": round(float(auc(pa, na)), 4),
            "E_auc": round(float(auc(pe, ne)), 4),
            "A_tpr1": round(float(tpr(pa, na, 0.01)), 4),
            "E_tpr1": round(float(tpr(pe, ne, 0.01)), 4),
            "A_tpr5": round(float(tpr(pa, na, 0.05)), 4),
            "E_tpr5": round(float(tpr(pe, ne, 0.05)), 4),
        }
        result["per_attack"][a] = row
        print(f"{a:10} {row['A_auc']:7.4f} {row['E_auc']:7.4f} "
              f"{row['A_tpr1']:7.1%} {row['E_tpr1']:7.1%} "
              f"{row['A_tpr5']:7.1%} {row['E_tpr5']:7.1%}")

    pa = [P[k]["_A"] for k in keys if P[k]["_A"] is not None]
    na = [N[k]["_A"] for k in keys if N[k]["_A"] is not None]
    pe = [P[k]["_E"] for k in keys if P[k]["_E"] is not None]
    ne = [N[k]["_E"] for k in keys if N[k]["_E"] is not None]
    d_e = [P[k]["_E"] - N[k]["_E"] for k in keys
           if P[k]["_E"] is not None and N[k]["_E"] is not None]
    result["overall"] = {
        "A_auc": round(float(auc(pa, na)), 4),
        "E_auc": round(float(auc(pe, ne)), 4),
        "A_tpr1": round(float(tpr(pa, na, 0.01)), 4),
        "E_tpr1": round(float(tpr(pe, ne, 0.01)), 4),
        "A_tpr5": round(float(tpr(pa, na, 0.05)), 4),
        "E_tpr5": round(float(tpr(pe, ne, 0.05)), 4),
        "E_minus_A_tpr1": round(float(tpr(pe, ne, 0.01) - tpr(pa, na, 0.01)), 4),
        "E_minus_A_tpr1_ci": boot_ci(
            lambda s: tpr([P[k]["_E"] for k in s if P[k]["_E"] is not None],
                          [N[k]["_E"] for k in s if N[k]["_E"] is not None], 0.01)
            - tpr([P[k]["_A"] for k in s if P[k]["_A"] is not None],
                  [N[k]["_A"] for k in s if N[k]["_A"] is not None], 0.01),
            args.boot),
        "mean_paired_phi_diff": round(float(np.mean(d_e)), 4) if d_e else None,
    }
    print(f"\n总体：A AUC={result['overall']['A_auc']:.4f}  "
          f"E AUC={result['overall']['E_auc']:.4f}")
    print(f"      A@1%={result['overall']['A_tpr1']:.1%}  "
          f"E@1%={result['overall']['E_tpr1']:.1%}  "
          f"Δ={result['overall']['E_minus_A_tpr1']:+.1%}  "
          f"CI={result['overall']['E_minus_A_tpr1_ci']}")
    print(f"      A@5%={result['overall']['A_tpr5']:.1%}  "
          f"E@5%={result['overall']['E_tpr5']:.1%}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
