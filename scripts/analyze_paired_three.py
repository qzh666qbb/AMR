"""难负例 + 目标模板条件下的 A / E / B1 三方对比。

- 负例：dataset5_paired_neg（同生成器、同风格、不同事件）。
- 校准：负例文档按编号对半；阈值只来自校准半区。
- 条件：只对每个载体配对的模板打分（target-conditioned），避免 100 选 1 的尾部饱和。
- A：整图窗口分数；E：事件子图分数；B1：6 词 / 步长 1 窗口分数。
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

MAIN_POS = ROOT / "experiments/swan_multiview_development1/dataset5_eval/FUSION_RECOVERY.json"
MAIN_NEG = ROOT / "experiments/swan_multiview_development1/dataset5_paired_neg_eval/FUSION_RECOVERY.json"
B1_POS_DIR = ROOT / "experiments/swan_multiview_development1/dataset5_eval/window6s1_pos"
B1_NEG_DIR = ROOT / "experiments/swan_multiview_development1/dataset5_paired_neg_eval/window6s1_neg"
OUT = ROOT / "experiments/swan_multiview_development1/dataset5_eval/PAIRED_THREE_WAY.json"
ATTACKS = ("baseline", "semicolon", "newline", "split", "merge", "comma", "space")


def load_single(path: Path):
    return json.loads(path.read_text())["per_instance"]


def load_shards(root: Path):
    rows = []
    for p in sorted(root.glob("s*/FUSION_RECOVERY.json")):
        rows.extend(json.loads(p.read_text())["per_instance"])
    return rows


def build_cols(rows, view):
    cols = [[] for _ in range(100)]
    for r in rows:
        vec = r.get(view)
        if not vec:
            continue
        for t, v in enumerate(vec):
            if v is not None:
                cols[t].append(v)
    return [sorted(c) for c in cols]


def phi_one(row, view, cache, cal_rows):
    t = row.get("template_index")
    vec = row.get(view)
    if vec is None or t is None or t >= len(vec) or vec[t] is None:
        return None
    key = (view, t)
    if key not in cache:
        col = sorted(r[view][t] for r in cal_rows
                     if r.get(view) and t < len(r[view]) and r[view][t] is not None)
        cache[key] = col
    col = cache[key]
    n = len(col)
    if not n:
        return None
    p = (1 + (n - bisect.bisect_left(col, vec[t]))) / (1 + n)
    return -2.0 * math.log(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-pos", type=Path, default=MAIN_POS)
    ap.add_argument("--main-neg", type=Path, default=MAIN_NEG)
    ap.add_argument("--b1-pos-dir", type=Path, default=B1_POS_DIR)
    ap.add_argument("--b1-neg-dir", type=Path, default=B1_NEG_DIR)
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--boot", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=20261022)
    args = ap.parse_args()

    mp = load_single(args.main_pos)
    mn = load_single(args.main_neg)
    bp = load_shards(args.b1_pos_dir)
    bn = load_shards(args.b1_neg_dir)
    print(f"A/E 正例 {len(mp)}  负例 {len(mn)}")
    print(f"B1  正例 {len(bp)}  负例 {len(bn)}")

    docs = sorted({r["doc"] for r in mn})
    half = len(docs) // 2
    cal_docs, test_docs = set(docs[:half]), set(docs[half:])
    mn_cal = [r for r in mn if r["doc"] in cal_docs]
    mn_test = [r for r in mn if r["doc"] in test_docs]
    bn_cal = [r for r in bn if r["doc"] in cal_docs]
    bn_test = [r for r in bn if r["doc"] in test_docs]
    print(f"负例文档：校准 {len(cal_docs)} / 测试 {len(test_docs)}")

    cache = {}
    for r in mp:
        r["_A"] = phi_one(r, "vec_d1", cache, mn_cal)
        r["_E"] = phi_one(r, "vec_d2", cache, mn_cal)
    for r in mn_test:
        r["_A"] = phi_one(r, "vec_d1", cache, mn_cal)
        r["_E"] = phi_one(r, "vec_d2", cache, mn_cal)
    cache_b1 = {}
    for r in bp:
        r["_B1"] = phi_one(r, "vec_d1", cache_b1, bn_cal)
    for r in bn_test:
        r["_B1"] = phi_one(r, "vec_d1", cache_b1, bn_cal)

    P = {(r["doc"], r["slot"], r["attack"]): r for r in mp}
    N = {(r["doc"], r["slot"], r["attack"]): r for r in mn_test}
    BP = {(r["doc"], r["slot"], r["attack"]): r for r in bp}
    BN = {(r["doc"], r["slot"], r["attack"]): r for r in bn_test}
    keys = sorted(set(P) & set(N) & set(BP) & set(BN))
    print(f"四方配对：{len(keys)}")
    if not keys:
        raise SystemExit("没有可配对样本")

    rng = np.random.default_rng(args.seed)
    by_doc = defaultdict(list)
    for k in keys:
        by_doc[k[0]].append(k)
    pdocs = sorted(by_doc)

    def vals(k):
        return P[k].get("_A"), P[k].get("_E"), BP[k].get("_B1"), \
               N[k].get("_A"), N[k].get("_E"), BN[k].get("_B1")

    def tpr_at(ks, det, q):
        """det: 0=A, 1=E, 2=B1；正例取 v[det]，负例取 v[det+3]。"""
        p, n = [], []
        for k in ks:
            v = vals(k)
            if v[det] is not None:
                p.append(v[det])
            if v[det + 3] is not None:
                n.append(v[det + 3])
        return tpr(p, n, q) if p and n else float("nan")

    def auc_at(ks, pi, ni):
        p, n = [], []
        for k in ks:
            v = vals(k)
            if v[pi] is not None:
                p.append(v[pi])
            if v[ni] is not None:
                n.append(v[ni])
        return auc(p, n) if p and n else float("nan")

    def boot_diff(det_a, det_b, q):
        out = []
        for _ in range(args.boot):
            sample = []
            for i in rng.integers(0, len(pdocs), len(pdocs)):
                sample.extend(by_doc[pdocs[i]])
            a = tpr_at(sample, det_a, q)
            b = tpr_at(sample, det_b, q)
            if np.isfinite(a) and np.isfinite(b):
                out.append(a - b)
        if not out:
            return [None, None]
        return [round(float(np.percentile(out, 2.5)), 4),
                round(float(np.percentile(out, 97.5)), 4)]

    result = {"paired": len(keys), "cal_docs": len(cal_docs),
              "test_docs": len(test_docs), "per_attack": {}}
    print(f"\n{'attack':10} {'A AUC':>7} {'E AUC':>7} {'B1 AUC':>7} "
          f"{'A@1%':>7} {'E@1%':>7} {'B1@1%':>7} {'A@5%':>7} {'E@5%':>7} {'B1@5%':>7}")
    for a in ATTACKS:
        ks = [k for k in keys if k[2] == a]
        row = {
            "n": len(ks),
            "A_auc": round(float(auc_at(ks, 0, 3)), 4),
            "E_auc": round(float(auc_at(ks, 1, 4)), 4),
            "B1_auc": round(float(auc_at(ks, 2, 5)), 4),
            "A_tpr1": round(float(tpr_at(ks, 0, 0.01)), 4),
            "E_tpr1": round(float(tpr_at(ks, 1, 0.01)), 4),
            "B1_tpr1": round(float(tpr_at(ks, 2, 0.01)), 4),
            "A_tpr5": round(float(tpr_at(ks, 0, 0.05)), 4),
            "E_tpr5": round(float(tpr_at(ks, 1, 0.05)), 4),
            "B1_tpr5": round(float(tpr_at(ks, 2, 0.05)), 4),
        }
        result["per_attack"][a] = row
        print(f"{a:10} {row['A_auc']:7.4f} {row['E_auc']:7.4f} {row['B1_auc']:7.4f} "
              f"{row['A_tpr1']:7.1%} {row['E_tpr1']:7.1%} {row['B1_tpr1']:7.1%} "
              f"{row['A_tpr5']:7.1%} {row['E_tpr5']:7.1%} {row['B1_tpr5']:7.1%}")

    result["overall"] = {
        "A_auc": round(float(auc_at(keys, 0, 3)), 4),
        "E_auc": round(float(auc_at(keys, 1, 4)), 4),
        "B1_auc": round(float(auc_at(keys, 2, 5)), 4),
        "A_tpr1": round(float(tpr_at(keys, 0, 0.01)), 4),
        "E_tpr1": round(float(tpr_at(keys, 1, 0.01)), 4),
        "B1_tpr1": round(float(tpr_at(keys, 2, 0.01)), 4),
        "A_tpr5": round(float(tpr_at(keys, 0, 0.05)), 4),
        "E_tpr5": round(float(tpr_at(keys, 1, 0.05)), 4),
        "B1_tpr5": round(float(tpr_at(keys, 2, 0.05)), 4),
        "B1_minus_E_tpr1": round(float(tpr_at(keys, 2, 0.01) - tpr_at(keys, 1, 0.01)), 4),
        "B1_minus_E_tpr1_ci": boot_diff(2, 1, 0.01),
        "B1_minus_A_tpr1": round(float(tpr_at(keys, 2, 0.01) - tpr_at(keys, 0, 0.01)), 4),
        "B1_minus_A_tpr1_ci": boot_diff(2, 0, 0.01),
        "E_minus_A_tpr1": round(float(tpr_at(keys, 1, 0.01) - tpr_at(keys, 0, 0.01)), 4),
        "E_minus_A_tpr1_ci": boot_diff(1, 0, 0.01),
    }
    o = result["overall"]
    print(f"\n总体 @1%: A={o['A_tpr1']:.1%}  E={o['E_tpr1']:.1%}  B1={o['B1_tpr1']:.1%}")
    print(f"总体 @5%: A={o['A_tpr5']:.1%}  E={o['E_tpr5']:.1%}  B1={o['B1_tpr5']:.1%}")
    print(f"B1−E @1% = {o['B1_minus_E_tpr1']:+.1%}  CI={o['B1_minus_E_tpr1_ci']}")
    print(f"B1−A @1% = {o['B1_minus_A_tpr1']:+.1%}  CI={o['B1_minus_A_tpr1_ci']}")
    print(f"E−A  @1% = {o['E_minus_A_tpr1']:+.1%}  CI={o['E_minus_A_tpr1_ci']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
