"""6 词滑窗基线（B1）与主实验检测器的同表比较。

B1 的定义：候选窗口缩到 6 词（步长 3），仍用窗口完整 AMR 与模板库匹配，
再对与载体重叠的窗口逐元素取最大。它与 E 的唯一差别是：作用域是"更小的窗口"
而不是"以谓词为锚点的子图"。

比较口径：每个检测器使用各自窗口配置下的零例分布定阈值，
在相同假阳率（1% / 5%）处比较载体级 AUC 与 TPR。
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_ed_ablation import auc, make, tpr  # noqa: E402
from analyze_doc_level_strict import Scope  # noqa: E402

MAIN_NULL = ROOT / "experiments/swan_multiview_development1/null_fp_audit1/PARTIAL_WINDOW_200.json"
MAIN_POS = ROOT / "experiments/swan_multiview_development1/dataset5_eval/FUSION_RECOVERY.json"
W6_NULL = ROOT / "experiments/swan_multiview_development1/null_fp_audit1/window6_null"
W6_POS = ROOT / "experiments/swan_multiview_development1/dataset5_eval/window6_pos"
OUT = ROOT / "experiments/swan_multiview_development1/dataset5_eval/WINDOW6_BASELINE.json"
ATTACKS = ("baseline", "semicolon", "newline", "split", "merge", "comma", "space")


def load_units(shard_dir: Path):
    units = []
    for p in sorted(shard_dir.glob("s*/PARTIAL_WINDOW.json")):
        units.extend(json.loads(p.read_text())["per_unit"])
    return units


def load_rows(shard_dir: Path):
    rows = []
    for p in sorted(shard_dir.glob("s*/FUSION_RECOVERY.json")):
        rows.extend(json.loads(p.read_text())["per_instance"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--w6-null", type=Path, default=W6_NULL)
    ap.add_argument("--w6-pos", type=Path, default=W6_POS)
    args = ap.parse_args()

    m_null = [u for u in json.loads(MAIN_NULL.read_text())["per_unit"]
              if u.get("vec_d1") and u.get("vec_d2")]
    m_cal = [u for u in m_null if u["cohort"] == "calibration"]
    m_test = [u for u in m_null if u["cohort"] == "independent_test"]
    m_rows = [r for r in json.loads(MAIN_POS.read_text())["per_instance"]
              if r.get("vec_d1") and r.get("vec_d2")]

    w_units = [u for u in load_units(args.w6_null) if u.get("vec_d1")]
    w_cal = [u for u in w_units if u["cohort"] == "calibration"]
    w_test = [u for u in w_units if u["cohort"] == "independent_test"]
    w_rows = [r for r in load_rows(args.w6_pos) if r.get("vec_d1")]
    print(f"主实验：校准 {len(m_cal)} / 测试 {len(m_test)} / 正例 {len(m_rows)}")
    print(f"6 词：  校准 {len(w_cal)} / 测试 {len(w_test)} / 正例 {len(w_rows)}")

    det_main = make(m_cal)
    sc = Scope(w_cal, "vec_d1", "empirical")

    def b1_raw(u):
        v = u.get("vec_d1")
        if not v:
            return None
        vals = [x for x in v if x is not None]
        return max(vals) if vals else None

    def b1_q(u):
        v = u.get("vec_d1")
        if not v:
            return None
        best = None
        for t, x in enumerate(v):
            if x is None:
                continue
            val = -2.0 * math.log(sc.p(t, x))
            best = val if best is None or val > best else best
        return best

    det = dict(det_main)
    det["B1 6词+原始"] = b1_raw
    det["B1 6词+分位"] = b1_q

    neg = {"main": m_test, "win6": w_test}
    pos = {"main": m_rows, "win6": w_rows}

    def split_scope(name):
        return "win6" if name.startswith("B1") else "main"

    print("\n" + "=" * 78)
    print(f"{'检测器':<18}{'AUC':>9}{'TPR@1%':>9}{'TPR@5%':>9}")
    print("-" * 78)
    payload = {"results": {}}
    scores = {}
    for name, f in det.items():
        s = split_scope(name)
        negv = [f(u) for u in neg[s]]
        posv = [f(r) for r in pos[s]]
        a = auc(posv, negv)
        t1, t5 = tpr(posv, negv, 0.01), tpr(posv, negv, 0.05)
        print(f"{name:<18}{a:>9.4f}{t1:>9.1%}{t5:>9.1%}")
        payload["results"][name] = {"auc": round(a, 4),
                                    "tpr1": round(t1, 4), "tpr5": round(t5, 4)}
        scores[name] = (posv, negv)

    # 逐攻击（1% 工作点）
    print("\n逐攻击 TPR@1%")
    print(f"{'attack':<10}" + "".join(f"{n[:14]:>16}" for n in det))
    for atk in ATTACKS:
        cells = f"{atk:<10}"
        for name, f in det.items():
            s = split_scope(name)
            sub = [r for r in pos[s] if r["attack"] == atk]
            negv = scores[name][1]
            cells += f"{tpr([f(r) for r in sub], negv, 0.01):>16.1%}"
        print(cells)
        payload.setdefault("per_attack", {})[atk] = {
            name: round(tpr([f(r) for r in pos[split_scope(name)] if r["attack"] == atk],
                            scores[name][1], 0.01), 4)
            for name, f in det.items()}

    # 聚类 bootstrap：B1 分位 vs E / A（同一载体/单元配对，分数预先算好）
    rng = np.random.default_rng(args.seed)
    b1_pos = {(r["doc"], r["slot"], r["attack"]): b1_q(r) for r in w_rows}
    b1_neg = {(u["cohort"], u["doc"], u.get("unit")): b1_q(u) for u in w_test}
    pos_pairs = [((r["doc"], r["slot"]), r) for r in m_rows
                 if (r["doc"], r["slot"], r["attack"]) in b1_pos]
    neg_pairs = [((u["cohort"], u["doc"]), u) for u in m_test
                 if (u["cohort"], u["doc"], u.get("unit")) in b1_neg]
    print(f"\n配对：正例 {len(pos_pairs)}/{len(m_rows)}，零例 {len(neg_pairs)}/{len(m_test)}")

    for base in ("E 局部+分位", "A 整图+原始"):
        pos_cl = {}
        for ck, r in pos_pairs:
            pos_cl.setdefault(ck, []).append(
                (det[base](r), b1_pos[(r["doc"], r["slot"], r["attack"])]))
        neg_doc = {}
        for ck, u in neg_pairs:
            neg_doc.setdefault(ck, []).append(
                (det[base](u), b1_neg[(u["cohort"], u["doc"], u.get("unit"))]))
        pk, nk = list(pos_cl), list(neg_doc)
        diffs = []
        for _ in range(args.boot):
            psel = [pos_cl[pk[i]] for i in rng.integers(0, len(pk), len(pk))]
            nsel = [neg_doc[nk[i]] for i in rng.integers(0, len(nk), len(nk))]
            pv_e = [x for c in psel for x, _ in c]
            pv_b = [x for c in psel for _, x in c]
            nv_e = [x for c in nsel for x, _ in c]
            nv_b = [x for c in nsel for _, x in c]
            diffs.append(tpr(pv_b, nv_b, 0.01) - tpr(pv_e, nv_e, 0.01))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        payload[f"delta_B1q_minus_{base}"] = {
            "mean": round(float(np.mean(diffs)), 4),
            "ci": [round(float(lo), 4), round(float(hi), 4)]}
        print(f"B1分位 − {base}: ΔTPR@1% = {np.mean(diffs):+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]")

    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
