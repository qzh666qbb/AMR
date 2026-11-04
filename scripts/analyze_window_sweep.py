"""窗口尺寸/步长扫描分析：6/8/10 词 × 步长 1/3。

每个配置使用各自窗口下重算的零例定阈值，在 1% / 5% 工作点比较
B1（短窗 + 分位）与 E / A。目的是检验“谓词锚定”的增量是否稳健。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_ed_ablation import auc, make, tpr  # noqa: E402
from analyze_doc_level_strict import Scope  # noqa: E402

NULL_ROOT = ROOT / "experiments/swan_multiview_development1/null_fp_audit1"
POS_ROOT = ROOT / "experiments/swan_multiview_development1/dataset5_eval"
MAIN_NULL = NULL_ROOT / "PARTIAL_WINDOW_200.json"
MAIN_POS = POS_ROOT / "FUSION_RECOVERY.json"
OUT = POS_ROOT / "WINDOW_SWEEP.json"
ATTACKS = ("baseline", "semicolon", "newline", "split", "merge", "comma", "space")


def dirs_for(L, S):
    if (L, S) == (6, 3):
        return NULL_ROOT / "window6_null", POS_ROOT / "window6_pos"
    return NULL_ROOT / f"window{L}s{S}_null", POS_ROOT / f"window{L}s{S}_pos"


def load_units(d):
    out = []
    for p in sorted(d.glob("s*/PARTIAL_WINDOW.json")):
        out.extend(json.loads(p.read_text())["per_unit"])
    return out


def load_rows(d):
    out = []
    for p in sorted(d.glob("s*/FUSION_RECOVERY.json")):
        out.extend(json.loads(p.read_text())["per_instance"])
    return out


def boot_ci(vals_pos, vals_neg, clusters, neg_docs, rng, draws=1000, q=0.01):
    pk, nk = list(clusters), list(neg_docs)
    out = []
    for _ in range(draws):
        p = [x for i in rng.integers(0, len(pk), len(pk)) for x in clusters[pk[i]]]
        n = [x for i in rng.integers(0, len(nk), len(nk)) for x in neg_docs[nk[i]]]
        out.append(tpr(p, n, q))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["6x3", "6x1", "8x1", "8x3", "10x1", "10x3"])
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261021)
    args = ap.parse_args()

    m_null = [u for u in json.loads(MAIN_NULL.read_text())["per_unit"]
              if u.get("vec_d1") and u.get("vec_d2")]
    m_cal = [u for u in m_null if u["cohort"] == "calibration"]
    m_test = [u for u in m_null if u["cohort"] == "independent_test"]
    m_rows = [r for r in json.loads(MAIN_POS.read_text())["per_instance"]
              if r.get("vec_d1") and r.get("vec_d2")]
    det_main = make(m_cal)
    e_neg = [det_main["E 局部+分位"](u) for u in m_test]
    e_pos = [det_main["E 局部+分位"](r) for r in m_rows]
    e_obs = {"auc": auc(e_pos, e_neg), "t1": tpr(e_pos, e_neg, 0.01),
             "t5": tpr(e_pos, e_neg, 0.05)}
    print(f"参照 E：AUC={e_obs['auc']:.4f} TPR@1%={e_obs['t1']:.1%} "
          f"TPR@5%={e_obs['t5']:.1%}")

    rng = np.random.default_rng(args.seed)
    payload = {"reference_E": {k: round(v, 4) for k, v in e_obs.items()}, "results": {}}
    print("\n" + "=" * 92)
    print(f"{'config':<9}{'units':>7}{'pos':>6}{'AUC':>9}{'TPR@1%':>9}"
          f"{'95%CI':>18}{'TPR@5%':>9}{'ΔvsE':>8}")
    print("-" * 92)
    for cfg in args.configs:
        L, S = (int(x) for x in cfg.split("x"))
        nd, pd = dirs_for(L, S)
        if not (nd.exists() and pd.exists()):
            print(f"{cfg:<9} MISSING")
            continue
        units = [u for u in load_units(nd) if u.get("vec_d1")]
        rows = [r for r in load_rows(pd) if r.get("vec_d1")]
        cal = [u for u in units if u["cohort"] == "calibration"]
        test = [u for u in units if u["cohort"] == "independent_test"]
        if not cal or not test or not rows:
            print(f"{cfg:<9} INCOMPLETE")
            continue
        sc = Scope(cal, "vec_d1", "empirical")

        def b1q(u):
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

        pv = [b1q(r) for r in rows]
        nv = [b1q(u) for u in test]
        a = auc(pv, nv)
        t1, t5 = tpr(pv, nv, 0.01), tpr(pv, nv, 0.05)
        clusters, neg_docs = {}, {}
        for r in rows:
            clusters.setdefault((r["doc"], r["slot"]), []).append(b1q(r))
        for u in test:
            neg_docs.setdefault((u["cohort"], u["doc"]), []).append(b1q(u))
        lo, hi = boot_ci(pv, nv, clusters, neg_docs, rng, args.boot)
        ci = f"[{lo:.1%},{hi:.1%}]"
        print(f"{cfg:<9}{len(units):>7}{len(rows):>6}{a:>9.4f}{t1:>9.1%}{ci:>18}"
              f"{t5:>9.1%}{t1-e_obs['t1']:>+8.1%}")
        payload["results"][cfg] = {
            "window": L, "stride": S, "units": len(units), "positives": len(rows),
            "auc": round(a, 4), "tpr1": round(t1, 4),
            "tpr1_ci": [round(lo, 4), round(hi, 4)], "tpr5": round(t5, 4),
            "delta_vs_E_tpr1": round(t1 - e_obs["t1"], 4),
        }

    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
