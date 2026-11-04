"""混合比例实验：文档中只有一部分句子携带水印时，文档级检出率如何变化。

两种阈值口径并列报告：
- strict：tau_u / tau_z 只由校准半区估计，测试半区实测 FPR（检验可迁移性）；
- matched：在测试半区的零例上匹配 5% 文档 FPR（标准 ROC 工作点，用于等 FPR 比较）。

文档长度固定 N=12（30 篇源文档最少 12 个槽位）；rho ∈ {1.0, 0.5, 0.25}。
载体命中数用超几何分布抽样，填充句命中数用二项分布抽样，计算量与数据量解耦。
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

from analyze_doc_level_strict import make_detectors, tau_at  # noqa: E402

NULL = ROOT / "experiments/swan_multiview_development1/null_fp_audit1/PARTIAL_WINDOW_200.json"
POS = ROOT / "experiments/swan_multiview_development1/dataset5_eval/FUSION_RECOVERY.json"
OUT = ROOT / "experiments/swan_multiview_development1/dataset5_eval/MIXED_RATIO.json"
ATTACKS = ("baseline", "semicolon", "newline", "split", "merge", "comma", "space")


def z_emp(g, n, p0):
    d = math.sqrt(n * p0 * (1 - p0))
    return 0.0 if d < 1e-9 else (g - p0 * n) / d


def rate(f, units, tau):
    return sum(1 for u in units if f(u) is not None and f(u) >= tau) / len(units)


def eval_ratios(f, tu, p0, p_f, tau_z, args, rng, N, src_docs, carriers):
    sd = math.sqrt(N * p0 * (1 - p0)) if p0 > 0 else 1.0
    hits = {}
    for key, rs in carriers.items():
        flags = [1 if (f(r) is not None and f(r) >= tu) else 0 for r in rs]
        hits[key] = (len(flags), sum(flags))

    out = {}
    for rho in args.ratios:
        k = max(1, int(round(rho * N)))
        per_doc, per_attack = {}, {a: [] for a in ATTACKS}
        for d in src_docs:
            doc_rates = []
            for a in ATTACKS:
                key = (d, a)
                if key not in hits:
                    continue
                n_s, h_s = hits[key]
                kk = min(k, n_s)
                gc = rng.hypergeometric(h_s, n_s - h_s, kk, size=args.draws)
                gf = rng.binomial(N - kk, p_f, size=args.draws)
                passed = ((gc + gf) - p0 * N) / sd >= tau_z
                doc_rates.append(float(passed.mean()))
                per_attack[a].extend(passed.tolist())
            per_doc[d] = float(np.mean(doc_rates))
        vals = np.array([per_doc[d] for d in src_docs])
        boot = [float(np.mean(rng.choice(vals, size=len(vals), replace=True)))
                for _ in range(args.boot)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        kept = {a: round(float(np.mean(v)), 4) for a, v in per_attack.items() if v}
        out[rho] = {"doc_tpr": round(float(np.mean(vals)), 4),
                    "doc_tpr_ci": [round(float(lo), 4), round(float(hi), 4)],
                    "per_attack": kept if rho == min(args.ratios) else {}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", type=Path, default=NULL)
    ap.add_argument("--pos", type=Path, default=POS)
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--doc-len", type=int, default=12)
    ap.add_argument("--ratios", nargs="+", type=float, default=[1.0, 0.5, 0.25])
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--q-unit", type=float, default=0.01)
    ap.add_argument("--q-doc", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20261021)
    args = ap.parse_args()

    null = [u for u in json.loads(args.null.read_text())["per_unit"]
            if u.get("vec_d1") and u.get("vec_d2")]
    calib = [u for u in null if u["cohort"] == "calibration"]
    test = [u for u in null if u["cohort"] == "independent_test"]
    rows = [r for r in json.loads(args.pos.read_text())["per_instance"]
            if r.get("vec_d1") and r.get("vec_d2")]
    carriers = {}
    for r in rows:
        carriers.setdefault((r["doc"], r["attack"]), []).append(r)
    src_docs = sorted({k[0] for k in carriers})
    rng = np.random.default_rng(args.seed)
    N = args.doc_len

    payload = {"config": {"doc_len": N, "ratios": args.ratios, "q_unit": args.q_unit,
                          "q_doc": args.q_doc, "draws": args.draws}, "results": {}}

    for tail in ("exp", "empirical"):
        det = make_detectors(calib, tail)
        for split in ("strict", "matched"):
            print("\n" + "=" * 104)
            print(f"尾部={tail}  阈值={split}  文档长度 N={N}  单元FPR目标={args.q_unit:.0%}  "
                  f"文档FPR目标={args.q_doc:.0%}")
            print("=" * 104)
            print(f"{'rho':>5} {'检测器':<18}{'零例FPR':>9}{'文档TPR':>9}{'95%CI':>18}")
            for name, f in det.items():
                cu = [f(u) for u in calib]
                tu_test = [f(u) for u in test]
                if split == "strict":
                    tu = tau_at(cu, args.q_unit)
                    p0 = rate(f, calib, tu)
                    p_f = rate(f, test, tu)
                    cal_g = rng.binomial(N, p0, size=200000)
                    tau_z = tau_at([z_emp(g, N, p0) for g in cal_g], args.q_doc)
                    test_g = rng.binomial(N, p_f, size=200000)
                    fpr = float(np.mean([z_emp(g, N, p0) >= tau_z for g in test_g]))
                else:
                    tu = tau_at(tu_test, args.q_unit)
                    p0 = p_f = rate(f, test, tu)
                    test_g = rng.binomial(N, p_f, size=200000)
                    tau_z = tau_at([z_emp(g, N, p0) for g in test_g], args.q_doc)
                    fpr = float(np.mean([z_emp(g, N, p0) >= tau_z for g in test_g]))

                ratios = eval_ratios(f, tu, p0, p_f, tau_z, args, rng, N,
                                     src_docs, carriers)
                for rho in args.ratios:
                    r = ratios[rho]
                    lo, hi = r["doc_tpr_ci"]
                    ci = f"[{lo:.1%},{hi:.1%}]"
                    print(f"{rho:>5.2f} {name:<18}{fpr:>9.2%}{r['doc_tpr']:>9.1%}"
                          f"{ci:>18}")
                    payload["results"][f"{tail}::{split}::rho{rho}::{name}"] = {
                        "tau_u": round(tu, 4), "tau_z": round(tau_z, 3),
                        "p0": round(p0, 4), "p_filler": round(p_f, 4),
                        "k_carriers": max(1, int(round(rho * N))), "doc_len": N,
                        "null_fpr": round(fpr, 4), "doc_tpr": r["doc_tpr"],
                        "doc_tpr_ci": r["doc_tpr_ci"], "per_attack": r["per_attack"],
                    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
