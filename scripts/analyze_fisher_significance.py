"""D7a（分位映射 + Fisher）与基线的配对 bootstrap 显著性检验。

为什么必须配对：D7a 与 D1 用的是同一批零假设单元与同一批正例载体，
两者高度相关。独立样本的 AUC 标准误（≈0.022）会严重高估差异的不确定性。
配对 bootstrap 在**同一批重采样索引**上同时重算两个统计量，才是正确的比较。

同时对 TPR@固定FPR 做配对 bootstrap（在重采样样本内重新定阈值）。
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import random
from pathlib import Path

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


def build_normal(units, key):
    n_tpl = len(units[0][key])
    mu, sigma = [], []
    for t in range(n_tpl):
        vals = [u[key][t] for u in units if u[key][t] is not None]
        if len(vals) < 2:
            mu.append(0.0); sigma.append(1.0); continue
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
        mu.append(m); sigma.append(max(math.sqrt(var), SIGMA_FLOOR))
    return mu, sigma


def make_stats(calib):
    ecdf = {s: build_ecdf(calib, f"vec_{s}") for s in ("d1", "d2")}

    def p_emp(scope, t, score):
        arr = ecdf[scope][t]
        n = len(arr)
        return (1 + n - bisect.bisect_left(arr, score)) / (1 + n)

    def fisher(u):
        v1, v2 = u.get("vec_d1"), u.get("vec_d2")
        if v1 is None or v2 is None:
            return None
        best = None
        for t in range(len(v1)):
            if v1[t] is None or v2[t] is None:
                continue
            x = -2.0 * (math.log(p_emp("d1", t, v1[t])) + math.log(p_emp("d2", t, v2[t])))
            if best is None or x > best:
                best = x
        return best

    def raw_whole(u):
        v = u.get("vec_d1")
        if v is None:
            return None
        vals = [x for x in v if x is not None]
        return max(vals) if vals else None

    fit_norm = {s: build_normal(calib, f"vec_{s}") for s in ("d1", "d2")}

    def zsum(u):
        v1, v2 = u.get("vec_d1"), u.get("vec_d2")
        if v1 is None or v2 is None:
            return None
        mu1, s1 = fit_norm["d1"]; mu2, s2 = fit_norm["d2"]
        best = None
        for t in range(len(v1)):
            if v1[t] is None or v2[t] is None:
                continue
            z = (v1[t] - mu1[t]) / s1[t] + (v2[t] - mu2[t]) / s2[t]
            if best is None or z > best:
                best = z
        return best

    return {"D7a 分位+Fisher": fisher, "D1 整图(raw)": raw_whole, "D6b z求和": zsum}


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


def tpr_at(pairs_pos, pairs_neg, q):
    """在给定样本内重新用零假设分位定阈值，再算正例命中率。"""
    neg_desc = sorted((v for v in pairs_neg if v is not None), reverse=True)
    pos_vals = [v for v in pairs_pos if v is not None]
    if not neg_desc or not pos_vals:
        return float("nan")
    k = max(1, math.ceil(q * len(neg_desc)))
    tau = neg_desc[min(k, len(neg_desc)) - 1]
    return sum(1 for v in pos_vals if v >= tau) / len(pos_vals)


def pct(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--null-file", type=Path, default=NULL / "PARTIAL.json",
                    help="零假设文件；对称口径用 PARTIAL_WINDOW_MERGED.json")
    ap.add_argument("--output", type=Path,
                    default=NULL / "QUANTILE_FISHER_SIGNIFICANCE.json")
    args = ap.parse_args()

    null_units = [u for u in json.loads(args.null_file.read_text())["per_unit"]
                  if u.get("vec_d1") and u.get("vec_d2")]
    pos_rows = [r for r in json.loads((POS / "FUSION_RECOVERY.json").read_text())["per_instance"]
                if r.get("vec_d1") and r.get("vec_d2")]
    calib = [u for u in null_units if u["cohort"] == "calibration"]
    test = [u for u in null_units if u["cohort"] == "independent_test"]
    stats = make_stats(calib)

    # 预先算好每个检测器在测试集与正例上的分数
    neg = {k: [f(u) for u in test] for k, f in stats.items()}
    pos = {k: [f(r) for r in pos_rows] for k, f in stats.items()}

    print(f"零假设测试 {len(test)}；正例 {len(pos_rows)}；bootstrap {args.draws} 次")
    print("\n观测 AUC：")
    for k in stats:
        print(f"  {k:<18} {auc(pos[k], neg[k]):.4f}")

    rng = random.Random(args.seed)
    ref_keys = [k for k in stats if k != "D7a 分位+Fisher"]
    diffs_auc = {k: [] for k in ref_keys}
    diffs_tpr = {q: {k: [] for k in ref_keys} for q in (0.01, 0.05)}
    for _ in range(args.draws):
        pi = [rng.randrange(len(pos_rows)) for _ in pos_rows]
        ni = [rng.randrange(len(test)) for _ in test]
        for k in ref_keys:
            pv_a = [pos["D7a 分位+Fisher"][i] for i in pi]
            nv_a = [neg["D7a 分位+Fisher"][i] for i in ni]
            pv_b = [pos[k][i] for i in pi]
            nv_b = [neg[k][i] for i in ni]
            diffs_auc[k].append(auc(pv_a, nv_a) - auc(pv_b, nv_b))
            for q in diffs_tpr:
                diffs_tpr[q][k].append(
                    tpr_at(pv_a, nv_a, q) - tpr_at(pv_b, nv_b, q))

    payload = {"draws": args.draws, "seed": args.seed,
               "observed_auc": {k: round(auc(pos[k], neg[k]), 4) for k in stats},
               "paired_delta": {}}
    print(f"\n{'对比':<28} {'ΔAUC':>8} {'95%CI':>22} {'P(Δ>0)':>8}")
    print("-" * 72)
    for k in ref_keys:
        d = diffs_auc[k]
        obs = auc(pos["D7a 分位+Fisher"], neg["D7a 分位+Fisher"]) - auc(pos[k], neg[k])
        lo, hi = pct(d, 0.025), pct(d, 0.975)
        pgt = sum(1 for x in d if x > 0) / len(d)
        print(f"D7a - {k:<22} {obs:>8.4f} [{lo:>8.4f}, {hi:>8.4f}] {pgt:>8.3f}")
        payload["paired_delta"][k] = {
            "delta_auc": round(obs, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "prob_greater": round(pgt, 4),
        }

    for q in diffs_tpr:
        print(f"\nTPR@{q:.0%}FPR 配对差：")
        for k in ref_keys:
            d = diffs_tpr[q][k]
            obs = (tpr_at(pos["D7a 分位+Fisher"], neg["D7a 分位+Fisher"], q)
                   - tpr_at(pos[k], neg[k], q))
            lo, hi = pct(d, 0.025), pct(d, 0.975)
            print(f"  D7a - {k:<22} {obs:>7.4f} [{lo:>7.4f}, {hi:>7.4f}]")
            payload["paired_delta"][k].setdefault("tpr", {})[f"q{q}"] = {
                "delta": round(obs, 4), "ci95": [round(lo, 4), round(hi, 4)]}

    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
