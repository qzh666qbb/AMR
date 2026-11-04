"""D8：零假设尾部去相关（针对 D7a 的瓶颈）。

**问题定位**：D7a 改善了 AUC（排序），但在固定 FPR 下无增益。原因是零假设尾部
由「对 100 个高度相关的模板取极值」决定。实测模板两两相关均值 0.477、中位 0.605、
15% 的模板对 >0.8 —— **有效独立检验数远小于 100**，极值统计被严重膨胀。

**方案**：按相关结构把模板聚成族，族内取代表，族间用**非极值**方式聚合
（均值 / Fisher / 计数），从而压轻零假设尾部。

  X(t)        = -2(ln p_whole(t) + ln p_sub(t))        模板级 Fisher（同 D7a）
  F_f         = max_{t in family f} X(t)               族内取代表
  z_f         = (F_f - μ_f) / σ_f                      族级零假设归一化
  score       = 聚合_f(z_f)                            族间聚合（见 AGGREGATORS）

对照组：max_f z_f 等价于原来的极值聚合（应当接近 D7a）。

同时实现「尾部校正的多重比较」变体：对模板级 p 做 Sidak 校正，
有效检验数取族数而非 100。
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


# ───────────────────── 基础工具 ─────────────────────

def build_ecdf(units: list, key: str) -> list:
    n_tpl = len(units[0][key])
    cols = [[] for _ in range(n_tpl)]
    for u in units:
        for t, v in enumerate(u[key]):
            if v is not None:
                cols[t].append(v)
    return [sorted(c) for c in cols]


def p_emp(sorted_asc: list, score: float) -> float:
    n = len(sorted_asc)
    if n == 0:
        return 1.0
    return (1 + n - bisect.bisect_left(sorted_asc, score)) / (1 + n)


def template_x(unit: dict, ecdf: dict) -> list:
    """模板级 Fisher 组合分数 X(t)。"""
    v1, v2 = unit.get("vec_d1"), unit.get("vec_d2")
    if v1 is None or v2 is None:
        return [None] * len(v1)
    out = []
    for t in range(len(v1)):
        if v1[t] is None or v2[t] is None:
            out.append(None)
        else:
            out.append(-2.0 * (math.log(p_emp(ecdf["d1"][t], v1[t]))
                               + math.log(p_emp(ecdf["d2"][t], v2[t]))))
    return out


def cluster_templates(calib: list, threshold: float) -> list:
    """按相关距离做层次聚类，返回每个模板的族标签。"""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    M = np.array([[v if v is not None else np.nan for v in u["vec_d1"]]
                  for u in calib], dtype=float)
    with np.errstate(invalid="ignore"):
        C = np.ma.corrcoef(np.ma.masked_invalid(M), rowvar=False).filled(0.0)
    C = np.clip(C, -1.0, 1.0)
    np.fill_diagonal(C, 1.0)
    dist = 1.0 - C
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    Z = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")
    return labels.tolist()


# ───────────────────── 族级统计量 ─────────────────────

class FamilyScorer:
    def __init__(self, calib: list, labels: list, ecdf: dict):
        self.labels = labels
        self.ecdf = ecdf
        self.n_fam = max(labels) + 1
        members = [[] for _ in range(self.n_fam)]
        for t, f in enumerate(labels):
            members[f].append(t)
        self.members = members
        # 族级零假设分布（均值/标准差 + 经验分布）
        fam_scores = [[] for _ in range(self.n_fam)]
        for u in calib:
            F = self.family_scores(u)
            for f, v in enumerate(F):
                if v is not None:
                    fam_scores[f].append(v)
        self.fam_mu = [float(np.mean(s)) if s else 0.0 for s in fam_scores]
        self.fam_sd = [max(float(np.std(s, ddof=1)) if len(s) > 1 else 1.0, SIGMA_FLOOR)
                       for s in fam_scores]
        self.fam_ecdf = [sorted(s) for s in fam_scores]

    def family_scores(self, unit: dict) -> list:
        X = template_x(unit, self.ecdf)
        out = []
        for members in self.members:
            vals = [X[t] for t in members if X[t] is not None]
            out.append(max(vals) if vals else None)
        return out

    def family_z(self, F: list) -> list:
        return [None if v is None else (v - self.fam_mu[f]) / self.fam_sd[f]
                for f, v in enumerate(F)]

    def family_p(self, F: list) -> list:
        return [None if v is None else p_emp(self.fam_ecdf[f], v)
                for f, v in enumerate(F)]

    # ── 聚合器 ──
    def agg_max(self, unit):          # 对照：等价于极值聚合
        z = [v for v in self.family_z(self.family_scores(unit)) if v is not None]
        return max(z) if z else None

    def agg_mean(self, unit):         # 均值：压轻尾部
        z = [v for v in self.family_z(self.family_scores(unit)) if v is not None]
        return sum(z) / len(z) if z else None

    def agg_sum(self, unit):
        z = [v for v in self.family_z(self.family_scores(unit)) if v is not None]
        return sum(z) if z else None

    def agg_fisher(self, unit):       # 族间 Fisher
        ps = [v for v in self.family_p(self.family_scores(unit)) if v is not None]
        if not ps:
            return None
        return -2.0 * sum(math.log(max(p, 1e-300)) for p in ps)

    def agg_count(self, unit, thr=2.0):  # 超阈值族数
        z = [v for v in self.family_z(self.family_scores(unit)) if v is not None]
        return float(sum(1 for v in z if v >= thr)) if z else None


def sidak_statistic(unit: dict, ecdf: dict, m_eff: int) -> float | None:
    """尾部校正的多重比较：模板级 p → Fisher 合并 p → Sidak 校正。

    注意：模板级的 X(t) 是 Fisher 统计量，**不是 p 值**，不能直接做 Sidak。
    必须先由 X(t) 换算出合并 p 值（X ~ chi2(4)），再做 Sidak。
    """
    from scipy.stats import chi2
    v1, v2 = unit.get("vec_d1"), unit.get("vec_d2")
    if v1 is None or v2 is None:
        return None
    best = None
    for t in range(len(v1)):
        if v1[t] is None or v2[t] is None:
            continue
        x = -2.0 * (math.log(p_emp(ecdf["d1"][t], v1[t]))
                    + math.log(p_emp(ecdf["d2"][t], v2[t])))
        p_comb = float(chi2.sf(x, df=4))
        p_comb = min(max(p_comb, 1e-300), 1.0 - 1e-12)
        p_adj = -math.expm1(m_eff * math.log1p(-p_comb))
        score = -math.log(max(p_adj, 1e-300))
        if best is None or score > best:
            best = score
    return best


# ───────────────────── 评估 ─────────────────────

def auc(pos, neg) -> float:
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


def curve(pos, neg, targets) -> list:
    nd = sorted([v for v in neg if v is not None], reverse=True)
    pv = [v for v in pos if v is not None]
    out = []
    for q in targets:
        k = max(1, math.ceil(q * len(nd)))
        tau = nd[min(k, len(nd)) - 1]
        out.append({"target": q, "tau": round(tau, 4),
                    "fpr": round(sum(1 for x in nd if x >= tau) / len(nd), 4),
                    "tpr": round(sum(1 for x in pv if x >= tau) / len(pv), 4)})
    return out


def pct(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--null-file", type=Path,
                    default=NULL / "PARTIAL_WINDOW_MERGED.json")
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--output", type=Path, default=NULL / "TAIL_DECORRELATION.json")
    args = ap.parse_args()

    null_units = [u for u in json.loads(args.null_file.read_text())["per_unit"]
                  if u.get("vec_d1") and u.get("vec_d2")]
    pos_rows = [r for r in json.loads((POS / "FUSION_RECOVERY.json").read_text())["per_instance"]
                if r.get("vec_d1") and r.get("vec_d2")]
    calib = [u for u in null_units if u["cohort"] == "calibration"]
    test = [u for u in null_units if u["cohort"] == "independent_test"]
    print(f"零例：校准 {len(calib)}，测试 {len(test)}；正例 {len(pos_rows)}")

    ecdf = {s: build_ecdf(calib, f"vec_{s}") for s in ("d1", "d2")}

    # ── 真实基线（与 D7a / D1 同口径，用于公平对比）──
    def baseline_d1(unit):
        v = unit.get("vec_d1")
        if v is None:
            return None
        vals = [x for x in v if x is not None]
        return max(vals) if vals else None

    def baseline_d7a(unit):
        X = template_x(unit, ecdf)
        vals = [x for x in X if x is not None]
        return max(vals) if vals else None

    baselines = {"D1 整图(raw)": baseline_d1, "D7a 分位+Fisher(极值)": baseline_d7a}
    base_neg = {k: [f(u) for u in test] for k, f in baselines.items()}
    base_pos = {k: [f(r) for r in pos_rows] for k, f in baselines.items()}
    print(f"\n真实基线：")
    for k in baselines:
        a = auc(base_pos[k], base_neg[k])
        cv = curve(base_pos[k], base_neg[k], [0.01, 0.05])
        print(f"  {k:<24} AUC={a:.4f}  TPR@1%={cv[0]['tpr']:.1%}  TPR@5%={cv[1]['tpr']:.1%}")

    targets = [0.01, 0.02, 0.05, 0.10, 0.20]
    results = {}
    payload = {"note": "按模板相关结构聚族，族间用非极值聚合以压轻零假设尾部",
               "null_calibration": len(calib), "null_test": len(test),
               "positive": len(pos_rows), "by_threshold": {}}

    for thr in args.thresholds:
        labels = cluster_templates(calib, thr)
        n_fam = max(labels) + 1
        fs = FamilyScorer(calib, labels, ecdf)
        variants = {
            "max_family(对照≈D7a)": fs.agg_max,
            "mean_family": fs.agg_mean,
            "sum_family": fs.agg_sum,
            "fisher_family": fs.agg_fisher,
            "count_family_z>=2": lambda u: fs.agg_count(u, 2.0),
            "sidak_tail_corrected": lambda u: sidak_statistic(u, ecdf, n_fam),
        }
        neg = {k: [f(u) for u in test] for k, f in variants.items()}
        pos = {k: [f(r) for r in pos_rows] for k, f in variants.items()}

        print(f"\n{'=' * 92}")
        print(f"相关阈值 {thr}  →  {n_fam} 个模板族（有效检验数从 100 降到 {n_fam}）")
        print("=" * 92)
        print(f"{'聚合方式':<24} {'AUC':>8} | " + " | ".join(
            f"{'@' + format(q, '.0%'):>12}" for q in targets))
        entry = {"n_families": n_fam,
                 "family_sizes": [labels.count(f) for f in range(n_fam)]}
        for k in variants:
            a = auc(pos[k], neg[k])
            cv = curve(pos[k], neg[k], targets)
            row = " | ".join(f"{c['tpr']:>12.1%}" for c in cv)
            print(f"{k:<24} {a:>8.4f} | {row}")
            entry[k] = {"auc": round(a, 4), "curve": cv}
            results[(thr, k)] = (pos[k], neg[k])
        payload["by_threshold"][str(thr)] = entry

    # ── 与真实基线的配对 bootstrap ──
    best_key = max(
        ((t, k) for (t, k) in results if not k.startswith("max_family")),
        key=lambda key: auc(results[key][0], results[key][1]))
    print(f"\n最佳非极值聚合：阈值 {best_key[0]}，{best_key[1]}")
    pos_a, neg_a = results[best_key]

    rng = np.random.default_rng(args.seed)

    def paired_report(label, pos_b, neg_b):
        d_auc, d_tpr = [], {q: [] for q in targets}
        for _ in range(args.draws):
            pi = rng.integers(0, len(pos_a), len(pos_a))
            ni = rng.integers(0, len(neg_a), len(neg_a))
            pa = [pos_a[i] for i in pi]; na = [neg_a[i] for i in ni]
            pb = [pos_b[i] for i in pi]; nb = [neg_b[i] for i in ni]
            d_auc.append(auc(pa, na) - auc(pb, nb))
            for q in targets:
                d_tpr[q].append(curve(pa, na, [q])[0]["tpr"]
                                - curve(pb, nb, [q])[0]["tpr"])
        obs = auc(pos_a, neg_a) - auc(pos_b, neg_b)
        print(f"\n  {label}")
        print(f"    ΔAUC          {obs:>8.4f}  "
              f"[{pct(d_auc, 0.025):>8.4f},{pct(d_auc, 0.975):>8.4f}]  "
              f"P(Δ>0)={sum(1 for x in d_auc if x > 0) / len(d_auc):.3f}")
        entry = {"delta_auc": round(obs, 4),
                 "ci95": [round(pct(d_auc, 0.025), 4), round(pct(d_auc, 0.975), 4)]}
        for q in targets:
            o = (curve(pos_a, neg_a, [q])[0]["tpr"]
                 - curve(pos_b, neg_b, [q])[0]["tpr"])
            lo, hi = pct(d_tpr[q], 0.025), pct(d_tpr[q], 0.975)
            print(f"    ΔTPR@{q:<4.0%}FPR {o:>8.4f}  [{lo:>8.4f},{hi:>8.4f}]  "
                  f"P(Δ>0)={sum(1 for x in d_tpr[q] if x > 0) / len(d_tpr[q]):.3f}")
            entry.setdefault("tpr", {})[f"q{q}"] = {
                "delta": round(o, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        return entry

    print(f"\n{'=' * 84}")
    print(f"配对 bootstrap（{args.draws} 次）：{best_key[1]} vs 真实基线")
    print("=" * 84)
    for bname in baselines:
        payload.setdefault("paired_vs_baseline", {})[bname] = paired_report(
            bname, base_pos[bname], base_neg[bname])

    payload["paired_vs_control"] = paired_report(
        "max_family(内部极值对照)",
        results[(best_key[0], "max_family(对照≈D7a)")][0],
        results[(best_key[0], "max_family(对照≈D7a)")][1])
    payload["best_variant"] = {"threshold": best_key[0], "name": best_key[1]}

    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
