"""严格拆分 + 尾部平滑的文档级审计。

修复三件事：
1. 阈值只用校准半区估计（tau_u、tau_z），测试半区只用于评估；
2. 分位统计量的尾部做平滑（指数尾 / GPD 尾），避免经验 p 值上界饱和；
3. 文档级给出聚类 bootstrap 置信区间（按源文档重抽，攻击为簇内重复）。

同时报告两种零模型：SWAN 惯例 lambda=0.25，以及实测零例 green 率 p0。
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NULL = ROOT / "experiments/swan_multiview_development1/null_fp_audit1/PARTIAL_WINDOW_200.json"
POS = ROOT / "experiments/swan_multiview_development1/dataset5_eval/FUSION_RECOVERY.json"
OUT = ROOT / "experiments/swan_multiview_development1/dataset5_eval/DOC_LEVEL_STRICT.json"
LAMBDA = 0.25
MATCH_N = 6


# ---------- 尾部模型 ----------
def fit_tail(arr, q=0.9):
    """对给定分布的超出部分拟合指数/GPD 尾（矩估计，避免额外依赖）。"""
    n = len(arr)
    if n < 50:
        return None
    u = arr[int(q * (n - 1))]
    exc = [x - u for x in arr if x > u]
    if len(exc) < 20:
        return None
    m = float(np.mean(exc))
    v = float(np.var(exc, ddof=1))
    if m <= 0 or v <= 0:
        return None
    xi = 0.5 * (1.0 - m * m / v)
    xi = float(min(max(xi, -0.4), 0.4))
    sigma = float(max(m * (1.0 + xi), 1e-6))
    return {"u": u, "pu": len(exc) / n, "sigma": sigma, "xi": xi}


def p_empirical(arr, s):
    n = len(arr)
    return (1.0 + n - bisect.bisect_left(arr, s)) / (1.0 + n)


def p_value(arr, s, tail, mode):
    if mode == "empirical" or tail is None or s <= tail["u"]:
        return p_empirical(arr, s)
    y = s - tail["u"]
    if mode == "exp":
        return max(tail["pu"] * math.exp(-y / tail["sigma"]), 1e-15)
    xi = tail["xi"]
    if abs(xi) < 1e-6:
        surv = math.exp(-y / tail["sigma"])
    else:
        base = 1.0 + xi * y / tail["sigma"]
        if base <= 0:
            return 1e-15
        surv = base ** (-1.0 / xi)
    return max(tail["pu"] * surv, 1e-15)


class Scope:
    """一个作用域（d1 / d2）的模板分数分布与尾部模型。"""

    def __init__(self, units, key, mode):
        n_tpl = len(units[0][key])
        self.arrays = []
        self.tails = []
        for t in range(n_tpl):
            arr = sorted(u[key][t] for u in units if u[key][t] is not None)
            self.arrays.append(arr)
            self.tails.append(fit_tail(arr) if mode in ("exp", "gpd") else None)
        self.mode = mode

    def p(self, t, s):
        return p_value(self.arrays[t], s, self.tails[t], self.mode)


def make_detectors(calib, mode):
    d1 = Scope(calib, "vec_d1", mode)
    d2 = Scope(calib, "vec_d2", mode)

    def A(u):
        v = u.get("vec_d1")
        if not v:
            return None
        vals = [x for x in v if x is not None]
        return max(vals) if vals else None

    def E(u):
        v = u.get("vec_d2")
        if not v:
            return None
        best = None
        for t, x in enumerate(v):
            if x is None:
                continue
            val = -2.0 * math.log(d2.p(t, x))
            best = val if best is None or val > best else best
        return best

    def D(u):
        v1, v2 = u.get("vec_d1"), u.get("vec_d2")
        if not v1 or not v2:
            return None
        best = None
        for t in range(len(v1)):
            if v1[t] is None or v2[t] is None:
                continue
            val = -2.0 * (math.log(d1.p(t, v1[t])) + math.log(d2.p(t, v2[t])))
            best = val if best is None or val > best else best
        return best

    return {"A 整图+原始": A, "E 局部+分位": E, "D 双域+分位Fisher": D}


# ---------- 统计工具 ----------
def tau_at(scores, q):
    s = sorted([x for x in scores if x is not None], reverse=True)
    if not s:
        return float("nan")
    k = max(1, math.ceil(q * len(s)))
    return s[min(k, len(s)) - 1]


def z_lambda(g, n):
    d = math.sqrt(n * LAMBDA * (1 - LAMBDA))
    return float("nan") if d < 1e-9 else (g - LAMBDA * n) / d


def z_emp(g, n, p0):
    d = math.sqrt(n * p0 * (1 - p0))
    return 0.0 if d < 1e-9 else (g - p0 * n) / d


def group_docs(units, key="doc"):
    out = {}
    for u in units:
        out.setdefault(u[key], []).append(u)
    return out


def doc_zs(docs, f, tau, zfun):
    out = {}
    for d, us in docs.items():
        g = sum(1 for u in us if f(u) is not None and f(u) >= tau)
        out[d] = zfun(g, len(us))
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def crossfit_fpr(docs, name, mode, q_u, q_d, zfun):
    """校准半区两折交叉：一半（重新拟合尾部）定阈值，另一半测 FPR。"""
    keys = sorted(docs)
    folds = [keys[::2], keys[1::2]]
    rates = []
    for i in range(2):
        fit_keys = folds[i]
        eval_keys = folds[1 - i]
        fit_units = [u for k in fit_keys for u in docs[k]]
        f = make_detectors(fit_units, mode)[name]
        tau = tau_at([f(u) for u in fit_units], q_u)
        zfit = {}
        for k in fit_keys:
            us = docs[k]
            zfit[k] = zfun(sum(1 for u in us if f(u) is not None and f(u) >= tau), len(us))
        tau_z = tau_at(list(zfit.values()), q_d)
        zev = [zfun(sum(1 for u in docs[k] if f(u) is not None and f(u) >= tau), len(docs[k]))
               for k in eval_keys]
        rates.append(sum(1 for v in zev if v >= tau_z) / len(zev))
    return float(np.mean(rates)), rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", type=Path, default=NULL)
    ap.add_argument("--pos", type=Path, default=POS)
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--q-unit", type=float, default=0.01)
    ap.add_argument("--q-doc", type=float, default=0.05)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261021)
    args = ap.parse_args()

    null = [u for u in json.loads(args.null.read_text())["per_unit"]
            if u.get("vec_d1") and u.get("vec_d2")]
    calib = [u for u in null if u["cohort"] == "calibration"]
    test = [u for u in null if u["cohort"] == "independent_test"]
    rows = [r for r in json.loads(args.pos.read_text())["per_instance"]
            if r.get("vec_d1") and r.get("vec_d2")]
    cal_docs, test_docs = group_docs(calib), group_docs(test)
    pos_docs = {}
    for r in rows:
        pos_docs.setdefault((r["doc"], r["attack"]), []).append(r)
    src_docs = sorted({k[0] for k in pos_docs})

    print(f"校准 {len(calib)} 单元 / {len(cal_docs)} 篇；"
          f"测试 {len(test)} 单元 / {len(test_docs)} 篇；"
          f"正例 {len(rows)} 实例 / {len(src_docs)} 源文档")

    rng = np.random.default_rng(args.seed)
    payload = {"config": {k: (str(v) if isinstance(v, Path) else v)
                          for k, v in vars(args).items()},
               "results": {}}

    for mode in ("empirical", "exp", "gpd"):
        det = make_detectors(calib, mode)
        print("\n" + "=" * 92)
        print(f"尾部模型：{mode}")
        print("=" * 92)
        print(f"{'检测器':<18}{'τ_u':>9}{'测试单元FPR':>12}{'τ_z':>8}"
              f"{'测试文档FPR':>12}{'文档TPR':>10}{'95%CI':>16}{'交叉FPR':>10}")
        for name, f in det.items():
            cu = [f(u) for u in calib]
            tu = tau_at(cu, args.q_unit)
            p0 = sum(1 for x in cu if x is not None and x >= tu) / len(cu)
            unit_fpr = sum(1 for u in test
                           if f(u) is not None and f(u) >= tu) / len(test)
            zfun = lambda g, n: z_emp(g, n, p0)
            cal_z = doc_zs(cal_docs, f, tu, zfun)
            tau_z = tau_at(list(cal_z.values()), args.q_doc)
            test_z = doc_zs(test_docs, f, tu, zfun)
            fpr = sum(1 for v in test_z.values() if v >= tau_z) / len(test_z)
            fpr_lo, fpr_hi = wilson(sum(1 for v in test_z.values() if v >= tau_z),
                                    len(test_z))

            # 文档 TPR：等 N 重抽 + 按源文档聚类 bootstrap
            per_doc = {}
            for d in src_docs:
                rs_list = [pos_docs[(d, a)] for a in
                           ("baseline", "semicolon", "newline", "split",
                            "merge", "comma", "space") if (d, a) in pos_docs]
                rates = []
                for _ in range(args.draws):
                    hit = 0
                    for rs in rs_list:
                        m = min(MATCH_N, len(rs))
                        idx = rng.choice(len(rs), size=m, replace=False)
                        g = sum(1 for i in idx
                                if f(rs[i]) is not None and f(rs[i]) >= tu)
                        hit += zfun(g, m) >= tau_z
                    rates.append(hit / max(1, len(rs_list)))
                per_doc[d] = float(np.mean(rates))
            tpr = float(np.mean(list(per_doc.values())))
            boot = []
            vals = np.array([per_doc[d] for d in src_docs])
            for _ in range(args.boot):
                boot.append(float(np.mean(rng.choice(vals, size=len(vals), replace=True))))
            lo, hi = np.percentile(boot, [2.5, 97.5])

            cf, cfr = crossfit_fpr(cal_docs, name, mode, args.q_unit, args.q_doc, zfun)

            # 固定 z=2.33（实测零模型）
            fpr233 = sum(1 for v in test_z.values() if v >= 2.33) / len(test_z)
            tpr233 = float(np.mean([
                np.mean([zfun(sum(1 for i in rng.choice(len(rs), size=min(MATCH_N, len(rs)),
                                                       replace=False)
                                    if f(rs[i]) is not None and f(rs[i]) >= tu),
                               min(MATCH_N, len(rs))) >= 2.33
                          for rs in [pos_docs[(d, a)] for a in
                                     ("baseline", "semicolon", "newline", "split",
                                      "merge", "comma", "space") if (d, a) in pos_docs]])
                for d in src_docs]))

            print(f"{name:<18}{tu:>9.4f}{unit_fpr:>12.2%}{tau_z:>8.3f}"
                  f"{fpr:>12.2%}{tpr:>10.1%}"
                  f"{f'[{lo:.1%},{hi:.1%}]':>16}{cf:>10.2%}")
            payload["results"][f"{mode}::{name}"] = {
                "tau_u": round(tu, 4), "p0_calib": round(p0, 4),
                "test_unit_fpr": round(unit_fpr, 4),
                "tau_z": round(tau_z, 3),
                "test_doc_fpr": round(fpr, 4),
                "test_doc_fpr_ci": [round(fpr_lo, 4), round(fpr_hi, 4)],
                "doc_tpr": round(tpr, 4),
                "doc_tpr_ci": [round(float(lo), 4), round(float(hi), 4)],
                "crossfit_doc_fpr": round(cf, 4), "crossfit_folds": [round(x, 4) for x in cfr],
                "z233": {"test_doc_fpr": round(fpr233, 4), "doc_tpr": round(tpr233, 4)},
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
