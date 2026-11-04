"""dataset5 逐攻击 TPR 明细（A vs E vs D），按实测 FPR 匹配阈值。"""
from __future__ import annotations

import bisect
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NULL = ROOT / "experiments/swan_multiview_development1/null_fp_audit1"
EV = ROOT / "experiments/swan_multiview_development1/dataset5_eval"


def build_ecdf(units, key):
    n = len(units[0][key])
    cols = [[] for _ in range(n)]
    for u in units:
        for t, v in enumerate(u[key]):
            if v is not None:
                cols[t].append(v)
    return [sorted(c) for c in cols]


ECDF = None


def p_emp(arr, s):
    n = len(arr)
    return 1.0 if n == 0 else (1 + n - bisect.bisect_left(arr, s)) / (1 + n)


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
    for t, x in enumerate(v):
        if x is None:
            continue
        val = -2.0 * math.log(p_emp(ECDF[t], x))
        if best is None or val > best:
            best = val
    return best


def D(u):
    v1, v2 = u.get("vec_d1"), u.get("vec_d2")
    if v1 is None or v2 is None:
        return None
    best = None
    for t in range(len(v1)):
        if v1[t] is None or v2[t] is None:
            continue
        val = -2.0 * (math.log(p_emp(ECDF[t], v1[t]))
                      + math.log(p_emp(ECDF[t], v2[t])))
        if best is None or val > best:
            best = val
    return best


def tau_at(scores, q):
    s = sorted([x for x in scores if x is not None], reverse=True)
    k = max(1, math.ceil(q * len(s)))
    return s[min(k, len(s)) - 1]


null = [u for u in json.loads((NULL / "PARTIAL_WINDOW_SCALED.json").read_text())["per_unit"]
        if u.get("vec_d1") and u.get("vec_d2")]
calib = [u for u in null if u["cohort"] == "calibration"]
test = [u for u in null if u["cohort"] == "independent_test"]
ECDF = build_ecdf(calib, "vec_d2")

rows = [r for r in json.loads((EV / "FUSION_RECOVERY.json").read_text())["per_instance"]
        if r.get("vec_d1") and r.get("vec_d2")]
det = {"A 整图+原始": A, "E 局部+分位": E, "D 双域+分位Fisher": D}
taus = {k: {q: tau_at([f(u) for u in test], q) for q in (0.01, 0.05)}
        for k, f in det.items()}

by = {}
for r in rows:
    by.setdefault(r["attack"], []).append(r)

out = {"thresholds": {k: {f"q{q}": round(t, 4) for q, t in v.items()}
                      for k, v in taus.items()},
       "per_attack": {}}
print(f"{'attack':<11} {'n':>4} | " + " | ".join(
    f"{k[:2]}@{q:.0%}" for q in (0.01, 0.05) for k in det))
print("-" * 78)
for atk in ("baseline", "semicolon", "newline", "split", "merge", "comma", "space"):
    rs = by.get(atk, [])
    if not rs:
        continue
    cells = f"{atk:<11} {len(rs):>4} | "
    entry = {"n": len(rs)}
    for q in (0.01, 0.05):
        for k, f in det.items():
            vals = [f(r) for r in rs]
            rate = sum(1 for v in vals if v is not None and v >= taus[k][q]) / len(rs)
            cells += f"{rate:>7.1%} | "
            entry[f"{k}_q{q}"] = round(rate, 4)
    print(cells)
    out["per_attack"][atk] = entry

(EV / "PER_ATTACK_TPR.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(f"\nwritten: {EV / 'PER_ATTACK_TPR.json'}")
