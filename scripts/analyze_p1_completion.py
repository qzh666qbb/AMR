"""补齐 P1：对 merge / split 计算 E 统计量（标点恢复对这两类无效）。

用 `dataset4_fusion1/FUSION_RECOVERY.json` 已缓存的 vec_d2，与 P1 的
"none" 列同公式，不需要重新解析。同时校验与 P1 的 none 列是否一致。
"""
from __future__ import annotations

import bisect
import json
import math
from pathlib import Path

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


def p_emp(arr, score):
    n = len(arr)
    return 1.0 if n == 0 else (1 + n - bisect.bisect_left(arr, score)) / (1 + n)


def e_stat(vec):
    best = None
    for t, v in enumerate(vec):
        if v is None:
            continue
        x = -2.0 * math.log(p_emp(ECDF[t], v))
        if best is None or x > best:
            best = x
    return best


def tau_at(scores, q):
    s = sorted([x for x in scores if x is not None], reverse=True)
    k = max(1, math.ceil(q * len(s)))
    return s[min(k, len(s)) - 1]


null_units = [u for u in json.loads((NULL / "PARTIAL_WINDOW_SCALED.json").read_text())["per_unit"]
              if u.get("vec_d2")]
calib = [u for u in null_units if u["cohort"] == "calibration"]
test = [u for u in null_units if u["cohort"] == "independent_test"]
ECDF = build_ecdf(calib, "vec_d2")

null_e = [e_stat(u["vec_d2"]) for u in test]
taus = {q: tau_at(null_e, q) for q in (0.01, 0.05)}

rows = json.loads((POS / "FUSION_RECOVERY.json").read_text())["per_instance"]
p1 = json.loads((NULL / "P1_PUNCT_BASELINE.json").read_text())

by_attack = {}
for r in rows:
    by_attack.setdefault(r["attack"], []).append(e_stat(r["vec_d2"]))

print("P1 补齐：E 统计量的载体恢复率（按攻击）")
print(f"阈值：q=1% τ={taus[0.01]:.4f}，q=5% τ={taus[0.05]:.4f}")
print()
print(f"{'attack':<12} {'n':>4} {'@1%':>9} {'@5%':>9}  {'P1的none列@1%':>14}  {'一致':>5}")
print("-" * 62)
p1_none = {r["attack"]: r for r in p1["rows"]}
for attack in ("baseline", "semicolon", "newline", "comma", "space", "split", "merge"):
    vals = [v for v in by_attack.get(attack, []) if v is not None]
    if not vals:
        continue
    r1 = sum(1 for v in vals if v >= taus[0.01]) / len(vals)
    r5 = sum(1 for v in vals if v >= taus[0.05]) / len(vals)
    ref = p1_none.get(attack, {}).get("none", {}).get("r1")
    ok = "✓" if ref is not None and abs(ref - r1) < 1e-6 else ("—" if ref is None else "✗")
    ref_s = f"{ref:>14.1%}" if ref is not None else f"{'(未测)':>14}"
    print(f"{attack:<12} {len(vals):>4} {r1:>9.1%} {r5:>9.1%}  {ref_s}  {ok:>5}")

print()
print("说明：split/merge 无法靠标点恢复（词序已变/新增边界不可知），")
print("      因此对这两类攻击，E 的 no-restoration 结果就是最终可达到的水平。")
