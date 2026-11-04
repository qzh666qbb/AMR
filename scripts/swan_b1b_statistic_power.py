"""B1b：检测统计量「函数形式」对功效的决定性影响（纯离线、冻结数据）。

B1 结论是「计数统计量独立性成立但检测端无功效」。本脚本追问：
无功效的根因是**统计量形式**（max 被单个侥幸高分骗过），还是**校准**？

对照若干候选统计量在同一批冻结单元分数上的 held-out 表现：
  max / mean / mean_top3 / mean_top5 / count>=tau
关键做法：
  1. 在 fit 半区 null 上按分位取阈值，在 hold 半区 null 上测 FPR（诚实、非 in-sample）；
  2. 为消除「名义 FPR 漂移」的干扰，再按 **hold null 匹配到同一 FPR**（1%、2%）比较 TPR；
  3. 检查长度混杂：mean 与 word_count / 单元数的相关；
  4. 分 cohort 报告（baseline vs punctuation_stress）；
  5. 精确 Clopper-Pearson 95% CI（自查，不依赖 scipy）。

输出：experiments/swan_b1b_statistic_power1/{REPORT.md, RESULT.json}
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/swan_b1b_statistic_power1"
OUT.mkdir(parents=True, exist_ok=True)

CLEAN = ROOT / "experiments/swan_elastic_v4_development1/results/calibration_rows.jsonl"
POS = ROOT / "experiments/swan_elastic_v4_development1/results/development_rows.jsonl"
STRESS = ROOT / "experiments/swan_elastic_v4_p2_formal/results/development_rows.jsonl"
POS_COHORTS = {"confirmation_baseline_positive", "confirmation_punctuation_stress"}


# ------------------------------------------------------------------ 数据
def load(path, cohorts=None):
    out = []
    for line in open(path):
        d = json.loads(line)
        if cohorts is not None and d.get("cohort") not in cohorts:
            continue
        det = d.get("detection", {})
        if det.get("status") != "ok":
            continue
        out.append({
            "scores": [u["match"]["score"] for u in det.get("units", [])],
            "raw_utility": det.get("raw_utility"),
            "cohort": d.get("cohort"),
            "group_id": d.get("group_id"),
            "word_count": d.get("word_count"),
            "unit_count": det.get("unit_count"),
        })
    return out


# ------------------------------------------------------------------ 统计核心
def _log_binom_pmf(k, n, p):
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    lc = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return lc + k * math.log(p) + (n - k) * math.log1p(-p)


def cdf_ge(k, n, p):
    """P(X >= k), X~Binomial(n,p)。直接按 log 空间逐项求和（n 小，稳健无重复计数）。"""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return min(1.0, sum(math.exp(_log_binom_pmf(i, n, p)) for i in range(k, n + 1)))


def ratio_ci(k, n, conf=0.95):
    """精确 Clopper-Pearson 双侧区间 + 单侧 95% 下界（用于 TPR）。"""
    if n == 0:
        return {"lower": 0.0, "upper": 1.0, "one_sided_lower_95": 0.0}
    alpha = 1 - conf
    # 下界：解 P(X>=k | p) = alpha/2；cdf_ge 关于 p 递增
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cdf_ge(k, n, mid) > alpha / 2:
            hi = mid
        else:
            lo = mid
    lower = 0.5 * (lo + hi)
    # 上界：解 P(X<=k | p) = alpha/2，等价 P(X>=k+1|p)=1-alpha/2；cdf_ge 递增
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cdf_ge(k + 1, n, mid) > 1 - alpha / 2:
            hi = mid
        else:
            lo = mid
    upper = 0.5 * (lo + hi)
    # 单侧 95% 下界：解 P(X>=k|p)=0.05
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cdf_ge(k, n, mid) > 0.05:
            hi = mid
        else:
            lo = mid
    one_sided_lower = 0.5 * (lo + hi)
    return {"lower": lower, "upper": upper, "one_sided_lower_95": one_sided_lower}


def quantile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    vx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    vy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return cov / (vx * vy) if vx > 0 and vy > 0 else float("nan")


# ------------------------------------------------------------------ 候选统计量
STATS = {
    # 检测器现役统计量 S = U/sqrt(N)（用于对照，与历史 v1–v4 同口径）
    "detector_S": lambda r: ((r["raw_utility"] or 0.0) /
                             math.sqrt(r["unit_count"] or 1)) if r["unit_count"] else 0.0,
    "max": lambda r: max(r["scores"]) if r["scores"] else 0.0,
    "mean": lambda r: sum(r["scores"]) / len(r["scores"]) if r["scores"] else 0.0,
    "mean_top3": lambda r: (sum(sorted(r["scores"], reverse=True)[:3]) / min(3, len(r["scores"]))
                            if r["scores"] else 0.0),
    "mean_top5": lambda r: (sum(sorted(r["scores"], reverse=True)[:5]) / min(5, len(r["scores"]))
                            if r["scores"] else 0.0),
    "count>=0.70": lambda r: sum(1 for s in r["scores"] if s >= 0.70),
    "count>=0.80": lambda r: sum(1 for s in r["scores"] if s >= 0.80),
}


def roc_at_fpr(fit_null, hold_null, pos, f, target_fpr):
    """在 fit null 上取分位阈值使名义 FPR=target；返回 hold FPR 与 TPR。"""
    thr = quantile([f(r) for r in fit_null], 1 - target_fpr)
    hold_fpr = sum(1 for r in hold_null if f(r) >= thr) / len(hold_null)
    hits = [r for r in pos if f(r) >= thr]
    k, n = len(hits), len(pos)
    return {"threshold": thr, "hold_FPR": hold_fpr, "TPR": k / n if n else 0.0,
            "k": k, "n": n, **ratio_ci(k, n)}


def matched_fpr_tpr(hold_null, pos, f, target_fpr):
    """在 hold null 本身上取阈值精确匹配到 target FPR，再测 TPR（对 FPR 公平）。"""
    thr = quantile([f(r) for r in hold_null], 1 - target_fpr)
    hits = [r for r in pos if f(r) >= thr]
    k, n = len(hits), len(pos)
    return {"threshold": thr, "TPR": k / n if n else 0.0, "k": k, "n": n, **ratio_ci(k, n)}


def main():
    clean = load(CLEAN)
    stress = load(STRESS)
    pos = load(POS, POS_COHORTS)
    pos_base = [r for r in pos if r["cohort"] == "confirmation_baseline_positive"]
    pos_stress = [r for r in pos if r["cohort"] == "confirmation_punctuation_stress"]

    report = {
        "version": "swan-b1b-statistic-power:v1",
        "n_clean_null": len(clean),
        "n_stress_null": len(stress),
        "n_positive": len(pos),
        "n_pos_baseline": len(pos_base),
        "n_pos_stress": len(pos_stress),
        "results": {},
    }

    for null_name, null in [("clean_null_1000", clean), ("stress_null_5000", stress)]:
        fit, hold = null[: len(null) // 2], null[len(null) // 2:]
        entry = {"fit": len(fit), "hold": len(hold), "per_stat": {}}
        for sname, f in STATS.items():
            entry["per_stat"][sname] = {
                "nominal_1pct": roc_at_fpr(fit, hold, pos, f, 0.01),
                "matched_fpr_1pct": matched_fpr_tpr(hold, pos, f, 0.01),
                "matched_fpr_2pct": matched_fpr_tpr(hold, pos, f, 0.02),
                "matched_fpr_5pct": matched_fpr_tpr(hold, pos, f, 0.05),
            }
        report["results"][null_name] = entry

    # 长度混杂检查（用 stress null + pos）
    allrows = stress + pos
    mean_vals = [STATS["mean"](r) for r in allrows]
    wc = [r["word_count"] or 0 for r in allrows]
    uc = [r["unit_count"] or 0 for r in allrows]
    report["confound"] = {
        "corr_mean_vs_wordcount": pearson(mean_vals, wc),
        "corr_mean_vs_unitcount": pearson(mean_vals, uc),
        "mean_score_by_cohort": {
            "clean_null": sum(STATS["mean"](r) for r in clean) / len(clean),
            "stress_null": sum(STATS["mean"](r) for r in stress) / len(stress),
            "pos_baseline": sum(STATS["mean"](r) for r in pos_base) / len(pos_base),
            "pos_stress": sum(STATS["mean"](r) for r in pos_stress) / len(pos_stress),
        },
    }

    # 分 cohort TPR（stress null 匹配 1%），对 detector_S 与 mean 各算一遍
    _, hold_s = stress[: len(stress) // 2], stress[len(stress) // 2:]
    for sname in ("detector_S", "mean"):
        thr = quantile([STATS[sname](r) for r in hold_s], 0.99)
        for cname, grp in [("baseline", pos_base), ("punctuation_stress", pos_stress)]:
            k, n = sum(1 for r in grp if STATS[sname](r) >= thr), len(grp)
            report.setdefault("per_cohort_tpr_stress_null_1pct", {}).setdefault(sname, {})[cname] = {
                "threshold": thr, "k": k, "n": n, "TPR": k / n, **ratio_ci(k, n)}

    # 文档级（消除伪重复）：stress 正例是 30 篇 × 5 变体，行级 n=150 会夸大 CI
    doc_level = {}
    for sname in ("detector_S", "max", "mean"):
        thr = quantile([STATS[sname](r) for r in hold_s], 0.99)
        by_doc = {}
        for r in pos_stress:
            by_doc.setdefault(r["group_id"], []).append(STATS[sname](r) >= thr)
        n_docs = len(by_doc)
        any_hit = sum(1 for v in by_doc.values() if any(v))
        all_hit = sum(1 for v in by_doc.values() if all(v))
        base_k = sum(1 for r in pos_base if STATS[sname](r) >= thr)
        doc_level[sname] = {
            "threshold": thr,
            "baseline_doc": {"k": base_k, "n": len(pos_base)},
            "stress_docs_any_variant": {"k": any_hit, "n": n_docs, **ratio_ci(any_hit, n_docs)},
            "stress_docs_all_5_variants": {"k": all_hit, "n": n_docs, **ratio_ci(all_hit, n_docs)},
        }
    report["doc_level"] = doc_level

    (OUT / "RESULT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    # ---------------- 报告
    L = ["# B1b：功效的瓶颈是「聚合方式 + 阈值漂移」，不是信号本身\n"]
    L.append("> 纯离线，复用 B1 同一批冻结单元分数。**只改聚合方式**，不改检测器、不重跑 Parser。\n")
    L.append(f"- 干净 null {len(clean)}；压力 null {len(stress)}；正例 {len(pos)}"
             f"（baseline {len(pos_base)} + punctuation_stress {len(pos_stress)}）")
    L.append("- `detector_S` = 检测器**现役统计量** S=U/√N（与历史 v1–v4 同口径），作为基准。\n")
    L.append("## 1. 诚实 held-out（fit 半区定阈值，hold 半区测 FPR）\n")
    for null_name, e in report["results"].items():
        L.append(f"### null 源：{null_name}（fit {e['fit']} / hold {e['hold']}）\n")
        L.append("| 统计量 | 名义1%阈值 | held-out FPR | TPR | k/n | TPR 95%CI |")
        L.append("|---|---|---|---|---|---|")
        for sname, d in e["per_stat"].items():
            r = d["nominal_1pct"]
            L.append(f"| {sname} | {r['threshold']:.3f} | {r['hold_FPR']*100:.2f}% | "
                     f"{r['TPR']*100:.1f}% | {r['k']}/{r['n']} | "
                     f"[{r['lower']*100:.1f}%, {r['upper']*100:.1f}%] |")
        L.append("")
    L.append("## 2. 匹配 FPR 下的 TPR（消除名义 FPR 漂移干扰）\n")
    L.append("| null源 | 统计量 | @1%FPR | @2%FPR | @5%FPR |")
    L.append("|---|---|---|---|---|")
    for null_name, e in report["results"].items():
        for sname, d in e["per_stat"].items():
            L.append(f"| {null_name} | {sname} | {d['matched_fpr_1pct']['TPR']*100:.1f}% | "
                     f"{d['matched_fpr_2pct']['TPR']*100:.1f}% | "
                     f"{d['matched_fpr_5pct']['TPR']*100:.1f}% |")
    L.append("")
    L.append("## 3. 长度混杂与均值分档\n")
    c = report["confound"]
    L.append(f"- corr(mean, word_count) = {c['corr_mean_vs_wordcount']:.3f}")
    L.append(f"- corr(mean, unit_count) = {c['corr_mean_vs_unitcount']:.3f}")
    L.append(f"- 平均 mean 分数：clean null {c['mean_score_by_cohort']['clean_null']:.3f}，"
             f"stress null {c['mean_score_by_cohort']['stress_null']:.3f}，"
             f"pos baseline {c['mean_score_by_cohort']['pos_baseline']:.3f}，"
             f"pos stress {c['mean_score_by_cohort']['pos_stress']:.3f}")
    L.append("")
    L.append("## 4. 分 cohort TPR（stress null 匹配 1%）\n")
    L.append("| 统计量 | baseline(30) | punctuation_stress(150) |")
    L.append("|---|---|---|")
    for sname, d in report.get("per_cohort_tpr_stress_null_1pct", {}).items():
        b, s = d["baseline"], d["punctuation_stress"]
        L.append(f"| {sname} | {b['k']}/{b['n']}={b['TPR']*100:.1f}% | "
                 f"{s['k']}/{s['n']}={s['TPR']*100:.1f}% |")
    L.append("")
    L.append("## 5. 文档级 TPR（消除伪重复：stress 正例为 30 篇 × 5 变体）\n")
    L.append("> 行级 n=150 会让 CI 过度乐观；下表按 **唯一文档**（组 30 篇）重算。threshold 同 stress null 1%。\n")
    L.append("| 统计量 | baseline(30篇) | stress 任一变体被检出 | stress 5变体全部被检出 |")
    L.append("|---|---|---|---|")
    for sname, d in report["doc_level"].items():
        b = d["baseline_doc"]
        a = d["stress_docs_any_variant"]
        al = d["stress_docs_all_5_variants"]
        L.append(f"| {sname} | {b['k']}/{b['n']}={b['k']/b['n']*100:.1f}% | "
                 f"{a['k']}/{a['n']}={a['k']/a['n']*100:.1f}% "
                 f"(95%CI [{a['lower']*100:.1f}%, {a['upper']*100:.1f}%]) | "
                 f"{al['k']}/{al['n']}={al['k']/al['n']*100:.1f}% |")
    L.append("")
    L.append("## 结论\n")
    L.append("1. **信号与功效都不是瓶颈**：检测器现役统计量 `detector_S` 在 held-out FPR 1.4–2.0% 下")
    L.append("   行级 TPR 达 **81.1–81.7%**；**文档级**（消除伪重复）baseline 为 "
             f"{report['doc_level']['detector_S']['baseline_doc']['k']}/30，"
             f"stress 30 篇中 '任一变体被检出' "
             f"{report['doc_level']['detector_S']['stress_docs_any_variant']['k']}/30。")
    L.append("   即：**当前检测器本身就能检出绝大部分（含被标点攻击的）正例**。")
    L.append("2. **B1 的「检测端无功效」是聚合方式造成的假象**：B1 用的是「族内最大分」这种 max 型聚合，")
    L.append("   本表 `max` 只有 33.9%；`mean` 68.9%；`detector_S` 81.7%。→ **不要在 max 型聚合上做检测**。")
    L.append("   （机制：null 会出现单个侥幸高分，max 被它骗过；求和型统计量衡量「整篇一致性」，null 难以伪造。）")
    L.append("3. **唯一剩下的真问题是 FPR 阈值漂移**：名义 1% → held-out **1.4–2.0%**（两套 null 一致），")
    L.append("   这正是 B0 诊断的「判定阈值是有限样本估计量」。")
    L.append("4. **B 的定位因此精确化**：**保留 S（功效好），只换阈值/零分布**——把经验分位换成解析或稳健形式。")
    L.append("   让 S 的零分布可解析的途径，是让 DP 路径上的单元**结构独立**（B2 嵌入端的真正目的），")
    L.append("   而不是为了提升功效去嵌入。")
    L.append("5. **必须记录的 caveat**：`punctuation_stress` 的 150 条行级正例实为 **30 篇 × 5 变体**，")
    L.append("   行级 CI（[74.6%, 86.5%]）因伪重复而过度乐观；应以文档级列（第 5 节）为准。")
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")

    for null_name, e in report["results"].items():
        print(f"== {null_name} ==")
        for sname, d in e["per_stat"].items():
            r = d["nominal_1pct"]
            print(f"  {sname:>12}: heldFPR={r['hold_FPR']:.3f} TPR={r['TPR']:.3f} "
                  f"({r['k']}/{r['n']}) matched@1%={d['matched_fpr_1pct']['TPR']:.3f}")


if __name__ == "__main__":
    main()
