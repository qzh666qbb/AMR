"""Independent calibration and full-denominator final confirmation metrics."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_swan_blind_eval import save_json, unit_threshold
from src.core.swan_blind_protocol import digest, official_two_stage_summary

CONDITIONS = ("baseline", "newline", "merge", "comma", "space")


def q95(values):
    k = math.ceil(.95 * (len(values) + 1))
    return sorted(values)[k-1] if values and k <= len(values) else None


def cp(k, n, alpha=.05):
    from scipy.stats import beta
    if not n:
        return None
    return [0. if not k else float(beta.ppf(alpha/2, k, n-k+1)),
            1. if k == n else float(beta.ppf(1-alpha/2, k+1, n-k))]


def statistic(row, unit):
    if not row["complete"] or unit is None:
        return None
    if row["method"].startswith("P0"):
        return official_two_stage_summary(row["unit_bank_max"])["z"]
    return sum(v >= unit for v in row["unit_bank_max"]) / row["N_expected"]


def calibrate(rows, method):
    negative = [r for r in rows if r["method"] == method
                and r["split"] == "calibration" and r["arm"] == "negative"]
    complete = [r for r in negative if r["complete"]]
    values = [v for r in complete for v in r["unit_bank_max"] if v is not None]
    saturated = False
    unit = .65 if method.startswith("P0") else None
    if not method.startswith("P0") and values:
        unit, saturated = unit_threshold(values, .01)
    groups = {}
    for source in ("generated", "human"):
        for condition in CONDITIONS:
            subset = [r for r in negative if r["source"] == source and r["attack"] == condition]
            stats = [statistic(r, unit) for r in subset]
            stats = [s for s in stats if s is not None]
            c = q95(stats)
            groups[f"{source}:{condition}"] = {
                "frozen_denominator": len(subset), "complete": len(stats),
                "threshold": c, "order_index": math.ceil(.95*(len(stats)+1)),
            }
    available = unit is not None and not saturated \
        and all(g["threshold"] is not None for g in groups.values())
    threshold = max(g["threshold"] for g in groups.values()) if available else None
    # A fraction cutoff of one rejects every possible document.
    trivial = not method.startswith("P0") and threshold == 1.
    available = available and not trivial
    return {"unit_threshold": unit, "unit_threshold_saturated": saturated,
            "unit_rule": "pooled-complete-new-calibration-unit-upper-1pct",
            "unit_count": len(values), "document_threshold": threshold,
            "document_rule": "max-source-condition-order-ceil(0.95*(n+1))",
            "groups": groups, "available": available,
            "trivial_document_threshold": trivial,
            "negative_manifest_sha256": digest([
                (r["record_id"], r["text_sha256"]) for r in negative])}


def metric(rows, decisions, available):
    n = len(rows)
    k = sum(decisions.get(r["record_id"]) is True for r in rows)
    complete = sum(r["complete"] for r in rows)
    unresolved = sum(decisions.get(r["record_id"]) is None for r in rows)
    return {
        "positive_decisions": k, "frozen_denominator": n,
        "rate_all": k/n if n and available else None,
        "complete_documents": complete, "complete_rate": complete/n if n else None,
        "rate_complete": k/complete if complete and available else None,
        "unresolved": unresolved,
        "cp95_all": cp(k, n) if available else None,
        "cp95_complete": cp(k, complete) if available else None,
        "cp99_complete": cp(k, complete, .01) if available else None,
        "source_articles": len({r["source_article_sha256"] for r in rows}),
        "failure_counts": dict(Counter(reason for r in rows for reason in r["failure_counts"])),
    }


def analyze(folder):
    frozen = json.loads((folder / "EVAL_FREEZE.json").read_text())
    scored = json.loads((folder / "DOCUMENT_SCORES.json").read_text())
    if scored["freeze_sha256"] != digest(frozen):
        raise RuntimeError("Score/freeze provenance differs")
    rows = scored["documents"]
    candidate = frozen["candidate_method"]
    methods = ["P0_raw", "P0_strict", candidate]
    if {r["method"] for r in rows} != set(methods):
        raise RuntimeError("Cannot analyze an incomplete confirmation")
    expected = {(m, source, arm, split, attack): n
                for m in methods for attack in CONDITIONS
                for source, arm, split, n in (
                    ("human", "negative", "calibration", 100),
                    ("human", "negative", "test", 200),
                    ("generated", "negative", "calibration", 100),
                    ("generated", "negative", "test", 100),
                    ("generated", "positive", "test", 50))}
    observed = Counter((r["method"], r["source"], r["arm"], r["split"], r["attack"]) for r in rows)
    if observed != Counter(expected) or len({r["record_id"] for r in rows}) != len(rows):
        raise RuntimeError("Frozen document denominators changed")
    cal, decisions, results = {}, {}, {}
    for method in methods:
        data = [r for r in rows if r["method"] == method]
        c = cal[method] = calibrate(rows, method)
        decisions[method] = {
            r["record_id"]: statistic(r, c["unit_threshold"]) > c["document_threshold"]
            if c["available"] and statistic(r, c["unit_threshold"]) is not None
            else None for r in data}
        groups = {}
        for condition in CONDITIONS:
            test = [r for r in data if r["split"] == "test" and r["attack"] == condition]
            groups[condition] = {
                "TPR": metric([r for r in test if r["arm"] == "positive"],
                              decisions[method], c["available"]),
                "generated_FPR": metric([r for r in test if r["source"] == "generated"
                                        and r["arm"] == "negative"],
                                       decisions[method], c["available"]),
                "human_FPR": metric([r for r in test if r["source"] == "human"],
                                   decisions[method], c["available"]),
            }
        results[method] = {"calibration": c, "conditions": groups}
    paired = {}
    if cal[candidate]["available"] and cal["P0_strict"]["available"]:
        import numpy as np
        values = []
        bykey = {(r["method"], r["document_id"], r["attack"]): r for r in rows
                 if r["arm"] == "positive" and r["split"] == "test"}
        ids = sorted({r["document_id"] for r in bykey.values()})
        for doc in ids:
            diffs = []
            for attack in ("merge", "comma", "space"):
                a, b = bykey[(candidate, doc, attack)], bykey[("P0_strict", doc, attack)]
                diffs.append(int(decisions[candidate][a["record_id"]] is True)
                             - int(decisions["P0_strict"][b["record_id"]] is True))
            values.append(sum(diffs)/3)
        rng = np.random.default_rng(2026102309)
        values = np.array(values)
        draws = rng.integers(0, len(values), (10000, len(values)))
        bounds = np.quantile(values[draws].mean(axis=1), [.025, .975])
        paired = {"difference": float(values.mean()), "cluster_bootstrap95": bounds.tolist(),
                  "independent_positive_source_articles": len(values), "replicates": 10000,
                  "unresolved_treated_as_no_positive_decision": True}
    human_gate = all(
        g["human_FPR"]["cp99_complete"] is not None
        and g["human_FPR"]["cp99_complete"][1] <= .05
        for g in results[candidate]["conditions"].values())
    coverage_gate = all(g[label]["complete_rate"] >= .8
                        for g in results[candidate]["conditions"].values()
                        for label in ("TPR", "human_FPR"))
    improvement_gate = bool(paired) and paired["cluster_bootstrap95"][0] > 0
    output = {
        "schema": "swan-last-round-confirmation/v1", "freeze_sha256": digest(frozen),
        "candidate_method": candidate, "methods": results, "paired_improvement": paired,
        "gates": {"positive_improvement": improvement_gate,
                  "human_fpr_5conditions_bonferroni99": human_gate,
                  "positive_and_human_completion80": coverage_gate},
        "usable_recovery_confirmed": improvement_gate and human_gate and coverage_gate,
        "no_successor": True, "line_closed": True,
        "scope": "frozen-generated-cohort-and-historical-wikitext-reference",
        "human_generated_metrics_pooled": False,
        "human_conditional_fpr_scope": "only documents completed by frozen pipeline",
        "calibration_uses_test": False, "full_denominators_verified": True,
        "formal_conformal_guarantee_claimed": False,
        "unit_and_document_thresholds_share_calibration_negatives": True,
    }
    save_json(folder / "CALIBRATION.json", cal)
    save_json(folder / "RESULTS.json", output)
    lines = ["# SWAN 最后一轮独立确认", "",
             "候选窗口为开发选择后冻结的 " + candidate + "。数值阈值仅用新增完整校准负例，"
             "人类与生成负例分别报告；失败样本保留在冻结分母。",
             "人类参考是固定版本历史 WikiText 文本，不代表全部人类文本。", "",
             "| 方法 | 条件 | TPR/50 | 生成 FPR/100 | 人类 FPR/200 | 人类完整率 |",
             "|---|---|---|---|---|---|"]
    def fmt(g):
        return "校准不可用" if g["rate_all"] is None else \
            f"{g['positive_decisions']}/{g['frozen_denominator']} ({100*g['rate_all']:.1f}%)"
    for method in methods:
        for condition, group in results[method]["conditions"].items():
            lines.append(f"| {method} | {condition} | {fmt(group['TPR'])} | "
                         f"{fmt(group['generated_FPR'])} | {fmt(group['human_FPR'])} | "
                         f"{group['human_FPR']['complete_documents']}/200 |")
    lines += ["", "预定性能门：" + json.dumps(output["gates"], ensure_ascii=False) + "。",
              "全部门通过，支持本协议范围内的恢复收益。" if output["usable_recovery_confirmed"]
              else "预定门未全部通过，不能主张可用恢复成功；按系统审计与恢复局限收口。",
              "", "逐条件全分母与完整文档区间、弃权原因和校准组数见 `RESULTS.json`。"
              "人类误报门采用五条件 Bonferroni 校正的双侧 99% Clopper–Pearson 区间；"
              "其适用性依赖来源文章独立与冻结处理的交换性假设。",
              "单元与文档阈值共用新增校准负例，因此顺序统计量规则只作为冻结的"
              "经验校准方法，不主张正式 conformal 保证。",
              "配对收益按 50 个正例来源文章聚类，不将三种变体当作独立样本。",
              "", "最终图、源文档注入诊断、开发多窗口比较与 SHA 另随归档交付。"]
    (folder / "REPORT.md").write_text("\n".join(lines) + "\n")
    return output


if __name__ == "__main__":
    import argparse
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", type=Path, required=True)
    print(json.dumps(analyze(cli.parse_args().output)["gates"]))
