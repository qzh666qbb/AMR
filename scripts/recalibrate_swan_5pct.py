"""Separate development recalibration using only calibration-negative scores."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_swan_blind_eval import ATTACKS, doc_threshold, metrics, save_json
from src.core.swan_blind_protocol import digest, official_two_stage_summary

RULE = {
    "schema": "swan-condition-calibration/v1",
    "target_document_fpr": .05,
    "threshold_rule": "max_over_conditions_of_complete_calibration_negative_empirical_95pct",
    "decision_rule": "statistic > threshold",
    "unit_threshold_rule": "retain_source_method_unit_threshold",
    "scope": "posthoc-development-calibration-rule-change",
    "numeric_threshold_uses_test": False,
    "policy_chosen_after_previous_test_observed": True,
    "independent_confirmatory_test": False,
    "failed_calibration_documents": "retain_in_status_no_partial_candidates_no_supplement",
    "failed_test_documents": "abstain_retain_full_denominator",
    "missing_condition_calibration": "unavailable",
    "new_generation_or_detector_calls": 0,
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statistic(row, unit):
    if not row["complete"] or unit is None:
        return None
    if row["method"].startswith("P0"):
        return official_two_stage_summary(row["unit_bank_max"])["z"]
    return sum(s >= unit for s in row["unit_bank_max"]) / row["N_expected"]


def threshold_from_calibration(rows, unit):
    """No test rows or positive labels are allowed in this selector."""
    if any(r["arm"] != "paired_negative" or r["split"] != "calibration" for r in rows):
        raise ValueError("Only calibration-negative documents may select threshold")
    conditions = {}
    for attack in ATTACKS:
        subset = [r for r in rows if r["attack"] == attack]
        complete = [r for r in subset if r["complete"]]
        values = [statistic(r, unit) for r in complete]
        values = [v for v in values if v is not None]
        c = doc_threshold(values, .05) if values else None
        conditions[attack] = {
            "negative_documents": len(subset),
            "negative_source_documents": len({r["document_id"] for r in subset}),
            "complete_documents": len(complete),
            "unresolved_documents": len(subset) - len(complete),
            "condition_empirical_threshold": c,
            "nontrivial_distribution_free_5pct_available": len(values) >= 19,
            "max_order_statistic_fpr_upper_one_sided_95":
                1 - .05 ** (1 / len(values)) if values else None,
        }
    available = unit is not None and all(
        c["condition_empirical_threshold"] is not None for c in conditions.values())
    threshold = max(c["condition_empirical_threshold"] for c in conditions.values()) \
        if available else None
    for attack, info in conditions.items():
        selected = [statistic(r, unit) for r in rows if r["attack"] == attack
                    and r["complete"]]
        info["FP_at_unified_threshold"] = sum(v > threshold for v in selected) \
            if available else None
        info["FPR_all_negative_documents"] = info["FP_at_unified_threshold"] / \
            info["negative_documents"] if available else None
        info["FPR_complete_negative_documents"] = info["FP_at_unified_threshold"] / \
            info["complete_documents"] if available else None
    return threshold, conditions


def evaluate(data, unit, threshold, available):
    stats = {r["record_id"]: statistic(r, unit) for r in data}
    decisions = {r["record_id"]: stats[r["record_id"]] > threshold
                 if available and stats[r["record_id"]] is not None else None for r in data}
    conditions = {}
    for attack in (*ATTACKS, "pooled"):
        subset = [r for r in data if r["split"] == "test"
                  and (attack == "pooled" or r["attack"] == attack)]
        conditions[attack] = {
            label: metrics([r for r in subset if r["arm"] == arm], decisions,
                           stats, threshold, available)
            for label, arm in (("TPR", "positive"), ("FPR", "paired_negative"))
        }
    cal = [r for r in data if r["split"] == "calibration" and r["arm"] == "paired_negative"]
    return {"document_threshold": threshold,
            "calibration": metrics(cal, decisions, stats, threshold, available),
            "conditions": conditions}


def percentage(metric):
    if metric["rate"] is None:
        return "未校准"
    return f"{metric['decisions_positive']}/{metric['denominator']}（{100*metric['rate']:.1f}%）"


def report(folder, result):
    lines = [
        "# 5% 目标下的阈值调整评估",
        "",
        "本次采用阈值调整路径，复用已完成的全文与盲检测评分。"
        "新增规则先对七类扰动分别计算完整校准负例的 95% 经验分位，"
        "再取最高值作为所有输入共用的文档阈值；检测时不需要知道扰动类别。"
        "单元阈值保持来源配置，判定仍使用严格大于。",
        "",
        "数值阈值只读取前 15 篇校准负例；后 15 对文档用于报告结果。"
        "调整规则是在上一轮测试结果已知后提出的，且文本经过既往探索，"
        "因此本次属于事后开发诊断，不能称为新的独立确认试验。",
        "没有增加负例、生成调用或重新运行解析/检测器。"
        "原运行的输入、候选、弃权与历史结果保持来源记录。",
        "",
        "来源为本目录的 [SOURCE_FREEZE.json](SOURCE_FREEZE.json)、"
        "[SOURCE_RESULTS.json](SOURCE_RESULTS.json) 和 "
        "[SOURCE_DOCUMENT_SCORES.json](SOURCE_DOCUMENT_SCORES.json)，"
        "规则冻结见 [POLICY.json](POLICY.json)，"
        "逐条件结果和区间见 [RESULTS.json](RESULTS.json)。",
        "",
        "## 汇总结果",
        "",
        "| 方法 | 原阈值 → 调整阈值 | 原 TPR → 调整 TPR | 原 FPR → 调整 FPR | 正/负弃权 |",
        "|---|---|---|---|---|",
    ]
    for method, measured in result["methods"].items():
        old, new = measured["original_5pct"], measured["condition_guard_5pct"]
        a, b = old["conditions"]["pooled"], new["conditions"]["pooled"]
        old_c, new_c = old["document_threshold"], new["document_threshold"]
        threshold = f"{old_c:.9g} → {new_c:.9g}" if new_c is not None and old_c is not None \
            else "未校准"
        lines.append(f"| {method} | {threshold} | {percentage(a['TPR'])} → "
                     f"{percentage(b['TPR'])} | {percentage(a['FPR'])} → "
                     f"{percentage(b['FPR'])} | {b['TPR']['unresolved']}/"
                     f"{b['FPR']['unresolved']} |")
    lines += [
        "",
        "汇总的 105 个变体来自 15 个源文档，采用源文档聚类 bootstrap；"
        "每个扰动仍只有 15 篇正例与 15 篇负例。弃权保留在全分母中，"
        "降低 FPR 的同时也降低 TPR，不能把弃权当作正确识别负例。",
        "",
        "## 各条件实测结果",
        "",
        "| 方法 | 扰动 | 原 TPR → 调整 TPR | 原 FPR → 调整 FPR | 正/负弃权 |",
        "|---|---|---|---|---|",
    ]
    for method, measured in result["methods"].items():
        for attack in ATTACKS:
            old = measured["original_5pct"]["conditions"][attack]
            new = measured["condition_guard_5pct"]["conditions"][attack]
            lines.append(f"| {method} | {attack} | {percentage(old['TPR'])} → "
                         f"{percentage(new['TPR'])} | {percentage(old['FPR'])} → "
                         f"{percentage(new['FPR'])} | {new['TPR']['unresolved']}/"
                         f"{new['FPR']['unresolved']} |")
    lines += [
        "",
        "## 校准与可用性",
        "",
        "| 方法 | 扰动 | 完整校准负例/15 | 条件 95% 阈值 | 统一阈值下校准 FP |",
        "|---|---|---|---|---|",
    ]
    for method, measured in result["methods"].items():
        for attack, info in measured["calibration_by_condition"].items():
            c = info["condition_empirical_threshold"]
            value = f"{c:.9g}" if c is not None else "未校准"
            lines.append(f"| {method} | {attack} | {info['complete_documents']}/"
                         f"{info['negative_documents']} | {value} | "
                         f"{info['FP_at_unified_threshold']} |")
    lines += [
        "",
        "当前每条件完整校准数不足 19，5% 经验分位实际上等于该条件最大负例分数。"
        "统一阈值因而等于全部完整校准负例的最大值，"
        "这比原来的混合 5% 分位更保守。校准侧零误报不是总体误报控制证明。",
        "",
        "## 结果解释与样本量",
        "",
    ]
    for method, measured in result["methods"].items():
        new = measured["condition_guard_5pct"]
        unresolved = new["conditions"]["pooled"]["FPR"]["unresolved"]
        residual = [f"{attack}：{percentage(new['conditions'][attack]['FPR'])}"
                    for attack in ATTACKS
                    if new["conditions"][attack]["FPR"]["rate"] is not None and
                    new["conditions"][attack]["FPR"]["rate"] > .05]
        if residual:
            lines.append(f"- {method} 仍有超过 5% 的条件级误报：{'；'.join(residual)}。"
                         "不能宣称所有扰动均已控制在 5%。")
        else:
            lines.append(f"- {method} 当前没有观测到超过 5% 的条件级误报；"
                         f"测试负例仍有 {unresolved}/105 弃权。"
                         "这是小样本观测，不是总体上限证明。")
    lines += [
        "- 每条件 0/15 误报的双侧 95% Clopper–Pearson 上界为 21.8%；"
        "七个相关变体不能算作七个新增独立负例。",
        "- 若只调整文档阈值且满足交换性，至少 19 个完整独立校准负例才能得到"
        "非平凡的 5% split-conformal 顺序统计量阈值；该结论是边际保证，"
        "不能自动给出所有扰动同时成立的保证。",
        "- 若要用零误报验证总体 FPR≤5%，单个预先固定条件在单侧 95% "
        "Clopper–Pearson 口径下至少需要 59 篇独立测试负例；"
        "采用本项目双侧 95% 口径则至少需要 72 篇。"
        "这是假设观测到零误报时的最低数，不是默认够用的实验规模。"
        "同时保证多个条件需另作置信水平调整；存在误报时也需更多样本。",
        "- 增样应新建版本并分离校准、独立人类测试；"
        "不能把已查看的后 15 篇负例并入校准后仍当作测试，"
        "也不能把配对生成文本当作独立人类文本。",
        "",
        "## 验证与输出",
        "",
        "原指标由来源文档分数独立重算；新阈值数值不依赖测试正负分数。"
        "阈值、完整分母、弃权、同分和逐条件指标见 "
        "[AUDIT.json](AUDIT.json) 与 [METRICS.csv](METRICS.csv)。"
        "所有输出的 SHA 见 [SHA256SUMS.json](SHA256SUMS.json)。"
        "尚未完成的窗口不写入本次实测表，后台完成后按同一规则追加。",
        "",
    ]
    (folder / "REPORT.md").write_text("\n".join(lines))


def run(source, output):
    output.mkdir(parents=True, exist_ok=True)
    before = {name: sha(source / name) for name in
              ("FREEZE.json", "RESULTS.json", "DOCUMENT_SCORES.json")}
    blobs = {name: (source / name).read_bytes() for name in before}
    after = {name: sha(source / name) for name in before}
    if before != after or any(hashlib.sha256(blobs[n]).hexdigest() != before[n] for n in before):
        raise ValueError("Source snapshot changed during read")
    freeze, previous, documents = (json.loads(blobs[name]) for name in before)
    if previous["freeze_sha256"] != digest(freeze) or documents["freeze_sha256"] != digest(freeze):
        raise ValueError("Source freeze binding mismatch")
    rows = documents["documents"]
    if set(previous["complete_methods"]) != {r["method"] for r in rows}:
        raise ValueError("Source methods and scores do not form a complete snapshot")
    policy = {**RULE, "source_freeze_sha256": digest(freeze),
              "calibration_ids": freeze["calibration_ids"],
              "test_ids": freeze["primary_test_ids"]}
    policy_path = output / "POLICY.json"
    if policy_path.exists() and json.loads(policy_path.read_text()) != policy:
        raise ValueError("Recalibration policy already frozen differently")
    save_json(policy_path, policy)
    for name, content in blobs.items():
        (output / ("SOURCE_" + name)).write_bytes(content)
    result = {
        "schema": "swan-5pct-recalibration/v1",
        "scope": RULE["scope"], "policy_sha256": digest(policy),
        "source_files_sha256": before,
        "source_freeze_sha256": digest(freeze),
        "complete_methods": previous["complete_methods"],
        "planned_methods": previous["planned_methods"],
        "independent_human_test": False, "new_negative_documents": 0,
        "target_is_nominal": True, "methods": {},
        "zero_false_positive_sample_size": {
            "one_sided_95_fpr_le_5pct": math.ceil(math.log(.05) / math.log(.95)),
            "two_sided_95_fpr_le_5pct": math.ceil(math.log(.025) / math.log(.95))},
    }
    csv_rows, checks = [], []
    for method in previous["complete_methods"]:
        measured = previous["methods"][method]
        data = [r for r in rows if r["method"] == method]
        if len(data) != 420:
            raise ValueError("Method lacks the full document cohort")
        unit = measured["unit_threshold"]
        cal = [r for r in data if r["arm"] == "paired_negative" and r["split"] == "calibration"]
        if {r["document_id"] for r in cal} != set(freeze["calibration_ids"]):
            raise ValueError("Calibration IDs differ from source freeze")
        threshold, by_condition = threshold_from_calibration(cal, unit)
        available = measured["availability"] == "available" and threshold is not None
        old = measured["operating_points"]["document_5pct"]
        replay = evaluate(data, unit, old["document_threshold"], measured["availability"] == "available")
        for attack, condition in old["conditions"].items():
            for label in ("TPR", "FPR"):
                for field in ("decisions_positive", "denominator", "rate", "unresolved"):
                    if condition[label][field] != replay["conditions"][attack][label][field]:
                        raise ValueError("Independent old-metric recomputation differs")
        new = evaluate(data, unit, threshold, available)
        result["methods"][method] = {
            "configuration_sha256": measured["configuration_sha256"],
            "unit_threshold": unit, "availability": "available" if available else "unavailable",
            "calibration_manifest_sha256": digest([(r["record_id"], r["text_sha256"]) for r in cal]),
            "calibration_by_condition": by_condition,
            "threshold_numeric_selection_uses_test": False,
            "original_5pct": old, "condition_guard_5pct": new}
        checks.append({"method": method, "original_metrics_recomputed": True,
                       "selector_received_only_calibration_negatives": True,
                       "calibration_source_documents": len({r["document_id"] for r in cal}),
                       "test_source_ids_disjoint": not set(freeze["calibration_ids"]) &
                           {r["document_id"] for r in data if r["split"] == "test"},
                       "condition_calibration_empirical_FPR_le_5pct":
                           all(v["FPR_complete_negative_documents"] <= .05
                               for v in by_condition.values()) if available else None})
        for strategy, op in (("original_5pct", old), ("condition_guard_5pct", new)):
            for attack, values in op["conditions"].items():
                a, b = values["TPR"], values["FPR"]
                csv_rows.append({
                    "method": method, "strategy": strategy, "condition": attack,
                    "nominal_target": .05, "unit_threshold": unit,
                    "document_threshold": op["document_threshold"],
                    "TP": a["decisions_positive"], "positive_denominator": a["denominator"],
                    "TPR": a["rate"], "positive_unresolved": a["unresolved"],
                    "FP": b["decisions_positive"], "negative_denominator": b["denominator"],
                    "FPR": b["rate"], "negative_unresolved": b["unresolved"]})
    save_json(output / "RESULTS.json", result)
    save_json(output / "AUDIT.json", {"policy_sha256": digest(policy),
                                    "source_sha_verified": True, "checks": checks,
                                    "policy_is_posthoc": True, "detector_calls": 0})
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)
    (output / "METRICS.csv").write_text(stream.getvalue())
    report(output, result)
    save_json(output / "SHA256SUMS.json", {
        p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()
        and p.name != "SHA256SUMS.json"})
    print(json.dumps({m: v["condition_guard_5pct"]["conditions"]["pooled"]
                      for m, v in result["methods"].items()}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
