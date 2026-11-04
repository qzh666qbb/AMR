"""Add completion-conditional intervals and historical controls to the report.

This reads completed aggregate outputs only. It does not select thresholds,
load a detector, or change the frozen scoring and analysis programs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_swan_blind_eval import interval, metrics

OUT = ROOT / "experiments/swan_multiview_development1/newline_negative_expansion1"
MARKER = "<!-- newline-completion-supplement-v1 -->"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def supplement(folder):
    folder = Path(folder)
    heartbeat = json.loads((folder / "heartbeat.json").read_text())
    audit = json.loads((folder / "AUDIT.json").read_text())
    result = json.loads((folder / "RESULTS.json").read_text())
    calibration = json.loads((folder / "CALIBRATION.json").read_text())
    hashes = json.loads((folder / "SHA256SUMS.json").read_text())
    if heartbeat["state"] != "complete" or not audit["ready"]:
        raise RuntimeError("Completed audited aggregate outputs required")
    for name, expected in hashes.items():
        if Path(name).name != name or sha(folder / name) != expected:
            raise RuntimeError("Aggregate SHA mismatch: " + name)
    if result["threshold_uses_test"] or not calibration["calibration_only"]:
        raise RuntimeError("Calibration scope differs from frozen protocol")
    if calibration["document_threshold"] != result["expanded_calibration_threshold"]["threshold"]:
        raise RuntimeError("Calibration and result threshold differ")

    controls = json.loads((folder / "HISTORICAL_CONTROL_SCORES.json").read_text())["documents"]
    negatives = [r for r in controls if r["arm"] == "paired_negative" and r["attack"] == "newline"]
    statistics = {
        r["record_id"]: sum(s >= result["unit_threshold"] for s in r["unit_bank_max"]) / r["N_expected"]
        if r["complete"] else None for r in negatives
    }
    new_threshold = calibration["document_threshold"]
    decisions = {k: value > new_threshold if value is not None else None
                 for k, value in statistics.items()}
    historical_new = metrics(negatives, decisions, statistics, new_threshold, True)
    if historical_new["denominator"] != 15:
        raise RuntimeError("Historical negative denominator changed")

    conditional = {}
    for key in ("fixed_threshold", "expanded_calibration_threshold"):
        m = result[key]["new_test_FPR"]
        complete = m["complete_decision_documents"]
        conditional[key] = {
            "decisions_positive": m["decisions_positive"],
            "denominator": complete,
            "rate": m["conditional_rate"],
            "binomial_interval_95": interval(m["decisions_positive"], complete) if complete else None,
            "scope": "completion-conditional-generated-test-documents",
        }

    lines = [
        MARKER, "", "## 完整文档条件指标与覆盖限制", "",
        "全分母指标统计每 100 篇输入中产生假阳判定的比例；弃权仍是不确定结果，"
        "不能解释为识别正确。下表只在完整文档中计算条件误报率，两类分母需一起查看。", "",
        "| 新测试口径 | 假阳 / 完整文档 | 条件 FPR | 条件 FPR 双侧 95% 区间 |",
        "|---|---:|---:|---|",
    ]
    for key, label in (("fixed_threshold", "原阈值"),
                       ("expanded_calibration_threshold", "新增校准阈值")):
        m = conditional[key]
        if m["binomial_interval_95"] is None:
            detail, bounds = "无完整文档", "不可用"
        else:
            lo, hi = m["binomial_interval_95"]
            detail = f"{100*m['rate']:.2f}%"
            bounds = f"[{100*lo:.2f}%, {100*hi:.2f}%]"
        lines.append(f"| {label} | {m['decisions_positive']}/{m['denominator']} | {detail} | {bounds} |")
    measured = conditional["expanded_calibration_threshold"]
    if measured["binomial_interval_95"] is not None and measured["binomial_interval_95"][1] <= .05:
        conclusion = "新增校准阈值的完整文档条件 FPR 双侧 95% 上界也不超过 5%。"
    else:
        conclusion = "新增校准阈值的完整文档条件 FPR 双侧 95% 上界不能确认不超过 5%。"
    lines += [
        "", conclusion,
        "上述二项区间在生成文档满足相应独立/交换性假设时解释；它们不提供人类文本、"
        "生成器变化或所有格式扰动的保证。", "",
        "## 其他扰动的历史正例对照", "",
        "以下正例均复用原后 15 篇源文档，只说明同一个阈值变化的开发影响。"
        "新负例仅评估 newline，不能由此推断其他扰动的 FPR。", "",
        "| 条件 | 原阈值 TP/15 | 新增校准阈值 TP/15 | 正例弃权 |",
        "|---|---:|---:|---:|",
    ]
    for attack, values in result["historical_positive_all_conditions"].items():
        old, new = values["fixed"], values["expanded"]
        if old["denominator"] != 15 or new["denominator"] != 15 or old["unresolved"] != new["unresolved"]:
            raise RuntimeError("Historical positive coverage changed")
        lines.append(f"| {attack} | {old['decisions_positive']}/15 | "
                     f"{new['decisions_positive']}/15 | {new['unresolved']}/15 |")
    old = result["historical_newline_control"]["FPR"]
    lines += [
        "", "## 原 15 篇负例的新阈值回看", "",
        f"原 newline 负例在原阈值下 FP={old['decisions_positive']}/15；"
        f"新增校准阈值下 FP={historical_new['decisions_positive']}/15"
        f"（{100*historical_new['rate']:.1f}%），弃权 {historical_new['unresolved']}/15。"
        "这只是已经查看过的历史队列回看，未加入新增校准或测试分母。", "",
        "本次增样未确认修复原队列的残余误报，不能将不同生成批次的 "
        "13.3% 与 1.0% 解释为同分布改善。重校准也没有降低新测试的实测误报数。", "",
        "补充复现：`scripts/supplement_swan_newline_expansion.py`；图表："
        "`scripts/plot_swan_newline_expansion.py`。检测分数与阈值未因补充分析改变。", "",
    ]

    report = folder / "REPORT.md"
    prefix = report.read_text().split(MARKER, 1)[0].rstrip()
    if "原 15 篇负例 / 新增校准阈值" not in prefix:
        positive = result["expanded_calibration_threshold"]["historical_TPR"]
        row = (f"| 原 15 篇负例 / 新增校准阈值 | {new_threshold:.9g} | "
               f"{historical_new['decisions_positive']}/15 ({100*historical_new['rate']:.1f}%) | "
               f"{positive['decisions_positive']}/15 ({100*positive['rate']:.1f}%) | "
               f"{historical_new['unresolved']} |")
        prefix = prefix.replace("\n\n单元阈值固定为", "\n" + row + "\n\n单元阈值固定为", 1)
    report.write_text(prefix + "\n\n" + "\n".join(lines) + "\n")
    artifact = {
        "at": datetime.now(timezone.utc).isoformat(),
        "schema": "swan-newline-completion-supplement/v1",
        "results_sha256": sha(folder / "RESULTS.json"),
        "calibration_sha256": sha(folder / "CALIBRATION.json"),
        "postprocessor_sha256": sha(Path(__file__)),
        "threshold_changed_by_supplement": False,
        "detector_calls": 0,
        "input_aggregate_SHA_verified": True,
        "conditional_test_FPR": conditional,
        "historical_negative_FPR_at_expanded_threshold": historical_new,
    }
    p = folder / "COMPLETION_SUPPLEMENT.json"
    p.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    hashes["REPORT.md"] = sha(report)
    hashes[p.name] = sha(p)
    (folder / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, ensure_ascii=False))


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", type=Path, default=OUT)
    supplement(cli.parse_args().output)
