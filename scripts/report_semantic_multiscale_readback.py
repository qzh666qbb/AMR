"""Render a sourced report and export real aggregate metrics."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]
import run_semantic_multiscale_readback as run


def main():
    run.verify()
    r = run.read(run.OUT / "RESULT.json")
    audit = run.read(run.OUT / "INDEPENDENT_AUDIT.json")
    if not audit["Go"] or audit["result_sha256"] != run.sha(run.OUT / "RESULT.json"):
        raise RuntimeError("Independent audit required")
    normalized = []
    for summary in r["summaries"]:
        subset = [x for x in r["rows"] if x["condition"] == summary["condition"]
                  and x["method"] == summary["method"]]
        normalized.append({
            "condition": summary["condition"], "method": summary["method"],
            "scenarios": 12, "planned_carriers": 96,
            "macro_f1_percent": 100 * summary["macro_f1"]["mean"],
            "macro_f1_ci_low_percent": 100 * summary["macro_f1"]["CI95"][0],
            "macro_f1_ci_high_percent": 100 * summary["macro_f1"]["CI95"][1],
            "historical_identity_f1_percent": 100 * summary["identity_f1"]["mean"],
            "correct_carriers": summary["correct_carriers"],
            "complete_message_correct": sum(x["complete_correct"] for x in subset),
            "wrong_unique_messages": summary["wrong_unique_messages"],
            "message_erasures": summary["message_erasures"],
            "emitted_candidates": summary["emitted"],
            "candidate_observations": summary["observations"],
            "source_result_sha256": run.sha(run.OUT / "RESULT.json")})
    run.atomic(run.OUT / "NORMALIZED_SUMMARY.json", {
        "schema": "semantic-score-msa-normalized-summary/v1",
        "source_result_sha256": run.sha(run.OUT / "RESULT.json"), "rows": normalized,
        "schema_note": "Raw RESULT summaries.complete_correct holds a rate interval, not an integer; counts independently derived from per-scenario rows."})
    with (run.OUT / "METRICS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(normalized[0]))
        writer.writeheader()
        writer.writerows(normalized)
    os.environ["MPLCONFIGDIR"] = str(run.PRIVATE / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    methods = ["legacy_W", "legacy_E1", "mean", "gate", "gate_wide", "attention"]
    captions = ["Legacy W", "Legacy E1", "Mean", "Gate", "Wide gate", "Attention"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=170,
                             gridspec_kw={"width_ratios": [1.55, 1]})
    selected = [next(s for s in normalized if s["condition"] == "paraphrase" and s["method"] == m)
                for m in methods]
    means = np.array([s["macro_f1_percent"] for s in selected])
    low = np.array([s["macro_f1_ci_low_percent"] for s in selected])
    high = np.array([s["macro_f1_ci_high_percent"] for s in selected])
    colors = ["#8b96a6", "#8b96a6", "#72a9b9", "#227b91", "#72a9b9", "#ba5c55"]
    axes[0].bar(range(6), means, color=colors, width=.67)
    axes[0].errorbar(range(6), means, yerr=[means-low, high-means],
                    fmt="none", color="#313a49", capsize=3, linewidth=1)
    for i, value in enumerate(means):
        axes[0].text(i, value + 1.1, f"{value:.2f}", ha="center", fontsize=9)
    axes[0].set_xticks(range(6), captions, rotation=20, ha="right")
    axes[0].set_ylabel("Blind event macro-F1 (%)")
    axes[0].set_title("Paraphrase: attention did not improve readback", loc="left", fontsize=11)
    axes[0].set_ylim(0, 48)
    axes[0].grid(axis="y", alpha=.2)
    axes[0].set_axisbelow(True)
    primary = r["primary"]
    axes[0].text(.02, .96,
                 f"Attention - gate: {100*primary['mean']:.2f} pp\n"
                 f"95% descriptive CI [{100*primary['CI95'][0]:.2f}, {100*primary['CI95'][1]:.2f}]",
                 transform=axes[0].transAxes, va="top", fontsize=9)
    axes[1].axis("off")
    table = [[condition.capitalize(), "0 / 12", "0 / 12"]
             for condition in run.CONDITIONS]
    cell = axes[1].table(cellText=table, colLabels=["Condition", "Gate", "Attention"],
                         cellLoc="center", loc="center", colWidths=[.42, .29, .29])
    cell.auto_set_font_size(False)
    cell.set_fontsize(11)
    cell.scale(1, 1.8)
    for (i, j), item in cell.get_celld().items():
        item.set_edgecolor("#dae0e7")
        if i == 0:
            item.set_facecolor("#edf2f6")
    axes[1].set_title("Complete message recovery", loc="left", fontsize=11)
    axes[1].text(.5, .19, "All 65,536 eight-symbol messages searched\nTies and erasures count as failures",
                 ha="center", fontsize=9, transform=axes[1].transAxes)
    fig.suptitle("Paper two: score-level multi-scale readback reevaluation",
                 x=.055, ha="left", fontsize=14, weight="bold")
    fig.text(.055, .02,
             "Source: RESULT.json. 12 old scenarios; grouped cross-validation; constructed/model-assisted references. No human gold or new test data.",
             fontsize=8, color="#596576")
    fig.tight_layout(rect=(0, .075, 1, .92))
    fig.savefig(run.OUT / "RESULTS.png", bbox_inches="tight")
    fig.savefig(run.OUT / "RESULTS.pdf", bbox_inches="tight")
    plt.close(fig)
    table_lines = []
    names = {"legacy_W": "原W", "legacy_E1": "原E1", "legacy_C": "原MiniLM",
             "s6": "短窗", "s12": "长窗", "ssub": "事件子图",
             "mean": "等权融合", "gate": "简单门控", "gate_wide": "宽门控",
             "attention": "跨尺度注意力"}
    for mode in run.ALL_MODES:
        clean = next(s for s in normalized if s["method"] == mode and s["condition"] == "clean")
        para = next(s for s in normalized if s["method"] == mode and s["condition"] == "paraphrase")
        table_lines.append(
            f"| {names[mode]} | {clean['macro_f1_percent']:.2f}% | {para['macro_f1_percent']:.2f}% "
            f"| {para['correct_carriers']}/96 | {para['complete_message_correct']}/12 |")
    ci = [100*x for x in primary["CI95"]]
    report = f"""# 第二篇多尺度注意力：集成后的开发复评

2026-10-26。多尺度模块已接入符号读回接口，完成84个模型的分组训练、
72版本预测封存、完整候选消息解码与独立审计。
**当前配置的注意力没有改善符号或完整消息读回。**
复述macro-F1为22.97%，简单门控28.93%；配对差
{100*primary['mean']:.2f}个百分点，场景bootstrap描述性95%区间
[{ci[0]:.2f}, {ci[1]:.2f}]。全部十种方法在四种正例条件下完整恢复均0/12。
来源：[原始结果](RESULT.json)、[规范化汇总](NORMALIZED_SUMMARY.json)。

![多尺度符号读回复评](RESULTS.png)

## 1. 本次集成和数据范围

同一候选事件的W6、W12、E1_12匹配分数形成3×24输入，沿用原跨尺度
Q/K/V架构并新增四类符号聚合头。没有用旧二分类checkpoint产生ABCD。
单视图、等权、门控、注意力与宽门控共同采用训练归一化、符号头和
缺失掩码；旧W/E1/C按原冻结参数另栏复现。
源码见[符号融合器](../../src/core/semantic_multiscale.py)、
[复评入口](../../scripts/run_semantic_multiscale_readback.py)，
方案见[专项设计](../../第二篇_多尺度注意力集成与复评方案.md)。

使用旧12场景、72版本、1523个真实封存候选。12折各8场景训练、
3场景校准、1场景测试，同一场景全部变体同组。
只对训练组原句/复述参考载体监督；epoch及拒绝阈值只读校准组。
测试过程独立加载模型与特征，不读标签或正确位置。
七种融合各12折共84个模型，另有三种legacy复现。
参考是构造位置及模型辅助复述标签，尚无真人金标准。
来源：[协议](PROTOCOL.md)、[冻结](FREEZE.json)、[分组](SPLITS.json)、
[训练与选择记录](TRAINING_RESULTS.json)、[预测封存](PREDICTION_SEAL.json)。

## 2. 符号读出结果

| 方法 | 原句盲事件macro-F1 | 复述盲事件macro-F1 | 复述正确载体，全分母 | 复述完整消息 |
|---|---:|---:|---:|---:|
{chr(10).join(table_lines)}

macro-F1逐场景按四类平均，再按12场景平均；它与旧identity-F1定义不同，
不能直接比较旧报告的百分比。旧identity-F1已逐项复现144项，
例如原W复述仍22.19%，attention复述22.81%，gate28.82%。
完整消息需要恢复八个符号，候选含全部65,536消息，未利用真实等频组成。
每条件12篇与96计划载体保留；并列、拒绝和缺失均不算完整成功。
来源：[结果逐行与聚合](RESULT.json)、[导出数据](METRICS.csv)。

## 3. 注意力的独立增量

主要对照attention−gate的复述macro-F1差为−5.96个百分点，
95%描述性区间[−11.65, −0.73]，没有通过预定正增益门。
与宽门控相比为−3.38个百分点，区间[−9.49, 3.11]；
与等权相比为−2.33个百分点，区间[−7.67, 2.52]。
与原W相比仅+1.74个百分点，区间[−6.96, 11.80]，也未确认提升。
这些对照已预先声明；不得在看到结果后改用最弱基线建立优势。
来源：[配对对照](RESULT.json)中的`paired_contrasts`。

注意力复述输出187个候选、门控199个，候选总数253；
正确身份分别33/96和39/96。前端候选召回和擦除也影响这些分数，
不能把结果仅归为图证据融合。无拒绝阈值时，已匹配位置正确身份全分母
为attention42/96、gate44/96、宽门控46/96；该条件仍不提供真实位置给推理。
它是评价中的定位配对诊断，不等于部署的“已知位置准确率”。
来源：[逐场景结果](RESULT.json)的`located_no_reject_correct_full_denominator`。

| 条件 | 简单门控完整恢复 | 注意力完整恢复 | 注意力错误唯一消息 / 并列擦除 |
|---|---:|---:|---:|
| 原句 | 0/12 | 0/12 | 10 / 2 |
| 复述 | 0/12 | 0/12 | 11 / 1 |
| 拆句 | 0/12 | 0/12 | 10 / 2 |
| 合句 | 0/12 | 0/12 | 8 / 4 |

完整恢复未改善，不能将单符号局部增益、查询排序或唯一最优输出当作
可靠通信。错误唯一消息仍是失败，增加擦除不改善主分母。
结果支持将attention保留为候选/消融，不支持将它写为第二篇已证实的核心优势。

## 4. 工程核验与资源

五项真实缓存工程测试通过：缺失与全缺失、有限梯度、模板重排、禁止目标
输入、checkpoint与批次重放、独立DP核验由对应测试覆盖。
训练CPU FP32、单线程；84个模型约20.05秒，预测和全部候选解码约32.23秒，
主评价和全空间复算约{r['evaluation_seconds']:.2f}秒；不含历史AMR解析/匹配成本。
本轮新增provider、AMR、embedding调用均0，不消耗GPU。
来源：[readiness](READINESS.json)、[预测封存计时](PREDICTION_SEAL.json)、
[执行结果计时](RESULT.json)、[测试源码](../../tests/test_semantic_multiscale.py)。

主审计复算31,457,280个候选代价，最大DP误差0；
独立审计另查84组模型隔离及归一化、504个留出版本模型重放、
480个macro-F1/旧F1、380个区间。归一化最大数值差
{audit['max_normalizer_error']:.3g}，低于1e-6。
72原预测和全部原参考SHA不变。
来源：[主审计](AUDIT.json)、[独立审计](INDEPENDENT_AUDIT.json)。

原`RESULT.json`聚合字段`complete_correct`因同名聚合键保存为率区间，
不是整数计数；逐场景整数记录完整。为保留冻结结果SHA，
[NORMALIZED_SUMMARY.json](NORMALIZED_SUMMARY.json)和CSV独立从逐场景重建
完整恢复计数，全部为0；未覆写原结果或冻结源码。

## 5. 对第二篇方案的影响

第二篇保留“核心约束匹配＋多尺度融合＋精确序列解码”的接口，
但方法验证必须分别证明约束增量和attention对gate/宽gate的增量。
本轮旧结构字母表和6/12词范围不能代替新事实字母表的验证。
当前attention配置的开发增益判定为No-Go；简单门控可作后续主要学习对照，
也尚未建立可靠通道。后续独立确认仍按真人事实门、新输入冻结和四符号
全部256消息执行，不能按本批测试结果继续调参直到过门。

只有旧12场景，单seed且交叉验证训练集重叠；区间是描述性开发证据。
同样的数据此前已被研究，标签还可能包含写入/解析/模型参考错误。
结果既不证明所有注意力方法无效，也不证明AMR没有语义信息；
它只说明本次公开字母表、缓存视图和训练合同下，attention没有优于门控。
无独立人类负例、完整检测阈值或来源认证，FPR/TPR未评估。

复现按[冻结协议](PROTOCOL.md)与[运行源码](../../scripts/run_semantic_multiscale_readback.py)
分prepare、fit、predict、evaluate；封存目录拒绝覆盖。
原密封文本、逐候选预测及checkpoint保留在私有缓存，工作区仅交付
聚合结果、数值行、图、SHA、协议与源码。
"""
    (run.OUT / "REPORT.md").write_text(report)
    public = [p for p in run.OUT.iterdir() if p.is_file() and p.name != "SHA256SUMS.json"]
    run.atomic(run.OUT / "SHA256SUMS.json", {p.name: run.sha(p) for p in sorted(public)})
    print(json.dumps({"report": str(run.OUT / "REPORT.md"), "chart": str(run.OUT / "RESULTS.png"),
                      "rows": len(normalized)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
