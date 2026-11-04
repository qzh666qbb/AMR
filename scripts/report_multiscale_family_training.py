"""Build the development report exclusively from audited aggregate results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_multiscale_family_training import DEFAULT

LABELS = {
    "U0": "原门控", "U1": "原门控＋行hard", "U2": "原门控＋分族hard",
    "U3": "原门控＋分族＋所选损失", "E1": "可信度门控＋行hard",
    "E2": "可信度门控＋分族", "E3": "可信度门控＋分族＋所选损失",
    "X1": "交叉注意力＋行hard", "X2": "交叉注意力＋分族",
    "X3": "交叉注意力＋分族＋所选损失",
}


def percent(x):
    return f"{100 * x:.2f}%"


def delta(row):
    mean = row["micro_delta"]["mean"] * 100
    lo, hi = np.array(row["mean_micro_delta_ci95"]) * 100
    return f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.input
    if (out / "REPORT.md").exists():
        raise FileExistsError("REPORT.md")
    summary = json.loads((out / "SUMMARY.json").read_text())
    protocol = json.loads((out / "RUN_PROTOCOL.json").read_text())
    checks = json.loads((out / "ANALYSIS_CHECKS.json").read_text())
    mining = json.loads((out / "MINING_DIAGNOSTICS.json").read_text())
    selection = json.loads((out / "SELECTION.json").read_text())
    lines = [
        "# 载体族高分负例训练开发验证",
        "",
        "日期：2026-10-23。版本：swan-family-hard-development-v1。",
        "",
        "## 1. 结果与证据范围",
        "",
        "已完成按族挖掘、最坏格式损失、原2维门控＋负例训练对照，以及兼顾最差条件FPR的模型选择。"
        "训练80条轨迹，先用训练/验证固定系数和epoch，再对10阶段、5种子、2种选择器的100个冻结端点评估。"
        "所有均值均为5个独立模型结果的平均，不构成集成。[协议](PROTOCOL.md)、"
        "[系数选择](SELECTION.json)、[聚合结果](SUMMARY.json)保留全部决策与逐种子结果。",
        "",
        "本轮主工作点为5%经验校准目标，另报告1%。数据是已多次查看的30篇源文档、oracle载体槽位缓存，"
        "不能解释为新的独立确认实验。独立human-test FPR、盲文档TPR/FPR均未测量。所有统计结论均限于本开发分区。",
        "",
        "按族挖掘使用每篇原行级预算，每族最高分格式只占一个hard名额；其余格式仍参加基础训练。"
        "旧按行策略含截止分数并列，新策略采用确定名额，因此报告实际名额扩张差异。"
        "新损失系数零值可选，零值胜出意味着训练/验证选择未接受额外惩罚，不能据此宣称新损失收益。",
        "",
    ]
    for left, right, description in (
        ("U2", "U1", "原门控按族挖掘"),
        ("E2", "E1", "可信度门控按族挖掘"),
        ("X2", "X1", "交叉注意力按族挖掘"),
    ):
        point = summary["comparisons"][f"{left}_guarded_minus_{right}_guarded"]["0.05"]
        lines.append(f"{description}的整体检出差为{delta(point['all']['tpr'])}个百分点，"
                     f"split假阳差为{delta(point['split']['fpr'])}个百分点。")
    lines += ["", "上述三项比较在同一模型族、同一epoch选择规则下隔离分族挖掘，"
              "不会将特征包或选择规则收益混入分族增益。"]
    selected = ", ".join(f"{stage}={protocol['coefficients'][choice['selected']]:.2f}"
                         for stage, choice in selection["coefficient_choices"].items())
    lines += [f"训练/验证选择的新损失系数为{selected}。", ""]
    passed = sum(row["engineering_rule"]["passed"] for row in summary["comparisons"].values())
    lines += [f"本轮{len(summary['comparisons'])}项预定训练/选择比较中，"
              f"{passed}项通过“降低split且保住检出”的严格开发判据。"
              "因此当前配置尚未建立符合目标的稳定收益，不能将去重成功等同于检测改善。",
              ""]
    base_gain = summary["comparisons"]["U1_bce_minus_U0_bce"]["0.05"]
    feature_gain = summary["comparisons"]["E1_bce_minus_U1_bce"]["0.05"]
    attention_family = summary["comparisons"]["X2_bce_minus_X1_bce"]["0.05"]
    lines += [
        f"补齐的原门控＋行hard对照，整体检出相对原门控提高{delta(base_gain['all']['tpr'])}个百分点，"
        f"同时整体假阳提高{delta(base_gain['all']['fpr'])}个百分点。"
        f"在相同行hard上再加可信度特征，检出差为{delta(feature_gain['all']['tpr'])}个百分点。"
        "这支持将训练权重与特征收益分别归档，不能把累计检出提升全部归因于特征。",
        "",
        f"注意力在原BCE选择下按族挖掘，split假阳差为{delta(attention_family['split']['fpr'])}个百分点，"
        f"检出差为{delta(attention_family['all']['tpr'])}个百分点。"
        "此处split改善区间不跨零，但检出下降超过预设容限；"
        "guarded选择下的相同训练改动又出现split上升，说明选择规则收益没有稳定迁移到保留区。",
        "",
    ]
    for selector, title in (("guarded", "兼顾最差条件FPR的epoch选择"),
                            ("bce", "原BCE epoch选择")):
        lines += [f"## {2 if selector == 'guarded' else 3}. 5%目标：{title}", "",
                  "| 阶段 | 检出率 | 整体FPR | split FPR | 最差条件FPR |",
                  "|---|---:|---:|---:|---:|"]
        for stage in protocol["stages"]:
            row = summary["methods"][f"{stage}_{selector}"]["working_points"]["0.05"]
            lines.append(f"| {stage} {LABELS[stage]} | "
                         f"{percent(row['all']['tpr']['micro']['mean'])} | "
                         f"{percent(row['all']['fpr']['micro']['mean'])} | "
                         f"{percent(row['split']['fpr']['micro']['mean'])} | "
                         f"{percent(row['worst_condition']['fpr']['mean'])} |")
        lines += ["", "最差条件FPR先对每个种子取七条件最大值，再对5种子求均值。"
                  "通过5%校准目标不代表每个测试条件均≤5%；逐种子最大值见SUMMARY.json。", ""]
    lines += ["## 4. 系数选择与正例保护", "",
              "每条轨迹的guarded候选相对该轨迹BCE参考，验证TPR最多下降2个百分点、正例BCE最多增加0.05。"
              "跨系数选择相对零损失分族对照，采用相同的种子平均保护边界。"
              "临时阈值只来自训练负例，是训练内代理工作点；最终校准区没有参与选择。",
              "",
              "| 新损失阶段 | 候选系数 | 验证TPR | 验证正例BCE | 平均最差FPR | 最差种子FPR | 选择 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for stage, choice in selection["coefficient_choices"].items():
        for coefficient, row in choice["candidates"].items():
            label = "选中" if coefficient == choice["selected"] else (
                "未过正例保护" if coefficient not in choice["eligible"] else "未选中")
            lines.append(f"| {stage} | {protocol['coefficients'][coefficient]:.2f} | "
                         f"{percent(row['mean_validation_tpr'])} | "
                         f"{row['mean_validation_positive_bce']:.4f} | "
                         f"{percent(row['mean_validation_worst_fpr'])} | "
                         f"{percent(row['max_seed_validation_worst_fpr'])} | {label} |")
    lines += [""]
    for stage, choice in selection["coefficient_choices"].items():
        lines.append(f"- {stage}所选系数为{protocol['coefficients'][choice['selected']]:.2f}，"
                     f"选择状态为`{choice['status']}`。")
    lines += ["", "全部种子可行规则未满足时，选择器仍保留最低条件风险的受保护候选，并明确记为不可行；"
              "不能把“选中”解读为5%条件约束过门。"
              "对非零损失，基础BCE与额外族损失使用同一次优化步，额外前向的计算成本应单列。",
              "",
              "## 5. 收益来源与配对区间", "",
              "下表是guarded选择器、5%目标下的逐步差。数值为百分点，括号为95%源文档配对bootstrap区间。"
              "全部区间未作多重比较校正，条件于现有模型、阈值和种子，不包含重训练/校准不确定性。",
              "",
              "| 比较 | 整体TPR差 [95%区间] | 整体FPR差 [95%区间] | split FPR差 [95%区间] | 严格开发判据 |",
              "|---|---:|---:|---:|---|"]
    for a, b in protocol["comparisons"]:
        row = summary["comparisons"][f"{a}_guarded_minus_{b}_guarded"]
        point = row["0.05"]
        lines.append(f"| {a}−{b} | {delta(point['all']['tpr'])} | "
                     f"{delta(point['all']['fpr'])} | {delta(point['split']['fpr'])} | "
                     f"{'通过' if row['engineering_rule']['passed'] else '未通过'} |")
    lines += ["", "严格开发判据同时要求：split FPR差区间上界<0；整体TPR差区间下界≥−2个百分点；"
              "全部种子split不升；新增方法每个种子最差条件FPR≤5%。"
              "X3−E3若使用不同系数，是各自验证优化后的整套配置比较，不能归因于纯注意力算子。",
              "",
              "## 6. 模型选择规则的独立影响", "",
              "同一训练轨迹和系数，仅改变选中epoch；两个模型分别使用同一算法在最终校准区重新定阈值。",
              "",
              "| 阶段 | guarded−BCE TPR差 [95%区间] | split FPR差 [95%区间] | guarded epoch（5种子） | BCE epoch（5种子） |",
              "|---|---:|---:|---|---|"]
    for stage in protocol["stages"]:
        row = summary["comparisons"][f"{stage}_guarded_minus_{stage}_bce"]["0.05"]
        epochs = {selector: [summary["runs"][f"{stage}_{selector}_{seed}"]["epoch"]
                             for seed in protocol["seeds"]]
                  for selector in ("guarded", "bce")}
        lines.append(f"| {stage} | {delta(row['all']['tpr'])} | {delta(row['split']['fpr'])} | "
                     f"{','.join(map(str, epochs['guarded']))} | {','.join(map(str, epochs['bce']))} |")
    lines += ["", "## 7. 挖掘质量审计", "",
              "以下聚合各阶段所选配置、全部5个种子的第26–30轮。额外重复格式占比="
              "1−选中不同载体族数/选中行数。split训练质量占比仅指加权BCE的负类质量，"
              "不把额外max-format梯度强行换算成样本权重。",
              "",
              "| 阶段 | hard行数 | 不同族数 | 额外重复格式占比 | split占hard行 | split占负训练质量 | 旧并列额外名额 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for stage, choice in selection["evaluation_methods"].items():
        if stage == "U0":
            continue
        rows = [mining[f"{choice['source_stage']}_{choice['coefficient']}_{seed}"]["last5"]
                for seed in protocol["seeds"]]
        mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        lines.append(f"| {stage} | {mean['selected_rows']:.1f} | {mean['distinct_families']:.1f} | "
                     f"{percent(mean['duplicate_extra_format_fraction'])} | "
                     f"{percent(mean['split_selected_fraction'])} | "
                     f"{percent(mean['split_negative_training_mass_fraction'])} | "
                     f"{mean['row_tie_budget_expansion']:.1f} |")
    lines += ["", "基础负类七格式等权时，split质量占14.29%。所有未选格式权重仍为正，"
              "正例权重逐项保持，负类每篇原总质量保持在1e-7绝对误差内。",
              "",
              "## 8. 条件检出与辅助1%工作点", "",
              "下表为guarded选择器5%目标。每个条件正负各236槽位，不能当作236篇独立文档。",
              "",
              "| 阶段 | 条件 | 检出率 | 假阳率均值 | 假阳率种子范围 |",
              "|---|---|---:|---:|---:|"]
    for stage in protocol["stages"]:
        points = summary["methods"][f"{stage}_guarded"]["working_points"]["0.05"]
        for attack, row in points.items():
            if attack in ("all", "worst_condition"):
                continue
            fpr = row["fpr"]["micro"]
            lines.append(f"| {stage} | {attack} | {percent(row['tpr']['micro']['mean'])} | "
                         f"{percent(fpr['mean'])} | {percent(fpr['min'])}–{percent(fpr['max'])} |")
    lines += ["", "1%目标仅辅助报告，未用于本轮系数优化。每个最终校准条件只有62槽位、4篇文档，"
              "因此本轮仍采用统一pooled阈值，不生成单条件1%分位保证。",
              "",
              "| 阶段（guarded） | 检出率 | 整体FPR | split FPR | 最差条件FPR |",
              "|---|---:|---:|---:|---:|"]
    for stage in protocol["stages"]:
        row = summary["methods"][f"{stage}_guarded"]["working_points"]["0.01"]
        lines.append(f"| {stage} | {percent(row['all']['tpr']['micro']['mean'])} | "
                     f"{percent(row['all']['fpr']['micro']['mean'])} | "
                     f"{percent(row['split']['fpr']['micro']['mean'])} | "
                     f"{percent(row['worst_condition']['fpr']['mean'])} |")
    execution = summary["execution"]
    lines += ["", "## 9. 工程复核与复现", "",
              f"80次训练总用时{execution['elapsed_seconds']:.1f}秒，"
              f"最大子进程峰值RSS为{execution['max_child_rss_bytes'] / 1024**2:.1f} MiB，"
              "使用CPU两线程。该时间包括缓存加载与融合训练，不含历史AMR/S2Match生成成本；"
              "新增上游调用均为0。",
              "",
              f"已独立核验{checks['negative_document_epoch_groups_verified']}个训练文档×epoch组合、"
              f"{checks['validation_epoch_metrics_recomputed']}个验证epoch，"
              f"重放{checks['endpoints_checkpoint_replayed']}个冻结端点，"
              f"复算{checks['independent_thresholds_recomputed']}个阈值、"
              f"{checks['independent_pooled_and_condition_rates_recomputed']}个整体/条件率，"
              f"并以直接正负对比较复核{checks['direct_pair_auroc_verified']}个AUROC。"
              f"旧A0/A2/A3共{checks['old_baseline_exact_replay_runs']}条BCE轨迹逐位一致。"
              "输入和冻结源码SHA全部通过。见[工程检查](ENGINEERING_CHECKS.json)、"
              "[分析检查](ANALYSIS_CHECKS.json)、[挖掘审计](MINING_DIAGNOSTICS.json)。",
              "",
              "复现使用新输出目录，复制RUN_PROTOCOL.json和PROTOCOL.md，先保持输入和源码SHA一致。"
              "执行顺序为train、select、evaluate、analysis、report；已存在输出一律拒绝覆盖。",
              "",
              "```bash",
              ".venv/bin/python scripts/run_multiscale_family_training.py --phase train",
              ".venv/bin/python scripts/run_multiscale_family_training.py --phase select",
              ".venv/bin/python scripts/run_multiscale_family_training.py --phase evaluate",
              ".venv/bin/python scripts/analyze_multiscale_family_training.py",
              ".venv/bin/python scripts/report_multiscale_family_training.py",
              "```",
              "",
              "上述默认目录是本轮已完成的归档，复现时各命令须传入--output或--input指向同一个新目录。"
              "完整模型、逐轮trace、执行日志、系数选择SHA和输入元数据都保留在本目录；"
              "未导出源/生成文本。固定种子不承诺跨硬件、跨PyTorch版本逐位一致。",
              "",
              "## 10. 限制与后续边界", "",
              "本轮可用于判断去重、负例损失、可信度特征和选择规则在现有缓存上的收益与取舍，"
              "不能证明语义质量改善、AMR机制因果关系或正式检测鲁棒性。"
              "训练内临时阈值与最终校准阈值来自不同分区，验证过门不保证测试最差条件过门；"
              "两个分区结果同时保留，不能只展示较低的一侧。",
              "",
              "当前保持行预算后，每轮91个族被选作hard，训练区共有125族，覆盖率约72.8%。"
              "这个设计隔离了同一行级名额下的去重变化，但不等于10%的族级预算。"
              "本轮负结果只约束这组预算、系数网格和选择代理；更小的族预算、其他损失形式未测试。"
              "验证每条件48槽位，FPR分辨率为约2.08个百分点，且只来自3篇文档，"
              "当前候选间的相同验证FPR不能证明其真实尾部风险相同。",
              "",
              "当前只归档这些开发候选，不按测试结果追加系数、改阈值、挑种子或补样。"
              "若未来改变实际检测流程，必须整体重新校准，并另测独立human-test FPR。",
              ""]
    with (out / "REPORT.md").open("x") as stream:
        stream.write("\n".join(lines))
    print(f"report={out / 'REPORT.md'} lines={len(lines)}")


if __name__ == "__main__":
    main()
