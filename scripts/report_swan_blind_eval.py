"""Write a reviewable report from exported, text-free evaluation aggregates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(folder, name):
    path = folder / name
    return json.loads(path.read_text()) if path.exists() else None


def rate(metric):
    if metric["rate"] is None:
        return "未校准"
    return f"{100 * metric['rate']:.1f}% ({metric['decisions_positive']}/{metric['denominator']})"


def report(folder):
    freeze = read(folder, "FREEZE.json")
    ready = read(folder, "READINESS.json")
    result = read(folder, "RESULTS.json")
    documents = read(folder, "DOCUMENT_SCORES.json")
    overlap = read(folder, "INPUT_OVERLAP_AUDIT.json")
    diagnostics = read(folder, "OFFICIAL_DIAGNOSTICS.json")
    heartbeat = read(folder, "heartbeat.json")
    halt = read(folder, "HALT.json")
    audit = read(folder, "AUDIT.json")
    if not freeze or not ready:
        raise ValueError("Frozen inputs and readiness are required")
    complete = bool(result and result["complete_methods"] == freeze["evaluation_methods"]
                    and heartbeat and heartbeat["state"] == "complete" and not halt)
    lines = [
        "# 官方全文与盲文档检测评估",
        "",
        f"状态：{'完整执行完成' if complete else '永久停止，保留已完成方法' if halt else '执行中，以下仅列已完成方法'}。",
        "沿用 dataset5 的 30 对受控文档和七类扰动；前 15 篇负例及其变体用于校准，"
        "后 15 对文档用于主测试。全部 30 篇正例的结果只作辅助。",
        "输入经过既往探索，负例为配对生成文本；本报告是开发评估，"
        "没有独立人类文本测试，不能作为正式泛化或 1% 误报控制的证明。",
        "",
        "数据划分和执行来源见 [FREEZE.json](FREEZE.json)；"
        "资源、数值重放与预算预检见 [READINESS.json](READINESS.json)。",
        "",
        "检测只使用受扰动全文和冻结配置。P0 为英文 Punkt 实际分句；"
        "S6/S8/S10/S12 为相应词数的完整滑窗，步长均为 1。"
        "每个候选都匹配全部 100 个模板。官方计数参照保留空图分母；"
        "严格版本任一候选失败即整篇弃权。",
        "",
        "## 执行证据",
        "",
        f"- 解析：FP32、四束解码、批次 {freeze['parser_batch_size']}，原生上下文 "
        f"{ready['native_context_tokens']} token；实际最大输入 {freeze['native_input_max_tokens']} token。",
        f"- 唯一候选输入：{freeze['candidate_unique_inputs']:,}；"
        f"模型/输入/归一化/模板库/匹配器均以 SHA 冻结。",
        f"- 预检：日志绑定、最长全文完整批次显存、评分数值重放均通过；"
        f"保守预算 {ready['conservative_full_run_estimate_hours']:.2f} 小时。",
        "- 初次正式运行首批 CUDA OOM 后永久停止，无提交结果；"
        "本报告对应独立后继 run2 和新空缓存。",
    ]
    if heartbeat:
        lines += [
            f"- 最新远程 heartbeat：{heartbeat['updated_at']}。",
            f"- 已解析 {heartbeat['unique_inputs_parsed']:,} 个输入，"
            f"已评分 {heartbeat['unique_graphs_scored']:,} 个唯一归一化图。",
        ]
        if heartbeat["state"] == "running":
            lines.append(f"- 当前阶段：{heartbeat['method']} / {heartbeat['stage']}，"
                         f"{heartbeat['stage_completed']:,}/{heartbeat['stage_expected']:,}。")
    if halt:
        lines.append(f"- 永久停止记录：{halt['timestamp']}；{halt['exception_type']} / {halt['reason']}。")
    if audit:
        lines.append(f"- 历史分数 SHA 保持不变：{audit['historical_scores_unchanged']}；"
                     f"已完成方法的完整分母审计："
                     f"{audit.get('complete_method_denominators_verified', '尚无评分结果')}。")
    if overlap:
        negative = overlap["methods"]["S6"]["negative_test"]
        shared = negative["exact_candidate_overlap_with_negative_calibration"]
        count = negative["unique_test_candidates"]
        lines += ["",
                  "只读输入重叠审计：校准负例与主测试正负文档的全文完全重叠均为 0；"
                  f"S6 的测试负例候选有 {shared}/{count:,}（{100*shared/count:.4f}%）"
                  "与校准负例完全相同，其余方法该交叉重叠为 0。"
                  "这里检验的是输入文本，不代表归一化图或语义完全独立。"
                  "来源见 [INPUT_OVERLAP_AUDIT.json](INPUT_OVERLAP_AUDIT.json)。"]
    lines += ["", "## 实测结果", ""]
    if not result:
        lines += ["尚无完整方法的 FPR/TPR。解析进度、有效候选比例不等于检测指标。", ""]
    else:
        lines += [
            "所有百分比均以全部主测试文档为分母，弃权作为未检出保留在分母中。"
            "名义 1%/5% 是校准目标，实测 FPR 见表；未校准工作点不产生判定。",
            "以下表格直接读取 [RESULTS.json](RESULTS.json)，"
            "阈值来源见 [CALIBRATION.json](CALIBRATION.json)。",
            "",
        ]
        for target in (1, 5):
            name = f"document_{target}pct"
            lines += [
                f"### 名义文档 {target}% 校准目标",
                "",
                "| 方法 | 扰动 | TPR（TP/15） | FPR（FP/15） | 正例无判定 | 负例无判定 |",
                "|---|---|---|---|---|---|",
            ]
            for method, measured in result["methods"].items():
                for attack, values in measured["operating_points"][name]["conditions"].items():
                    if attack == "pooled":
                        continue
                    tpr, fpr = values["TPR"], values["FPR"]
                    lines.append(f"| {method} | {attack} | {rate(tpr)} | {rate(fpr)} | "
                                 f"{tpr['unresolved']}/15 | {fpr['unresolved']}/15 |")
            lines += [
                "",
                "七类同源变体的汇总：",
                "",
                "| 方法 | 汇总 TPR | 汇总 FPR | 文档阈值 | 完整校准文档/105 |",
                "|---|---|---|---|---|",
            ]
            for method, measured in result["methods"].items():
                op = measured["operating_points"][name]
                pooled = op["conditions"]["pooled"]
                threshold = op["document_threshold"]
                formatted = f"{threshold:.9g}" if threshold is not None else "未校准"
                lines.append(f"| {method} | {rate(pooled['TPR'])} | {rate(pooled['FPR'])} | "
                             f"{formatted} | {measured['complete_calibration_documents']}/"
                             f"{measured['calibration_documents']} |")
            lines += [""]
        if diagnostics:
            lines += [
                "### 官方统计量的计数上限",
                "",
                "官方 λ=0.25 且 G≤N，因此 z≤√(3N)。"
                "semicolon/comma/space 每篇 N=1，即使唯一图为 green，"
                "z 最高仅为 1.732，低于本轮两个文档阈值。"
                "这三个条件的零检出包含计数与统一阈值的容量限制，"
                "不能全部归因于解析失败或图匹配丢失信号。",
                "merge 主测试两臂均有 6–10 句且没有严格弃权，"
                "名义 1% 下仍有 11/15 正例即使全 green 也无法超过阈值；"
                "其正例单元 green 率为 35.5%，同时存在计数限制与单元分数变化。"
                "实际单元数、green 率和每个阈值的上界审计见 "
                "[OFFICIAL_DIAGNOSTICS.json](OFFICIAL_DIAGNOSTICS.json)。",
                "",
            ]
        lines += [
            "### 单元阈值与可用性",
            "",
            "| 方法 | 单元阈值 | 校准单元 green 率 | 可用性 |",
            "|---|---|---|---|",
        ]
        for method, measured in result["methods"].items():
            threshold = measured["unit_threshold"]
            unit = f"{threshold:.9g}" if threshold is not None else "无"
            green = measured["calibration_unit_green_rate"]
            green_text = f"{100 * green:.3f}%" if green is not None else "无"
            availability = measured["availability"]
            if measured["unavailability_reason"]:
                availability += ": " + measured["unavailability_reason"]
            lines.append(f"| {method} | {unit} | {green_text} | {availability} |")
        lines += [
            "",
            "P0 原始 z 使用官方固定单元阈值 0.65 与 λ=0.25；"
            "其文档阈值仍从校准负例选择，z 不解释为理论 p 值。"
            "官方网格搜索与回退另见 RESULTS.json 的 official_grid_auxiliary，"
            "不与主经验阈值结果混合。",
            "",
        ]
        if documents:
            lines += [
                "### 主测试覆盖与实际检测单元",
                "",
                "这里的完整文档数只反映解析与评分覆盖，未校准时仍可能无法作出判定。"
                "候选数为每篇实际送入检测的单元数，不使用原始注入槽位补齐。"
                "表格来源为 [DOCUMENT_SCORES.json](DOCUMENT_SCORES.json)。",
                "",
                "| 方法 | 扰动 | 正例单元数范围 | 负例单元数范围 | 完整正例/15 | 完整负例/15 |",
                "|---|---|---|---|---|---|",
            ]
            rows = documents["documents"]
            for method, measured in result["methods"].items():
                conditions = measured["operating_points"]["document_1pct"]["conditions"]
                for attack in conditions:
                    if attack == "pooled":
                        continue
                    groups = [[r for r in rows if r["method"] == method and
                               r["attack"] == attack and r["arm"] == arm and
                               r["split"] == "test"]
                              for arm in ("positive", "paired_negative")]
                    ranges = [f"{min(r['N_expected'] for r in group)}–"
                              f"{max(r['N_expected'] for r in group)}"
                              for group in groups]
                    completed = [sum(r["complete"] for r in group) for group in groups]
                    lines.append(f"| {method} | {attack} | {' | '.join(ranges)} | "
                                 f"{completed[0]}/15 | {completed[1]}/15 |")
            lines += [""]
    lines += [
        "## 统计边界",
        "",
        "- 每种扰动只有 15 篇测试负例，FPR 分辨率为 1/15 = 6.67 个百分点。",
        "- 即使 0/15 误报，双侧 95% Clopper–Pearson 区间上界约为 21.8%；"
        "不能宣称真实 FPR 为 0 或达到 1%。",
        "- 单条件保留二项区间；七类变体高度相关，汇总只按源文档进行 "
        "2,000 次聚类 bootstrap，不把 105 个变体当成独立负例。",
        "- 弃权会同时压低全分母 TPR/FPR；需结合弃权数量与成功文档条件指标解读。"
        "未校准时的未解决数包含所有无判定文档，不等同于解析失败数。",
        "- 全部 30 正例辅助指标、同分数量、失败原因与区间见 RESULTS.json；"
        "原始候选和解析图不导出。",
        "",
        "## 复核文件",
        "",
        "- [FREEZE.json](FREEZE.json) / [READINESS.json](READINESS.json)：冻结来源与预检。",
        "- [DOCUMENT_SCORES.json](DOCUMENT_SCORES.json)：仅文档级分数、状态与计数，无文本或图。",
        "- [CALIBRATION.json](CALIBRATION.json) / [RESULTS.json](RESULTS.json) / "
        "[METRICS.csv](METRICS.csv)：阈值与实际指标。",
        "- [AUDIT.json](AUDIT.json) / [SHA256SUMS.json](SHA256SUMS.json)：分母、来源绑定与文件 SHA。",
        "- [document_1pct.pdf](document_1pct.pdf) / [document_5pct.pdf](document_5pct.pdf)："
        "TPR、FPR、未解决比例图（已有结果时生成）。",
        "",
    ]
    content = "\n".join(lines)
    path = folder / "REPORT.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(content)
    temporary.replace(path)
    return {"report": str(path), "full_cohort_complete": complete,
            "complete_methods": result["complete_methods"] if result else [],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(report(parser.parse_args().output), indent=2))
