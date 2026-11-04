"""Write concise, seed-inclusive summaries without fitting or selecting models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def aggregate(rows, field):
    values = [r[field] for r in rows if r.get(field) is not None]
    return {"mean": statistics.mean(values), "min": min(values), "max": max(values)} if values else None


def main(out):
    result = read(out / "RESULTS.json")
    audit = read(out / "FINAL_AUDIT.json")
    metric_audit = read(out / "METRIC_AUDIT.json")
    assert audit["passed"] and (out / "COMPLETE.json").exists()
    assert metric_audit["passed"]
    assert metric_audit["freeze_sha256"] == sha(out / "FREEZE.json")
    assert metric_audit["results_sha256"] == sha(out / "RESULTS.json")
    assert metric_audit["predictions_sha256"] == sha(out / "private/PREDICTIONS.json")
    summary = {
        "freeze_sha256": sha(out / "FREEZE.json"), "audit_passed": True,
        "metric_audit": metric_audit,
        "clean_gate": result["clean_gate"], "official_signal": result["official_signal"]["populations"],
        "feature_shapes": result["feature_shapes"], "conditions": {},
        "seed_summary": "Mean of metrics across all five fixed seeds; no prediction ensemble",
        "independent_human_test_fpr": None, "no_document_1_percent_fpr_claim": True,
        "no_original_stage1b_gate_rewrite": True}
    if (out / "EXECUTION_LINEAGE.json").exists():
        summary["execution_lineage"] = read(out / "EXECUTION_LINEAGE.json")
    if (out / "PREFILL_PLAN.json").exists():
        prefill = read(out / "PREFILL_COMPLETE.json")
        plan = read(out / "PREFILL_PLAN.json")
        assert prefill["freeze_sha256"] == summary["freeze_sha256"]
        assert prefill["source_sha256"] == plan["source_sha256"]
        workers = [read(p) for p in sorted((out / "private/prefill").glob("*.json"))]
        assert len(workers) == plan["workers"] * len(prefill["completed_phases"])
        assert all(r["freeze_sha256"] == summary["freeze_sha256"] and
                   len(r["fixed_sample_replays"]) == 3 and
                   all(s["exact_parse_sha"] for s in r["fixed_sample_replays"]) for r in workers)
        summary["parser_prefill"] = {
            **prefill, "fixed_worker_replays_verified": 3 * len(workers),
            "helper_fresh_parses": sum(r["fresh_parses"] for r in workers)}
        revision_path = out / "PREFILL_CAPACITY_REVISION.json"
        if revision_path.exists():
            revision = read(revision_path)
            assert revision["freeze_sha256"] == summary["freeze_sha256"]
            assert revision["revised_plan_sha256"] == sha(out / "PREFILL_PLAN.json")
            assert revision["prior_plan_sha256"] == sha(out / "PREFILL_PLAN_THREE_WORKER.json")
            assert revision["before_any_helper_started"]
            assert revision["helper_workers"] == plan["workers"]
            summary["prefill_capacity_revision"] = revision
    for condition, populations in result["conditions"].items():
        methods = populations["readout"]["methods"]
        rows = {}
        for mode in ("raw_s6", "s6", "mean", "gate", "attention"):
            for view in ("carrier", "blind"):
                selected = [v for k, v in methods.items()
                            if k.startswith(mode + "_") and k.endswith("/" + view)]
                if not selected:
                    continue
                row = {"fixed_models": len(selected), "auroc": aggregate(selected, "auroc"),
                       "paired_win": aggregate(selected, "paired_win"),
                       "positive_abstain": aggregate(selected, "positive_abstain"),
                       "negative_abstain": aggregate(selected, "negative_abstain")}
                working = [v["working_points"]["0.05"] for v in selected]
                row["exploratory_5_percent"] = {
                    "tpr": aggregate(working, "tpr"), "fpr": aggregate(working, "fpr"),
                    "statuses": sorted({v["status"] for v in working})}
                rows[mode + "/" + view] = row
        summary["conditions"][condition] = {"readout_pairs": populations["readout"]["pairs"],
                                             "methods": rows}
    changes = {}
    clean = summary["conditions"]["clean"]["methods"]
    for condition, group in summary["conditions"].items():
        if condition == "clean":
            continue
        changes[condition] = {}
        baseline = group["methods"]["raw_s6/blind"]["auroc"]["mean"]
        for mode in ("raw_s6", "s6", "mean", "gate", "attention"):
            key = mode + "/blind"
            row = group["methods"][key]
            changes[condition][mode] = {
                "auroc_change_from_clean": row["auroc"]["mean"] - clean[key]["auroc"]["mean"],
                "auroc_difference_from_attacked_raw_s6": row["auroc"]["mean"] - baseline,
                "paired_win_change_from_clean": row["paired_win"]["mean"] - clean[key]["paired_win"]["mean"],
                "positive_abstain": row["positive_abstain"],
                "negative_abstain": row["negative_abstain"]}
    summary["readout_blind_changes"] = changes
    write(out / "private/ARTIFACT_SHA256SUMS.json", {
        str(p.relative_to(out)): sha(p)
        for p in sorted((out / "private").rglob("*"))
        if p.is_file() and p.name != "ARTIFACT_SHA256SUMS.json"
        and "__pycache__" not in p.parts and p.suffix in (".json", ".jsonl", ".py", ".pt")})
    summary["private_artifact_manifest_sha256"] = sha(out / "private/ARTIFACT_SHA256SUMS.json")
    write(out / "SUMMARY.json", summary)
    signal = summary["official_signal"]
    gate = summary["clean_gate"]
    lines = ["# 新语料检测与冻结模型迁移：结论", "",
             "24 对原文全部保留，原句检测先于窗口特征；没有补样、重新生成或按分数筛选。", "",
             f"开发信号门：**{'Go' if gate['passed'] else 'No-Go'}**。"
             f"6 对决策文档的原句载体 AUROC={gate['carrier_auroc']:.3f}，"
             f"原句均分的文档配对胜率={gate['document_paired_win']:.3f}。", "",
             f"独立读出 12 对原文：原句载体 AUROC={signal['readout']['carrier_auroc']:.3f}，"
             f"文档 AUROC={signal['readout']['document_auroc']:.3f}，"
             f"文档配对胜率={signal['readout']['document_paired_win']:.3f}。"
             "读出结果未用于信号门、模型或阈值选择。", "",
             f"新算 clean 载体张量为 `{summary['feature_shapes']['clean_carrier']}`，"
             f"盲窗口张量为 `{summary['feature_shapes']['blind_all_conditions']}`。", ""]
    if not gate["passed"]:
        lines += ["按冻结协议停止攻击阶段。当前生成通道尚未提供稳定的原句检测信号，"
                  "不能将任何窗口或融合结果解释为攻击后的水印恢复。"
                  "此前 0/24 是完整图覆盖率；本轮判断使用真实 SWAN bank-max 匹配分数。", ""]
    else:
        lines += ["原句信号通过开发门后，按冻结配置完成复述与句界攻击。"
                  "攻击评分只使用待测文本提取窗口；已知原句位置仅用于 clean 载体诊断。", ""]
    labels = {"raw_s6": "原始 S6", "s6": "归一化 S6", "mean": "等权融合",
              "gate": "门控（5 种子均值）", "attention": "注意力（5 种子均值）"}
    for condition, group in summary["conditions"].items():
        lines += [f"## {condition}：独立读出", "",
                  "| 方法 | 盲文档 AUROC | 盲文档配对胜率 | 载体 AUROC | 载体 TPR / FPR（探索性5%） |",
                  "|---|---:|---:|---:|---:|"]
        for mode, label in labels.items():
            blind = group["methods"].get(mode + "/blind")
            carrier = group["methods"].get(mode + "/carrier")
            def val(row, key):
                return f"{row[key]['mean']:.3f}" if row and row.get(key) else "—"
            wp = carrier["exploratory_5_percent"] if carrier else None
            lines.append(f"| {label} | {val(blind, 'auroc')} | {val(blind, 'paired_win')} | "
                         f"{val(carrier, 'auroc')} | {val(wp, 'tpr')} / {val(wp, 'fpr')} |")
        lines.append("")
    if changes:
        lines += ["## clean 到扰动后的盲文档变化", "",
                  "下表为同一批 12 对读出文档的 AUROC 变化。Δclean 表示扰动减去 clean；"
                  "ΔS6 表示该条件下融合减去原始 S6。正值只表示排序差异，不代表达到某个文档误报率。", "",
                  "| 条件 | S6 Δclean | 等权 Δclean / ΔS6 | 门控 Δclean / ΔS6 | 注意力 Δclean / ΔS6 |",
                  "|---|---:|---:|---:|---:|"]
        for condition, rows in changes.items():
            values = []
            for mode in ("mean", "gate", "attention"):
                row = rows[mode]
                values.append(f"{row['auroc_change_from_clean']:+.3f} / "
                              f"{row['auroc_difference_from_attacked_raw_s6']:+.3f}")
            lines.append(f"| {condition} | {rows['raw_s6']['auroc_change_from_clean']:+.3f} | "
                         + " | ".join(values) + " |")
        lines.append("")
    lines += ["## 解释范围", "",
              "门控和注意力报告全部固定种子的指标均值，范围与逐种子值见 SUMMARY.json 和 REPORT.md；"
              "没有选择最好种子，也没有组成预测集成。旧权重和归一化器逐文件 SHA 核对一致。", "",
              "载体工作点仅用 6 篇校准负文档的 48 个相关原句。1% 载体校准不足，"
              "文档 1%/5% 校准均不足；没有独立人类文本 FPR。"
              "正负文档平均约 115/141 词，长度与生成风格可能影响分数。", "",
              "旧缓存的模板列顺序未保存独立身份凭据；当前产生缓存的代码按同一个 100 模板 JSON 顺序输出，"
              "本轮沿用该顺序，历史列身份仍属于溯源限制。"
              "新评分保留完整尾窗和浮点分数，修复事件 BFS 的确定顺序与悬空边，"
              "S2Match 使用图对 SHA 随机流、保留 5 次搜索。"
              "这些输入与实现差异限制了对旧模型泛化失败原因的归因。", "",
              "所有特征均按与锚重叠的候选窗口逐模板取最大值，再应用冻结模型；"
              "盲文档使用 S12 锚，载体诊断使用已知原句锚。"
              "盲文档的阈值没有从载体阈值迁移。"
              "私有文本、特征、逐行记录与模型的 SHA 清单保留于 private/ARTIFACT_SHA256SUMS.json。", "",
              f"最终审计通过：{audit['records']} 个配对条件记录，"
              f"{audit['candidate_score_cells_checked']:,} 个候选分数单元，"
              f"模型输出最大重算误差 {audit['max_prediction_error']:.2g}；"
              "固定样本 AMR 与全库分数重放通过。"
              f"无效解析 {audit['invalid_parse_spans']}/{audit['unique_parses']}，"
              "完整分母保留，未删除失败正例。", "",
              f"另以秩统计独立复核 {metric_audit['metrics_checked']} 组 AUROC/配对胜率和 "
              f"{metric_audit['working_points_checked']} 个校准工作点；"
              f"{metric_audit['expected_unique_parse_inputs']:,} 个预定解析键均存在，"
              "全部缓存的 AMR SHA 与内容一致。", "",
              "[冻结协议](PROTOCOL.md) · [完整结果](REPORT.md) · [最终审计](FINAL_AUDIT.json) · "
              "[指标复核](METRIC_AUDIT.json) · [SHA 清单](SHA256SUMS.json)", ""]
    if "execution_lineage" in summary:
        lineage = summary["execution_lineage"]
        lines += ["## 执行溯源", "",
                  f"上一轮 {lineage['predecessor_run']} 保留 STOP：逐条件累计工作量的保守预测为 "
                  f"{lineage['predecessor_resource_projection_hours']:.3f} 小时，超过 12 小时门。"
                  f"本轮以 {lineage['successor_feature_unique_inputs']:,} 个原句阶段后唯一文本键估计资源，"
                  "沿用同一文本、模型和匹配算法；从独立缓存重新运行，旧消耗计入预算。", ""]
    if "parser_prefill" in summary:
        prefill = summary["parser_prefill"]
        phases = ", ".join(prefill["completed_phases"]) or "未启用"
        peak = (f"全 GPU 峰值显存占比 {prefill['peak_global_gpu_memory_fraction']:.1%}。"
                if prefill["fixed_worker_replays_verified"] else
                "额外解析进程未启动，没有并发重放或峰值显存观测值。")
        lines += [f"解析缓存的实际启用阶段：{phases}；并发上限 {prefill['workers']} 个额外进程，"
                  "使用 FP32、batch1、4 beams。"
                  f"{prefill['fixed_worker_replays_verified']} 次固定样本的 AMR SHA 精确重放通过。"
                  f"{peak}"
                  "它们只填充本轮解析缓存，匹配和模型评分仍由冻结流程执行。", ""]
        if "prefill_capacity_revision" in summary:
            revision = summary["prefill_capacity_revision"]
            lines += [
                "三路预填在其他 GPU 任务占用下未能启动，随后在任一额外进程启动前修订为一路。"
                f"单进程预留 {revision['reservation_mib']:,} MiB，依据预检测得的"
                f" {revision['observed_peak_reserved_mib']:,.1f} MiB 峰值加 25% 并向上取整；"
                f"另保留 {revision['additional_headroom_mib']} MiB 余量，全 GPU 80% 门限不变。"
                "旧计划及修订 SHA 均保留，见 PREFILL_CAPACITY_REVISION.json。", ""]
    (out / "OVERVIEW.md").write_text("\n".join(lines))
    write(out / "SHA256SUMS.json", {
        p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()
        and p.name not in ("SHA256SUMS.json", "heartbeat.json") and not p.name.endswith(".log")})
    print(json.dumps({"summary_written": True, "clean_gate": gate["passed"],
                      "readout_pairs": signal["readout"]["pairs"]}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    main(ap.parse_args().out.resolve())
