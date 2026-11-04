"""Write the reviewable report for an existing-corpus symbol/order diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def pct(value):
    return f"{100 * value:.2f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    identity = read(out / "IDENTITY_RESULT.json")
    order = read(out / "ORDER_RESULT.json")
    paired = read(out / "ORDER_PAIRED_READOUT.json")
    selfread = read(out / "TEMPLATE_SELF_READ.json")
    coverage = read(out / "PARAPHRASE_COVERAGE.json")
    layers = read(out / "LAYER_DIAGNOSIS.json")
    validation = read(out / "VALIDATION.json")
    orig = identity["readout"]["AMR_original_carrier"]
    soft = order["readout"]["AMR_original_carrier/positive/soft"]
    neg = order["readout"]["AMR_original_carrier/unwatermarked/soft"]
    lines = [
        "# 原句与复述的符号读出、顺序贡献诊断", "",
        "2026-10-25 · 新增分析版本 · 已有 Stage1B run2 语料上的探索性诊断", "",
        f"**原句 AMR 符号 BA 仍为 {pct(orig['balanced_accuracy'])}。"
        f"保留置信权重后，软 DP 的真实顺序平均胜率为 {pct(soft['mean_pairwise_win_rate'])}，"
        f"但只有 {soft['unique_correct_top1']}/12 篇实现唯一第一。"
        "目前支持“分数中有顺序信号”，仍不足以支持可靠信息读回。**", "",
        "本次使用原 24 对语料、原 A/B/C/D 字典和固定温度 T=0.10。"
        "原句按已知载体位置评分。MiniLM 原型由全部 96 个开发正例的计划标签建立，"
        "每类 24 个样本；没有梯度训练，没有按识别效果筛选样本或调整读出参数。"
        "这不是训练好的序列识别模型。", "",
        "12 篇开发、12 篇读出均已被此前分析观察过，本报告不恢复独立验证地位。"
        "计划符号是生成意图，不能当成文本实际实现的人工语义标签。", "",
        "## 1. 符号是否读得出来", "",
        "| 表示与观察方式 | 开发 BA | 读出 BA | 读出已评分/计划 | A/B/C/D 召回 |",
        "|---|---:|---:|---:|---|",
    ]
    names = {
        "AMR_original_carrier": "AMR 原句完整载体（已知位置）",
        "C_original_carrier": "Embedding 原句完整载体（已知位置）",
        "AMR_AI_mapped_sentence_mean": "AMR 复述关联句均值（AI 映射，缓存不全）",
        "C_AI_mapped_sentence_mean": "Embedding 复述关联句均值（AI 映射）",
    }
    for method in names:
        dev, test = identity["development"][method], identity["readout"][method]
        lines.append(f"| {names[method]} | {pct(dev['balanced_accuracy'])} | "
                     f"{pct(test['balanced_accuracy'])} | {test['scored']}/{test['planned']} | "
                     + " / ".join(pct(x) for x in test["recall_ABCD"]) + " |")
    lines += [
        "",
        "全部正例的每个条件均保留 192 个计划位置，开发/读出各 96 个。"
        "复述取已封存 AI evidence_span 重叠的全部句子，并等权平均，"
        "不选择分数最高的句子；任一关联句缺解析或分数，整个位置记擦除。"
        "没有位置映射的条目也记擦除。该表衡量与计划标签的一致性，"
        "不是独立人工核验后的符号正确率，也不是盲事件身份 F1。", "",
        "复述 AMR 缓存覆盖：", "",
        "| 分区 | 正例已评分句/全部句 | 负例已评分句/全部句 |",
        "|---|---:|---:|",
    ]
    frozen = read(out / "FREEZE.json")
    for population in ("development", "readout"):
        cells = []
        for arm in ("positive", "unwatermarked"):
            rows = [r for r in coverage if r["doc_id"] in frozen["split"][population] and r["arm"] == arm]
            cells.append(f"{sum(r['q_available'] for r in rows)}/{len(rows)}")
        lines.append(f"| {population} | {' | '.join(cells)} |")
    lines += [
        "",
        "缺失按未知处理，不将缓存可用子集当成全量复述结果。"
        "Embedding 则覆盖全部句子，但 AI 位置映射仍不能代替独立事件对应关系。",
        "",
        "### 原句 AMR 读出混淆", "",
        "| 真值 | 读成 A | 读成 B | 读成 C | 读成 D | 擦除 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, row in zip("ABCD", orig["confusion_ABCD_erasure"]):
        lines.append(f"| {label} | " + " | ".join(f"{v:g}" for v in row) + " |")
    lines += [
        "",
        "A/C 的召回为零，低 BA 不是均匀分布的小误差。"
        "完整 NLL、Brier、Top-2、擦除原因及背景符号偏向见 "
        "[IDENTITY_RESULT.json](IDENTITY_RESULT.json) 与 "
        "[BACKGROUND_RESULT.json](BACKGROUND_RESULT.json)。", "",
        "## 2. 失败在哪一层", "",
        "| 层 | 已有证据 | 可以得出的结论 |",
        "|---|---|---|",
        "| 提示契约 | X 被描述为普通未指定概念，官方对应 amr-unknown | 工程解释不一致已确认；尚无修订后的生成对照 |",
        "| 生成实现 | 18/24 目标含 X，14/24 要求多段姓名，实际开发原句图均缺这些结构 | 目标与实际图错位；不能据图缺失独自判为生成失败 |",
        "| 解析保留 | 明确命令丢失 imperative；花盆被解析成人物 | 有具体解析损失实例，不能估计全量解析错误率 |",
        f"| 分类聚合 | 直接输入全部 24 张完整目标模板图 | 四类 BA={pct(selfread['metrics']['balanced_accuracy'])}；这是分类器独立自检 |",
        "| 剩余归因 | 尚无独立文本实现标签与人工语义图 | 生成/解析/分类因果错误率仍为未知 |", "",
        "模板自检的输入本身就是目标模板，所有目标与自身的 S2Match 分数为 1。"
        "它可以单独检查分组聚合是否把目标图读成别的类；"
        "即使自检通过，也不能把自然文本中剩余失败全部归因于生成或解析。"
        "若自检失败，失败项支持分类聚合存在问题，不说明文本已经成功写入。", "",
        "固定开发样本 24 个原句位置及 24 个复述位置的完整目标图覆盖均为 0/24，"
        "它是严格图覆盖，和模糊匹配的符号读出不是同一个指标。"
        "逐位置记录保留未知归因，见 [LAYER_DIAGNOSIS.json](LAYER_DIAGNOSIS.json)。", "",
        "已形成[提示修订草案](../../semantic_sequence_stage1_prompt_repair1/PROMPT_REPAIR.md)，"
        "没有新生成或重采样。特别是 :polarity X 不能默认视为普通否定，"
        "实际实施前仍需逐模板核对其语义要求。", "",
        "## 3. 顺序是否提供额外信息", "",
        "每篇候选固定为 A/B/C/D 各两次的全部 2,520 个排列。"
        "硬方法只取单个最高权重符号；并列最高视为擦除。"
        "软方法保留四类归一化权重，匹配成本如下，插入/删除成本均为 1：", "",
        "$$", r"\ell(q_i,a)=-\frac{\log(\max(q_i(a),10^{-6}))}{\log 4}.", "$$", "",
        "每个候选通过精确 DP 搜索全部编辑对齐路径。"
        "这不是训练，也不是文本重分句的认证搜索。无序对照只看符号频数，"
        "全部候选严格同分，因此胜率恒为 50%。", "",
        "| 观察序列 | 方法 | 正例平均胜率 | 无水印参考胜率 | 正例平均名次 | 唯一正确第一 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    stream_names = {
        "AMR_original_carrier": "AMR 原句载体，已知位置",
        "C_original_carrier": "Embedding 原句载体，已知位置",
        "C_sentence_stream": "Embedding 复述完整句流，实际文本顺序",
        "AMR_sentence_stream_incomplete": "AMR 复述句流，缓存不完整",
    }
    for method, name in stream_names.items():
        for mode in ("hard", "soft"):
            pos = order["readout"][f"{method}/positive/{mode}"]
            background = order["readout"][f"{method}/unwatermarked/{mode}"]
            lines.append(f"| {name} | {mode} | {pct(pos['mean_pairwise_win_rate'])} | "
                         f"{pct(background['mean_pairwise_win_rate'])} | "
                         f"{pos['mean_rank']:.2f} | {pos['unique_correct_top1']}/12 |")
    delta = paired["AMR_original_carrier/soft"]
    lo, hi = soft["descriptive_document_bootstrap_95"]
    dlo, dhi = delta["descriptive_document_bootstrap_95"]
    lines += [
        "",
        f"原句 AMR 软 DP 的平均胜率为 {pct(soft['mean_pairwise_win_rate'])}，"
        f"文档 bootstrap 的描述性 95% 区间为 [{pct(lo)}, {pct(hi)}]。"
        f"无水印配对参考为 {pct(neg['mean_pairwise_win_rate'])}；"
        f"正例减背景参考为 {100 * delta['mean_positive_minus_background_reference']:.2f} 个百分点，"
        f"描述性区间为 [{100*dlo:.2f}, {100*dhi:.2f}] 个百分点，"
        f"{delta['positive_greater_documents']}/12 对正例更高。", "",
        "无水印文本没有真实计划符号。这里将配对正例的公开计划序列用作背景参考，"
        "仅检验相同评分对背景是否也有偏向，不能称为无水印解码准确率、TPR 或 FPR。"
        "背景参考本身高于 50%，意味着不能只看正例超出 50% 就断言水印特异性。", "",
        "复述句流在评分时只使用复述文本的分句顺序，不读取源槽位映射。"
        "它可以检验当前表达顺序中是否仍有可利用信号，"
        "但每句仍可能包含多个事件，不能叫盲事件恢复。"
        "缺失 AMR 句的软匹配成本对四类都是 1；该行仅展示缓存不全时的诊断，"
        "不作 AMR 与 Embedding 复述能力的公平比较。", "",
        "全部方法均报告 hard/soft，不根据读出结果挑赢家或重新调温度。"
        "不训练密钥状态机，不进行消息编码，不声称完成局部插删或重排鲁棒性验证。", "",
        "## 4. 当前可执行结论", "",
        "继续优先处理符号身份。原句的软分数可用于顺序排序，"
        "但排名有信号与正确解码整条序列必须分开。"
        "先修提示契约并做独立语义核验，再决定是否修解析/分类或调整字典。"
        "现有数据不足以支持直接投入序列模型训练。", "",
        "AMR 全量复述评分、独立人工生成实现率、正式盲事件身份 F1、"
        "攻击鲁棒性仍未完成。本分析没有恢复原迁移 manifest，"
        "其整卡显存门触发后的永久 HALT 状态保持原样。", "",
        "## 5. 复现与文件", "",
        f"独立小图编辑路径穷举 {validation['small_exhaustive_path_cases']} 个用例通过；"
        "2,520 候选唯一性与等频约束、均匀/擦除平局、成本精确重放均通过。"
        "原句 35/96 的读出结果与旧诊断逐项复算一致。"
        f"MiniLM 输入最长 {validation['embedding_max_tokens']} tokens，无截断；"
        f"重复编码最大误差 {validation['embedding_replay_max_absolute_error']:.3g}，门限 1e-6。"
        "输入文件 SHA 在完成后全部不变。", "",
        "- [冻结参数](FREEZE.json) · [输入与源码 SHA](SOURCE_SHA256.json)",
        "- [身份结果](IDENTITY_RESULT.json) · [目标图自检](TEMPLATE_SELF_READ.json)",
        "- [顺序汇总](ORDER_RESULT.json) · [逐文档名次](ORDER_PER_DOCUMENT.json)",
        "- [配对差值](ORDER_PAIRED_READOUT.json) · [复述缓存覆盖](PARAPHRASE_COVERAGE.json)",
        "- [分层诊断](LAYER_DIAGNOSIS.json) · [验证记录](VALIDATION.json)",
        "- [预测封存](PREDICTION_SEAL.json) · [收口状态](STATUS.json)", "",
        "运行脚本是 [run_symbol_sequence_diagnostic.py](../../../scripts/run_symbol_sequence_diagnostic.py)。"
        "模型使用固定 revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41，"
        "9 个资产 SHA 与仓库已有模型清单完全一致。"
        "文本、实际图、原型向量和完整候选成本放在 private/，"
        "聚合报告没有复制封存原文或访问配置。", "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in out.iterdir() if p.is_file() and p.name != "OUTPUT_SHA256.json"}
    manifest["report_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"report": str(out / "REPORT.md"), "files_sealed": len(manifest)}))


if __name__ == "__main__":
    main()
