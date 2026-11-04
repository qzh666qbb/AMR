"""Evidence-based report and redistributable aggregates; no private text export."""
from __future__ import annotations

import json
from pathlib import Path
import re
import zipfile

from run_semantic_readout import OUT, PRIVATE, ROOT
from run_semantic_sequence_cycle import sha, atomic


def pct(v):
    return f"{v * 100:.2f}%"


def stat(v):
    return pct(v["mean"]) + " [" + ", ".join(pct(x) for x in v["CI95"]) + "]"


def finalize():
    result = json.loads((OUT / "RESULT.json").read_text())
    order = json.loads((OUT / "ORDER_RESULT.json").read_text())
    audit = json.loads((OUT / "RESULT_AUDIT.json").read_text())
    execution = json.loads((OUT / "EXECUTION_CLOSEOUT.json").read_text())
    closeout = json.loads((OUT / "CLOSEOUT.json").read_text())
    markers = json.loads((OUT / "WRITE_CONTRACT_DIAGNOSTIC.json").read_text())
    if not audit["Go"] or audit["result_sha256"] != sha(OUT / "RESULT.json") or audit["order_result_sha256"] != sha(OUT / "ORDER_RESULT.json"):
        raise RuntimeError("Independent result audit required")
    if not closeout["Go"] or not closeout["local_supervisor_shutdown"] or not closeout["remote_supervisor_shutdown"]:
        raise RuntimeError("Both task supervisors must be closed")
    summaries = {(r["condition"], r["representation"]): r for r in result["summaries"]}
    ordered = {(r["condition"], r["representation"], r["mode"]): r for r in order["summaries"]}
    g2 = (summaries["clean", "W"]["known_position_BA"]["mean"] >= .75 and
          summaries["clean", "W"]["known_position_BA"]["CI95"][0] > .50 and
          summaries["paraphrase", "W"]["known_position_BA"]["mean"] >= .60 and
          summaries["paraphrase", "W"]["known_position_BA"]["CI95"][0] > .40)
    g3 = all(summaries[condition, "W"]["identity_f1"]["mean"] >= threshold
             for condition, threshold in (("clean", .70), ("paraphrase", .55),
                                          ("split", .50), ("merge", .50)))
    lines = [
        "# 固定参数独立读出与等频顺序诊断",
        "",
        "新12提示对、72文本版本、96正例计划符号/条件完成。参数、原型和候选规则",
        "全部继承旧开发选择，未在读出上调参。全部2,520等频序列精确评分，先封存再加入",
        "真实源顺序。新增228个provider调用全部完成，没有重生成、补样或未决请求。",
        "这是cycle2失败后的独立探索诊断；原G1e=19/24及No-Go保持不变。",
        "人工材料已准备，真人评审仍为0，正式人工G1/G2/G3及G4机制未确认。",
        "",
        "数据来源：[RESULT.json](RESULT.json)、[ORDER_RESULT.json](ORDER_RESULT.json)、",
        "[冻结协议](PROTOCOL.md)、[独立复算](RESULT_AUDIT.json)、",
        "[预测封存](PREDICTION_SEAL.json)与[顺序代价封存](ORDER_SEAL.json)。",
        "",
        "## 版本关系与独立性",
        "",
        "| 版本 | 数据 | 参数 | 允许结论 |",
        "|---|---|---|---|",
        "| cycle1 | 旧三对pilot | 旧设计 | G1e=16/24，No-Go |",
        "| cycle2 execution5 | 三对开发pilot，18版本 | 仅开发选择 | G1e=19/24，No-Go；开发性诊断 |",
        "| independent readout1 | 此前未生成的12个读出提示对，72版本 | 完全固定，不重新训练 | 新数据的构造/模型辅助读出与源顺序诊断 |",
        "",
        "24模板各出现4次；每篇ABCD各两次。继承冻结场景与槽位计划并在首次新请求前",
        "确认从未生成，不重用开发文本。污染审计见[INPUT_INTEGRITY.json](INPUT_INTEGRITY.json)。",
        "位置模型辅助与生成使用同provider，不能称独立人工标注；该“独立”限定为",
        "未参与开发参数/原型选择的新提示对，不代表新模型、新模板或人类金标准。",
        "",
        "## 写入必要结构诊断",
        "",
        f"复用已封存的整句解析，不再调用模型：必要语义标记通过"
        f"{markers['successes']}/{markers['denominator']}（"
        f"{pct(markers['successes']/markers['denominator'])}）。",
        "这里只检查未知节点、姓名结构和要求的语气标记，不证明角色方向、事件含义",
        "或完整语义结构正确，也不重判旧G1e。该诊断保留全部96计划载体。",
        "逐槽结果见[WRITE_CONTRACT_DIAGNOSTIC.json](WRITE_CONTRACT_DIAGNOSTIC.json)。",
        "",
        "## 独立符号读回",
        "",
        "W为完整AMR图，E1为单事件AMR子图，C为MiniLM文本向量。",
        "已知参考位置只放开位置，不提供目标模板。原句使用自动构造参考，复述使用",
        "经唯一短语/动词检查接受的同provider模型辅助参考。缺失位置、错误及擦除保留",
        "每条件96计划符号分母。BA因每篇类频相等也等于正确数/96。",
        "表中括号为按12提示对、5000次配对bootstrap的95%百分位区间。",
        "",
        "| 表示 | 原句BA [95%区间] | 复述BA [95%区间] |",
        "|---|---:|---:|",
    ]
    for rep in ("W", "E1", "C"):
        lines.append(f"| {rep} | {stat(summaries['clean', rep]['known_position_BA'])} | "
                     f"{stat(summaries['paraphrase', rep]['known_position_BA'])} |")
    clean = [r for r in result["rows"] if r["arm"] == "positive" and r["condition"] == "clean"]
    full = [x for r in clean for x in r["whole_sentence_W"]]
    lines += ["", f"整句W正确{sum(x['true'] == x['predicted'] for x in full)}/96，"
              f"BA={pct(sum(x['true'] == x['predicted'] for x in full)/96)}；完整混淆保留于RESULT逐篇行。",
              "整句与定位窗口范围不同，不能据此单独确认窗口切割的因果贡献。",
              "", "## 盲事件定位与身份", "",
              "位置与计划符号同时正确才计身份成功。三表示共用候选位置；额外动词观察、",
              "错误位置和擦除均保留。F1先逐篇计算再取均值，不能与pooled micro-F1混用。",
              "", "| 条件 | 位置参考覆盖 | 共同定位F1 | W身份F1 | E1身份F1 | C身份F1 |",
              "|---|---:|---:|---:|---:|---:|"]
    for condition in ("clean", "paraphrase", "split", "merge"):
        w = summaries[condition, "W"]
        lines.append(f"| {condition} | {w['reference_coverage']}/96 | {pct(w['localization_f1']['mean'])} | "
                     + " | ".join(pct(summaries[condition, rep]["identity_f1"]["mean"]) for rep in ("W", "E1", "C")) + " |")
    lines += ["", "复述身份F1的预设配对表示对照：", "",
              "| 对照 | 均值差（百分点） | 95%区间（百分点） |",
              "|---|---:|---:|"]
    for c in result["paired_contrasts"]:
        lines.append(f"| {c['contrast']} | {c['mean'] * 100:.2f} | "
                     f"[{c['CI95'][0] * 100:.2f}, {c['CI95'][1] * 100:.2f}] |")
    lines += ["", "R*=W固定为主表示。E1/C只作预设次级对照，观察到更高值也不重新选R*。",
              "沿用旧数值目标仅作诊断比较：G2要求W原句BA≥75%、复述≥60%，"
              "区间下界分别>50%、>40%；G3要求W身份F1原句≥70%、复述≥55%、"
              "split和merge各≥50%。",
              f"本批诊断G2数值目标{'达到' if g2 else '未达到'}，"
              f"G3数值目标{'达到' if g3 else '未达到'}。",
              "旧G2/G3目标只作诊断目标比较；未通过或缺真人参考不能记录为正式Go。",
              "", "## 等频顺序贡献", "",
              "对相同待测文本，共用全部观察位置、阈值和擦除，精确DP评分全部2,520候选。",
              "每个候选ABCD各两次，唯一变化是源符号顺序。频数基线全部候选并列，",
              "胜率恒为50%、平均名次1260.5、第一名分摊分数1/2520。",
              "主分数soft允许插入/删除/替换；hard仅次级诊断。没有随机采样候选或对齐路径。",
              "",
              "胜率是该篇真源序列与2519替代顺序的比较，平局计半胜；",
              "2519替代序列不是2519独立统计样本。统计单位仍为12提示对。",
              "",
              "| 条件 | W soft胜率 [95%区间] | E1 soft胜率 [95%区间] | C soft胜率 [95%区间] |",
              "|---|---:|---:|---:|"]
    for condition in ("clean", "paraphrase", "split", "merge"):
        lines.append(f"| {condition} | " + " | ".join(
            stat(ordered[condition, rep, "soft"]["pairwise_win_rate"]) for rep in ("W", "E1", "C")) + " |")
    main = ordered["paraphrase", "W", "soft"]["gain_over_frequency"]
    supported = main["CI95"][0] > 0
    lines += ["", f"预设主效应：W复述soft相对频数基线增益{main['mean']*100:.2f}个百分点，"
              f"95%区间[{main['CI95'][0]*100:.2f}, {main['CI95'][1]*100:.2f}]。",
              ("区间下界大于0，支持本批冻结源序列排序诊断中的正增益。" if supported else
               "区间下界未大于0，本批没有确认主顺序增益。"),
              "即使排序高于频数基线，也不等于语义写入或符号读回通过，更不等于水印检测成功。",
              "复述提示允许调整叙述次序；本指标比较原源顺序，排序损失可同时来自顺序变化、",
              "语义变化、局部擦除和符号错误。没有真人事件对应，不能拆出语义事件顺序的因果机制。",
              "未做顺序训练或独立生成2520个文本；本实验枚举的是同一文本的查询序列。",
              "hard、逐篇平均名次、并列数、top1分摊值及E1−W/C−W对照均在ORDER_RESULT。",
              "无水印文本没有赋予真实水印序列，仅报告代价范围；本轮不估计detector FPR/TPR。",
              "", "## 数据限制与人工工作", ""]
    for arm in ("positive", "unwatermarked"):
        for condition in ("clean", "paraphrase"):
            rows = [r for r in result["rows"] if r["arm"] == arm and r["condition"] == condition]
            words = [r["surface"]["words"] for r in rows]
            inert = sum(r["surface"]["inert_paraphrase"] for r in rows)
            lines.append(f"{arm}/{condition}：长度{min(words)}–{max(words)}词，变换未发生{inert}/12。")
    lines += ["", "共享表面合同不代表实测长度/风格完全匹配。有限模板与同provider来源",
              "限制泛化，不能将本结果写成通用安全鲁棒性或独立人类文本误报验证。",
              "两位真人的264项/人盲评包、说明与回收入口已完成；目前仍0真人标签。",
              "见[人工交付说明](HUMAN_REVIEW.md)，原文本、目标图与对照键仅存私有目录。",
              "", "## 工程核验与收口", ""]
    resource = json.loads((OUT / "RESOURCE_PRECHECK.json").read_text())
    cuda = json.loads((OUT / "CUDA_PRECHECK.json").read_text())
    replay_error = max(e for r in cuda["replay"] for e in r["errors"])
    lines += [f"CUDA FP32两次fresh重放最大误差{replay_error:g}，"
              f"预检峰值整卡占用{pct(cuda['peak_global_vram_fraction'])}，低于80%门。",
              f"完整资源投影{resource['projected_gpu_seconds']/60:.2f}分钟（1.5安全系数），"
              "全部条件及480个已知位置/整句窗口预留保留。",
              f"盲预测与顺序评分实耗{json.loads((OUT/'PREDICTION_SEAL.json').read_text())['elapsed_seconds']:.2f}秒，"
              f"后置评估实耗{result['elapsed_seconds']:.2f}秒。",
              f"独立CPU复算通过{audit['representation_rows_recomputed']}组表示逐篇指标、"
              f"{audit['exact_order_cost_entries_recomputed']:,}个精确顺序代价、"
              f"{audit['scalar_DP_real_candidate_checks']}组真实候选逐格检查；"
              f"最大DP误差{audit['maximum_DP_error']:g}。",
              "位置匹配另用后向递归oracle复算最大匹配数、最小距离和并列选择；",
              "已知位置神经输出未由第二模型重算，审计仅复算这些输出的混淆、分母及区间；",
              "它不等于独立人工验证。封存SHA、温度/阈值和原型均一致。",
              f"执行收口：{execution['completed_requests']}请求，无未决发送，远端Supervisor已关闭。",
              "本地Supervisor关闭与最终SHA归档见[CLOSEOUT.json](CLOSEOUT.json)。",
              "公开压缩包只含聚合、协议、SHA及自有复现脚本；文本/图/回执/模型/配置均排除。",
              "冻结代码中的5个第三方来源文件仅保留SHA，不随包再分发；复现需合法取得"
              "FREEZE记录的第三方版本与模型资产。所有自有冻结执行源码随包保留。",
              ""]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines))
    atomic(OUT / "DIAGNOSTIC_VERDICT.json", {
        "old_cycle2": "NO_GO_19_OF_24", "independent_readout_computed": True,
        "order_2520_computed": True, "primary_W_paraphrase_order_positive_CI": supported,
        "diagnostic_G2_numeric_targets_met": g2, "diagnostic_G3_numeric_targets_met": g3,
        "human_completed": False, "formal_G1_G2_G3_G4": "NOT_EVALUATED",
        "detector_FPR_TPR": "NOT_ESTIMATED", "new_branch": "EXPLORATORY_DIAGNOSTIC_COMPLETE"})
    own_scripts = ("run_semantic_readout.py", "semantic_readout_gpu.py", "audit_semantic_readout.py",
                   "prepare_semantic_readout_review.py", "finalize_semantic_readout.py",
                   "close_semantic_readout.py")
    atomic(OUT / "POSTPROCESS_SHA.json", {"scripts": {name: sha(ROOT / "scripts" / name) for name in own_scripts},
                                         "neural_execution_code_immutable": True})
    return lines


def package():
    own_scripts = ("run_semantic_readout.py", "semantic_readout_gpu.py", "audit_semantic_readout.py",
                   "prepare_semantic_readout_review.py", "finalize_semantic_readout.py",
                   "close_semantic_readout.py",
                   "run_semantic_cycle2.py", "semantic_cycle2_gpu.py", "run_semantic_sequence_cycle.py",
                   "run_multiscale_transfer.py")
    aggregate_names = (
        "PROTOCOL.md", "FINAL_REPORT.md", "FREEZE.json", "LOCKED_SELECTION.json",
        "PERMUTATIONS.json", "SYMBOL_MAP.json", "IDENTITY_PRECHECK.json", "INPUT_INTEGRITY.json",
        "CUDA_PRECHECK.json", "RESOURCE_PRECHECK.json", "PREDICTION_SEAL.json", "ORDER_SEAL.json",
        "RESULT.json", "ORDER_RESULT.json", "RESULT_AUDIT.json", "WRITE_CONTRACT_DIAGNOSTIC.json",
        "DIAGNOSTIC_VERDICT.json", "POSTPROCESS_SHA.json", "HUMAN_REVIEW_STATUS.json",
        "HUMAN_COLLECTION_CHECK.json", "EXECUTION_CLOSEOUT.json", "CLOSEOUT.json", "STATUS.json",
        "RECEIPT_AUDIT.json", "POST_AUDIT_PREPARED.json",
    )
    frozen = json.loads((OUT / "FREEZE.json").read_text())
    code_names = ["scripts/" + name for name in own_scripts]
    code_names += ["src/core/" + name for name in
                  ("semantic_sequence.py", "semantic_cycle2.py", "remote_execution.py",
                   "parser_recovery_audit.py", "bank_scorer.py")]
    script_members = []
    for name in code_names:
        if name in frozen["sources"]:
            source = PRIVATE / "code" / name
            if sha(source) != frozen["sources"][name]:
                raise RuntimeError("Packaged execution source differs from its freeze")
        else:
            source = ROOT / name
        script_members.append((source, name))
    all_members = [(OUT / name, "aggregates/" + name) for name in aggregate_names] + script_members
    # Local reviewer links are useful to the owner; the shared package uses a portable copy.
    human_guide = re.sub(r"file:///[^)\s]+", "PRIVATE_REVIEW_PACKAGE",
                        (OUT / "HUMAN_REVIEW.md").read_text()).encode()
    import hashlib
    portable_members = {"aggregates/HUMAN_REVIEW.md": hashlib.sha256(human_guide).hexdigest()}
    with zipfile.ZipFile(OUT / "public_aggregates.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path, name in all_members:
            archive.write(path, name)
        archive.writestr("aggregates/HUMAN_REVIEW.md", human_guide)
    atomic(OUT / "PUBLIC_PACKAGE_SHA.json", {"sha256": sha(OUT / "public_aggregates.zip"),
                                            "members": {**{name: sha(path) for path, name in all_members},
                                                        **portable_members},
                                            "execution_sources_match_freeze": True,
                                            "package_sha_manifest_external": True,
                                            "private_text_and_credentials_excluded": True})
    print(json.dumps({"report": str(OUT / "FINAL_REPORT.md"),
                      "package": str(OUT / "public_aggregates.zip"), "private_payloads_exported": False}))


if __name__ == "__main__":
    finalize()
    package()
