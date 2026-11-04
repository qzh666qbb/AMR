"""Align the closed SWAN audit, claims, figures, and numerical replay package."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_swan_blind_eval import save_json
from scripts.replay_swan_public_aggregates import replay

EXP = ROOT / "experiments/swan_multiview_development1"
OUT = EXP / "final_completion1"
OLD = EXP / "blind_eval1/run2"
REVISED = EXP / "threshold5_reassessment1/P0_S6_S8_S10_S12"
METHODS = ("P0_raw", "P0_strict", "S6", "S8", "S10", "S12")
CONDITIONS = ("baseline", "semicolon", "newline", "split", "merge", "comma", "space")


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts
            and p.name != "SHA256SUMS.json" and p.suffix != ".pyc"}


def table(result):
    lines = ["| 方法 | TP/105 | FP/105 | 正例弃权/105 | 负例弃权/105 |",
             "|---|---|---|---|---|"]
    for method in METHODS:
        op = result["methods"][method]["condition_guard_5pct"]["conditions"]["pooled"]
        a, b = op["TPR"], op["FPR"]
        lines.append(
            f"| {method} | {a['decisions_positive']}/105（{100*a['rate']:.1f}%） | "
            f"{b['decisions_positive']}/105（{100*b['rate']:.1f}%） | "
            f"{a['unresolved']}/105 | {b['unresolved']}/105 |")
    return "\n".join(lines)


def figures(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    data = [result["methods"][m]["condition_guard_5pct"]["conditions"]["pooled"]
            for m in METHODS]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.3), constrained_layout=True)
    colors = ["#54616e", "#8593a0", "#147b70", "#bd7940", "#98705c", "#b3a293"]
    for index, label in enumerate(("TPR", "FPR")):
        values = [100*d[label]["rate"] for d in data]
        axes[index].bar(range(6), values, color=colors)
        axes[index].set_title(("Positive decisions", "False-positive decisions")[index])
        axes[index].set_ylim(0, 100 if index == 0 else 16)
        for i, value in enumerate(values):
            axes[index].text(i, value+(.9 if index == 0 else .3), f"{value:.1f}",
                             ha="center", fontsize=9)
    x = np.arange(6)
    axes[2].bar(x-.18, [100*d["TPR"]["unresolved_rate"] for d in data],
                width=.36, color="#54616e", label="Positive")
    axes[2].bar(x+.18, [100*d["FPR"]["unresolved_rate"] for d in data],
                width=.36, color="#bd7940", label="Negative")
    axes[2].set_title("Abstentions")
    axes[2].set_ylim(0, 105)
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(range(6), METHODS, rotation=35, ha="right")
        ax.set_ylabel("Full-denominator rate (%)")
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("Development audit: conservative 5% calibration rule", fontsize=13)
    fig.text(.5, -.025, "105 correlated variants from 15 test source documents per arm; "
             "zero observed FP is not population FPR control.", ha="center", fontsize=9)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"development_summary.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    labels = (("TPR", "rate", "Positive decisions (%)"),
              ("FPR", "rate", "False-positive decisions (%)"),
              ("TPR", "unresolved_rate", "Positive abstentions (%)"),
              ("FPR", "unresolved_rate", "Negative abstentions (%)"))
    for ax, (label, field, title) in zip(axes.flat, labels):
        matrix = [[100*result["methods"][m]["condition_guard_5pct"]
                   ["conditions"][c][label][field] for c in CONDITIONS] for m in METHODS]
        image = ax.imshow(matrix, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(7), CONDITIONS, rotation=30, ha="right")
        ax.set_yticks(range(6), METHODS)
        ax.set_title(title)
        for i in range(6):
            for j in range(7):
                ax.text(j, i, f"{matrix[i][j]:.1f}", ha="center", va="center",
                        color="white" if matrix[i][j] > 65 else "#15252a", fontsize=8)
    fig.colorbar(image, ax=list(axes.flat), shrink=.8, label="Percent")
    fig.suptitle("Each condition: 15 positives and 15 negatives; abstentions retained")
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"development_conditions.{suffix}", dpi=180)
    plt.close(fig)


def stop_description(closure):
    ready = load(OUT / "READINESS.json")
    if closure["independent_confirmation_completed"]:
        result = load(OUT / "RESULTS.json")
        return ("新增评分与校准完整完成。预定性能门为 "
                + json.dumps(result["gates"], ensure_ascii=False)
                + "；只有全部通过才支持冻结协议范围内的可用恢复收益。")
    if ready:
        return (
            f"新增确认在 `{closure['reason']}` 停止。"
            f"精确重放门：{ready['exact_parser_and_score_replay']}；"
            f"保守预计计算 {ready['conservative_estimate_seconds']/3600:.2f} 小时，"
            f"预检剩余预算 {ready['remaining_seconds']/3600:.2f} 小时，"
            f"另需归档预留 {ready['archive_reserve_seconds']/60:.0f} 分钟。"
            "完整 readiness 见 `READINESS.json`。"
            "预检输入不构成正式测试结果，没有新增 FPR、TPR 或注入成功率。")
    failure = load(OUT / "EXECUTION_FAILURE_AUDIT.json")
    generation = load(OUT / "GENERATION_FAILURE_AUDIT.json")
    detail = ""
    if failure:
        detail += ("执行冻结文件创建时发生AttributeError：`sha_file(__file__)`"
                   "将字符串传给要求Path的函数。EVAL_FREEZE未提交，"
                   "新增CUDA readiness、AMR推理与detector评分均未开始。")
    if generation:
        detail += (f"独立输入审计同时发现生成文档合格数为0/250，"
                   f"{generation['surface_rule_counts']['surface_failed_slots']}/4000"
                   "个句子不符合冻结表面规则；生成校准不可用。"
                   "这是生成通道资格失败，不能归因于检测器失效或注入信号不足。")
    return (f"新增确认在 `{closure['reason']}` 停止；没有完整正式确认指标。"
            + detail + "两项失败分别归档，原HALT保留，不修复重跑或换候选。")


def claims(closure):
    entries = [
        ("C01", "confirmed-code-and-development", "官方两阶段检测使用实际 N、完整 bank-max、0.65 与 λ=0.25。",
         "PROTOCOL.md §2；blind_eval1/run2/OFFICIAL_DIAGNOSTICS.json"),
        ("C02", "confirmed-development", "semicolon/comma/space 在本开发语料被分为单句，z≤√(3N) 阻止其越过统一阈值。",
         "blind_eval1/run2/OFFICIAL_DIAGNOSTICS.json"),
        ("C03", "bounded-development", "merge 的零检出同时涉及计数限制和单元分数变化，单一因果机制未识别。",
         "blind_eval1/run2/OFFICIAL_DIAGNOSTICS.json"),
        ("C04", "bounded-development", "保守 5% 规则下 S6 检出 52/105、误报 2/105，newline 仍误报 2/15。",
         "threshold5_reassessment1/P0_S6_S8_S10_S12/RESULTS.json"),
        ("C05", "bounded-development", "S8/S10/S12 的零实测误报伴随高弃权，不能推出优于 S6。",
         "threshold5_reassessment1/P0_S6_S8_S10_S12/RESULTS.json"),
        ("C06", "confirmed-audit", "失败窗口最终图可解码，但含 token 忽略或结构恢复风险；严格解析规则未放宽。",
         "abstention_audit1/REPORT.md；abstention_audit1/OLD_S6_CACHE_AUDIT.json"),
        ("C07", "bounded-development", "newline 新旧生成批次的 FPR 不可解释为同分布的 13.3%→1% 修复。",
         "newline_negative_expansion1/REPORT.md"),
        ("C08", "frozen-design", "S8 由开发数据按既定排序选择，并在新增响应前冻结；不称为全面最优。",
         "final_completion1/CANDIDATE_SELECTION.json"),
        ("C09", "confirmed-input-audit", "300 篇历史 WikiText 来源冻结；生成目标250篇、4000次单次请求，不按分数筛样。",
         "final_completion1/SOURCE_FREEZE.json；final_completion1/INPUT_AUDIT.json"),
        ("C10", "confirmed" if closure["usable_recovery_confirmed"] else "not-established",
         "冻结协议内的可用恢复收益。", "final_completion1/CLOSURE.json；final_completion1/RESULTS.json"),
        ("C11", "retracted", "旧稿事件子图额外贡献约15pp、SAQS全面优越以及作用域错配的单一因果解释。",
         "历史 oracle 与域不匹配对照；新稿不保留该主张"),
        ("C12", "not-tested", "AMR 特有机制、对复述/翻译的鲁棒性、长期部署和通用人类文本 FPR 保证。",
         "不在本次冻结确认范围"),
        ("C13", "confirmed-input-audit", "生成资格审计0/250合格，1912/4000句不符合表面规则；无法校准生成负例。",
         "final_completion1/GENERATION_FAILURE_AUDIT.json"),
        ("C14", "confirmed-execution-audit", "新增执行因字符串路径SHA调用失败在冻结创建时永久停止，未开始AMR推理或评分。",
         "final_completion1/EXECUTION_FAILURE_AUDIT.json；final_completion1/HALT.json"),
    ]
    content = {"schema": "swan-final-claim-registry/v1",
               "line_closed": True, "no_successor": True,
               "claims": [{"id": i, "status": s, "claim": c, "evidence": e}
                          for i, s, c, e in entries]}
    save_json(OUT / "CLAIM_REGISTRY.json", content)
    lines = ["# 最终主张登记", "",
             "状态只针对本实验线；其他分支、历史 oracle 或域不匹配零例不填补验证缺口。",
             "", "| ID | 状态 | 主张 | 证据 |", "|---|---|---|---|"]
    lines.extend(f"| {i} | {s} | {c} | {e} |" for i, s, c, e in entries)
    (OUT / "CLAIM_REGISTRY.md").write_text("\n".join(lines)+"\n")


def align_documents(closure, result):
    history = EXP / "history"
    history.mkdir(exist_ok=True)
    old_draft = history / "PAPER_DRAFT_before_final_closure.md"
    if not old_draft.exists():
        old_draft.write_text("> 历史工作稿，含已撤回主张；不得作为现行结论。\n\n"
                             + (EXP / "PAPER_DRAFT.md").read_text())
    description = stop_description(closure)
    paper = f"""# SWAN 句界依赖与固定盲窗恢复的系统审计

本稿为最终技术审计稿，实验线已关闭：`{closure['reason']}`。现行证据以
[最终报告](final_completion1/FINAL_REPORT.md)、
[主张登记](final_completion1/CLAIM_REGISTRY.md)及其 SHA 清单为准。
本稿不预设投稿 venue，也不主张提出了新的鲁棒语义水印机制。

## 摘要

SWAN 将实际句子解析为 AMR，并取与全部秘密模板匹配的最高分。格式编辑改变
检测端观察到的句子单元，因此既可能改变图分数，也可能改变文档计数统计量。
本审计区分官方源码行为、正负全文对称的原始计数参照与严格覆盖适配，并在
30 对生成式开发文档上完整比较官方分句及6、8、10、12词的步长1盲窗口。
主测试为后15对源文档，每篇有七种相关变体。保守5%经验校准规则下，官方严格
适配检出44/105；S6检出52/105，但误报2/105，newline条件仍为2/15。S8、
S10、S12检出分别为11/105、5/105、4/105，伴随大量弃权。单句条件的
$z\\le\\sqrt{{3N}}$ 上限解释了部分零检出，不能把失效全部归因于AMR表征损坏。
最后一次确认冻结300篇历史WikiText人类参考与250篇逐句生成目标，只比较
官方适配和预先选定S8。{description}
本稿保留句界依赖及固定窗口的工程限制，不保留旧稿事件子图额外贡献约15pp。

## 1 研究目标与验证链

| 环节 | 本次回答 |
|---|---|
| 研究目标 | 句界格式编辑如何使 SWAN 文档检测失效 |
| 攻击下的实际瓶颈 | 实际句数与统一z阈值的计数限制；单元匹配变化；严格解析导致全文弃权 |
| 有针对性的工程对照 | 固定密集盲窗与完整bank-max，避免依赖原始槽位 |
| 必要验证与边界 | 逐条件TP/FP、完整分母、弃权、未见生成语料及历史人类参考；只按完成阶段下结论 |

这条验证链没有识别AMR特有的因果收益，也没有实现学习式边界恢复、事件序列、
多尺度融合或可证明鲁棒性。它们不作为本稿的机制或实验结果。

## 2 检测协议

对实际检测单元 $u$，$s(u)=\\max_{{t\\in\\mathcal{{B}}}}\\mathrm{{S2Match}}
(\\mathrm{{TemplateNormalize}}(\\mathrm{{AMR}}(u)),t)$，模板库大小为100。
官方两阶段参照取 $s(u)\\ge0.65$ 为green，并计算

$$
z=\\frac{{G-0.25N}}{{\\sqrt{{N\\cdot0.25\\cdot0.75}}}}.
$$

官方human wrapper只取前五句，因此公平比较采用正负全文对称适配，并保留
原始空图计数和严格覆盖两个版本。P0_raw是原始计数行为参照，并非完整原始
wrapper的逐比特复现。S2Match随机过程使用冻结的确定性适配。

纯窗只接收受扰动全文，按空白词连续枚举所有窗口，步长为1并覆盖尾部。
规划和聚合不读取原始句界、槽位或目标模板。窗口以green比例为文档分数，
不把重叠窗口代入独立二项z检验。任一候选的上下文、选中束诊断、图归一化或
bank评分失败，严格版本整篇弃权；弃权保留在分母中。
详见[协议](PROTOCOL.md)与[开发冻结](blind_eval1/run2/FREEZE.json)。

## 3 数据、阈值与统计

开发数据为30对经过既往探索的生成文档；前15篇负例校准，后15对主测试。
baseline、semicolon、newline、split、merge、comma、space均送入真实分句。
merge引入连词and，不能将全部变换称为严格词汇不变或未经核验的语义等价攻击。
历史已知槽位结果及S12/6配置仅作为历史材料，不与新S12/1混表。

同时保留原名义1%/5%工作点与事后保守5%规则。后者对各条件完整校准负例取
95%经验阈值，再取最大值；阈值数值只用校准数据，规则设计已见过开发测试结果。
它是开发诊断，不是独立确认。105个变体来自15个源文档，不视为105篇独立文本。
每条件0/15误报的双侧95%区间上界为21.8%，不能证明总体FPR达标。

## 4 开发结果与失效归因

{table(result)}

表中采用同一保守5%经验规则；来源为
[完整窗口重评](threshold5_reassessment1/P0_S6_S8_S10_S12/REPORT.md)。
原名义工作点另见[开发报告](blind_eval1/run2/REPORT.md)，不追溯替换。

![开发检测与弃权](final_completion1/development_summary.png)

![逐条件检测与弃权](final_completion1/development_conditions.png)

semicolon/comma/space在本开发语料各有一个实际句子，即使全green，
$z\\le\\sqrt{{3N}}=1.732$仍不能越过校准阈值。P0_raw在这些条件没有严格
弃权却同样零检出，说明失效不能全部归因于严格解析。merge仍有6–10句，
同时存在计数限制和单元分数变化，没有识别单一因果机制。
参见[官方诊断](blind_eval1/run2/OFFICIAL_DIAGNOSTICS.json)。

S6在merge/comma/space分别检出9/15、7/15、8/15，但newline误报仍为2/15。
较长窗口的低实测误报伴随大量全文弃权，不能据此宣称更稳健。失败窗口的最终
归一化图可解码，但部分解析诊断涉及token忽略或结构恢复；结构可读不能证明
原语义完好。本次保留严格规则，未放宽解析来提高完整率。
参见[弃权审计](abstention_audit1/REPORT.md)。

newline增样批次的生成提示与旧逐槽流程不同。其完整测试负例为59/100，
条件FPR为1/59，95%区间为[0.04%,9.09%]；新阈值回看旧负例为3/15。
它没有确认同分布误报修复。参见[增样报告](newline_negative_expansion1/REPORT.md)。

## 5 最后一次确认

S8在新增响应前按冻结开发排序选择：先最小化最坏条件FPR，再比较破句界TPR、
完整率、成本及窗长。这是选择规则的结果，不表示S8全面优于S6。
人类参考冻结100篇校准、200篇测试；生成目标冻结100篇校准负例、50篇
正例、100篇测试负例，每篇16次单次请求，上限4000次。不按分数剔除失败正例。

新单元与文档阈值重新校准，来源分别报告。预定成功门同时要求：破句界配对
收益区间下界大于0、五条件人类完整文档FPR的Bonferroni校正99%区间上界
不超过5%、五条件正例与人类测试负例完整率均至少80%。
单元与文档阈值共享校准负例，不主张正式conformal保证。

{description}

实际完成内容及停止证据见[最后一轮归档](final_completion1/FINAL_REPORT.md)。
人类参考来自历史WikiText重建，不代表所有文体、语言或全部人类文本。
"""
    paper += """
## 6 结论与复现范围

本审计确认了冻结开发语料下的句界依赖，并量化固定盲窗的检出、误报和弃权代价。
它没有建立事件子图的额外收益、AMR特有机制或通用人类文本误报保证。
新确认只能以实际完成评分和冻结门为依据，未完成或不可校准的状态不计为零误报。
实验线已关闭，后续仅核验与归档，不追加实验或创建后继。

公开包支持从无原文的单元分数聚合重算开发阈值、判定、计数与区间；若确认完成，
亦重算确认指标。公开包不包含原始文本、秘密模板、凭证、本地配置、模型或
无再分发权的第三方payload，不能据此独立重跑生成和AMR解析。
版本、SHA、复现范围与命令见[公开聚合包](final_completion1/public_aggregates/README.md)。
历史草稿保存在history目录；旧图和旧分数不进入本稿主结论。
"""
    (EXP / "PAPER_DRAFT.md").write_text(paper)
    protocol = (EXP / "PROTOCOL.md").read_text()
    protocol = protocol.split("\n## 14. 最终归档")[0].rstrip()+"\n"
    protocol = protocol.replace(
        "四个纯窗尺度仍在运行，不能把此阶段记为全队列完成。",
        "以上为P0阶段记录；现已完成全部纯窗尺度，完整结果见第11.6节。")
    marker = "### 最后一轮独立确认与收口记录"
    protocol = re.sub(re.escape(marker)+r".*?(?=\n#{1,3} |\Z)", "", protocol,
                      count=1, flags=re.S)
    # Move the follower's appended §11.6 next to the other §11 results.
    appended = re.search(r"\n### 11\.6 本轮全部方法实际结果.*\Z", protocol, re.S)
    if appended:
        block = appended.group(0).rstrip()+"\n\n"
        protocol = protocol[:appended.start()].rstrip()+"\n\n"
        protocol = protocol.replace("\n## 12.", block+"## 12.", 1)
    protocol += (f"\n## 14. 最终归档\n\n实验线已关闭：`{closure['reason']}`。\n\n"
                 + description + "\n\n主张、全部窗口结果及复现范围统一以"
                 "[最终报告](final_completion1/FINAL_REPORT.md)和"
                 "[主张登记](final_completion1/CLAIM_REGISTRY.md)为准。"
                 "第1–13节保留各阶段历史口径；未完成状态不改写为成功。\n")
    status_line = ("\n> 当前状态：实验线已关闭；现行结论见"
                   " [最终归档](final_completion1/FINAL_REPORT.md)。"
                   "下文保留各阶段协议和历史结果。\n")
    protocol = protocol.replace(status_line, "")
    protocol = protocol.replace("\n", "\n"+status_line, 1)
    (EXP / "PROTOCOL.md").write_text(protocol)
    fig_note = """# 图表证据范围

当前技术审计稿的主图为final_completion1/development_summary与development_conditions。
独立确认图仅在确认评分完整完成时存在，且必须结合校准可用性、弃权和冻结门阅读。
本目录fig1–fig3来自旧作用域/域不匹配实验，含已撤回的机制或优越性解释。
fig4_window_sweep属于已知槽位oracle诊断，S12使用步长6；不能作为全文盲测主图。
旧图保留历史文件，不进入当前主张登记。
"""
    (EXP / "figures/EVIDENCE_SCOPE.md").write_text(fig_note)
    (EXP / "README.md").write_text(
        "# SWAN 句界依赖系统审计\n\n"
        f"实验线关闭：`{closure['reason']}`；不追加实验或创建后继。\n\n"
        "- [最终归档与结论](final_completion1/FINAL_REPORT.md)\n"
        "- [最终主张登记](final_completion1/CLAIM_REGISTRY.md)\n"
        "- [当前技术稿](PAPER_DRAFT.md)\n"
        "- [历史及执行协议](PROTOCOL.md)\n"
        "- [公开聚合复现包](final_completion1/public_aggregates/README.md)\n\n"
        "其他旧报告、oracle图与独立分支保持历史记录，不能填补本次验证缺口。\n")


def package(closure, result):
    p = OUT / "public_aggregates"
    p.mkdir(exist_ok=True)
    # Remove four earlier flat entry-point copies; preserve their exact source in
    # a clearly labelled code inventory only when needed for aggregate replay.
    for name in ("swan_last_round_inputs.py", "run_swan_last_round.py",
                 "analyze_swan_last_round.py", "audit_swan_last_round_cache.py"):
        (p / name).unlink(missing_ok=True)
    files = {
        OUT: ("FINAL_REPORT.md", "REPORT.md", "RESULTS.json", "CALIBRATION.json",
              "DOCUMENT_SCORES.json", "INJECTION_DIAGNOSTIC.json", "INPUT_QUALITY_AUDIT.json",
              "CLOSURE.json", "LOCAL_REPLAY_VERIFICATION.json", "CLAIM_REGISTRY.json",
              "CLAIM_REGISTRY.md", "READINESS_GATE_AUDIT.json",
              "GENERATION_FAILURE_AUDIT.json", "EXECUTION_FAILURE_AUDIT.json", "HALT.json",
              "JOB_SHUTDOWN_VERIFICATION.json",
              "development_summary.pdf", "development_summary.png",
              "development_conditions.pdf", "development_conditions.png",
              "confirmation.pdf", "confirmation.png"),
    }
    for folder, names in files.items():
        for name in names:
            if (folder / name).exists():
                shutil.copyfile(folder / name, p / name)
    for original, destination in (
        (OLD / "DOCUMENT_SCORES.json", "DEVELOPMENT_DOCUMENT_SCORES.json"),
        (REVISED / "RESULTS.json", "DEVELOPMENT_5PCT_RESULTS.json"),
        (OLD / "OFFICIAL_DIAGNOSTICS.json", "OFFICIAL_DIAGNOSTICS.json"),
        (OUT / "CANDIDATE_SELECTION.json", "CANDIDATE_SELECTION.json"),
    ):
        shutil.copyfile(original, p / destination)
    for file in sorted((EXP / "abstention_audit1").glob("*.json")):
        if file.name != "SHA256SUMS.json":
            destination = "ABSTENTION_RESULTS.json" if file.name == "RESULTS.json" else file.name
            shutil.copyfile(file, p / destination)
    sources = (
        "scripts/replay_swan_public_aggregates.py",
        "scripts/analyze_swan_blind_eval.py", "scripts/analyze_swan_last_round.py",
        "scripts/recalibrate_swan_5pct.py", "src/core/swan_blind_protocol.py",
        "src/core/multiscale_blind.py",
    )
    for name in sources:
        target = p / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    freeze = load(OLD / "FREEZE.json")
    input_audit = load(OUT / "INPUT_AUDIT.json")
    if input_audit:
        save_json(p / "INPUT_AUDIT_SUMMARY.json", {
            k: input_audit[k] for k in (
                "source_family_overlap", "generation_dispatches", "dispatch_cap_pass",
                "all_frozen_documents_retained", "generated_eligible",
                "human_documents", "input_failure_counts", "eligible_by_arm_split",
                "input_manifest_sha256", "injection_failure_exclusion")})
    save_json(p / "IMPLEMENTATION_VERSION.json", {
        "development_environment_versions": freeze["assets"]["versions"],
        "frozen_runtime_settings": freeze["assets"]["settings"],
        "resource_group_fingerprints": {
            k: v["fingerprint"] for k, v in freeze["assets"]["resource_groups"].items()},
        "development_scoring_freeze_sha256": result["source_freeze_sha256"],
        "historical_oracle_scores_sha256": freeze["historical_scores_sha256"],
        "public_source_sha256": {name: sha(ROOT / name) for name in sources},
        "source_has_runtime_credentials": False,
        "published_remote_access_or_runtime_configuration": False,
    })
    readiness = load(OUT / "READINESS.json")
    if readiness:
        save_json(p / "READINESS_SUMMARY.json", {
            key: readiness[key] for key in (
                "ready", "cuda", "dtype", "exact_parser_and_score_replay",
                "longest_valid_full_batch_checked", "conservative_estimate_seconds",
                "remaining_seconds", "archive_reserve_seconds", "gpu_peak_allocated_bytes",
                "gpu_total_bytes", "native_context_rejections")})
    (p / "requirements-replay.txt").write_text("numpy>=1.26,<3\nscipy>=1.13,<2\n")
    (p / "README.md").write_text(
        "# SWAN 系统审计：公开数值聚合复现包\n\n"
        f"实验线关闭：`{closure['reason']}`；候选选择不表示独立确认成功。\n\n"
        "本包可重算开发保守5%阈值、各条件判定、计数、完整分母、弃权和区间。"
        "若RESULTS.json存在，亦重算最后确认的校准、全部指标与冻结门。"
        "全局SHA清单覆盖全部交付文件，数值容差为1e-12。\n\n"
        "在本目录执行：\n\n"
        "```bash\npython -m pip install -r requirements-replay.txt\n"
        "python source/scripts/replay_swan_public_aggregates.py --package .\n```\n\n"
        "确认重算仅用临时的candidate_method包装数值输入，原始freeze SHA保留在"
        "原聚合中；该包装不是原始实验冻结文件，不声称重建了全部执行资产。"
        "重算不会读取原文、加载模型、生成文本或调用detector/API。\n\n"
        "本包不含原始人类/生成文本、秘密模板、模型、凭证、本地配置、远程访问元数据"
        "或无再分发权限的第三方payload。因此它支持数值聚合复现，不能独立重跑"
        "生成、AMR解析及图匹配。保留代码依赖完整，未提供可启动新实验的控制器。\n\n"
        "主结论见FINAL_REPORT.md及CLAIM_REGISTRY.md；历史开发结果不能充当"
        "通用人类文本FPR证明。SOURCE_VERSION.json固定人类参考来源版本。"
        "DEVELOPMENT_5PCT_RESULTS.json及DEVELOPMENT_DOCUMENT_SCORES.json是"
        "开发数值输入；REPORT.md/RESULTS.json仅在新确认完整结束时存在。\n")
    replacements = {
        "../abstention_audit1/REPORT.md": "ABSTENTION_REPORT.md",
        "../abstention_audit1/OLD_S6_CACHE_AUDIT.json": "OLD_S6_CACHE_AUDIT.json",
        "../blind_eval1/run2/REPORT.md": "DEVELOPMENT_REPORT.md",
        "../threshold5_reassessment1/": "DEVELOPMENT_5PCT_REPORT.md",
        "public_aggregates/README.md": "README.md",
    }
    for name in ("FINAL_REPORT.md", "CLAIM_REGISTRY.md"):
        content = (p / name).read_text()
        for old, new in replacements.items():
            content = content.replace(old, new)
        (p / name).write_text(content)
    # The copied historical reports use their original filenames; fix links
    # where the public files have distinct development names.
    for name in ("DEVELOPMENT_REPORT.md", "DEVELOPMENT_5PCT_REPORT.md"):
        if (p / name).exists():
            content = (p / name).read_text()
            content = content.replace("(RESULTS.json)", "(DEVELOPMENT_RESULTS.json)"
                                      if name == "DEVELOPMENT_REPORT.md"
                                      else "(DEVELOPMENT_5PCT_RESULTS.json)")
            content = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                             lambda m: m.group(0) if (p / m.group(2)).exists()
                             else m.group(1)+"（原执行目录证据）", content)
            (p / name).write_text(content)
    save_json(p / "SHA256SUMS.json", inventory(p))
    verification = replay(p)
    save_json(OUT / "PUBLIC_REPLAY_VERIFICATION.json", verification)
    return p


def finalize():
    closure = load(OUT / "CLOSURE.json")
    if not closure or not closure["line_closed"] or not closure["no_successor"]:
        raise RuntimeError("Only a formally closed line may be finalized")
    result = load(REVISED / "RESULTS.json")
    if set(result["complete_methods"]) != set(METHODS):
        raise RuntimeError("Development window comparison incomplete")
    readiness = load(OUT / "READINESS.json")
    if readiness:
        gates = {
            "exact_replay": readiness["exact_parser_and_score_replay"],
            "gpu_peak_below80pct": readiness["gpu_peak_allocated_bytes"]
            < .8*readiness["gpu_total_bytes"],
            "gpu_current_usage_below80pct": readiness["gpu_total_bytes"]
            - readiness["gpu_free_after_pilot_bytes"] < .8*readiness["gpu_total_bytes"],
            "compute_plus_archive_within_remaining_budget":
            readiness["conservative_estimate_seconds"]+readiness["archive_reserve_seconds"]
            < readiness["remaining_seconds"],
        }
        if all(gates.values()) != readiness["ready"]:
            raise RuntimeError("Readiness aggregate gates disagree")
        save_json(OUT / "READINESS_GATE_AUDIT.json", {
            "gates": gates, "readiness_sha256": sha(OUT / "READINESS.json"),
            "all_gates_pass": all(gates.values()), "new_inference_calls": 0})
    claims(closure)
    figures(result)
    report = (OUT / "FINAL_REPORT.md").read_text()
    marker = "\n## 最终证据核验与主张对齐"
    report = report.split(marker)[0].rstrip()
    report += ("\n"+marker+"\n\n"+stop_description(closure)+"\n\n"
               +table(result)+"\n\n"
               "本表采用事后保守5%规则；105个同源变体来自15个测试源文档。"
               "每条件0/15误报的95%上界为21.8%，零观测不等于总体达标。\n\n"
               "开发队列完整处理150,128个唯一输入、86,309个唯一图。"
               "共享缓存只复用相同输入或归一化图，不能据此给出各方法冷启动耗时。\n\n"
               "![开发检测与弃权](development_summary.png)\n\n"
               "![逐条件检测与弃权](development_conditions.png)\n\n"
               "旧稿的事件子图额外15pp、SAQS全面优越、单一作用域因果解释已撤回。"
               "研究目标→攻击瓶颈→固定盲窗对照→验证边界见更新后的技术稿。"
               "正式主张见[登记表](CLAIM_REGISTRY.md)。\n\n"
               "公开包支持无原文数值聚合复现，不包含重跑生成、解析及bank匹配"
               "所需私有/第三方资产。命令和版本见"
               "[复现说明](public_aggregates/README.md)。\n")
    if readiness:
        failures = [name for name, passed in gates.items() if not passed]
        report += ("\n就绪门独立复核：" + ("全部通过" if not failures else
                   "未通过 "+", ".join(failures))
                   + "。数值与来源SHA见 `READINESS_GATE_AUDIT.json`。\n")
    quality = load(OUT / "INPUT_QUALITY_AUDIT.json")
    if quality:
        report += ("\n生成匹配质量：主题配对 "
                   f"{quality['topic_pairs']} 对，配对平均绝对词数差 "
                   f"{quality['paired_mean_absolute_word_count_gap']:.1f}。"
                   "该审计未进行独立语义或人类质量评估，也不按此结果筛样或改阈值。"
                   "各组完成数、模型标识与调用token见 `INPUT_QUALITY_AUDIT.json`。\n")
    (OUT / "FINAL_REPORT.md").write_text(report)
    align_documents(closure, result)
    public = package(closure, result)
    save_json(OUT / "ARTIFACT_VERIFICATION.json", {
        "at": datetime.now(timezone.utc).isoformat(),
        "line_closed": True, "no_successor": True,
        "paper_protocol_claims_aligned": True,
        "historical_draft_preserved": True,
        "canonical_development_figures_generated": True,
        "public_replay_verified": True,
        "public_manifest_entries": len(inventory(public)),
        "public_raw_text_graphs_credentials_runtime_config_included": False,
    })
    save_json(OUT / "DELIVERY_SHA256SUMS.json", {
        str(p.relative_to(EXP)): sha(p) for p in (
            EXP / "README.md", EXP / "PAPER_DRAFT.md", EXP / "PROTOCOL.md",
            EXP / "figures/EVIDENCE_SCOPE.md", OUT / "FINAL_REPORT.md",
            OUT / "CLAIM_REGISTRY.md", OUT / "CLAIM_REGISTRY.json",
            OUT / "CLOSURE.json", OUT / "ARTIFACT_VERIFICATION.json",
            OUT / "PUBLIC_REPLAY_VERIFICATION.json", public / "SHA256SUMS.json")})
    # Exclude mutable supervisor heartbeats from the final scientific manifest.
    save_json(OUT / "SHA256SUMS.json", {
        p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file()
        and p.name not in ("SHA256SUMS.json", "heartbeat.json", "WORKFLOW_STATUS.json")
        and "HEARTBEAT" not in p.name and not p.name.endswith((".tmp", ".transfer"))})
    print(json.dumps({"artifacts_finalized": True, "reason": closure["reason"],
                      "public_aggregate_replay_verified": True}))


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    finalize()
