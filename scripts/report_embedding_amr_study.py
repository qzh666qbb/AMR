"""Generate a text-free evidence report only from sealed complete evaluations."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core.embedding_amr_runtime import atomic, file_sha, sealed_scores
from src.core.embedding_amr_evaluation import evaluate, fit

ARMS = {"amr_only": "AMR 单路", "embedding_amr": "句向量辅助 AMR",
        "embedding_without_relation": "关闭关系核验"}


def read(path):
    return json.loads(path.read_text())


def verify(folder):
    terminal = read(folder / "TERMINAL.json")
    for name, sha in terminal["files"].items():
        assert file_sha(folder / name) == sha
    return terminal


def report(inputs, study, reviews, out):
    verify(study)
    verify(reviews)
    protocol = read(inputs / "PROTOCOL.json")
    calibration = read(study / "CALIBRATION.json")
    assert file_sha(inputs / "PROTOCOL.json") == read(
        inputs / "INPUT_TERMINAL.json")["protocol_sha256"]
    assert fit(sealed_scores(study / "calibration/SCORES.json"), .05) == calibration
    evaluation = read(study / "EVALUATION.json")
    assert evaluate(sealed_scores(study / "test/SCORES.json"), calibration) == evaluation
    subgroup = read(study / "FULLY_WRITTEN_SUBGROUP.json")
    quality = read(reviews / "SUMMARY.json")
    metrics = evaluation["metrics"]
    out.mkdir()
    quality_rows = quality["conditions"]
    lines = [
        "# 句向量辅助 AMR：完整短文本基准复评",
        "",
        "三臂与三条件的重新校准、测试及 API 评审均已执行并封存。"
        "结果属于历史短文本基准复评，不能直接外推到正常长度文本或新的独立正例。",
        "",
        "写入与检测绑定同一公开 Bank50。全部 30 篇写入尝试进入主分母，"
        "其中 21 篇五个写入槽位全部接受；另 9 篇部分接受。"
        "各条件以 1,000 篇自然来源文本校准，另 1,000 篇来源隔离的文本测误报。"
        "阈值在测试评分前封存；校准、测试和正例均未按检测分数或质量筛选。",
        "",
        "## 检测结果",
        "",
        "以下误报率和检出率采用相同的名义 5% 校准要求。"
        "各方法实际测试误报率仍需逐项查看，名义水平不等于实际误报完全相同。",
        "",
        "| 条件 | 方法 | 测试误报 | 全部写入尝试检出 | 全部写入成功子组检出 |",
        "|---|---|---:|---:|---:|",
    ]
    for condition in protocol["conditions"]:
        for arm, name in ARMS.items():
            null = next(m for m in metrics if (m["condition"], m["arm"], m["kind"])
                        == (condition, arm, "human"))
            positive = next(m for m in metrics if (m["condition"], m["arm"], m["kind"])
                            == (condition, arm, "watermarked"))
            sub = next(m for m in subgroup["metrics"] if (m["condition"], m["arm"]) == (condition, arm))
            cell = lambda m: f'{m["hits"]}/{m["denominator"]} ({m["rate"]:.2%})'
            lines.append(f"| {condition} | {name} | {cell(null)} | {cell(positive)} | {cell(sub)} |")
    lines += ["", "完整 95% 精确区间、擦除数及逐来源配对判定保存在机器结果中。",
              "", "## 融合与 AMR 单路的配对比较", "",
              "| 条件 | 融合独有检出 | AMR 单路独有检出 | 检出率差 | Holm 调整 p |",
              "|---|---:|---:|---:|---:|"]
    for comparison in evaluation["paired_comparisons"]:
        if comparison["kind"] != "watermarked" or comparison["control"] != "amr_only":
            continue
        c = comparison
        lines.append(f'| {c["condition"]} | {c["fusion_only_positive"]} | '
                     f'{c["control_only_positive"]} | '
                     f'{c["rate_difference_fusion_minus_control"]:+.2%} | '
                     f'{c["holm_p_all_reported_contrasts"]:.4g} |')
    gains = [c for c in evaluation["paired_comparisons"]
             if c["kind"] == "watermarked" and c["control"] == "amr_only"
             and c["rate_difference_fusion_minus_control"] > 0
             and c["holm_p_all_reported_contrasts"] < .05]
    interpretation = (
        "部分压力条件出现经多重比较调整后的检出率差异，但仍需结合实际误报与文本质量解释；"
        "本基准不能证明高质量改写下的普遍融合收益。"
        if gains else
        "本基准未检出句向量辅助相对 AMR 单路的统计显著增益。"
        "这不等于证明两者等效；30 篇既有正例的功效有限。")
    lines += ["", interpretation, "", "## 模型评审", "",
              f'评审者为 `{quality["model"]}`，共 {quality["denominator"]} 条；'
              "仅接收原文与候选文本，不读取 Bank、分数、判定或写入接受标签。"
              "标签是模型判断，不是外部事实查证。", "",
              "| 条件 | 含义保持 | 自然度 | 连贯性 | 合理性 | 四项全过 |",
              "|---|---:|---:|---:|---:|---:|"]
    for condition, q in quality_rows.items():
        cells = [f'{q[k]}/{q["denominator"]}' for k in
                 ("meaning", "natural", "coherent", "plausible", "all_pass")]
        lines.append("| " + condition + " | " + " | ".join(cells) + " |")
    lines += ["",
              "joined 删除表面句末标点，split 再按每六个词插入句号；词汇序列保持不变。"
              "模型评审显示编辑后的自然度明显不足，因此仅称为标点结构压力测试。"
              "不把这些结果包装为高质量语义改写攻击的鲁棒性收益。",
              "", "## 审计边界", "",
              "- 真实 BART AMR、MiniLM、官方归一化与完整 Bank 的 S2Match；FP32。",
              "- 前两个解析文档批次精确重放，逐篇检测结果重放；日志绑定失败则停止。",
              "- 候选和对应选择发生在 Bank 评分前；同一候选池、相同准入与去重规则。",
              "- 完整分母保留解析擦除和部分写入；21 篇子组为冻结协议预先指定。",
              "- 这是公开 Bank 的历史短文本复评，不是私钥安全证明或全重分句证书。",
              "- 关系同构核验只证明自动图中的一致性，无法单凭自对应证明解析正确。",
              "",
              f'冻结价格口径 API 用量约 ${quality["cost_at_frozen_prices_usd"]:.6f}；'
              "实际结算以服务商账单为准。",
              ""]
    (out / "完整评估报告.md").write_text("\n".join(lines))
    atomic(out / "检测评估结果.json", evaluation)
    atomic(out / "全部写入成功子组.json", subgroup)
    atomic(out / "模型评审结果.json", quality)
    atomic(out / "最终审计.json", {**read(study / "FINAL_AUDIT.json"),
                                   "review_complete": quality["status_counts"].get("review_complete") == 90,
                                   "interpretation": interpretation,
                                   "calibration_sha256": file_sha(study / "CALIBRATION.json"),
                                   "input_protocol_sha256": file_sha(inputs / "PROTOCOL.json"),
                                   "review_terminal_sha256": file_sha(reviews / "TERMINAL.json"),
                                   "evaluation_terminal_sha256": file_sha(study / "TERMINAL.json")})
    amendment = study.parent / "EXECUTION_AMENDMENT.json"
    if amendment.exists():
        atomic(out / "运行修订记录.json", read(amendment))
    if (study / "PROFILE_EQUIVALENCE.json").exists():
        atomic(out / "修复前后等价性.json", read(study / "PROFILE_EQUIVALENCE.json"))
    atomic(out / "SHA256.json", {p.name: file_sha(p) for p in sorted(out.iterdir())})
    print(json.dumps({"status": "report_complete", "directory": str(out)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("inputs", "study", "reviews", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    import fcntl
    with (a.study / ".report_export.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if a.output.exists():
            for name, sha in read(a.output / "SHA256.json").items():
                assert file_sha(a.output / name) == sha
            print(json.dumps({"status": "report_already_complete"}))
        else:
            report(a.inputs, a.study, a.reviews, a.output)
