"""Close the final line with available evidence, including explicit stops."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_swan_blind_eval import save_json
from scripts.analyze_swan_last_round import analyze

EXP = ROOT / "experiments/swan_multiview_development1"
OUT = EXP / "final_completion1"


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def hashes(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.iterdir()) if p.is_file()
            and p.name not in ("SHA256SUMS.json", "heartbeat.json", "WORKFLOW_STATUS.json")
            and "GUARD_HEARTBEAT" not in p.name and "GENERATION_HEARTBEAT" not in p.name
            and not p.name.endswith((".tmp", ".transfer"))}


def archive(reason):
    OUT.mkdir(parents=True, exist_ok=True)
    # Deadline archive gets the final atomic remote aggregates after stopping.
    if reason == "deadline" and (OUT / "REMOTE_LAUNCH_RECEIPT.json").exists():
        try:
            from scripts.follow_swan_last_round import retrieve
            retrieve()
        except Exception as exc:
            save_json(OUT / "FINAL_TRANSFER_LIMIT.json", {
                "exception_type": type(exc).__name__, "raw_payload_exported": False})
    previous = load(OUT / "CLOSURE.json")
    if previous and previous["reason"] == "confirmation_complete":
        return
    if not (OUT / "INPUT_AUDIT.json").exists() and (OUT / "private/calls").exists():
        from scripts.swan_last_round_inputs import audit_generation
        audit_generation()
    outcome = load(OUT / "RESULTS.json")
    verification = None
    if reason == "confirmation_complete":
        maximum_error = 0.
        def compare(expected, actual):
            nonlocal maximum_error
            if isinstance(expected, dict):
                if not isinstance(actual, dict) or set(expected) != set(actual):
                    raise RuntimeError("Local replay fields differ")
                for key in expected:
                    compare(expected[key], actual[key])
            elif isinstance(expected, list):
                if not isinstance(actual, list) or len(expected) != len(actual):
                    raise RuntimeError("Local replay array shape differs")
                for a, b in zip(expected, actual):
                    compare(a, b)
            elif isinstance(expected, float) and isinstance(actual, (int, float)):
                delta = abs(expected-actual)
                maximum_error = max(maximum_error, delta)
                if delta > 1e-12:
                    raise RuntimeError("Local floating statistics replay differs")
            elif type(expected) is not type(actual) or expected != actual:
                raise RuntimeError("Local replay decisions or counts differ")
        with tempfile.TemporaryDirectory(prefix="swan-final-replay-") as tmp:
            p = Path(tmp)
            for n in ("EVAL_FREEZE.json", "DOCUMENT_SCORES.json"):
                (p/n).write_bytes((OUT/n).read_bytes())
            recomputed = analyze(p)
            compare(outcome, recomputed)
            compare(load(OUT/"CALIBRATION.json"), load(p/"CALIBRATION.json"))
        verification = {"aggregate_decisions_and_calibration_exact": True,
                        "statistics_max_absolute_difference": maximum_error,
                        "statistics_tolerance": 1e-12,
                        "original_remote_aggregates_preserved": True,
                        "result_sha256": hashlib.sha256((OUT/"RESULTS.json").read_bytes()).hexdigest()}
        save_json(OUT / "LOCAL_REPLAY_VERIFICATION.json", verification)
    old = load(EXP / "blind_eval1/run2/RESULTS.json")
    selection = load(OUT / "CANDIDATE_SELECTION.json")
    audit = load(OUT / "INPUT_AUDIT.json")
    source = load(OUT / "SOURCE_FREEZE.json")
    status = {
        "schema": "swan-last-round-closure/v1", "at": datetime.now(timezone.utc).isoformat(),
        "reason": reason, "line_closed": True, "no_successor": True,
        "independent_confirmation_completed": reason == "confirmation_complete",
        "usable_recovery_confirmed": bool(outcome and outcome["usable_recovery_confirmed"])
        if reason == "confirmation_complete" else False,
        "completed_development_methods": old["complete_methods"] if old else [],
        "planned_development_methods": old["planned_methods"] if old else [],
        "selected_candidate": selection["method"] if selection else None,
        "human_sources_frozen": source["selected_articles"] if source else 0,
        "generated_documents_audited": 250 if audit else 0,
        "generation_dispatches": audit["generation_dispatches"] if audit else
            len(list((OUT/"private/calls").glob("*.dispatch.json"))),
        "aggregate_verification": verification,
    }
    save_json(OUT / "CLOSURE.json", status)
    lines = ["# SWAN 句界依赖与固定盲窗：最终归档", "",
             f"最终状态：`{reason}`。本实验线关闭，不补样、不换窗口、不创建后继。",
             "只有完成的阶段提供实测结论；截止、校准不可用或 readiness 失败均保留为正式限制。", "",
             "## 弃权归因", "",
             "新负例测试 41/100 弃权中，39 篇由 46 个唯一失败窗口的严格解析诊断触发，"
             "2 篇为输入问题。旧 newline 正例 10/15 弃权涉及 13 个失败窗口。",
             "这些最终归一化图均可解码，但部分诊断明确忽略 token 或恢复结构；"
             "结构可读不等于原语义完好，本轮保留严格规则。",
             "完整聚合及类别见 [弃权报告](../abstention_audit1/REPORT.md) 和"
             " [旧缓存审计](../abstention_audit1/OLD_S6_CACHE_AUDIT.json)。", "",
             "## 开发多窗口比较", "",
             "已完成方法：" + (", ".join(status["completed_development_methods"]) or "无") + "。",
             "全部冻结方法：" + (", ".join(status["planned_development_methods"]) or "无") + "。",
             "原开发队列、名义 1%/5%、同一保守 5% 规则与完整分母见"
             " [开发报告](../blind_eval1/run2/REPORT.md) 及"
             " [阈值重评](../threshold5_reassessment1/)。",
             "候选选择仅属于开发选择，不能充当独立确认。", "",
             "## 新语料与独立确认", "",
             f"已冻结人类来源文章 {status['human_sources_frozen']} 篇；"
             "目标为 100 篇校准与 200 篇测试。生成目标为 100 篇校准负例、"
             "50 篇正例与 100 篇测试负例，每篇 16 个单次请求。",
             f"已审计生成文档 {status['generated_documents_audited']} 篇。"
             "样本失败不补齐，未完成样本也不得从分母移除。",
             "来源版本与 SHA 见 `SOURCE_FREEZE.json`，两臂共用提示与参数见"
             " `GENERATION_FREEZE.json`；私有原文与响应不在公开聚合包中。",
             "生成完成后，词数差异、主题配对、实际模型标识和调用 token 汇总见"
             " `INPUT_QUALITY_AUDIT.json`。这些是匹配与成本诊断，不替代独立语义质量评估。",
             f"冻结候选：{status['selected_candidate'] or '未进入选择阶段'}。",
             "确认完成，本地重算的判定与计数一致，统计浮点容差为 1e-12。" if verification
             else "独立确认未完整完成，因此没有成功的正式确认指标。",
             "若完成，逐条件人类与生成 FPR、TPR、弃权、区间、校准和预定门见"
             " [独立确认报告](REPORT.md) 与 `RESULTS.json`。"
             if verification else "停止标记与阶段状态保留在本目录。",
             "", "## 结论边界", "",
             "预定全部门通过，支持冻结协议范围内的恢复收益。"
             if status["usable_recovery_confirmed"] else
             "不能主张可用恢复成功。保留句界依赖与恢复局限的系统审计结论。",
             "人类参考来自历史 WikiText Wikipedia，并经过固定重建；"
             "误报结论不能外推为所有人类文本的总体保证。生成未实现模板的正例"
             "按生成意图保留，目标模板诊断不参与阈值或样本筛选。",
             "全分母 FPR 与完整文档 FPR 同时报告，不将弃权当作正确判断。",
             "", "## 复现材料", "",
             "执行入口：`scripts/follow_swan_last_round.py`、"
             "`scripts/swan_last_round_inputs.py`、`scripts/run_swan_last_round.py`、"
             "`scripts/analyze_swan_last_round.py` 与 `scripts/guard_swan_last_round.py`。",
             "控制文件固定 24 小时预算。旧结果、旧输入和永久 HALT 不追溯替换。",
             "公开包仅含聚合结果、SHA、脚本与版本元数据；原始文本、凭证、配置及"
             "第三方 payload 排除。"]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n")
    if verification:
        plot(outcome)
    # Link the same authoritative closure from both current protocol and draft.
    marker = "### 最后一轮独立确认与收口记录"
    for name in ("PROTOCOL.md", "PAPER_DRAFT.md"):
        p = EXP / name
        content = p.read_text()
        if marker in content:
            content = re.sub(
                re.escape(marker) + r".*?(?=\n#{1,3} |\Z)", "",
                content, count=1, flags=re.S).rstrip() + "\n"
        content += ("\n\n" + marker + "\n\n"
                    f"最终状态 `{reason}`，实验线已关闭。已完成阶段、未完成部分、"
                    "弃权归因与独立确认边界统一以"
                    " [最终归档](final_completion1/FINAL_REPORT.md) 和"
                    " `final_completion1/CLOSURE.json` 为准；不授权新增实验。\n")
        p.write_text(content)
    save_json(EXP / "abstention_audit1/SHA256SUMS.json", hashes(EXP / "abstention_audit1"))
    save_json(OUT / "SHA256SUMS.json", hashes(OUT))
    package = OUT / "public_aggregates"
    package.mkdir(exist_ok=True)
    allowlist = (
        "FINAL_REPORT.md", "REPORT.md", "RESULTS.json", "CALIBRATION.json",
        "INJECTION_DIAGNOSTIC.json", "INPUT_QUALITY_AUDIT.json",
        "CLOSURE.json", "LOCAL_REPLAY_VERIFICATION.json",
        "confirmation.pdf", "confirmation.png",
    )
    for n in allowlist:
        p = OUT / n
        if p.exists():
            (package / n).write_bytes(p.read_bytes())
    old_folder = EXP / "blind_eval1/run2"
    for name, destination in (
        ("REPORT.md", "DEVELOPMENT_REPORT.md"),
        ("RESULTS.json", "DEVELOPMENT_RESULTS.json"),
        ("METRICS.csv", "DEVELOPMENT_METRICS.csv"),
    ):
        if (old_folder/name).exists():
            (package/destination).write_bytes((old_folder/name).read_bytes())
    complete_tag = "_".join("P0" if m == "P0_raw" else m
                           for m in status["completed_development_methods"] if m != "P0_strict")
    revised = EXP / "threshold5_reassessment1" / complete_tag
    if (revised/"REPORT.md").exists():
        (package/"DEVELOPMENT_5PCT_REPORT.md").write_bytes((revised/"REPORT.md").read_bytes())
    public_report = (package/"FINAL_REPORT.md").read_text()
    public_report = public_report.replace(
        "../abstention_audit1/REPORT.md", "ABSTENTION_REPORT.md").replace(
        "../abstention_audit1/OLD_S6_CACHE_AUDIT.json", "OLD_S6_CACHE_AUDIT.json").replace(
        "../blind_eval1/run2/REPORT.md", "DEVELOPMENT_REPORT.md").replace(
        "../threshold5_reassessment1/", "DEVELOPMENT_5PCT_REPORT.md")
    (package/"FINAL_REPORT.md").write_text(public_report)
    # Entry points use only relative protocol paths and public provider URLs.
    for n in ("swan_last_round_inputs.py", "run_swan_last_round.py",
              "analyze_swan_last_round.py", "audit_swan_last_round_cache.py"):
        (package / n).write_bytes((ROOT / "scripts" / n).read_bytes())
    (package / "ABSTENTION_REPORT.md").write_bytes(
        (EXP / "abstention_audit1/REPORT.md").read_bytes())
    (package / "OLD_S6_CACHE_AUDIT.json").write_bytes(
        (EXP / "abstention_audit1/OLD_S6_CACHE_AUDIT.json").read_bytes())
    if source:
        save_json(package / "SOURCE_VERSION.json", {
            k: source[k] for k in ("repository", "revision", "source_file",
                                  "source_sha256", "license", "reconstruction")})
    save_json(package / "SHA256SUMS.json", hashes(package))
    from scripts.finalize_swan_audit_artifacts import finalize
    finalize()
    print(json.dumps(status), flush=True)


def plot(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = list(result["methods"])
    conditions = ["baseline", "newline", "merge", "comma", "space"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for j, label in enumerate(("TPR", "human_FPR", "generated_FPR")):
        for i, method in enumerate(methods):
            values = [result["methods"][method]["conditions"][a][label]["rate_all"]
                      for a in conditions]
            if any(v is None for v in values):
                continue
            axes[j].plot(range(5), [100*v for v in values], marker="o", label=method)
        axes[j].set_xticks(range(5), conditions, rotation=25)
        axes[j].set_title(label.replace("_", " "))
        axes[j].set_ylabel("All-document rate (%)")
        axes[j].grid(alpha=.2)
    axes[0].legend(fontsize=8)
    axes[1].axhline(5, color="gray", linestyle="--", linewidth=1)
    axes[2].axhline(5, color="gray", linestyle="--", linewidth=1)
    fig.savefig(OUT / "confirmation.pdf")
    fig.savefig(OUT / "confirmation.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--reason", required=True)
    archive(cli.parse_args().reason)
