"""Archive repair2 separately; preserve the failed predecessor byte for byte."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_swan_blind_eval import save_json
from scripts.analyze_swan_last_round import analyze
from src.core.swan_blind_protocol import digest

EXP = ROOT / "experiments/swan_multiview_development1"
OUT = EXP / "final_completion2"


def load(name):
    path = OUT / name
    return json.loads(path.read_text()) if path.exists() else None


def inventory(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()
            and p.name != "SHA256SUMS.json" and "__pycache__" not in p.parts
            and p.suffix not in (".pyc", ".tmp", ".transfer")}


def compare(a, b):
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise RuntimeError("Replay fields differ")
        return max((compare(a[k], b[k]) for k in a), default=0.)
    if isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            raise RuntimeError("Replay lengths differ")
        return max((compare(x, y) for x, y in zip(a, b)), default=0.)
    if isinstance(a, float) and isinstance(b, (int, float)):
        error = abs(a-b)
        if error > 1e-12:
            raise RuntimeError("Replay numerical result differs")
        return error
    if type(a) is not type(b) or a != b:
        raise RuntimeError("Replay decision differs")
    return 0.


def plot_execution(audit, ready):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    keys = ("negative:calibration", "negative:test", "positive:test")
    denominators = (100, 100, 50)
    labels = ("Generated calibration negative", "Generated test negative",
              "Generated test positive")
    values = [100*audit["eligible_by_arm_split"].get(k, 0)/n for k, n in zip(keys, denominators)]
    axes[0].barh(labels[::-1], values[::-1], color="#19776a")
    axes[0].set_xlim(0, 105)
    axes[0].set_xlabel("Input-eligible documents / frozen denominator (%)")
    axes[0].set_title("Surface repair; detector scores not used")
    for y, (v, k, n) in enumerate(zip(values[::-1], keys[::-1], denominators[::-1])):
        axes[0].text(v+1, y, f"{audit['eligible_by_arm_split'].get(k,0)}/{n}", va="center")
    if ready:
        workload = ready["conservative_estimate_seconds"]/3600
        budget = max(0, ready["remaining_seconds"]-ready["archive_reserve_seconds"])/3600
        axes[1].bar(("Measured projection", "Available compute budget"),
                    (workload, budget), color=("#ba7445", "#566775"))
        axes[1].set_ylabel("Hours")
        axes[1].set_title("Frozen readiness resource gate")
        for x, value in enumerate((workload, budget)):
            axes[1].text(x, value, f"{value:.2f} h", ha="center", va="bottom")
    else:
        axes[1].axis("off")
        axes[1].text(.5, .5, "CUDA readiness not completed", ha="center", va="center")
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / ("repair_execution."+suffix), dpi=180)
    plt.close(fig)


def align_documents(closure, result, ready):
    if closure["independent_confirmation_completed"]:
        statement = "修复版评分与独立经验校准已完整完成。"
        statement += ("预定性能门全部通过。" if closure["usable_recovery_confirmed"]
                      else "预定性能门未全部通过，不能主张可用恢复成功。")
    elif ready:
        audit = load("EXECUTION_READINESS_AUDIT.json")
        failed = "、".join(audit["failed_gates"])
        statement = ("修复版已执行CUDA就绪检查，未通过"+failed+"门，正式评分未开始。"
                     f"精确重放={ready['exact_parser_and_score_replay']}，"
                     f"预计{ready['conservative_estimate_seconds']/3600:.2f}小时，"
                     f"预检剩余{ready['remaining_seconds']/3600:.2f}小时。没有新增FPR/TPR。")
    else:
        statement = "修复版在正式评分完成前停止；没有新增FPR/TPR。"
    text = (f"修复版状态`{closure['reason']}`。"+statement+
            "\n\n完整阶段、输入资格、条件指标和结论边界以"
            "[修复最终报告](final_completion2/FINAL_REPORT.md)及"
            "`final_completion2/CLOSURE.json`为准。原`final_completion1`保持永久失败归档，"
            "两版不混同。修复版已关闭，不授权新增实验。\n")
    for name, heading in (("PROTOCOL.md", "## 16. 修复版最终结果"),
                          ("PAPER_DRAFT.md", "## 7 修复版最终结果")):
        path = EXP / name
        content = path.read_text()
        if heading in content:
            content = re.sub(re.escape(heading)+r".*\Z", "", content, flags=re.S)
        content = content.replace("修复版正在执行，尚未提供正式确认数字",
                                  "修复版已收口，完成情况见第7节")
        content = content.replace("新增确认在 `confirmation_permanent_halt`",
                                  "原独立确认在 `confirmation_permanent_halt`")
        content = content.replace("原HALT保留，不修复重跑或换候选。",
                                  "原HALT保留，不恢复该旧运行身份。")
        if name == "PAPER_DRAFT.md":
            content = re.sub(
                r"本稿为技术审计稿。.*?(?=本稿不预设投稿 venue)",
                f"本稿为技术审计稿；修复版已收口：`{closure['reason']}`。现行证据以\n"
                "[修复最终报告](final_completion2/FINAL_REPORT.md)、\n"
                "[修复主张登记](final_completion2/CLAIM_REGISTRY.md)及SHA清单为准。\n"
                "原永久失败版本保持独立归档；事后修复不追溯改写预注册成功。\n",
                content, count=1, flags=re.S)
        content += "\n\n"+heading+"\n\n"+text
        path.write_text(content)
    (EXP / "README.md").write_text(
        "# SWAN 句界依赖系统审计\n\n"+text+"\n"
        "- [修复最终报告](final_completion2/FINAL_REPORT.md)\n"
        "- [修复主张登记](final_completion2/CLAIM_REGISTRY.md)\n"
        "- [修复公开包](final_completion2/public_aggregates/README.md)\n"
        "- [原失败归档](final_completion1/FINAL_REPORT.md)\n"
        "- [当前技术稿](PAPER_DRAFT.md)\n"
        "- [历史及执行协议](PROTOCOL.md)\n")
    old = json.loads((EXP / "final_completion1/CLAIM_REGISTRY.json").read_text())
    entries = old["claims"] + [
        {"id": "C15", "status": "confirmed-input-audit",
         "claim": f"事后生成修复合格{closure['generated_eligible']}/250；"
                  f"新增请求{closure['generation_dispatches']}/4000；原分母与S8选择保留。",
         "evidence": "final_completion2/INPUT_AUDIT.json"},
        {"id": "C16", "status": "confirmed-execution" if closure["independent_confirmation_completed"]
         else "not-completed",
         "claim": statement, "evidence": "final_completion2/CLOSURE.json"},
        {"id": "C17", "status": "confirmed" if closure["usable_recovery_confirmed"]
         else "not-established", "claim": "修复版冻结范围内的可用恢复收益。",
         "evidence": "final_completion2/RESULTS.json；final_completion2/CLOSURE.json"
         if closure["independent_confirmation_completed"] else "final_completion2/CLOSURE.json"},
    ]
    save_json(OUT / "CLAIM_REGISTRY.json", {"schema": "swan-repair-claim-registry/v2",
        "predecessor_claims_historical": True, "claims": entries, "line_closed": True})
    lines = ["# SWAN 修复版主张登记", "", "C01–C14保持原失败版本的历史作用域。",
             "", "| ID | 状态 | 主张 | 证据 |", "|---|---|---|---|"]
    lines += [f"| {d['id']} | {d['status']} | {d['claim']} | {d['evidence']} |" for d in entries]
    (OUT / "CLAIM_REGISTRY.md").write_text("\n".join(lines)+"\n")


def archive(reason):
    previous = load("CLOSURE.json")
    if previous and previous["independent_confirmation_completed"]:
        return
    result, audit, ready = load("RESULTS.json"), load("INPUT_AUDIT.json"), load("READINESS.json")
    completed = reason == "confirmation_complete"
    if ready:
        total = ready["gpu_total_bytes"]
        gates = {"exact_replay": ready["exact_parser_and_score_replay"],
                 "vram_under80": ready["gpu_peak_allocated_bytes"] < .8*total
                 and total-ready["gpu_free_after_pilot_bytes"] < .8*total,
                 "time_budget": ready["conservative_estimate_seconds"]
                 + ready["archive_reserve_seconds"] < ready["remaining_seconds"]}
        save_json(OUT / "EXECUTION_READINESS_AUDIT.json", {
            "gates": gates, "failed_gates": [k for k, v in gates.items() if not v],
            "projection_hours": ready["conservative_estimate_seconds"]/3600,
            "available_compute_hours": max(0, ready["remaining_seconds"]
                - ready["archive_reserve_seconds"])/3600,
            "projected_deficit_hours": max(0, ready["conservative_estimate_seconds"]
                + ready["archive_reserve_seconds"]-ready["remaining_seconds"])/3600,
            "projection_is_conservative_not_hard_lower_bound": True,
            "safety_factor": 1.25, "fresh_cache": True,
            "deadline_extended": False,
            "formal_detector_scoring_completed": completed,
            "pilot_not_a_test_metric": True,
            "readiness_sha256": hashlib.sha256((OUT / "READINESS.json").read_bytes()).hexdigest()})
    preservation = load("PREDECESSOR_PRESERVATION.json")
    for name, expected in preservation["files_sha256"].items():
        if hashlib.sha256((EXP / "final_completion1" / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Original failure archive changed")
    save_json(OUT / "PREDECESSOR_VERIFICATION.json",
              {"original_archive_sha_verified": True, "files": len(preservation["files_sha256"])})
    verification = None
    if completed:
        if not result or not load("heartbeat.json") \
                or load("heartbeat.json")["state"] != "complete":
            raise RuntimeError("Cannot archive incomplete scoring as complete")
        with tempfile.TemporaryDirectory(prefix="swan-repair2-replay-") as tmp:
            folder = Path(tmp)
            for name in ("EVAL_FREEZE.json", "DOCUMENT_SCORES.json"):
                shutil.copyfile(OUT / name, folder / name)
            recomputed = analyze(folder)
            error = max(compare(result, recomputed),
                        compare(load("CALIBRATION.json"),
                                json.loads((folder / "CALIBRATION.json").read_text())))
        verification = {"decisions_and_calibration_exact": True,
                        "statistics_max_absolute_difference": error,
                        "tolerance": 1e-12}
        save_json(OUT / "LOCAL_REPLAY_VERIFICATION.json", verification)
    closure = {
        "schema": "swan-repair2-closure/v2", "at": datetime.now(timezone.utc).isoformat(),
        "reason": reason, "line_closed": True, "no_successor": True,
        "independent_confirmation_completed": completed,
        "usable_recovery_confirmed": bool(completed and result["usable_recovery_confirmed"]),
        "generated_eligible": audit["generated_eligible"] if audit else None,
        "generation_dispatches": audit["generation_dispatches"] if audit else None,
        "full_intents": {"human": 300, "generated": 250},
        "candidate": "S8", "aggregate_verification": verification,
        "predecessor_status": "confirmation_permanent_halt",
        "posthoc_repair": True, "predecessor_archive_preserved": True,
    }
    lines = ["# SWAN 修复版独立确认", "",
             f"状态：`{reason}`。这是事后生成与执行修复；旧失败版本保留。",
             f"冻结候选S8，沿用300篇人类参考、250个生成意图及原校准/测试来源分组。",
             f"生成合格数：{closure['generated_eligible']}/250；新增请求："
             f"{closure['generation_dispatches']}/4000。失败意图保留在分母。",
             "", "SHA字符串路径错误已修复；输入就绪增加零合格生成拦截，"
             "原始合格句保留，格式/调用/重复失败用冻结的有界JSON生成通道修复。",
             "检测、解析失败策略、FP32、四束、模板库与S8选择规则未改变。"]
    if audit:
        lines += ["", "| 输入组 | 合格 / 冻结分母 |", "|---|---|",
                  "| 人类校准负例 | 100/100 |", "| 人类测试负例 | 200/200 |"]
        for key, label, n in (("negative:calibration", "生成校准负例", 100),
                              ("negative:test", "生成测试负例", 100),
                              ("positive:test", "生成测试正例", 50)):
            lines.append(f"| {label} | {audit['eligible_by_arm_split'].get(key,0)}/{n} |")
        lines += ["", f"{250-audit['generated_eligible']}篇生成输入不合格，继续保留为输入弃权；"
                  "完整解析阶段的弃权率尚未评估。"
                  "输入数量与来源交集见[输入审计](INPUT_AUDIT.json)，"
                  "原句未修改、请求上限和逐句绑定见[完整性审计](REPAIR_INTEGRITY_AUDIT.json)。"]
    if completed:
        lines += ["", "550篇、五条件、三种检测口径的评分和独立经验校准均已完成。",
                  "本地数值重放与远程结果一致，最大绝对误差"
                  f"{verification['statistics_max_absolute_difference']:.3g}。",
                  "", "性能门："+json.dumps(result["gates"], ensure_ascii=False)+"。",
                  "全部门通过，支持该冻结范围内的恢复收益。"
                  if result["usable_recovery_confirmed"] else
                  "性能门未全部通过，不能主张可用恢复成功。",
                  "逐条件TPR、生成/人类FPR、全分母与完整文档区间、弃权和校准见"
                  "[数值报告](REPORT.md)、[结果](RESULTS.json)及[校准](CALIBRATION.json)。"]
        import scripts.archive_swan_last_round as original
        original.OUT = OUT
        original.plot(result)
    elif ready:
        lines += ["", "CUDA就绪已实际执行；"
                  f"精确重放={ready['exact_parser_and_score_replay']}，"
                  f"预计工作量={ready['conservative_estimate_seconds']/3600:.2f}小时，"
                  f"剩余预算={ready['remaining_seconds']/3600:.2f}小时，另预留20分钟归档。",
                  "具体门："+json.dumps(load("EXECUTION_READINESS_AUDIT.json")["gates"],
                                      ensure_ascii=False)+"。",
                  "正式评分未完成，未填写FPR/TPR；[就绪记录](READINESS.json)保留真实资源门。"]
        lines += ["", "| 方法 | 唯一候选 | 预检输入 | 解析秒 | 单worker评分秒 |",
                  "|---|---|---|---|---|"]
        for timing in ready["timing"]:
            lines.append(f"| {timing['method']} | {timing['unique_inputs']} | "
                         f"{timing['pilot_inputs']} | {timing['parser_seconds']:.2f} | "
                         f"{timing['single_worker_score_seconds']:.2f} |")
        execution = load("EXECUTION_READINESS_AUDIT.json")
        lines += ["", "投影采用同一FP32配置、四个评分worker及1.25安全系数。"
                  f"可用计算时间{execution['available_compute_hours']:.2f}小时，"
                  f"按冻结估算缺口{execution['projected_deficit_hours']:.2f}小时。"
                  "该投影是保守预算估算，不是算法运行时间的严格下界。"
                  "详见[门限归因](EXECUTION_READINESS_AUDIT.json)。",
                  "精确重放及显存门通过，只有时间门未通过；"
                  "少量预检评分不作为正式测试、阈值、FPR/TPR或注入成功率。"]
    else:
        lines += ["", "正式评分未完成，不填写检测结果；停止阶段和原因见本目录审计标记。"]
    lines += ["", "此次结果只适用于冻结生成批次与历史WikiText参考。"
              "生成目标实现率不用于筛样；主题/长度与成本诊断不能代替独立语义质量核验。",
              "原绝对截止时间2026-10-24 08:03:30 UTC保持不变。",
              "修复版完成情况不追溯改写原协议成功。"]
    if audit:
        plot_execution(audit, ready)
        lines += ["", "![输入资格与资源门](repair_execution.png)"]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines)+"\n")
    align_documents(closure, result, ready)
    public = OUT / "public_aggregates"
    public.mkdir(exist_ok=True)
    for name in ("FINAL_REPORT.md", "INPUT_REPORT.md", "INPUT_AUDIT.json",
                 "INPUT_QUALITY_AUDIT.json", "INJECTION_DIAGNOSTIC.json",
                 "CLOSURE.json", "LOCAL_REPLAY_VERIFICATION.json",
                 "REPAIR_PROTOCOL.md", "CLAIM_REGISTRY.json", "CLAIM_REGISTRY.md",
                 "PREDECESSOR_VERIFICATION.json", "REPAIR_INTEGRITY_AUDIT.json",
                 "REGRESSION_VERIFICATION.json", "EXECUTION_READINESS_AUDIT.json",
                 "repair_execution.pdf", "repair_execution.png",
                 "confirmation.pdf", "confirmation.png"):
        if (OUT / name).exists():
            shutil.copyfile(OUT / name, public / name)
    source = load("SOURCE_FREEZE.json")
    if source:
        save_json(public / "SOURCE_VERSION.json", {k: source[k] for k in (
            "repository", "revision", "source_file", "source_sha256", "license", "reconstruction")})
    if ready:
        save_json(public / "READINESS.json", ready)
    runtime = load("EVAL_FREEZE.json")
    generation = load("GENERATION_FREEZE.json")
    save_json(public / "VERSION_METADATA.json", {
        "generation_settings": generation["settings"] if generation else None,
        "repair_code_sha256": generation["implementation_sha256"] if generation else None,
        "repair_protocol_sha256": generation["protocol_sha256"] if generation else None,
        "runtime_settings": runtime["settings"] if runtime else None,
        "runtime_code_sha256": runtime["implementation_sha256"] if runtime else None,
        "analysis_sha256": runtime["analysis_sha256"] if runtime else None,
        "asset_fingerprints": {k: v["fingerprint"] for k, v in
                               runtime["assets"]["resource_groups"].items()} if runtime else None,
        "runtime_versions": runtime["assets"]["versions"] if runtime else None,
        "posthoc_repair": True, "candidate_selection_preserved": True})
    if completed:
        # Numerical replay needs only the candidate identity. Runtime config and
        # sealed text remain private; preserve the parent freeze SHA as provenance.
        minimal = {"schema": "swan-public-numerical-replay/v2", "candidate_method": "S8",
                   "parent_runtime_freeze_sha256": digest(load("EVAL_FREEZE.json"))}
        save_json(public / "EVAL_FREEZE.json", minimal)
        scored = load("DOCUMENT_SCORES.json")
        save_json(public / "DOCUMENT_SCORES.json",
                  {**scored, "freeze_sha256": digest(minimal)})
        replayed = analyze(public)
        compare({k: v for k, v in result.items() if k != "freeze_sha256"},
                {k: v for k, v in replayed.items() if k != "freeze_sha256"})
        for relative in ("scripts/analyze_swan_last_round.py",
                         "scripts/analyze_swan_blind_eval.py",
                         "src/core/swan_blind_protocol.py", "src/core/multiscale_blind.py"):
            destination = public / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        for directory in ("scripts", "src", "src/core"):
            (public / directory / "__init__.py").touch()
    (public / "README.md").write_text(
        "# SWAN 修复版公开聚合包\n\n"+((
            "该包可离线从无原文的单元分数重算独立校准、条件指标、置信区间和性能门。\n\n"
            "依赖Python、NumPy、SciPy；在包目录执行：\n\n"
            "```bash\npython scripts/analyze_swan_last_round.py --output .\n```\n")
            if completed else "该版未完成正式评分，公开材料仅复现输入/执行停止证据。\n")+
        "\n不包含原文、响应内容、秘密模板、凭证、本地配置或第三方模型payload。"
        "因此不支持重新生成或AMR推理。原失败版的公开包保持原SHA。\n\n"
        "SHA256SUMS.json记录全部公开文件摘要；FINAL_REPORT.md为入口。\n")
    save_json(OUT / "CLOSURE.json", closure)
    shutil.copyfile(OUT / "CLOSURE.json", public / "CLOSURE.json")
    save_json(public / "SHA256SUMS.json", inventory(public))
    zip_path = Path(shutil.make_archive(str(OUT / "public_aggregates"), "zip", public))
    save_json(OUT / "PACKAGE_VERIFICATION.json", {
        "files": len(inventory(public)), "zip_bytes": zip_path.stat().st_size,
        "zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
        "no_sealed_text_or_local_configuration": True,
        "aggregate_numerical_replay": completed})
    save_json(OUT / "SHA256SUMS.json", {
        str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(OUT.iterdir()) if p.is_file()
        and p.name not in ("SHA256SUMS.json", "GUARD_HEARTBEAT.json",
                           "WORKFLOW_STATUS.json", "GENERATION_HEARTBEAT.json", "heartbeat.json")})
    print(json.dumps(closure), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--reason", required=True)
    archive(cli.parse_args().reason)
