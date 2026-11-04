"""Create a fresh successor from verified frozen inputs, never from result caches."""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime
import importlib.util
from pathlib import Path
import shutil


def prepare(root, parent, out, runner, expected_parent_sha):
    spec = importlib.util.spec_from_file_location("recovery_runner", runner)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.sha(parent / "FREEZE.json") == expected_parent_sha
    old = m.read(parent / "FREEZE.json")
    m.verify(parent, old)
    assert (parent / "HALT.json").exists()
    assert not out.exists(), "Successor identity must be new"
    old_source = parent / "private/code/scripts/run_multiscale_transfer.py"
    old_nodes = {n.name: ast.dump(n, include_attributes=False)
                 for n in ast.parse(old_source.read_text()).body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    new_nodes = {n.name: ast.dump(n, include_attributes=False)
                 for n in ast.parse(runner.read_text()).body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    allowed_changes = {"Parser", "work", "main"}
    unchanged = sorted(set(old_nodes) - allowed_changes)
    assert all(old_nodes[n] == new_nodes[n] for n in unchanged), "Scientific code changed"
    # Bound earlier missing ledgers conservatively using the whole frozen-run
    # interval for both the primary worker and its one possible helper.
    halt = m.read(parent / "HALT.json")
    duration = (datetime.fromisoformat(halt["at"]) -
                datetime.fromisoformat(old["frozen_at"])).total_seconds()
    assert duration > 0
    prior = old["resource_gate"]["prior_gpu_seconds"] + 2 * duration
    out.mkdir(parents=True)
    (out / "private").mkdir()
    (out / ".gitignore").write_text("private/\n*.log\n")
    files = {"private/DOCUMENTS.json", "private/BANK.json", "private/propbank_verbs.json",
             "WORKLOAD.json", "INPUT_INVENTORY.json"}
    files.update(old["sources"])
    files.update(old["normalizer_assets"])
    files.update(row["file"] for row in old["checkpoints"])
    for name in sorted(files):
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / name, target)
    shutil.copy2(runner, out / "private/code/scripts/run_multiscale_transfer.py")
    freeze = copy.deepcopy(old)
    freeze["version"] = "new-text-multiscale-transfer/v4-resource-recovery1"
    freeze["frozen_at"] = m.now()
    freeze["sources"] = {name: m.sha(out / name) for name in old["sources"]}
    freeze["resource_gate"]["prior_gpu_seconds"] = prior
    freeze["execution"] = {
        "resumable_gpu_pressure": True, "parser_processes": 1,
        "prefill_helpers": 0, "model_reservation_mib": 3072, "parse_headroom_mib": 512,
        "gpu_pressure_exit_code": 75,
        "resource_pressure": "Checkpoint, release process, wait for admission, resume",
        "unbound_parser_warning": "Unchanged permanent HALT",
        "resource_budget": "Unchanged 12 cumulative conservative GPU hours / 24 hours",
    }
    science_keys = ["split", "documents_sha256", "original_generation_manifest_sha256",
                    "checkpoints", "bank_sha256", "bank_order", "normalizer_assets",
                    "propbank_verbs_sha256", "parser", "scorer", "features",
                    "calibration", "clean_gate", "stage_order", "conditions",
                    "no_generation", "no_score_filtering"]
    assert all(freeze[key] == old[key] for key in science_keys)
    m.atomic(out / "FREEZE.json", freeze)
    m.atomic(out / "RECOVERY_LINEAGE.json", {
        "at": m.now(), "parent_freeze_sha256": expected_parent_sha,
        "parent_halt_sha256": m.sha(parent / "HALT.json"),
        "prior_gpu_seconds_upper_bound": prior,
        "prior_accounting": "Earlier recorded time plus twice predecessor freeze-to-HALT interval; "
                            "includes helper conservatively, not a measured CUDA runtime",
        "parent_interval_seconds": duration, "scientific_fields_unchanged": science_keys,
        "unchanged_code_nodes": unchanged,
        "new_runner_sha256": m.sha(runner), "preparation_sha256": m.sha(__file__),
        "predecessor_parse_or_score_cache_imported": False,
        "parent_halt_preserved": True,
    })
    protocol = (parent / "PROTOCOL.md").read_text()
    (out / "PROTOCOL.md").write_text(
        "# 2026-11-02 资源恢复后继运行\n\n"
        "用户要求继续完成因显存门中断的迁移实验。原HALT保持原样；"
        "本目录使用同一24对文本、13个模型、Bank、FP32/batch1/4 beams、"
        "评分、分区和校准规则，不导入前驱计算缓存，不生成或补样。\n\n"
        "仅运行一个解析进程，不启动prefill。整卡80%作为资源准入线："
        "资源不足或CUDA OOM保存暂停记录，释放本任务进程后等待资源；"
        "不标记科学失败，不停止其他任务。无法归属的解析诊断仍永久HALT。"
        "沿用12小时累计保守计算预算，计入前驱主进程和helper时间上界；"
        "不会因当前用户希望跑完而静默取消预算或科学门。\n\n"
        "完整结果只在COMPLETE和独立FINAL_AUDIT通过后报告。"
        "预期结果未知；该小样本协议没有足够独立文档用于1%/5%文档FPR。\n\n"
        "以下科学协议继承前驱：\n\n" + protocol)
    m.verify(out, freeze)
    print({"prepared": True, "freeze_sha256": m.sha(out / "FREEZE.json"),
           "prior_conservative_hours": prior / 3600,
           "science_fields_identical": len(science_keys), "models": len(freeze["checkpoints"]),
           "cached_results_imported": 0})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runner", type=Path, required=True)
    ap.add_argument("--expected-parent-sha", required=True)
    a = ap.parse_args()
    prepare(a.root, a.parent, a.out, a.runner, a.expected_parent_sha)
