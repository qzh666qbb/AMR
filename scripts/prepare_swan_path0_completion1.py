"""Assemble an isolated, hash-bound path0 runtime without modifying old evidence."""
import ast
import json
from pathlib import Path
import shutil
import sys
import tarfile

import run_swan_path0_completion1 as run
from swan_final1_common import freeze, load, now, sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/swan_path0_completion1"
SNAPSHOT = Path("/tmp/swan-path0-completion1-runtime")
PARENT = ROOT / "archive/experiments/swan_elastic_v4_path0_development3"


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256(source) != sha256(target):
        raise ValueError("Existing snapshot differs: " + str(target))
    if not target.exists():
        shutil.copy2(source, target)


def core_snapshot():
    pending = ["__init__", "elastic_v4_path0_development", "elastic_v4_path0_inputs",
               "elastic_v4_path0_preflight"]
    selected = set()
    original = load(PARENT / "RUNTIME_MANIFEST.json")["code_sha256"]
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        selected.add(name)
        path = ROOT / "src/core" / (name + ".py")
        if name.startswith("elastic_v4_path0"):
            path = ROOT / "archive/src_core" / (name + ".py")
        relative = "src/core/" + name + ".py"
        if sha256(path) != original[relative]:
            raise ValueError("Original dependency differs: " + name)
        copy(path, SNAPSHOT / relative)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                pending.append(node.module)
    for relative in ("src/__init__.py", "src/core/__init__.py"):
        copy(ROOT / relative, SNAPSHOT / relative)
    return sorted(selected)


def main():
    if (OUTPUT / "FREEZE.json").exists():
        raise RuntimeError("Already frozen; use the existing package")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in ("PROTOCOL.md", "INPUT_MANIFEST.json", "LOCAL_HISTORY.json", "REMOTE_HISTORY.json",
                 "sealed/calibration.jsonl", "sealed/null.jsonl", "sealed/positive_source.jsonl",
                 "sealed/source_rows.jsonl"):
        copy(PARENT / name, OUTPUT / name)
    copy(ROOT / "archive/experiments/swan_elastic_v4_path0_development1/PROTOCOL.md",
         OUTPUT / "ORIGINAL_SCIENTIFIC_PROTOCOL.md")
    core = core_snapshot()
    names = {p for p in run.generation.pilot.DEPENDENCIES if p.startswith("scripts/")}
    names.update("scripts/" + p for p in (
        "run_swan_supplement2.py", "run_swan_supplement4.py", "swan_supplement4_runtime.py",
        "run_swan_path0_completion1.py", "test_swan_path0_completion1.py",
        "test_swan_supplement4.py", "run_research_completion_queue.py",
        "prepare_swan_path0_completion1.py"))
    for relative in sorted(names):
        copy(ROOT / relative, SNAPSHOT / relative)
    # Immutable shared assets use previously verified hashes; deployment checks every byte.
    previous = load(ROOT / "experiments/swan_count_neutral_supplement4/FREEZE.json")
    assets = {p: h for p, h in previous["dependencies"].items()
              if p.startswith(("models/", "third_party/", "vectors/", ".nltk_data/"))}
    for path, expected in load(PARENT / "RUNTIME_MANIFEST.json")["model_asset_sha256"].items():
        relative = path.split("/workspace/swan-elastic-reconstruct/", 1)[1]
        if relative in assets and assets[relative] != expected:
            raise ValueError("Model identity differs from original path0")
        assets[relative] = expected
    bank_path = "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_100.json"
    assets[bank_path] = sha256(ROOT / bank_path)
    if assets[bank_path] != "7e51393dacc5473ac9b3d96a12ba2a2591503432277ede174a125864e755a777":
        raise ValueError("Bank100 identity differs")
    sources = [json.loads(line) for line in (OUTPUT / "sealed/positive_source.jsonl").read_text().splitlines()]
    bank = run.read_bank()
    freeze(OUTPUT / "GENERATION_ASSIGNMENT.json", run.assignment(sources, bank))
    (OUTPUT / "RUN_AMENDMENT.md").write_text(
        "# path0 同源补全后继\n\n2026-11-02，用户授权“都补一补吧”。原 Stop、旧 Null 结果和来源保持。\n\n"
        "PROTOCOL.md 是原 INPUT_MANIFEST 绑定的 development2 工程协议，逐字保留；"
        "ORIGINAL_SCIENTIFIC_PROTOCOL.md 是其引用的完整科学协议。\n\n"
        "旧逐行结果及压缩候选证据不完整，因此新缓存重算1000校准、200×6 Null及50×6正例。"
        "核心依赖逐字等于原runtime；两臂共享AMR，Bank100、batch16/request128、FP32、"
        "hops3/span80/capgap12/longgap40、center .65、green .75、p2=p3=.18不变。TF32关闭。\n\n"
        "先执行全部50个正例生成，再开始检测。四层固定2句×9–12词、3句×9–10词、"
        "4句×9–10词、4句×11–12词，共最多161句。每句官方10模板×5尝试，"
        "S2Match≥.75，50输出tokens、temperature .7、top_p .9。模板抽样沿用已验证官方RNG，"
        "全部候选模板索引及长度提示在GENERATION_ASSIGNMENT.json中预封。"
        "deepseek-flash，最多8050次请求，累计25美元保护，保守最多22.73964美元；"
        "记录usage估算，不将其称为已核账单。没有历史调用复用。\n\n"
        "首个耗尽句槽使该文档自动资格失败，继续下一冻结来源，全部失败保留50篇分母；"
        "不按检测结果重新生成。所有生成完毕后两臂分别从row0校准，保留六个alpha曲线和完整证据。"
        "自动资格至少40/50仅为工程样本充分性；独立真人质量缺失时不作最终开发Go。"
        "两名独立真人盲审grammar/coherence/no-AMR-leak，双方三项均通过才为真人质量通过；"
        "全部50篇提供材料，缺失/分歧保留分母并记未通过，不能由模型代填。\n\n"
        "CUDA准入低于3072MiB、parse余量低于512MiB、OOM或磁盘余量低于1GiB保存断点exit75。"
        "这不是方法失败；磁盘须等待可用资源，不删旧数据。"
        "只把同步绑定到候选的生成parser/normalizer/template诊断记拒绝；"
        "检测沿用原恢复审计及非法边剔除，无法绑定的诊断或未知消费永久halt。"
        "所有已付响应原样重放，HTTP重试0。没有72小时或整卡80%永久停止门。\n"
    )
    (OUTPUT / "HUMAN_REVIEW.md").write_text(
        "# 独立质量回收\n\n生成结束自动提供private/human_review/PACKET.json及两份空CSV。"
        "每位评审独立阅读语境与文本，判断英语语法完整、主题连贯、无AMR/占位符泄露，"
        "按0/1填三项，并填写真实reviewer_id、is_human=true及理由。"
        "不得读取HUMAN_LINKAGE、目标Bank、模型分数或另一人的答卷。"
        "不要求复述语境原有事实；生成允许合理的虚构细节。"
        "不存在答卷时保持pending，自动门不得充作真人结论。\n"
    )
    for path in OUTPUT.rglob("*"):
        if path.is_file():
            copy(path, SNAPSHOT / "experiments/swan_path0_completion1" / path.relative_to(OUTPUT))
    dependencies = {str(p.relative_to(SNAPSHOT)): sha256(p)
                    for p in SNAPSHOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    dependencies.update(assets)
    freeze(OUTPUT / "FREEZE.json", {
        "at": now(), "experiment": OUTPUT.name, "dependencies": dependencies,
        "user_instruction": "都补一补吧", "old_evidence_preserved": True,
        "source_replacement": False, "fresh_cache_required": True,
        "original_core_dependencies": core,
        "generator": run.budget.MODEL, "generation_max_calls": run.CALL_CAP,
        "generation_max_nanodollars": run.COST_CAP,
        "generation_worst_reserve_nanodollars": run.CALL_CAP * run.budget.MAX_RESERVE,
        "positive_denominator": 50, "calibration_sources": 1000, "null_sources": 200,
        "human_review_is_external": True, "detection_after_all_generation": True,
    })
    copy(OUTPUT / "FREEZE.json", SNAPSHOT / "experiments/swan_path0_completion1/FREEZE.json")
    archive = Path("/tmp/swan-completion-w03-20261102.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(SNAPSHOT.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                tar.add(path, arcname=str(path.relative_to(SNAPSHOT)), recursive=False)
    print(json.dumps({"status": "prepared", "core_dependencies": len(core),
                      "freeze_sha256": sha256(OUTPUT / "FREEZE.json"),
                      "archive": str(archive), "archive_sha256": sha256(archive),
                      "archive_bytes": archive.stat().st_size, "new_calls": 0}))


if __name__ == "__main__":
    main()
