"""Freeze and package the authorized successor with an actual CUDA parser worker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_semantic_fact_carrier_pilot2 as v2
from semantic_fact_raw_amr import validate_inputs

p = v2.runtime
DEFAULT_OUT = ROOT / "experiments/semantic_fact_layers_v3_parse1"
SOURCES = [
    "scripts/prepare_semantic_fact_layers_v3.py",
    "scripts/run_semantic_fact_layers_v3_gpu.py",
    "scripts/run_semantic_fact_layers_v3.py",
    "scripts/semantic_fact_raw_amr.py",
    "scripts/run_multiscale_transfer.py",
    "src/core/parser_recovery_audit.py",
    "src/core/semantic_fact_layers.py",
    "src/core/semantic_fact_carrier.py",
    "tests/test_semantic_fact_layers.py",
]
GPU_SOURCES = [
    "scripts/run_semantic_fact_layers_v3_gpu.py",
    "scripts/semantic_fact_raw_amr.py",
    "scripts/run_multiscale_transfer.py",
    "src/core/parser_recovery_audit.py",
]


def prepare(out, package):
    if out.exists():
        raise RuntimeError("Use a new version; never overwrite an existing freeze")
    p.verify(v2.OUT, v2.PRIVATE)
    source = v2.OUT / "private_review/fact_layers"
    model = v2.OUT / "private_review/model_review"
    seals = [(source, "INPUT_SEAL.json"), (model, "ANNOTATION_SEAL.json")]
    for folder, name in seals:
        seal = p.load(folder / name)
        for filename, expected in seal["files"].items():
            if p.sha(folder / filename) != expected:
                raise RuntimeError("Sealed input or model-reference drift")
    data = p.load(source / "PARSER_INPUTS_NO_TARGET.json")
    rows = validate_inputs(data)
    protected = p.load(v2.PRIVATE / "V1_PROTECTED_SHA256.json")
    for name, expected in protected.items():
        if p.sha(ROOT / name) != expected:
            raise RuntimeError("Historical v1 protected source drift")
    # Capture all current public v2 and earlier execution bytes, in addition to
    # their existing seals. Private source files stay protected by the seals.
    historical = dict(protected)
    for folder in (v2.OUT, ROOT / "experiments/semantic_fact_layers_v2_execution1"):
        for path in folder.rglob("*"):
            if path.is_file() and "private_review" not in path.parts:
                historical[str(path.relative_to(ROOT))] = p.sha(path)
    out.mkdir(parents=True)
    (out / ".gitignore").write_text("private/\nprivate_review/\n*.log\n")
    private = out / "private"
    private.mkdir(mode=0o700)
    code = private / "code"
    for name in SOURCES:
        destination = code / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    shutil.copy2(source / "PARSER_INPUTS_NO_TARGET.json", private / "PARSER_INPUTS_NO_TARGET.json")
    p.atomic(private / "HISTORICAL_SHA256.json", historical)
    protocol = (
        "# 192条原AMR生成与归因重跑\n\n"
        "新版本仅执行用户明确授权的开发解析，保持载体v2的192条原文、ID、"
        "模型参考标签与真人门限不变。每个输入独立解析一次；前两条各重放一次"
        "用于确定性检查，共194次推理，无缓存替代、补样或文字改写。\n\n"
        "解析器只接收原文字节及不透明ID/SHA；整段固定位置输入，不带目标、"
        "模型事实、计划或分类标签。整段多句AMR的事实完整性尚未验证。\n\n"
        "Linux CUDA RTX3090、BART amrlib、FP32、batch=1、4 beams，原生1024"
        "上下文。禁止截断、FP16、量化、detector评分和图重写。资产、代码及"
        "输入先做SHA核验；每条原子落盘，Supervisor与flock托管。"
        "全局显存占用须低于80%，时间预算7200秒。\n\n"
        "未绑定warning/异常永久停止并保存inflight证据；已绑定warning、"
        "恢复到非零beam、空图或无效roundtrip输出保留但排除合格图集合，"
        "仍保留全192条分母。已知torch.load FutureWarning只在加载阶段"
        "单独记录。重放SHA不一致立即停止。\n\n"
        "原AMR存在不等于已读出图事实。原图与独立修订图的事实证据、"
        "ABCD输出和真人声明继续按实际状态记录；原归因函数对全部192条"
        "真实调用，禁止伪造确认。此轮不运行256消息盲恢复。\n"
    )
    (out / "PROTOCOL.md").write_text(protocol, encoding="utf-8")
    freeze = {
        "version": "semantic-fact-layers/v3-raw-amr-1", "at": p.now(),
        "authorization": "User: 修复准备脚本，调用解析器生成原 AMR，然后重新运行 192 条输入与归因代码",
        "scope": "Authorized developmental raw AMR; formal human gate remains unconfirmed",
        "rows": 192, "unique_texts": len({r["input_text_sha256"] for r in rows}),
        "parser": p.load(ROOT / "experiments/semantic_sequence_readout1/FREEZE.json")["parser"],
        "parser_settings": {"dtype": "float32", "batch_size": 1, "num_beams": 4,
                            "native_context": 1024, "tf32": False, "cache_used": False},
        "input_mode": "whole_paragraph_fixed_window",
        "multi_sentence_semantic_completeness_validated": False,
        "input_sha256": p.sha(private / "PARSER_INPUTS_NO_TARGET.json"),
        "source_input_seal_sha256": p.sha(source / "INPUT_SEAL.json"),
        "source_annotation_seal_sha256": p.sha(model / "ANNOTATION_SEAL.json"),
        "historical_sha_manifest_sha256": p.sha(private / "HISTORICAL_SHA256.json"),
        "source_sha256": {name: p.sha(ROOT / name) for name in SOURCES},
        "gpu_code_sha256": {name: p.sha(ROOT / name) for name in GPU_SOURCES},
        "protocol_sha256": p.sha(out / "PROTOCOL.md"),
        "replay_case_ids": [r["case_id"] for r in rows[:2]], "extra_replay_calls": 2,
        "resource_gate": {"seconds": 7200, "total_vram_fraction_strictly_below": 0.8},
        "no_provider_detector_embedding_or_blind_decode": True,
        "human_attestations_may_not_be_changed": True,
    }
    p.atomic(out / "FREEZE.json", freeze)
    package.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(package, "w:gz") as tar:
        for name in ("FREEZE.json", "PROTOCOL.md", "private/PARSER_INPUTS_NO_TARGET.json"):
            tar.add(out / name, arcname=name)
        for name in GPU_SOURCES:
            tar.add(code / name, arcname="private/code/" + name)
    package.chmod(0o600)
    p.atomic(out / "PREPARATION.json", {
        "at": p.now(), "status": "FROZEN_ACTUAL_PARSER_WORKER_READY",
        "rows": 192, "package_sha256": p.sha(package),
        "freeze_sha256": p.sha(out / "FREEZE.json"),
        "target_or_reference_labels_in_gpu_package": False,
        "historical_protected_files": len(historical), "new_parser_calls": 0,
    })
    print(json.dumps({"status": "FROZEN", "rows": 192, "gpu_package_sha256": p.sha(package)}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--package", type=Path, default=Path("/tmp/semantic_fact_layers_v3_parse1.tar.gz"))
    args = ap.parse_args()
    prepare(args.out, args.package)
