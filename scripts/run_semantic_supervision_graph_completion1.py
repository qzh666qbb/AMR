"""Fresh, target-free full AMR for all frozen supervision candidates/windows."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_multiscale_transfer as runtime
from semantic_fact_raw_amr import audit_raw, replay_signature

P2 = ROOT / "第二篇_语义消息写入与读回"
PARENT = P2 / "02_实验/semantic_multiscale_supervision2"
SOURCE = P2 / "08_本地原始资产/cache_amr/semantic_multiscale_supervision2"
DEFAULT = PARENT / "graph_completion1"
CODE = (
    "scripts/run_semantic_supervision_graph_completion1.py",
    "scripts/run_multiscale_transfer.py", "scripts/semantic_fact_raw_amr.py",
    "scripts/run_research_completion_queue.py", "scripts/test_semantic_supervision_graph_completion1.py",
    "src/core/parser_recovery_audit.py",
)


def immutable(path, value):
    if path.exists():
        if runtime.read(path) != value:
            raise RuntimeError("Immutable artifact differs: " + path.name)
    else:
        runtime.atomic(path, value)


def workload(request):
    if set(request) != {"alphabet_version", "core_fact_matcher_implemented",
                        "parser_calls", "views", "documents"}:
        raise ValueError("Unexpected request fields")
    if request["views"] != ["R1_full", "R2_full", "R3_core"]:
        raise ValueError("Frozen three-view definition differs")
    unique, occurrences, keys = {}, [], set()
    for doc in request["documents"]:
        if set(doc) != {"candidate_status", "candidates", "contract", "key", "text", "text_sha256"}:
            raise ValueError("Only blind document fields allowed")
        if doc["key"] in keys or runtime.digest(doc["text"]) != doc["text_sha256"]:
            raise ValueError("Document identity mismatch")
        keys.add(doc["key"])
        if set(doc["contract"]) != {"role_one", "role_two", "object"}:
            raise ValueError("Only public contract allowed")
        for ci, candidate in enumerate(doc["candidates"]):
            if set(candidate) != {"anchor_span", "anchor_text", "contexts"}:
                raise ValueError("Unexpected candidate fields")
            a, b = candidate["anchor_span"]
            if not 0 <= a < b <= len(doc["text"]) or doc["text"][a:b] != candidate["anchor_text"]:
                raise ValueError("Candidate anchor differs")
            if len(candidate["contexts"]) != 3:
                raise ValueError("Missing context view")
            for vi, context in enumerate(candidate["contexts"]):
                if set(context) != {"requested_sentences", "span", "text"}:
                    raise ValueError("Unexpected context fields")
                lo, hi = context["span"]
                if (context["requested_sentences"] != vi + 1 or not 0 <= lo <= a < b <= hi <= len(doc["text"])
                        or doc["text"][lo:hi] != context["text"]):
                    raise ValueError("Window differs from literal frozen text")
                digest = runtime.digest(context["text"])
                row = {"case_id": digest, "input_text": context["text"],
                       "input_text_sha256": digest}
                if digest in unique and unique[digest] != row:
                    raise ValueError("Text hash collision")
                unique[digest] = row
                occurrences.append({"key": doc["key"], "candidate_index": ci,
                                    "view_index": vi, "text_sha256": digest,
                                    "context_span": [lo, hi], "anchor_span": [a, b]})
    return {"unique_texts": [unique[k] for k in sorted(unique)], "occurrences": occurrences,
            "documents": len(keys), "candidates": len(occurrences) // 3}


def prepare(out):
    if out.exists():
        raise RuntimeError("Choose a new successor directory")
    parent = runtime.read(PARENT / "FREEZE.json")
    checked = {}
    for name, expected in parent["private_sha256"].items():
        if runtime.sha(SOURCE / name) != expected:
            raise RuntimeError("Original private input drift")
        checked[name] = expected
    for name, key in (("SPLITS.json", "split_sha256"), ("MEMBERSHIP.json", "membership_sha256"),
                      ("PROTOCOL.md", "protocol_sha256")):
        if runtime.sha(PARENT / name) != parent[key]:
            raise RuntimeError("Original supervision contract drift")
    for old, expected in parent["input_sha256"].items():
        source = P2 / "08_本地原始资产/cache_amr" / old.split("/.cache/amr/", 1)[1]
        if runtime.sha(source) != expected:
            raise RuntimeError("Original generation asset drift")
    request = runtime.read(SOURCE / "FEATURE_REQUEST_NO_TARGET.json")
    work = workload(request)
    if (work["documents"], work["candidates"]) != (192, 1320):
        raise RuntimeError("Full frozen denominator required")
    private = out / "private"
    private.mkdir(parents=True, mode=0o700)
    (out / ".gitignore").write_text("private/\n*.log\n")
    shutil.copyfile(SOURCE / "FEATURE_REQUEST_NO_TARGET.json", private / "REQUEST.json")
    for name in ("SPLITS.json", "MEMBERSHIP.json"):
        shutil.copyfile(PARENT / name, out / name)
    for name in CODE:
        dest = private / "code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    runtime.atomic(private / "WORKLOAD.json", work)
    transfer = P2 / "02_实验/semantic_sequence_stage1_run2/multiscale_transfer4_recovery1"
    parser = runtime.read(transfer / "FREEZE.json")["parser"]
    protocol = """# 新事实监督：完整原始图补全

固定原192文本、1320候选、每候选1/2/3句上下文。逐字SHA核验，不重切候选。
同文本只解析一次，但所有3960候选视图及重复视图关系保留。
使用冻结BART、FP32、4束、原生1024边界；全原始PENMAN保留，不先切事件子图。
beam0、无恢复日志、无截断且roundtrip一致才作为可用图。
绑定的解析异常/原生长度超限记擦除，保留全部分母；未知日志/身份漂移永久停止此版本。
单GPU进程、新缓存。显存/OOM/磁盘不足原子保存后暂停恢复，无总时间截断。
先解析及封存全部图；不加载计划符号、真人标签、助手答案或任何旧特征缓存。
这一步不是模板得分或训练。新24模板仅来自训练场景的真人实际事实；
模板、共同匹配器、核心合法性、R3裁剪和特征要另行封存。
旧开发test已经暴露，不得升级为独立确认。该任务不产生FPR/TPR或真人语义判断。
"""
    (out / "PROTOCOL.md").write_text(protocol)
    runtime.atomic(out / "FREEZE.json", {
        "version": "semantic-supervision-raw-graph-completion/v1",
        "parent_freeze_sha256": runtime.sha(PARENT / "FREEZE.json"),
        "original_inputs_verified": checked, "original_generation_files_verified": len(parent["input_sha256"]),
        "parser": parser, "counts": {"documents": 192, "candidates": 1320, "view_occurrences": 3960,
                                   "unique_texts": len(work["unique_texts"])},
        "files": {str(p.relative_to(out)): runtime.sha(p) for p in sorted(out.rglob("*"))
                  if p.is_file() and p.name != ".gitignore"},
        "execution": {"resumable_gpu_pressure": True, "single_parser": True,
                      "fresh_cache": True, "resource_pause_exit_code": 75},
        "resource_gate": {"vram_fraction_max": 0.8},
        "target_access": False, "human_labels_accessed": False, "feature_release": False,
    })
    runtime.atomic(out / "VALIDATION.json", {"frozen_counts": runtime.read(out / "FREEZE.json")["counts"],
                                            "inputs_verified": True, "cuda_run": False})


def verify(out):
    f = runtime.read(out / "FREEZE.json")
    for name, expected in f["files"].items():
        if runtime.sha(out / name) != expected:
            raise RuntimeError("Frozen artifact drift: " + name)
    work = workload(runtime.read(out / "private/REQUEST.json"))
    if work != runtime.read(out / "private/WORKLOAD.json"):
        raise RuntimeError("Workload identity differs")
    return f, work


def native_erasure(row, freeze_sha, tokens):
    return {"amr": None, "amr_sha256": None, "valid": False, "case_id": row["case_id"],
            "text_sha256": row["input_text_sha256"], "token_length": tokens, "events": [],
            "attempts": [], "decode_events": [], "beam_lengths": [], "clipped": False,
            "seconds": 0., "freeze_sha256": freeze_sha, "reason": "native_input_limit"}


def run(out, assets):
    f, work = verify(out)
    if (out / "HALT.json").exists():
        raise RuntimeError("Integrity halt cannot resume")
    if (out / "RAW_GRAPH_SEAL.json").exists():
        seal = runtime.read(out / "RAW_GRAPH_SEAL.json")
        if seal["freeze_sha256"] != runtime.sha(out / "FREEZE.json"):
            raise RuntimeError("Output freeze changed")
        for name, expected in seal["files"].items():
            if runtime.sha(out / name) != expected:
                raise RuntimeError("Sealed graph differs")
    else:
        import torch
        parser = runtime.Parser(assets, out, f)
        prechecks = []
        for row in work["unique_texts"][:2]:
            pair = [{**parser.fresh(row["input_text"]), "case_id": row["case_id"]} for _ in range(2)]
            if replay_signature(pair[0]) != replay_signature(pair[1]):
                raise RuntimeError("Raw graph two-replay mismatch")
            prechecks.append({"text_sha256": row["case_id"], "replay_identical": True,
                              "structurally_eligible": audit_raw(pair[0], row, parser.freeze_sha)["structurally_eligible"]})
        runtime.atomic(out / "GPU_PRECHECK.json", {
            "passed": True, "replays": prechecks, "dtype": "float32",
            "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)})
        for i, row in enumerate(work["unique_texts"]):
            if shutil.disk_usage(out).free < 512 * 1024**2:
                raise runtime.ResourcePause("disk_headroom_below_512MiB")
            path = out / "private/raw_graphs" / (row["case_id"] + ".json")
            if path.exists():
                record = runtime.read(path)
                audit_raw(record, row, parser.freeze_sha)
            else:
                tokens = len(parser.model.tokenizer(row["input_text"], truncation=False)["input_ids"])
                record = (native_erasure(row, parser.freeze_sha, tokens) if tokens >= parser.model.max_sent_len
                          else {**parser.fresh(row["input_text"]), "case_id": row["case_id"]})
                audit_raw(record, row, parser.freeze_sha)
                immutable(path, record)
            runtime.atomic(out / "HEARTBEAT.json", {
                "stage": "raw_graphs", "completed_unique_texts": i + 1,
                "total_unique_texts": len(work["unique_texts"]), "at": runtime.now()})
        immutable(out / "RAW_GRAPH_SEAL.json", {
            "freeze_sha256": parser.freeze_sha, "target_access": False, "human_label_access": False,
            "files": {str(p.relative_to(out)): runtime.sha(p)
                      for p in sorted((out / "private/raw_graphs").glob("*.json"))}})
    audits = {}
    for row in work["unique_texts"]:
        record = runtime.read(out / "private/raw_graphs" / (row["case_id"] + ".json"))
        audits[row["case_id"]] = audit_raw(record, row, runtime.sha(out / "FREEZE.json"))
    result = {
        "status": "RAW_GRAPH_COMPUTATION_COMPLETE_FEATURES_PENDING",
        "counts": f["counts"], "eligible_unique_graphs": sum(r["structurally_eligible"] for r in audits.values()),
        "eligible_view_occurrences": sum(audits[r["text_sha256"]]["structurally_eligible"] for r in work["occurrences"]),
        "exclusion_reasons_unique": dict(Counter(reason for r in audits.values() for reason in r["exclusion_reasons"])),
        "training_runs": 0, "human_labels": 0, "formal_FPR": None, "formal_TPR": None,
    }
    immutable(out / "RESULT.json", result)
    immutable(out / "AUDIT.json", {"full_denominator": len(audits) == f["counts"]["unique_texts"],
                                  "all_3960_views_retained": len(work["occurrences"]) == 3960,
                                  "labels_accessed": False, "feature_or_human_closure": False})
    immutable(out / "TERMINAL.json", {"status": result["status"],
                                     "result_sha256": runtime.sha(out / "RESULT.json"),
                                     "audit_sha256": runtime.sha(out / "AUDIT.json")})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare", "verify", "run"))
    ap.add_argument("--out", type=Path, default=DEFAULT)
    ap.add_argument("--assets", type=Path, default=ROOT)
    args = ap.parse_args()
    if args.action == "prepare":
        prepare(args.out)
    elif args.action == "verify":
        verify(args.out)
    else:
        with (args.out / "private/worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            start = time.monotonic()
            try:
                run(args.out, args.assets)
            except Exception as exc:
                pause = isinstance(exc, runtime.ResourcePause) or type(exc).__name__ == "OutOfMemoryError"
                runtime.atomic(args.out / ("PAUSED.json" if pause else "HALT.json"), {
                    "type": type(exc).__name__, "at": runtime.now(), "resume_allowed": pause,
                    "scientific_result": None})
                if pause:
                    sys.exit(75)
                raise
            finally:
                runtime.atomic(args.out / "private/sessions" / (str(time.time_ns()) + ".json"),
                               {"elapsed_seconds": time.monotonic() - start, "at": runtime.now()})
    print(json.dumps({"action": args.action, "ok": True}))


if __name__ == "__main__":
    main()
