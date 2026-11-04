"""Complete the original 24-template paraphrase diagnostic in a new cache.

The scoring stage consumes text and the public dictionary only. Intended symbols
and model-assisted spans are read only after the sentence predictions are sealed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import run_multiscale_transfer as runtime
import run_symbol_sequence_diagnostic as diagnostic
from src.core.semantic_sequence import (
    equal_composition_sequences, exact_costs, rank_truth, unordered_costs,
)

P2 = ROOT / "第二篇_语义消息写入与读回"
BASE = P2 / "02_实验/semantic_sequence_stage1_run2"
DEFAULT = BASE / "symbol_paraphrase_completion1"


def frozen(path, value):
    if path.exists():
        if runtime.read(path) != value:
            raise RuntimeError("Immutable record differs: " + path.name)
    else:
        runtime.atomic(path, value)


def resolve_history(relative):
    old = ROOT / relative
    if old.is_file():
        return old
    if relative.startswith("experiments/semantic_sequence_stage1_run2/"):
        return BASE / relative.split("semantic_sequence_stage1_run2/", 1)[1]
    raise FileNotFoundError(relative)


def prepare(out):
    import spacy
    if out.exists():
        raise RuntimeError("Choose a new successor; preparation never overwrites")
    old = BASE / "symbol_order_diagnostic1"
    old_contract = runtime.read(old / "FREEZE.json")
    transfer = BASE / "multiscale_transfer4_recovery1"
    contract = runtime.read(transfer / "FREEZE.json")
    manifests = {}
    for name in ("MANIFEST_GENERATION.sha256", "ai_review1/MANIFEST_AI_REVIEW.sha256"):
        manifest = BASE / name
        count = 0
        for line in manifest.read_text().splitlines():
            expected, relative = line.split(None, 1)
            if runtime.sha(resolve_history(relative.strip())) != expected:
                raise RuntimeError("Original generation/reference SHA differs")
            count += 1
        manifests[name] = {"sha256": runtime.sha(manifest), "verified_files": count}
    symbol_map = runtime.read(BASE / "SYMBOL_MAP.json")
    mapping = {int(k): "ABCD".index(v) for k, v in symbol_map["template_to_symbol"].items()}
    pool = sorted(mapping)
    groups = [[i for i, t in enumerate(pool) if mapping[t] == s] for s in range(4)]
    if len(pool) != 24 or list(map(len, groups)) != [6] * 4:
        raise RuntimeError("Original 24-template dictionary differs")
    if runtime.sha(BASE / "SYMBOL_MAP.json") != old_contract["symbol_map_sha256"]:
        raise RuntimeError("Original symbol map changed")
    splitter = spacy.blank("en")
    splitter.add_pipe("sentencizer")
    streams, origins = [], {}
    for p in sorted((BASE / "private/documents").glob("d*.json")):
        d = runtime.read(p)
        origins[str(p.relative_to(ROOT))] = runtime.sha(p)
        for arm, payload in sorted(d["arms"].items()):
            if runtime.digest(payload["paraphrase"]) != payload["paraphrase_sha256"]:
                raise RuntimeError("Paraphrase identity changed")
            units = diagnostic.segment(payload["paraphrase"], splitter)
            for u in units:
                u["text"] = diagnostic.normalize(u["text"])
                u["text_sha256"] = runtime.digest(u["text"])
            streams.append({"doc_id": d["doc_id"], "arm": arm, "units": units})
    if len(streams) != 48:
        raise RuntimeError("Expected all 24 documents in both arms")
    # Check that sentence boundaries are exactly those of the original diagnostic.
    old_coverage = runtime.read(old / "PARAPHRASE_COVERAGE.json")
    if isinstance(old_coverage, dict):
        old_coverage = old_coverage["rows"]
    before = {(r["doc_id"], r["arm"], r["unit"]): r["text_sha256"] for r in old_coverage}
    after = {(r["doc_id"], r["arm"], i): u["text_sha256"]
             for r in streams for i, u in enumerate(r["units"])}
    if before != after:
        raise RuntimeError("Sentence segmentation differs from original diagnostic")
    private = out / "private"
    private.mkdir(parents=True, mode=0o700)
    (out / ".gitignore").write_text("private/\n*.log\n")
    shutil.copytree(old / "private/code", private / "code")
    for name in ("BANK.json", "propbank_verbs.json"):
        shutil.copyfile(old / "private" / name, private / name)
    shutil.copytree(old / "private/spacy_model", private / "spacy_model")
    for name in ("scripts/run_multiscale_transfer.py",
                 "scripts/run_symbol_sequence_diagnostic.py",
                 "scripts/run_symbol_paraphrase_completion1.py",
                 "scripts/run_research_completion_queue.py", "src/core/semantic_sequence.py"):
        target = private / "code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    shutil.copyfile(BASE / "private/ASSIGNMENTS.jsonl", private / "EVALUATOR_ASSIGNMENTS.jsonl")
    shutil.copyfile(BASE / "private/ai_review1/EVALUATOR_AI_LABELS.jsonl",
                    private / "EVALUATOR_AI_LABELS.jsonl")
    for name in ("IDENTITY_RESULT.json", "ORDER_RESULT.json", "FREEZE.json",
                 "private/ORDER_COSTS.npz", "private/ORDER_KEYS.json"):
        target = private / "original_diagnostic" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old / name, target)
    runtime.atomic(private / "INPUTS.json", streams)
    runtime.atomic(out / "SOURCE_SHA256.json", origins)
    files = {str(p.relative_to(out)): runtime.sha(p) for p in sorted(private.rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts}
    protocol = """# 原24模板复述读回补全

原 Stage1B 的全部24篇、两臂复述、192个正例计划位置，不生成或筛掉文本。
逐句边界与旧诊断逐项SHA相同；模板池和温度0.10不变。
使用同一冻结BART解析器和官方归一化，FP32、4束、原生1024边界。
新缓存逐句解析；只有beam0、无恢复/解码告警、未截断的图可计分。
绑定到句子的失败记擦除并保留分母，未知日志停止本版本。
显存压力/OOM/磁盘不足保存断点并退出75，由队列恢复，不判方法失败。
先固定所有句子预测和2520个等频序列成本，再加载计划符号及模型辅助位置映射。
报告全量解析覆盖、模型辅助映射下的符号身份、真实文本顺序下的精确DP排名。
模型辅助映射不是盲身份真值；开发/原readout各12篇，均已暴露，不称独立确认。
旧embedding成本仅作相同句子流、相同2520候选下的对照，无新拟合。
当前补全不衡量提示修复贡献；该项需要另立同来源配对生成对照。
保留全部旧Stop、结果及SHA，不提供正式FPR/TPR或语义正确性结论。
"""
    (out / "PROTOCOL.md").write_text(protocol)
    runtime.atomic(out / "FREEZE.json", {
        "version": "original-symbol-paraphrase-completion/v1",
        "authorization": "2026-11-02 user requested all audited gaps to be completed",
        "parser": contract["parser"], "scorer": old_contract["scorer"],
        "split": old_contract["split"], "temperature": 0.10,
        "pool": pool, "groups": groups, "template_to_symbol": mapping,
        "inputs": {"streams": 48, "sentence_occurrences": len(after),
                   "unique_texts": len(set(after.values())), "positive_positions": 192},
        "original_seals": manifests, "old_freeze_sha256": runtime.sha(old / "FREEZE.json"),
        "files": files, "protocol_sha256": runtime.sha(out / "PROTOCOL.md"),
        "execution": {"resumable_gpu_pressure": True, "single_parser": True,
                      "resource_pause_exit_code": 75, "fresh_cache": True,
                      "workload_limit": "all frozen sentence texts; no elapsed-time truncation"},
        "resource_gate": {"vram_fraction_max": 0.8},
        "human_labels": False, "prompt_contribution": "SEPARATE_PAIRED_GENERATION_REQUIRED",
    })
    runtime.atomic(out / "VALIDATION.json", {
        "original_generation_and_AI_seals": manifests, "segmentation_identical": True,
        "old24_dictionary_preserved": True, "cuda_run": False,
        "inputs": runtime.read(out / "FREEZE.json")["inputs"],
        "DP_oracle": diagnostic.oracle_validation(equal_composition_sequences()),
    })


def verify(out):
    f = runtime.read(out / "FREEZE.json")
    for name, value in f["files"].items():
        if runtime.sha(out / name) != value:
            raise RuntimeError("Frozen dependency drift: " + name)
    if runtime.sha(out / "PROTOCOL.md") != f["protocol_sha256"]:
        raise RuntimeError("Protocol drift")
    return f


def graph_scores(parser, scorer, text, f):
    parser.parse(text)
    record = runtime.read(parser.out / "private/parses" / (runtime.digest(text) + ".json"))
    if not diagnostic.eligible_parse(record, parser.freeze_sha):
        return None, "parser_ineligible"
    try:
        norm = scorer.normalize(record["amr"], "official")
    except ValueError:
        return None, "normalization_invalid"
    values = [scorer.pair(norm, scorer.bank[t]) for t in f["pool"]]
    q = diagnostic.weights(values, f["groups"])
    return (q.tolist(), None) if q is not None else (None, "dictionary_score_invalid")


def verify_outputs(out, seal):
    if seal["freeze_sha256"] != runtime.sha(out / "FREEZE.json"):
        raise RuntimeError("Output freeze differs")
    for name, value in seal["files"].items():
        if runtime.sha(out / name) != value:
            raise RuntimeError("Saved prediction changed")


def score(out, assets):
    f = verify(out)
    if (out / "HALT.json").exists():
        raise RuntimeError("Integrity halt is immutable")
    if (out / "PREDICTION_SEAL.json").exists():
        verify_outputs(out, runtime.read(out / "PREDICTION_SEAL.json"))
        return
    import torch
    inputs = runtime.read(out / "private/INPUTS.json")
    texts = list(dict.fromkeys(u["text"] for s in inputs for u in s["units"]))
    parser = runtime.Parser(assets, out, f)
    scorer = runtime.Scorer(assets, out)
    if runtime.sha(scorer.glove) != f["scorer"]["glove_sha256"]:
        raise RuntimeError("Scorer vector identity changed")
    precheck = []
    for text in texts[:2]:
        a, b = parser.fresh(text), parser.fresh(text)
        if (a["amr_sha256"], diagnostic.eligible_parse(a, parser.freeze_sha)) != (
                b["amr_sha256"], diagnostic.eligible_parse(b, parser.freeze_sha)):
            raise RuntimeError("Fresh parser replay differs")
        path = out / "private/parses" / (runtime.digest(text) + ".json")
        if path.exists():
            c = runtime.read(path)
            if (c["amr_sha256"], c["valid"]) != (a["amr_sha256"], a["valid"]):
                raise RuntimeError("Resume parser replay differs")
        else:
            runtime.atomic(path, a)
        q1, reason1 = graph_scores(parser, scorer, text, f)
        q2, reason2 = graph_scores(parser, scorer, text, f)
        if (q1, reason1) != (q2, reason2):
            raise RuntimeError("Fresh dictionary scoring replay differs")
        precheck.append({"text_sha256": runtime.digest(text), "exact_replay": True})
    runtime.atomic(out / "GPU_PRECHECK.json", {
        "passed": True, "replays": precheck, "dtype": "float32",
        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "freeze_sha256": runtime.sha(out / "FREEZE.json"),
    })
    for i, text in enumerate(texts):
        if shutil.disk_usage(out).free < 512 * 1024**2:
            raise runtime.ResourcePause("disk_headroom_below_512MiB")
        path = out / "private/predictions" / (runtime.digest(text) + ".json")
        if path.exists():
            r = runtime.read(path)
            parse = out / r["parse_path"]
            if r["freeze_sha256"] != runtime.sha(out / "FREEZE.json") or r[
                    "text_sha256"] != runtime.digest(text) or runtime.sha(parse) != r["parse_sha256"]:
                raise RuntimeError("Prediction resume identity drift")
            continue
        q, reason = graph_scores(parser, scorer, text, f)
        parse_path = "private/parses/" + runtime.digest(text) + ".json"
        frozen(path, {"text_sha256": runtime.digest(text), "q": q, "reason": reason,
                      "freeze_sha256": runtime.sha(out / "FREEZE.json"),
                      "parse_path": parse_path, "parse_sha256": runtime.sha(out / parse_path)})
        runtime.atomic(out / "HEARTBEAT.json", {
            "stage": "scoring", "completed_unique_texts": i + 1, "total": len(texts),
            "parser_calls": parser.calls, "at": runtime.now()})
        runtime.atomic(out / "private/SCORING_DIAGNOSTICS.json", scorer.events)
    # Predictions and all candidate costs are sealed before reference access.
    candidates = equal_composition_sequences()
    costs, keys = [], []
    for s in inputs:
        qs = [runtime.read(out / "private/predictions" / (u["text_sha256"] + ".json"))["q"]
              for u in s["units"]]
        for mode in ("hard", "soft", "unordered"):
            values = unordered_costs(qs, candidates) if mode == "unordered" else exact_costs(qs, candidates, mode)
            if mode == "unordered" and np.ptp(values) != 0:
                raise RuntimeError("Equal-frequency control varies")
            costs.append(values)
            keys.append([s["doc_id"], s["arm"], mode])
    np.savez_compressed(out / "private/ORDER_COSTS.npz", costs=np.stack(costs), candidates=candidates)
    runtime.atomic(out / "private/ORDER_KEYS.json", keys)
    paths = list((out / "private/predictions").glob("*.json"))
    paths += list((out / "private/parses").glob("*.json"))
    paths += [out / "private/ORDER_COSTS.npz", out / "private/ORDER_KEYS.json"]
    frozen(out / "PREDICTION_SEAL.json", {
        "freeze_sha256": runtime.sha(out / "FREEZE.json"), "unique_texts": len(texts),
        "files": {str(p.relative_to(out)): runtime.sha(p) for p in sorted(paths)},
        "reference_access_before_seal": False,
    })


def evaluate(out):
    f = verify(out)
    seal = runtime.read(out / "PREDICTION_SEAL.json")
    verify_outputs(out, seal)
    assignments = {r["doc_id"]: r for r in diagnostic.jsonl(out / "private/EVALUATOR_ASSIGNMENTS.jsonl")}
    labels = {(r["doc_id"], r["planned_carrier_slot"]): r for r in
              diagnostic.jsonl(out / "private/EVALUATOR_AI_LABELS.jsonl")}
    inputs = runtime.read(out / "private/INPUTS.json")
    identity_rows, coverage = [], []
    for s in inputs:
        for u in s["units"]:
            r = runtime.read(out / "private/predictions" / (u["text_sha256"] + ".json"))
            u["q"], u["reason"] = r["q"], r["reason"]
            coverage.append({"doc_id": s["doc_id"], "arm": s["arm"],
                             "available": r["q"] is not None, "reason": r["reason"]})
        if s["arm"] != "positive":
            continue
        for slot in assignments[s["doc_id"]]["slots"]:
            span = labels[(s["doc_id"], slot["slot"])]["paraphrase"].get("evidence_span")
            linked = [u for u in s["units"]
                      if span and max(span[0], u["span"][0]) < min(span[1], u["span"][1])]
            q = np.mean([u["q"] for u in linked], axis=0).tolist() if linked and all(
                u["q"] is not None for u in linked) else None
            identity_rows.append({
                "doc_id": s["doc_id"], "slot": slot["slot"],
                "target": f["template_to_symbol"][str(slot["intended_template"])], "q": q,
                "reason": None if q is not None else (
                    "mapping_missing" if not linked else "parse_or_score_invalid"),
            })
    if len(identity_rows) != 192:
        raise RuntimeError("Position denominator incomplete")
    arrays = np.load(out / "private/ORDER_COSTS.npz", allow_pickle=False)
    candidates = equal_composition_sequences()
    if not np.array_equal(candidates, arrays["candidates"]):
        raise RuntimeError("Candidate set drift")
    keys = runtime.read(out / "private/ORDER_KEYS.json")
    order = []
    for (doc_id, arm, mode), costs in zip(keys, arrays["costs"]):
        truth = [s["intended_state"] for s in assignments[doc_id]["slots"]]
        rank = rank_truth(costs, candidates, truth)
        s = next(s for s in inputs if s["doc_id"] == doc_id and s["arm"] == arm)
        order.append({**rank, "doc_id": doc_id, "arm": arm, "mode": mode,
                      "observations": len(s["units"]),
                      "all_erased": all(u["q"] is None for u in s["units"])})
    identity_results, order_results, baseline = {}, {}, {}
    old_root = out / "private/original_diagnostic"
    old_arrays = np.load(old_root / "private/ORDER_COSTS.npz", allow_pickle=False)
    old_keys = runtime.read(old_root / "private/ORDER_KEYS.json")
    if not np.array_equal(old_arrays["candidates"], candidates):
        raise RuntimeError("Embedding comparator candidates differ")
    for split in ("development", "readout"):
        ids = f["split"][split]
        identity_results[split] = diagnostic.identity([r for r in identity_rows if r["doc_id"] in ids])
        order_results[split] = {}
        for arm in ("positive", "unwatermarked"):
            for mode in ("hard", "soft", "unordered"):
                rows = [r for r in order if r["doc_id"] in ids and r["arm"] == arm and r["mode"] == mode]
                order_results[split][arm + "/" + mode] = diagnostic.order_summary(rows)
        paired = []
        for key, cost in zip(old_keys, old_arrays["costs"]):
            method, condition, doc_id, arm, mode = key
            if method == "C_sentence_stream" and condition == "paraphrase" and doc_id in ids and arm == "positive" and mode == "soft":
                truth = [s["intended_state"] for s in assignments[doc_id]["slots"]]
                c = rank_truth(cost, candidates, truth)["pairwise_win_rate"]
                a = next(r["pairwise_win_rate"] for r in order
                         if (r["doc_id"], r["arm"], r["mode"]) == (doc_id, arm, mode))
                paired.append({"doc_id": doc_id, "AMR": a, "embedding": c, "AMR_minus_C": a - c})
        if len(paired) != 12:
            raise RuntimeError("Paired embedding comparator incomplete")
        baseline[split] = paired
    result = {
        "status": "COMPUTATION_COMPLETE_HUMAN_TRUTH_UNAVAILABLE",
        "freeze_sha256": runtime.sha(out / "FREEZE.json"), "sentence_occurrences": len(coverage),
        "scored_sentence_occurrences": sum(r["available"] for r in coverage),
        "erasure_reasons": dict(Counter(r["reason"] for r in coverage if not r["available"])),
        "identity_model_assisted_mapping": identity_results, "equal_frequency_order": order_results,
        "paired_embedding_order": baseline, "independent_human_rows": 0,
        "prompt_contribution": f["prompt_contribution"],
        "formal_FPR": None, "formal_TPR": None,
        "old_incomplete_identity": runtime.read(old_root / "IDENTITY_RESULT.json"),
    }
    frozen(out / "private/IDENTITY_ROWS.json", identity_rows)
    frozen(out / "private/ORDER_ROWS.json", order)
    frozen(out / "RESULT.json", result)
    frozen(out / "AUDIT.json", {
        "all_48_streams": len(inputs) == 48, "all_192_positions": len(identity_rows) == 192,
        "all_frozen_texts_attempted": len(list((out / "private/predictions").glob("*.json"))) == f["inputs"]["unique_texts"],
        "predictions_sealed_before_reference": True, "same_2520_candidates_as_embedding": True,
        "old_results_unchanged": True, "human_or_prompt_effect_closure": False,
    })
    frozen(out / "TERMINAL.json", {"status": "COMPUTATION_COMPLETE",
                                  "result_sha256": runtime.sha(out / "RESULT.json"),
                                  "audit_sha256": runtime.sha(out / "AUDIT.json")})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare", "verify", "run", "evaluate"))
    ap.add_argument("--out", type=Path, default=DEFAULT)
    ap.add_argument("--assets", type=Path, default=ROOT)
    args = ap.parse_args()
    if args.action == "prepare":
        prepare(args.out)
        print(json.dumps(runtime.read(args.out / "VALIDATION.json")))
        return
    verify(args.out)
    if args.action == "verify":
        print(json.dumps({"frozen_SHA_verified": True}))
        return
    with (args.out / "private/worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        try:
            if args.action == "run":
                score(args.out, args.assets)
            evaluate(args.out)
        except Exception as exc:
            is_oom = type(exc).__name__ == "OutOfMemoryError"
            pause = isinstance(exc, runtime.ResourcePause) or is_oom
            runtime.atomic(args.out / ("PAUSED.json" if pause else "HALT.json"), {
                "type": type(exc).__name__, "at": runtime.now(),
                "scientific_result": None, "resume_allowed": pause,
            })
            if pause:
                sys.exit(75)
            raise
        finally:
            runtime.atomic(args.out / "private/sessions" / (str(time.time_ns()) + ".json"),
                           {"elapsed_seconds": time.monotonic() - started, "at": runtime.now()})


if __name__ == "__main__":
    main()
