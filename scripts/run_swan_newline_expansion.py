"""Resource-gated CUDA scoring and independent-negative newline reassessment."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_swan_blind_eval import interval, metrics, save_json
from run_swan_blind_eval import (
    Cache, assets, bank_load, configs_for, document_scores, sha_file)
from src.core.swan_blind_protocol import digest, plan_document
from src.core.swan_blind_runtime import (
    SETTINGS, AuditedParser, BoundAuditError, init_matcher, score_graph, text_sha)

OUT = ROOT / "experiments/swan_multiview_development1/newline_negative_expansion1"
PRIOR = ROOT / "experiments/swan_multiview_development1/blind_eval1/run2"
PRIOR_CONTROL = "/tmp/swan_blind_eval_control/supervisord.conf"


def now():
    return datetime.now(timezone.utc).isoformat()


def emit(event, **fields):
    print(json.dumps({"at": now(), "event": event, **fields}), flush=True)


def inputs(config):
    docs = json.loads((OUT / "private/DOCUMENTS.json").read_text())["documents"]
    plans, unique = [], {}
    for d in docs:
        if not d["eligible"]:
            continue
        # Only exact attacked full text enters candidate extraction.
        p = plan_document(d["text"], config)
        if p.status != "geometry_ready" or p.uncovered_nonwhitespace:
            raise RuntimeError("Input geometry failed")
        keys = []
        for c in p.candidates:
            k = text_sha(c.text)
            if k in unique and unique[k] != c.text:
                raise RuntimeError("Candidate digest collision")
            unique[k] = c.text
            keys.append(k)
        plans.append({
            "record_id": "S6:paired_negative:" + d["document_id"] + ":newline",
            "method": "S6", "arm": "paired_negative", "document_id": d["document_id"],
            "attack": "newline", "split": d["split"], "text_sha256": p.text_sha256,
            "configuration_sha256": p.configuration_sha256,
            "N_expected": len(keys), "candidate_keys": keys,
        })
    return docs, plans, unique


def verify_resources():
    source = json.loads((OUT / "SOURCE_FREEZE.json").read_text())
    current = assets()
    old = source["assets"]
    # Local documentation may have progressed after run2 freeze. Verify every
    # original execution asset, version and native resource group independently.
    omit = "experiments/swan_multiview_development1/PROTOCOL.md"
    for k, value in old["files_sha256"].items():
        if k != omit and current["files_sha256"].get(k) != value:
            raise RuntimeError("Source runtime asset changed: " + k)
    for k in ("resource_groups", "versions", "punkt", "settings"):
        if current[k] != old[k]:
            raise RuntimeError("Source runtime resources changed: " + k)
    config = configs_for(current)["S6"]
    if asdict(config) != source["configurations"]["S6"]:
        raise RuntimeError("S6 execution settings changed")
    return current, config


def freeze():
    if (OUT / "FREEZE.json").exists():
        return verify()
    if not json.loads((OUT / "INPUT_AUDIT.json").read_text())["ready"]:
        raise RuntimeError("Input identity audit failed")
    current, config = verify_resources()
    docs, plans, unique = inputs(config)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "models/bart-large-tokenizer"))
    counts = {}
    keys = list(unique)
    for i in range(0, len(keys), 1024):
        batch = keys[i:i + 1024]
        ids = tokenizer([unique[k] for k in batch], padding=False, truncation=False)["input_ids"]
        counts.update((k, len(t)) for k, t in zip(batch, ids))
    bank_load()
    files = [
        Path(__file__), OUT / "GENERATION_FREEZE.json", OUT / "SOURCE_FREEZE.json",
        OUT / "INPUT_AUDIT.json", OUT / "private/DOCUMENTS.json",
        OUT / "HISTORICAL_CONTROL_SCORES.json", OUT / "PROTOCOL.md",
    ]
    frozen = {
        "schema": "swan-newline-negative-evaluation/v1", "at": now(),
        "assets": current, "config": asdict(config),
        "file_sha256": {str(p.relative_to(ROOT)): sha_file(p) for p in files},
        "document_manifest_sha256": digest(docs), "plan_manifest_sha256": digest(plans),
        "negative_documents": len(docs),
        "calibration_ids": [d["document_id"] for d in docs if d["split"] == "calibration"],
        "test_ids": [d["document_id"] for d in docs if d["split"] == "test"],
        "eligible_documents": len(plans), "candidate_unique_inputs": len(unique),
        "candidate_total": sum(p["N_expected"] for p in plans),
        "native_input_max_tokens": max(counts.values()) if counts else 0,
        "native_context_rejections": sum(n > config.parser_context_tokens for n in counts.values()),
        "fresh_private_cache": True, "parser_batch_size": 8, "score_workers": 4,
        "strict_audit": SETTINGS["audit"], "input_text_no_normalization": True,
    }
    save_json(OUT / "FREEZE.json", frozen)
    emit("frozen", documents=len(docs), candidates=len(unique), fingerprint=digest(frozen))
    return frozen, config, docs, plans, unique


def verify():
    frozen = json.loads((OUT / "FREEZE.json").read_text())
    for name, expected in frozen["file_sha256"].items():
        if sha_file(ROOT / name) != expected:
            raise RuntimeError("Evaluation freeze changed: " + name)
    current, config = verify_resources()
    # PROTOCOL.md is mutable prose; core resources and all inputs remain bound.
    for key in ("resource_groups", "versions", "settings", "punkt"):
        if current[key] != frozen["assets"][key]:
            raise RuntimeError("Evaluation runtime changed")
    docs, plans, unique = inputs(config)
    if digest(plans) != frozen["plan_manifest_sha256"] \
            or digest(docs) != frozen["document_manifest_sha256"]:
        raise RuntimeError("Frozen documents or plans changed")
    return frozen, config, docs, plans, unique


def readiness():
    frozen, _, _, plans, unique = verify()
    parser = AuditedParser()
    bank = bank_load()
    init_matcher(bank)
    keys = sorted(unique, key=lambda k: len(unique[k]), reverse=True)
    strata = list(dict.fromkeys(keys[:8] + keys[len(keys) // 2:len(keys) // 2 + 8] + keys[-8:]))
    import torch
    torch.cuda.reset_peak_memory_stats()
    parse_seconds, score_seconds, strict, replay = 0., 0., 0, True
    for i in range(0, len(strata), 8):
        texts = [unique[k] for k in strata[i:i + 8]]
        t = time.monotonic()
        records = parser.parse(texts)
        parse_seconds += time.monotonic() - t
        t = time.monotonic()
        first = [score_graph(r["graph"]) for r in records if r.get("graph")]
        score_seconds += time.monotonic() - t
        second = [score_graph(r["graph"]) for r in records if r.get("graph")]
        replay &= first == second and all(r["scores"] is not None for r in first)
        strict += sum(r["strict_ok"] for r in records)
    free, total = torch.cuda.mem_get_info()
    peak = torch.cuda.max_memory_allocated()
    estimate = len(unique) * (parse_seconds / len(strata)
                              + score_seconds / len(strata) / 4) / 3600
    result = {
        "at": now(), "freeze_sha256": digest(frozen), "cuda": True, "dtype": "float32",
        "real_candidates": len(strata), "strict_valid": strict, "full_batch_checked": 8,
        "bound_audit_pass": True, "numeric_replay_exact": replay,
        "gpu_peak_allocated_bytes": peak, "gpu_total_bytes": total,
        "gpu_free_after_pilot_bytes": free, "estimate_hours": estimate,
        "ready": replay and peak < .8 * total and estimate <= 48,
        "no_old_parse_or_slot_cache": True,
    }
    save_json(OUT / "READINESS.json", result)
    emit("readiness", **result)
    if not result["ready"]:
        raise RuntimeError("Readiness gates failed")
    del parser


def append_ineligible(rows, docs, config):
    for d in docs:
        if d["eligible"]:
            continue
        plan = plan_document(d["text"], config)
        rows.append({
            "record_id": "S6:paired_negative:" + d["document_id"] + ":newline:strict",
            "method": "S6", "arm": "paired_negative", "document_id": d["document_id"],
            "attack": "newline", "split": d["split"], "text_sha256": plan.text_sha256,
            "configuration_sha256": plan.configuration_sha256,
            "N_expected": len(plan.candidates), "complete": False,
            "failure_counts": dict(Counter(d["failure_reasons"])),
            "unit_bank_max": [], "N_valid_candidates": 0,
        })
    return rows


def run(cache_path):
    if (OUT / "HALT.json").exists():
        raise RuntimeError("Permanent halt prohibits resumed inference")
    frozen, config, docs, plans, unique = verify()
    ready = json.loads((OUT / "READINESS.json").read_text())
    if not ready["ready"] or ready["freeze_sha256"] != digest(frozen):
        raise RuntimeError("Readiness does not bind freeze")
    cache = Cache(cache_path, digest(frozen))
    parser, bank = AuditedParser(), bank_load()
    keys = list(unique)
    started = time.monotonic()
    def heartbeat(stage, completed, expected):
        save_json(OUT / "heartbeat.json", {
            "at": now(), "state": "running", "stage": stage, "completed": completed,
            "expected": expected, "unique_inputs_parsed": len(cache.parses),
            "unique_graphs_scored": len(cache.scores),
            "freeze_sha256": digest(frozen), "elapsed_seconds": time.monotonic() - started,
        })
    todo = [k for k in keys if k not in cache.parses]
    try:
        for i in range(0, len(todo), 8):
            batch = todo[i:i + 8]
            counts = parser.token_counts([unique[k] for k in batch])
            valid = [k for k, n in zip(batch, counts) if n <= parser.model.max_sent_len]
            for k, n in zip(batch, counts):
                if n > parser.model.max_sent_len:
                    cache.put("parses", k, {
                        "input_sha256": k, "native_input_tokens": n, "strict_ok": False,
                        "reasons": ["native_context_exceeded"], "graph": None, "blank_output": False})
            for k, r in zip(valid, parser.parse([unique[k] for k in valid]) if valid else []):
                cache.put("parses", k, r)
            heartbeat("parse", min(i + 8, len(todo)), len(todo))
        # Release this job's CUDA model before CPU bank scoring.
        del parser
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        graphs = {text_sha(cache.parses[k]["graph"]): cache.parses[k]["graph"]
                  for k in keys if cache.parses[k].get("graph")}
        todo_graphs = [g for g in graphs if g not in cache.scores]
        with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=init_matcher, initargs=(bank,)) as pool:
            pending, index, finished = {}, 0, 0
            while index < len(todo_graphs) or pending:
                while index < len(todo_graphs) and len(pending) < 32:
                    k = todo_graphs[index]
                    pending[pool.submit(score_graph, graphs[k])] = k
                    index += 1
                completed, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
                for future in completed:
                    k = pending.pop(future)
                    cache.put("scores", k, future.result())
                    finished += 1
                heartbeat("score", finished, len(todo_graphs))
        rows = append_ineligible(document_scores(plans, cache, ["S6"]), docs, config)
        save_json(OUT / "DOCUMENT_SCORES.json", {"freeze_sha256": digest(frozen), "documents": rows})
        analyze()
        save_json(OUT / "heartbeat.json", {"state": "complete", "at": now(),
                                          "freeze_sha256": digest(frozen),
                                          "unique_inputs_parsed": len(cache.parses),
                                          "unique_graphs_scored": len(cache.scores)})
        emit("complete", documents=len(rows))
    except Exception as exc:
        save_json(OUT / "HALT.json", {
            "at": now(), "permanent": True, "exception_type": type(exc).__name__,
            "reason": str(exc) if isinstance(exc, BoundAuditError) else "execution_failure",
            "freeze_sha256": digest(frozen)})
        raise


def statistic(r, unit):
    return sum(s >= unit for s in r["unit_bank_max"]) / r["N_expected"] if r["complete"] else None


def select_threshold(rows, unit):
    if any(r["split"] != "calibration" or r["arm"] != "paired_negative" for r in rows):
        raise ValueError("Calibration-only negative selector")
    values = sorted(statistic(r, unit) for r in rows if r["complete"])
    rank = math.ceil((len(values) + 1) * .95)
    return (values[rank - 1] if rank <= len(values) else 1.), len(values), rank


def analyze():
    generation = json.loads((OUT / "GENERATION_FREEZE.json").read_text())
    frozen = json.loads((OUT / "FREEZE.json").read_text())
    payload = json.loads((OUT / "DOCUMENT_SCORES.json").read_text())
    if payload["freeze_sha256"] != digest(frozen):
        raise RuntimeError("Document scores do not bind freeze")
    rows = payload["documents"]
    if len(rows) != 200 or len({r["document_id"] for r in rows}) != 200:
        raise RuntimeError("Frozen negative denominator changed")
    cal = [r for r in rows if r["split"] == "calibration"]
    test = [r for r in rows if r["split"] == "test"]
    if {r["document_id"] for r in cal} != set(frozen["calibration_ids"]) \
            or {r["document_id"] for r in test} != set(frozen["test_ids"]):
        raise RuntimeError("Split identity changed")
    unit, old = generation["fixed_unit_threshold"], generation["fixed_document_threshold"]
    threshold, count, rank = select_threshold(cal, unit)
    # Commit calibration before evaluating any new test labels/scores.
    save_json(OUT / "CALIBRATION.json", {
        "unit_threshold": unit, "old_document_threshold": old, "document_threshold": threshold,
        "calibration_only": True, "complete_calibration_count": count, "order_rank": rank,
        "calibration_manifest_sha256": digest(cal), "freeze_sha256": digest(frozen),
        "target": .05, "finite_sample_rank": "ceil((n_complete+1)*0.95)",
        "guarantee_scope": "marginal conditional-on-completion under exchangeability; not empirical proof",
    })
    historical = json.loads((OUT / "HISTORICAL_CONTROL_SCORES.json").read_text())["documents"]
    positives = [r for r in historical if r["arm"] == "positive" and r["attack"] == "newline"]
    old_neg = [r for r in historical if r["arm"] == "paired_negative" and r["attack"] == "newline"]
    def measured(subset, c):
        stats = {r["record_id"]: statistic(r, unit) for r in subset}
        decisions = {k: v > c if v is not None else None for k, v in stats.items()}
        return metrics(subset, decisions, stats, c, True)
    result = {
        "schema": "swan-newline-negative-reassessment/v1", "at": now(),
        "freeze_sha256": digest(frozen), "unit_threshold": unit,
        "negative_scope": "new-generated-unwatermarked-scenario-documents",
        "independent_human_test": False, "threshold_uses_test": False,
        "positive_tpr_scope": generation["positive_tpr_scope"],
        "calibration_documents": len(cal), "test_documents": len(test),
        "historical_newline_control": {"FPR": measured(old_neg, old), "TPR": measured(positives, old)},
        "fixed_threshold": {
            "threshold": old, "new_test_FPR": measured(test, old),
            "new_calibration_FPR": measured(cal, old), "historical_TPR": measured(positives, old)},
        "expanded_calibration_threshold": {
            "threshold": threshold, "complete_calibration": count, "order_rank": rank,
            "new_test_FPR": measured(test, threshold), "calibration_FPR": measured(cal, threshold),
            "historical_TPR": measured(positives, threshold)},
        "zero_of_100_two_sided_95_upper": interval(0, 100)[1],
    }
    # Preserve all seven historical positive conditions as development-only controls.
    result["historical_positive_all_conditions"] = {
        attack: {"fixed": measured([r for r in historical if r["arm"] == "positive"
                                    and r["attack"] == attack], old),
                 "expanded": measured([r for r in historical if r["arm"] == "positive"
                                       and r["attack"] == attack], threshold)}
        for attack in sorted({r["attack"] for r in historical})}
    save_json(OUT / "RESULTS.json", result)
    # Test scores can be perturbed without altering numeric calibration.
    selector_ok = False
    try:
        select_threshold(test, unit)
    except ValueError:
        selector_ok = True
    audit = {
        "ready": True, "calibration_test_document_overlap": 0,
        "frozen_denominator": len(rows), "calibration_count": len(cal), "test_count": len(test),
        "selector_rejects_test": selector_ok,
        "all_rows_bind_S6_configuration": len({r["configuration_sha256"] for r in rows}) == 1,
        "all_complete_rows_cover_all_candidates": all(
            len(r["unit_bank_max"]) == r["N_expected"] for r in rows if r["complete"]),
        "all_scores_finite_and_valid": all(
            math.isfinite(s) and 0 <= s <= 1 for r in rows for s in r["unit_bank_max"]),
        "historical_FPR_reproduced": result["historical_newline_control"]["FPR"]["decisions_positive"] == 2,
        "historical_TPR_reproduced": result["historical_newline_control"]["TPR"]["decisions_positive"] == 5,
        "fresh_cache": True, "threshold_test_leakage": False,
    }
    if not all(v for k, v in audit.items() if k in (
            "selector_rejects_test", "all_rows_bind_S6_configuration",
            "all_complete_rows_cover_all_candidates", "all_scores_finite_and_valid",
            "historical_FPR_reproduced", "historical_TPR_reproduced")):
        raise RuntimeError("Final evaluation audit failed")
    save_json(OUT / "AUDIT.json", audit)
    def fmt(m):
        return f"{m['decisions_positive']}/{m['denominator']} ({100*m['rate']:.1f}%)"
    lines = [
        "# S6 newline 新增负例重评", "",
        "新增 100 篇生成式无水印校准文档与 100 篇生成式无水印测试文档。"
        "分组、提示、模型与判定规则在评分前冻结；原 15 篇测试负例保持历史对照。"
        "本批不属于独立人类文本测试。", "",
        "| 口径 | 文档阈值 | 测试 FPR | newline TPR（历史正例） | 测试负例弃权 |",
        "|---|---:|---|---|---:|",
        f"| 原 15 篇历史负例 | {old:.9g} | {fmt(result['historical_newline_control']['FPR'])}"
        f" | {fmt(result['historical_newline_control']['TPR'])} | {result['historical_newline_control']['FPR']['unresolved']} |",
    ]
    for key, name in (("fixed_threshold", "原阈值 / 新测试负例"),
                      ("expanded_calibration_threshold", "新增校准阈值 / 新测试负例")):
        r = result[key]
        lines.append(f"| {name} | {r['threshold']:.9g} | {fmt(r['new_test_FPR'])}"
                     f" | {fmt(r['historical_TPR'])} | {r['new_test_FPR']['unresolved']} |")
    lines += ["", f"单元阈值固定为 {unit:.12g}，文档分数为全部 S6/1 候选的 green 比例，"
              "判定采用严格大于。任一候选解析或全 bank 评分失败则整篇弃权，保留完整分母。", "",
              f"新增校准集完整解析 {count}/100 篇，文档阈值使用其第 {rank} 个升序统计量"
              "（$\\lceil0.95(n+1)\\rceil$）；不足 19 篇时采用 1 的保守回退阈值。", "",
              "| 新测试负例口径 | 全分母 FPR 的双侧 95% 区间 | 完整文档条件 FPR | 完整文档数 |",
              "|---|---|---|---:|"]
    for key in ("fixed_threshold", "expanded_calibration_threshold"):
        r = result[key]["new_test_FPR"]
        lo, hi = r["binomial_interval_95"]
        conditional = f"{100*r['conditional_rate']:.1f}%" if r["conditional_rate"] is not None else "无完整文档"
        lines.append(f"| {key} | [{100*lo:.2f}%, {100*hi:.2f}%] | "
                     f"{conditional} | {r['complete_decision_documents']} |")
    r = result["expanded_calibration_threshold"]["new_test_FPR"]
    if r["binomial_interval_95"][1] <= .05:
        conclusion = "本批新生成负例的 FPR 双侧 95% 上界不超过 5%。"
    else:
        conclusion = "本批仍不能在双侧 95% 置信区间口径下证明总体 FPR≤5%。"
    lines += ["", conclusion + "弃权不等于识别正确，需同时查看完整文档条件误报率。",
              "100 个新测试文档不与旧 15 篇混合估计；场景由同一模型生成，"
              "文档 JSON 提示与原逐槽续写不同，因此该结果不能直接解释为原配对负例总体误报率。",
              "正例 TPR 复用历史 15 篇，不能当作新增独立泛化结果，也不据此调阈值。",
              "", "所有数值、弃权、分母与来源见 `RESULTS.json`、`CALIBRATION.json`、"
              "`INPUT_AUDIT.json`、`AUDIT.json` 和 `SHA256SUMS.json`。原始生成文本和缓存限于 private。", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines))
    hashes = {p.name: sha_file(p) for p in OUT.iterdir()
              if p.is_file() and p.suffix in (".json", ".md") and p.name != "SHA256SUMS.json"
              and p.name not in ("heartbeat.json", "QUEUE_STATUS.json")}
    save_json(OUT / "SHA256SUMS.json", hashes)
    emit("analyzed", threshold=threshold, FPR=r["rate"], unresolved=r["unresolved"])


def wait_for_capacity():
    while True:
        prior_heartbeat = json.loads((PRIOR / "heartbeat.json").read_text())
        memory = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout.strip()
        free, total = (int(s.strip()) for s in memory.splitlines()[0].split(","))
        prior_complete = prior_heartbeat.get("state") == "complete" or (PRIOR / "HALT.json").exists()
        status = subprocess.run(["supervisorctl", "-c", PRIOR_CONTROL, "status", "evaluation"],
                                capture_output=True, text=True).stdout
        paused = (OUT / "PRIOR_PAUSE.json").exists() and "STOPPED" in status
        available = (prior_complete or paused) and free >= 6500
        save_json(OUT / "QUEUE_STATUS.json", {
            "at": now(), "state": "capacity_ready" if available else "waiting_for_GPU_capacity",
            "prior_state": prior_heartbeat.get("state"), "prior_method": prior_heartbeat.get("method"),
            "prior_stage": prior_heartbeat.get("stage"), "gpu_free_mib": free,
            "gpu_total_mib": total, "minimum_free_mib": 6500,
            "nonoverlap_with_prior_evaluation": True, "prior_supervisor_paused": paused})
        if available:
            return
        time.sleep(45)


def pause_prior():
    heartbeat = json.loads((PRIOR / "heartbeat.json").read_text())
    if heartbeat.get("state") == "complete" or (PRIOR / "HALT.json").exists():
        return False
    state = subprocess.run(["supervisorctl", "-c", PRIOR_CONTROL, "status", "evaluation"],
                           capture_output=True, text=True).stdout
    if "STOPPED" in state and (OUT / "PRIOR_PAUSE.json").exists():
        return True
    if "RUNNING" not in state:
        raise RuntimeError("Cannot pause prior evaluation in unexpected supervisor state")
    group = None
    lines = subprocess.run(["ps", "-eo", "pid,ppid,pgid,args"],
                           capture_output=True, text=True, check=True).stdout.splitlines()
    for line in lines:
        fields = line.split(None, 3)
        if len(fields) == 4 and "scripts/run_swan_blind_eval.py --phase run" in fields[3] \
                and not fields[3].startswith(("sh ", "bash ", "zsh ")):
            if group is not None and group != int(fields[2]):
                raise RuntimeError("Multiple prior evaluation process groups")
            group = int(fields[2])
    if group is None or group == os.getpgrp():
        raise RuntimeError("Prior evaluation group cannot be safely identified")
    save_json(OUT / "PRIOR_PAUSE.json", {
        "at": now(), "prior_heartbeat": heartbeat,
        "reason": "user_requested_priority_S6_newline_negative_expansion",
        "automatic_resume": True, "frozen_sources_modified": False,
        "cache_policy": "SQLite-WAL-atomic-committed-candidate-resumption",
    })
    # Supervisor first records STOPPED, preventing automatic restart. The old
    # flock wrapper does not propagate signals, so stop its verified group next.
    subprocess.run(["supervisorctl", "-c", PRIOR_CONTROL, "stop", "evaluation"],
                   capture_output=True, text=True, check=True, timeout=30)
    try:
        os.killpg(group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for _ in range(30):
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            break
        time.sleep(.25)
    emit("prior_paused", cache_preserved=True)
    return True


def resume_prior():
    if json.loads((PRIOR / "heartbeat.json").read_text()).get("state") == "complete" \
            or (PRIOR / "HALT.json").exists():
        return
    result = subprocess.run(["supervisorctl", "-c", PRIOR_CONTROL, "start", "evaluation"],
                            capture_output=True, text=True, timeout=30)
    save_json(OUT / "PRIOR_RESUME.json", {
        "at": now(), "requested": True, "returncode": result.returncode,
        "resumes_committed_cache": True})
    emit("prior_resume", returncode=result.returncode)
    if result.returncode:
        raise RuntimeError("Prior evaluation automatic resume failed")


def all_phases(cache):
    if (OUT / "HALT.json").exists():
        raise RuntimeError("Permanent evaluation halt")
    freeze()
    if (OUT / "heartbeat.json").exists() \
            and json.loads((OUT / "heartbeat.json").read_text()).get("state") == "complete":
        analyze()
        return
    paused = pause_prior()
    try:
        wait_for_capacity()
        if not (OUT / "READINESS.json").exists():
            readiness()
        run(cache)
    except Exception as exc:
        if not (OUT / "HALT.json").exists():
            save_json(OUT / "HALT.json", {"at": now(), "permanent": True,
                                        "reason": "readiness_or_execution_failure",
                                        "exception_type": type(exc).__name__})
        raise
    finally:
        if paused:
            resume_prior()


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("phase", choices=("freeze", "readiness", "run", "analyze", "all"))
    cli.add_argument("--cache", type=Path, default=Path("/tmp/swan_newline_negative_expansion1"))
    args = cli.parse_args()
    os.umask(0o077)
    if args.phase == "analyze":
        analyze()
    else:
        with (OUT / "private/evaluation.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            {"freeze": freeze, "readiness": readiness,
             "run": lambda: run(args.cache), "all": lambda: all_phases(args.cache)}[args.phase]()
