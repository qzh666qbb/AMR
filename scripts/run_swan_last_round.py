"""Fresh CUDA confirmation; one candidate, strict audit, independent cache."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from dataclasses import asdict
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_swan_blind_eval import save_json
from scripts.analyze_swan_last_round import CONDITIONS, analyze
from scripts.prepare_swan_blind_definitions import load_attack_functions
from scripts.run_swan_blind_eval import assets, configs_for, bank_load, Cache, sha_file
from src.core.swan_blind_protocol import ProtocolConfig, digest, plan_document
from src.core.swan_blind_runtime import SETTINGS, AuditedParser, init_matcher, score_graph, text_sha

OUT = ROOT / "experiments/swan_multiview_development1/final_completion1"
CACHE = Path("/tmp/swan_last_round_confirmation1")


def now():
    return datetime.now(timezone.utc).isoformat()


def remaining():
    c = json.loads((OUT / "CONTROL.json").read_text())
    if (OUT / "DEADLINE_STOP.json").exists() or (OUT / "HALT.json").exists():
        return 0
    return (datetime.fromisoformat(c["deadline_at"]) - datetime.now(timezone.utc)).total_seconds()


def check_time():
    if remaining() <= 0:
        raise TimeoutError("Immutable final-round deadline")


def validate_input_readiness(audit, docs):
    expected = {("human", "negative", "calibration"): 100,
                ("human", "negative", "test"): 200,
                ("generated", "negative", "calibration"): 100,
                ("generated", "negative", "test"): 100,
                ("generated", "positive", "test"): 50}
    observed = Counter((d["source"], d["arm"], d["split"]) for d in docs)
    if observed != Counter(expected) or len({d["document_id"] for d in docs}) != 550:
        raise RuntimeError("Frozen input denominators changed")
    eligible = Counter((d["source"], d["arm"], d["split"]) for d in docs if d["eligible"])
    if not audit["all_frozen_documents_retained"] or audit["source_family_overlap"] \
            or not audit["dispatch_cap_pass"] or any(eligible[k] < 19 for k in expected):
        raise RuntimeError("Input readiness failed")
    if audit["generated_eligible"] != sum(
            d["eligible"] for d in docs if d["source"] == "generated"):
        raise RuntimeError("Input eligibility audit differs")


def inputs(configs):
    docs = json.loads((OUT / "private/DOCUMENTS.json").read_text())["documents"]
    attacks = load_attack_functions()
    plans, unique, injection = [], {}, []
    for method, config in configs.items():
        for d in docs:
            for attack in CONDITIONS:
                text, _ = attacks[attack]([{"text": t} for t in d["sentences"]])
                p = plan_document(text, config)
                reasons = list(d["failure_reasons"]) + list(p.reasons)
                keys = []
                if not reasons:
                    for candidate in p.candidates:
                        key = text_sha(candidate.text)
                        if key in unique and unique[key] != candidate.text:
                            raise RuntimeError("Candidate SHA collision")
                        unique[key] = candidate.text
                        keys.append(key)
                plans.append({
                    "record_id": f"{method}:{d['document_id']}:{attack}",
                    "method": method, "document_id": d["document_id"], "source": d["source"],
                    "source_article_sha256": d["source_article_sha256"],
                    "arm": d["arm"], "split": d["split"], "attack": attack,
                    "text_sha256": p.text_sha256, "configuration_sha256": p.configuration_sha256,
                    "N_expected": len(p.candidates), "candidate_keys": keys,
                    "input_failure_reasons": reasons,
                })
    for d in docs:
        if d["arm"] != "positive":
            continue
        for index, text in enumerate(d["sentences"]):
            key = text_sha(text) if d["eligible"] else None
            if key:
                unique[key] = text
            injection.append({"document_id": d["document_id"], "slot": index,
                              "target": d["target_template_ids"][index], "key": key})
    return plans, unique, injection


def freeze():
    check_time()
    if (OUT / "EVAL_FREEZE.json").exists():
        raise RuntimeError("Evaluation already frozen")
    old = ROOT / "experiments/swan_multiview_development1/blind_eval1/run2"
    heartbeat = json.loads((old / "heartbeat.json").read_text())
    if heartbeat["state"] != "complete" or (old / "HALT.json").exists():
        raise RuntimeError("Old queue must complete before fresh GPU work")
    selection = json.loads((OUT / "CANDIDATE_SELECTION.json").read_text())
    audit = json.loads((OUT / "INPUT_AUDIT.json").read_text())
    docs = json.loads((OUT / "private/DOCUMENTS.json").read_text())["documents"]
    validate_input_readiness(audit, docs)
    if sha_file(OUT / "private/DOCUMENTS.json") != audit["input_manifest_sha256"]:
        raise RuntimeError("New manifest changed")
    asset = assets()
    configs = configs_for(asset)
    configs = {m: configs[m] for m in ("P0", selection["method"])}
    plans, unique, injection = inputs(configs)
    if not unique or len(unique) > 500_000:
        raise RuntimeError("New candidate cap exceeded")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "models/bart-large-tokenizer"))
    lengths = []
    texts = list(unique.values())
    for i in range(0, len(texts), 1024):
        check_time()
        lengths += [len(t) for t in tokenizer(
            texts[i:i+1024], truncation=False, padding=False)["input_ids"]]
    bank_load()
    value = {
        "schema": "swan-last-round-eval/v1", "created_at": now(),
        "assets": asset, "candidate_method": selection["method"],
        "configurations": {m: asdict(c) for m, c in configs.items()},
        "candidate_selection_sha256": sha_file(OUT / "CANDIDATE_SELECTION.json"),
        "input_manifest_sha256": sha_file(OUT / "private/DOCUMENTS.json"),
        "input_audit_sha256": sha_file(OUT / "INPUT_AUDIT.json"),
        "generation_freeze_sha256": sha_file(OUT / "GENERATION_FREEZE.json"),
        "source_freeze_sha256": sha_file(OUT / "SOURCE_FREEZE.json"),
        "implementation_sha256": sha_file(__file__),
        "analysis_sha256": sha_file(ROOT / "scripts/analyze_swan_last_round.py"),
        "output_relative_path": str(OUT.relative_to(ROOT)),
        "cache_path": str(CACHE),
        "plan_sha256": digest(plans), "injection_manifest_sha256": digest(injection),
        "unique_candidates": len(unique), "document_plans": len(plans),
        "native_max_input_tokens": max(lengths),
        "native_context_rejections": sum(
            n > configs["P0"].parser_context_tokens for n in lengths),
        "settings": SETTINGS, "score_workers": 4, "fresh_cache": True,
        "unit_calibration": "pooled-complete-new-negative-units-upper-1pct",
        "document_calibration": "max-two-sources-five-conditions-order-ceil(0.95*(n+1))",
        "candidate_totals": dict(Counter({
            m: sum(p["N_expected"] for p in plans if p["method"] == m) for m in configs})),
        "conditions": CONDITIONS, "no_successor": True,
    }
    save_json(OUT / "EVAL_FREEZE.json", value)
    return value


def verify():
    check_time()
    frozen = json.loads((OUT / "EVAL_FREEZE.json").read_text())
    if sha_file(__file__) != frozen["implementation_sha256"] \
            or sha_file(ROOT / "scripts/analyze_swan_last_round.py") != frozen["analysis_sha256"] \
            or sha_file(OUT / "private/DOCUMENTS.json") != frozen["input_manifest_sha256"] \
            or sha_file(OUT / "INPUT_AUDIT.json") != frozen["input_audit_sha256"] \
            or sha_file(OUT / "GENERATION_FREEZE.json") != frozen["generation_freeze_sha256"] \
            or str(OUT.relative_to(ROOT)) != frozen["output_relative_path"] \
            or str(CACHE) != frozen["cache_path"]:
        raise RuntimeError("Frozen code or data changed")
    if assets() != frozen["assets"]:
        raise RuntimeError("Execution assets changed")
    configs = {m: ProtocolConfig(**c) for m, c in frozen["configurations"].items()}
    plans, unique, injection = inputs(configs)
    if digest(plans) != frozen["plan_sha256"] or digest(injection) != frozen["injection_manifest_sha256"]:
        raise RuntimeError("Frozen candidate geometry changed")
    return frozen, plans, unique, injection


def readiness():
    frozen, plans, unique, _ = verify()
    parser = AuditedParser()
    bank = bank_load()
    init_matcher(bank)
    import torch
    torch.cuda.reset_peak_memory_stats()
    timing, replay, estimate = [], True, 0.
    for method in ("P0", frozen["candidate_method"]):
        keys = list(dict.fromkeys(k for p in plans if p["method"] == method
                                  for k in p["candidate_keys"]))
        # Full longest valid batch plus deterministic spread for each method.
        valid_lengths = {}
        for i in range(0, len(keys), 1024):
            check_time()
            batch = keys[i:i+1024]
            lengths = parser.token_counts([unique[k] for k in batch])
            valid_lengths.update((k, n) for k, n in zip(batch, lengths)
                                 if n <= parser.model.max_sent_len)
        valid = sorted(valid_lengths, key=lambda k: (valid_lengths[k], k), reverse=True)
        representatives = list(dict.fromkeys(valid[:8] +
            [valid[(len(valid)-1)*i//7] for i in range(8)]))
        if not representatives:
            raise RuntimeError("No valid representative input")
        texts = [unique[k] for k in representatives]
        start = time.monotonic()
        records = parser.parse(texts)
        parse_seconds = time.monotonic()-start
        second = parser.parse(texts)
        replay &= records == second
        start = time.monotonic()
        first = [score_graph(r["graph"]) for r in records if r["graph"]]
        score_seconds = time.monotonic()-start
        second_scores = [score_graph(r["graph"]) for r in records if r["graph"]]
        replay &= bool(first) and any(r["strict_ok"] for r in records) \
            and first == second_scores and all(s["scores"] is not None for s in first)
        seconds = len(keys) * (parse_seconds/len(texts)
                              + score_seconds/max(1, len(first))/4)
        estimate += seconds
        timing.append({"method": method, "unique_inputs": len(keys),
                       "pilot_inputs": len(texts), "parser_seconds": parse_seconds,
                       "single_worker_score_seconds": score_seconds,
                       "conservative_no_reuse_seconds": seconds})
    free, total = torch.cuda.mem_get_info()
    peak = torch.cuda.max_memory_allocated()
    # Safety factor, and a fixed 20-minute allowance for aggregation/archival.
    estimate = 1.25 * estimate
    value = {"created_at": now(), "freeze_sha256": digest(frozen),
             "cuda": True, "dtype": "float32", "exact_parser_and_score_replay": replay,
             "longest_valid_full_batch_checked": True, "timing": timing,
             "conservative_estimate_seconds": estimate, "remaining_seconds": remaining(),
             "archive_reserve_seconds": 1200, "gpu_peak_allocated_bytes": peak,
             "gpu_total_bytes": total, "gpu_free_after_pilot_bytes": free,
             "native_context_rejections": frozen["native_context_rejections"],
             "ready": replay and peak < .8*total and total-free < .8*total
             and estimate + 1200 < remaining()}
    save_json(OUT / "READINESS.json", value)
    if not value["ready"]:
        save_json(OUT / "READINESS_STOP.json", {
            "at": now(), "permanent": True, "reason": "frozen_resource_or_replay_gate",
            "no_successor": True, "ready": value})
        return False
    return True


def summaries(plans, cache):
    output = []
    for p in plans:
        failures = Counter(p["input_failure_reasons"])
        maxima, raw = [], []
        raw_unavailable = bool(failures)
        for key in p["candidate_keys"]:
            r = cache.parses[key]
            s = cache.scores.get(text_sha(r["graph"])) if r.get("graph") else None
            score_ok = s is not None and s["scores"] is not None and len(s["scores"]) == 100
            if not r["strict_ok"]:
                failures.update(r["reasons"])
            if not score_ok:
                failures.update([s["reason"] if s else "missing_bank_scores"])
            maxima.append(max(s["scores"]) if score_ok and r["strict_ok"] else None)
            if r.get("blank_output"):
                raw.append(None)
            elif score_ok:
                raw.append(max(s["scores"]))
            else:
                raw_unavailable = True
        data = {k: v for k, v in p.items() if k not in (
            "candidate_keys", "input_failure_reasons")}
        if p["method"] == "P0":
            output.append({**data, "method": "P0_raw",
                           "record_id": p["record_id"]+":raw", "complete": not raw_unavailable,
                           "failure_counts": {"raw_or_input_failure": 1} if raw_unavailable else {},
                           "unit_bank_max": raw if not raw_unavailable else []})
            data["method"] = "P0_strict"
        output.append({**data, "record_id": p["record_id"]+":strict",
                       "complete": not failures, "failure_counts": dict(failures),
                       "unit_bank_max": maxima if not failures else []})
    return output


def run():
    frozen, plans, unique, injection = verify()
    ready = json.loads((OUT / "READINESS.json").read_text())
    if not ready["ready"] or ready["freeze_sha256"] != digest(frozen):
        raise RuntimeError("Readiness unavailable")
    cache = Cache(CACHE, digest(frozen))
    parser, bank = AuditedParser(), bank_load()
    start = time.monotonic()
    def beat(stage, done, expected):
        save_json(OUT / "heartbeat.json", {
            "at": now(), "state": "running", "stage": stage, "done": done,
            "expected": expected, "parsed_unique_inputs": len(cache.parses),
            "scored_unique_graphs": len(cache.scores), "elapsed_seconds": time.monotonic()-start,
            "remaining_seconds": remaining(), "freeze_sha256": digest(frozen)})
    todo = [k for k in unique if k not in cache.parses]
    for i in range(0, len(todo), SETTINGS["batch_size"]):
        check_time()
        batch = todo[i:i+SETTINGS["batch_size"]]
        lengths = parser.token_counts([unique[k] for k in batch])
        valid = []
        for k, n in zip(batch, lengths):
            if n > parser.model.max_sent_len:
                cache.put("parses", k, {"input_sha256": k, "native_input_tokens": n,
                    "strict_ok": False, "reasons": ["native_context_exceeded"],
                    "graph": None, "blank_output": False})
            else:
                valid.append(k)
        if valid:
            for key, record in zip(valid, parser.parse([unique[k] for k in valid])):
                cache.put("parses", key, record)
        beat("parse", min(i+SETTINGS["batch_size"], len(todo)), len(todo))
    graphs = {text_sha(r["graph"]): r["graph"] for r in cache.parses.values() if r.get("graph")}
    todo = [k for k in graphs if k not in cache.scores]
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn"),
                             initializer=init_matcher, initargs=(bank,)) as pool:
        pending, index, finished = {}, 0, 0
        while index < len(todo) or pending:
            check_time()
            while index < len(todo) and len(pending) < 32:
                key = todo[index]
                pending[pool.submit(score_graph, graphs[key])] = key
                index += 1
            completed, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
            for future in completed:
                key = pending.pop(future)
                cache.put("scores", key, future.result())
                finished += 1
            beat("score", finished, len(todo))
    save_json(OUT / "DOCUMENT_SCORES.json", {
        "freeze_sha256": digest(frozen), "documents": summaries(plans, cache)})
    target_scores, bank_maxima, failures = [], [], Counter()
    for slot in injection:
        r = cache.parses.get(slot["key"])
        s = cache.scores.get(text_sha(r["graph"])) if r and r.get("graph") else None
        if r and r["strict_ok"] and s and s["scores"] is not None:
            target_scores.append(s["scores"][slot["target"]])
            bank_maxima.append(max(s["scores"]))
        else:
            failures.update(r["reasons"] if r else ["input_ineligible"])
    save_json(OUT / "INJECTION_DIAGNOSTIC.json", {
        "frozen_positive_slots": 800, "strict_scored_slots": len(target_scores),
        "target_mean_score": sum(target_scores)/len(target_scores) if target_scores else None,
        "bankmax_mean_score": sum(bank_maxima)/len(bank_maxima) if bank_maxima else None,
        "target_green_ge065": sum(s >= .65 for s in target_scores),
        "bankmax_green_ge065": sum(s >= .65 for s in bank_maxima),
        "full_denominator": 800, "failure_counts": dict(failures),
        "used_for_selection_or_calibration": False,
    })
    analyze(OUT)
    save_json(OUT / "heartbeat.json", {
        "at": now(), "state": "complete", "freeze_sha256": digest(frozen),
        "parsed_unique_inputs": len(cache.parses), "scored_unique_graphs": len(cache.scores),
        "elapsed_seconds": time.monotonic()-start})


def workflow():
    try:
        if not (OUT / "EVAL_FREEZE.json").exists():
            freeze()
        if not (OUT / "READINESS.json").exists():
            if not readiness():
                return
        run()
    except TimeoutError:
        save_json(OUT / "DEADLINE_STOP.json", {
            "at": now(), "permanent": True, "reason": "final_24h_budget_expired",
            "no_successor": True})
    except Exception as exc:
        save_json(OUT / "HALT.json", {
            "at": now(), "permanent": True, "exception_type": type(exc).__name__,
            "reason": "frozen_confirmation_execution_failure", "no_successor": True})
        raise


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--phase", choices=("freeze", "readiness", "run", "workflow"), required=True)
    cli.add_argument("--output", type=Path, default=OUT)
    cli.add_argument("--cache", type=Path, default=CACHE)
    args = cli.parse_args()
    OUT, CACHE = args.output.resolve(), args.cache.resolve()
    {"freeze": freeze, "readiness": readiness, "run": run, "workflow": workflow}[args.phase]()
