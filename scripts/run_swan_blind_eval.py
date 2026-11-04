"""Fresh Linux CUDA evaluation with asset freeze, persistent cache and heartbeat."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core.swan_blind_protocol import (
    METHODS, ProtocolConfig, configuration_fingerprint, digest, plan_document,
    segmenter_metadata)
from src.core.swan_blind_runtime import (
    SETTINGS, OFFICIAL, RUNTIME, AuditedParser, BoundAuditError, graph_valid,
    init_matcher, score_graph, text_sha)
from scripts.prepare_swan_blind_definitions import ATTACKS, EXP, load_attack_functions
from scripts.analyze_swan_blind_eval import analyze, save_json

OUTPUT = EXP / "blind_eval1/run2"
OLD_SHA = "906a23abfe7ee9fecfff7b186e7faac9c85e26164978b5d7dc8f0542a6417578"


def sha_file(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def assets():
    import amrlib
    import en_core_web_sm
    import nltk
    nltk.data.path.insert(0, str(ROOT / ".nltk_data"))
    from nltk.corpus import propbank
    # Trigger corpus path resolution before recording the actual archive.
    propbank.ensure_loaded()
    paths = [
        ROOT / "src/core/swan_blind_protocol.py", ROOT / "src/core/multiscale_blind.py",
        ROOT / "src/core/swan_blind_runtime.py",
        ROOT / "scripts/prepare_swan_blind_definitions.py",
        ROOT / "scripts/run_swan_blind_eval.py",
        ROOT / "scripts/analyze_swan_blind_eval.py",
        ROOT / "scripts/run_dataset4_attack_recovery.py", EXP / "PROTOCOL.md",
        EXP / "dataset5/documents.json", EXP / "dataset5_paired_neg/documents.json",
        OFFICIAL / "utils/amr_utils.py",
        OFFICIAL / "amr_bank/banks/amr_bank_100.json",
        OFFICIAL / "detection/parse_machine_text.py",
        OFFICIAL / "detection/detect_from_parsed_amrs.py",
        ROOT / "vectors/glove.6B.100d.txt", ROOT / ".nltk_data/corpora/propbank.zip",
    ]
    folders = [
        ("model", ROOT / "models/model_parse_xfm_bart_large-v0_1_0"),
        ("tokenizer", ROOT / "models/bart-large-tokenizer"),
        ("s2match", RUNTIME),
        ("amrlib", Path(amrlib.__file__).parent),
        ("spacy_model", Path(en_core_web_sm.__file__).parent),
    ]
    hashes = {str(p.relative_to(ROOT)): sha_file(p) for p in paths}
    groups = {}
    for label, folder in folders:
        group = {str(p.relative_to(folder)): sha_file(p) for p in sorted(folder.rglob("*"))
                 if p.is_file() and "__pycache__" not in p.parts
                 and ".git" not in p.parts and p.suffix not in (".pyc", ".log")
                 and not p.name.startswith(".")}
        if not group:
            raise RuntimeError("Missing frozen resource group: " + label)
        groups[label] = {"files_sha256": group, "fingerprint": digest(group)}
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "amrlib", "penman", "spacy", "nltk",
                 "numpy", "scipy", "tokenizers", "safetensors")}
    return {"files_sha256": hashes, "resource_groups": groups,
            "versions": versions, "punkt": segmenter_metadata(), "settings": SETTINGS}


def bank_load():
    path = OFFICIAL / "amr_bank/banks/amr_bank_100.json"
    entries = json.loads(path.read_text())
    bank = [e["normalized_template_amr"] for e in entries]
    if len(bank) != 100 or len(set(bank)) != 100 or not all(graph_valid(g) for g in bank):
        raise RuntimeError("Bank missing, duplicate or invalid")
    return bank


def configs_for(asset):
    groups = asset["resource_groups"]
    parser_sha = digest({"model": groups["model"]["fingerprint"],
                         "amrlib": groups["amrlib"]["fingerprint"],
                         "versions": asset["versions"], "settings": SETTINGS})
    norm_sha = digest({
        "official": asset["files_sha256"][
            "third_party/SWAN_official_2ca0c207_clean/utils/amr_utils.py"],
        "spacy": groups["spacy_model"]["fingerprint"],
        "propbank": asset["files_sha256"][".nltk_data/corpora/propbank.zip"],
        "adapter": asset["files_sha256"]["src/core/swan_blind_runtime.py"]})
    match_sha = digest({
        "runtime": groups["s2match"]["fingerprint"],
        "vectors": asset["files_sha256"]["vectors/glove.6B.100d.txt"],
        "settings": SETTINGS["score"], "versions": asset["versions"]})
    model_config = json.loads((ROOT /
        "models/model_parse_xfm_bart_large-v0_1_0/config.json").read_text())
    context = model_config["task_specific_params"]["parse_amr"]["max_in_len"]
    return {m: ProtocolConfig(
        m, batch_size=SETTINGS["batch_size"], bank_sha256=asset["files_sha256"][
            "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_100.json"],
        parser_sha256=parser_sha, parser_tokenizer_sha256=groups["tokenizer"]["fingerprint"],
        normalizer_sha256=norm_sha, matcher_sha256=match_sha,
        parser_context_tokens=context) for m in METHODS}


def cohort(configs):
    docs = {arm: json.loads((EXP / name / "documents.json").read_text())["documents"]
            for arm, name in (("positive", "dataset5"),
                              ("paired_negative", "dataset5_paired_neg"))}
    ids = {arm: sorted(d["document_id"] for d in data) for arm, data in docs.items()}
    if ids["positive"] != ids["paired_negative"] or len(ids["positive"]) != 30 \
            or len(set(ids["positive"])) != 30:
        raise RuntimeError("Frozen cohort mismatch")
    negative_by_id = {d["document_id"]: d for d in docs["paired_negative"]}
    if any(len(d["slots"]) != len(negative_by_id[d["document_id"]]["slots"])
           for d in docs["positive"]):
        raise RuntimeError("Frozen paired slot counts mismatch")
    calibration_ids = set(ids["paired_negative"][:15])
    attacks = load_attack_functions()
    plans, unique = [], {}
    for method in METHODS:
        for arm, data in docs.items():
            for document in sorted(data, key=lambda d: d["document_id"]):
                for attack in ATTACKS:
                    # Oracle data is restricted to attack construction.
                    text, _ = attacks[attack]([{"text": s["text"]}
                                              for s in document["slots"]])
                    plan = plan_document(text, configs[method])
                    if plan.status != "geometry_ready" or plan.uncovered_nonwhitespace:
                        raise RuntimeError("Text geometry readiness failed")
                    keys = []
                    for candidate in plan.candidates:
                        key = text_sha(candidate.text)
                        if key in unique and unique[key] != candidate.text:
                            raise RuntimeError("Candidate digest collision")
                        unique[key] = candidate.text
                        keys.append(key)
                    plans.append({
                        "record_id": f"{method}:{arm}:{document['document_id']}:{attack}",
                        "method": method, "arm": arm, "document_id": document["document_id"],
                        "attack": attack, "split": "calibration" if
                        document["document_id"] in calibration_ids else "test",
                        "text_sha256": plan.text_sha256,
                        "configuration_sha256": plan.configuration_sha256,
                        "N_expected": len(keys), "candidate_keys": keys})
    return plans, unique, ids


def freeze():
    if (OUTPUT / "FREEZE.json").exists():
        raise RuntimeError("Freeze already exists; verify it instead of replacing")
    if sha_file(EXP / "dataset5_eval/PROTOCOL_RESULTS.json") != OLD_SHA:
        raise RuntimeError("Historical scores changed")
    asset = assets()
    configs = configs_for(asset)
    plans, unique, ids = cohort(configs)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "models/bart-large-tokenizer"))
    keys, lengths = list(unique), {}
    for i in range(0, len(keys), 1024):
        batch = keys[i:i + 1024]
        tokens = tokenizer([unique[k] for k in batch], truncation=False, padding=False)
        lengths.update((k, len(t)) for k, t in zip(batch, tokens["input_ids"]))
    bank_load()
    manifest = {
        "schema": "swan-blind-eval-freeze/v1", "created_at": now(), "assets": asset,
        "configurations": {m: asdict(c) for m, c in configs.items()},
        "configuration_sha256": {m: configuration_fingerprint(c)
                                 for m, c in configs.items()},
        "source_ids": ids, "calibration_ids": ids["paired_negative"][:15],
        "primary_test_ids": ids["paired_negative"][15:],
        "evaluation_methods": ["P0_raw", "P0_strict", "S6", "S8", "S10", "S12"],
        "attacks": ATTACKS,
        "input_manifest_sha256": digest([{k: v for k, v in p.items()
                                         if k != "candidate_keys"} for p in plans]),
        "plan_manifest_sha256": digest(plans),
        "document_plans": len(plans), "candidate_unique_inputs": len(unique),
        "candidate_totals": dict(Counter({
            m: sum(p["N_expected"] for p in plans if p["method"] == m) for m in METHODS})),
        "native_input_max_tokens": max(lengths.values()),
        "native_context_rejections": sum(n > configs["P0"].parser_context_tokens
                                         for n in lengths.values()),
        "score_workers": 4, "parser_batch_size": SETTINGS["batch_size"],
        "predecessor": "blind_eval1:permanent-first-batch-CUDA-OOM-no-committed-results",
        "private_cache_policy": "fresh-exact-input-and-normalized-graph-under-freeze",
        "historical_scores_sha256": OLD_SHA,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_json(OUTPUT / "FREEZE.json", manifest)
    return manifest


def verify():
    frozen = json.loads((OUTPUT / "FREEZE.json").read_text())
    if frozen["assets"] != assets() or sha_file(EXP /
        "dataset5_eval/PROTOCOL_RESULTS.json") != OLD_SHA:
        raise RuntimeError("Frozen resource/source changed")
    configs = {m: ProtocolConfig(**c) for m, c in frozen["configurations"].items()}
    plans, unique, _ = cohort(configs)
    if digest(plans) != frozen["plan_manifest_sha256"]:
        raise RuntimeError("Frozen plan manifest changed")
    return frozen, configs, plans, unique


def readiness():
    frozen, configs, plans, unique = verify()
    parser = AuditedParser()
    bank = bank_load()
    init_matcher(bank)
    # Time long sentences separately; a long P0 input must not inflate every
    # short-window estimate through amrlib's batch padding and beam decoding.
    strata = []
    for method in METHODS:
        keys = list(dict.fromkeys(k for p in plans if p["method"] == method
                                  for k in p["candidate_keys"]))
        ranked = sorted(keys, key=lambda k: len(unique[k]), reverse=True)
        representatives = ranked[:2] + ranked[len(ranked) // 2:len(ranked) // 2 + 2] \
            + ranked[-2:]
        strata.append((method, len(keys), list(dict.fromkeys(representatives))))
    import torch
    torch.cuda.reset_peak_memory_stats()
    records, timing = [], []
    parser_seconds, score_seconds, replay_ok = 0., 0., True
    hours = 0.
    for method, count, representatives in strata:
        texts = [unique[k] for k in representatives]
        started = time.monotonic()
        trial = parser.parse(texts)
        parsed_seconds = time.monotonic() - started
        started = time.monotonic()
        first = [score_graph(r["graph"]) for r in trial if r["graph"]]
        scored_seconds = time.monotonic() - started
        replay = [score_graph(r["graph"]) for r in trial if r["graph"]]
        replay_ok &= first == replay and all(r["scores"] is not None for r in first)
        parser_seconds += parsed_seconds
        score_seconds += scored_seconds
        # Summing per-scale unique counts overcounts cross-scale cache reuse.
        estimate = count * (parsed_seconds / len(texts)
                            + scored_seconds / max(len(first), 1) / 4) / 3600
        hours += estimate
        timing.append({"method": method, "unique_inputs": count,
                       "real_candidates": len(texts), "parser_seconds": parsed_seconds,
                       "score_seconds_single_worker": scored_seconds,
                       "estimated_hours_without_cross_scale_cache_reuse": estimate})
        records.extend(trial)
    # Verify full batches of the longest observed inputs and short windows.
    window_keys = list(dict.fromkeys(k for p in plans if p["method"] == "S12"
                                    for k in p["candidate_keys"]))[:SETTINGS["batch_size"]]
    parser.parse([unique[k] for k in window_keys])
    p0_keys = list(dict.fromkeys(k for p in plans if p["method"] == "P0"
                                for k in p["candidate_keys"]))
    p0_keys.sort(key=lambda k: len(unique[k]), reverse=True)
    parser.parse([unique[k] for k in p0_keys[:SETTINGS["batch_size"]]])
    free, total = torch.cuda.mem_get_info()
    peak = torch.cuda.max_memory_allocated()
    result = {
        "freeze_sha256": digest(frozen), "created_at": now(),
        "input_and_assets_verified": True, "cuda": True, "dtype": "float32",
        "native_input_max_tokens": frozen["native_input_max_tokens"],
        "native_context_rejections": frozen["native_context_rejections"],
        "native_context_tokens": parser.model.max_sent_len,
        "native_output_tokens": parser.model.max_graph_len,
        "real_candidates": len(records), "bound_audit_pass": True,
        "strict_valid_candidates": sum(r["strict_ok"] for r in records),
        "candidate_failure_reasons": dict(Counter(
            reason for r in records for reason in r["reasons"])),
        "discarded_beam_events": sum(
            e.get("beam") is not None and e["beam"] != r["selected_beam"]
            for r in records for e in r["events"]),
        "numeric_replay_exact": replay_ok, "parser_seconds": parser_seconds,
        "score_seconds_single_worker": score_seconds,
        "timing_by_scale": timing, "full_batch_memory_checked": SETTINGS["batch_size"],
        "longest_p0_full_batch_checked": True,
        "conservative_full_run_estimate_hours": hours,
        "gpu_peak_allocated_bytes": peak, "gpu_total_bytes": total,
        "gpu_free_after_pilot_bytes": free,
        "ready": replay_ok and peak < .8 * total and hours <= 48,
        "no_old_parse_or_slot_cache": True,
    }
    save_json(OUTPUT / "READINESS.json", result)
    if not result["ready"]:
        raise RuntimeError("Readiness gate failed")
    return result


class Cache:
    def __init__(self, folder, fingerprint):
        folder.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(folder / "cache.sqlite")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS parses (key TEXT PRIMARY KEY, value TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, value TEXT)")
        old = self.db.execute("SELECT value FROM meta WHERE key='freeze'").fetchone()
        if old and old[0] != fingerprint:
            raise RuntimeError("Private cache belongs to another freeze")
        self.db.execute("INSERT OR IGNORE INTO meta VALUES ('freeze', ?)", (fingerprint,))
        self.db.commit()
        self.parses = {k: json.loads(v) for k, v in self.db.execute("SELECT * FROM parses")}
        self.scores = {k: json.loads(v) for k, v in self.db.execute("SELECT * FROM scores")}

    def put(self, table, key, value):
        if table not in ("parses", "scores"):
            raise ValueError("Invalid cache table")
        self.db.execute(f"INSERT OR REPLACE INTO {table} VALUES (?, ?)",
                        (key, json.dumps(value, allow_nan=False)))
        self.db.commit()
        getattr(self, table)[key] = value


def document_scores(plans, cache, completed_methods):
    rows = []
    for p in plans:
        if p["method"] not in completed_methods:
            continue
        failures, maxima, raw, native, audit_counts = Counter(), [], [], [], Counter()
        raw_unavailable = False
        for key in p["candidate_keys"]:
            r = cache.parses[key]
            native.append(r["native_input_tokens"])
            audit_counts["selected_beam_events"] += sum(
                e.get("beam") == r.get("selected_beam") for e in r.get("events", []))
            audit_counts["discarded_beam_events"] += sum(
                e.get("beam") is not None and e["beam"] != r.get("selected_beam")
                for e in r.get("events", []))
            s = cache.scores.get(text_sha(r["graph"])) if r.get("graph") else None
            score_ok = s is not None and s["scores"] is not None and len(s["scores"]) == 100
            if not r["strict_ok"]:
                failures.update(r["reasons"])
            if not score_ok:
                failures.update([s["reason"] if s else "missing_bank_scores"])
            maxima.append(max(s["scores"]) if r["strict_ok"] and score_ok else None)
            if r.get("blank_output"):
                raw.append(None)
            elif score_ok:
                raw.append(max(s["scores"]))
            else:
                raw_unavailable = True
                raw.append(None)
        metadata = {k: v for k, v in p.items() if k != "candidate_keys"}
        metadata.update(
            native_input_max_tokens=max(native), candidate_audit_counts=dict(audit_counts))
        if p["method"] == "P0":
            rows.append({
                **metadata, "method": "P0_raw", "record_id": p["record_id"] + ":raw",
                "complete": not raw_unavailable, "failure_counts": {
                    "raw_nonblank_scoring_failure": 1} if raw_unavailable else {},
                "blank_candidates": sum(s is None for s in raw),
                "unit_bank_max": raw if not raw_unavailable else []})
            method = "P0_strict"
        else:
            method = p["method"]
        rows.append({**metadata, "method": method,
                     "record_id": p["record_id"] + ":strict",
                     "complete": not failures, "failure_counts": dict(failures),
                     "unit_bank_max": maxima if not failures else [],
                     "N_valid_candidates": sum(v is not None for v in maxima)})
    return rows


def run(cache_path):
    if (OUTPUT / "HALT.json").exists():
        raise RuntimeError("Permanent halt prohibits replay of this run")
    frozen, _, plans, unique = verify()
    ready = json.loads((OUTPUT / "READINESS.json").read_text())
    if not ready["ready"] or ready["freeze_sha256"] != digest(frozen):
        raise RuntimeError("Readiness does not bind this freeze")
    cache = Cache(cache_path, digest(frozen))
    parser, bank, completed = AuditedParser(), bank_load(), []
    if (OUTPUT / "STATE.json").exists():
        completed = json.loads((OUTPUT / "STATE.json").read_text())["completed_methods"]
    started = time.monotonic()

    def heartbeat(stage, method, done, expected):
        save_json(OUTPUT / "heartbeat.json", {
            "updated_at": now(), "state": "running", "stage": stage, "method": method,
            "stage_completed": done, "stage_expected": expected,
            "unique_inputs_parsed": len(cache.parses), "unique_graphs_scored": len(cache.scores),
            "completed_methods": completed, "elapsed_seconds": time.monotonic() - started,
            "freeze_sha256": digest(frozen)})

    try:
        with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=init_matcher, initargs=(bank,)) as pool:
            for method in METHODS:
                if method in completed:
                    continue
                keys = list(dict.fromkeys(k for p in plans if p["method"] == method
                                          for k in p["candidate_keys"]))
                todo = [k for k in keys if k not in cache.parses]
                batch_size = SETTINGS["batch_size"]
                for i in range(0, len(todo), batch_size):
                    batch = todo[i:i + batch_size]
                    counts = parser.token_counts([unique[k] for k in batch])
                    valid = [k for k, n in zip(batch, counts)
                             if n <= parser.model.max_sent_len]
                    for k, n in zip(batch, counts):
                        if n > parser.model.max_sent_len:
                            cache.put("parses", k, {
                                "input_sha256": k, "native_input_tokens": n,
                                "strict_ok": False, "reasons": ["native_context_exceeded"],
                                "graph": None, "blank_output": False})
                    if valid:
                        records = parser.parse([unique[k] for k in valid])
                        for key, record in zip(valid, records):
                            cache.put("parses", key, record)
                    heartbeat("parse", method, min(i + batch_size, len(todo)), len(todo))
                graphs = {text_sha(cache.parses[k]["graph"]): cache.parses[k]["graph"]
                          for k in keys if cache.parses[k].get("graph")}
                todo_graphs = [g for g in graphs if g not in cache.scores]
                pending, index, finished = {}, 0, 0
                while index < len(todo_graphs) or pending:
                    while index < len(todo_graphs) and len(pending) < 32:
                        key = todo_graphs[index]
                        pending[pool.submit(score_graph, graphs[key])] = key
                        index += 1
                    ready_futures, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
                    for future in ready_futures:
                        key = pending.pop(future)
                        cache.put("scores", key, future.result())
                        finished += 1
                    heartbeat("score", method, finished, len(todo_graphs))
                completed.append(method)
                save_json(OUTPUT / "DOCUMENT_SCORES.json", {
                    "freeze_sha256": digest(frozen),
                    "documents": document_scores(plans, cache, completed)})
                save_json(OUTPUT / "STATE.json", {"completed_methods": completed,
                                                "updated_at": now()})
                analyze(OUTPUT)
                print(json.dumps({"method_completed": method, "timestamp": now()}), flush=True)
        save_json(OUTPUT / "heartbeat.json", {
            "updated_at": now(), "state": "complete", "completed_methods": completed,
            "unique_inputs_parsed": len(cache.parses), "unique_graphs_scored": len(cache.scores),
            "freeze_sha256": digest(frozen)})
    except Exception as exc:
        save_json(OUTPUT / "HALT.json", {
            "timestamp": now(), "permanent": True, "exception_type": type(exc).__name__,
            "reason": str(exc) if isinstance(exc, BoundAuditError) else "execution_failure",
            "completed_methods": completed, "freeze_sha256": digest(frozen)})
        raise


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--phase", choices=("freeze", "readiness", "run"), required=True)
    cli.add_argument("--cache", type=Path, default=Path("/tmp/swan_blind_eval1_run2"))
    args = cli.parse_args()
    if args.phase == "freeze":
        print(json.dumps(freeze(), indent=2))
    elif args.phase == "readiness":
        print(json.dumps(readiness(), indent=2))
    else:
        run(args.cache)
