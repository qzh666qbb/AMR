"""Freeze, CUDA-parse and audit the first write/read gate of cycle1."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata as md
import json
import logging
import os
from pathlib import Path
import re
import sys
import time

from run_semantic_sequence_cycle import (
    ROOT, OUT, PRIVATE, BANK, atomic, digest, required_markers, sha, verify,
)


def prepare(out, private):
    import shutil
    verify(out, private)
    if (out / "PARSER_FREEZE.json").exists():
        raise RuntimeError("Parser input already frozen")
    if not (out / "GENERATION_COMPLETE.json").exists():
        raise RuntimeError("Generation incomplete")
    assignments = json.loads((private / "ASSIGNMENTS.json").read_text())
    bank = json.loads(BANK.read_text())
    inputs = []
    audit = []
    for doc_id in (0, 1, 2):
        doc = json.loads((private / "documents" / f"d{doc_id:02d}.json").read_text())
        if doc["split"] != "development":
            raise RuntimeError("Pilot must be development only")
        for slot in assignments[doc_id]["slots"]:
            i = slot["slot"]
            text = doc["arms"]["positive"]["slots"][i]
            call = json.loads((private / "calls" / f"d{doc_id:02d}-positive-s{i}.response.json").read_text())
            inputs.append({"input_id": f"d{doc_id:02d}-s{i}", "text": text,
                           "text_sha256": digest(text)})
            audit.append({"input_id": f"d{doc_id:02d}-s{i}", "doc_id": doc_id,
                          "slot": i, "template_id": slot["template_id"], "symbol": slot["symbol"],
                          "target": bank[slot["template_id"]]["normalized_template_amr"],
                          "required_markers": slot["markers"],
                          "generation_usable": call["complete"] and call["finish_reason"] == "stop" and bool(text)})
    assert len(inputs) == 24
    atomic(private / "INPUTS.json", inputs)
    atomic(private / "TARGETS.json", audit)
    old = ROOT / "experiments/semantic_sequence_stage1_run2/alignment_review1"
    old_freeze = json.loads((old / "FREEZE.json").read_text())
    snapshots = private / "snapshots"
    snapshots.mkdir(exist_ok=True)
    sources = [Path(__file__), ROOT / "scripts/run_semantic_sequence_cycle.py",
               ROOT / "scripts/audit_semantic_sequence_alignment.py",
               ROOT / "src/core/parser_recovery_audit.py"]
    for source in sources:
        shutil.copyfile(source, snapshots / source.name)
    shutil.copyfile(old / "private/propbank_verbs.json", private / "propbank_verbs.json")
    freeze = {"schema": "semantic-sequence-cycle-parser/v1",
              "generation_freeze_sha256": sha(out / "FREEZE.json"),
              "input_sha256": sha(private / "INPUTS.json"),
              "targets_sha256": sha(private / "TARGETS.json"),
              "snapshots_sha256": {p.name: sha(p) for p in snapshots.iterdir() if p.is_file()},
              "parser_assets": old_freeze["parser"]["inference_assets"],
              "normalizer_source_sha256": sha(ROOT / "third_party/SWAN_official_2ca0c207_clean/utils/amr_utils.py"),
              "propbank_verbs_sha256": sha(private / "propbank_verbs.json"),
              "inputs": 24, "dtype": "float32", "beams": 4,
              "replay_input_ids": [r["input_id"] for r in inputs[:3]],
              "vram_limit": .8, "unbound_diagnostic": "permanent_halt",
              "whole_graph_coverage_is_not_gate": True}
    atomic(out / "PARSER_FREEZE.json", freeze)


def parse(out, private, model_root):
    import fcntl
    import amrlib
    import amrlib.models.parse_xfm.inference as inference_module
    import penman
    from penman.models.amr import model as amr_model
    import torch
    from audit_semantic_sequence_alignment import recovery_namespace

    lock = (private / "parse.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / "PARSER_HALT.json").exists():
        raise RuntimeError("Permanent parser halt")
    freeze = json.loads((out / "PARSER_FREEZE.json").read_text())
    freeze_sha = sha(out / "PARSER_FREEZE.json")
    try:
        if sha(private / "INPUTS.json") != freeze["input_sha256"]:
            raise RuntimeError("Input SHA mismatch")
        for name, expected in freeze["snapshots_sha256"].items():
            if sha(private / "snapshots" / name) != expected:
                raise RuntimeError("Snapshot SHA mismatch")
        for asset in freeze["parser_assets"]:
            if sha(model_root / asset["path"]) != asset["sha256"]:
                raise RuntimeError("Model asset SHA mismatch")
        if sys.platform != "linux" or not torch.cuda.is_available():
            raise RuntimeError("Linux/CUDA required")
        torch.manual_seed(20261024)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        ns = recovery_namespace(private / "snapshots/parser_recovery_audit.py")
        events = []
        with ns["capture_recovery_logs"]({"phase": "load", "span_index": None}, events):
            model = amrlib.load_stog_model(
                model_dir=str(model_root / "models/model_parse_xfm_bart_large-v0_1_0"),
                tok_name_or_path=str(model_root / "models/bart-large-tokenizer"),
                device="cuda:0", batch_size=1, num_beams=4)
        if events:
            atomic(private / "LOAD_EVENTS.json", events)
            raise RuntimeError("Unbound load diagnostics")
        if not all(p.dtype == torch.float32 for p in model.model.parameters()):
            raise RuntimeError("FP32 required")
        values = json.loads((private / "INPUTS.json").read_text())
        lengths = {r["input_id"]: len(model.tokenizer(r["text"], truncation=False)["input_ids"])
                   for r in values}
        if max(lengths.values()) >= model.max_sent_len:
            raise RuntimeError("Native input boundary reached; no truncation")
        peak_ratio = 0

        def memory_gate():
            nonlocal peak_ratio
            free, total = torch.cuda.mem_get_info()
            ratio = (total - free) / total
            peak_ratio = max(peak_ratio, ratio)
            if ratio >= .8:
                raise RuntimeError("Global GPU memory gate")

        generate = model.model.generate
        generated_lengths = []

        def monitored(*args, **kwargs):
            memory_gate()
            output = generate(*args, **kwargs)
            memory_gate()
            for row in output:
                ids = row.tolist()
                while ids and ids[-1] == model.tokenizer.pad_token_id:
                    ids.pop()
                generated_lengths.append((len(ids), bool(ids) and ids[-1] == model.tokenizer.eos_token_id))
            return output

        model.model.generate = monitored
        records = private / "PARSES"
        records.mkdir(exist_ok=True)
        started = time.monotonic()
        replay = []

        def infer(row):
            generated_lengths.clear()
            before = time.monotonic()
            outputs, audits = ns["parse_with_recovery_audit"](model, [row["text"]], inference_module)
            audit = audits[0]
            decode_events = []
            valid = False
            with ns["capture_recovery_logs"](
                    {"phase": "roundtrip", "span_index": 0, "beam_index": None}, decode_events):
                try:
                    graph = penman.decode(outputs[0], model=amr_model)
                    again = penman.decode(penman.encode(graph, model=amr_model), model=amr_model)
                    valid = (bool(graph.instances()) and graph.top in graph.variables()
                             and graph.top == again.top and sorted(graph.triples) == sorted(again.triples))
                except Exception:
                    valid = False
            chosen_events = [e for e in audit["events"] if e["beam_index"] == audit["selected_beam"]]
            clipped = any(n >= model.max_graph_len and not eos for n, eos in generated_lengths)
            strict = valid and not chosen_events and not decode_events and audit["selected_beam"] == 0 and not clipped
            return {**row, "amr": outputs[0], "amr_sha256": digest(outputs[0]),
                    "recovery_audit": audit, "decode_events": decode_events,
                    "roundtrip_valid": valid, "output_limit_hit": clipped,
                    "strict_valid": strict, "input_tokens": lengths[row["input_id"]],
                    "seconds": time.monotonic() - before, "freeze_sha256": freeze_sha}

        # Replay readiness precedes the other 21 fixed inputs.
        for row in values[:3]:
            memory_gate()
            first, second = infer(row), infer(row)
            exact = (first["amr_sha256"] == second["amr_sha256"]
                     and first["recovery_audit"] == second["recovery_audit"]
                     and first["strict_valid"] == second["strict_valid"])
            replay.append({"input_id": row["input_id"], "exact": exact})
            atomic(records / f"{row['input_id']}.json", first)
            if not exact:
                raise RuntimeError("Replay mismatch")
        atomic(out / "CUDA_PRECHECK.json", {"exact_replay": replay, "vram_under80": peak_ratio < .8,
               "asset_sha_verified": True, "precision": "FP32",
               "max_input_tokens": max(lengths.values()), "freeze_sha256": freeze_sha})
        for i, row in enumerate(values[3:], 4):
            memory_gate()
            atomic(records / f"{row['input_id']}.json", infer(row))
            atomic(out / "parse_heartbeat.json", {"completed": i, "total": 24,
                   "elapsed_seconds": round(time.monotonic() - started, 2)})
        atomic(out / "PARSE_COMPLETE.json", {
            "inputs": 24, "replay_extra_calls": 3, "exact_replay": all(r["exact"] for r in replay),
            "elapsed_seconds": time.monotonic() - started, "peak_global_memory_ratio": peak_ratio,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "versions": {p: md.version(p) for p in ("torch", "amrlib", "transformers", "penman")},
            "freeze_sha256": freeze_sha})
        print(json.dumps({"parsed": 24, "replay": True, "seconds": round(time.monotonic() - started, 2)}))
    except BaseException as exc:
        atomic(out / "PARSER_HALT.json", {"reason_class": type(exc).__name__, "permanent": True})
        raise


def actual_markers(amr):
    import penman
    from penman.models.amr import model as amr_model
    graph = penman.decode(amr, model=amr_model)
    names = {t for s, r, t in graph.triples if r == ":name"}
    return {"unknown_nodes": sum(r == ":instance" and t == "amr-unknown" for s, r, t in graph.triples),
            "name_sizes": sorted(sum(s == node and bool(re.fullmatch(r":op\d+", r))
                                    for s, r, t in graph.triples) for node in names),
            "imperative": any(r == ":mode" and t == "imperative" for s, r, t in graph.triples),
            "polite": any(r == ":polite" and t == "+" for s, r, t in graph.triples)}


def markers_satisfied(required, actual):
    counts = Counter(actual["name_sizes"])
    return {"unknown": actual["unknown_nodes"] >= required["unknown_nodes"],
            "names": all(counts[n] >= count for n, count in Counter(required["name_sizes"]).items()),
            "imperative": not required["imperative"] or actual["imperative"],
            "polite": not required["polite"] or actual["polite"]}


def summarize(out, private):
    import re
    import shutil
    from audit_semantic_sequence_alignment import normalization_namespace, structure, covers
    freeze = json.loads((out / "PARSER_FREEZE.json").read_text())
    if not (out / "PARSE_COMPLETE.json").exists():
        raise RuntimeError("Complete parser receipt required")
    if sha(private / "TARGETS.json") != freeze["targets_sha256"]:
        raise RuntimeError("Target SHA mismatch")
    # The normalizer expects out/private; stage it entirely outside the workspace.
    normalizer_root = private / "normalizer_runtime"
    (normalizer_root / "private").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(private / "propbank_verbs.json", normalizer_root / "private/propbank_verbs.json")
    if sha(private / "propbank_verbs.json") != freeze["propbank_verbs_sha256"]:
        raise RuntimeError("Normalizer asset mismatch")
    if sha(ROOT / "third_party/SWAN_official_2ca0c207_clean/utils/amr_utils.py") != freeze["normalizer_source_sha256"]:
        raise RuntimeError("Normalizer source mismatch")
    normalize, _ = normalization_namespace(ROOT, normalizer_root)
    rows = []
    for target in json.loads((private / "TARGETS.json").read_text()):
        parsed = json.loads((private / "PARSES" / f"{target['input_id']}.json").read_text())
        if (digest(parsed["amr"]) != parsed["amr_sha256"]
                or digest(parsed["text"]) != parsed["text_sha256"]
                or parsed["freeze_sha256"] != sha(out / "PARSER_FREEZE.json")):
            raise RuntimeError("Parse identity mismatch")
        actual = actual_markers(parsed["amr"])
        checks = markers_satisfied(target["required_markers"], actual)
        checks["strict_parse"] = parsed["strict_valid"]
        checks["generation_usable"] = target["generation_usable"]
        normalized = normalize(parsed["amr"])
        whole = covers(structure(normalized), structure(target["target"]), rooted=False)
        rows.append({"input_id": target["input_id"], "doc_id": target["doc_id"],
                     "template_id": target["template_id"], "symbol": target["symbol"],
                     "required": target["required_markers"], "actual": actual,
                     "checks": checks, "g1e_success": all(checks.values()),
                     "whole_target_covered_diagnostic": whole,
                     "parse_sha256": sha(private / "PARSES" / f"{target['input_id']}.json")})
    lengths = {arm: [] for arm in ("positive", "unwatermarked")}
    for doc_id in (0, 1, 2):
        doc = json.loads((private / "documents" / f"d{doc_id:02d}.json").read_text())
        for arm in lengths:
            lengths[arm].extend(len(t.split()) for t in doc["arms"][arm]["slots"])
    count = sum(r["g1e_success"] for r in rows)
    status = "G1E_GO_FORMAL_G1_PENDING" if count >= 20 else "NO_GO_WRITE_READ_PILOT"
    result = {"status": status, "denominator": 24, "g1e_successes": count,
              "minimum_required": 20, "g1e_rate": count / 24,
              "whole_target_covered_diagnostic": sum(r["whole_target_covered_diagnostic"] for r in rows),
              "failed_checks": {key: sum(not r["checks"][key] for r in rows) for key in rows[0]["checks"]},
              "carrier_mean_words": {arm: sum(v) / len(v) for arm, v in lengths.items()},
              "formal_human_G1": "NOT_EVALUATED",
              "symbol_G2": "NOT_EVALUATED", "blind_G3": "NOT_EVALUATED", "order_G4": "NOT_EVALUATED",
              "automated_markers_are_not_semantic_truth": True,
              "no_detector_or_symbol_scores_computed": True, "rows": rows}
    atomic(out / "WRITE_READ_RESULTS.json", result)
    atomic(out / "STATUS.json", {"stage": 1, "status": status,
        "released_pairs_completed": 3, "unreleased_pairs": 21,
        "stages": {"write_read": status, "symbol_recognition": "NOT_STARTED",
                   "blind_localization": "NOT_STARTED", "order": "ENGINEERING_VERIFIED_ONLY"}})
    print(json.dumps({k: result[k] for k in ("status", "g1e_successes", "denominator",
                                           "whole_target_covered_diagnostic", "failed_checks")}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare", "parse", "summarize"])
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--private", type=Path, default=PRIVATE)
    p.add_argument("--model-root", type=Path, default=ROOT)
    args = p.parse_args()
    os.umask(0o077)
    if args.action == "parse":
        parse(args.out, args.private, args.model_root)
    else:
        globals()[args.action](args.out, args.private)


if __name__ == "__main__":
    main()
