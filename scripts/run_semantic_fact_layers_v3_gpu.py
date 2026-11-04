"""CUDA-only raw AMR worker; never receives planned symbols or model labels."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata as metadata
import json
import platform
from pathlib import Path
import sys
import time
import warnings

from semantic_fact_raw_amr import audit_raw, replay_signature, validate_inputs
from run_multiscale_transfer import Parser, atomic, now, read, sha


def run(root, out):
    frozen = read(out / "FREEZE.json")
    freeze_sha = sha(out / "FREEZE.json")
    private = out / "private"
    if (out / "HALT.json").exists():
        raise RuntimeError("Permanent halt exists; this manifest cannot resume")
    if (private / "INFLIGHT_CASE.json").exists():
        raise RuntimeError("Unresolved interrupted inference; no silent retry")
    if sha(private / "PARSER_INPUTS_NO_TARGET.json") != frozen["input_sha256"]:
        raise RuntimeError("GPU input SHA differs")
    for name, expected in frozen["gpu_code_sha256"].items():
        if sha(private / "code" / name) != expected:
            raise RuntimeError("GPU code SHA differs")
    if sha(out / "PROTOCOL.md") != frozen["protocol_sha256"]:
        raise RuntimeError("Protocol SHA differs")
    rows = validate_inputs(read(private / "PARSER_INPUTS_NO_TARGET.json"))
    for asset in frozen["parser"]["assets"]:
        if sha(root / asset["path"]) != asset["sha256"]:
            raise RuntimeError("Model/tokenizer asset SHA differs")
    import torch
    if platform.system() != "Linux" or not torch.cuda.is_available():
        raise RuntimeError("Linux NVIDIA CUDA required")
    device = torch.cuda.get_device_properties(0)
    if "3090" not in device.name:
        raise RuntimeError("Frozen RTX3090 resource differs")
    started = time.monotonic()
    completed, replays = {}, {}
    for row in rows:
        path = private / "raw_amr" / (row["case_id"] + ".json")
        if path.exists():
            record = read(path)
            audit_raw(record, row, freeze_sha)
            completed[row["case_id"]] = record
    for row in rows[:2]:
        path = private / "replays" / (row["case_id"] + ".json")
        if path.exists():
            record = read(path)
            audit_raw(record, row, freeze_sha)
            if row["case_id"] not in completed or replay_signature(record) != replay_signature(completed[row["case_id"]]):
                raise RuntimeError("Persisted replay differs")
            replays[row["case_id"]] = record
    prior_seconds = read(out / "heartbeat.json").get("elapsed_seconds", 0) if (out / "heartbeat.json").exists() else 0
    peak_fraction = 0

    def memory_gate():
        nonlocal peak_fraction
        free, total = torch.cuda.mem_get_info()
        fraction = (total - free) / total
        peak_fraction = max(peak_fraction, fraction)
        if fraction >= frozen["resource_gate"]["total_vram_fraction_strictly_below"]:
            raise RuntimeError("Total CUDA VRAM gate failed")
        return fraction

    def elapsed():
        return prior_seconds + time.monotonic() - started

    def heartbeat(status):
        atomic(out / "heartbeat.json", {
            "at": now(), "status": status, "completed": len(completed), "total": 192,
            "replay_completed": len(replays), "elapsed_seconds": elapsed(),
            "peak_total_vram_fraction": peak_fraction,
            "freeze_sha256": freeze_sha,
        })

    memory_gate()
    atomic(out / "READINESS.json", {
        "at": now(), "status": "PASS_GPU_INPUT_ASSET_CODE_GATES",
        "os": platform.system(), "gpu": device.name, "total_vram_bytes": device.total_memory,
        "packages": {name: metadata.version(name) for name in ("torch", "amrlib", "transformers", "penman")},
        "cuda": torch.version.cuda, "assets_sha_verified": len(frozen["parser"]["assets"]),
        "inputs": 192, "input_and_code_sha_verified": True,
        "initial_total_vram_fraction": peak_fraction,
    })
    heartbeat("LOADING_PARSER")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parser = Parser(root, out, frozen)
    atomic(private / "LOAD_LIBRARY_WARNINGS.json", [
        {"category": w.category.__name__, "message": str(w.message)} for w in caught])
    if any(not (w.category is FutureWarning and "torch.load" in str(w.message)
                and "weights_only=False" in str(w.message)) for w in caught):
        raise RuntimeError("Unexpected unbound model-load warning")
    memory_gate()
    heartbeat("PARSING")

    def infer(row, phase):
        if elapsed() >= frozen["resource_gate"]["seconds"]:
            raise RuntimeError("Frozen replay/run time budget exhausted")
        memory_gate()
        atomic(private / "INFLIGHT_CASE.json", {"case_id": row["case_id"], "phase": phase, "at": now()})
        with warnings.catch_warnings(record=True) as inference_warnings:
            warnings.simplefilter("always")
            record = parser.fresh(row["input_text"])
        if inference_warnings:
            atomic(private / "UNBOUND_PYTHON_WARNINGS.json", [
                {"category": w.category.__name__, "message": str(w.message)} for w in inference_warnings])
            raise RuntimeError("Unbound Python inference warning; permanent halt")
        record["case_id"] = row["case_id"]
        record["phase"] = phase
        record["audit"] = audit_raw(record, row, freeze_sha)
        record["total_vram_fraction_after"] = memory_gate()
        # Extra graph decoding is diagnostic only, preserving original PENMAN.
        decode_events = []
        graph = None
        if record["amr"]:
            with warnings.catch_warnings(record=True) as graph_warnings:
                warnings.simplefilter("always")
                try:
                    with parser.capture({"phase": "inventory", "span_index": 0, "beam_index": None}, decode_events):
                        graph = parser.penman.decode(record["amr"], model=parser.amr_model)
                except Exception:
                    if record["audit"]["structurally_eligible"]:
                        raise
            if graph_warnings:
                raise RuntimeError("Unbound Python inventory warning")
        record["graph_inventory"] = {
            "top": graph.top if graph else None,
            "triples": [list(t) for t in graph.triples] if graph else [],
            "instances": len(graph.instances()) if graph else 0,
            "edges": len(graph.edges()) if graph else 0,
            "attributes": len(graph.attributes()) if graph else 0,
            "events": decode_events,
            "semantic_facts_inferred": False,
        }
        if decode_events and record["audit"]["structurally_eligible"]:
            raise RuntimeError("Inventory disagrees with clean audited roundtrip")
        return record

    for index, row in enumerate(rows):
        case_id = row["case_id"]
        if case_id not in completed:
            record = infer(row, "primary")
            atomic(private / "raw_amr" / (case_id + ".json"), record)
            completed[case_id] = record
            (private / "INFLIGHT_CASE.json").unlink()
            heartbeat("PARSING")
        if index < 2 and case_id not in replays:
            replay = infer(row, "determinism_replay")
            atomic(private / "replays" / (case_id + ".json"), replay)
            if replay_signature(replay) != replay_signature(completed[case_id]):
                raise RuntimeError("Exact parser replay differs; permanent halt")
            replays[case_id] = replay
            (private / "INFLIGHT_CASE.json").unlink()
            heartbeat("PARSING")
    if len(completed) != 192 or len(replays) != 2:
        raise RuntimeError("Incomplete primary/replay denominator")
    if sha(private / "PARSER_INPUTS_NO_TARGET.json") != frozen["input_sha256"]:
        raise RuntimeError("Inputs changed during inference")
    ordered = [completed[row["case_id"]] for row in rows]
    atomic(out / "PARSER_RESULT.json", {
        "at": now(), "status": "COMPLETE", "primary_parser_calls": 192,
        "replay_parser_calls": 2, "total_parser_calls": 194,
        "raw_penman_present": sum(bool(r["amr"]) for r in ordered),
        "structurally_eligible": sum(r["audit"]["structurally_eligible"] for r in ordered),
        "excluded": sum(not r["audit"]["structurally_eligible"] for r in ordered),
        "parse_status_counts": dict(Counter(r["audit"]["parse_status"] for r in ordered)),
        "exclusion_reason_counts": dict(Counter(reason for r in ordered for reason in r["audit"]["exclusion_reasons"])),
        "bound_diagnostic_rows": sum(bool(r["events"] or r["decode_events"]) for r in ordered),
        "fallback_beam_rows": sum(r["audit"]["selected_beam"] not in (None, 0) for r in ordered),
        "replay_exact_match": True, "peak_total_vram_fraction": peak_fraction,
        "elapsed_seconds": elapsed(), "sum_primary_inference_seconds": sum(r["seconds"] for r in ordered),
        "token_length_min": min(r["token_length"] for r in ordered),
        "token_length_max": max(r["token_length"] for r in ordered),
        "freeze_sha256": freeze_sha,
        "multi_sentence_semantic_completeness_validated": False,
        "semantic_fact_or_ABCD_predictions": 0,
        "provider_detector_embedding_blind_decode_calls": 0,
    })
    heartbeat("COMPLETE")
    files = [out / name for name in ("FREEZE.json", "PROTOCOL.md", "READINESS.json",
                                     "PARSER_RESULT.json", "heartbeat.json")]
    files += [p for p in private.rglob("*.json") if "code" not in p.parts]
    atomic(out / "REMOTE_RESULT_SHA256.json", {str(p.relative_to(out)): sha(p) for p in files})
    print(json.dumps(read(out / "PARSER_RESULT.json"), sort_keys=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    try:
        run(args.root, args.out)
    except Exception as exc:
        atomic(args.out / "HALT.json", {
            "at": now(), "status": "PERMANENT_HALT", "exception_type": type(exc).__name__,
            "message": str(exc), "no_silent_retry": True,
        })
        # Private traceback is unnecessary; bounded diagnostic state is preserved.
        print(json.dumps({"status": "PERMANENT_HALT", "exception_type": type(exc).__name__}))
        raise SystemExit(2)
