"""Prefill only fresh text-key parser caches after the frozen stage gates."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def load(out):
    source = out / "private/code/scripts/run_multiscale_transfer.py"
    spec = importlib.util.spec_from_file_location("frozen_transfer_prefill", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.insert(0, str(out / "private/code"))
    return module


def atomic_unique(m, path, value):
    """Avoid sharing the worker's temporary filename when identical parses race."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".prefill-{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        prior = m.read(path)
        assert prior["amr_sha256"] == value["amr_sha256"] and prior["valid"] == value["valid"]
        assert prior["freeze_sha256"] == value["freeze_sha256"]
    temporary.replace(path)


def plan(root, out, workers):
    m = load(out)
    assert not (out / "PREFILL_PLAN.json").exists()
    freeze = m.read(out / "FREEZE.json")
    m.verify(out, freeze)
    m.atomic(out / "PREFILL_PLAN.json", {
        "at": m.now(), "freeze_sha256": m.sha(out / "FREEZE.json"),
        "source_sha256": m.sha(root / "scripts/prefill_multiscale_transfer.py"),
        "workers": workers, "gpu_memory_fraction_max": 0.8,
        "worker_memory_reservation_mib": 3072, "memory_headroom_mib": 768,
        "parser_config": freeze["parser"], "samples": m.read(out / "AUDIT_PLAN.json")["samples"],
        "stage_gates": ["SIGNAL_COMPLETE.json", "PRECHECK.json", "CLEAN_COMPLETE.json"],
        "scope": "Parser caches only; no scoring, fitting, threshold or input changes",
        "cache_scope": "Current run only; no predecessor or legacy cache import",
        "parallel_single_input_replay": "Every worker must match three fixed cached parse SHAs exactly",
        "on_unbound_diagnostic_or_mismatch": "Stop primary and permanent HALT"})
    print(json.dumps({"prefill_plan_written": True, "workers": workers}))


def verify(m, root, out):
    p = m.read(out / "PREFILL_PLAN.json")
    assert p["freeze_sha256"] == m.sha(out / "FREEZE.json")
    assert p["source_sha256"] == m.sha(root / "scripts/prefill_multiscale_transfer.py")
    m.verify(out, m.read(out / "FREEZE.json"))
    return p


def worker(root, out, phase, shard):
    from src.core.multiscale_blind import blind_geometry
    m = load(out)
    p = verify(m, root, out)
    freeze = m.read(out / "FREEZE.json")
    assert m.read(out / "SIGNAL_COMPLETE.json")["sentences"] == 384
    assert m.read(out / "PRECHECK.json")["passed"]
    if phase == "attacks":
        assert m.read(out / "CLEAN_GATE.json")["passed"]
        assert (out / "CLEAN_COMPLETE.json").exists()
    documents = m.read(out / "private/DOCUMENTS.json")
    docs = {d["doc_id"]: d for d in documents}
    parser = m.Parser(root, out, freeze)
    replay = []
    for sample in p["samples"]:
        arm = docs[sample["doc_id"]]["arms"][sample["arm"]]
        text = (arm["carriers"][sample["index"]]["text"] if sample["kind"] == "original_sentence"
                else blind_geometry(arm["texts"]["clean"])[sample["kind"]][sample["index"]]["text"])
        prior = m.read(out / "private/parses" / (m.digest(text) + ".json"))
        current = parser.fresh(text)
        assert current["amr_sha256"] == prior["amr_sha256"] and current["valid"] == prior["valid"]
        replay.append({**sample, "exact_parse_sha": True})
    # Text-hash assignment is independent of labels, scores, or parser quality.
    conditions = ["clean"] if phase == "clean" else freeze["conditions"][1:]
    texts = sorted(m.unique_parse_inputs(documents, conditions), key=m.digest)
    texts = [t for t in texts if int(m.digest(t), 16) % p["workers"] == shard]
    completed, hits = 0, 0
    heartbeat = out / "private/prefill" / f"{phase}_{shard}.json"
    m.atomic(heartbeat, {
        "at": m.now(), "phase": phase, "shard": shard, "assigned": len(texts),
        "processed": 0, "fresh_parses": 0, "cache_hits": 0, "fixed_sample_replays": replay,
        "freeze_sha256": parser.freeze_sha,
        "peak_reserved_bytes": parser.torch.cuda.max_memory_reserved(0)})
    for index, text in enumerate(texts):
        if (out / "COMPLETE.json").exists():
            break
        if any((out / n).exists() for n in ("HALT.json", "STOP.json")):
            raise RuntimeError("Primary stop marker observed")
        path = out / "private/parses" / (m.digest(text) + ".json")
        if path.exists():
            prior = m.read(path)
            assert prior["freeze_sha256"] == parser.freeze_sha and prior["text_sha256"] == m.digest(text)
            hits += 1
        else:
            atomic_unique(m, path, parser.fresh(text))
            completed += 1
        if index % 25 == 0 or index == len(texts) - 1:
            m.atomic(heartbeat, {
                "at": m.now(), "phase": phase, "shard": shard, "assigned": len(texts),
                "processed": index + 1, "fresh_parses": completed, "cache_hits": hits,
                "fixed_sample_replays": replay, "freeze_sha256": parser.freeze_sha,
                "peak_reserved_bytes": parser.torch.cuda.max_memory_reserved(0)})
    print(json.dumps({"phase": phase, "shard": shard, "fresh_parses": completed, "cache_hits": hits}))


def gpu_memory():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True)
    used, total = [float(v.strip()) for v in result.stdout.split(",")]
    return used, total


def coordinate(root, out):
    m = load(out)
    p = verify(m, root, out)
    processes, streams = [], []
    started, peak, completed_phases = time.monotonic(), 0.0, []
    try:
        for phase, gate in (("clean", "PRECHECK.json"), ("attacks", "CLEAN_COMPLETE.json")):
            while not (out / gate).exists():
                if any((out / n).exists() for n in ("HALT.json", "STOP.json")):
                    print("Primary stopped; no further prefill.", flush=True)
                    return
                if time.monotonic() - started > 24 * 3600:
                    raise RuntimeError("Prefill stage wait budget reached")
                time.sleep(2)
            if phase == "clean" and not m.read(out / "PRECHECK.json")["passed"]:
                print("Feature precheck stopped; no prefill.", flush=True)
                return
            if phase == "attacks" and not m.read(out / "CLEAN_GATE.json")["passed"]:
                break
            while True:
                used, total = gpu_memory()
                reserved = p["workers"] * p["worker_memory_reservation_mib"] + p["memory_headroom_mib"]
                if (out / "COMPLETE.json").exists():
                    m.atomic(out / "PREFILL_COMPLETE.json", {
                        "at": m.now(), "freeze_sha256": m.sha(out / "FREEZE.json"),
                        "source_sha256": p["source_sha256"], "workers": p["workers"],
                        "completed_phases": completed_phases, "status": "primary_completed_during_capacity_wait",
                        "peak_global_gpu_memory_fraction": peak, "seconds": time.monotonic() - started,
                        "no_score_or_threshold_changes": True})
                    return
                if used + reserved < p["gpu_memory_fraction_max"] * total:
                    break
                if any((out / n).exists() for n in ("HALT.json", "STOP.json")):
                    print("Primary stopped during capacity wait.", flush=True)
                    return
                m.atomic(out / "PREFILL_HEARTBEAT.json", {
                    "at": m.now(), "phase": phase, "status": "waiting_for_gpu_capacity",
                    "used_mib": used, "required_additional_mib": reserved})
                time.sleep(2)
            processes = []
            for shard in range(p["workers"]):
                log = out / "private/prefill" / f"{phase}_{shard}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                stream = log.open("a")
                streams.append(stream)
                processes.append(subprocess.Popen([
                    sys.executable, "-u", str(root / "scripts/prefill_multiscale_transfer.py"), "worker",
                    "--root", str(root), "--out", str(out), "--phase", phase, "--shard", str(shard)],
                    stdout=stream, stderr=subprocess.STDOUT))
            while any(proc.poll() is None for proc in processes):
                used, total = gpu_memory()
                peak = max(peak, used / total)
                if peak >= p["gpu_memory_fraction_max"]:
                    raise RuntimeError("Global GPU memory gate failed")
                if any(proc.poll() not in (None, 0) for proc in processes):
                    raise RuntimeError("Prefill worker failed; inspect private logs")
                if any((out / n).exists() for n in ("HALT.json", "STOP.json")):
                    raise RuntimeError("Primary halt or stop observed")
                time.sleep(2)
            assert all(proc.returncode == 0 for proc in processes)
            completed_phases.append(phase)
        m.atomic(out / "PREFILL_COMPLETE.json", {
            "at": m.now(), "freeze_sha256": m.sha(out / "FREEZE.json"),
            "source_sha256": p["source_sha256"], "workers": p["workers"],
            "completed_phases": completed_phases, "status": "completed",
            "peak_global_gpu_memory_fraction": peak, "seconds": time.monotonic() - started,
            "no_score_or_threshold_changes": True, "all_fixed_worker_replays_passed": True})
    except BaseException as error:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        failure = {
            "at": m.now(), "reason": "Parser prefill failure: " + type(error).__name__,
            "private_traceback": "private/prefill_coordinator.log",
            "freeze_sha256": m.sha(out / "FREEZE.json")}
        m.atomic(out / "PREFILL_FAILURE.json", failure)
        if not (out / "HALT.json").exists():
            m.atomic(out / "HALT.json", failure)
        subprocess.run(["/usr/bin/supervisorctl", "-c", str(out / "private/supervisor.conf"),
                        "stop", out.name], capture_output=True)
        traceback.print_exc()
        raise
    finally:
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("plan", "coordinate", "worker"))
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--phase", choices=("clean", "attacks"))
    ap.add_argument("--shard", type=int)
    args = ap.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    # Load frozen sources before importing project packages in worker().
    load(out)
    if args.mode == "plan":
        assert 1 <= args.workers <= 3
        plan(root, out, args.workers)
    elif args.mode == "worker":
        worker(root, out, args.phase, args.shard)
    else:
        coordinate(root, out)
