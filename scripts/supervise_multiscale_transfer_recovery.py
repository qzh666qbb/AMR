"""Serialize a recoverable transfer worker and both frozen final audits."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(path.read_text())


def write(path, payload):
    payload["at_unix"] = time.time()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def main(root, out, tools):
    lock = (out / "private/recovery_supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    freeze = read(out / "FREEZE.json")
    plan = read(out / "RECOVERY_CONTROL.json")
    assert plan["freeze_sha256"] == hashlib.sha256((out / "FREEZE.json").read_bytes()).hexdigest()
    for name, expected in plan["tool_sha256"].items():
        assert hashlib.sha256((tools / name).read_bytes()).hexdigest() == expected
    environment = dict(os.environ, PYTHONPATH=str(out / "private/code"),
                       OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
    start = time.monotonic()
    state = out / "RECOVERY_STATUS.json"
    launches = 0
    oom_pauses = 0

    def capacity():
        result = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"], check=True, capture_output=True,
            text=True, timeout=10)
        used, total = map(float, result.stdout.strip().split(","))
        return used, total

    def admit(phase):
        while True:
            used, total = capacity()
            if used + 3072 < freeze["resource_gate"]["vram_fraction_max"] * total:
                return
            write(state, {"status": "waiting_for_gpu_capacity", "next_phase": phase,
                          "used_mib": used, "total_mib": total, "launches": launches})
            if time.monotonic() - start > 24 * 3600:
                raise RuntimeError("Resource wait exceeded 24 hours; no scientific outcome")
            time.sleep(30)

    while not (out / "COMPLETE.json").exists():
        if any((out / name).exists() for name in ("HALT.json", "STOP.json")):
            write(state, {"status": "stopped", "reason": "Worker gate; inspect original marker"})
            return
        admit("worker")
        launches += 1
        write(state, {"status": "running", "launches": launches, "parser_processes": 1})
        with (out / "private/worker.log").open("a") as stream:
            result = subprocess.run(
                [sys.executable, "-u", str(out / "private/code/scripts/run_multiscale_transfer.py"),
                 "work", "--root", str(root), "--out", str(out)],
                env=environment, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode == 75 and (out / "PAUSED.json").exists():
            pause = read(out / "PAUSED.json")
            oom_pauses += "OutOfMemoryError" in pause["reason"]
            if oom_pauses >= 2:
                write(state, {"status": "resource_blocked",
                              "reason": "Repeated CUDA OOM requires new capacity diagnosis",
                              "scientific_outcome": None})
                return
            continue
        if result.returncode != 0:
            write(state, {"status": "worker_failed", "exit_code": result.returncode})
            return
        if not (out / "COMPLETE.json").exists():
            write(state, {"status": "stopped", "reason": "Precheck or scientific gate"})
            return

    admit("final_audit")
    write(state, {"status": "auditing", "launches": launches})
    with (out / "private/final_audit.log").open("a") as stream:
        commands = [
            [sys.executable, str(tools / "audit_multiscale_transfer.py"), "audit",
             "--root", str(root), "--out", str(out)],
            [sys.executable, str(tools / "verify_multiscale_transfer_metrics.py"),
             "--out", str(out)],
        ]
        for command in commands:
            result = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                write(state, {"status": "audit_failed", "exit_code": result.returncode})
                return
    assert read(out / "FINAL_AUDIT.json")["passed"]
    write(state, {"status": "complete", "final_audit_passed": True, "launches": launches})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tools", type=Path, required=True)
    a = ap.parse_args()
    main(a.root, a.out, a.tools)
