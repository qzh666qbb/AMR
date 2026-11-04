"""Run the frozen confirmation protocol with idle-sleep protection and status."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_segmentation_intervention import ROOT, write_json


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def snapshot(folder, prior_calls, now=None):
    now = time.time() if now is None else now
    counts = {name: len(list((folder / name).glob("*.json")))
              for name in ("baselines", "injection", "attacks", "quality", "evaluated")}
    active = []
    for path in (folder / "calls").glob("*.json"):
        if path.name in prior_calls:
            continue
        row = json.loads(path.read_text())
        if row["status"] == "in_flight":
            started = datetime.fromisoformat(row["started_at"]).timestamp()
            active.append({"file": path.name, "age_seconds": max(0, now - started)})
    return {"counts": counts, "active_calls": active}


def read_credential(path):
    keys = [line.strip() for line in path.read_text().splitlines()
            if line.strip().startswith("sk-")]
    if len(keys) != 1:
        raise ValueError("Expected exactly one credential in the supplied file")
    return keys[0]


def stop_process(process, group=False):
    if process is None or process.poll() is not None:
        return
    try:
        if group:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if group:
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()
    except ProcessLookupError:
        process.wait()


def start_sleep_guard():
    guard = subprocess.Popen(
        ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(20):
            if guard.poll() is not None:
                raise RuntimeError("Idle-sleep guard exited unexpectedly")
            assertions = subprocess.check_output(
                ["/usr/bin/pmset", "-g", "assertions"], text=True, timeout=5)
            if any(f"pid {guard.pid}(caffeinate)" in line
                   and "PreventUserIdleSystemSleep" in line
                   for line in assertions.splitlines()):
                return guard
            time.sleep(.1)
        raise RuntimeError("Could not verify the idle-sleep assertion")
    except BaseException:
        stop_process(guard)
        raise


def runner_command(folder, limit, max_calls):
    return [sys.executable, "-u", str(ROOT / "scripts/run_confirmation_experiment.py"),
            "run", "--output", str(folder), "--limit", str(limit),
            "--max-calls", str(max_calls), "--recover-incomplete-injection"]


def supervise(folder, credential_path, limit, max_calls, wall_limit):
    logs = folder / "run_logs"
    logs.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    status_path = logs / "status.json"
    prior_calls = {p.name for p in (folder / "calls").glob("*.json")}
    guard = child = None
    status = "starting"
    error_type = None
    command = runner_command(folder, limit, max_calls)
    log_path = logs / f"{run_id}.log"
    heartbeat_path = logs / f"{run_id}.jsonl"

    def publish():
        payload = {
            "run_id": run_id, "updated_at": utc_now(), "status": status,
            "supervisor_pid": os.getpid(),
            "runner_pid": child.pid if child else None,
            "runner_returncode": child.poll() if child else None,
            "idle_sleep_guard_pid": guard.pid if guard else None,
            "error_type": error_type, "request_wall_limit_seconds": wall_limit,
            "log": str(log_path), "target_paragraphs": limit,
            **snapshot(folder, prior_calls),
        }
        write_json(status_path, payload)
        with heartbeat_path.open("a") as output:
            output.write(json.dumps(payload) + "\n")
        print(json.dumps(payload), flush=True)
        return payload

    with (logs / "supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another supervised run holds the experiment lock") from None
        try:
            publish()
            env = os.environ.copy()
            env["DEEPSEEK_API_KEY"] = read_credential(credential_path)
            guard = start_sleep_guard()
            with log_path.open("x") as output:
                child = subprocess.Popen(command, cwd=ROOT, env=env,
                                         stdout=output, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                status = "running"
                while child.poll() is None:
                    state = publish()
                    if guard.poll() is not None:
                        status = "sleep_guard_failed"
                        break
                    if any(r["age_seconds"] > wall_limit for r in state["active_calls"]):
                        status = "request_wall_timeout"
                        break
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
                if status == "running":
                    status = "runner_succeeded" if child.returncode == 0 else "runner_failed"
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "supervisor_failed"
            error_type = type(exc).__name__
            raise
        finally:
            stop_process(child, group=True)
            stop_process(guard)
            publish()
    return 0 if status == "runner_succeeded" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/confirmation_three_arm30")
    parser.add_argument("--credential-file", type=Path, default=ROOT / "api.txt")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--max-calls", type=int, default=7000)
    parser.add_argument("--request-wall-limit", type=float, default=300)
    parser.add_argument("--check-sleep-guard", action="store_true",
                        help="Verify a real assertion, then release it; no API calls")
    args = parser.parse_args()
    if not 1 <= args.limit <= 30 or args.max_calls < 1 or args.request_wall_limit <= 0:
        parser.error("Invalid paragraph, call or timeout limit")

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    if args.check_sleep_guard:
        guard = None
        try:
            guard = start_sleep_guard()
            print(json.dumps({"idle_sleep_assertion_verified": True, "pid": guard.pid}))
        finally:
            stop_process(guard)
        print(json.dumps({"idle_sleep_assertion_released": True}))
        return
    raise SystemExit(supervise(args.output.resolve(), args.credential_file,
                              args.limit, args.max_calls, args.request_wall_limit))


if __name__ == "__main__":
    main()
