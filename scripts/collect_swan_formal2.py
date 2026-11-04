"""Collect audited aggregates and follow all phases until machine results exist."""
import argparse
import base64
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/swan_research_completion_20261102/formal_continuation2"
COMMAND = ("PYTHONPATH=/tmp/swan_formal2_stage "
           "/workspace/swan-elastic-reconstruct/.venv/bin/python "
           "/tmp/swan_formal2_stage/swan_formal2_snapshot.py --export")
METRIC_JOBS = ("calibration", "normal", "formal_alignment", "confirmation",
               "endpoint", "multiparser")


def atomic(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def collect(payload):
    stamp = datetime.fromtimestamp(payload["at_unix"], timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = OUT / ("snapshot_" + stamp)
    target.mkdir(mode=0o700, exist_ok=False)
    hashes = {}
    for name, record in payload.pop("exports").items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".json":
            raise ValueError("Unexpected artifact path")
        data = base64.b64decode(record["base64"], validate=True)
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("Artifact hash mismatch")
        destination = target / name
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        destination.write_bytes(data)
        hashes[str(destination.relative_to(OUT))] = record["sha256"]
    frozen = json.loads((target / "formal_null/FREEZE.json").read_text())
    for name, key in (("deploy_swan_formal_successor2.py", "DEPLOYMENT_SOURCE.py"),
                      ("swan_null_successor2.py", "private/code/scripts/swan_null_successor2.py"),
                      ("test_swan_null_successor2.py", "private/code/scripts/test_swan_null_successor2.py")):
        assert hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest() == frozen["files"][key]
    confirmation = target / "confirmation/FREEZE.json"
    if confirmation.exists():
        dependencies = json.loads(confirmation.read_text())["dependencies"]
        for name in ("swan_confirmation_successor2.py", "test_swan_confirmation_successor2.py",
                     "deploy_swan_confirmation_successor2.py"):
            key = ("experiments/swan_v4_independent_confirmation2/DEPLOYMENT_SOURCE.py"
                   if name.startswith("deploy_") else "scripts/" + name)
            assert hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest() == dependencies[key]
    atomic(OUT / "AUDIT.json", {
        "status": "PASS", "at_unix": payload["at_unix"], "snapshot": target.name,
        "byte_exact_files_verified": len(hashes), "files": hashes,
        "local_runtime_code_matches": True, "private_text_exported": False,
        "all_experiments_complete": all(j["machine_closure"] for j in payload["jobs"].values())
    })
    # Publish status only after successful byte-exact export and code checks.
    atomic(OUT / "LATEST_STATUS.json", payload)
    return payload


def fetch():
    result = subprocess.run([sys.executable, "/tmp/swan_transfer_remote_20261102.py"],
                            input=COMMAND, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError("Remote aggregate query failed: exit " + str(result.returncode))
    return collect(json.loads(result.stdout))


def states(snapshot):
    return {key: {
        "queue": value.get("queue/QUEUE_STATUS.json", {}).get("status"),
        "phase": value.get("HEARTBEAT.json", {}).get("phase",
                                                   value.get("HEARTBEAT.json", {}).get("stage")),
        "halt": bool(value.get("HALT.json")), "machine_closure": value["machine_closure"],
    } for key, value in snapshot["jobs"].items()}


def blockers(snapshot):
    found = {}
    for key, job in snapshot["jobs"].items():
        queue = job.get("queue/QUEUE_STATUS.json", {})
        reasons = [name for name in ("HALT.json", "PAUSED.json") if job.get(name)]
        if job.get("verification_error"):
            reasons.append("completion_verification_failed")
        for name, value in job.items():
            if name.startswith("queue/") and name.endswith("_TERMINAL.json"):
                if value.get("status") in ("failed", "blocked", "paused"):
                    reasons.append(value["status"])
        if queue.get("status") in ("queue_error", "failed", "blocked", "paused"):
            reasons.append(queue["status"])
        if not job["machine_closure"] and job.get("scheduler_alive") is False:
            reasons.append("scheduler_exited_without_success")
        coordinator = job.get("COORDINATOR_STATUS.json", {}).get("status", "")
        if any(word in coordinator.lower() for word in ("fail", "error", "halt", "blocked")):
            reasons.append("coordinator_" + coordinator)
        if reasons:
            found[key] = sorted(set(reasons))
    return found


def audited_metrics(snapshot):
    return {name: {
        "result_sha256": job["RESULT.json"]["sha256"],
        "audit_sha256": job["AUDIT.json"]["sha256"],
        "metrics": job["aggregate_metrics"],
    } for name, job in snapshot["jobs"].items()
        if name in METRIC_JOBS and job.get("machine_closure")
        and job.get("completion_contract_verified") and "aggregate_metrics" in job}


def emit(kind, **fields):
    event = {"event": kind, "at_unix": time.time(), "workers_interrupted": False, **fields}
    with (OUT / "MONITOR_EVENTS.jsonl").open("a") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    atomic(OUT / "MONITOR_EVENT.json", event)
    print(json.dumps(event, ensure_ascii=False), flush=True)


def watch(interval):
    with (OUT / ".monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        baseline, prior_blockers, known_metrics = None, None, set()
        prior = OUT / "MACHINE_METRICS.json"
        if prior.exists():
            known_metrics = set(json.loads(prior.read_text())["jobs"])
        errors, last_success = 0, None
        emit("monitor_started", pid=os.getpid(), interval_seconds=interval,
             stop_condition="all_six_metric_jobs_audited", expiry=None)
        while True:
            atomic(OUT / "MONITOR_STATUS.json", {
                "status": "querying", "pid": os.getpid(), "at_unix": time.time(),
                "last_success_unix": last_success, "consecutive_query_errors": errors})
            try:
                snapshot = fetch()
                metrics = audited_metrics(snapshot)
                current = states(snapshot)
                blocked = blockers(snapshot)
            except Exception as exc:
                errors += 1
                atomic(OUT / "MONITOR_STATUS.json", {
                    "status": "observation_unavailable", "pid": os.getpid(),
                    "at_unix": time.time(), "last_success_unix": last_success,
                    "consecutive_query_errors": errors, "error_type": type(exc).__name__,
                    "experiment_failure_inferred": False})
                if errors == 3:
                    emit("observation_unavailable", consecutive_errors=errors,
                         error_type=type(exc).__name__, experiment_failure_inferred=False)
                time.sleep(min(interval, 60))
                continue
            if errors:
                emit("observation_restored", preceding_errors=errors)
            errors, last_success = 0, snapshot["at_unix"]
            if baseline is not None and current != baseline:
                emit("phase_transition", before=baseline, after=current)
            if blocked != prior_blockers and blocked:
                emit("experiment_attention_required", jobs=blocked)
            if prior_blockers and not blocked:
                emit("experiment_attention_cleared")
            baseline, prior_blockers = current, blocked
            newly_ready = sorted(set(metrics) - known_metrics)
            complete = all(name in metrics for name in METRIC_JOBS)
            atomic(OUT / "MACHINE_METRICS.json", {
                "at_unix": last_success, "jobs": metrics,
                "all_machine_metric_jobs_audited": complete,
                "scientific_closure": False,
                "scope": "Machine endpoints only; independent A2 and human quality remain pending.",
                "missing_metric_jobs": sorted(set(METRIC_JOBS) - set(metrics))})
            if newly_ready:
                emit("audited_metrics_available", jobs=newly_ready,
                     metrics_path=str(OUT / "MACHINE_METRICS.json"),
                     scientific_closure=False)
            known_metrics = set(metrics)
            atomic(OUT / "MONITOR_STATUS.json", {
                "status": "machine_metrics_complete" if complete else "watching",
                "pid": os.getpid(), "at_unix": time.time(),
                "last_success_unix": last_success, "consecutive_query_errors": 0,
                "metric_jobs_audited": sorted(metrics), "blockers": blocked})
            if complete:
                emit("all_machine_metrics_audited", scientific_closure=False,
                     metrics_path=str(OUT / "MACHINE_METRICS.json"))
                return
            time.sleep(interval)


def notify(after):
    """A separate background command returns an event without stopping the watcher."""
    relevant = {"audited_metrics_available", "all_machine_metrics_audited",
                "experiment_attention_required", "observation_unavailable"}
    while True:
        path = OUT / "MONITOR_EVENTS.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event["at_unix"] > after and event["event"] in relevant:
                    print(json.dumps(event, ensure_ascii=False, indent=2), flush=True)
                    return
        status_path = OUT / "MONITOR_STATUS.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            stale = time.time() - status["at_unix"] > 300
            try:
                os.kill(status["pid"], 0)
                alive = True
            except ProcessLookupError:
                alive = False
            if stale or not alive:
                print(json.dumps({"event": "monitor_unresponsive", "at_unix": time.time(),
                                  "pid_alive": alive, "status_stale": stale}))
                return
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--notify-after", type=float)
    parser.add_argument("--interval", type=int, default=45)
    args = parser.parse_args()
    if not 1 <= args.interval <= 60:
        parser.error("interval must be between 1 and 60 seconds")
    if args.notify_after is not None:
        notify(args.notify_after)
        return
    if args.source:
        snapshot = collect(json.loads(args.source.read_text()))
    elif not args.watch:
        snapshot = fetch()
    else:
        watch(args.interval)
        return
    print(json.dumps({"status": "PASS", "at_unix": snapshot["at_unix"],
                      "states": states(snapshot), "gpu": snapshot["gpu"]}, indent=2))


if __name__ == "__main__":
    main()
