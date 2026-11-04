"""Enforce the immutable final-round deadline for explicitly named jobs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--control", required=True, type=Path)
    cli.add_argument("--supervisorctl", required=True)
    cli.add_argument("--config", required=True)
    cli.add_argument("--jobs", required=True, nargs="+")
    cli.add_argument("--archive-script", type=Path)
    args = cli.parse_args()
    control = json.loads(args.control.read_text())
    deadline = datetime.fromisoformat(control["deadline_at"])
    if deadline.tzinfo is None:
        raise RuntimeError("Deadline requires timezone")
    identity = hashlib.sha256(json.dumps({
        k: control[k] for k in ("started_at", "deadline_at", "no_successor")
    }, sort_keys=True).encode()).hexdigest()
    folder = args.control.parent
    while True:
        current = json.loads(args.control.read_text())
        if hashlib.sha256(json.dumps({
            k: current[k] for k in ("started_at", "deadline_at", "no_successor")
        }, sort_keys=True).encode()).hexdigest() != identity:
            raise RuntimeError("Immutable budget changed")
        utc = datetime.now(timezone.utc)
        save(folder / "GUARD_HEARTBEAT.json", {
            "updated_at": utc.isoformat(), "deadline_at": deadline.isoformat(),
            "guard_identity_sha256": identity, "guarded_jobs": args.jobs,
            "seconds_remaining": max(0, (deadline - utc).total_seconds()),
        })
        if utc >= deadline:
            save(folder / "DEADLINE_STOP.json", {
                "timestamp": utc.isoformat(), "reason": "final_24h_budget_expired",
                "deadline_at": deadline.isoformat(), "permanent": True,
                "preserve_atomic_cache": True, "no_successor": True,
            })
            outcomes = {}
            for job in args.jobs:
                # Only supplied program names; never shut down a server.
                pid_result = subprocess.run([
                    args.supervisorctl, "-c", args.config, "pid", job,
                ], capture_output=True, text=True, timeout=15)
                isolated_group = None
                try:
                    pid = int(pid_result.stdout.strip())
                    if pid > 1 and os.getpgid(pid) == pid:
                        isolated_group = pid
                except (ValueError, ProcessLookupError):
                    pass
                outcome = subprocess.run([
                    args.supervisorctl, "-c", args.config, "stop", job,
                ], capture_output=True, text=True, timeout=55)
                # Stop the supervisor first to suppress automatic restart, then
                # terminate any surviving children in its verified isolated group.
                if isolated_group is not None:
                    try:
                        os.killpg(isolated_group, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                outcomes[job] = {"returncode": outcome.returncode,
                                 "isolated_group_termination": isolated_group is not None,
                                 "status": outcome.stdout.strip()[-500:]}
            save(folder / "GUARD_STOP_OUTCOME.json", outcomes)
            if args.archive_script:
                completed = subprocess.run([
                    sys.executable, str(args.archive_script), "--reason", "deadline",
                ], capture_output=True, text=True, timeout=180)
                save(folder / "GUARD_ARCHIVE_OUTCOME.json", {
                    "returncode": completed.returncode,
                    "archive_attempted": True,
                })
            return
        time.sleep(min(20, max(.1, (deadline - utc).total_seconds())))


if __name__ == "__main__":
    main()
