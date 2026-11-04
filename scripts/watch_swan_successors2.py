"""Bounded read-only monitor; worker execution is independent of this process."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/swan_research_completion_20261102/sequential_successors2"
HELPER = Path("/tmp/swan_transfer_remote_20261102.py")
COMMAND = "/workspace/swan-elastic-reconstruct/.venv/bin/python /tmp/swan_successors2_snapshot.py"


def state(snapshot):
    values = {}
    for label, job in snapshot["jobs"].items():
        queue = job.get("queue/QUEUE_STATUS.json", {})
        values[label] = (queue.get("status"), queue.get("job"),
                         job.get("HEARTBEAT.json", {}).get("phase"), bool(job.get("HALT.json")),
                         bool(job.get("PAUSED.json")), job["machine_closure"])
    return values


def main():
    baseline = state(json.loads((OUT / "LATEST_STATUS.json").read_text()))
    deadline = time.monotonic() + 24 * 3600
    while time.monotonic() < deadline:
        result = subprocess.run([sys.executable, str(HELPER)], input=COMMAND, text=True,
                                capture_output=True, timeout=90)
        if result.returncode:
            event = {"event": "monitor_connection_failed", "returncode": result.returncode,
                     "at_unix": time.time(), "workers_interrupted": False}
            break
        snapshot = json.loads(result.stdout)
        path = OUT / "MONITOR_LATEST.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
        temp.replace(path)
        current = state(snapshot)
        if current != baseline:
            event = {"event": "successor_phase_transition", "at_unix": time.time(),
                     "before": baseline, "after": current, "workers_interrupted": False}
            break
        time.sleep(45)
    else:
        event = {"event": "monitor_window_complete", "at_unix": time.time(),
                 "workers_interrupted": False}
    (OUT / "MONITOR_EVENT.json").write_text(json.dumps(event, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(event, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
