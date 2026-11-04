"""Fast, read-only progress inspection; no model loading or network requests."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def process_exists(pid):
    if type(pid) is not int or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def snapshot(folder, now=None):
    now = now or datetime.now(timezone.utc)
    logs = [json.loads(p.read_text()) for p in (folder / "run_logs").glob("*.json")]
    latest = max(logs, key=lambda r: r["updated_at"]) if logs else {}
    path = folder / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    protocol = manifest.get("protocol", {})
    comparison = "sentence_slots_per_arm" in protocol
    grounded = protocol.get("workflow") == "grounded_generation"
    injection = protocol.get("workflow") == "grounded_injection"
    alive = process_exists(latest.get("pid"))
    age = max(0, (now - datetime.fromisoformat(latest["updated_at"])).total_seconds()) if latest else None
    if latest.get("status") == "complete" and summary.get("status") in (
        "complete_development_run", "complete_development_comparison", "complete_grounded_comparison",
        "complete_grounded_injection_pilot"
    ):
        state = "complete"
    elif latest.get("status") == "failed":
        state = "failed"
    elif alive:
        state = "running" if age is not None and age <= 45 else "process_present_heartbeat_stale"
    elif latest:
        state = "stopped_without_completion"
    else:
        state = "not_started"
    calls = [json.loads(p.read_text()) for p in (folder / "calls").glob("*.json")]
    completed_slots = (len(list((folder / "sentences").rglob("*.json"))) if comparison or injection
                       else len(list((folder / "injection").glob("*.json"))))
    planned_slots = (protocol["sentence_slots_per_arm"] * len(protocol["arms"]) if comparison
                     else protocol.get("injection", {}).get("max_sentence_slots"))
    if injection:
        planned_slots = protocol["total_sentence_slots"]
    return {
        "state": state, "last_stage": latest.get("stage"), "last_arm": latest.get("arm"),
        "last_slot": latest.get("slot"), "last_prompt_id": latest.get("prompt_id"),
        "last_error_type": latest.get("error_type"),
        "heartbeat_age_seconds": age, "pid_present": alive,
        "completed_sentence_slots": None if grounded else completed_slots,
        "planned_sentence_slots": None if grounded else planned_slots,
        "completed_generation_calls": len(list((folder / "generated").glob("*.json"))) if grounded else None,
        "planned_generation_calls": protocol.get("generation_call_count") if grounded else None,
        "baseline_prefixes": len(list((folder / ("evaluated" if comparison or grounded or injection else "baselines")).glob("*.json"))),
        "quality_reviews": (sum(c["scope"].startswith(("style-", "facts-")) and c["status"] == "complete"
                                for c in calls) if grounded or injection else len(list((folder / "quality").glob("*.json")))),
        "mechanism_tests": len(list((folder / "mechanism").glob("*.json"))),
        "api_attempted": len(calls), "api_complete": sum(c["status"] == "complete" for c in calls),
        "api_in_flight": sum(c["status"] == "in_flight" for c in calls),
        "api_failed_or_uncertain": sum(c["status"] not in ("complete", "in_flight") for c in calls),
        "summary_status": summary.get("status"),
        "note": "PID presence is only a liveness hint; stale heartbeat does not prove a hung request.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/swan_sentence_growth1")
    parser.add_argument("--watch-seconds", type=float, default=0)
    parser.add_argument("--interval", type=float, default=15)
    args = parser.parse_args()
    if args.watch_seconds < 0 or args.interval <= 0:
        parser.error("watch-seconds must be nonnegative; interval must be positive")
    deadline = time.monotonic() + args.watch_seconds
    while True:
        state = snapshot(args.output)
        print(json.dumps(state, indent=2), flush=True)
        remaining = deadline - time.monotonic()
        if remaining <= 0 or state["state"] in ("complete", "failed", "stopped_without_completion"):
            break
        time.sleep(min(args.interval, remaining))


if __name__ == "__main__":
    main()
