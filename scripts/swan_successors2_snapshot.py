"""Read-only successor progress export; never export text or provider credentials."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path("/workspace/swan-elastic-reconstruct")
JOBS = {
    "calibration": ("independent_calibration2_runtime", "independent_calibration2_queue", "calibration"),
    "normal": ("normal_detection2_runtime", "normal_detection2_queue", "normal_detection"),
    "endpoint": ("independent_endpoint2_runtime", "independent_endpoint_successors2_queue", "endpoint"),
    "multiparser": ("multiparser2_runtime", "independent_endpoint_successors2_queue", "multiparser"),
}


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for data in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(data)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text())


def verify_receipt(out, queue, job_id):
    """Recheck the queue's frozen completion contract, not merely its exit code."""
    manifest = queue / "MANIFEST.json"
    receipt = read(queue / (job_id + "_TERMINAL.json"))
    if receipt.get("status") != "success" or receipt.get("returncode") != 0:
        return False
    assert receipt["job"] == job_id
    assert receipt["schema"] == "research-completion-queue/v2"
    assert receipt["manifest_sha256"] == sha(manifest)
    job = next(j for j in read(manifest)["jobs"] if j["id"] == job_id)
    observed, values = {}, {}
    for name, spec in job["completion"].items():
        path = Path(spec["path"])
        data = path.read_bytes()
        values[name] = json.loads(data)
        observed[name] = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}
        if spec.get("sha256"):
            assert observed[name]["sha256"] == spec["sha256"]
        for key, expected in spec.get("equals", {}).items():
            value = values[name]
            for part in key.split("."):
                value = value[part]
            assert value == expected
    for name, spec in job["completion"].items():
        for key, target in spec.get("sha256_fields", {}).items():
            value = values[name]
            for part in key.split("."):
                value = value[part]
            assert value == observed[target]["sha256"]
    assert observed == receipt["artifacts"]
    terminal = read(out / "TERMINAL.json")
    assert terminal["result_sha256"] == sha(out / "RESULT.json")
    assert terminal["audit_sha256"] == sha(out / "AUDIT.json")
    return True


def verify_files(out, seal):
    count = 0
    for name, expected in seal.get("files", {}).items():
        assert sha(out / name) == expected
        count += 1
    return count


def verify_metrics(out, label):
    audit = read(out / "AUDIT.json")
    assert audit["freeze_sha256"] == sha(out / "FREEZE.json")
    count = 0
    if label == "calibration":
        assert len(audit["row_files"]) == 4000
        for name, expected in audit["row_files"].items():
            assert sha(out / name) == expected
            count += 1
        for field, name in (("raw_graph_seal_sha256", "RAW_GRAPH_SEAL.json"),
                            ("thresholds_sha256", "THRESHOLDS.json"),
                            ("bank_binding_sha256", "private/BANK_BINDING.json")):
            assert audit[field] == sha(out / name)
        count += verify_files(out, read(out / "RAW_GRAPH_SEAL.json"))
    elif label == "normal":
        count += verify_files(out, audit)
        for name in ("INPUT_SEAL.json", "RAW_GRAPH_SEAL.json"):
            count += verify_files(out, read(out / name))
        assert len(read(out / "private/SCORES.json")) == 6600
    elif label in ("endpoint", "multiparser"):
        for name, expected in audit["seals"].items():
            assert sha(out / name) == expected
            count += verify_files(out, read(out / name))
    return count


def snapshot(export=False):
    payload = {"at_unix": time.time(), "free_bytes": shutil.disk_usage(ROOT).free,
               "jobs": {}, "exports": {}}
    for label, (directory, queue_name, job_id) in JOBS.items():
        out, queue = ROOT / directory, ROOT / queue_name
        state = {"runtime_exists": out.exists(), "machine_closure": False,
                 "scientific_closure": False}
        filenames = ("FREEZE.json", "HEARTBEAT.json", "GPU_PRECHECK.json", "HALT.json",
                     "PAUSED.json", "RESULT.json", "AUDIT.json", "TERMINAL.json",
                     "THRESHOLDS.json", "QUEUE_VALIDATION.json", "CARRY_FORWARD_COST.json",
                     "readiness/RESULT.json", "readiness/AUDIT.json", "readiness/TERMINAL.json")
        for name in filenames:
            path = out / name
            if not path.is_file():
                continue
            value = read(path)
            state[name] = {"sha256": sha(path), **{k: value[k] for k in (
                "status", "phase", "completed", "complete", "total", "counts", "version",
                "reason_type", "dtype", "normalizer_double_replay", "S2Match_self_replays",
                "elapsed_seconds", "rows") if k in value}}
            if export:
                payload["exports"][label + "/" + name.replace("/", "_")] = {
                    "sha256": sha(path), "base64": base64.b64encode(path.read_bytes()).decode()}
        for name in ("PREPARATION.json", "LAUNCH.json", "QUEUE_STATUS.json", job_id + "_TERMINAL.json"):
            path = queue / name
            if not path.is_file():
                continue
            value = read(path)
            state["queue/" + name] = {"sha256": sha(path), **{k: value[k] for k in (
                "status", "job", "jobs", "reason", "returncode") if k in value}}
            if export:
                payload["exports"][label + "/queue_" + name] = {
                    "sha256": sha(path), "base64": base64.b64encode(path.read_bytes()).decode()}
        for kind, relative in (("raw_graph_records", "private/raw_graphs"), ("scored_rows", "private/rows"),
                               ("normal_scored_rows", "private/scores"), ("writer_slots", "private/slots"),
                               ("completed_documents", "private/documents")):
            state[kind] = len(list((out / relative).glob("*.json")))
        logs = sorted((out / "private").glob("static_*.log")) + sorted(
            (out / "private").glob("regression_*.log"))
        state["regression_tests_passed"] = sum(
            int(match.group(1)) for p in logs
            for match in re.finditer(r"Ran (\d+) tests? in", p.read_text())
            if p.read_text().rstrip().endswith("OK"))
        launch = queue / "LAUNCH.json"
        if launch.exists():
            try:
                os.kill(read(launch)["pid"], 0)
                state["scheduler_alive"] = True
            except ProcessLookupError:
                state["scheduler_alive"] = False
        terminal = out / "TERMINAL.json"
        receipt = queue / (job_id + "_TERMINAL.json")
        if terminal.exists() and receipt.exists():
            try:
                state["machine_closure"] = verify_receipt(out, queue, job_id)
                if state["machine_closure"] and label in (
                        "calibration", "normal", "endpoint", "multiparser"):
                    state["final_payload_files_verified"] = verify_metrics(out, label)
                    state["aggregate_metrics"] = read(out / "RESULT.json")
                state["completion_contract_verified"] = state["machine_closure"]
            except (AssertionError, KeyError, ValueError, OSError, StopIteration) as exc:
                state["machine_closure"] = False
                state["verification_error"] = type(exc).__name__
        payload["jobs"][label] = state
    payload["gpu"] = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader"], text=True).strip()
    if not export:
        payload.pop("exports")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    print(json.dumps(snapshot(args.export), sort_keys=True))
