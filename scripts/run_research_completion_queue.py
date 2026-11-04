"""Versioned dependency queue. A resource lock is not evidence of success."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


VERSION = "research-completion-queue/v2"


class EvidenceError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    payload = {**payload, "updated_at_unix": time.time()}
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def absolute(value):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise EvidenceError("Manifest paths must be absolute")
    return Path(value)


def load_plan(manifest):
    plan = read(manifest)
    if plan.get("schema") != VERSION or "predecessor_lock" in plan:
        raise EvidenceError("Use a new v2 manifest with explicit requires and completion")
    seen, paths = set(), set()
    for job in plan["jobs"]:
        name = job["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in seen:
            raise EvidenceError("Invalid or duplicate job id")
        absolute(job["cwd"])
        if not job["argv"] or not all(isinstance(s, str) for s in job["argv"]):
            raise EvidenceError("argv must be a nonempty string list")
        if not job["code_sha256"]:
            raise EvidenceError("Code hashes required")
        for path in job["code_sha256"]:
            absolute(path)
        for dep in job["requires"]:
            if "manifest" in dep:
                if absolute(dep["manifest"]).resolve() == manifest.resolve():
                    raise EvidenceError("Local dependencies must use job_id only")
                if not re.fullmatch(r"[0-9a-f]{64}", dep["manifest_sha256"]):
                    raise EvidenceError("External dependency needs frozen manifest SHA")
            elif dep["job_id"] not in seen:
                raise EvidenceError("Local dependencies must precede consumers")
        completion = job["completion"]
        if not {"terminal", "result", "seal"} <= completion.keys():
            raise EvidenceError("Completion needs terminal, result and seal")
        if not completion["terminal"].get("equals"):
            raise EvidenceError("Terminal success condition must be explicit")
        for spec in completion.values():
            path = absolute(spec["path"]).resolve()
            if path in paths:
                raise EvidenceError("Jobs must not share completion paths")
            paths.add(path)
            for target in spec.get("sha256_fields", {}).values():
                if target not in completion:
                    raise EvidenceError("Artifact hash link has no target")
        seen.add(name)
    retry = plan.get("resource_retry", {})
    if not isinstance(retry.get("max_retries", 3), int) or retry.get("max_retries", 3) < 0:
        raise EvidenceError("Invalid resource retry count")
    if not 0 <= retry.get("delay_seconds", 30) <= 60:
        raise EvidenceError("Invalid resource retry delay")
    if plan.get("resource_lock"):
        absolute(plan["resource_lock"])
    return plan


def field(value, dotted):
    for part in dotted.split("."):
        value = value[part]
    return value


def artifacts(job):
    """Check semantic completion conditions, hash links and exact current bytes."""
    snapshots, values = {}, {}
    for name, spec in job["completion"].items():
        path = Path(spec["path"])
        if not path.is_file():
            raise EvidenceError("missing_completion:" + name)
        payload = path.read_bytes()
        values[name] = json.loads(payload)
        snapshots[name] = {"path": str(path),
                           "sha256": hashlib.sha256(payload).hexdigest()}
        if spec.get("sha256") and spec["sha256"] != snapshots[name]["sha256"]:
            raise EvidenceError("fixed_artifact_sha:" + name)
        for key, expected in spec.get("equals", {}).items():
            if field(values[name], key) != expected:
                raise EvidenceError("completion_condition:" + name + ":" + key)
    for name, spec in job["completion"].items():
        for key, target in spec.get("sha256_fields", {}).items():
            if field(values[name], key) != snapshots[target]["sha256"]:
                raise EvidenceError("artifact_hash_link:" + name + ":" + key)
    return snapshots


def verify_code(job):
    for path, expected in job["code_sha256"].items():
        if sha(path) != expected:
            raise EvidenceError("code_sha_changed")


def receipt_path(manifest, job_id):
    return manifest.parent / (job_id + "_TERMINAL.json")


def dependencies(manifest, plan, job, visiting):
    receipts = []
    for dep in job["requires"]:
        parent = Path(dep.get("manifest", str(manifest)))
        parent_identity = sha(parent)
        if "manifest" in dep and parent_identity != dep["manifest_sha256"]:
            raise EvidenceError("parent_manifest_sha_changed")
        parent_plan = plan if parent == manifest else load_plan(parent)
        producer = next((j for j in parent_plan["jobs"] if j["id"] == dep["job_id"]), None)
        if producer is None:
            raise EvidenceError("parent_job_missing")
        path = receipt_path(parent, producer["id"])
        verify_success(parent, parent_plan, producer, visiting)
        receipt_sha = sha(path)
        if dep.get("terminal_sha256") and receipt_sha != dep["terminal_sha256"]:
            raise EvidenceError("parent_terminal_sha_changed")
        receipts.append({"manifest": str(parent), "manifest_sha256": parent_identity,
                         "job_id": producer["id"], "terminal_sha256": receipt_sha})
    return receipts


def verify_success(manifest, plan, job, visiting):
    key = (str(manifest.resolve()), job["id"])
    if key in visiting:
        raise EvidenceError("dependency_cycle")
    record = read(receipt_path(manifest, job["id"]))
    if (record.get("schema") != VERSION or record.get("manifest_sha256") != sha(manifest)
            or record.get("job") != job["id"] or record.get("status") != "success"
            or record.get("returncode") != 0):
        raise EvidenceError("parent_or_resume_not_successful")
    verify_code(job)
    if artifacts(job) != record["artifacts"]:
        raise EvidenceError("artifact_changed_since_success")
    if dependencies(manifest, plan, job, visiting | {key}) != record["dependencies"]:
        raise EvidenceError("dependency_changed_since_success")
    return record


def execute(manifest):
    manifest = manifest.resolve()
    plan = load_plan(manifest)
    out, identity = manifest.parent, sha(manifest)
    base = {"schema": VERSION, "manifest_sha256": identity}
    states = {}
    with ExitStack() as stack:
        owner = stack.enter_context((out / ".queue.lock").open("a"))
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = out / "QUEUE_STATUS.json"
        write(status, {**base, "status": "waiting_for_resource"})
        if plan.get("resource_lock"):
            lock = stack.enter_context(Path(plan["resource_lock"]).open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX)
        for job in plan["jobs"]:
            name = job["id"]
            terminal = receipt_path(manifest, name)
            common = {**base, "job": name}
            if sha(manifest) != identity:
                raise EvidenceError("Queue manifest changed during execution")
            if terminal.exists():
                existing = read(terminal)
                if (existing.get("manifest_sha256") != identity
                        or existing.get("schema") != VERSION or existing.get("job") != name):
                    raise EvidenceError("Queue receipt identity changed")
                if existing["status"] == "success":
                    try:
                        verify_success(manifest, plan, job, set())
                        states[name] = "success"
                    except (EvidenceError, OSError, ValueError, KeyError):
                        # Retain historical receipt; never silently rerun a completed job.
                        states[name] = "invalidated"
                else:
                    states[name] = existing["status"]
                continue
            try:
                parents = dependencies(manifest, plan, job, set())
            except (EvidenceError, OSError, ValueError, KeyError) as exc:
                states[name] = "blocked"
                write(terminal, {**common, "status": "blocked", "returncode": None,
                                 "reason": type(exc).__name__})
                continue
            started = out / (name + "_STARTED.json")
            try:
                verify_code(job)
                if started.exists():
                    raise EvidenceError("interrupted_job_requires_review")
                if any(Path(s["path"]).exists() for s in job["completion"].values()):
                    raise EvidenceError("stale_completion_before_launch")
                write(started, {**common, "dependencies": parents})
            except (EvidenceError, OSError) as exc:
                states[name] = "failed"
                write(terminal, {**common, "status": "failed", "returncode": None,
                                 "reason": str(exc) if isinstance(exc, EvidenceError)
                                 else type(exc).__name__})
                continue
            retry = plan.get("resource_retry", {})
            attempts = 0
            while True:
                attempts += 1
                rc, snapshots, reason = None, {}, None
                try:
                    verify_code(job)
                    if sha(manifest) != identity:
                        raise EvidenceError("Queue manifest changed during execution")
                    if dependencies(manifest, plan, job, set()) != parents:
                        raise EvidenceError("dependency_changed_before_launch")
                    write(status, {**common, "status": "running", "attempt": attempts})
                    with (out / (name + ".log")).open("a") as stream:
                        process = subprocess.run(
                            job["argv"], cwd=job["cwd"], stdout=stream,
                            stderr=subprocess.STDOUT,
                            env=dict(os.environ, PYTHONUNBUFFERED="1"))
                    rc = process.returncode
                    if rc == 75 and attempts <= retry.get("max_retries", 3):
                        write(status, {**common, "status": "resource_pause",
                                       "scientific_result": None})
                        time.sleep(retry.get("delay_seconds", 30))
                        continue
                    state = "paused" if rc in (75, 76) else "failed"
                    if rc == 0:
                        verify_code(job)
                        if sha(manifest) != identity:
                            raise EvidenceError("Queue manifest changed during execution")
                        snapshots = artifacts(job)
                        if dependencies(manifest, plan, job, set()) != parents:
                            raise EvidenceError("dependency_changed_during_job")
                        state = "success"
                except (EvidenceError, OSError, ValueError, KeyError) as exc:
                    state = "failed"
                    reason = str(exc) if isinstance(exc, EvidenceError) else type(exc).__name__
                states[name] = state
                write(terminal, {**common, "status": state, "returncode": rc,
                                 "attempts": attempts, "reason": reason,
                                 "artifacts": snapshots, "dependencies": parents})
                break
        overall = ("failed" if any(s in ("failed", "invalidated") for s in states.values())
                   else "paused" if "paused" in states.values()
                   else "blocked" if "blocked" in states.values() else "complete")
        write(status, {**base, "status": overall, "jobs": states})
    return 0 if overall == "complete" else 76 if overall == "paused" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        raise SystemExit(execute(args.manifest))
    except Exception as exc:
        write(args.manifest.parent / "QUEUE_ERROR.json",
              {"schema": VERSION, "status": "queue_error", "error_type": type(exc).__name__})
        raise
