"""Prepare and queue endpoint/multiparser successors behind verified calibration."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path("/workspace/swan-elastic-reconstruct")
BANK = ROOT / "independent_banks2_runtime"
CAL = ROOT / "independent_calibration2_runtime"
EP = ROOT / "independent_endpoint2_runtime"
MULTI = ROOT / "multiparser2_runtime"
GROUP = ROOT / "independent_endpoint_successors2_queue"
CAL_QUEUE = ROOT / "independent_calibration2_queue/MANIFEST.json"
PYTHON = ROOT / ".venv/bin/python"
SPECS = {
    "endpoint": {
        "out": EP, "parent": ROOT / "independent_endpoint1_runtime",
        "parent_sha": "911598f6792d4438f4b2e9ef4be0a590cf69e130bb2773778e552e0044da59ce",
        "runner": "scripts/run_swan_independent_endpoint1.py",
        "tests": ("test_swan_independent_endpoint1.py",),
        "status": "MACHINE_ENDPOINT_COMPLETE_HUMAN_QUALITY_PENDING",
    },
    "multiparser": {
        "out": MULTI, "parent": ROOT / "multiparser1_runtime",
        "parent_sha": "47217e6ae90ca9622291a305481182089a5d9f4f6afc64261ff448fdb6df1373",
        "runner": "scripts/run_swan_multiparser1.py",
        "tests": ("test_swan_amrbart_audited1.py", "test_swan_multiparser1.py"),
        "status": "MULTIPARSER_MACHINE_COMPARISON_COMPLETE_HUMAN_QUALITY_PENDING",
    },
}
PATCHES = ("scripts/run_multiscale_transfer.py", "src/core/parser_recovery_audit.py",
           "scripts/run_research_completion_queue.py")


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text())


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(".tmp")
    with temp.open("w") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(p)


def copy_checked(source, target, expected):
    assert sha(source) == expected, "Source changed: " + source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert sha(target) == expected


def environment(out):
    code = out / "private/code"
    return dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, [
        BANK / "private/python_env", code, code / "scripts"])),
        NLTK_DATA=str(BANK / "private/code/.nltk_data"), HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
        AMRBART_ASSETS=str(ROOT), CUBLAS_WORKSPACE_CONFIG=":4096:8")


def arguments(label, action):
    spec = SPECS[label]
    args = [str(spec["out"] / "private/code" / spec["runner"]), action,
            "--out", str(spec["out"]), "--bank", str(BANK), "--calibration", str(CAL)]
    if label == "multiparser":
        args += ["--endpoint", str(EP)]
    return args


def configured_key():
    old_manifest = read(SPECS["endpoint"]["parent"] / "private/queue/MANIFEST.json")
    keys = [job["argv"][job["argv"].index("--key-file") + 1] for job in old_manifest["jobs"]
            if "--key-file" in job["argv"]]
    assert len(keys) == 1 and Path(keys[0]).is_file()
    return keys[0]


def execute(label):
    spec = SPECS[label]
    out = spec["out"]
    args = arguments(label, "run") + ["--assets", str(ROOT)]
    if label == "endpoint":
        args += ["--key-file", configured_key()]
    result = subprocess.run([str(PYTHON), *args], cwd=out / "private/code", env=environment(out))
    if result.returncode:
        raise SystemExit(result.returncode)
    terminal, audit = read(out / "TERMINAL.json"), read(out / "AUDIT.json")
    assert terminal["status"] == spec["status"]
    assert terminal["result_sha256"] == sha(out / "RESULT.json")
    assert terminal["audit_sha256"] == sha(out / "AUDIT.json")
    assert audit["freeze_sha256"] == sha(out / "FREEZE.json")
    count = 0
    for name, expected in audit["seals"].items():
        assert sha(out / name) == expected
        seal = read(out / name)
        if "files" in seal:
            assert seal["freeze_sha256"] == sha(out / "FREEZE.json")
            for relative, digest in seal["files"].items():
                assert sha(out / relative) == digest
                count += 1
    write(out / "QUEUE_VALIDATION.json", {"status": "PASS", "seal_links_verified": True,
        "sealed_payload_files_verified": count, "freeze_sha256": sha(out / "FREEZE.json"),
        "terminal_sha256": sha(out / "TERMINAL.json"), "audit_sha256": sha(out / "AUDIT.json")})


def prepare():
    assert not GROUP.exists() and all(not s["out"].exists() for s in SPECS.values())
    assert shutil.disk_usage(ROOT).free > 2 * 1024**3
    GROUP.mkdir(mode=0o700)
    cal_f = read(CAL / "FREEZE.json")
    for directory in (BANK, CAL):
        for name, expected in read(directory / "FREEZE.json")["files"].items():
            assert sha(directory / name) == expected
    summary = {}
    for label, spec in SPECS.items():
        parent, out = spec["parent"], spec["out"]
        assert sha(parent / "FREEZE.json") == spec["parent_sha"]
        old = read(parent / "FREEZE.json")
        # Both original queues stopped at prerequisite intake, before any provider dispatch.
        provider_files = [p for p in (parent / "private/provider").rglob("*") if p.is_file()]
        assert not provider_files, "Existing provider evidence needs explicit cumulative journal binding"
        assert not list((parent / "private/slots").glob("*.json"))
        out.mkdir(mode=0o700)
        for name, expected in old["files"].items():
            assert not any(s in Path(name).parts for s in ("raw_graphs", "rows", "sessions", "queue", "provider"))
            copy_checked(parent / name, out / name, expected)
        additions = {}
        for name, expected in cal_f["files"].items():
            if name.startswith("private/code/") and (
                    not (out / name).exists() or name.removeprefix("private/code/") in PATCHES):
                copy_checked(CAL / name, out / name, expected)
                additions[name] = expected
        copy_checked(Path(__file__), out / "DEPLOYMENT_SOURCE.py", sha(Path(__file__)))
        sessions = [{"name": p.name, "sha256": sha(p), **read(p)}
                    for p in sorted((parent / "private/sessions").glob("*.json"))]
        write(out / "CARRY_FORWARD_COST.json", {
            "prior_worker_sessions": sessions,
            "prior_worker_elapsed_seconds": sum(s.get("seconds", 0) for s in sessions),
            "prior_provider_files": 0, "prior_provider_calls": 0,
            "historical_cost_reset": False,
            "note": "Bank/calibration costs remain separately carried; worker seconds are not GPU kernel time."})
        write(out / "PARENT_PROVENANCE.json", {
            "parent_freeze_sha256": spec["parent_sha"],
            "parent_halt_sha256": sha(parent / "HALT.json"),
            "source_files_sha256": old["files"], "repaired_or_added_files": additions,
            "new_caches": True, "bank_freeze_sha256": sha(BANK / "FREEZE.json"),
            "calibration_freeze_sha256": sha(CAL / "FREEZE.json")})
        (out / "AMENDMENT.md").write_text(
            "# Verified dependency successor v2\n\n"
            "User authorized completing remaining experiments sequentially. Preserve old HALTs.\n"
            "Keep all original source allocations, models, FP32, native contexts, scoring,\n"
            "writing/attack parameters, denominators, source splits and budget limits.\n"
            "Only verified parser metadata handling and successful-parent queue bindings change.\n"
            "Use fresh caches. Endpoint requires completed calibration and matching private Banks.\n"
            "Endpoint remains 72 paired sources, four arms, 16 slots and 16 candidates.\n"
            "Writer cap stays 230400 requests/$40, attack cap 4608/$10, no unknown-request retry.\n"
            "Multiparser uses the same sealed A-arm texts and has zero provider-call budget.\n"
            "Native CUDA replay remains mandatory when its GPU turn arrives.\n"
            "All actual human quality remains pending; machine completion is not scientific closure.\n")
        frozen = copy.deepcopy(old)
        frozen.update(version=old["version"].replace("/v1", "/v2-parser-metadata-repair"),
                      previous_freeze_sha256=spec["parent_sha"],
                      carry_forward_cost_sha256=sha(out / "CARRY_FORWARD_COST.json"))
        if label == "endpoint":
            frozen.update(bank_freeze_sha256=sha(BANK / "FREEZE.json"),
                          calibration_freeze_sha256=sha(CAL / "FREEZE.json"))
        else:
            frozen["parents"] = {"bank": sha(BANK / "FREEZE.json"),
                                "calibration": sha(CAL / "FREEZE.json"),
                                "endpoint": sha(EP / "FREEZE.json")}
        frozen["files"] = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
        write(out / "FREEZE.json", frozen)
        for asset in frozen["parser"]["assets"]:
            assert sha(ROOT / asset["path"]) == asset["sha256"]
        for name, expected in frozen.get("amrbart_assets", {}).items():
            assert sha(ROOT / name) == expected
        assert sha(ROOT / "vectors/glove.6B.100d.txt") == frozen["glove_sha256"]
        commands = [arguments(label, "verify")] + [
            ["-m", "unittest", "discover", "-s", str(out / "private/code/scripts"), "-p", pattern, "-v"]
            for pattern in spec["tests"]]
        records = []
        for index, args in enumerate(commands):
            logpath = out / "private" / ("regression_%d.log" % index)
            with logpath.open("w") as stream:
                result = subprocess.run([str(PYTHON), *args], cwd=out / "private/code",
                    env=environment(out), stdout=stream, stderr=subprocess.STDOUT)
            records.append({"index": index, "returncode": result.returncode, "log_sha256": sha(logpath)})
            if result.returncode:
                print(logpath.read_text()[-5000:])
                raise RuntimeError("Successor static validation failed: " + label)
        summary[label] = {"status": "PREPARED", "freeze_sha256": sha(out / "FREEZE.json"),
            "frozen_files": len(frozen["files"]), "checks": records, "prior_provider_calls": 0,
            "real_cuda_precheck": "pending_serial_gpu_turn"}
    write(GROUP / "PREPARATION.json", {"status": "PREPARED", "jobs": summary,
                                      "free_bytes": shutil.disk_usage(ROOT).free})
    print(json.dumps(read(GROUP / "PREPARATION.json"), indent=2))


def launch():
    assert read(GROUP / "PREPARATION.json")["status"] == "PREPARED"
    assert not (GROUP / "MANIFEST.json").exists()
    # Resolve the already authorized credential reference without exporting it.
    configured_key()
    jobs = []
    for label, spec in SPECS.items():
        out, status = spec["out"], spec["status"]
        f = read(out / "FREEZE.json")
        args = [str(out / "DEPLOYMENT_SOURCE.py"), label]
        env = environment(out)
        argv = ["/usr/bin/env"] + [k + "=" + env[k] for k in (
            "PYTHONPATH", "NLTK_DATA", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
            "TOKENIZERS_PARALLELISM", "PYTHONDONTWRITEBYTECODE", "AMRBART_ASSETS",
            "CUBLAS_WORKSPACE_CONFIG")] + [str(PYTHON), *args]
        completion = {
            "terminal": {"path": str(out / "TERMINAL.json"), "equals": {"status": status},
                         "sha256_fields": {"result_sha256": "result", "audit_sha256": "seal"}},
            "result": {"path": str(out / "RESULT.json"), "equals": {"status": status}},
            "seal": {"path": str(out / "AUDIT.json"), "equals": {"freeze_sha256": sha(out / "FREEZE.json")}},
            "validation": {"path": str(out / "QUEUE_VALIDATION.json"),
                           "equals": {"status": "PASS", "seal_links_verified": True},
                           "sha256_fields": {"terminal_sha256": "terminal", "audit_sha256": "seal"}},
        }
        seal_names = (("WRITING_COMPLETE.json", "ATTACK_GENERATION_COMPLETE.json", "RAW_GRAPH_SEAL.json",
                       "SCORING_COMPLETE.json", "SELECTIONS_COMPLETE.json") if label == "endpoint" else
                      ("INPUT_BINDING.json", "RAW_AMRLIB_SEAL.json", "RAW_AMRBART_SEAL.json",
                       "SCORING_COMPLETE.json", "THRESHOLDS.json"))
        for index, name in enumerate(seal_names):
            key = "artifact_" + str(index)
            completion[key] = {"path": str(out / name)}
        checks = {str(out / n): h for n, h in f["files"].items()}
        checks[str(out / "FREEZE.json")] = sha(out / "FREEZE.json")
        jobs.append({"id": label, "cwd": str(out / "private/code"), "argv": argv,
            "requires": ([{"manifest": str(CAL_QUEUE), "manifest_sha256": sha(CAL_QUEUE),
                           "job_id": "calibration"}] if label == "endpoint" else [{"job_id": "endpoint"}]),
            "code_sha256": checks, "completion": completion})
    manifest = GROUP / "MANIFEST.json"
    write(manifest, {"schema": "research-completion-queue/v2", "jobs": jobs,
        "resource_lock": str(ROOT / "parser_repair_successors1_20261103/gpu.lock"),
        "resource_retry": {"max_retries": 120, "delay_seconds": 30}})
    manifest.chmod(0o600)
    queuepath = EP / "private/code/scripts/run_research_completion_queue.py"
    spec = importlib.util.spec_from_file_location("queue", queuepath)
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    queue.load_plan(manifest)
    with (GROUP / "scheduler.log").open("a") as log:
        process = subprocess.Popen([str(PYTHON), str(queuepath), str(manifest)],
            cwd=GROUP, env=environment(EP), stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    write(GROUP / "LAUNCH.json", {"status": "QUEUE_LAUNCHED", "pid": process.pid,
        "manifest_sha256": sha(manifest),
        "jobs": {k: {"freeze_sha256": sha(s["out"] / "FREEZE.json")} for k, s in SPECS.items()},
        "writer_cap_calls": 230400, "writer_cap_usd": 40,
        "attack_cap_calls": 4608, "attack_cap_usd": 10, "at_unix": time.time()})
    print(json.dumps(read(GROUP / "LAUNCH.json"), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "launch", "endpoint", "multiparser"))
    action = parser.parse_args().action
    execute(action) if action in SPECS else globals()[action]()
