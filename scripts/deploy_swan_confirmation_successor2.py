"""Freeze and queue the original W04 cohort with conservative missing-response handling."""
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
PARENT_ROOT = ROOT / "v4_confirmation1_runtime"
PARENT = PARENT_ROOT / "experiments/swan_v4_independent_confirmation1"
RUNTIME = ROOT / "v4_confirmation2_runtime"
OUT = RUNTIME / "experiments/swan_v4_independent_confirmation2"
GROUP = ROOT / "v4_confirmation2_queue"
STAGE = Path("/tmp/swan_confirmation2_stage")
PYTHON = ROOT / ".venv/bin/python"
PARENT_SHA = "bf2107a793efd00092714d2533c0d737f778f7cabad2d5f95ef684d1daf05824"


def sha(p):
    result = hashlib.sha256()
    with Path(p).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, payload):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def copy_checked(source, target, expected):
    assert sha(source) == expected
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert sha(target) == expected
    else:
        shutil.copyfile(source, target)
    assert sha(target) == expected


def environment():
    return dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
                PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(RUNTIME / "scripts"),
                NLTK_DATA=str(RUNTIME / ".nltk_data"))


def prepare():
    assert not RUNTIME.exists() and not GROUP.exists()
    assert sha(PARENT / "FREEZE.json") == PARENT_SHA
    assert shutil.disk_usage(ROOT).free > 3 * 1024**3
    previous = read(PARENT / "FREEZE.json")
    refs = {"FREEZE.json": PARENT_SHA}
    RUNTIME.mkdir(mode=0o700)
    GROUP.mkdir(mode=0o700)
    for name, expected in previous["dependencies"].items():
        source, target = PARENT_ROOT / name, RUNTIME / name
        assert sha(source) == expected
        target.parent.mkdir(parents=True, exist_ok=True)
        if name.startswith(("models/", "vectors/", "third_party/", ".nltk_data/")):
            target.symlink_to(source.resolve())
        else:
            copy_checked(source, target, expected)
        if source.is_relative_to(PARENT):
            relative = str(source.relative_to(PARENT))
            refs[relative] = expected
            copy_checked(source, OUT / relative, expected)
    for pattern in ("HALT.json", "GENERATION_READINESS.json", "private/documents/*.json",
                    "private/slots/*.json", "private/attempts/*.json", "private/pilot/journal.jsonl",
                    "private/pilot/responses/*.json", "private/local_errors/*.json"):
        refs.update({str(p.relative_to(PARENT)): sha(p) for p in PARENT.glob(pattern)})
    inherited = {n: h for n, h in refs.items() if n.startswith("private/documents/")}
    assert len(inherited) == 30
    for name, expected in inherited.items():
        copy_checked(PARENT / name, OUT / name, expected)
    # Original queue is retained in its old runtime. The successor uses verified v2 receipts.
    queue_path = RUNTIME / "scripts/run_research_completion_queue.py"
    shutil.copyfile(STAGE / queue_path.name, queue_path)
    for name in ("swan_confirmation_successor2.py", "test_swan_confirmation_successor2.py"):
        copy_checked(STAGE / name, RUNTIME / "scripts" / name, sha(STAGE / name))
    copy_checked(Path(__file__), OUT / "DEPLOYMENT_SOURCE.py", sha(Path(__file__)))
    copy_checked(STAGE / "API_DISPOSITION.json", OUT / "API_DISPOSITION.json",
                 sha(STAGE / "API_DISPOSITION.json"))
    case = next(c for c in read(OUT / "API_DISPOSITION.json")["cases"]
                if c["experiment"] == "W04_confirmation")
    assert sha(PARENT / "private/pilot/journal.jsonl") == case["journal_sha256"]
    assert case["phase_cap_nanodollars"] == 25_000_000_000
    write(OUT / "SOURCE_REFERENCE.json", {
        "parent": str(PARENT), "files": refs, "inherited_documents": inherited,
        "parent_dependencies": previous["dependencies"], "old_halt_preserved": True,
        "inherited_validation": "read-only original contract and nested original slot/attempt SHA"})
    write(OUT / "MISSING_RESPONSE_POLICY.json", {
        "known_missing_call_sha256": case["call_id_sha256"],
        "known_missing_reserve_nanodollars": case["unknown_request_retained_reserve_nanodollars"],
        "parent_calls_nonreplayable": True,
        "missing_response_policy": "source unavailable; full source denominator and full reserve retained",
        "future_invalid_model_usage_schema_or_journal": "permanent_halt",
        "max_cumulative_calls": 40550, "max_cumulative_nanodollars": 25_000_000_000,
        "source_replacement": False, "detector_after_complete_generation": True,
        "fresh_generation_and_detector_cache": True, "human_review_supplied": False,
        "unchanged": ["250 sources", "1000 calibration", "1000 null", "Bank100", "FP32 parser",
                      "prompts", "assignments", "scoring", "threshold rule", "8500 score rows"]})
    (OUT / "AMENDMENT.md").write_text(
        "# W04 independent confirmation successor\n\n"
        "Continue the original 250 sources without replacement. Preserve the old HALT and all\n"
        "original request/response and document evidence. Inherit 30 documents byte-for-byte;\n"
        "validate them against the original contract, slots and attempts through a read-only adapter.\n"
        "Missing responses stop their source, retain its denominator and full reserved budget,\n"
        "and are never sent again. Apply the same frozen rule to future missing responses.\n"
        "Keep invalid provider identity, usage/schema and journal failures as permanent HALTs.\n"
        "Keep original prompts, assignments, length gates, Bank100, FP32 and score/threshold rules.\n"
        "Keep the cumulative 40550-call/$25 cap, including inherited usage and unknown reserve once.\n"
        "Generation completes before 1000 calibration + 6000 null + 1500 positive scoring rows.\n"
        "Use a fresh detector cache. Real CUDA replay and <80% VRAM admission precede new calls.\n"
        "Machine completion is not independent human quality completion.\n")
    frozen = copy.deepcopy(previous)
    frozen.update(version="v4-confirmation/v2-missing-response-source-failure",
                  experiment=OUT.name, parent_freeze_sha256=PARENT_SHA)
    frozen["dependencies"] = {
        str(p.relative_to(RUNTIME)): sha(p)
        for p in RUNTIME.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    write(OUT / "FREEZE.json", frozen)
    sys.path[:0] = [str(RUNTIME), str(RUNTIME / "scripts")]
    import swan_confirmation_successor2 as runner
    runner.configure()
    runner.verify()
    log = runner.journal()
    missing = [cid for cid, c in log.previous.calls.items()
               if c["terminal"] is None or c["terminal"]["kind"] == "uncertain"]
    assert len(missing) == 1
    assert hashlib.sha256(missing[0].encode()).hexdigest() == case["call_id_sha256"]
    assert log.totals()["exposure_nanodollars"] == case["budget_exposure_nanodollars"]
    assert log.totals()["calls"] == 1624
    assert {runner.source_of(cid) for cid in log.previous.calls} == (
        {Path(p).stem for p in inherited} | runner.unavailable(log.previous))
    for s in runner.run.read_sources("positive_source"):
        p = OUT / "private/documents" / (s["allocation_id"] + ".json")
        if p.exists():
            runner.validate_document(s, p)
    tests = []
    for index, pattern in enumerate(("test_swan_confirmation_successor2.py",
                                     "test_swan_v4_confirmation1.py",
                                     "test_swan_v4_confirmation1_sources.py")):
        path = OUT / "private" / ("regression_%d.log" % index)
        with path.open("w") as stream:
            result = subprocess.run([str(PYTHON), "-m", "unittest", "discover",
                "-s", "scripts", "-p", pattern, "-v"], cwd=RUNTIME, env=environment(),
                stdout=stream, stderr=subprocess.STDOUT)
        tests.append({"pattern": pattern, "returncode": result.returncode, "sha256": sha(path)})
        if result.returncode:
            print(path.read_text()[-8000:])
            raise RuntimeError("Successor regression failed")
    write(GROUP / "PREPARATION.json", {
        "status": "PREPARED", "freeze_sha256": sha(OUT / "FREEZE.json"),
        "original_dependencies": len(previous["dependencies"]), "parent_evidence_files": len(refs),
        "successor_dependencies": len(frozen["dependencies"]),
        "inherited_documents": 30, "inherited_qualified": 1, "budget": log.totals(), "tests": tests,
        "new_generation_calls": 0, "scored_rows": 0,
        "GPU_readiness": "pending_shared_GPU_turn", "free_bytes": shutil.disk_usage(ROOT).free})
    print(json.dumps(read(GROUP / "PREPARATION.json"), indent=2))


def launch():
    prep = read(GROUP / "PREPARATION.json")
    assert prep["status"] == "PREPARED" and prep["freeze_sha256"] == sha(OUT / "FREEZE.json")
    assert not (GROUP / "MANIFEST.json").exists()
    old = read(PARENT / "private/queue/MANIFEST.json")
    keys = {j["argv"][j["argv"].index("--key-file") + 1]
            for j in old["jobs"] if "--key-file" in j["argv"]}
    assert len(keys) == 1 and Path(next(iter(keys))).is_file()
    runner = RUNTIME / "scripts/swan_confirmation_successor2.py"
    checks = {str(RUNTIME / p): h for p, h in read(OUT / "FREEZE.json")["dependencies"].items()
              if p.startswith(("scripts/", "src/"))}
    checks[str(OUT / "FREEZE.json")] = sha(OUT / "FREEZE.json")
    checks[str(OUT / "SOURCE_REFERENCE.json")] = sha(OUT / "SOURCE_REFERENCE.json")
    jobs = []
    for action, folder, status in (
        ("generate", OUT / "generation", "GENERATION_COMPLETE_MISSING_RETAINED"),
        ("score", OUT, "computational_complete_pending_human_quality"),
    ):
        argv = [str(PYTHON), str(runner), action]
        if action == "generate":
            argv += ["--key-file", next(iter(keys))]
        completion = {
            "terminal": {"path": str(folder / "TERMINAL.json"), "equals": {"status": status},
                         "sha256_fields": {"result_sha256": "result", "audit_sha256": "seal"}},
            "result": {"path": str(folder / "RESULT.json"), "equals": {"status": status}},
            "seal": {"path": str(folder / "AUDIT.json")},
        }
        if action == "score":
            completion["verified"] = {"path": str(OUT / "FINAL_VERIFICATION.json"),
                                      "equals": {"status": "PASS", "rows": 8500},
                                      "sha256_fields": {"result_sha256": "result", "audit_sha256": "seal"}}
            completion["calibration"] = {"path": str(OUT / "results/CALIBRATION_control.json")}
        jobs.append({"id": action, "argv": argv, "cwd": str(RUNTIME), "code_sha256": checks,
                     "requires": [] if action == "generate" else [{"job_id": "generate"}],
                     "completion": completion})
    manifest = GROUP / "MANIFEST.json"
    write(manifest, {"schema": "research-completion-queue/v2", "jobs": jobs,
                    "resource_lock": str(ROOT / "parser_repair_successors1_20261103/gpu.lock"),
                    "resource_retry": {"max_retries": 120, "delay_seconds": 30}})
    queue_path = RUNTIME / "scripts/run_research_completion_queue.py"
    spec = importlib.util.spec_from_file_location("queue2", queue_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load_plan(manifest)
    with (GROUP / "scheduler.log").open("a") as stream:
        process = subprocess.Popen([str(PYTHON), str(queue_path), str(manifest)],
            cwd=RUNTIME, env=environment(), stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    write(GROUP / "LAUNCH.json", {"status": "QUEUE_LAUNCHED", "pid": process.pid,
                                "manifest_sha256": sha(manifest), "at_unix": time.time()})
    print(json.dumps(read(GROUP / "LAUNCH.json")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("prepare", "launch"))
    globals()[ap.parse_args().action]()
