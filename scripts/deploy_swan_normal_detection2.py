"""Repair W02 detection using unchanged sealed sources, without provider calls."""
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
PARENT = ROOT / "normal_completion1_runtime"
BANK = ROOT / "independent_banks2_runtime"
OUT = ROOT / "normal_detection2_runtime"
GROUP = ROOT / "normal_detection2_queue"
CODE = OUT / "private/code"
PYTHON = ROOT / ".venv/bin/python"
CAL = ROOT / "independent_calibration2_runtime"
CAL_QUEUE = ROOT / "independent_calibration2_queue/MANIFEST.json"
PARENT_SHA = "8e8391060be15d67d1bf68c1366d256196c86c35a3c5985d64b4f6fe9a84c22e"
BANK_SHA = "dec02bd4be005497afc854502084d9d2b9e1dae2889a70c63b2fea331f06533c"
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


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(".tmp")
    with temp.open("w") as stream:
        stream.write(json.dumps(obj, sort_keys=True, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(p)


def checked_copy(source, destination, expected):
    assert sha(source) == expected
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    assert sha(destination) == expected


def environment():
    return dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, [
        BANK / "private/python_env", CODE, CODE / "scripts"])),
        NLTK_DATA=str(BANK / "private/code/.nltk_data"), HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")


def check_source():
    ref = read(OUT / "SOURCE_REFERENCE.json")
    assert ref["root"] == str(PARENT)
    for name, expected in ref["files"].items():
        assert sha(PARENT / name) == expected, "Original source changed: " + name
    assert sha(PARENT / "FREEZE.json") == PARENT_SHA
    assert sha(BANK / "FREEZE.json") == BANK_SHA
    for name, expected in read(BANK / "FREEZE.json")["files"].items():
        assert sha(BANK / name) == expected
    assert sha(OUT / "private/INPUTS.json") == ref["files"]["private/INPUTS.json"]
    return ref


def prepare():
    assert not OUT.exists() and not GROUP.exists()
    assert sha(PARENT / "FREEZE.json") == PARENT_SHA
    assert sha(BANK / "FREEZE.json") == BANK_SHA
    assert shutil.disk_usage(ROOT).free > 3 * 1024**3
    previous = read(PARENT / "FREEZE.json")
    refs = {**previous["files"], "FREEZE.json": PARENT_SHA,
            "DETECT_HALT.json": sha(PARENT / "DETECT_HALT.json")}
    for sealname in ("MAIN_GENERATION_SEAL.json", "INPUT_SEAL.json"):
        seal = read(PARENT / sealname)
        assert seal["freeze_sha256"] == PARENT_SHA
        refs[sealname] = sha(PARENT / sealname)
        refs.update(seal["files"])
    for name, expected in refs.items():
        assert sha(PARENT / name) == expected
    docs = read(PARENT / "private/INPUTS.json")
    assert len(docs) == 6600
    identities = {role: {d["source_id"] for d in docs if d["role"] == role}
                  for role in ("calibration", "null_test", "positive")}
    assert {k: len(v) for k, v in identities.items()} == {
        "calibration": 1000, "null_test": 1000, "positive": 200}
    assert sum(map(len, identities.values())) == len(set.union(*identities.values()))
    assert len({(d["source_id"], d["condition"]) for d in docs}) == 6600
    for role, ids in identities.items():
        assert {(d["source_id"], d["condition"]) for d in docs if d["role"] == role} == {
            (sid, c) for sid in ids for c in ("clean", "A", "B")}
    OUT.mkdir(mode=0o700)
    GROUP.mkdir(mode=0o700)
    for name, expected in previous["files"].items():
        assert not any(s in Path(name).parts for s in ("raw_graphs", "scores", "sessions", "queue"))
        checked_copy(PARENT / name, OUT / name, expected)
    for name, expected in read(CAL / "FREEZE.json")["files"].items():
        if not name.startswith("private/code/"):
            continue
        if not (OUT / name).exists() or name.removeprefix("private/code/") in PATCHES:
            checked_copy(CAL / name, OUT / name, expected)
    checked_copy(PARENT / "private/INPUTS.json", OUT / "private/INPUTS.json",
                 refs["private/INPUTS.json"])
    checked_copy(Path(__file__), OUT / "DEPLOYMENT_SOURCE.py", sha(Path(__file__)))
    write(OUT / "SOURCE_REFERENCE.json", {"root": str(PARENT), "files": refs,
        "read_only": True, "old_halt_preserved": True, "source_rows": 6600,
        "original_raw_graphs_reused": False, "LLM_Null_not_used": True})
    sessions = [{"name": p.name, "sha256": sha(p), **read(p)}
                for p in sorted((PARENT / "private/sessions").glob("*.json"))]
    write(OUT / "CARRY_FORWARD_COST.json", {"parent_sessions": sessions,
        "historical_cost_reset": False, "provider_calls_budget": 0,
        "main_generation_seal_sha256": refs["MAIN_GENERATION_SEAL.json"],
        "old_detect_graphs": len(list((PARENT / "private/raw_graphs").glob("*.json"))),
        "LLM_Null_unknown_request_remains_unresolved": True,
        "note": "Reuse sealed source/generation evidence only. Wall time is not exact GPU time."})
    (OUT / "AMENDMENT.md").write_text(
        "# W02 normal detection repair v2\n\n"
        "User authorized sequential completion. Reuse the exact sealed 6600 input rows:\n"
        "1000 calibration, 1000 independent original Null and all 200 generation attempts,\n"
        "with clean/A/B conditions and all missing-positive denominators retained.\n"
        "Keep the official Bank50, FP32 parser, native length, normalization, matcher and\n"
        "threshold rules. Use the verified metadata parser repair and a fresh graph/score cache.\n"
        "Reference original generation and input seals without rewriting historical records.\n"
        "No provider calls or LLM Null intake; W05 A2/formal three-arm work remains separate.\n"
        "Zero eligible positive documents cannot support positive retention or quality success.\n")
    f = copy.deepcopy(previous)
    f.update(version="normal-length-detection/v2-parser-metadata-repair",
             parent_detection_freeze_sha256=PARENT_SHA, bank_environment_freeze_sha256=BANK_SHA,
             source_reference_sha256=sha(OUT / "SOURCE_REFERENCE.json"),
             new_provider_calls_budget=0, original_graph_cache_reused=False)
    f["files"] = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
    write(OUT / "FREEZE.json", f)
    check_source()
    records = []
    for index, pattern in enumerate(("test_swan_normal_completion1.py", "test_swan_independent_calibration1.py")):
        logpath = OUT / "private" / ("regression_%d.log" % index)
        with logpath.open("w") as stream:
            result = subprocess.run([str(PYTHON), "-m", "unittest", "discover", "-s",
                str(CODE / "scripts"), "-p", pattern, "-v"], cwd=CODE, env=environment(),
                stdout=stream, stderr=subprocess.STDOUT)
        records.append({"test": pattern, "returncode": result.returncode, "log_sha256": sha(logpath)})
        if result.returncode:
            print(logpath.read_text()[-5000:])
            raise RuntimeError("Normal detection readiness failed")
    write(GROUP / "PREPARATION.json", {"status": "PREPARED", "freeze_sha256": sha(OUT / "FREEZE.json"),
        "source_files_verified": len(refs), "frozen_files": len(f["files"]), "checks": records,
        "rows": len(docs), "unique_parse_texts": len({s for d in docs for s in d["sentences"]}),
        "positive_sources": 200, "positive_available_rows": sum(
            bool(d["sentences"]) for d in docs if d["role"] == "positive"),
        "provider_calls_budget": 0, "free_bytes": shutil.disk_usage(ROOT).free})
    print(json.dumps(read(GROUP / "PREPARATION.json"), indent=2))


def run():
    sys.path[:0] = [str(BANK / "private/python_env"), str(CODE), str(CODE / "scripts")]
    import run_swan_normal_completion1 as normal
    import run_multiscale_transfer as rt
    from run_swan_independent_banks1 import Normalizer
    from run_swan_independent_calibration1 import Matcher
    from swan_final1_common import freeze, single_writer
    with single_writer(OUT):
        started = time.monotonic()
        try:
            assert not (OUT / "HALT.json").exists(), "New successor required after integrity HALT"
            normal.verify(OUT)
            ref = check_source()
            if (OUT / "TERMINAL.json").exists():
                terminal = read(OUT / "TERMINAL.json")
                assert terminal["result_sha256"] == sha(OUT / "RESULT.json")
                assert terminal["audit_sha256"] == sha(OUT / "AUDIT.json")
                normal.verify_seal(OUT, OUT / "AUDIT.json")
                return
            normalizer, matcher = Normalizer(BANK), Matcher(OUT, ROOT)
            graph = "(w / want-01 :ARG0 (b / boy) :ARG1 (g / go-01 :ARG0 b))"
            pair = [normalizer(graph) for _ in range(2)]
            assert pair[0] == pair[1] and pair[0]["eligible"]
            assert [matcher.fresh(pair[0]["template"], pair[0]["template"]) for _ in range(2)] == [1., 1.]
            matcher.db.close()
            freeze(OUT / "NORMALIZER_MATCHER_PRECHECK.json", {
                "status": "PASS", "normalizer_double_replay": True, "matcher_double_replay": [1., 1.]})
            normal.file_seal(OUT, OUT / "INPUT_SEAL.json", [
                OUT / "private/INPUTS.json", OUT / "SOURCE_REFERENCE.json"])
            freeze(OUT / "MAIN_GENERATION_SEAL.json", {
                "freeze_sha256": sha(OUT / "FREEZE.json"),
                "files": {"SOURCE_REFERENCE.json": sha(OUT / "SOURCE_REFERENCE.json")},
                "source_main_generation_seal_sha256": ref["files"]["MAIN_GENERATION_SEAL.json"],
                "original_records_referenced_read_only": True})
            normal.score_rows(OUT, ROOT, BANK, read(OUT / "private/INPUTS.json"))
            check_source()
        except Exception as exc:
            pause = isinstance(exc, rt.ResourcePause) or type(exc).__name__ == "OutOfMemoryError"
            rt.atomic(OUT / ("PAUSED.json" if pause else "HALT.json"), {
                "reason_type": type(exc).__name__, "resume_allowed": pause, "scientific_result": None})
            if pause:
                raise SystemExit(75)
            raise
        finally:
            rt.atomic(OUT / "private/sessions" / (str(time.time_ns()) + ".json"),
                      {"stage": "detect", "seconds": time.monotonic() - started})


def launch():
    assert read(GROUP / "PREPARATION.json")["status"] == "PREPARED"
    assert not (GROUP / "MANIFEST.json").exists()
    f = read(OUT / "FREEZE.json")
    status = "W02_MACHINE_COMPLETE_QUALITY_PENDING"
    completion = {
        "terminal": {"path": str(OUT / "TERMINAL.json"), "equals": {"status": status},
                     "sha256_fields": {"result_sha256": "result", "audit_sha256": "seal"}},
        "result": {"path": str(OUT / "RESULT.json"), "equals": {"status": status}},
        "seal": {"path": str(OUT / "AUDIT.json"), "equals": {"freeze_sha256": sha(OUT / "FREEZE.json")}},
        "graphs": {"path": str(OUT / "RAW_GRAPH_SEAL.json"), "equals": {"freeze_sha256": sha(OUT / "FREEZE.json")}},
        "scores": {"path": str(OUT / "private/SCORES.json")},
        "thresholds": {"path": str(OUT / "THRESHOLDS.json")},
    }
    env = environment()
    argv = ["/usr/bin/env"] + [k + "=" + env[k] for k in (
        "PYTHONPATH", "NLTK_DATA", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
        "TOKENIZERS_PARALLELISM", "PYTHONDONTWRITEBYTECODE")] + [
            str(PYTHON), str(OUT / "DEPLOYMENT_SOURCE.py"), "run"]
    checks = {str(OUT / name): h for name, h in f["files"].items()}
    checks[str(OUT / "FREEZE.json")] = sha(OUT / "FREEZE.json")
    manifest = GROUP / "MANIFEST.json"
    write(manifest, {"schema": "research-completion-queue/v2",
        "resource_lock": str(ROOT / "parser_repair_successors1_20261103/gpu.lock"),
        "resource_retry": {"max_retries": 120, "delay_seconds": 30}, "jobs": [{
            "id": "normal_detection", "cwd": str(CODE), "argv": argv, "code_sha256": checks,
            "requires": [{"manifest": str(CAL_QUEUE), "manifest_sha256": sha(CAL_QUEUE),
                          "job_id": "readiness"}], "completion": completion}]})
    queuepath = CODE / "scripts/run_research_completion_queue.py"
    spec = importlib.util.spec_from_file_location("q", queuepath)
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    queue.load_plan(manifest)
    with (GROUP / "scheduler.log").open("a") as log:
        process = subprocess.Popen([str(PYTHON), str(queuepath), str(manifest)], cwd=CODE,
            env=environment(), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    write(GROUP / "LAUNCH.json", {"status": "QUEUE_LAUNCHED", "pid": process.pid,
        "manifest_sha256": sha(manifest), "freeze_sha256": sha(OUT / "FREEZE.json"),
        "provider_calls_budget": 0, "at_unix": time.time()})
    print(json.dumps(read(GROUP / "LAUNCH.json"), indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("prepare", "run", "launch"))
    globals()[ap.parse_args().action]()
