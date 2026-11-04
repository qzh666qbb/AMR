"""Deploy W05 missing-response continuation and dependent formal scoring."""
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
OUT = ROOT / "normal_formal2_runtime"
GROUP = ROOT / "normal_formal2_queue"
GPU_GROUP = ROOT / "normal_formal2_scoring_queue"
CODE = OUT / "private/code"
BANK = ROOT / "independent_banks2_runtime"
CAL = ROOT / "independent_calibration2_runtime"
ALIGNMENT = ROOT / "atomic_alignment_completion2_runtime"
SCORING = ROOT / "atomic_alignment_scoring1_runtime"
OVERLAY = ROOT / "atomic_scoring_dependencies1/overlay"
PYTHON = ROOT / ".venv/bin/python"
STAGE = Path("/tmp/swan_formal2_stage")
PARENT_SHA = "8e8391060be15d67d1bf68c1366d256196c86c35a3c5985d64b4f6fe9a84c22e"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(data)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def checked_copy(source, target, expected):
    assert sha(source) == expected, "Source SHA differs: " + str(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert sha(target) == expected


def environment():
    return dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, [
        OVERLAY, BANK / "private/python_env", CODE, CODE / "scripts"])),
        NLTK_DATA=str(BANK / "private/code/.nltk_data"),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")


def check_source():
    ref = read(OUT / "SOURCE_REFERENCE.json")
    assert ref["root"] == str(PARENT)
    assert sha(PARENT / "FREEZE.json") == PARENT_SHA
    for name, expected in ref["files"].items():
        assert sha(PARENT / name) == expected, "Parent drift: " + name
    for path, expected in read(OUT / "ASSET_REFERENCE.json")["files"].items():
        assert sha(Path(path)) == expected, "Runtime dependency drift"
    return ref


def prepare():
    assert not OUT.exists() and not GROUP.exists() and not GPU_GROUP.exists()
    assert sha(PARENT / "FREEZE.json") == PARENT_SHA
    assert shutil.disk_usage(ROOT).free > 3 * 1024**3
    previous = read(PARENT / "FREEZE.json")
    refs = dict(previous["files"], **{"FREEZE.json": PARENT_SHA})
    for name in ("MAIN_GENERATION_SEAL.json", "INPUT_SEAL.json"):
        seal = read(PARENT / name)
        assert seal["freeze_sha256"] == PARENT_SHA
        refs[name] = sha(PARENT / name)
        refs.update(seal["files"])
    for pattern in ("*_HALT.json", "private/null_documents/*.json",
                    "private/null_slots/*.json", "private/llm_null/journal.jsonl",
                    "private/llm_null/responses/*.json"):
        refs.update({str(p.relative_to(PARENT)): sha(p) for p in PARENT.glob(pattern)})
    for name, expected in refs.items():
        assert sha(PARENT / name) == expected
    disposition = read(STAGE / "API_DISPOSITION.json")
    case = next(x for x in disposition["cases"] if x["experiment"] == "W05_LLM_Null")
    assert sha(PARENT / "private/llm_null/journal.jsonl") == case["journal_sha256"]
    assert case["phase_cap_nanodollars"] == 15_000_000_000
    assert case["unknown_request_replay_allowed"] is False
    for folder in (OUT, GROUP, GPU_GROUP):
        folder.mkdir(mode=0o700)
    for name, expected in previous["files"].items():
        checked_copy(PARENT / name, OUT / name, expected)
    # Share the verified metadata repair in a new runtime; never alter old frozen code.
    for name, expected in read(CAL / "FREEZE.json")["files"].items():
        if name.startswith("private/code/") and (
                not (OUT / name).exists() or name.removeprefix("private/code/") in (
                    "scripts/run_multiscale_transfer.py", "src/core/parser_recovery_audit.py",
                    "scripts/run_research_completion_queue.py")):
            checked_copy(CAL / name, OUT / name, expected)
    for name in ("swan_null_successor2.py", "test_swan_null_successor2.py"):
        checked_copy(STAGE / name, CODE / "scripts" / name, sha(STAGE / name))
    checked_copy(Path(__file__), OUT / "DEPLOYMENT_SOURCE.py", sha(Path(__file__)))
    checked_copy(STAGE / "API_DISPOSITION.json", OUT / "API_DISPOSITION.json",
                 sha(STAGE / "API_DISPOSITION.json"))
    inherited = {p: h for p, h in refs.items() if p.startswith("private/null_documents/")}
    for name in ("private/INPUTS.json", *inherited):
        checked_copy(PARENT / name, OUT / name, refs[name])
    for name, expected in read(PARENT / "MAIN_GENERATION_SEAL.json")["files"].items():
        checked_copy(PARENT / name, OUT / name, expected)
    for name in ("models", "vectors"):
        (CODE / name).symlink_to(ROOT / name, target_is_directory=True)
    # Keep the predeclared four-arm family, including pending A2, for simultaneous bounds.
    statistics = CODE / "scripts/swan_atomic_calibration1.py"
    original = statistics.read_text()
    old = "tests = len(arms) * len(conditions) * len(null_kinds)"
    assert original.count(old) == 1
    statistics.write_text(original.replace(
        old, "tests = max(len(ARMS), len(arms)) * len(conditions) * len(null_kinds)"))
    assets = {}
    for folder, filename in ((BANK, "FREEZE.json"), (SCORING, "RUNTIME_ENVIRONMENT.json")):
        spec = read(folder / filename)
        assets[str(folder / filename)] = sha(folder / filename)
        for name, expected in spec["files"].items():
            path = Path(name) if Path(name).is_absolute() else folder / name
            assert sha(path) == expected
            assets[str(path)] = expected
    for item in previous["parser"]["assets"]:
        assets[str(ROOT / item["path"])] = item["sha256"]
    assets[str(ROOT / "vectors/glove.6B.100d.txt")] = previous["glove_sha256"]
    assets[str(ALIGNMENT / "FREEZE.json")] = previous["alignment_parent_freeze_sha256"]
    for name, expected in {**previous["encoder_files"], **previous["surface_files"]}.items():
        assets[str(ALIGNMENT / name)] = expected
    write(OUT / "ASSET_REFERENCE.json", {"files": assets})
    write(OUT / "SOURCE_REFERENCE.json", {
        "root": str(PARENT), "files": refs, "inherited_null_documents": inherited,
        "old_halt_preserved": True, "new_graph_and_score_cache": True})
    write(OUT / "MISSING_RESPONSE_POLICY.json", {
        "version": "source-failure-no-replay/v1",
        "known_missing_call_sha256": case["call_id_sha256"],
        "known_missing_reserve_nanodollars": case["unknown_request_retained_reserve_nanodollars"],
        "all_original_request_ids_nonreplayable": True,
        "new_missing_response": "stop source, retain denominator/reserve, continue next source",
        "invalid_model_usage_schema_or_journal": "permanent_halt",
        "budgets": {"calls": 640000, "nanodollars": 15_000_000_000},
        "unchanged": ["sources", "model", "prompts", "temperature", "length_gate",
                      "attempt_cap", "scoring", "threshold_rule"],
        "multiplicity": "24 comparisons retained even while A2 is missing",
        "inference": "generated-null full-attempt FPR plus coverage; conditional FPR separate",
        "human_A2_required": True})
    (OUT / "AMENDMENT.md").write_text(
        "# W05 formal continuation v2\n\n"
        "Continue the original 800 null sources. Preserve all original records and HALTs.\n"
        "Inherit completed null documents byte-for-byte. Missing responses stop that source;\n"
        "retain its allocation and full reserved cost, never redispatch its requests.\n"
        "Apply this precommitted rule to future missing responses. Invalid model identity,\n"
        "usage/schema, input or journal integrity remains a permanent halt.\n"
        "Use the same model, prompt, 16 slots and 50 attempts, temperature, source split,\n"
        "150+ word design, score/threshold rule and cumulative $15 / 640000 request cap.\n"
        "Use fresh repaired parser caches and the verified CPU encoder dependency overlay.\n"
        "All 1800 sources x 3 conditions remain. Preserve 24-comparison upper bounds.\n"
        "Report generation coverage separately: an unavailable LLM text cannot establish\n"
        "text-conditional specificity. Human A2/quality cannot be supplied by this runner.\n")
    f = copy.deepcopy(previous)
    f.update(version="normal-length-formal/v2-missing-response-source-failure",
             parent_freeze_sha256=PARENT_SHA,
             bank_environment_freeze_sha256=sha(BANK / "FREEZE.json"),
             missing_response_policy_sha256=sha(OUT / "MISSING_RESPONSE_POLICY.json"))
    f["files"] = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob("*"))
                  if p.is_file() and not any(x in p.parts for x in ("models", "vectors", "__pycache__"))}
    write(OUT / "FREEZE.json", f)
    check_source()
    # Read-only accounting admission must match the approved disposition exactly.
    sys.path[:0] = [str(OVERLAY), str(BANK / "private/python_env"), str(CODE), str(CODE / "scripts")]
    import swan_null_successor2 as runner
    j = runner.Journal(OUT, PARENT, None)
    uncertain = [cid for cid, call in j.previous.calls.items()
                 if call["terminal"] is None or call["terminal"]["kind"] == "uncertain"]
    assert len(uncertain) == 1
    assert hashlib.sha256(uncertain[0].encode()).hexdigest() == case["call_id_sha256"]
    assert j.totals()["exposure_nanodollars"] == case["budget_exposure_nanodollars"]
    assert len(inherited) == 9
    assert {runner.source_of(cid) for cid in j.previous.calls} == (
        {Path(p).stem for p in inherited} | runner.unavailable_sources(j.previous))
    tests = []
    for index, pattern in enumerate(("test_swan_null_successor2.py",
                                     "test_swan_normal_completion1.py", "test_atomic*2.py")):
        path = OUT / "private" / ("regression_%d.log" % index)
        with path.open("w") as stream:
            result = subprocess.run([str(PYTHON), "-m", "unittest", "discover", "-s",
                str(CODE / "scripts"), "-p", pattern, "-v"], cwd=CODE, env=environment(),
                stdout=stream, stderr=subprocess.STDOUT)
        tests.append({"pattern": pattern, "returncode": result.returncode, "sha256": sha(path)})
        if result.returncode:
            print(path.read_text()[-6000:])
            raise RuntimeError("Formal successor regression failed")
    from run_atomic_alignment_completion2 import Encoder
    encoder = Encoder(ALIGNMENT / "private/encoder")
    text = ["Alice inspected the equipment."]
    assert (encoder.encode(text) == encoder.encode(text)).all()
    from swan_supplement1_sources import tokenizer
    assert tokenizer()(text[0]) == text
    write(GROUP / "PREPARATION.json", {
        "status": "PREPARED", "freeze_sha256": sha(OUT / "FREEZE.json"),
        "source_files_verified": len(refs), "asset_files_verified": len(assets),
        "inherited_null_documents": len(inherited), "missing_sources": 1,
        "budget": j.totals(), "tests": tests, "CPU_encoder_replay": True,
        "CUDA_precheck": "deferred_to_serial_GPU_turn", "new_calls": 0,
        "free_bytes": shutil.disk_usage(ROOT).free})
    print(json.dumps(read(GROUP / "PREPARATION.json"), indent=2))


def queue_module():
    spec = importlib.util.spec_from_file_location("queue2", CODE / "scripts/run_research_completion_queue.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def spawn(folder, manifest):
    with (folder / "scheduler.log").open("a") as stream:
        process = subprocess.Popen([str(PYTHON), str(CODE / "scripts/run_research_completion_queue.py"),
            str(manifest)], cwd=CODE, env=environment(), stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    write(folder / "LAUNCH.json", {"status": "QUEUE_LAUNCHED", "pid": process.pid,
                                 "manifest_sha256": sha(manifest), "at_unix": time.time()})


def launch():
    assert read(GROUP / "PREPARATION.json")["status"] == "PREPARED"
    assert not (GROUP / "MANIFEST.json").exists()
    f = read(OUT / "FREEZE.json")
    checks = {str(OUT / p): h for p, h in f["files"].items()}
    checks[str(OUT / "FREEZE.json")] = sha(OUT / "FREEZE.json")
    checks.update(read(OUT / "ASSET_REFERENCE.json")["files"])
    def job(name, action, folder, status, requires, extras):
        return {"id": name, "cwd": str(CODE),
                "argv": [str(PYTHON), str(OUT / "DEPLOYMENT_SOURCE.py"), action],
                "code_sha256": checks, "requires": requires, "completion": {
                    "terminal": {"path": str(folder / "TERMINAL.json"), "equals": {"status": status},
                                 "sha256_fields": {"result_sha256": "result", "audit_sha256": "seal"}},
                    "result": {"path": str(folder / "RESULT.json"), "equals": {"status": status}},
                    "seal": {"path": str(folder / "AUDIT.json"),
                             "equals": {"freeze_sha256": sha(OUT / "FREEZE.json")}} if folder == OUT
                            else {"path": str(folder / "AUDIT.json")},
                    **extras}}
    null_job = job("null", "generate", OUT, "NULL_GENERATION_COMPLETE_MISSING_RETAINED", [],
                   {"generation": {"path": str(OUT / "LLM_NULL_GENERATION_SEAL.json")}})
    manifest = GROUP / "MANIFEST.json"
    write(manifest, {"schema": "research-completion-queue/v2", "jobs": [null_job],
                     "resource_retry": {"max_retries": 0, "delay_seconds": 0}})
    scoring_job = job(
        "alignment", "score", OUT / "alignment", "NORMAL_LENGTH_A0_A1_COMPLETE_A2_AND_QUALITY_PENDING",
        [{"manifest": str(manifest), "manifest_sha256": sha(manifest), "job_id": "null"}],
        {"metrics": {"path": str(OUT / "alignment/COVERAGE_AND_INFERENCE.json")},
         "thresholds": {"path": str(OUT / "alignment/MACHINE_CALIBRATION.json")},
         "pretest": {"path": str(OUT / "alignment/BASELINE_SELECTION.json")}})
    write(GPU_GROUP / "MANIFEST.json", {"schema": "research-completion-queue/v2",
        "resource_lock": str(ROOT / "parser_repair_successors1_20261103/gpu.lock"),
        "resource_retry": {"max_retries": 120, "delay_seconds": 30}, "jobs": [scoring_job]})
    queue = queue_module()
    queue.load_plan(manifest)
    queue.load_plan(GPU_GROUP / "MANIFEST.json")
    spawn(GROUP, manifest)
    with (GPU_GROUP / "coordinator.log").open("a") as stream:
        process = subprocess.Popen([str(PYTHON), str(OUT / "DEPLOYMENT_SOURCE.py"), "coordinate"],
            cwd=CODE, env=environment(), stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
    write(GPU_GROUP / "COORDINATOR.json", {"pid": process.pid, "at_unix": time.time(),
                                         "status": "STARTED"})
    print(json.dumps({"generation": read(GROUP / "LAUNCH.json"),
                      "scoring": read(GPU_GROUP / "COORDINATOR.json")}))


def generate():
    check_source()
    import swan_null_successor2 as runner
    previous = read(PARENT / "private/queue/MANIFEST.json")
    keys = {j["argv"][j["argv"].index("--key-file") + 1]
            for j in previous["jobs"] if "--key-file" in j["argv"]}
    assert len(keys) == 1
    key = runner.normal.pilot.credential.read_credential(Path(next(iter(keys))))
    runner.execute(OUT, PARENT, lambda body: runner.budget.transport(body, key), check_source)


def coordinate():
    """Wait for CPU production before acquiring the shared GPU lock."""
    import fcntl
    with (GPU_GROUP / ".coordinator.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = GROUP / "MANIFEST.json"
        queue = queue_module()
        plan = queue.load_plan(manifest)
        while True:
            receipt = GROUP / "null_TERMINAL.json"
            if receipt.exists():
                if read(receipt)["status"] != "success":
                    write(GPU_GROUP / "COORDINATOR_STATUS.json", {
                        "status": "blocked", "parent_status": read(receipt)["status"]})
                    return
                queue.verify_success(manifest, plan, plan["jobs"][0], set())
                assert not (GPU_GROUP / "LAUNCH.json").exists()
                spawn(GPU_GROUP, GPU_GROUP / "MANIFEST.json")
                write(GPU_GROUP / "COORDINATOR_STATUS.json", {"status": "GPU_QUEUED"})
                return
            write(GPU_GROUP / "COORDINATOR_STATUS.json", {"status": "waiting_for_null_generation"})
            time.sleep(30)


def score():
    import run_swan_normal_completion1 as normal
    import swan_normal_alignment1 as alignment
    from swan_final1_common import freeze, single_writer
    check_source()
    with single_writer(OUT):
        try:
            normal.verify(OUT)
            normal.verify_seal(OUT, OUT / "LLM_NULL_GENERATION_SEAL.json")
            normal.file_seal(OUT, OUT / "MAIN_GENERATION_SEAL.json", [
                OUT / name for name in read(PARENT / "MAIN_GENERATION_SEAL.json")["files"]])
            normal.file_seal(OUT, OUT / "INPUT_SEAL.json", [
                OUT / "private/INPUTS.json", OUT / "SOURCE_REFERENCE.json"])
            alignment.run(OUT, ROOT, ALIGNMENT)
            meta = read(OUT / "alignment/private/METADATA.json")
            pred = read(OUT / "alignment/private/PREDICTIONS.json")
            fitted = read(OUT / "alignment/MACHINE_CALIBRATION.json")
            choice = read(OUT / "alignment/BASELINE_SELECTION.json")["selected_family"]
            rows = alignment.endpoint_rows(pred, meta, choice)
            from swan_atomic_calibration1 import reject, binomial
            coverage = {}
            for kind in ("human", "llm"):
                coverage[kind] = {}
                for condition in normal.CONDITIONS:
                    selected = [r for r in rows if r["kind"] == kind
                                and r["role"] == "null_test" and r["condition"] == condition]
                    coverage[kind][condition] = {}
                    for arm in fitted["arms"]:
                        available = [r for r in selected if r["scores"][arm] is not None]
                        hits = sum(reject(r["scores"][arm], fitted["thresholds"][arm]) for r in available)
                        coverage[kind][condition][arm] = {
                            "full_n": len(selected), "available_n": len(available),
                            "unavailable_n": len(selected) - len(available),
                            "conditional_available_FPR": binomial(hits, len(available), .05/24)
                            if available else None}
            result = read(OUT / "alignment/RESULT.json")
            assert result["FPR_simultaneous_test_count"] == 24
            freeze(OUT / "alignment/COVERAGE_AND_INFERENCE.json", {
                "coverage": coverage, "machine_result_sha256": sha(OUT / "alignment/RESULT.json"),
                "generation_result_sha256": sha(OUT / "RESULT.json"),
                "LLM_full_denominator_FPR": "generation-and-detection pipeline over attempts",
                "conditional_available_FPR": "selection-conditional descriptive result; no general guarantee",
                "independent_human_A2": False, "independent_quality": False})
            check_source()
        except Exception as exc:
            pause = isinstance(exc, normal.rt.ResourcePause) or type(exc).__name__ == "OutOfMemoryError"
            write(OUT / "alignment" / ("PAUSED.json" if pause else "HALT.json"),
                  {"reason_type": type(exc).__name__, "automatic_restart_allowed": pause})
            if pause:
                raise SystemExit(75)
            raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("prepare", "launch", "generate", "coordinate", "score"))
    globals()[ap.parse_args().action]()
