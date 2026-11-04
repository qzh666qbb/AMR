"""Freeze and run the repaired calibration successor, with verified dependencies."""
import argparse
import copy
import gc
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
PARENT = ROOT / "independent_calibration1_runtime"
BANK = ROOT / "independent_banks2_runtime"
OUT = ROOT / "independent_calibration2_runtime"
GROUP = ROOT / "independent_calibration2_queue"
CODE = OUT / "private/code"
PYTHON = ROOT / ".venv/bin/python"
BANK_QUEUE = ROOT / "parser_repair_successors1_20261103/MANIFEST.json"
PINS = {
    "parent": "d3ea8da4585773f9c35111856cebbed2f2a5b975d8afb0a6f0dddf5004b3649e",
    "bank": "dec02bd4be005497afc854502084d9d2b9e1dae2889a70c63b2fea331f06533c",
    "bank_receipt": "2f6cae3abac53a05f51767757362d3f1fc5c8ef56cc68af442f948ab9cd06b2e",
}
PATCHES = ("scripts/run_multiscale_transfer.py", "src/core/parser_recovery_audit.py",
           "scripts/run_research_completion_queue.py")


def sha(p):
    digest = hashlib.sha256()
    with Path(p).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(p):
    return json.loads(p.read_text())


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(".tmp")
    with temp.open("w") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(p)


def copy_checked(source, target, expected):
    assert sha(source) == expected, "Source SHA drift: " + source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert sha(target) == expected


def environment():
    return dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, [
        BANK / "private/python_env", CODE, CODE / "scripts"])),
        NLTK_DATA=str(BANK / "private/code/.nltk_data"),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")


def queue_module():
    path = BANK / "private/code/scripts/run_research_completion_queue.py"
    spec = importlib.util.spec_from_file_location("successor_queue", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_bank():
    assert sha(BANK / "FREEZE.json") == PINS["bank"]
    assert sha(BANK_QUEUE.parent / "banks_TERMINAL.json") == PINS["bank_receipt"]
    queue = queue_module()
    plan = queue.load_plan(BANK_QUEUE)
    job = next(j for j in plan["jobs"] if j["id"] == "banks")
    queue.verify_success(BANK_QUEUE, plan, job, set())
    f = read(BANK / "FREEZE.json")
    for name, expected in f["files"].items():
        assert sha(BANK / name) == expected, "Bank dependency drift"
    raw = read(BANK / "RAW_GRAPH_SEAL.json")
    for name, expected in raw["files"].items():
        assert sha(BANK / name) == expected, "Bank graph drift"
    assert len(raw["files"]) == 2329
    return f


def prepare():
    assert not OUT.exists() and not GROUP.exists(), "Use fresh successor directories"
    assert sha(PARENT / "FREEZE.json") == PINS["parent"]
    previous, bank = read(PARENT / "FREEZE.json"), verify_bank()
    assert shutil.disk_usage(ROOT).free > 3 * 1024**3
    OUT.mkdir(mode=0o700)
    GROUP.mkdir(mode=0o700)
    for name, expected in previous["files"].items():
        assert not any(s in Path(name).parts for s in ("raw_graphs", "rows", "sessions", "queue"))
        copy_checked(PARENT / name, OUT / name, expected)
    additions = {}
    for name, expected in bank["files"].items():
        if not name.startswith("private/code/"):
            continue
        target = OUT / name
        if not target.exists() or name.removeprefix("private/code/") in PATCHES:
            copy_checked(BANK / name, target, expected)
            additions[name] = expected
    copy_checked(Path(__file__), OUT / "DEPLOYMENT_SOURCE.py", sha(Path(__file__)))
    sessions = [{"name": p.name, "sha256": sha(p), **read(p)}
                for p in sorted((PARENT / "private/sessions").glob("*.json"))]
    write(OUT / "CARRY_FORWARD_COST.json", {
        "parent_sessions": sessions,
        "prior_worker_elapsed_seconds": sum(s.get("seconds", 0) for s in sessions),
        "bank_carry_forward_sha256": sha(BANK / "CARRY_FORWARD_COST.json"),
        "bank_sessions": [{"name": p.name, "sha256": sha(p), **read(p)}
                          for p in sorted((BANK / "private/sessions").glob("*.json"))],
        "prior_calibration_graphs": len(list((PARENT / "private/raw_graphs").glob("*.json"))),
        "prior_calibration_rows": len(list((PARENT / "private/rows").glob("*.json"))),
        "provider_calls_budget": 0, "historical_cost_reset": False,
        "note": "Session wall time is not exact GPU kernel time; shared Bank costs remain separate.",
    })
    write(OUT / "PARENT_PROVENANCE.json", {
        "parent_freeze_sha256": PINS["parent"], "parent_halt_sha256": sha(PARENT / "HALT.json"),
        "bank_freeze_sha256": PINS["bank"], "bank_queue_sha256": sha(BANK_QUEUE),
        "bank_success_receipt_sha256": PINS["bank_receipt"],
        "original_inputs_sha256": sha(PARENT / "private/INPUTS.json"),
        "added_or_repaired_files": additions, "new_cache": True,
    })
    (OUT / "AMENDMENT.md").write_text(
        "# Independent calibration successor v2\n\n"
        "User authorized sequential completion on 2026-11-03. Bind completed Bank v2.\n"
        "Preserve v1 HALT and input bytes. Same 400 calibration + 400 independent original\n"
        "sources, five conditions, five flows, FP32, native context, official normalizer,\n"
        "S2Match, alpha .01 and full erasure denominators. Only the verified parser metadata\n"
        "repair, runtime closure, parent binding and queue dependency contract change.\n"
        "No generated samples, provider requests, old graph cache or old score reuse.\n"
        "Real parser/normalizer/matcher replay must pass before full computation.\n"
        "All raw graphs precede scoring. Thresholds seal before the first Null score.\n"
        "This endpoint provides original-source FPR only, not TPR or human quality evidence.\n")
    frozen = copy.deepcopy(previous)
    frozen.update(version="independent-bank-calibration/v2-parser-metadata-repair",
                  parent_calibration_freeze_sha256=PINS["parent"],
                  parent_bank_freeze_sha256=PINS["bank"],
                  carry_forward_cost_sha256=sha(OUT / "CARRY_FORWARD_COST.json"))
    frozen["files"] = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
    write(OUT / "FREEZE.json", frozen)
    assert sha(OUT / "private/INPUTS.json") == sha(PARENT / "private/INPUTS.json")
    records = []
    tests = [
        [str(CODE / "scripts/run_swan_independent_calibration1.py"), "verify",
         "--out", str(OUT), "--bank", str(BANK)],
        ["-m", "unittest", "discover", "-s", str(CODE / "scripts"),
         "-p", "test_swan_independent_calibration1.py", "-v"],
        ["-m", "unittest", "tests.test_swan_parser_recovery_audit",
         "tests.test_research_completion_queue", "-v"],
    ]
    for index, args in enumerate(tests):
        logpath = OUT / "private" / ("static_%d.log" % index)
        with logpath.open("w") as log:
            result = subprocess.run([str(PYTHON), *args], cwd=CODE, env=environment(),
                                    stdout=log, stderr=subprocess.STDOUT)
        records.append({"check": index, "returncode": result.returncode, "log_sha256": sha(logpath)})
        if result.returncode:
            print(logpath.read_text()[-5000:])
            raise RuntimeError("Static readiness failed")
    write(GROUP / "PREPARATION.json", {
        "status": "PREPARED", "freeze_sha256": sha(OUT / "FREEZE.json"),
        "counts": frozen["counts"], "frozen_files": len(frozen["files"]),
        "bank_files_verified": len(bank["files"]), "bank_graphs_verified": 2329,
        "checks": records, "new_provider_calls": 0, "free_bytes": shutil.disk_usage(ROOT).free})
    print(json.dumps(read(GROUP / "PREPARATION.json"), indent=2))


def readiness():
    sys.path[:0] = [str(BANK / "private/python_env"), str(CODE), str(CODE / "scripts")]
    import run_multiscale_transfer as rt
    import run_swan_independent_calibration1 as cal
    from run_swan_independent_banks1 import Normalizer
    from semantic_fact_raw_amr import audit_raw, replay_signature
    start = time.monotonic()
    f = cal.verify(OUT, BANK)
    verify_bank()
    assert sha(ROOT / "vectors/glove.6B.100d.txt") == f["glove_sha256"]
    ready = OUT / "readiness"
    assert not (ready / "TERMINAL.json").exists()
    normalizer = Normalizer(BANK)
    matcher = cal.Matcher(OUT, ROOT)
    synthetic = "(w / want-01 :ARG0 (b / boy) :ARG1 (g / go-01 :ARG0 b))"
    normalized = [normalizer(synthetic) for _ in range(2)]
    assert normalized[0] == normalized[1] and normalized[0]["eligible"]
    self_scores = [matcher.fresh(normalized[0]["template"], normalized[0]["template"]) for _ in range(2)]
    assert self_scores == [1., 1.]
    parser = rt.Parser(ROOT, OUT, f)
    texts = sorted({s for d in read(OUT / "private/INPUTS.json") for s in d["sentences"]}, key=rt.digest)
    cases = []
    for text in texts:
        if len(parser.model.tokenizer(text, truncation=False)["input_ids"]) >= 1024:
            continue
        records = [{**parser.fresh(text), "case_id": rt.digest(text)} for _ in range(2)]
        assert replay_signature(records[0]) == replay_signature(records[1])
        checks = [audit_raw(r, {"case_id": rt.digest(text), "input_text_sha256": rt.digest(text)},
                            parser.freeze_sha) for r in records]
        for i, record in enumerate(records):
            write(ready / "private" / (rt.digest(text) + "_" + str(i) + ".json"), record)
        cases.append({"text_sha256": rt.digest(text), "two_replays_equal": True,
                      "eligible": checks[0]["structurally_eligible"]})
        if len(cases) == 2:
            break
    assert len(cases) == 2
    del parser
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    matcher.db.close()
    cal.verify(OUT, BANK)
    write(ready / "RESULT.json", {"status": "PASS", "dtype": "float32", "cases": cases,
        "normalizer_double_replay": True, "S2Match_self_replays": self_scores,
        "new_provider_calls": 0, "elapsed_seconds": time.monotonic() - start})
    write(ready / "AUDIT.json", {"freeze_sha256": sha(OUT / "FREEZE.json"),
        "source_files_verified": True, "bank_success_verified": True,
        "raw_records": {str(p.relative_to(ready)): sha(p) for p in (ready / "private").glob("*.json")}})
    write(ready / "TERMINAL.json", {"status": "PASS", "result_sha256": sha(ready / "RESULT.json"),
                                   "audit_sha256": sha(ready / "AUDIT.json")})
    print(json.dumps(read(ready / "RESULT.json"), indent=2))


def launch():
    assert read(GROUP / "PREPARATION.json")["status"] == "PREPARED"
    assert not (GROUP / "MANIFEST.json").exists()
    f = read(OUT / "FREEZE.json")
    checks = {str(OUT / n): h for n, h in f["files"].items()}
    checks[str(OUT / "FREEZE.json")] = sha(OUT / "FREEZE.json")
    bankdep = {"manifest": str(BANK_QUEUE), "manifest_sha256": sha(BANK_QUEUE),
               "job_id": "banks", "terminal_sha256": PINS["bank_receipt"]}
    jobs = []
    for action, folder, status in (
        ("readiness", OUT / "readiness", "PASS"),
        ("calibration", OUT, "CALIBRATION_AND_INDEPENDENT_NULL_COMPLETE"),
    ):
        completion = {
            "terminal": {"path": str(folder / "TERMINAL.json"), "equals": {"status": status},
                         "sha256_fields": {"result_sha256": "result", "audit_sha256": "seal"}},
            "result": {"path": str(folder / "RESULT.json"), "equals": {"status": status}},
            "seal": {"path": str(folder / "AUDIT.json"),
                     "equals": {"freeze_sha256": sha(OUT / "FREEZE.json")}},
        }
        if action == "readiness":
            args = [str(OUT / "DEPLOYMENT_SOURCE.py"), "readiness"]
            requires = [bankdep]
            completion["result"]["equals"].update(normalizer_double_replay=True,
                                                  S2Match_self_replays=[1., 1.])
        else:
            args = [str(CODE / "scripts/run_swan_independent_calibration1.py"), "run",
                    "--out", str(OUT), "--bank", str(BANK), "--assets", str(ROOT)]
            requires = [bankdep, {"job_id": "readiness"}]
            completion["result"]["equals"].update(rows=4000, calibration_sources=400,
                                                   independent_human_null_sources=400)
            completion["seal"]["equals"]["scored_all_4000_condition_rows"] = True
            completion["seal"]["sha256_fields"] = {"thresholds_sha256": "thresholds",
                "raw_graph_seal_sha256": "graphs", "bank_binding_sha256": "binding"}
            completion["thresholds"] = {"path": str(OUT / "THRESHOLDS.json")}
            completion["graphs"] = {"path": str(OUT / "RAW_GRAPH_SEAL.json"),
                                   "equals": {"freeze_sha256": sha(OUT / "FREEZE.json")}}
            completion["binding"] = {"path": str(OUT / "private/BANK_BINDING.json")}
        env = environment()
        argv = ["/usr/bin/env"] + [k + "=" + env[k] for k in (
            "PYTHONPATH", "NLTK_DATA", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
            "TOKENIZERS_PARALLELISM", "PYTHONDONTWRITEBYTECODE")] + [str(PYTHON), *args]
        jobs.append({"id": action, "cwd": str(CODE), "argv": argv,
                     "code_sha256": checks, "requires": requires, "completion": completion})
    manifest = GROUP / "MANIFEST.json"
    write(manifest, {"schema": "research-completion-queue/v2", "jobs": jobs,
                     "resource_lock": str(ROOT / "parser_repair_successors1_20261103/gpu.lock"),
                     "resource_retry": {"max_retries": 120, "delay_seconds": 30}})
    queue_module().load_plan(manifest)
    with (GROUP / "scheduler.log").open("a") as log:
        process = subprocess.Popen([str(PYTHON), str(CODE / "scripts/run_research_completion_queue.py"),
                                    str(manifest)], cwd=CODE, env=environment(),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(GROUP / "LAUNCH.json", {"status": "QUEUE_LAUNCHED", "pid": process.pid,
        "manifest_sha256": sha(manifest), "freeze_sha256": sha(OUT / "FREEZE.json"),
        "counts": f["counts"], "provider_calls_budget": 0, "at_unix": time.time()})
    print(json.dumps(read(GROUP / "LAUNCH.json"), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "readiness", "launch"))
    globals()[parser.parse_args().action]()
