"""Freeze a new document-planning run without changing any predecessor."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import deploy_swan_writer_repair2 as prior

ROOT, PYTHON, BANK = prior.ROOT, prior.PYTHON, prior.BANK
STAGE = Path("/tmp/swan_writer3_stage")
PREVIOUS = ROOT / "normal_writer_repair3_full1"
read, sha, write, checked_copy = prior.read, prior.sha, prior.write, prior.checked_copy


def prepare(out):
    assert not out.exists(), "Use a new runtime"
    prior.verify(PREVIOUS)
    terminal, result = read(PREVIOUS / "TERMINAL.json"), read(PREVIOUS / "RESULT.json")
    assert terminal["result_sha256"] == sha(PREVIOUS / "RESULT.json")
    assert terminal["audit_sha256"] == sha(PREVIOUS / "AUDIT.json")
    for name, expected in read(PREVIOUS / "AUDIT.json")["files"].items():
        assert sha(PREVIOUS / name) == expected
    old = read(PREVIOUS / "FREEZE.json")
    references = read(PREVIOUS / "REFERENCE.json")["files"]
    for name in ("FREEZE.json", "RESULT.json", "AUDIT.json", "TERMINAL.json"):
        references[str(PREVIOUS / name)] = sha(PREVIOUS / name)
    for name, expected in old["files"].items():
        if name.startswith("private/code/") or name in ("private/SOURCES.json", "private/amr_bank_50.json"):
            checked_copy(PREVIOUS / name, out / name, expected)
            references[str(PREVIOUS / name)] = expected
    for source in read(out / "private/SOURCES.json"):
        sid = source["allocation_id"]
        for folder in ("plans", "reviews"):
            filename = sid + ("-plan.json" if folder == "reviews" else ".json")
            source_path = PREVIOUS / "private" / folder / filename
            checked_copy(source_path, out / "private/initial" / folder / filename, sha(source_path))
            references[str(source_path)] = sha(source_path)
    for name in ("swan_writer_repair3.py", "swan_writer3_budget.py", "test_swan_writer_repair3.py",
                 "deploy_swan_writer_repair2.py", "deploy_swan_writer_repair3.py"):
        checked_copy(STAGE / name, out / "private/code/scripts" / name, sha(STAGE / name))
    checked_copy(STAGE / "deploy_swan_writer_repair3.py", out / "DEPLOYMENT_SOURCE.py",
                 sha(STAGE / "deploy_swan_writer_repair3.py"))
    checked_copy(STAGE / "PROTOCOL.md", out / "PROTOCOL.md", sha(STAGE / "PROTOCOL.md"))
    write(out / "REFERENCE.json", {"files": references})
    write(out / "FREEZE.json", {
        "version": out.name, "stage": "document_plan_and_semantic_guard",
        "parent_budget": {"calls": result["budget"]["cumulative_calls"],
                          "exposure_nanodollars": result["budget"]["cumulative_exposure_nanodollars"]},
        "parser": old["parser"], "glove_sha256": old["glove_sha256"],
        "execution": old["execution"], "resource_gate": old["resource_gate"],
        "sources": old["sources"], "confirmation_sources_used": 0,
        "counts": {"sources": 4, "slots": 16, "attempts_per_slot": 4, "plan_revisions": 1},
        "caps": {"new_calls": 1100, "new_nanodollars": 3_000_000_000},
        "provider": {"model": "deepseek-flash", "temperature": .7, "top_p": .9,
                     "max_output": {"plan": 2400, "review": 600, "candidate": 50},
                     "max_request_bytes": 24000, "transport_retries": 0},
        "acceptance": "Bank50 max>=0.75 AND 7-12 words AND model semantic guard",
        "plan_before_scores": True, "review_scores_visible": False,
        "reviewer_independent_human": False, "automatic_scale_up": False,
        "old_cohort_modified": False,
        "failed_slots": "retain planned sentence ONLY in diagnostic draft; never count as written",
        "files": {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
    })
    for name in ("models", "vectors"):
        (out / "private/code" / name).symlink_to(ROOT / name, target_is_directory=True)
    prior.verify(out)
    with (out / "private/tests.log").open("w") as log:
        done = subprocess.run([str(PYTHON), "-m", "unittest", "discover",
            "-s", str(out / "private/code/scripts"), "-p", "test_swan_writer_repair*.py", "-v"],
            env=prior.environment(out), stdout=log, stderr=subprocess.STDOUT)
    assert done.returncode == 0, "See private/tests.log"
    write(out / "DEPLOYMENT.json", {"status": "FROZEN_TESTED", "freeze_sha256": sha(out / "FREEZE.json"),
                                   "reference_files": len(references), "new_provider_calls": 0})
    print(json.dumps(read(out / "DEPLOYMENT.json")))


def run(out):
    sys.path[:0] = prior.environment(out)["PYTHONPATH"].split(os.pathsep)
    import swan_writer_repair3 as writer
    manifest = read(ROOT / "v4_confirmation1_runtime/experiments/swan_v4_independent_confirmation1/private/queue/MANIFEST.json")
    keys = [j["argv"][j["argv"].index("--key-file") + 1] for j in manifest["jobs"] if "--key-file" in j["argv"]]
    assert len(keys) == 1
    key = writer.normal.pilot.credential.read_credential(Path(keys[0]))
    try:
        writer.run(out, ROOT, BANK, lambda body: writer.budget.transport(body, key), lambda: prior.verify(out))
    except (writer.normal.rt.ResourcePause, writer.budget.Pause) as exc:
        write(out / "PAUSED.json", {"reason_type": type(exc).__name__, "reason": str(exc)})
        raise SystemExit(75)
    except Exception as exc:
        write(out / "HALT.json", {"reason_type": type(exc).__name__, "automatic_restart_allowed": False})
        raise


def launch(out):
    assert not (out / "LAUNCH.json").exists(), "Already launched"
    prior.verify(out)
    with (out / "private/run.log").open("a") as log:
        process = subprocess.Popen([str(PYTHON), str(out / "DEPLOYMENT_SOURCE.py"), "run", "--name", out.name],
            env=prior.environment(out), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    write(out / "LAUNCH.json", {"pid": process.pid, "at_unix": time.time()})
    print(json.dumps({"status": "LAUNCHED", "runtime": out.name, "pid": process.pid}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "launch", "run"))
    parser.add_argument("--name", default="normal_writer_repair3_full2")
    args = parser.parse_args()
    assert args.name.startswith("normal_writer_repair3_") and "/" not in args.name
    {"prepare": prepare, "launch": launch, "run": run}[args.action](ROOT / args.name)
