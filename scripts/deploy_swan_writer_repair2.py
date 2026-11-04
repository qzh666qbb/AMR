"""Freeze and run bounded writer development probes; preserve all old runtimes."""
import argparse
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
CAL = ROOT / "independent_calibration2_runtime"
BANK = ROOT / "independent_banks2_runtime"
STAGE = Path("/tmp/swan_writer2_stage")
PYTHON = ROOT / ".venv/bin/python"
PARENT_SHA = "8e8391060be15d67d1bf68c1366d256196c86c35a3c5985d64b4f6fe9a84c22e"


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        assert path.read_text() == data, "Immutable file changed"
    else:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(data)
        temporary.replace(path)


def checked_copy(source, target, expected):
    assert sha(source) == expected
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert sha(target) == expected


def environment(out):
    code = out / "private/code"
    return dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, [
        ROOT / "atomic_scoring_dependencies1/overlay", BANK / "private/python_env",
        code, code / "scripts"])), NLTK_DATA=str(code / ".nltk_data"),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")


def verify(out):
    f = read(out / "FREEZE.json")
    assert sha(PARENT / "FREEZE.json") == PARENT_SHA
    for name, expected in f["files"].items():
        assert sha(out / name) == expected, "Frozen file changed: " + name
    for name, expected in read(out / "REFERENCE.json")["files"].items():
        assert sha(Path(name)) == expected, "Read-only dependency changed"
    return f


def prepare(out, stage, predecessor=None):
    assert not (out / "FREEZE.json").exists() and not (out / "LAUNCH.json").exists()
    assert not (out / "private/writer2/journal.jsonl").exists()
    assert shutil.disk_usage(ROOT).free > 2 * 1024**3
    assert sha(PARENT / "FREEZE.json") == PARENT_SHA
    original, repaired = read(PARENT / "FREEZE.json"), read(CAL / "FREEZE.json")
    references = {str(PARENT / "FREEZE.json"): PARENT_SHA}
    for name, expected in original["files"].items():
        checked_copy(PARENT / name, out / name, expected)
        references[str(PARENT / name)] = expected
    # Copy repaired parsing code into this new runtime only.
    for name, expected in repaired["files"].items():
        if name.startswith("private/code/"):
            checked_copy(CAL / name, out / name, expected)
    for name in ("swan_writer_repair2.py", "test_swan_writer_repair2.py"):
        checked_copy(STAGE / name, out / "private/code/scripts" / name, sha(STAGE / name))
    checked_copy(STAGE / "deploy_swan_writer_repair2.py", out / "DEPLOYMENT_SOURCE.py",
                 sha(STAGE / "deploy_swan_writer_repair2.py"))
    checked_copy(STAGE / "PROTOCOL.md", out / "PROTOCOL.md", sha(STAGE / "PROTOCOL.md"))
    main_seal = read(PARENT / "MAIN_GENERATION_SEAL.json")
    references[str(PARENT / "MAIN_GENERATION_SEAL.json")] = sha(PARENT / "MAIN_GENERATION_SEAL.json")
    references.update({str(PARENT / p): h for p, h in main_seal["files"].items()})
    for path, expected in references.items():
        assert sha(path) == expected
    parent_budget = read(PARENT / "MAIN_BUDGET.json")
    if predecessor:
        terminal = read(predecessor / "TERMINAL.json")
        assert terminal["result_sha256"] == sha(predecessor / "RESULT.json")
        assert terminal["audit_sha256"] == sha(predecessor / "AUDIT.json")
        result = read(predecessor / "RESULT.json")
        parent_budget = {
            "calls": result["budget"]["cumulative_calls"],
            "exposure_nanodollars": result["budget"]["cumulative_exposure_nanodollars"]}
        for name in ("FREEZE.json", "RESULT.json", "AUDIT.json", "TERMINAL.json"):
            references[str(predecessor / name)] = sha(predecessor / name)
        for name, expected in read(predecessor / "AUDIT.json")["files"].items():
            assert sha(predecessor / name) == expected, "Predecessor output changed"
    if stage == "full_document":
        assert predecessor and result["stage"] == "proposal_probe"
        assert result["accepted_sentences"] >= 12
        assert result["budget"]["new_version"]["calls"] == 16
        review = read(STAGE / "MODEL_REVIEW_GATE.json")
        assert review["predecessor_result_sha256"] == sha(predecessor / "RESULT.json")
        assert review["quality_passed"] is True and review["reviewer_type"] == "assistant_model"
        checked_copy(STAGE / "MODEL_REVIEW_GATE.json", out / "MODEL_REVIEW_GATE.json",
                     sha(STAGE / "MODEL_REVIEW_GATE.json"))
    design = read(PARENT / "private/ALIGNMENT_DESIGN.json")
    selected_ids = design["positive_development"][:8 if stage == "proposal_probe" else 4]
    byid = {r["allocation_id"]: r for r in read(PARENT / "private/sources/main.json")}
    assert not set(selected_ids) & set(design["positive_test"])
    write(out / "private/SOURCES.json", [byid[i] for i in selected_ids])
    # Pin normalizer code, models and repaired parser assets without copying models.
    for name, expected in read(BANK / "FREEZE.json")["files"].items():
        references[str(BANK / name)] = expected
    environment_path = BANK / "PARENT_NORMALIZER_ENVIRONMENT.json"
    normalizer_environment = read(environment_path)
    references[str(environment_path)] = sha(environment_path)
    references.update({str(BANK / p): h for p, h in normalizer_environment["files"].items()})
    references.update({str(ROOT / a["path"]): a["sha256"] for a in repaired["parser"]["assets"]})
    references[str(ROOT / "vectors/glove.6B.100d.txt")] = original["glove_sha256"]
    write(out / "REFERENCE.json", {"files": references})
    write(out / "FREEZE.json", {
        "version": out.name, "stage": stage, "parent_freeze_sha256": PARENT_SHA,
        "parent_budget": parent_budget, "parser": repaired["parser"],
        "execution": {"resumable_gpu_pressure": True},
        "resource_gate": {"vram_fraction_max": .8},
        "glove_sha256": original["glove_sha256"],
        "sources": selected_ids, "confirmation_sources_used": 0,
        "proposal": "topic-specific collective + has/have been + transitive verb-ing + concrete bare object; exclude nominalized propositions; 7-9 words",
        "acceptance": "all original Bank50 entries; maximum S2Match >=0.75; original 7-12 word sentence gate",
        "parser_rule": "FP32 CUDA; native limit; beam0 and diagnostic/roundtrip audit",
        "counts": {"sources": len(selected_ids), "slots": 2 if stage == "proposal_probe" else 16,
                   "attempts_per_slot": 1 if stage == "proposal_probe" else 50},
        "caps": {"new_calls": 16 if stage == "proposal_probe" else 3200,
                 "new_nanodollars": 3_000_000_000,
                 "cumulative_parent_calls": 160000, "cumulative_parent_nanodollars": 40_000_000_000},
        "probe_go_gate": "at least 12/16 candidate acceptance plus model semantic review; otherwise no full document run",
        "full_document_gate": "at least 3/4 complete documents and acceptable model quality; no automatic expansion",
        "old_cohort_modified": False, "formal_TPR_or_FPR": False,
        "files": {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*"))
                  if p.is_file() and "__pycache__" not in p.parts},
    })
    for name in ("models", "vectors"):
        (out / "private/code" / name).symlink_to(ROOT / name, target_is_directory=True)
    verify(out)
    with (out / "private/tests.log").open("w") as log:
        completed = subprocess.run(
            [str(PYTHON), "-m", "unittest", "discover", "-s", str(out / "private/code/scripts"),
             "-p", "test_swan_writer_repair2.py", "-v"],
            env=environment(out), stdout=log, stderr=subprocess.STDOUT)
    assert completed.returncode == 0, "Writer tests failed; see private/tests.log"
    write(out / "DEPLOYMENT.json", {"status": "FROZEN_TESTED", "reference_files": len(references),
                                   "freeze_sha256": sha(out / "FREEZE.json"), "new_provider_calls": 0})
    print(json.dumps(read(out / "DEPLOYMENT.json")))


def run(out):
    sys.path[:0] = environment(out)["PYTHONPATH"].split(os.pathsep)
    import swan_writer_repair2 as writer
    old_manifest = read(ROOT / "v4_confirmation1_runtime/experiments/swan_v4_independent_confirmation1/private/queue/MANIFEST.json")
    keys = [job["argv"][job["argv"].index("--key-file") + 1]
            for job in old_manifest["jobs"] if "--key-file" in job["argv"]]
    assert len(keys) == 1
    key = writer.normal.pilot.credential.read_credential(Path(keys[0]))
    try:
        writer.run(out, ROOT, BANK, lambda body: writer.budget.transport(body, key), lambda: verify(out))
    except (writer.normal.rt.ResourcePause, writer.budget.Pause) as exc:
        write(out / "PAUSED.json", {"reason_type": type(exc).__name__, "reason": str(exc)})
        raise SystemExit(75)
    except Exception as exc:
        write(out / "HALT.json", {"reason_type": type(exc).__name__, "automatic_restart_allowed": False})
        raise


def launch(out):
    verify(out)
    with (out / "private/run.log").open("a") as log:
        process = subprocess.Popen(
            [str(PYTHON), str(out / "DEPLOYMENT_SOURCE.py"), "run", "--name", out.name],
            env=environment(out), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    write(out / "LAUNCH.json", {"pid": process.pid, "at_unix": time.time()})
    print(json.dumps({"status": "LAUNCHED", "pid": process.pid, "runtime": out.name}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "launch"))
    parser.add_argument("--name", default="normal_writer_repair2_probe1")
    parser.add_argument("--stage", choices=("proposal_probe", "full_document"), default="proposal_probe")
    parser.add_argument("--predecessor")
    args = parser.parse_args()
    assert args.name.startswith("normal_writer_repair2_") and "/" not in args.name
    out = ROOT / args.name
    if args.action == "prepare":
        prepare(out, args.stage, ROOT / args.predecessor if args.predecessor else None)
    elif args.action == "launch":
        launch(out)
    else:
        run(out)
