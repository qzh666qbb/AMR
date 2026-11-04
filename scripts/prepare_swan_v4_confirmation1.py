"""Snapshot verified historical detector dependencies and new confirmation code."""
import argparse
from pathlib import Path
import shutil
import tarfile

from swan_final1_common import freeze, load, now, sha256
import prepare_swan_path0_completion1 as previous
import run_swan_v4_confirmation1 as run
import swan_v4_confirmation1_sources as sources

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/swan_v4_independent_confirmation1"
SNAPSHOT = Path("/tmp/swan-v4-confirmation1-runtime")
FILES = (
    "PROTOCOL.md", "SOURCE_CONTRACT.json", "INPUT_MANIFEST.json", "SOURCE_AUDIT.json",
    "private/HISTORY.json", "private/SOURCE_POOL.jsonl",
    "sealed/calibration.jsonl", "sealed/null.jsonl", "sealed/positive_source.jsonl",
    "RUN_AMENDMENT.md", "HUMAN_REVIEW.md",
)
CODE = (
    "scripts/run_swan_v4_confirmation1.py", "scripts/swan_v4_confirmation1_scoring.py",
    "scripts/test_swan_v4_confirmation1.py", "scripts/prepare_swan_v4_confirmation1.py",
    "scripts/test_swan_v4_confirmation1_sources.py",
)


def snapshot():
    if (OUT / "FREEZE.json").exists():
        raise ValueError("Frozen runtime cannot be modified")
    sources.audit()
    old = load(ROOT / "experiments/swan_path0_completion1/FREEZE.json")
    # Use only original runtime byte identities. No current core may substitute.
    for relative, expected in old["dependencies"].items():
        if relative.startswith(("scripts/", "src/")):
            path = previous.SNAPSHOT / relative
            if sha256(path) != expected:
                raise ValueError("Historical runtime drift: " + relative)
            previous.copy(path, SNAPSHOT / relative)
    for relative in sorted(set(CODE + sources.CODE)):
        if relative in CODE:
            (SNAPSHOT / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, SNAPSHOT / relative)
        else:
            previous.copy(ROOT / relative, SNAPSHOT / relative)
    bank = run.shared.read_bank()
    assigned = run.shared.assignment(run.read_sources("positive_source"), bank)
    freeze(OUT / "GENERATION_ASSIGNMENT.json", assigned)
    for relative in (*FILES, "GENERATION_ASSIGNMENT.json"):
        previous.copy(OUT / relative, SNAPSHOT / OUT.relative_to(ROOT) / relative)
    print("Unfrozen snapshot ready for no-provider tests.")


def seal():
    if (OUT / "FREEZE.json").exists():
        raise ValueError("Already frozen")
    snapshot()
    old = load(ROOT / "experiments/swan_path0_completion1/FREEZE.json")
    assets = {p: h for p, h in old["dependencies"].items()
              if p.startswith(("models/", "third_party/", "vectors/", ".nltk_data/"))}
    dependencies = {
        str(p.relative_to(SNAPSHOT)): sha256(p) for p in SNAPSHOT.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    dependencies.update(assets)
    freeze(OUT / "FREEZE.json", {
        "at": now(), "experiment": OUT.name, "user_instruction": "都补一补吧",
        "dependencies": dependencies, "original_core_dependencies": old["original_core_dependencies"],
        "positive_sources": 250, "calibration_sources": 1000, "null_sources": 1000,
        "scoring_rows": 8500, "fixed_arm": "control", "alpha": .01,
        "source_manifest_sha256": sha256(OUT / "INPUT_MANIFEST.json"),
        "generation_max_calls": run.CALL_CAP, "generation_max_nanodollars": run.COST_CAP,
        "generation_worst_reserve_nanodollars": run.CALL_CAP * run.BUDGET.MAX_RESERVE,
        "fresh_cache_required": True, "detection_after_all_generation": True,
        "human_review_external": True, "budget_pause_exit": 76, "resource_pause_exit": 75,
        "old_evidence_preserved": True,
    })
    previous.copy(OUT / "FREEZE.json", SNAPSHOT / OUT.relative_to(ROOT) / "FREEZE.json")
    archive = Path("/tmp/swan-completion-w04-20261102.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(SNAPSHOT.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                tar.add(path, arcname=str(path.relative_to(SNAPSHOT)), recursive=False)
    print({"freeze_sha256": sha256(OUT / "FREEZE.json"), "archive": str(archive),
           "archive_sha256": sha256(archive), "archive_bytes": archive.stat().st_size})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("snapshot", "seal"))
    args = parser.parse_args()
    (snapshot if args.action == "snapshot" else seal)()
