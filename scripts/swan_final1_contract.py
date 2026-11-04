"""Freeze/check final1 code and local assets without importing inference libraries."""
from __future__ import annotations

import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys

from swan_final1_budget import Journal, budget_estimate
from swan_final1_common import ROOT, OUT, assert_hashes, freeze, load, now, sha256
from swan_final1_quality import blank_review_row
from swan_final1_sources import audit_identities, check_sources

BANK = ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_50.json"
EXPECTED_BANK = "f0a88c1af73d12078586d10ef68804c12ed5b79cabc07c9f6c2a34bb084ef5a4"
CONTROL_FILES = ("README.md", "protocol.md", "analysis_plan.md", "quality_protocol.md",
                 "STOP_RULES.md", "claims.json", "pricing_snapshot.md")
EXTRA_CODE = ("scripts/run_swan_positive_collection.py", "scripts/swan_reframing_statistics.py")


def git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def inventory(paths):
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha256(p)
            for p in sorted(set(paths)) if p.is_file()}


def tree_files(path):
    return [p for p in path.rglob("*") if p.is_file()
            and "__pycache__" not in p.parts and p.suffix != ".pyc"]


def environment_lock():
    old = load(ROOT / "experiments/swan_positive_collection1/execution_contract_v3.json")
    resources = []
    for group in old["dependencies"]["resources"].values():
        for relative in group:
            path = ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(f"Required local asset missing: {relative}")
            resources.append(path)
    checkouts = {}
    for name, info in old["official_checkouts"].items():
        folder = ROOT / info["root"]
        head = git(folder, "rev-parse", "HEAD")
        if head != info["head"] or git(folder, "status", "--porcelain=v1", "--untracked-files=all"):
            raise ValueError(f"Official checkout not exact and clean: {name}")
        checkouts[name] = {"root": info["root"], "head": head,
                           "tree": git(folder, "rev-parse", "HEAD^{tree}")}
        resources.extend(folder / name for name in git(folder, "ls-files").splitlines())
    for relative in ("models/model_parse_xfm_bart_large-v0_1_0", "models/bart-large-tokenizer",
                     "third_party/swan_s2match_runtime_711a231d",
                     ".nltk_data/tokenizers/punkt_tab/english"):
        resources.extend(tree_files(ROOT / relative))
    # NLTK prefers extracted corpora when present; freeze both potential inputs.
    propbank = ROOT / ".nltk_data/corpora/propbank"
    if propbank.exists():
        resources.extend(tree_files(propbank))
    distributions, installed_files = {}, []
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"].lower().replace("_", "-")
        files = list(dist.files or [])
        records = [dist.locate_file(f) for f in files if str(f).endswith(".dist-info/RECORD")]
        if len(records) != 1:
            raise ValueError(f"Distribution has no unique RECORD: {name}")
        if name in distributions:
            raise ValueError(f"Duplicate installed distribution: {name}")
        distributions[name] = {"version": dist.version, "record_sha256": sha256(records[0]),
                                "record_path": str(records[0])}
        # Detect modified installed code/data, not merely unchanged RECORD manifests.
        for item in files:
            path = dist.locate_file(item)
            if "__pycache__" not in path.parts and path.suffix != ".pyc":
                if not path.is_file():
                    raise FileNotFoundError(f"Installed distribution file missing: {name}/{item}")
                installed_files.append(path.resolve())
    if distributions["en-core-web-sm"]["version"] != "3.8.0":
        raise ValueError("Unexpected frozen syntax model")
    for name, info in old["python_environment"]["official_requirements"].items():
        if distributions[name]["version"] != info["required"]:
            raise ValueError(f"Official dependency version changed: {name}")
    resources.extend((Path(sys.executable), Path(sys.prefix) / "pyvenv.cfg"))
    return {"python_executable": sys.executable, "prefix": sys.prefix,
            "python_version": platform.python_version(), "platform": platform.platform(),
            "official_checkouts": checkouts, "distributions": distributions,
            "resources": inventory(resources), "installed_files": inventory(installed_files),
            "inference_libraries_imported": False}


def code_paths():
    return sorted(set([*ROOT.glob("scripts/swan_final1_*.py"),
                       *ROOT.glob("scripts/test_swan_final1_*.py"),
                       ROOT / "scripts/run_swan_count_neutral_final1.py",
                       *[ROOT / p for p in EXTRA_CODE]]))


def prepare(folder=OUT):
    folder = Path(folder)
    if (folder / "manifest.json").exists():
        return check(folder)
    audit_identities(folder)
    if not BANK.is_file() or sha256(BANK) != EXPECTED_BANK:
        raise ValueError("Bank50 location/hash differs")
    for path in code_paths():
        if not path.is_file():
            raise FileNotFoundError(path)
    freeze(folder / "budget_estimate.json", budget_estimate())
    freeze(folder / "quality_schema.json", {
        "baseline": blank_review_row("filled_only_after_generation", "baseline"),
        "pair": blank_review_row("filled_only_after_transformation", "pair"),
        "reviewer_ids": [None, None], "completed_at": None, "independent_review": None,
    })
    freeze(folder / "environment_lock.json", environment_lock())
    # Main pipeline must be implemented and tested before this contract can freeze.
    for required in ("swan_final1_analysis.py", "swan_final1_detection.py"):
        if not (ROOT / "scripts" / required).is_file():
            raise ValueError(f"Final analysis pipeline not ready: {required}")
    artifacts = [folder / p for p in (*CONTROL_FILES, "source_manifest.json",
                                      "prior_identity_snapshot.json", "budget_estimate.json",
                                      "quality_schema.json", "environment_lock.json")]
    manifest = freeze(folder / "manifest.json", {
        "schema_version": 1, "experiment": "swan_count_neutral_final1", "frozen_at": now(),
        "status": "frozen_preexecution", "paid_execution_authorized": False,
        "code_hashes": inventory(code_paths()), "artifact_hashes": inventory(artifacts),
        "bank_path": str(BANK.relative_to(ROOT)), "bank_sha256": EXPECTED_BANK,
        "provider_revision": None, "provider_revision_limit": "No immutable vendor revision available",
        "real_model_parser_scorer_detector_observations_at_freeze": 0,
    })
    for phase in ("pilot", "main"):
        budget = budget_estimate()["phases"][phase]
        freeze(folder / f"authorization_{phase}_template.json", {
            "phase": phase, "manifest_sha256": sha256(folder / "manifest.json"),
            "budget_sha256": sha256(folder / "budget_estimate.json"),
            "max_calls": budget["max_calls"], "max_cost_usd": budget["max_cost_usd"],
            "authorized_by": None, "authorization_text": None, "authorized_at": None,
        })
    return {"state": manifest["status"], "manifest_sha256": sha256(folder / "manifest.json")}


def check(folder=OUT, full_environment=True):
    folder = Path(folder)
    manifest = load(folder / "manifest.json")
    assert_hashes(manifest["code_hashes"])
    assert_hashes(manifest["artifact_hashes"])
    if set(manifest["code_hashes"]) != set(inventory(code_paths())):
        raise ValueError("Final1 code inventory changed")
    check_sources(folder)
    lock = load(folder / "environment_lock.json")
    if lock["prefix"] != sys.prefix or lock["python_version"] != platform.python_version():
        raise ValueError("Use the frozen .venv-swan-positive Python")
    if full_environment:
        if environment_lock() != lock:
            raise ValueError("Execution environment or assets changed")
    if sha256(BANK) != EXPECTED_BANK:
        raise ValueError("Bank50 changed")
    accounting = {}
    for phase in ("pilot", "main"):
        if (folder / "private" / phase / "journal.jsonl").exists():
            accounting[phase] = Journal(
                folder, phase, 1, None, sha256(folder / "manifest.json"),
                allow_halted=True).totals()
    return {"state": "frozen_hashes_verified", "manifest_sha256": sha256(folder / "manifest.json"),
            "environment_verified": full_environment, "accounting": accounting,
            "paid_calls_recorded": sum(r["calls"] for r in accounting.values())}
