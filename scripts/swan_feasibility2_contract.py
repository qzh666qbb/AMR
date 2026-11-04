"""Freeze and verify the independent feasibility2 experiment contract."""
from __future__ import annotations

from pathlib import Path
import platform
import sys

from swan_feasibility2_budget import Journal, budget_estimate
from swan_feasibility2_common import ROOT, OUT, assert_hashes, freeze, load, now, sha256
from swan_feasibility2_sources import (
    HUMAN_RANGE,
    PILOT_COUNT,
    POSITIVE_RANGE,
    allocate,
    check_sources,
)
from swan_final1_contract import environment_lock
from swan_final1_quality import blank_review_row

BANK = ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_50.json"
EXPECTED_BANK = "f0a88c1af73d12078586d10ef68804c12ed5b79cabc07c9f6c2a34bb084ef5a4"
CONTROL_FILES = (
    "README.md",
    "protocol.md",
    "analysis_plan.md",
    "quality_protocol.md",
    "STOP_RULES.md",
)
CODE_FILES = (
    "scripts/run_swan_normal_length_feasibility2.py",
    "scripts/swan_feasibility2_budget.py",
    "scripts/swan_feasibility2_common.py",
    "scripts/swan_feasibility2_contract.py",
    "scripts/swan_feasibility2_generation.py",
    "scripts/swan_feasibility2_sources.py",
    "scripts/swan_feasibility2_workflow.py",
    "scripts/test_swan_feasibility2.py",
    "scripts/run_swan_positive_collection.py",
    "scripts/swan_anchor1_sources.py",
    "scripts/swan_final1_budget.py",
    "scripts/swan_final1_common.py",
    "scripts/swan_final1_contract.py",
    "scripts/swan_final1_quality.py",
    "scripts/swan_final1_runtime.py",
    "scripts/swan_final1_transform.py",
    "scripts/swan_final1_workflow.py",
)


def inventory(paths):
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for path in sorted(set(paths))
        if path.is_file()
    }


def code_paths():
    return [ROOT / relative for relative in CODE_FILES]


def prepare(folder=OUT):
    folder = Path(folder)
    if (folder / "manifest.json").exists():
        return check(folder)
    allocate(folder)
    check_sources(folder)
    if not BANK.is_file() or sha256(BANK) != EXPECTED_BANK:
        raise ValueError("Bank50 location or hash changed")
    missing = [str(path) for path in code_paths() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required code missing: {missing}")
    freeze(folder / "budget_estimate.json", budget_estimate())
    freeze(folder / "quality_schema.json", {
        "kind": "baseline",
        "review_A": blank_review_row("filled_after_generation", "baseline"),
        "review_B": blank_review_row("filled_after_generation", "baseline"),
        "reviewers_must_be_distinct_humans": True,
        "detection_before_quality_freeze": False,
    })
    freeze(folder / "environment_lock.json", environment_lock())
    artifacts = [
        folder / relative for relative in (
            *CONTROL_FILES,
            "source_manifest.json",
            "prior_identity_snapshot.json",
            "budget_estimate.json",
            "quality_schema.json",
            "environment_lock.json",
        )
    ]
    manifest = freeze(folder / "manifest.json", {
        "schema_version": 1,
        "experiment": "swan_normal_length_feasibility2",
        "frozen_at": now(),
        "status": "frozen_preexecution",
        "paid_execution_authorized": False,
        "code_hashes": inventory(code_paths()),
        "artifact_hashes": inventory(artifacts),
        "bank_path": str(BANK.relative_to(ROOT)),
        "bank_sha256": EXPECTED_BANK,
        "pilot_count": PILOT_COUNT,
        "positive_source_range": list(POSITIVE_RANGE),
        "human_source_range": list(HUMAN_RANGE),
        "prior_failed_cohort_modified": False,
        "final1_sealed_content_access": False,
        "credential_file_access": False,
        "real_model_parser_scorer_detector_observations_at_freeze": 0,
    })
    budget = budget_estimate()["pilot"]
    freeze(folder / "authorization_pilot_template.json", {
        "phase": "pilot",
        "manifest_sha256": sha256(folder / "manifest.json"),
        "budget_sha256": sha256(folder / "budget_estimate.json"),
        "max_calls": budget["max_calls"],
        "max_cost_usd": budget["max_cost_usd"],
        "authorized_by": None,
        "authorization_text": None,
        "key_rotation_attested": None,
        "authorized_at": None,
    })
    return {
        "state": manifest["status"],
        "manifest_sha256": sha256(folder / "manifest.json"),
    }


def check(folder=OUT, full_environment=True):
    folder = Path(folder)
    manifest = load(folder / "manifest.json")
    if manifest["experiment"] != "swan_normal_length_feasibility2" \
            or manifest["prior_failed_cohort_modified"] \
            or manifest["final1_sealed_content_access"] \
            or manifest["credential_file_access"]:
        raise ValueError("Experiment identity or isolation contract changed")
    assert_hashes(manifest["code_hashes"])
    assert_hashes(manifest["artifact_hashes"])
    if set(manifest["code_hashes"]) != set(inventory(code_paths())):
        raise ValueError("Feasibility2 code inventory changed")
    check_sources(folder)
    lock = load(folder / "environment_lock.json")
    if lock["prefix"] != sys.prefix or lock["python_version"] != platform.python_version():
        raise ValueError("Use the frozen .venv-swan-positive Python")
    if full_environment and environment_lock() != lock:
        raise ValueError("Execution environment or assets changed")
    if sha256(BANK) != EXPECTED_BANK:
        raise ValueError("Bank50 changed")
    accounting = None
    if (folder / "private/pilot/journal.jsonl").exists():
        accounting = Journal(
            folder, 1, None, sha256(folder / "manifest.json"), allow_halted=True
        ).totals()
    return {
        "state": "frozen_hashes_verified",
        "manifest_sha256": sha256(folder / "manifest.json"),
        "environment_verified": full_environment,
        "accounting": accounting,
        "paid_calls_recorded": accounting["calls"] if accounting else 0,
    }
