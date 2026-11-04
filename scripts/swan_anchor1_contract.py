"""Freeze and verify the source-anchored experiment contract."""
from __future__ import annotations

from pathlib import Path
import platform
import sys

from swan_anchor1_budget import Journal, budget_estimate
from swan_anchor1_common import ROOT, OUT, assert_hashes, freeze, load, now, sha256
from swan_anchor1_sources import allocate, check_sources
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
    "claims.json",
    "pricing_snapshot.md",
)
REUSED_CODE = (
    "scripts/run_swan_positive_collection.py",
    "scripts/swan_reframing_statistics.py",
    "scripts/swan_final1_common.py",
    "scripts/swan_final1_budget.py",
    "scripts/swan_final1_runtime.py",
    "scripts/swan_final1_transform.py",
    "scripts/swan_final1_quality.py",
    "scripts/swan_final1_workflow.py",
)


def inventory(paths):
    return {
        str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): sha256(path)
        for path in sorted(set(paths))
        if path.is_file()
    }


def code_paths():
    paths = [
        *ROOT.glob("scripts/swan_anchor1_*.py"),
        *ROOT.glob("scripts/swan_final1_*.py"),
        ROOT / "scripts/run_swan_source_anchored1.py",
        ROOT / "scripts/test_swan_anchor1.py",
        *(ROOT / relative for relative in REUSED_CODE),
    ]
    return sorted(set(paths))


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
        "baseline": blank_review_row("filled_only_after_generation", "baseline"),
        "pair": blank_review_row("filled_only_after_transformation", "pair"),
        "reviewer_ids": [None, None],
        "completed_at": None,
        "independent_review": None,
    })
    freeze(folder / "environment_lock.json", environment_lock())
    artifacts = [
        folder / relative
        for relative in (
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
        "experiment": "swan_source_anchored1",
        "frozen_at": now(),
        "status": "frozen_preexecution",
        "paid_execution_authorized": False,
        "code_hashes": inventory(code_paths()),
        "artifact_hashes": inventory(artifacts),
        "bank_path": str(BANK.relative_to(ROOT)),
        "bank_sha256": EXPECTED_BANK,
        "provider_revision": None,
        "provider_revision_limit": "No immutable vendor revision available",
        "positive_source_range": ["validation", 0, 13863],
        "human_source_range": ["train", 1700000, 1809584],
        "final1_source_shards": [13, 14, 15, 16],
        "final1_sealed_content_access": False,
        "credential_file_access": False,
        "real_model_parser_scorer_detector_observations_at_freeze": 0,
    })
    for phase in ("pilot", "main"):
        budget = budget_estimate()["phases"][phase]
        freeze(folder / f"authorization_{phase}_template.json", {
            "phase": phase,
            "manifest_sha256": sha256(folder / "manifest.json"),
            "budget_sha256": sha256(folder / "budget_estimate.json"),
            "max_calls": budget["max_calls"],
            "max_cost_usd": budget["max_cost_usd"],
            "authorized_by": None,
            "authorization_text": None,
            "authorized_at": None,
        })
    return {
        "state": manifest["status"],
        "manifest_sha256": sha256(folder / "manifest.json"),
    }


def check(folder=OUT, full_environment=True):
    folder = Path(folder)
    manifest = load(folder / "manifest.json")
    if manifest["experiment"] != "swan_source_anchored1" \
            or manifest["final1_sealed_content_access"] \
            or manifest["credential_file_access"]:
        raise ValueError("Experiment identity or isolation contract changed")
    assert_hashes(manifest["code_hashes"])
    assert_hashes(manifest["artifact_hashes"])
    if set(manifest["code_hashes"]) != set(inventory(code_paths())):
        raise ValueError("Anchor1 code inventory changed")
    check_sources(folder)
    lock = load(folder / "environment_lock.json")
    if lock["prefix"] != sys.prefix or lock["python_version"] != platform.python_version():
        raise ValueError("Use the frozen .venv-swan-positive Python")
    if full_environment and environment_lock() != lock:
        raise ValueError("Execution environment or assets changed")
    if sha256(BANK) != EXPECTED_BANK:
        raise ValueError("Bank50 changed")
    accounting = {}
    for phase in ("pilot", "main"):
        if (folder / "private" / phase / "journal.jsonl").exists():
            accounting[phase] = Journal(
                folder,
                phase,
                1,
                None,
                sha256(folder / "manifest.json"),
                allow_halted=True,
            ).totals()
    return {
        "state": "frozen_hashes_verified",
        "manifest_sha256": sha256(folder / "manifest.json"),
        "environment_verified": full_environment,
        "accounting": accounting,
        "paid_calls_recorded": sum(row["calls"] for row in accounting.values()),
    }
