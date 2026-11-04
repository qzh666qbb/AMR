"""Independently audit corrected SWAN positive-collection provenance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/swan_positive_collection1"
OFFICIAL_ROOT = ROOT / "third_party/SWAN_official_2ca0c207_clean"
LEGACY_ROOT = ROOT / "third_party/SWAN"
EXPECTED_COMMIT = "2ca0c2071c8d0c6bb3739b2b2bcfb34cfe63eb2a"
EXPECTED_TREE = "2195cce5bb83eb2c965c8753f00e899324c8d0eb"
EXPECTED_ORIGIN = "https://github.com/amazon-science/SWAN.git"
EXPECTED_MISMATCHES = {
    "detection/parse_machine_text.py",
    "injection/injection_amr_utils.py",
    "injection/watermark_generation.py",
    "utils/amr_utils.py",
    "utils/bedrock_utils.py",
}
REMOTE_ZERO_KEYS = (
    "paid_api_calls",
    "model_probe_calls",
    "generation_calls",
    "parser_calls",
    "s2match_calls",
    "detector_calls",
    "attack_calls",
)
AUDIT_PATH = OUTPUT / "audit_v2.json"
REPORT_PATH = OUTPUT / "report_v2.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_json(name: str):
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(OFFICIAL_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def current_packages(saved_requirements: dict) -> dict:
    states = {}
    for name, saved in saved_requirements.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        states[name] = {
            "required": saved["required"],
            "installed": installed,
            "exact_match": installed == saved["required"],
        }
    return states


def all_true(value) -> bool:
    if isinstance(value, dict):
        return all(all_true(item) for item in value.values())
    if isinstance(value, list):
        return all(all_true(item) for item in value)
    return value is True


def verify_official_checkout(correction: dict) -> dict:
    frozen = correction["official_checkout"]
    paths = git(
        "ls-tree",
        "-r",
        "--name-only",
        EXPECTED_COMMIT,
    ).splitlines()
    working_paths = git("ls-files").splitlines()
    file_checks = {}
    for relative in paths:
        record = frozen["tracked_files"].get(relative, {})
        path = OFFICIAL_ROOT / relative
        file_checks[relative] = (
            path.is_file()
            and record.get("git_blob_oid")
            == git("rev-parse", f"{EXPECTED_COMMIT}:{relative}")
            and record.get("sha256") == sha256(path)
            and record.get("bytes") == path.stat().st_size
        )
    return {
        "head_matches": git("rev-parse", "HEAD") == EXPECTED_COMMIT,
        "tree_matches": (
            git("rev-parse", f"{EXPECTED_COMMIT}^{{tree}}")
            == EXPECTED_TREE
        ),
        "origin_matches": (
            git("config", "--get", "remote.origin.url")
            == EXPECTED_ORIGIN
        ),
        "checkout_clean": not git(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ),
        "exact_36_path_inventory": (
            len(paths) == 36
            and paths == working_paths
            and set(paths) == set(frozen["tracked_files"])
        ),
        "all_blob_sha256_and_sizes_match": all(file_checks.values()),
    }


def verify_legacy_comparison(correction: dict) -> dict:
    official_files = correction["official_checkout"]["tracked_files"]
    observed_mismatches = set()
    all_legacy_paths_present = True
    all_records_match = True
    frozen_rows = {
        row["path"]: row
        for row in correction["legacy_vendor_comparison"]["mismatches"]
    }
    for relative, official in official_files.items():
        legacy_path = LEGACY_ROOT / relative
        if not legacy_path.is_file():
            all_legacy_paths_present = False
            continue
        legacy_hash = sha256(legacy_path)
        if legacy_hash != official["sha256"]:
            observed_mismatches.add(relative)
            row = frozen_rows.get(relative, {})
            all_records_match &= (
                row.get("official_git_blob_oid")
                == official["git_blob_oid"]
                and row.get("official_sha256") == official["sha256"]
                and row.get("legacy_sha256") == legacy_hash
            )
    text_official = sha256(OFFICIAL_ROOT / "utils/text_utils.py")
    text_legacy = sha256(LEGACY_ROOT / "utils/text_utils.py")
    return {
        "all_official_paths_present_in_legacy": all_legacy_paths_present,
        "exact_five_mismatch_set": (
            observed_mismatches == EXPECTED_MISMATCHES
        ),
        "mismatch_records_match": all_records_match,
        "text_utils_byte_identical": text_official == text_legacy,
        "frozen_counts_match": (
            correction["legacy_vendor_comparison"][
                "mismatched_file_count"
            ]
            == 5
            and correction["legacy_vendor_comparison"][
                "matched_file_count"
            ]
            == 31
            and correction["legacy_vendor_comparison"][
                "missing_file_count"
            ]
            == 0
        ),
    }


def verify_supersession(correction: dict) -> dict:
    legacy = correction["supersession"]["legacy_artifact_sha256"]
    sources = correction["supersession"]["source_artifact_sha256"]
    return {
        "all_v1_artifacts_preserved": all(
            (OUTPUT / name).is_file()
            and sha256(OUTPUT / name) == expected
            for name, expected in legacy.items()
        ),
        "all_source_artifacts_preserved": all(
            (OUTPUT / name).is_file()
            and sha256(OUTPUT / name) == expected
            for name, expected in sources.items()
        ),
        "supersession_scope_is_execution_only": (
            correction["supersession"]["scope"]
            == "execution_provenance_only"
        ),
        "v1_execution_claims_invalidated": (
            correction["supersession"][
                "runtime_v1_valid_for_execution"
            ]
            is False
            and correction["supersession"][
                "pre_execution_manifest_v1_valid_for_execution"
            ]
            is False
            and correction["supersession"][
                "audit_v1_runtime_provenance_conclusion_valid"
            ]
            is False
        ),
        "source_allocation_remains_valid": correction["supersession"][
            "source_allocation_v1_valid"
        ],
    }


def verify_runtime(correction: dict, runtime: dict) -> dict:
    source = load_json("source_manifest.json")
    old_runtime_hash = sha256(OUTPUT / "runtime.json")
    dependency_hashes_match = True
    for group in runtime["dependencies"].values():
        for relative, expected in group.items():
            if isinstance(expected, dict):
                expected = expected["sha256"]
            dependency_hashes_match &= (
                (ROOT / relative).is_file()
                and sha256(ROOT / relative) == expected
            )
    saved_packages = runtime["python_environment"]["requirements"]
    observed_packages = current_packages(saved_packages)
    mismatch_names = [
        name
        for name, state in observed_packages.items()
        if not state["exact_match"]
    ]
    gaps = runtime["protocol_implementation_gaps"]
    gap_ids = {gap["id"] for gap in gaps if gap["resolved"] is False}
    expected_gaps = {
        "template_rng_unseeded",
        "per_attempt_trace_unavailable",
        "acceptance_state_not_persisted",
        "claude_4_5_top_p_omitted",
    }
    return {
        "correction_hash_matches": (
            runtime["provenance_correction_sha256"]
            == sha256(OUTPUT / "provenance_correction.json")
        ),
        "official_inventory_matches_correction": (
            runtime["official_swan"] == correction["official_checkout"]
        ),
        "source_manifest_hash_matches": (
            runtime["source_lineage"]["source_manifest_sha256"]
            == sha256(OUTPUT / "source_manifest.json")
        ),
        "source_manifest_v1_link_is_intact": (
            source["runtime_sha256"] == old_runtime_hash
            and runtime["source_lineage"]["source_manifest_runtime_v1_sha256"]
            == old_runtime_hash
            and runtime["source_lineage"]["runtime_v1_link_matches"]
        ),
        "source_allocation_count_and_cohorts_match": (
            len(source["sources"]) == 35
            and source["cohort_counts"] == {"pilot": 5, "main": 30}
            and runtime["source_lineage"]["source_count"] == 35
            and runtime["source_lineage"]["cohorts"]
            == {"pilot": 5, "main": 30}
        ),
        "all_dependency_hashes_match": dependency_hashes_match,
        "package_state_matches": (
            observed_packages == saved_packages
            and mismatch_names
            == runtime["python_environment"]["mismatches"]
        ),
        "dependency_state_is_not_misreported_ready": (
            runtime["python_environment"]["all_exact"] is False
            and runtime["python_environment"]["mismatch_count"] > 0
            and runtime["execution_ready"] is False
        ),
        "four_implementation_gaps_recorded": gap_ids == expected_gaps,
        "effective_top_p_is_honestly_omitted": (
            runtime["generation_protocol"]["top_p"] == 0.9
            and runtime["effective_official_behavior"]["remote_top_p"]
            is None
        ),
        "template_rng_is_honestly_unseeded": (
            runtime["effective_official_behavior"]["template_rng_seed"]
            is None
        ),
        "execution_locked": (
            runtime["execution_authorized"] is False
            and runtime["remote_model"]["model_probe_authorized"] is False
            and runtime["remote_model"]["generation_authorized"] is False
        ),
    }


def verify_pre_execution(runtime: dict, pre_execution: dict) -> dict:
    expected_hashes = {
        "protocol_sha256": runtime["protocol_sha256"],
        "provenance_correction_sha256": sha256(
            OUTPUT / "provenance_correction.json"
        ),
        "runtime_v2_sha256": sha256(OUTPUT / "runtime_v2.json"),
        "source_manifest_sha256": sha256(
            OUTPUT / "source_manifest.json"
        ),
        "prior_identity_snapshot_sha256": sha256(
            OUTPUT / "prior_identity_snapshot.json"
        ),
        "allocation_journal_sha256": sha256(
            OUTPUT / "allocation_journal.json"
        ),
    }
    forbidden = (
        "model_probe.json",
        "quality_packet.json",
        "quality_labels.json",
        "original_detection.json",
        "calls",
        "generation_traces",
        "baselines",
    )
    counts = pre_execution["resource_counts"]
    return {
        "all_hashes_match": pre_execution["hashes"] == expected_hashes,
        "source_preparation_complete": (
            pre_execution["completed"]["all_35_sources_frozen"]
            and pre_execution["completed"][
                "official_36_file_blob_and_sha256_inventory_frozen"
            ]
        ),
        "execution_not_ready_or_authorized": (
            pre_execution["execution_ready"] is False
            and pre_execution["execution_authorized"] is False
        ),
        "all_remote_and_compute_counts_zero": all(
            counts[key] == 0 for key in REMOTE_ZERO_KEYS
        ),
        "only_one_prior_public_download": (
            counts["public_dataset_http_requests"] == 1
        ),
        "no_downstream_artifacts": not any(
            (OUTPUT / name).exists() for name in forbidden
        ),
    }


def build_audit(created_at: Optional[str] = None) -> dict:
    correction = load_json("provenance_correction.json")
    runtime = load_json("runtime_v2.json")
    pre_execution = load_json("pre_execution_manifest_v2.json")
    checks = {
        "official_checkout": verify_official_checkout(correction),
        "legacy_vendor_comparison": verify_legacy_comparison(correction),
        "supersession": verify_supersession(correction),
        "runtime_v2": verify_runtime(correction, runtime),
        "pre_execution_v2": verify_pre_execution(runtime, pre_execution),
    }
    passed = all_true(checks)
    source = load_json("source_manifest.json")
    return {
        "schema_version": 2,
        "created_at": created_at or utc_now(),
        "experiment": "swan_positive_collection1",
        "status": (
            "verified_source_frozen_execution_corrected_not_ready"
            if passed
            else "audit_failed"
        ),
        "decision": (
            "block_execution_pending_protocol_dependency_and_budget_gates"
            if passed
            else "do_not_execute_audit_failed"
        ),
        "checks": checks,
        "all_checks_pass": passed,
        "official_provenance": {
            "commit": runtime["official_swan"]["commit"],
            "tree_oid": runtime["official_swan"]["tree_oid"],
            "tracked_files": runtime["official_swan"][
                "tracked_file_count"
            ],
            "legacy_mismatches": correction[
                "legacy_vendor_comparison"
            ]["mismatched_file_count"],
        },
        "allocation": {
            "sources": len(source["sources"]),
            "pilot": source["cohort_counts"]["pilot"],
            "main": source["cohort_counts"]["main"],
            "source_manifest_preserved": True,
        },
        "readiness": {
            "execution_ready": runtime["execution_ready"],
            "package_mismatches": runtime["python_environment"][
                "mismatches"
            ],
            "implementation_gaps": [
                gap["id"]
                for gap in runtime["protocol_implementation_gaps"]
                if not gap["resolved"]
            ],
            "execution_blocked_by": runtime["execution_blocked_by"],
        },
        "resource_counts": pre_execution["resource_counts"],
        "claim_boundary": (
            "This audit verifies corrected official-code provenance and "
            "preservation of the frozen 35-source allocation. It reports no "
            "watermarked document, detector result, attack result, or ASR."
        ),
    }


def report_text(audit: dict) -> str:
    packages = ", ".join(audit["readiness"]["package_mismatches"])
    gaps = "\n".join(
        f"- `{gap}`" for gap in audit["readiness"]["implementation_gaps"]
    )
    return f"""# SWAN positive collection provenance correction

## Status

The v2 offline audit passed. The 35-source allocation remains frozen and
valid, but execution is neither ready nor authorized.

## Corrected provenance

- Official commit: `{audit["official_provenance"]["commit"]}`
- Official tree: `{audit["official_provenance"]["tree_oid"]}`
- Official tracked files: {audit["official_provenance"]["tracked_files"]}
- Legacy vendor mismatches: {audit["official_provenance"]["legacy_mismatches"]}
- Frozen allocation: {audit["allocation"]["pilot"]} pilot + {audit["allocation"]["main"]} main

The old `runtime.json`, `pre_execution_manifest.json`, and `audit.json` are
preserved but superseded for execution provenance. They hashed a local vendor
with five files that differ from the official commit. Source allocation is
unchanged because `utils/text_utils.py`, the only upstream code used for
prompt allocation, is byte-identical.

## Readiness blockers

Official requirement mismatches: {packages}.

The official path also has these unresolved protocol/implementation gaps:

{gaps}

The clean official checkout has no installed S2Match runtime, the parser model
is not wired to amrlib's default `model_stog` path, and frozen GloVe/NLTK
environment bindings are absent. Installing one missing package would not make
the protocol execution-ready.

## Resource boundary

The prior public C4 retrieval count is 1. Paid API, model probe, generation,
parser, S2Match, detector, and attack counts remain 0. No downstream result is
reported. Execution stays blocked pending a frozen protocol correction,
dependency completion, explicit paid-budget authorization, and a returned
model identity probe.
"""


def write_frozen(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"Frozen audit artifact differs: {path}")
    path.write_text(text, encoding="utf-8")


def create() -> dict:
    if AUDIT_PATH.exists() or REPORT_PATH.exists():
        if not AUDIT_PATH.is_file() or not REPORT_PATH.is_file():
            raise RuntimeError("Partial v2 audit output requires review")
        return check()
    audit = build_audit()
    if not audit["all_checks_pass"]:
        raise ValueError("Corrected provenance audit failed")
    write_frozen(
        AUDIT_PATH,
        json.dumps(audit, indent=2, ensure_ascii=True) + "\n",
    )
    write_frozen(REPORT_PATH, report_text(audit))
    return audit


def check() -> dict:
    saved = load_json("audit_v2.json")
    live = build_audit(created_at=saved["created_at"])
    if saved != live:
        raise ValueError("Frozen v2 audit differs from live evidence")
    if REPORT_PATH.read_text(encoding="utf-8") != report_text(saved):
        raise ValueError("Frozen v2 report differs from audit")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("create", "check"))
    args = parser.parse_args()
    result = create() if args.stage == "create" else check()
    print(
        json.dumps(
            {
                "status": result["status"],
                "all_checks_pass": result["all_checks_pass"],
                "official_provenance": result["official_provenance"],
                "allocation": result["allocation"],
                "execution_ready": result["readiness"]["execution_ready"],
                "resource_counts": result["resource_counts"],
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
