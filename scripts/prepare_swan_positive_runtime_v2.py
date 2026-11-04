"""Correct and freeze SWAN positive-collection execution provenance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/swan_positive_collection1"
OFFICIAL_ROOT = ROOT / "third_party/SWAN_official_2ca0c207_clean"
LEGACY_ROOT = ROOT / "third_party/SWAN"
EXPECTED_COMMIT = "2ca0c2071c8d0c6bb3739b2b2bcfb34cfe63eb2a"
EXPECTED_TREE = "2195cce5bb83eb2c965c8753f00e899324c8d0eb"
EXPECTED_ORIGIN = "https://github.com/amazon-science/SWAN.git"
EXPECTED_TRACKED_FILES = 36
EXPECTED_MISMATCHES = {
    "detection/parse_machine_text.py",
    "injection/injection_amr_utils.py",
    "injection/watermark_generation.py",
    "utils/amr_utils.py",
    "utils/bedrock_utils.py",
}
TEXT_UTILS_PATH = "utils/text_utils.py"

CORRECTION_PATH = OUTPUT / "provenance_correction.json"
RUNTIME_V2_PATH = OUTPUT / "runtime_v2.json"
PRE_EXECUTION_V2_PATH = OUTPUT / "pre_execution_manifest_v2.json"

LEGACY_ARTIFACTS = (
    "runtime.json",
    "pre_execution_manifest.json",
    "audit.json",
    "report.md",
)
SOURCE_ARTIFACTS = (
    "prior_identity_snapshot.json",
    "source_manifest.json",
    "allocation_journal.json",
)
PROVENANCE_CODE = (
    "scripts/prepare_swan_positive_collection.py",
    "scripts/audit_swan_positive_collection.py",
    "scripts/test_swan_positive_collection.py",
    "scripts/prepare_swan_positive_runtime_v2.py",
    "scripts/audit_swan_positive_runtime_v2.py",
    "scripts/test_swan_positive_runtime_v2.py",
)
RESOURCE_GROUPS = {
    "amr_parser_weights": (
        "models/model_parse_xfm_bart_large-v0_1_0/amrlib_meta.json",
        "models/model_parse_xfm_bart_large-v0_1_0/config.json",
        (
            "models/model_parse_xfm_bart_large-v0_1_0/"
            "model_parse_xfm_bart_large.json"
        ),
        (
            "models/model_parse_xfm_bart_large-v0_1_0/"
            "pytorch_model.bin"
        ),
    ),
    "amr_tokenizer": (
        "models/bart-large-tokenizer/config.json",
        "models/bart-large-tokenizer/merges.txt",
        "models/bart-large-tokenizer/tokenizer.json",
        "models/bart-large-tokenizer/tokenizer_config.json",
        "models/bart-large-tokenizer/vocab.json",
    ),
    "glove": ("vectors/glove.6B.100d.txt",),
    "nltk": (
        ".nltk_data/corpora/propbank.zip",
        ".nltk_data/tokenizers/punkt_tab/english/abbrev_types.txt",
        ".nltk_data/tokenizers/punkt_tab/english/collocations.tab",
        ".nltk_data/tokenizers/punkt_tab/english/ortho_context.tab",
        ".nltk_data/tokenizers/punkt_tab/english/sent_starters.txt",
    ),
}
ZERO_RESOURCE_COUNTS = {
    "public_dataset_http_requests": 1,
    "paid_api_calls": 0,
    "model_probe_calls": 0,
    "generation_calls": 0,
    "parser_calls": 0,
    "s2match_calls": 0,
    "detector_calls": 0,
    "attack_calls": 0,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_json(path: Path, payload) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"Frozen artifact differs: {path}")
    path.write_text(text, encoding="utf-8")


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(OFFICIAL_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def official_checkout_evidence() -> dict:
    if not OFFICIAL_ROOT.is_dir():
        raise FileNotFoundError(f"Official checkout missing: {OFFICIAL_ROOT}")
    head = git("rev-parse", "HEAD")
    tree = git("rev-parse", f"{EXPECTED_COMMIT}^{{tree}}")
    origin = git("config", "--get", "remote.origin.url")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    tracked_paths = git(
        "ls-tree",
        "-r",
        "--name-only",
        EXPECTED_COMMIT,
    ).splitlines()
    worktree_paths = git("ls-files").splitlines()
    if head != EXPECTED_COMMIT:
        raise ValueError(f"Official checkout HEAD differs: {head}")
    if tree != EXPECTED_TREE:
        raise ValueError(f"Official tree differs: {tree}")
    if origin != EXPECTED_ORIGIN:
        raise ValueError(f"Official origin differs: {origin}")
    if status:
        raise ValueError("Official checkout is not clean")
    if (
        len(tracked_paths) != EXPECTED_TRACKED_FILES
        or tracked_paths != worktree_paths
    ):
        raise ValueError("Official checkout tracked-file inventory differs")

    files = {}
    for relative in tracked_paths:
        path = OFFICIAL_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Official tracked file missing: {relative}")
        files[relative] = {
            "git_blob_oid": git(
                "rev-parse",
                f"{EXPECTED_COMMIT}:{relative}",
            ),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    return {
        "repository": EXPECTED_ORIGIN.removesuffix(".git"),
        "origin": origin,
        "local_root": str(OFFICIAL_ROOT.relative_to(ROOT)),
        "commit": head,
        "tree_oid": tree,
        "clean": True,
        "tracked_file_count": len(files),
        "tracked_files": files,
    }


def compare_legacy_vendor(official: dict) -> dict:
    matched = []
    mismatches = []
    missing = []
    for relative, official_record in official["tracked_files"].items():
        legacy_path = LEGACY_ROOT / relative
        if not legacy_path.is_file():
            missing.append(relative)
            continue
        legacy_hash = sha256(legacy_path)
        if legacy_hash == official_record["sha256"]:
            matched.append(relative)
        else:
            mismatches.append(
                {
                    "path": relative,
                    "official_git_blob_oid": official_record[
                        "git_blob_oid"
                    ],
                    "official_sha256": official_record["sha256"],
                    "legacy_sha256": legacy_hash,
                }
            )
    mismatch_paths = {row["path"] for row in mismatches}
    if missing:
        raise ValueError(f"Legacy vendor is missing tracked files: {missing}")
    if mismatch_paths != EXPECTED_MISMATCHES:
        raise ValueError(
            "Unexpected official/legacy mismatch set: "
            f"{sorted(mismatch_paths)}"
        )
    if TEXT_UTILS_PATH not in matched:
        raise ValueError("Prompt extractor is not byte-identical")
    return {
        "legacy_root": str(LEGACY_ROOT.relative_to(ROOT)),
        "comparison_scope": (
            "The 36 paths tracked by the official commit; legacy-only "
            "auxiliary files are outside this comparison."
        ),
        "matched_file_count": len(matched),
        "mismatched_file_count": len(mismatches),
        "missing_file_count": len(missing),
        "matched_paths": matched,
        "mismatches": mismatches,
        "prompt_extractor": {
            "path": TEXT_UTILS_PATH,
            "byte_identical": True,
            "official_sha256": official["tracked_files"][
                TEXT_UTILS_PATH
            ]["sha256"],
            "legacy_sha256": sha256(LEGACY_ROOT / TEXT_UTILS_PATH),
        },
    }


def legacy_artifact_hashes() -> dict[str, str]:
    hashes = {}
    for name in (*LEGACY_ARTIFACTS, *SOURCE_ARTIFACTS):
        path = OUTPUT / name
        if not path.is_file():
            raise FileNotFoundError(f"Frozen v1 artifact missing: {path}")
        hashes[name] = sha256(path)
    return hashes


def requirement_state() -> dict:
    requirements_path = OFFICIAL_ROOT / "requirements.txt"
    requirements = {}
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "==" not in line:
            raise ValueError(f"Unpinned official requirement: {line}")
        name, required = line.split("==", 1)
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        requirements[name] = {
            "required": required,
            "installed": installed,
            "exact_match": installed == required,
        }
    mismatches = [
        name
        for name, state in requirements.items()
        if not state["exact_match"]
    ]
    return {
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "python_base_prefix": sys.base_prefix,
        "virtual_environment": sys.prefix != sys.base_prefix,
        "python_version": sys.version.split()[0],
        "requirements_sha256": sha256(requirements_path),
        "requirements": requirements,
        "exact_match_count": len(requirements) - len(mismatches),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "all_exact": not mismatches,
    }


def hash_resource_groups() -> dict:
    groups = {}
    for name, relatives in RESOURCE_GROUPS.items():
        records = {}
        for relative in relatives:
            path = ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(f"Required resource missing: {path}")
            records[relative] = {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        groups[name] = records
    return groups


def hash_provenance_code() -> dict:
    records = {}
    for relative in PROVENANCE_CODE:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Provenance code missing: {path}")
        records[relative] = sha256(path)
    return records


def manual_readiness() -> dict:
    try:
        amrlib_root = Path(
            importlib.util.find_spec("amrlib").submodule_search_locations[0]
        )
    except (AttributeError, TypeError):
        amrlib_root = None
    default_model = (
        amrlib_root / "data/model_stog" if amrlib_root is not None else None
    )
    s2match_dir = (
        OFFICIAL_ROOT
        / "amr-metric-suite/py3-Smatch-and-S2match/smatch"
    )
    spacy_model = importlib.util.find_spec("en_core_web_sm") is not None
    glove_env = os.environ.get("GLOVE_VECTORS_PATH")
    nltk_env = os.environ.get("NLTK_DATA")
    return {
        "spacy_model_en_core_web_sm_available": spacy_model,
        "official_checkout_s2match_runtime_available": (
            (s2match_dir / "s2match.py").is_file()
            and (s2match_dir / "amr_py3.py").is_file()
            and (s2match_dir / "helpers.py").is_file()
        ),
        "amrlib_default_model_stog_available": bool(
            default_model is not None and default_model.exists()
        ),
        "amrlib_default_model_stog_path": (
            str(default_model) if default_model is not None else None
        ),
        "glove_vectors_path_env": glove_env,
        "glove_vectors_path_env_resolves": bool(
            glove_env and Path(glove_env).expanduser().is_file()
        ),
        "nltk_data_env": nltk_env,
        "nltk_data_env_includes_frozen_root": bool(
            nltk_env
            and str(ROOT / ".nltk_data") in nltk_env.split(os.pathsep)
        ),
        "note": (
            "Availability is checked without importing SWAN, loading a model, "
            "running a parser, or invoking S2Match."
        ),
    }


def build_correction() -> dict:
    official = official_checkout_evidence()
    comparison = compare_legacy_vendor(official)
    legacy_hashes = legacy_artifact_hashes()
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "experiment": "swan_positive_collection1",
        "status": "verified_execution_provenance_correction",
        "official_checkout": official,
        "legacy_vendor_comparison": comparison,
        "source_allocation_impact": {
            "source_manifest_preserved": True,
            "allocation_order_preserved": True,
            "prompt_text_preserved": True,
            "basis": (
                "Source selection used only C4 row order, identity checks, "
                "and utils/text_utils.py; that extractor is byte-identical "
                "between the legacy vendor and official commit."
            ),
        },
        "supersession": {
            "scope": "execution_provenance_only",
            "legacy_artifact_sha256": {
                name: legacy_hashes[name] for name in LEGACY_ARTIFACTS
            },
            "source_artifact_sha256": {
                name: legacy_hashes[name] for name in SOURCE_ARTIFACTS
            },
            "legacy_artifacts_preserved": True,
            "runtime_v1_valid_for_execution": False,
            "pre_execution_manifest_v1_valid_for_execution": False,
            "audit_v1_runtime_provenance_conclusion_valid": False,
            "source_allocation_v1_valid": True,
            "replacement_runtime": "runtime_v2.json",
            "replacement_pre_execution_manifest": (
                "pre_execution_manifest_v2.json"
            ),
            "replacement_audit": "audit_v2.json",
        },
        "claim_boundary": (
            "This correction establishes exact official source provenance "
            "and preserves the already frozen source allocation. It does not "
            "authorize or report model, generation, parser, S2Match, "
            "detector, or attack execution."
        ),
    }


def build_runtime_v2(correction: dict) -> dict:
    old_runtime = load_json(OUTPUT / "runtime.json")
    source_manifest = load_json(OUTPUT / "source_manifest.json")
    packages = requirement_state()
    manual = manual_readiness()
    resources = hash_resource_groups()
    implementation_gaps = [
        {
            "id": "template_rng_unseeded",
            "resolved": False,
            "evidence": (
                "Official watermark_generation.py uses random.choice but "
                "defines no seed argument or random.seed call."
            ),
        },
        {
            "id": "per_attempt_trace_unavailable",
            "resolved": False,
            "evidence": (
                "Official injection returns only the final candidate, total "
                "attempt count, and final template; it does not persist every "
                "request, response, parsed candidate, or S2Match score."
            ),
        },
        {
            "id": "acceptance_state_not_persisted",
            "resolved": False,
            "evidence": (
                "Official sampling metadata omits score and accepted/fallback "
                "state, so the frozen baseline gate cannot be reconstructed."
            ),
        },
        {
            "id": "claude_4_5_top_p_omitted",
            "resolved": False,
            "evidence": (
                "Official bedrock_utils.py removes top_p for Claude Sonnet "
                "4.5 although the protocol records top_p=0.9."
            ),
        },
    ]
    dependency_blockers = {
        "official_requirement_mismatches": packages["mismatches"],
        "official_checkout_s2match_runtime_missing": not manual[
            "official_checkout_s2match_runtime_available"
        ],
        "amrlib_default_parser_wiring_missing": not manual[
            "amrlib_default_model_stog_available"
        ],
        "glove_environment_binding_missing": not manual[
            "glove_vectors_path_env_resolves"
        ],
        "nltk_environment_binding_missing": not manual[
            "nltk_data_env_includes_frozen_root"
        ],
    }
    return {
        "schema_version": 2,
        "created_at": utc_now(),
        "status": (
            "official_runtime_frozen_remote_execution_unauthorized_"
            "not_ready"
        ),
        "execution_authorized": False,
        "execution_ready": False,
        "protocol_sha256": old_runtime["protocol_sha256"],
        "asset_audit_manifest_sha256": old_runtime[
            "asset_audit_manifest_sha256"
        ],
        "provenance_correction_sha256": sha256(CORRECTION_PATH),
        "supersedes": {
            "runtime_v1_sha256": sha256(OUTPUT / "runtime.json"),
            "pre_execution_manifest_v1_sha256": sha256(
                OUTPUT / "pre_execution_manifest.json"
            ),
            "scope": "execution_provenance",
        },
        "source_lineage": {
            "source_manifest_sha256": sha256(
                OUTPUT / "source_manifest.json"
            ),
            "source_manifest_runtime_v1_sha256": source_manifest[
                "runtime_sha256"
            ],
            "runtime_v1_link_matches": (
                source_manifest["runtime_sha256"]
                == sha256(OUTPUT / "runtime.json")
            ),
            "source_count": len(source_manifest["sources"]),
            "cohorts": source_manifest["cohort_counts"],
            "valid_under_correction": True,
            "basis": "Byte-identical official and legacy prompt extractor.",
        },
        "dataset": old_runtime["dataset"],
        "allocation": old_runtime["allocation"],
        "official_swan": correction["official_checkout"],
        "generation_protocol": old_runtime["generation"],
        "effective_official_behavior": {
            "template_rng_seed": None,
            "remote_temperature": old_runtime["generation"]["temperature"],
            "remote_top_p": None,
            "remote_top_p_reason": (
                "Official Bedrock wrapper removes top_p for Claude Sonnet 4.5."
            ),
            "per_attempt_journal": False,
            "saved_acceptance_state": False,
        },
        "remote_model": {
            **old_runtime["remote_model"],
            "model_probe_authorized": False,
            "generation_authorized": False,
        },
        "dependencies": {
            "official_swan_tracked_files": {
                str(
                    OFFICIAL_ROOT.relative_to(ROOT) / relative
                ): record["sha256"]
                for relative, record in correction["official_checkout"][
                    "tracked_files"
                ].items()
            },
            "provenance_and_audit_code": hash_provenance_code(),
            **resources,
        },
        "python_environment": packages,
        "manual_runtime_readiness": manual,
        "dependency_blockers": dependency_blockers,
        "protocol_implementation_gaps": implementation_gaps,
        "resource_policy": old_runtime["resource_policy"],
        "resource_counts": ZERO_RESOURCE_COUNTS,
        "execution_blocked_by": [
            "unresolved protocol-to-official-implementation gaps",
            "non-matching official Python requirements",
            "missing S2Match runtime inside the official checkout",
            "missing amrlib default parser-model wiring",
            "missing frozen GloVe and NLTK environment bindings",
            "explicit paid-budget authorization not granted",
            "returned model identity not probed",
        ],
    }


def build_pre_execution_v2(runtime: dict) -> dict:
    return {
        "schema_version": 2,
        "created_at": utc_now(),
        "status": (
            "source_preparation_complete_execution_provenance_corrected_"
            "remote_unauthorized_not_ready"
        ),
        "execution_authorized": False,
        "execution_ready": False,
        "completed": {
            "protocol_frozen": True,
            "prior_identity_snapshot_frozen": True,
            "all_35_sources_frozen": True,
            "pilot_main_membership_frozen": True,
            "official_checkout_persisted": True,
            "official_head_tree_and_cleanliness_verified": True,
            "official_36_file_blob_and_sha256_inventory_frozen": True,
            "legacy_vendor_mismatch_set_frozen": True,
            "parser_tokenizer_glove_nltk_files_frozen": True,
        },
        "pending": {
            "protocol_implementation_gap_resolution": True,
            "exact_dependency_environment": True,
            "s2match_runtime_installation": True,
            "parser_model_wiring": True,
            "execution_environment_binding": True,
            "explicit_paid_budget_authorization": True,
            "returned_model_identity_probe": True,
            "model_probe_json": True,
            "remote_generation": True,
        },
        "cohorts": runtime["source_lineage"]["cohorts"],
        "requested_model": runtime["remote_model"]["requested"],
        "returned_model": None,
        "hashes": {
            "protocol_sha256": runtime["protocol_sha256"],
            "provenance_correction_sha256": sha256(CORRECTION_PATH),
            "runtime_v2_sha256": sha256(RUNTIME_V2_PATH),
            "source_manifest_sha256": sha256(
                OUTPUT / "source_manifest.json"
            ),
            "prior_identity_snapshot_sha256": sha256(
                OUTPUT / "prior_identity_snapshot.json"
            ),
            "allocation_journal_sha256": sha256(
                OUTPUT / "allocation_journal.json"
            ),
        },
        "resource_counts": ZERO_RESOURCE_COUNTS,
        "next_valid_transition": (
            "Resolve and freeze the protocol-to-official implementation "
            "gaps and exact local dependencies. Paid execution remains "
            "blocked until explicit budget authorization and one frozen "
            "returned-model identity probe."
        ),
    }


def create() -> dict:
    if any(
        path.exists()
        for path in (CORRECTION_PATH, RUNTIME_V2_PATH, PRE_EXECUTION_V2_PATH)
    ):
        if not all(
            path.is_file()
            for path in (
                CORRECTION_PATH,
                RUNTIME_V2_PATH,
                PRE_EXECUTION_V2_PATH,
            )
        ):
            raise RuntimeError("Partial v2 provenance freeze requires audit")
        return check()

    correction = build_correction()
    freeze_json(CORRECTION_PATH, correction)
    runtime = build_runtime_v2(correction)
    freeze_json(RUNTIME_V2_PATH, runtime)
    pre_execution = build_pre_execution_v2(runtime)
    freeze_json(PRE_EXECUTION_V2_PATH, pre_execution)
    return check()


def check() -> dict:
    correction = load_json(CORRECTION_PATH)
    runtime = load_json(RUNTIME_V2_PATH)
    pre_execution = load_json(PRE_EXECUTION_V2_PATH)
    live_official = official_checkout_evidence()
    live_comparison = compare_legacy_vendor(live_official)
    if correction["official_checkout"] != live_official:
        raise ValueError("Frozen official checkout evidence differs")
    if correction["legacy_vendor_comparison"] != live_comparison:
        raise ValueError("Frozen legacy comparison differs")
    for name, expected in correction["supersession"][
        "legacy_artifact_sha256"
    ].items():
        if sha256(OUTPUT / name) != expected:
            raise ValueError(f"Superseded v1 artifact changed: {name}")
    for name, expected in correction["supersession"][
        "source_artifact_sha256"
    ].items():
        if sha256(OUTPUT / name) != expected:
            raise ValueError(f"Frozen source artifact changed: {name}")
    if (
        runtime["provenance_correction_sha256"]
        != sha256(CORRECTION_PATH)
        or runtime["official_swan"] != live_official
        or runtime["python_environment"] != requirement_state()
        or runtime["manual_runtime_readiness"] != manual_readiness()
        or runtime["execution_authorized"]
        or runtime["execution_ready"]
    ):
        raise ValueError("Runtime v2 lock or live state differs")
    for group in runtime["dependencies"].values():
        for relative, expected in group.items():
            if isinstance(expected, dict):
                expected = expected["sha256"]
            if sha256(ROOT / relative) != expected:
                raise ValueError(f"Frozen v2 dependency changed: {relative}")
    expected_pre_hashes = {
        "protocol_sha256": runtime["protocol_sha256"],
        "provenance_correction_sha256": sha256(CORRECTION_PATH),
        "runtime_v2_sha256": sha256(RUNTIME_V2_PATH),
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
    if (
        pre_execution["hashes"] != expected_pre_hashes
        or pre_execution["execution_authorized"]
        or pre_execution["execution_ready"]
        or pre_execution["resource_counts"] != ZERO_RESOURCE_COUNTS
    ):
        raise ValueError("Pre-execution v2 lock or hashes differ")
    return {
        "status": pre_execution["status"],
        "official_commit": runtime["official_swan"]["commit"],
        "official_tracked_files": runtime["official_swan"][
            "tracked_file_count"
        ],
        "legacy_mismatch_files": correction[
            "legacy_vendor_comparison"
        ]["mismatched_file_count"],
        "source_count": runtime["source_lineage"]["source_count"],
        "execution_authorized": runtime["execution_authorized"],
        "execution_ready": runtime["execution_ready"],
        "package_mismatches": runtime["python_environment"][
            "mismatches"
        ],
        "resource_counts": runtime["resource_counts"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("create", "check"))
    args = parser.parse_args()
    result = create() if args.stage == "create" else check()
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
