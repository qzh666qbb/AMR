"""Supplemental non-JSON history audit for frozen SWAN source identities."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/swan_positive_collection1"
SHARD_NEEDLE = b"c4-train.00008-of-00512"
TEXT_MARKERS = ("article", "original", "passage", "prompt", "source", "text")
LEGACY_AUDIT = OUTPUT / "history_format_audit.json"
V2_AUDIT = OUTPUT / "history_format_audit_v2.json"
V3_AUDIT = OUTPUT / "history_format_audit_v3.json"
V4_AUDIT = OUTPUT / "history_format_audit_v4.json"
V5_AUDIT = OUTPUT / "history_format_audit_v5.json"
V6_AUDIT = OUTPUT / "history_format_audit_v6.json"
V7_AUDIT = OUTPUT / "history_format_audit_v7.json"


def canonical(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def text_hash(text: str) -> str:
    return hashlib.sha256(canonical(text).casefold().encode()).hexdigest()


def url_hash(url: str) -> str:
    return hashlib.sha256(canonical(url).encode()).hexdigest()


def contains_bytes(path: Path, needle: bytes) -> bool:
    overlap = len(needle) - 1
    previous = b""
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value = previous + block
            if needle in value:
                return True
            previous = value[-overlap:] if overlap else b""
    return False


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def audit() -> dict:
    source_manifest = json.loads(
        (OUTPUT / "source_manifest.json").read_text()
    )
    selected_text = {
        row["normalized_source_sha256"]
        for row in source_manifest["sources"]
    }
    selected_url = {
        row["url_sha256"]
        for row in source_manifest["sources"]
        if row["url"]
    }

    csv.field_size_limit(sys.maxsize)
    csv_files = sorted((ROOT / "experiments").rglob("*.csv"))
    csv_columns = 0
    csv_values = 0
    text_collisions = []
    url_collisions = []
    for path in csv_files:
        with path.open(
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        ) as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            text_columns = [
                column
                for column in reader.fieldnames
                if any(
                    marker in column.casefold()
                    for marker in TEXT_MARKERS
                )
            ]
            url_columns = [
                column
                for column in reader.fieldnames
                if "url" in column.casefold()
            ]
            csv_columns += len(text_columns) + len(url_columns)
            for row_number, row in enumerate(reader, start=2):
                for column in text_columns:
                    value = row.get(column) or ""
                    if canonical(value):
                        csv_values += 1
                        if text_hash(value) in selected_text:
                            text_collisions.append(
                                {
                                    "path": str(path.relative_to(ROOT)),
                                    "row": row_number,
                                    "column": column,
                                }
                            )
                for column in url_columns:
                    value = row.get(column) or ""
                    if canonical(value):
                        csv_values += 1
                        if url_hash(value) in selected_url:
                            url_collisions.append(
                                {
                                    "path": str(path.relative_to(ROOT)),
                                    "row": row_number,
                                    "column": column,
                                }
                            )

    arrow_files = sorted((ROOT / "experiments").rglob("*.arrow"))
    missing_metadata = []
    for path in arrow_files:
        for name in ("dataset_info.json", "state.json"):
            companion = path.parent / name
            if not companion.is_file():
                missing_metadata.append(str(companion.relative_to(ROOT)))

    roots = [
        ROOT / "experiments",
        ROOT / "scripts",
        ROOT / "RESEARCH_STATE.md",
        ROOT / "CLAIM_EVIDENCE_MATRIX.md",
        ROOT / "CURRENT_RESULTS.md",
    ]
    files_scanned = 0
    locator_hits = []
    for root in roots:
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if (
                not path.is_file()
                or path == Path(__file__).resolve()
                or OUTPUT in path.parents
            ):
                continue
            files_scanned += 1
            if contains_bytes(path, SHARD_NEEDLE):
                locator_hits.append(str(path.relative_to(ROOT)))

    return {
        "schema_version": 1,
        "experiment": "swan_positive_collection1",
        "scope": (
            "Supplemental CSV collision and prior shard-locator audit; Arrow "
            "files are inventoried for metadata coverage, not value-parsed."
        ),
        "csv": {
            "files": len(csv_files),
            "identity_like_columns": csv_columns,
            "nonempty_values_checked": csv_values,
            "selected_text_collisions": text_collisions,
            "selected_url_collisions": url_collisions,
        },
        "arrow": {
            "files": len(arrow_files),
            "missing_dataset_metadata_or_state": missing_metadata,
            "values_parsed": False,
        },
        "prior_shard_8_locator": {
            "files_scanned": files_scanned,
            "hits_outside_new_collection": locator_hits,
        },
        "checks": {
            "no_csv_text_collision": not text_collisions,
            "no_csv_url_collision": not url_collisions,
            "all_arrow_datasets_have_metadata_and_state": (
                not missing_metadata
            ),
            "no_prior_shard_8_locator": not locator_hits,
        },
    }


def build_v2(current: dict) -> dict:
    legacy = json.loads(LEGACY_AUDIT.read_text(encoding="utf-8"))
    stable_sections = ("csv", "arrow", "checks")
    if any(current[key] != legacy[key] for key in stable_sections):
        raise ValueError("Supplemental history conclusions changed")
    legacy_locator = legacy["prior_shard_8_locator"]
    current_locator = current["prior_shard_8_locator"]
    if (
        legacy_locator["hits_outside_new_collection"]
        != current_locator["hits_outside_new_collection"]
        or current_locator["files_scanned"]
        < legacy_locator["files_scanned"]
    ):
        raise ValueError("Supplemental locator evidence regressed")
    return {
        "schema_version": 2,
        "experiment": "swan_positive_collection1",
        "status": "verified_after_provenance_correction",
        "supersedes": {
            "path": str(LEGACY_AUDIT.relative_to(ROOT)),
            "sha256": file_sha256(LEGACY_AUDIT),
            "legacy_artifact_preserved": True,
        },
        "reason": (
            "The scan universe gained provenance scripts after the v1 "
            "snapshot; identity collision and locator conclusions are "
            "unchanged."
        ),
        "files_scanned_delta": (
            current_locator["files_scanned"]
            - legacy_locator["files_scanned"]
        ),
        "current_evidence": current,
    }


def build_v3(current: dict) -> dict:
    prior = json.loads(V2_AUDIT.read_text(encoding="utf-8"))
    prior_evidence = prior["current_evidence"]
    stable_sections = ("csv", "arrow", "checks")
    if any(
        current[key] != prior_evidence[key]
        for key in stable_sections
    ):
        raise ValueError("Supplemental history conclusions changed")
    prior_locator = prior_evidence["prior_shard_8_locator"]
    current_locator = current["prior_shard_8_locator"]
    if (
        prior_locator["hits_outside_new_collection"]
        != current_locator["hits_outside_new_collection"]
        or current_locator["files_scanned"]
        < prior_locator["files_scanned"]
    ):
        raise ValueError("Supplemental locator evidence regressed")
    return {
        "schema_version": 3,
        "experiment": "swan_positive_collection1",
        "status": "verified_after_execution_contract",
        "supersedes": {
            "path": str(V2_AUDIT.relative_to(ROOT)),
            "sha256": file_sha256(V2_AUDIT),
            "legacy_artifact_preserved": True,
        },
        "reason": (
            "The scan universe gained execution-contract scripts after the "
            "v2 snapshot; identity collision and locator conclusions are "
            "unchanged."
        ),
        "files_scanned_delta": (
            current_locator["files_scanned"]
            - prior_locator["files_scanned"]
        ),
        "current_evidence": current,
    }


def build_v4(current: dict) -> dict:
    prior = json.loads(V3_AUDIT.read_text(encoding="utf-8"))
    prior_evidence = prior["current_evidence"]
    stable_sections = ("csv", "arrow", "checks")
    if any(
        current[key] != prior_evidence[key]
        for key in stable_sections
    ):
        raise ValueError("Supplemental history conclusions changed")
    prior_locator = prior_evidence["prior_shard_8_locator"]
    current_locator = current["prior_shard_8_locator"]
    if (
        prior_locator["hits_outside_new_collection"]
        != current_locator["hits_outside_new_collection"]
        or current_locator["files_scanned"]
        < prior_locator["files_scanned"]
    ):
        raise ValueError("Supplemental locator evidence regressed")
    return {
        "schema_version": 4,
        "experiment": "swan_positive_collection1",
        "status": "verified_after_budget_estimator",
        "supersedes": {
            "path": str(V3_AUDIT.relative_to(ROOT)),
            "sha256": file_sha256(V3_AUDIT),
            "legacy_artifact_preserved": True,
        },
        "reason": (
            "The scan universe gained the budget-estimation script after "
            "the v3 snapshot; identity collision and locator conclusions "
            "are unchanged."
        ),
        "files_scanned_delta": (
            current_locator["files_scanned"]
            - prior_locator["files_scanned"]
        ),
        "current_evidence": current,
    }


def build_v5(current: dict) -> dict:
    prior = json.loads(V4_AUDIT.read_text(encoding="utf-8"))
    prior_evidence = prior["current_evidence"]
    stable_sections = ("csv", "arrow", "checks")
    if any(
        current[key] != prior_evidence[key]
        for key in stable_sections
    ):
        raise ValueError("Supplemental history conclusions changed")
    prior_locator = prior_evidence["prior_shard_8_locator"]
    current_locator = current["prior_shard_8_locator"]
    if (
        prior_locator["hits_outside_new_collection"]
        != current_locator["hits_outside_new_collection"]
        or current_locator["files_scanned"]
        < prior_locator["files_scanned"]
    ):
        raise ValueError("Supplemental locator evidence regressed")
    delta = (
        current_locator["files_scanned"]
        - prior_locator["files_scanned"]
    )
    if delta != 113:
        raise ValueError("Unexpected local-preflight scan-universe delta")
    return {
        "schema_version": 5,
        "experiment": "swan_positive_collection1",
        "status": "verified_after_local_backend_preflight",
        "supersedes": {
            "path": str(V4_AUDIT.relative_to(ROOT)),
            "sha256": file_sha256(V4_AUDIT),
            "legacy_artifact_preserved": True,
        },
        "reason": (
            "The scan universe gained 3 local-preflight scripts and 110 "
            "local-preflight evidence files after the v4 snapshot; identity "
            "collision and locator conclusions are unchanged."
        ),
        "files_scanned_delta": delta,
        "current_evidence": current,
    }


def build_v6(current: dict) -> dict:
    prior = json.loads(V5_AUDIT.read_text(encoding="utf-8"))
    prior_evidence = prior["current_evidence"]
    stable_sections = ("csv", "arrow", "checks")
    if any(
        current[key] != prior_evidence[key]
        for key in stable_sections
    ):
        raise ValueError("Supplemental history conclusions changed")
    prior_locator = prior_evidence["prior_shard_8_locator"]
    current_locator = current["prior_shard_8_locator"]
    if (
        prior_locator["hits_outside_new_collection"]
        != current_locator["hits_outside_new_collection"]
        or current_locator["files_scanned"]
        < prior_locator["files_scanned"]
    ):
        raise ValueError("Supplemental locator evidence regressed")
    delta = (
        current_locator["files_scanned"]
        - prior_locator["files_scanned"]
    )
    if delta != 8:
        raise ValueError("Unexpected DeepSeek-preflight scan-universe delta")
    return {
        "schema_version": 6,
        "experiment": "swan_positive_collection1",
        "status": "verified_after_deepseek_preflight_freeze",
        "supersedes": {
            "path": str(V5_AUDIT.relative_to(ROOT)),
            "sha256": file_sha256(V5_AUDIT),
            "legacy_artifact_preserved": True,
        },
        "reason": (
            "The scan universe gained 3 DeepSeek-preflight scripts and 5 "
            "pre-execution evidence files after the v5 snapshot; identity "
            "collision and locator conclusions are unchanged."
        ),
        "files_scanned_delta": delta,
        "current_evidence": current,
    }


def build_v7(current: dict) -> dict:
    prior = json.loads(V6_AUDIT.read_text(encoding="utf-8"))
    prior_evidence = prior["current_evidence"]
    stable_sections = ("csv", "arrow", "checks")
    if any(
        current[key] != prior_evidence[key]
        for key in stable_sections
    ):
        raise ValueError("Supplemental history conclusions changed")
    prior_locator = prior_evidence["prior_shard_8_locator"]
    current_locator = current["prior_shard_8_locator"]
    if (
        prior_locator["hits_outside_new_collection"]
        != current_locator["hits_outside_new_collection"]
        or current_locator["files_scanned"]
        < prior_locator["files_scanned"]
    ):
        raise ValueError("Supplemental locator evidence regressed")
    delta = (
        current_locator["files_scanned"]
        - prior_locator["files_scanned"]
    )
    if delta != 100:
        raise ValueError("Unexpected DeepSeek-execution scan-universe delta")
    return {
        "schema_version": 7,
        "experiment": "swan_positive_collection1",
        "status": "verified_after_deepseek_preflight_execution",
        "supersedes": {
            "path": str(V6_AUDIT.relative_to(ROOT)),
            "sha256": file_sha256(V6_AUDIT),
            "legacy_artifact_preserved": True,
        },
        "reason": (
            "The scan universe gained the bounded DeepSeek execution "
            "wrappers, immutable call/score traces, amendments and audits; "
            "identity collision and locator conclusions are unchanged."
        ),
        "files_scanned_delta": delta,
        "current_evidence": current,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-check",
        action="store_true",
        help="Check the immutable v1 artifact instead of writing v2.",
    )
    args = parser.parse_args()
    current = audit()
    if not all(current["checks"].values()):
        raise ValueError("Supplemental history-format audit failed")

    if args.legacy_check:
        path = LEGACY_AUDIT
        result = current
    else:
        path = V7_AUDIT
        result = build_v7(current)
    text = json.dumps(result, indent=2, ensure_ascii=True) + "\n"
    if path.exists() and path.read_text() != text:
        raise ValueError("Frozen supplemental audit differs")
    path.write_text(text)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
