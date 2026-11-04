"""Allocate unseen C4 sources for the independent feasibility pilot."""
from __future__ import annotations

from pathlib import Path

from swan_anchor1_sources import (
    REVISION,
    history_snapshot,
    identity,
    screening,
    stream_rows,
)
from swan_feasibility2_common import OUT, freeze, load, now, sha256

PILOT_COUNT = 20
POSITIVE_RANGE = ("train", 1_600_000, 1_650_000)
HUMAN_RANGE = ("train", 1_650_000, 1_650_800)
HUMAN_COHORTS = (("calibration", 400), ("test", 400))


def select(rows, snapshot, count=PILOT_COUNT):
    seen_text = set(snapshot["normalized_text_sha256"])
    seen_url = set(snapshot["url_sha256"])
    selected, screened = [], []
    for split, index, row in rows:
        result = screening(row, split, index, seen_text, seen_url)
        screened.append(result)
        if not result["selected"]:
            continue
        ordinal = len(selected)
        selected.append({
            **result,
            "phase": "pilot",
            "ordinal": ordinal,
            "allocation_id": f"feasibility2-pilot-{ordinal:04d}",
        })
        seen_text.update({
            result["normalized_source_sha256"],
            result["normalized_window_sha256"],
            *(identity(anchor) for anchor in result["source_anchors"]),
        })
        if result["url"]:
            seen_url.add(result["url_sha256"])
        if len(selected) == count:
            return selected, screened
    raise RuntimeError(f"Fixed source scan exhausted ({len(selected)}/{count})")


def human_locators():
    split, start, stop = HUMAN_RANGE
    if start + sum(count for _, count in HUMAN_COHORTS) != stop:
        raise ValueError("Human control range must exactly match cohort sizes")
    rows, cursor = [], start
    for phase, count in HUMAN_COHORTS:
        for ordinal in range(count):
            rows.append({
                "allocation_id": f"feasibility2-{phase}-{ordinal:04d}",
                "phase": phase,
                "ordinal": ordinal,
                "split": split,
                "row": cursor,
                "locator": f"datasets-server/realnewslike/{split}:{cursor}",
                "dataset_revision": REVISION,
                "content_materialized": False,
            })
            cursor += 1
    return rows


def allocate(folder=OUT):
    folder = Path(folder)
    if (folder / "source_manifest.json").exists():
        return check_sources(folder)
    snapshot_path = folder / "prior_identity_snapshot.json"
    snapshot = load(snapshot_path) if snapshot_path.exists() else freeze(
        snapshot_path, history_snapshot(excluded=folder)
    )
    positives, screened = select(
        stream_rows(*POSITIVE_RANGE), snapshot, PILOT_COUNT
    )
    files = {
        "sealed/pilot_sources.json": positives,
        "sealed/source_screening.json": screened,
        "sealed/human_locators.json": human_locators(),
    }
    hashes = {}
    for relative, value in files.items():
        path = folder / relative
        freeze(path, value)
        hashes[relative] = sha256(path)
    return freeze(folder / "source_manifest.json", {
        "created_at": now(),
        "dataset_revision": REVISION,
        "dataset_endpoint": "https://datasets-server.huggingface.co/rows",
        "positive_range": list(POSITIVE_RANGE),
        "human_range": list(HUMAN_RANGE),
        "pilot_count": PILOT_COUNT,
        "human_cohorts": dict(HUMAN_COHORTS),
        "replacement_allowed": False,
        "window_rule": "first contiguous 10 sentences; each 8-35 words; total 100-220",
        "prior_identity_snapshot_sha256": sha256(snapshot_path),
        "sealed_file_hashes": hashes,
        "rows_scanned": len(screened),
        "rows_excluded": len(screened) - len(positives),
        "last_locator": screened[-1]["locator"],
        "final1_positive_source_region": "train parquet shards 13-16",
        "anchor1_positive_source_region": "validation rows 0-13862",
        "final1_sealed_content_access": False,
        "credential_file_access": False,
        "model_parser_scorer_calls_at_allocation": 0,
    })


def check_sources(folder=OUT):
    folder = Path(folder)
    manifest = load(folder / "source_manifest.json")
    if manifest["pilot_count"] != PILOT_COUNT \
            or manifest["positive_range"] != list(POSITIVE_RANGE) \
            or manifest["human_range"] != list(HUMAN_RANGE) \
            or manifest["human_cohorts"] != dict(HUMAN_COHORTS) \
            or manifest["replacement_allowed"] \
            or manifest["final1_sealed_content_access"] \
            or manifest["credential_file_access"]:
        raise ValueError("Feasibility2 source isolation contract changed")
    if sha256(folder / "prior_identity_snapshot.json") != \
            manifest["prior_identity_snapshot_sha256"]:
        raise ValueError("Historical identity snapshot changed")
    for relative, expected in manifest["sealed_file_hashes"].items():
        if sha256(folder / relative) != expected:
            raise ValueError(f"Sealed source changed: {relative}")
    return manifest
