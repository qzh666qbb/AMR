"""Isolate administrator keys from all human-facing adjudication folders."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import run_semantic_fact_carrier_pilot as pilot

HUMAN_FILES = {"facts.csv", "pairs.csv", "reviewer.json", "INSTRUCTIONS.md", "text_words.txt"}


def copy_human_files(source, target):
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    for path in source.iterdir():
        if path.name in HUMAN_FILES and path.is_file() and not path.is_symlink():
            shutil.copy2(path, target / path.name)
            (target / path.name).chmod(0o600)
    # No keys, manifests, source identifiers or labels from another reviewer.
    return [p.name for p in target.iterdir() if p.is_file()]


def collect(out, private, check_release=True):
    pilot.verify(out, private)
    if check_release:
        release = pilot.load(out / "REVIEW_RELEASE_FREEZE.json")
        for source, expected in release["code_sha256"].items():
            if pilot.sha(pilot.ROOT / source) != expected:
                raise RuntimeError("Review release source drift")
        if pilot.sha(out / "FREEZE.json") != release["generation_freeze_sha256"]:
            raise RuntimeError("Generation freeze drift")
    review = out / "private_review"
    if (review / "adjudicator/KEYS.json").exists():
        raise RuntimeError("Administrator key must not be in a human handoff folder")
    # The frozen collector executes in an external workspace. Its internal keys
    # never enter the reviewer/adjudicator packets returned to the user.
    working = private / "human_collection_workspace"
    working.mkdir(parents=True, exist_ok=True, mode=0o700)
    freeze = pilot.load(out / "FREEZE.json")
    for name in ("FREEZE.json", *freeze["public_sha256"]):
        shutil.copy2(out / name, working / name)
    admin = working / "private_review/admin"
    admin.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name in ("KEYS.json", "MANIFEST.json"):
        shutil.copy2(review / "admin" / name, admin / name)
    for name in ("reviewer1", "reviewer2", "adjudicator"):
        source = review / name
        if source.exists():
            copy_human_files(source, working / "private_review" / name)
    result = pilot.collect(working, private)
    if (working / "private_review/adjudicator").exists():
        copy_human_files(working / "private_review/adjudicator", review / "adjudicator")
    for name in ("COLLECTION_ERRORS.json", "DISPUTES.json", "RESOLVED.json"):
        source = admin / name
        if source.exists():
            shutil.copy2(source, review / "admin" / name)
    outcome = pilot.load(working / "HUMAN_COLLECTION.json")
    outcome["collector_release"] = "isolated_administrator_keys:v1"
    pilot.atomic(out / "HUMAN_COLLECTION.json", outcome)
    return result


def seal(out):
    if (out / "REVIEW_RELEASE_FREEZE.json").exists():
        raise RuntimeError("Review release is immutable")
    tests = pilot.load(out / "REVIEW_RELEASE_TESTS.json")
    sources = ["scripts/collect_semantic_fact_carrier_reviews.py",
               "scripts/test_semantic_fact_review_handoff.py"]
    hashes = {p: pilot.sha(pilot.ROOT / p) for p in sources}
    if tests["status"] != "PASS" or tests["code_sha256"] != hashes:
        raise RuntimeError("Review release tests missing or stale")
    pilot.atomic(out / "REVIEW_RELEASE_FREEZE.json", {
        "at": pilot.now(), "release": "isolated_administrator_keys:v1",
        "generation_freeze_sha256": pilot.sha(out / "FREEZE.json"),
        "code_sha256": hashes, "tests_sha256": pilot.sha(out / "REVIEW_RELEASE_TESTS.json"),
        "generation_sources_changed": False, "new_provider_calls": 0,
        "human_gate_or_scoring_changed": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("collect", "seal"))
    parser.add_argument("--out", type=Path, default=pilot.OUT)
    parser.add_argument("--private", type=Path, default=pilot.PRIVATE)
    args = parser.parse_args()
    if args.command == "seal":
        seal(args.out)
        print(json.dumps({"status": "SEALED"}))
    else:
        with pilot.Exclusive(args.private):
            print(json.dumps(collect(args.out, args.private), ensure_ascii=False, indent=2))
