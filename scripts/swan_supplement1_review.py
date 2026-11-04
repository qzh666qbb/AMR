"""Validate two independent, complete human review returns without filling labels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from swan_final1_common import freeze, load, now, sha256, single_writer
from swan_supplement1_sources import OUT


def read_review(path, items):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(items) or {r["item_id"] for r in rows} != items:
        raise ValueError("Review rows must match every blind item exactly once")
    if len({r["item_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate review item")
    identifiers = {r["reviewer_id"].strip() for r in rows}
    if len(identifiers) != 1 or not next(iter(identifiers), ""):
        raise ValueError("One nonempty reviewer identifier required per file")
    scores = {}
    for row in rows:
        if row["is_human"].strip().lower() != "yes":
            raise ValueError("Human declaration missing; tool cannot infer it")
        values = {}
        for version in ("1", "2"):
            for metric in ("grammar", "coherence"):
                value = row[f"version{version}_{metric}"].strip()
                if value not in ("1", "2", "3", "4", "5"):
                    raise ValueError("Every score must be an integer from 1 to 5")
                values[f"{version}_{metric}"] = int(value)
        flags = [row[k].strip().lower() for k in
                 ("version1_no_AMR_leak", "version2_no_AMR_leak", "meaning_preserved")]
        if any(value not in ("yes", "no") for value in flags):
            raise ValueError("Complete yes/no labels required")
        scores[row["item_id"]] = all(value == "yes" for value in flags) and all(
            values[f"{v}_grammar"] >= 4 and values[f"{v}_coherence"] >= 3 for v in ("1", "2"))
    return next(iter(identifiers)), scores


def finalize(reviewer1, reviewer2):
    packet_path = OUT / "private/human_review/pilot_packet.json"
    packet = load(packet_path)
    if not (OUT / "PILOT_RESULT.json").exists():
        raise ValueError("Pilot must be terminal before the human gate can be finalized")
    pilot = load(OUT / "PILOT_RESULT.json")
    items = {r["item_id"] for r in packet["items"]}
    if not items:
        raise ValueError("No eligible review items; pilot remains Stop")
    id1, first = read_review(reviewer1, items)
    id2, second = read_review(reviewer2, items)
    if id1 == id2 or sha256(reviewer1) == sha256(reviewer2):
        raise ValueError("Two distinct independent reviewer returns required")
    passing = [item for item in sorted(items) if first[item] and second[item]]
    passed = pilot["engineering_pass"] >= 6 and len(passing) >= 6 and pilot["stage"] != "Halt"
    return freeze(OUT / "HUMAN_GATE.json", {
        "at": now(), "gate": "Go" if passed else "Stop",
        "pilot_denominator": 8, "review_items": len(items), "quality_pass": len(passing),
        "passing_blind_item_ids": passing, "engineering_pass": pilot["engineering_pass"],
        "reviewer_count": 2, "identity_basis": "self_declared_independent_humans",
        "packet_sha256": sha256(packet_path),
        "review_file_sha256": [sha256(reviewer1), sha256(reviewer2)],
        "main_started": False,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("finalize",))
    parser.add_argument("--reviewer1", required=True, type=Path)
    parser.add_argument("--reviewer2", required=True, type=Path)
    args = parser.parse_args()
    with single_writer(OUT):
        result = finalize(args.reviewer1, args.reviewer2)
    print(json.dumps(result))
