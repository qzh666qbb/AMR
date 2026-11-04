"""Freeze score-blind detector-fit selections for WP2-v5."""

import argparse
import hashlib
import json
from pathlib import Path


PROTOCOL_SHA256 = (
    "1af73c61817afce878d88ce29c5f626a981b2382af92ded634331cf9663c2c24"
)
PRIMARY_CANDIDATE = "STRUCTURAL_SEPARATOR_NATIVE_512"
TARGETS = (32, 64, 96, 128, 192, 256, 300, 384, 512, 768, 1024)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def load_rows(path):
    raw = path.read_bytes()
    rows = [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
        if line
    ]
    return raw, rows


def eligible_rows(rows):
    selected = []
    seen = set()
    for row in rows:
        identity = (row["raw_text_sha256"], row["row_index"])
        if identity in seen:
            raise ValueError(f"duplicate row identity: {identity}")
        seen.add(identity)
        if (
            row["cohort"] == "detector_fit"
            and row["candidate_eligible"][PRIMARY_CANDIDATE]
            and row["unit_count"] >= 2
        ):
            selected.append(row)
    return selected


def freeze_selection(rows):
    available = eligible_rows(rows)
    if len(available) < len(TARGETS) + 1:
        raise ValueError("fewer than 12 eligible detector-fit documents")

    output = []
    for target in TARGETS:
        chosen = min(
            available,
            key=lambda row: (
                abs(row["atom_count"] - target),
                row["raw_text_sha256"],
                row["row_index"],
            ),
        )
        available.remove(chosen)
        output.append((f"nearest_{target}", target, chosen))

    chosen = min(
        available,
        key=lambda row: (
            -row["atom_count"],
            row["raw_text_sha256"],
            row["row_index"],
        ),
    )
    output.append(("maximum_remaining", None, chosen))
    return [
        {
            "selection_slot": slot,
            "atom_target": target,
            "row_index": row["row_index"],
            "raw_text_sha256": row["raw_text_sha256"],
            "structural_tape_sha256": row["structural_tape_sha256"],
            "raw_field_type": row["raw_field_type"],
            "atom_count": row["atom_count"],
            "unit_count": row["unit_count"],
            "max_native_tokens_p0": row["max_native_tokens"],
            "contains_raw_text": False,
        }
        for slot, target, row in output
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--semstamp-rows", type=Path, required=True)
    parser.add_argument("--simmark-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol_bytes = args.protocol.read_bytes()
    observed_protocol_sha = sha256_bytes(protocol_bytes)
    if observed_protocol_sha != PROTOCOL_SHA256:
        raise ValueError("WP2-v5 protocol SHA256 mismatch")

    inputs = {}
    systems = {}
    for system_id, path in (
        ("semstamp", args.semstamp_rows),
        ("simmark", args.simmark_rows),
    ):
        raw, rows = load_rows(path)
        inputs[system_id] = {
            "path": str(path),
            "sha256": sha256_bytes(raw),
            "rows": len(rows),
        }
        systems[system_id] = {
            "eligible_detector_fit_rows": len(eligible_rows(rows)),
            "documents": freeze_selection(rows),
        }

    result = {
        "schema_version": 1,
        "created_on": "2026-09-22",
        "audit_mode": "wp2_v5_score_blind_selection",
        "status": "frozen",
        "protocol": {
            "path": str(args.protocol),
            "sha256": observed_protocol_sha,
        },
        "primary_candidate": PRIMARY_CANDIDATE,
        "atom_targets": list(TARGETS),
        "maximum_documents_per_system": 12,
        "inputs": inputs,
        "systems": systems,
        "execution_attestation": {
            "model_loads": 0,
            "model_score_reads": 0,
            "calibration_score_reads": 0,
            "test_score_reads": 0,
            "positive_score_reads": 0,
            "watermark_generation_calls": 0,
            "provider_api_calls": 0,
        },
        "contains_raw_text": False,
    }
    encoded = (
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("ascii")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(json.dumps({
        "status": result["status"],
        "output_sha256": sha256_bytes(encoded),
        "systems": {
            system_id: {
                "selected": len(system["documents"]),
                "atom_counts": [
                    row["atom_count"] for row in system["documents"]
                ],
                "unit_counts": [
                    row["unit_count"] for row in system["documents"]
                ],
            }
            for system_id, system in systems.items()
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
