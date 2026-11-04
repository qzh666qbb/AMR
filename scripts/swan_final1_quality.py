"""Blind human review export and fail-closed label freeze. No scoring imports."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from swan_final1_common import digest, freeze, load, now, sha256

BASE_RATINGS = ("coherence", "naturalness", "readability", "context_compatibility")
BASE_ISSUES = ("truncation", "repetition", "topic_splice", "obvious_factual_conflict",
               "role_conflict", "negation_modality_time_place_quantity_conflict",
               "unclear_reference", "amr_placeholder_leak")
PAIR_RATINGS = tuple(f"{side}_{key}" for side in ("left", "right")
                     for key in ("coherence", "naturalness", "readability"))
PAIR_PRESERVES = ("facts", "events_roles", "negation", "modality", "time", "place",
                  "quantity", "reference", "logical_relation", "all_propositions")
PAIR_ISSUES = ("fragment", "run_on", "comma_splice", "topic_splice", "truncation")


def blank_review_row(blind_id, kind):
    ratings = BASE_RATINGS if kind == "baseline" else PAIR_RATINGS
    issues = BASE_ISSUES if kind == "baseline" else PAIR_ISSUES
    row = {"blind_id": blind_id, **{key: None for key in (*ratings, *issues)},
           "notes": ""}
    if kind == "pair":
        row.update({f"preserves_{key}": None for key in PAIR_PRESERVES})
        row["related_natural_merge"] = None
    else:
        row["external_facts_not_verifiable"] = None
    return row


def export_packets(records, folder, kind, contract_sha256, created_at=None):
    """records is a whitelist text export, not injection/detector logs.

    Baseline: allocation_id, source_excerpt, text.
    Pair: allocation_id, condition_id, original_text, transformed_text.
    """
    if kind not in {"baseline", "pair"}:
        raise ValueError("Unknown quality kind")
    created_at = created_at or now()
    folder = Path(folder)
    expected_fields = ({"allocation_id", "source_excerpt", "text"} if kind == "baseline"
                       else {"allocation_id", "condition_id", "original_text", "transformed_text"})
    if any(set(row) != expected_fields for row in records):
        raise ValueError("Quality export accepts only whitelisted text fields")
    keys = [(r["allocation_id"], r.get("condition_id")) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate review item")
    private, summaries = [], []
    for reviewer_slot in ("A", "B"):
        rows = []
        for source in records:
            key = [source["allocation_id"], source.get("condition_id")]
            blind_id = digest(["final1-quality", kind, key])[:24]
            swap = int(digest(["final1-side", reviewer_slot, key])[-1], 16) % 2 == 1
            row = {"blind_id": blind_id}
            if kind == "baseline":
                row.update(source_excerpt=source["source_excerpt"], text=source["text"])
            else:
                texts = [source["original_text"], source["transformed_text"]]
                if swap:
                    texts.reverse()
                row.update(left=texts[0], right=texts[1])
            rows.append(row)
            private.append({"reviewer_slot": reviewer_slot, "blind_id": blind_id,
                            "allocation_id": source["allocation_id"],
                            "condition_id": source.get("condition_id"),
                            "original_on_right": swap if kind == "pair" else None,
                            "input_sha256": digest(source)})
        rows.sort(key=lambda r: digest(["final1-order", reviewer_slot, r["blind_id"]]))
        packet = {"schema_version": 1, "created_at": created_at, "kind": kind,
                  "reviewer_slot": reviewer_slot, "items": rows,
                  "instructions_file": "quality_protocol.md"}
        packet_path = folder / f"packet_{reviewer_slot}.json"
        freeze(packet_path, packet)
        template = {
            "schema_version": 1, "reviewer_id": None, "completed_at": None,
            "independent_review": None, "packet_sha256": sha256(packet_path),
            "reviews": [blank_review_row(r["blind_id"], kind) for r in rows],
        }
        freeze(folder / f"review_{reviewer_slot}_template.json", template)
        summaries.append({"slot": reviewer_slot, "packet_sha256": sha256(packet_path)})
    freeze(folder / "private_index.json", {
        "created_at": created_at, "contract_sha256": contract_sha256,
        "kind": kind, "packets": summaries, "items": private,
    })
    return {"items_per_reviewer": len(records), "kind": kind,
            "private_index_sha256": sha256(folder / "private_index.json")}


def parse_utc(value):
    if not isinstance(value, str):
        raise ValueError("Missing timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("UTC timestamp with timezone required")
    return parsed


def row_pass(row, kind):
    ratings = BASE_RATINGS if kind == "baseline" else PAIR_RATINGS
    issues = BASE_ISSUES if kind == "baseline" else PAIR_ISSUES
    if any(type(row.get(k)) is not int or not 1 <= row[k] <= 5 for k in ratings):
        raise ValueError("Missing or invalid rating")
    boolean_fields = list(issues)
    if kind == "pair":
        boolean_fields += [f"preserves_{k}" for k in PAIR_PRESERVES]
        boolean_fields += ["related_natural_merge"]
    else:
        boolean_fields += ["external_facts_not_verifiable"]
    if any(type(row.get(k)) is not bool for k in boolean_fields):
        raise ValueError("Missing or invalid boolean label")
    ok = all(row[k] >= 4 for k in ratings) and not any(row[k] for k in issues)
    if kind == "pair":
        ok = ok and all(row[f"preserves_{k}"] for k in PAIR_PRESERVES) \
            and row["related_natural_merge"]
    return ok


def validated_review(review, packet, packet_hash):
    identity = review.get("reviewer_id")
    if not isinstance(identity, str) or not identity.strip() or \
            identity.strip().casefold() in {"none", "null"}:
        raise ValueError("Real reviewer identity required")
    if review.get("independent_review") is not True or review.get("packet_sha256") != packet_hash:
        raise ValueError("Independence declaration or packet hash invalid")
    if parse_utc(review.get("completed_at")) < parse_utc(packet["created_at"]):
        raise ValueError("Review predates packet")
    expected = {row["blind_id"] for row in packet["items"]}
    rows = review.get("reviews")
    if not isinstance(rows, list) or len(rows) != len(expected) or \
            {r.get("blind_id") for r in rows} != expected:
        raise ValueError("Review IDs missing, duplicated or changed")
    return {r["blind_id"]: row_pass(r, packet["kind"]) for r in rows}


def agreement(left, right):
    n = len(left)
    if n != len(right):
        raise ValueError("Unpaired ratings")
    if not n:
        return {"n": 0, "agreement": None, "kappa": None}
    observed = sum(a == b for a, b in zip(left, right)) / n
    p, q = sum(left) / n, sum(right) / n
    expected = p*q + (1-p)*(1-q)
    return {"n": n, "agreement": observed,
            "kappa": (observed - expected) / (1 - expected) if expected < 1 else None}


def build_decision(folder, contract_sha256, frozen_at):
    """Pure replay of packets, private mapping and original reviewer labels."""
    folder = Path(folder)
    index = load(folder / "private_index.json")
    if index["contract_sha256"] != contract_sha256:
        raise ValueError("Quality contract changed")
    reviews, labels, packet_hashes = [], [], []
    for slot in ("A", "B"):
        packet_path = folder / f"packet_{slot}.json"
        packet = load(packet_path)
        expected_hash = next(r["packet_sha256"] for r in index["packets"] if r["slot"] == slot)
        if sha256(packet_path) != expected_hash:
            raise ValueError("Packet content changed since export")
        review = load(folder / f"review_{slot}.json")
        if parse_utc(review.get("completed_at")) > parse_utc(frozen_at):
            raise ValueError("Review timestamp lies after freeze")
        reviews.append(review)
        labels.append(validated_review(review, packet, expected_hash))
        packet_hashes.append(expected_hash)
    if reviews[0]["reviewer_id"].strip().casefold() == reviews[1]["reviewer_id"].strip().casefold():
        raise ValueError("Two different reviewers required")
    mappings = [r for r in index["items"] if r["reviewer_slot"] == "A"]
    if {r["blind_id"] for r in mappings} != set(labels[0]) or set(labels[0]) != set(labels[1]):
        raise ValueError("Private mapping does not match both packets")
    decisions = [{
        "allocation_id": r["allocation_id"], "condition_id": r["condition_id"],
        "blind_id": r["blind_id"], "reviewer_a_pass": labels[0][r["blind_id"]],
        "reviewer_b_pass": labels[1][r["blind_id"]],
        "quality_pass": labels[0][r["blind_id"]] and labels[1][r["blind_id"]],
    } for r in mappings]
    return {
        "schema_version": 1, "frozen_at": frozen_at, "kind": index["kind"],
        "contract_sha256": contract_sha256, "status": "quality_frozen_before_detection",
        "private_index_sha256": sha256(folder / "private_index.json"),
        "packet_sha256": packet_hashes,
        "review_sha256": [sha256(folder / f"review_{s}.json") for s in ("A", "B")],
        "reviewer_ids": [r["reviewer_id"] for r in reviews],
        "decisions": decisions,
        "agreement": agreement([r["reviewer_a_pass"] for r in decisions],
                               [r["reviewer_b_pass"] for r in decisions]),
        "quality_pass_count": sum(r["quality_pass"] for r in decisions),
    }


def verify_decision(folder, contract_sha256):
    folder = Path(folder)
    saved = load(folder / "decision.json")
    actual = build_decision(folder, contract_sha256, saved["frozen_at"])
    if saved != actual:
        raise ValueError("Quality decision differs from original reviewer evidence")
    return actual


def finalize(folder, contract_sha256, experiment_folder, frozen_at=None):
    """No hand-entered gate flag; all decisions recomputed from original labels."""
    folder, experiment_folder = Path(folder), Path(experiment_folder)
    if (folder / "decision.json").exists():
        return verify_decision(folder, contract_sha256)
    for name in ("detection", "calibration_scores", "test_scores"):
        if (experiment_folder / name).exists():
            raise PermissionError("Quality labels must freeze before any final detection")
    return freeze(folder / "decision.json",
                  build_decision(folder, contract_sha256, frozen_at or now()))
