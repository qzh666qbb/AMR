"""Human event supervision and four independent data roles for paper two."""
from __future__ import annotations

from datetime import datetime, timezone
import math
import random

from .semantic_fact_carrier import classify_facts, validate_evidence

VERSION = "semantic-human-supervision:v2"
ROLES = ("train", "validation", "calibration", "test")
KINDS = ("carrier", "background", "duplicate", "unknown")
FACT_FIELDS = ("sender", "recipient", "object", "transfer_status")


def split_scenes(scenes, seed=20261026):
    """Freeze before labels: 12/4/4/4 development scenes, all variants together."""
    scenes = sorted(scenes)
    if len(scenes) != 24 or len(set(scenes)) != 24:
        raise ValueError("Exactly 24 distinct development scenes required")
    random.Random(seed).shuffle(scenes)
    groups = dict(zip(ROLES, (scenes[:12], scenes[12:16], scenes[16:20], scenes[20:])))
    validate_split(groups, scenes)
    return groups


def validate_split(groups, scenes):
    if set(groups) != set(ROLES) or any(not groups[r] for r in ROLES):
        raise ValueError("Four nonempty data roles required")
    flattened = [x for role in ROLES for x in groups[role]]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(scenes):
        raise ValueError("Scene overlap or incomplete split")


def require_human(metadata, other_codes=()):
    if (metadata.get("reviewer_type") != "human"
            or metadata.get("human_attestation") is not True
            or metadata.get("independent_attestation") is not True
            or metadata.get("no_model_assistance_attestation") is not True):
        raise ValueError("Independent human declaration required; model labels are not gold")
    code = metadata.get("reviewer_code", "").strip()
    if not code or code in other_codes:
        raise ValueError("Distinct real reviewer codes required")
    try:
        completed = datetime.fromisoformat(metadata["completed_at"].replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Valid completion timestamp required") from error
    if completed.tzinfo is None or completed > datetime.now(timezone.utc):
        raise ValueError("Timezone required and completion cannot be in the future")
    return code


def event_symbol(event, contract):
    if (event["kind"] != "carrier" or event["explicit_status"] != "yes"
            or event.get("readability") != "clear"):
        return None
    return classify_facts(event, contract).symbol


def validate_inventory(text, contract, events, status):
    """Unknown is explicit; failure to match a contract never implies background."""
    if status not in ("complete", "unknown", "unavailable") or not isinstance(events, list):
        raise ValueError("Inventory status and explicit event array required")
    if status == "unavailable" and (text or events):
        raise ValueError("Only genuinely missing text is unavailable")
    seen = set()
    for event in events:
        if not isinstance(event, dict) or not event.get("event_id"):
            raise ValueError("Every event requires a local id")
        if event["event_id"] in seen:
            raise ValueError("Duplicate event id")
        seen.add(event["event_id"])
        if event.get("kind") not in ("carrier", "background", "unknown"):
            raise ValueError("Inventory events cannot be labelled duplicate")
        if not event.get("meaning", "").strip() or not event.get("notes", "").strip():
            raise ValueError("Meaning and notes required")
        if not validate_evidence(text, [event.get("anchor")]):
            raise ValueError("Literal event anchor required")
        evidence = event.get("evidence")
        if not isinstance(evidence, dict) or set(evidence) != set(FACT_FIELDS):
            raise ValueError("Four fact evidence fields required")
        for field in FACT_FIELDS:
            value = event.get(field)
            spans = evidence[field]
            if not isinstance(value, str) or not value.strip() or not isinstance(spans, list):
                raise ValueError("Fact value and evidence array required")
            if value == "unknown":
                if spans:
                    raise ValueError("Unknown fact cannot have asserted fact evidence")
            elif not validate_evidence(text, spans):
                raise ValueError("Known fact requires literal evidence")
        if event.get("transfer_status") not in ("completed", "planned_not_occurred", "other", "unknown"):
            raise ValueError("Invalid occurrence state")
        if event.get("explicit_status") not in ("yes", "no", "unknown"):
            raise ValueError("Explicit-state judgement required")
        if event.get("readability") not in ("clear", "unclear", "contradictory"):
            raise ValueError("Readability judgement required")
        if event["readability"] == "clear" and any(event[k] == "unknown" for k in FACT_FIELDS):
            raise ValueError("Clear event cannot hide unknown facts")
        if event["kind"] == "carrier" and event_symbol(event, contract) is None:
            raise ValueError("Carrier requires a complete actual contract fact")
        if event["kind"] == "background":
            if not event.get("background_reason", "").strip():
                raise ValueError("Background requires a positive non-contract explanation")
            if classify_facts(event, contract).symbol is not None:
                raise ValueError("Complete contract fact cannot be relabelled background")
    return events


def validate_candidate(row, text, events):
    kind = row.get("kind")
    if kind not in KINDS or not row.get("notes", "").strip():
        raise ValueError("Candidate kind and reason required")
    links = row.get("event_ids")
    if not isinstance(links, list) or len(links) != len(set(links)):
        raise ValueError("Unique event-id array required")
    bank = {e["event_id"]: e for e in events}
    if any(link not in bank for link in links):
        raise ValueError("Candidate references an unknown event")
    if not validate_evidence(text, row.get("evidence")):
        raise ValueError("Candidate linkage needs literal text evidence")
    if kind == "unknown":
        return row
    if len(links) != 1:
        raise ValueError("Known candidate must correspond to one event")
    event = bank[links[0]]
    if kind == "background" and event["kind"] != "background":
        raise ValueError("Unmatched or uncertain event is not background")
    if kind in ("carrier", "duplicate") and event["kind"] != "carrier":
        raise ValueError("Carrier/duplicate requires a verified carrier event")
    a, b = row["anchor_span"]
    if not any(s["start"] <= a < b <= s["end"] for s in row["evidence"]):
        raise ValueError("Linkage evidence must include this candidate anchor")
    if kind == "carrier" and [event["anchor"]["start"], event["anchor"]["end"]] != [a, b]:
        raise ValueError("Canonical carrier must match the independently listed primary anchor")
    return row


def normalize_document(payload, fact, candidates):
    events = fact["events"]
    validate_inventory(payload["text"], payload["contract"], events, fact["inventory_status"])
    if len(candidates) != len(payload["candidates"]):
        raise ValueError("All shared candidates must be labelled, including unknown")
    canonical, rows = {}, []
    for i, (anchor, label) in enumerate(zip(payload["candidates"], candidates)):
        if label["anchor_span"] != anchor["anchor_span"]:
            raise ValueError("Candidate order/anchor edited")
        validate_candidate(label, payload["text"], events)
        symbol = None
        if label["kind"] in ("carrier", "duplicate"):
            event = next(e for e in events if e["event_id"] == label["event_ids"][0])
            symbol = event_symbol(event, payload["contract"])
            if label["kind"] == "carrier":
                if event["event_id"] in canonical:
                    raise ValueError("Two canonical observations for one event")
                canonical[event["event_id"]] = i
        rows.append({"index": i, "kind": label["kind"], "symbol": symbol,
                     "event_ids": label["event_ids"]})
    refs = [{"event_id": e["event_id"], "symbol": event_symbol(e, payload["contract"]),
             "candidate_index": canonical.get(e["event_id"])}
            for e in events if e["kind"] == "carrier"]
    missing = sum(r["candidate_index"] is None for r in refs)
    return {"candidate_labels": rows, "references": refs, "missing_events": missing,
            "inventory_status": fact["inventory_status"],
            "unknown_events": sum(e["kind"] == "unknown" for e in events)}


def identity_metrics(documents):
    """All assigned carrier slots and extra outputs stay in F1 denominators."""
    tp, fp, fn = [0] * 4, [0] * 4, [0] * 4
    expected = emitted = background = unknown = duplicates = missed = 0
    for doc in documents:
        predictions, rows = doc["predictions"], doc["candidate_labels"]
        if len(predictions) != len(rows):
            raise ValueError("Predictions must retain every candidate")
        expected += len(doc["references"])
        missed += doc["missing_events"]
        target = doc["planned_symbol"]
        if target not in "ABCD" or len(target) != 1:
            raise ValueError("One original allocated target slot required")
        c = "ABCD".index(target)
        correct = {r["candidate_index"] for r in doc["references"]
                   if r["symbol"] == target and r["candidate_index"] is not None
                   and predictions[r["candidate_index"]] == target}
        # Pilot contract is one allocated slot; extra same-class events remain FP.
        credited = min(correct) if correct else None
        tp[c] += int(credited is not None)
        fn[c] += int(credited is None)
        for i, prediction in enumerate(predictions):
            if prediction is None:
                continue
            if prediction not in "ABCD" or len(prediction) != 1:
                raise ValueError("Invalid emitted symbol")
            emitted += 1
            if i != credited:
                fp["ABCD".index(prediction)] += 1
            background += rows[i]["kind"] == "background"
            unknown += rows[i]["kind"] == "unknown"
            duplicates += rows[i]["kind"] == "duplicate"
    f1 = [2 * tp[c] / max(2 * tp[c] + fp[c] + fn[c], 1) for c in range(4)]
    return {"macro_f1": sum(f1) / 4, "tp": tp, "fp": fp, "fn": fn,
            "assigned_slots": len(documents), "actual_carriers": expected,
            "missing_events": missed, "emitted": emitted,
            "background_emissions": background, "unknown_emissions": unknown,
            "duplicate_emissions": duplicates,
            "slot_success_rate": sum(tp) / max(len(documents), 1)}


def choose_epoch(curve):
    """Threshold-free validation identity first, then fewer extras, then epoch."""
    if not curve:
        raise ValueError("Validation selection curve required")
    return max(curve, key=lambda x: (x["metrics"]["macro_f1"],
                                    -sum(x["metrics"]["fp"]), -x["epoch"]))


def calibrate_rejection(records, predict, thresholds):
    """Fixed checkpoint; calibration may choose thresholds only."""
    if not records or any(r.get("role") != "calibration" for r in records):
        raise ValueError("Only a nonempty independent calibration population is allowed")
    grid = list(thresholds)
    if not grid or len(grid) > 21:
        raise ValueError("Rejection grid must share the declared 24-configuration budget")
    curve = []
    for presence, confidence in grid:
        if not all(math.isfinite(x) and 0 <= x <= 1.01 for x in (presence, confidence)):
            raise ValueError("Invalid rejection threshold")
        docs = [{**r, "predictions": predict(r, presence, confidence)} for r in records]
        curve.append({"presence": presence, "confidence": confidence,
                      "metrics": identity_metrics(docs)})
    chosen = max(curve, key=lambda x: (x["metrics"]["macro_f1"],
                                      -sum(x["metrics"]["fp"]),
                                      x["presence"], x["confidence"]))
    return {"selection": chosen, "curve": curve,
            "role": "calibration", "checkpoint_updated": False,
            "empirical_only": True, "error_rate_guarantee": False}
