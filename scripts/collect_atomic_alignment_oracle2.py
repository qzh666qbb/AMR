"""Immutable successor: complete independent pair reviews plus third adjudication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from collect_atomic_alignment_oracle1 import pair_labels, review_errors
from run_multiscale_transfer import atomic, read, sha
from src.core import atomic_alignment_arms as arms
from src.core.multiview_views import Event


def resolve(common_rows, packets, hashes, freeze_sha, seal_sha, adjudication=None):
    """The third reviewer may resolve exactly the disputed pairs, never omissions."""
    if len(packets) != 2 or len(hashes) != 2:
        raise ValueError("Exactly two original independent submissions required")
    errors = [review_errors(p, freeze_sha, seal_sha) for p in packets]
    if packets[0].get("reviewer_code") == packets[1].get("reviewer_code"):
        errors[0].append("reviewers_not_distinct")
    expected_rows = set(common_rows)
    mappings = []
    for index, packet in enumerate(packets):
        rows = packet.get("rows", [])
        mapping = {r.get("row_id"): r for r in rows}
        if len(mapping) != len(rows) or set(mapping) != expected_rows:
            errors[index].append("all_source_conditions_required")
        mappings.append(mapping)
    if any(errors):
        return {}, [], {"reviewer_errors": errors, "status": "PENDING_COMPLETE_REVIEWS"}
    merged, disputed = {}, {}
    for rid, common in common_rows.items():
        try:
            labels = [pair_labels(m[rid], common) for m in mappings]
        except ValueError as exc:
            errors[0].append(f"{rid}: {exc}")
            continue
        merged[rid] = {}
        for pair in common:
            key = pair["candidate_index"], pair["event_id"]
            if labels[0][key] == labels[1][key]:
                merged[rid][key] = labels[0][key]
            else:
                disputed.setdefault(rid, []).append(dict(pair))
    if any(errors):
        return {}, [], {"reviewer_errors": errors, "status": "PENDING_COMPLETE_REVIEWS"}
    disputes = [{"row_id": rid, "pairs": pairs} for rid, pairs in disputed.items()]
    if not disputed:
        if adjudication is not None:
            raise ValueError("Unexpected adjudication when no pair is disputed")
        return merged, [], {"status": "COMPLETE_SUPPORT", "adjudicated_pairs": 0}
    if adjudication is None:
        return {rid: labels for rid, labels in merged.items() if rid not in disputed}, disputes, {
            "status": "PENDING_ADJUDICATION", "adjudicated_pairs": 0}
    third_errors = review_errors(adjudication, freeze_sha, seal_sha)
    if adjudication.get("reviewer_code") in {p["reviewer_code"] for p in packets}:
        third_errors.append("adjudicator_not_independent_third_person")
    if adjudication.get("original_submission_sha256") != hashes:
        third_errors.append("adjudication_original_submissions_changed")
    entries = adjudication.get("rows", [])
    third_rows = {r.get("row_id"): r for r in entries}
    if len(third_rows) != len(entries) or set(third_rows) != set(disputed):
        third_errors.append("exact_disputed_rows_required")
    resolved = {}
    if not third_errors:
        for rid, common in disputed.items():
            try:
                resolved[rid] = pair_labels(third_rows[rid], common)
            except ValueError as exc:
                third_errors.append(f"{rid}: {exc}")
    if third_errors:
        return {rid: labels for rid, labels in merged.items() if rid not in disputed}, disputes, {
            "status": "PENDING_VALID_ADJUDICATION", "adjudicator_errors": third_errors}
    for rid, labels in resolved.items():
        merged[rid].update(labels)
    return merged, disputes, {"status": "COMPLETE_SUPPORT",
                             "adjudicated_pairs": sum(map(len, resolved.values()))}


def collect(experiment, submissions, out, adjudication=None):
    if out.exists():
        raise RuntimeError("Use a new immutable collection directory")
    f = read(experiment / "FREEZE.json")
    code_root = experiment / "private/code" if (experiment / "private/code").exists() else ROOT
    for path, value in f["code_sha256"].items():
        if sha(code_root / path) != value:
            raise RuntimeError("Frozen prediction dependency changed")
    observation_seal = (read(experiment / "OBSERVATION_SEAL.json")
                        if (experiment / "OBSERVATION_SEAL.json").exists() else f)
    if observation_seal is not f and observation_seal["freeze_sha256"] != sha(experiment / "FREEZE.json"):
        raise RuntimeError("Observation seal belongs to another freeze")
    for filename, field in (("BLIND_INPUTS.json", "blind_inputs_sha256"),
                            ("POST_SELECTION_SCORES.json", "post_selection_scores_sha256")):
        if sha(experiment / "private" / filename) != observation_seal[field]:
            raise RuntimeError("Frozen common observations changed")
    seal = read(experiment / "PREDICTION_SEAL.json")
    if seal["freeze_sha256"] != sha(experiment / "FREEZE.json") or seal[
            "predictions_sha256"] != sha(experiment / "private/PREDICTIONS.json"):
        raise RuntimeError("Prediction seal changed")
    scores = read(experiment / "private/POST_SELECTION_SCORES.json")
    common = {r["row_id"]: r["universe"] for r in scores}
    score_map = {r["row_id"]: r for r in scores}
    inputs = read(experiment / "private/BLIND_INPUTS.json")
    if len(common) != len(scores) or set(common) != {r["row_id"] for r in inputs}:
        raise RuntimeError("Frozen observation denominator changed")
    labels, disputes, status = resolve(
        common, [read(p) for p in submissions], [sha(p) for p in submissions],
        sha(experiment / "FREEZE.json"), sha(experiment / "PREDICTION_SEAL.json"),
        read(adjudication) if adjudication else None)
    rows = []
    for row in inputs:
        rid = row["row_id"]
        value = None
        if rid in labels:
            selection = arms.oracle(row["candidates"], [Event(**e) for e in row["events"]], labels[rid])
            value = {**selection, **arms.fixed_statistic(
                selection["selected"], score_map[rid]["observations"])}
        rows.append({"row_id": rid, "A2": value,
                     "status": "SUPPORT_ORACLE_COMPUTED" if value is not None else "PENDING_HUMANS"})
    out.mkdir(parents=True)
    atomic(out / "RESULT.json", {
        **status, "rows": rows, "total_rows": len(inputs),
        "completed_support_rows": sum(r["A2"] is not None for r in rows),
        "formal_FPR": None, "formal_TPR": None, "identity_F1": None,
        "full_fact_inventory_required_for_identity_endpoints": True,
        "human_identity_mechanically_verified": False,
    })
    atomic(out / "DISAGREEMENTS.json", disputes)
    atomic(out / "AUDIT.json", {
        "prediction_seal_sha256": sha(experiment / "PREDICTION_SEAL.json"),
        "freeze_sha256": sha(experiment / "FREEZE.json"),
        "submission_sha256": [sha(p) for p in submissions],
        "adjudication_sha256": sha(adjudication) if adjudication else None,
        "all_row_denominators_preserved": True,
        "selected_pairs_only_review_forbidden": True,
        "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (
            Path(__file__), ROOT / "scripts/collect_atomic_alignment_oracle1.py")},
    })
    return read(out / "RESULT.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--reviewer1", type=Path, required=True)
    parser.add_argument("--reviewer2", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.experiment, [args.reviewer1, args.reviewer2],
                             args.out, args.adjudication)))
