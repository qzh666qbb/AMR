"""Collect two independent full-universe support reviews after A0/A1 sealing."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core import atomic_alignment_arms as arms
from src.core.multiview_views import Event
from run_multiscale_transfer import atomic, read, sha


def review_errors(packet, frozen_sha, prediction_sha):
    errors = []
    for name in ("human_attestation", "independent_attestation", "no_model_or_target_answers_used"):
        if packet.get(name) is not True:
            errors.append(name)
    if not isinstance(packet.get("reviewer_code"), str) or not packet["reviewer_code"].strip():
        errors.append("reviewer_code")
    for name, expected in (("freeze_sha256", frozen_sha), ("prediction_seal_sha256", prediction_sha)):
        if packet.get(name) != expected:
            errors.append(name)
    try:
        value = datetime.fromisoformat(packet["completed_at"])
        if value.utcoffset() is None:
            errors.append("timestamp_timezone_missing")
    except (KeyError, TypeError, ValueError):
        errors.append("completed_at")
    return errors


def pair_labels(row, common):
    expected = {(p["candidate_index"], p["event_id"]) for p in common}
    labels = {}
    for p in row.get("pairs", []):
        key = (p.get("candidate_index"), p.get("event_id"))
        if key in labels or key not in expected:
            raise ValueError("Duplicate or foreign Oracle pair")
        if type(p.get("same_fact_supported")) is not bool:
            raise ValueError("Unknown/unreviewed Oracle pair remains pending")
        if not isinstance(p.get("rationale"), str) or not p["rationale"].strip():
            raise ValueError("Every support judgement needs an independent rationale")
        labels[key] = p["same_fact_supported"]
    if set(labels) != expected:
        raise ValueError("Missing full-universe pair labels")
    return labels


def collect(experiment, submissions, out):
    if out.exists():
        raise RuntimeError("Use a new immutable Oracle collection directory")
    f = read(experiment / "FREEZE.json")
    for name, value in f["code_sha256"].items():
        if sha(ROOT / name) != value:
            raise RuntimeError("Frozen three-arm code drift")
    private = experiment / "private"
    for name, field in (("BLIND_INPUTS.json", "blind_inputs_sha256"),
                        ("POST_SELECTION_SCORES.json", "post_selection_scores_sha256")):
        if sha(private / name) != f[field]:
            raise RuntimeError("Frozen common observations changed")
    seal = read(experiment / "PREDICTION_SEAL.json")
    if seal["freeze_sha256"] != sha(experiment / "FREEZE.json") or seal[
            "predictions_sha256"] != sha(private / "PREDICTIONS.json"):
        raise RuntimeError("A0/A1 prediction seal changed")
    inputs = read(private / "BLIND_INPUTS.json")
    score_rows = {r["row_id"]: r for r in read(private / "POST_SELECTION_SCORES.json")}
    packets = [read(p) for p in submissions]
    errors = [review_errors(p, sha(experiment / "FREEZE.json"),
                            sha(experiment / "PREDICTION_SEAL.json")) for p in packets]
    if packets[0].get("reviewer_code") == packets[1].get("reviewer_code"):
        for e in errors:
            e.append("reviewers_not_distinct")
    mappings = []
    for i, packet in enumerate(packets):
        rows = packet.get("rows", [])
        mapping = {r.get("row_id"): r for r in rows}
        if len(mapping) != len(rows) or set(mapping) != {r["row_id"] for r in inputs}:
            errors[i].append("all_source_conditions_required")
        mappings.append(mapping)
    results, disputes = [], []
    for row in inputs:
        common = score_rows[row["row_id"]]["universe"]
        status = {"row_id": row["row_id"], "A2": None, "status": "PENDING_HUMAN_SUPPORT"}
        if not any(errors):
            try:
                labels = [pair_labels(m[row["row_id"]], common) for m in mappings]
            except ValueError as exc:
                status["reason"] = str(exc)
            else:
                different = [p for p in common if labels[0][(p["candidate_index"], p["event_id"])] !=
                             labels[1][(p["candidate_index"], p["event_id"])]]
                if different:
                    disputes.append({"row_id": row["row_id"], "pairs": different,
                                     "needs_independent_third_reviewer": True})
                    status["status"] = "PENDING_ADJUDICATION"
                else:
                    selection = arms.oracle(row["candidates"], [Event(**e) for e in row["events"]], labels[0])
                    status.update(status="ORACLE_SUPPORT_COMPUTED",
                                  A2={**selection, **arms.fixed_statistic(selection["selected"],
                                          score_rows[row["row_id"]]["observations"])})
        results.append(status)
    out.mkdir(parents=True)
    atomic(out / "RESULT.json", {
        "status": "SUPPORT_ORACLE_COMPUTED" if all(r["A2"] is not None for r in results) else "NOT_READY",
        "rows": results, "total_rows": len(inputs), "reviewer_errors": errors,
        "formal_FPR": None, "formal_TPR": None, "identity_F1": None,
        "reason_no_formal_metrics": "Source-isolated null calibration, human fact inventory and independent test still required",
        "human_identity_or_entailment_mechanically_proven": False,
    })
    atomic(out / "DISAGREEMENTS.json", disputes)
    atomic(out / "AUDIT.json", {
        "freeze_sha256": sha(experiment / "FREEZE.json"),
        "submission_sha256": [sha(p) for p in submissions], "all_row_denominators_preserved": True,
        "alignment_predictions_presealed": True, "Oracle_from_full_common_universe": True,
        "code_sha256": sha(Path(__file__)),
    })
    return read(out / "RESULT.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--reviewer1", type=Path, required=True)
    parser.add_argument("--reviewer2", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.experiment, [args.reviewer1, args.reviewer2], args.out)))
