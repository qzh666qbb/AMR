#!/usr/bin/env python3
"""No-compute closure audit for the source-anchored pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from swan_anchor1_budget import Journal
from swan_anchor1_common import OUT, digest, load, sha256, single_writer


ROOT = Path(__file__).resolve().parents[1]
CLOSURE = OUT / "pilot_closure"
MANIFEST_SHA256 = "3fa20cc2b23a6ed0d7faafd5929e1d378f15346b6966a8af10d60c8f92cb6427"
EXPECTED_UNITS = {
    "anchor1-pilot-0000": 6,
    "anchor1-pilot-0001": 10,
    "anchor1-pilot-0002": 10,
    "anchor1-pilot-0003": 2,
    "anchor1-pilot-0004": 10,
    "anchor1-pilot-0005": 2,
    "anchor1-pilot-0006": 10,
    "anchor1-pilot-0007": 10,
}
WARNING_CONCEPTS = (
    "cause-01",
    "more",
    "ordinal-entity",
    "name",
    "temporal-quantity",
)


def json_bytes(value):
    return (
        json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def frozen_bytes(path, expected):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != expected:
            raise ValueError(f"Frozen artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="freeze-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(expected)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def bytes_sha256(value):
    return hashlib.sha256(value).hexdigest()


def relative(path):
    return str(Path(path).relative_to(ROOT))


def aggregate(paths):
    hashes = {relative(path): sha256(path) for path in sorted(paths)}
    return {"count": len(hashes), "aggregate_sha256": digest(hashes)}


def trace_path(allocation_id, slot, attempt):
    return OUT / "private/pilot/generation_attempts" / (
        f"{allocation_id}-slot-{slot:02d}-attempt-{attempt:02d}.json"
    )


def verify_trace_chain(allocation_id, slot, expected_hashes, journal, call_ids):
    traces = []
    for attempt, expected_hash in enumerate(expected_hashes):
        path = trace_path(allocation_id, slot, attempt)
        if sha256(path) != expected_hash:
            raise ValueError(f"Attempt trace hash changed: {path.name}")
        trace = load(path)
        call_id = f"{allocation_id}-slot-{slot:02d}-attempt-{attempt:02d}"
        if trace["call_id"] != call_id or trace["attempt"] != attempt:
            raise ValueError(f"Attempt identity changed: {path.name}")
        call = journal.calls.get(call_id)
        if not call or not call["terminal"] \
                or call["terminal"]["kind"] != "complete" \
                or trace["call_sha256"] != call["terminal"]["response_sha256"]:
            raise ValueError(f"Attempt journal linkage changed: {path.name}")
        if trace["accepted"]:
            raise ValueError(f"Unexpected accepted attempt: {path.name}")
        if trace["surface"]["pass"]:
            if trace["status"] != "complete" \
                    or not isinstance(trace["score"], (int, float)) \
                    or not 0 <= trace["score"] < .75:
                raise ValueError(f"Surface-qualified score changed: {path.name}")
        elif trace["status"] != "surface_rejected" \
                or trace["score"] is not None:
            raise ValueError(f"Surface rejection changed: {path.name}")
        call_ids.add(call_id)
        traces.append(trace)
    return traces


def build():
    if sha256(OUT / "manifest.json") != MANIFEST_SHA256:
        raise ValueError("Frozen manifest hash changed")

    journal = Journal(
        OUT, "pilot", 1, None, MANIFEST_SHA256, allow_halted=True
    )
    accounting = journal.totals()
    document_paths = sorted((OUT / "documents/pilot").glob("*.json"))
    documents = [load(path) for path in document_paths]
    ids = [row["allocation_id"] for row in documents]
    if ids != list(EXPECTED_UNITS):
        raise ValueError("Pilot allocation identities or order changed")

    call_ids = set()
    rows = []
    completed_slots = 0
    fallback_slots = 0
    failed_slots = 0
    surface_rejected_attempts = 0
    surface_qualified_attempts = 0

    for document in documents:
        allocation_id = document["allocation_id"]
        units = EXPECTED_UNITS[allocation_id]
        expected_complete = units == 10
        expected_reason = (
            "ten_slots_complete"
            if expected_complete else "no_surface_eligible_candidate"
        )
        if document["manifest_sha256"] != MANIFEST_SHA256 \
                or document["num_detection_units"] != units \
                or len(document["slot_sha256"]) != units \
                or document["generation_complete"] != expected_complete \
                or document["structural_pass"] != expected_complete \
                or document["termination"] != expected_reason:
            raise ValueError(f"Document decision changed: {allocation_id}")

        for slot, expected_slot_hash in enumerate(document["slot_sha256"]):
            path = OUT / "private/pilot/generation_slots" / (
                f"{allocation_id}-slot-{slot:02d}.json"
            )
            if sha256(path) != expected_slot_hash:
                raise ValueError(f"Document-to-slot hash changed: {path.name}")
            value = load(path)
            if value["allocation_id"] != allocation_id \
                    or value["slot"] != slot \
                    or value["attempts"] != 30 \
                    or value["status"] != "fallback" \
                    or value["accepted"] \
                    or not value["fallback"] \
                    or len(value["attempt_trace_sha256"]) != 30:
                raise ValueError(f"Completed slot decision changed: {path.name}")
            traces = verify_trace_chain(
                allocation_id,
                slot,
                value["attempt_trace_sha256"],
                journal,
                call_ids,
            )
            eligible = [trace for trace in traces if trace["surface"]["pass"]]
            if not eligible:
                raise ValueError(f"Fallback lacks an eligible candidate: {path.name}")
            selected = min(
                eligible, key=lambda row: (-row["score"], row["attempt"])
            )
            if value["selected_attempt"] != selected["attempt"] \
                    or value["score"] != selected["score"]:
                raise ValueError(f"Fallback selection changed: {path.name}")
            completed_slots += 1
            fallback_slots += 1
            surface_rejected_attempts += sum(
                not trace["surface"]["pass"] for trace in traces
            )
            surface_qualified_attempts += sum(
                trace["surface"]["pass"] for trace in traces
            )

        if not expected_complete:
            slot = units
            path = OUT / "private/pilot/generation_failures" / (
                f"{allocation_id}-slot-{slot:02d}.json"
            )
            failure = load(path)
            if failure["allocation_id"] != allocation_id \
                    or failure["slot"] != slot \
                    or failure["reason"] != "no_surface_eligible_candidate" \
                    or failure["attempts"] != 30 \
                    or len(failure["attempt_trace_sha256"]) != 30:
                raise ValueError(f"Failed slot decision changed: {path.name}")
            traces = verify_trace_chain(
                allocation_id,
                slot,
                failure["attempt_trace_sha256"],
                journal,
                call_ids,
            )
            if any(trace["surface"]["pass"] for trace in traces):
                raise ValueError(f"Failed slot has an eligible candidate: {path.name}")
            failed_slots += 1
            surface_rejected_attempts += 30

        rows.append({
            "allocation_id": allocation_id,
            "termination": document["termination"],
            "generation_complete": document["generation_complete"],
            "generated_slots": units,
            "num_detection_units": document["num_detection_units"],
            "structural_pass": document["structural_pass"],
            "identity_pass": document["identity_pass"],
            "all_selected_surface_pass":
                document["all_selected_surface_pass"],
        })

    probe = journal.calls.get("anchor1-pilot-model-probe")
    if not probe or not probe["terminal"] \
            or probe["terminal"]["kind"] != "complete":
        raise ValueError("Model probe is not terminal")
    expected_call_ids = call_ids | {"anchor1-pilot-model-probe"}
    if set(journal.calls) != expected_call_ids:
        raise ValueError("Journal call identities differ from attempt provenance")

    response_paths = sorted((OUT / "private/pilot/responses").glob("*.json"))
    counts = {
        "allocated": len(documents),
        "terminal_documents": len(documents),
        "generation_complete": sum(row["generation_complete"] for row in rows),
        "structural_pass": sum(row["structural_pass"] for row in rows),
        "structural_fail": sum(not row["structural_pass"] for row in rows),
        "completed_slots": completed_slots,
        "accepted_slots": completed_slots - fallback_slots,
        "fallback_slots": fallback_slots,
        "failed_slots": failed_slots,
        "generation_attempts": len(call_ids),
        "surface_qualified_attempts": surface_qualified_attempts,
        "surface_rejected_attempts": surface_rejected_attempts,
        "responses": len(response_paths),
        "baseline_reviewed": 0,
        "transformed_allocations": 0,
        "pair_reviewed": 0,
        "detection_units_evaluated": 0,
    }
    expected_counts = {
        "allocated": 8,
        "terminal_documents": 8,
        "generation_complete": 5,
        "structural_pass": 5,
        "structural_fail": 3,
        "completed_slots": 60,
        "accepted_slots": 0,
        "fallback_slots": 60,
        "failed_slots": 3,
        "generation_attempts": 1890,
        "responses": 1891,
    }
    if any(counts[key] != value for key, value in expected_counts.items()):
        raise ValueError("Pilot attrition counts changed")
    if surface_qualified_attempts + surface_rejected_attempts != 1890:
        raise ValueError("Attempt surface accounting changed")

    gate = {
        "counts": {
            "allocated": 8,
            "structure_pass": 5,
            "baseline_quality_pass": None,
            "C2_constructed": None,
            "C2_quality_pass": None,
            "common_quality_pairs": None,
        },
        "gates": {"structural_upper_bound": False},
        "passed": False,
        "reason": "Too few structurally eligible documents",
        "branch": "C",
        "unperformed_quality_is_pending_not_failed": True,
    }

    absent = [
        OUT / "authorization_main.json",
        OUT / "private/main/journal.jsonl",
        OUT / "transforms/pilot.json",
        OUT / "quality/pilot/baseline/decision.json",
        OUT / "quality/pilot/pair/decision.json",
        OUT / "detection_plan.json",
        OUT / "results.json",
    ]
    if any(path.exists() for path in absent) \
            or any((OUT / "documents/main").glob("*.json")):
        raise ValueError("A prohibited post-pilot or main artifact exists")

    fixed_paths = [
        OUT / "manifest.json",
        OUT / "source_manifest.json",
        OUT / "environment_lock.json",
        OUT / "budget_estimate.json",
        OUT / "authorization_pilot.json",
        OUT / "private/pilot/journal.jsonl",
        OUT / "PILOT_CONCLUSION.md",
        Path(__file__).resolve(),
        *sorted((OUT / "price_checks").glob("*.json")),
    ]
    fixed_hashes = {relative(path): sha256(path) for path in fixed_paths}
    groups = {
        "documents": aggregate(document_paths),
        "slot_starts": aggregate(
            (OUT / "private/pilot/slot_starts").glob("*.json")
        ),
        "generation_slots": aggregate(
            (OUT / "private/pilot/generation_slots").glob("*.json")
        ),
        "generation_failures": aggregate(
            (OUT / "private/pilot/generation_failures").glob("*.json")
        ),
        "generation_attempts": aggregate(
            (OUT / "private/pilot/generation_attempts").glob("*.json")
        ),
        "responses": aggregate(response_paths),
    }
    if {key: value["count"] for key, value in groups.items()} != {
        "documents": 8,
        "slot_starts": 63,
        "generation_slots": 60,
        "generation_failures": 3,
        "generation_attempts": 1890,
        "responses": 1891,
    }:
        raise ValueError("Artifact inventory counts changed")

    conclusion = (OUT / "PILOT_CONCLUSION.md").read_text(encoding="utf-8")
    warnings_disclosed = all(value in conclusion for value in WARNING_CONCEPTS)
    parser_warning_stop_resolved = False
    checks = {
        "manifest_sha256_verified": True,
        "journal_chain_and_accounting_verified":
            accounting["calls"] == 1891
            and accounting["actual_peak_nanodollars"] == 491967300
            and accounting["exposure_nanodollars"] == 491967300
            and accounting["uncertain_or_invalid_calls"] == 0,
        "all_eight_allocations_terminal": len(documents) == 8,
        "attempt_response_slot_document_chains_verified": True,
        "structural_attrition_is_5_of_8":
            counts["structural_pass"] == 5
            and counts["structural_fail"] == 3,
        "all_completed_slots_are_fallbacks":
            counts["fallback_slots"] == 60
            and counts["accepted_slots"] == 0,
        "pilot_gate_failed_before_quality": gate["passed"] is False,
        "quality_is_unperformed_not_failed": True,
        "main_and_downstream_artifacts_absent": True,
        "parser_warning_limitation_disclosed": warnings_disclosed,
        "parser_warning_stop_resolved": parser_warning_stop_resolved,
    }
    required = {
        key: value
        for key, value in checks.items()
        if key != "parser_warning_stop_resolved"
    }
    if not all(required.values()):
        failed = [key for key, value in required.items() if not value]
        raise ValueError(f"Pilot closure checks failed: {failed}")

    attrition = {
        "schema_version": 1,
        "experiment": "swan_source_anchored1",
        "phase": "pilot",
        "documents": rows,
        "counts": counts,
    }
    artifact_hashes = {
        "schema_version": 1,
        "experiment": "swan_source_anchored1",
        "phase": "pilot",
        "fixed_inputs": fixed_hashes,
        "artifact_groups": groups,
        "stable_aggregate_sha256": digest({
            "fixed_inputs": fixed_hashes,
            "artifact_groups": groups,
        }),
    }
    attrition_raw = json_bytes(attrition)
    artifacts_raw = json_bytes(artifact_hashes)
    audit = {
        "schema_version": 1,
        "experiment": "swan_source_anchored1",
        "phase": "pilot",
        "status": "closure_verified_with_unresolved_parser_warning_stop",
        "decision": "branch_C_stop_source_anchored_route",
        "pilot_passed": False,
        "failure_reason": "structural_eligibility_5_of_8",
        "checks": checks,
        "blocking_findings": [{
            "code": "parser_warning_stop_unresolved",
            "source": "operator_observed_stderr_not_machine_recoverable",
            "concepts": list(WARNING_CONCEPTS),
            "effect": "main_remains_prohibited",
        }],
        "gate": gate,
        "accounting": accounting,
        "manifest_sha256": MANIFEST_SHA256,
        "attrition_sha256": bytes_sha256(attrition_raw),
        "artifact_hashes_sha256": bytes_sha256(artifacts_raw),
        "new_api_calls": 0,
        "new_neural_parser_calls": 0,
        "new_s2match_calls": 0,
        "new_detector_calls": 0,
        "confirmatory_attack_tests_performed": False,
    }
    audit_raw = json_bytes(audit)
    report = """# source-anchored Pilot Closure Audit

Status: **Branch C closure verified; parser-warning stop remains unresolved.**

The no-compute replay linked 1,890 generation attempts to 1,891 terminal
responses (including the model probe), 60 completed slots, three failed slots,
and eight terminal documents. Five of eight documents passed structure, below
the frozen 6/8 gate. All 60 completed slots were deterministic fallbacks after
30 attempts; no slot reached the 0.75 acceptance threshold.

Quality review was not performed and is not counted as failed. No quality
decision, transform, detection result, or main authorization exists. The
replay made zero API, neural parser, S2Match, or detector calls.

The run conclusion discloses parser diagnostics observed on stderr. Those
messages were not persisted by the runner and therefore cannot be independently
reconstructed from saved artifacts. Under `STOP_RULES.md`, this remains an
unresolved audit stop and independently prohibits main execution.
"""
    report_raw = report.encode("utf-8")
    recovery = {
        "schema_version": 1,
        "status": "complete_no_compute_pilot_closure",
        "decision": "branch_C_stop_source_anchored_route",
        "files_checked": sum(value["count"] for value in groups.values())
            + len(fixed_hashes),
        "stable_aggregate_sha256": artifact_hashes["stable_aggregate_sha256"],
        "network_forbidden": True,
        "main_source_content_forbidden": True,
        "credential_files_forbidden": True,
        "neural_amr_parser_forbidden": True,
        "s2match_forbidden": True,
        "detector_forbidden": True,
        "new_api_calls": 0,
        "new_neural_parser_calls": 0,
        "new_s2match_calls": 0,
        "new_detector_calls": 0,
        "attrition_sha256": bytes_sha256(attrition_raw),
        "artifact_hashes_sha256": bytes_sha256(artifacts_raw),
        "audit_sha256": bytes_sha256(audit_raw),
        "audit_report_sha256": bytes_sha256(report_raw),
    }
    return {
        "attrition.json": attrition_raw,
        "artifact_hashes.json": artifacts_raw,
        "audit.json": audit_raw,
        "audit_report.md": report_raw,
        "recovery_verification.json": json_bytes(recovery),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="require closure artifacts to exist and match the no-compute replay",
    )
    args = parser.parse_args()
    with single_writer(OUT):
        outputs = build()
        if args.verify_only:
            missing = [name for name in outputs if not (CLOSURE / name).exists()]
            if missing:
                raise FileNotFoundError(f"Missing closure artifacts: {missing}")
        for name, value in outputs.items():
            frozen_bytes(CLOSURE / name, value)
    print(json.dumps({
        "status": "verified" if args.verify_only else "frozen",
        "output_files": sorted(outputs),
        "new_api_calls": 0,
        "new_neural_parser_calls": 0,
        "new_s2match_calls": 0,
        "new_detector_calls": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
