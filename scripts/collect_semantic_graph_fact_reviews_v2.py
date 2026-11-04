"""Collect independent human evidence against a separately sealed v2 reader.

Reuse the frozen literal-evidence and real-human attestation validators. No
assistant answer, intended symbol, or administrative plan is used as human gold.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import collect_semantic_graph_fact_reviews as validation
from src.core.semantic_fact_carrier import classify_facts
from src.core.semantic_fact_layers import actual_signature
from src.core.semantic_graph_fact_reader_v2 import VERSION, read_graph_facts


def bound_packet(folder, sources, binding):
    packet = validation.read_packet(folder, sources)
    if packet["meta"].get("review_binding_sha256") != binding:
        error = "missing_or_wrong_v2_review_binding"
        packet["errors"].append(error)
        packet["checked"] = {}
        for case_id in sources:
            packet["row_errors"].setdefault(case_id, []).append(error)
    return packet


def adjudicator_identity_errors(meta, reviewer_codes, binding):
    errors = validation.identity_errors(meta, adjudicator=True)
    if meta.get("independent_attestation") is not True:
        errors.append("missing_independent_adjudicator_attestation")
    if meta.get("reviewer_code", "").strip() in reviewer_codes:
        errors.append("adjudicator_not_distinct")
    if meta.get("review_binding_sha256") != binding:
        errors.append("missing_or_wrong_v2_review_binding")
    return errors


def collect(experiment, reviewer1, reviewer2, output, adjudicator=None):
    experiment, output = Path(experiment), Path(output)
    if output.exists():
        raise RuntimeError("Use a new collection directory")
    freeze = validation.load(experiment / "FREEZE.json")
    if freeze["reader_version"] != VERSION:
        raise RuntimeError("This collector accepts only the sealed v2 reader")
    for name, expected in freeze["code_sha256"].items():
        if validation.file_sha(ROOT / name) != expected:
            raise RuntimeError("Frozen v2 collection code drift: " + name)
    private = experiment / "private_review"
    for name, key in [
        ("FORMAL_CANONICAL_INPUTS.json", "canonical_sha256"),
        ("GRAPH_ONLY_READOUTS_V2.json", "graph_readouts_sha256"),
        ("FOCUS171_NO_ANSWERS.json", "focus171_sha256"),
    ]:
        if validation.file_sha(private / name) != freeze[key]:
            raise RuntimeError("Frozen v2 review source drift: " + name)
    canonical = validation.load(private / "FORMAL_CANONICAL_INPUTS.json")
    sources = {r["case_id"]: r for r in canonical}
    raw_rows = validation.load(private / "GRAPH_ONLY_READOUTS_V2.json")
    raw = {r["case_id"]: r["graph_readback"] for r in raw_rows}
    focus_rows = validation.load(private / "FOCUS171_NO_ANSWERS.json")
    focus = {r["case_id"] for r in focus_rows}
    if len(canonical) != len(sources) or len(sources) != 192 or set(raw) != set(sources):
        raise RuntimeError("Full frozen 192-row identity drift")
    if len(focus_rows) != 171 or len(focus) != 171 or not focus <= set(sources):
        raise RuntimeError("Frozen 171-row focus identity drift")
    for source in sources.values():
        if validation.digest(source["text"]) != source["input_text_sha256"]:
            raise RuntimeError("Original text SHA differs")
        if validation.digest(source["original_penman"]) != source["original_penman_sha256"]:
            raise RuntimeError("Original graph SHA differs")
        if validation.digest(json.dumps(source["contract"], sort_keys=True)) != source["contract_sha256"]:
            raise RuntimeError("Public contract SHA differs")
        result = raw[source["case_id"]]
        if result["version"] != VERSION or result["original_penman_sha256"] != source["original_penman_sha256"]:
            raise RuntimeError("Readout version or original-graph identity differs")
    packets = [bound_packet(folder, sources, freeze["review_binding_sha256"])
               for folder in (reviewer1, reviewer2)]
    codes = [p["meta"].get("reviewer_code", "").strip() for p in packets]
    distinct = bool(all(codes) and codes[0] != codes[1])
    adjudications, adjudicator_errors = {}, []
    if adjudicator:
        meta = validation.load(Path(adjudicator) / "reviewer.json")
        adjudicator_errors = adjudicator_identity_errors(meta, codes, freeze["review_binding_sha256"])
        records = validation.load(Path(adjudicator) / "adjudications.json")
        adjudications = {r.get("case_id"): r for r in records}
        if len(adjudications) != len(records):
            adjudicator_errors.append("duplicate_adjudication")
    results, disputes = [], []
    for case_id, source in sources.items():
        result = {
            "case_id": case_id, "condition": source["condition"], "focus171": case_id in focus,
            "structurally_eligible": source["structurally_eligible"],
            "independent_evidence_valid": False, "formal_complete": False,
            "original_restoration_verified": False, "revised_restoration_verified": False,
            "reader_version": VERSION,
        }
        left, right = [p["checked"].get(case_id) for p in packets]
        if not distinct or not left or not right:
            result.update(
                status="PENDING_INDEPENDENT_HUMAN_EVIDENCE",
                missing_or_invalid_evidence={str(i + 1): p["row_errors"].get(case_id, [])
                                            for i, p in enumerate(packets)})
            results.append(result)
            continue
        disagreements = [phase for phase in left
                         if validation.facts_key(left[phase]["facts"]) != validation.facts_key(right[phase]["facts"])]
        chosen = left
        if disagreements:
            adjudication = adjudications.get(case_id, {})
            errors = list(adjudicator_errors)
            if adjudication.get("reviewed") is not True or not adjudication.get("rationale", "").strip():
                errors.append("missing_adjudication_and_rationale")
            resolution = adjudication.get("resolution", {})
            for phase in ("raw_graph", "text", "revision"):
                errors += [phase + ":" + e for e in validation.validate_answer(
                    resolution.get(phase, {}), source, phase)]
            if errors:
                result.update(status="PENDING_THIRD_HUMAN_ADJUDICATION",
                              disagreement_phases=disagreements)
                disputes.append({"case_id": case_id, "disagreement_phases": disagreements,
                                 "reviewer1": left, "reviewer2": right, "errors": errors})
                results.append(result)
                continue
            chosen = resolution
        # Check every independently fact-equivalent revision, including harder
        # variants; a successful easier variant cannot substitute for another.
        revisions = [chosen] if disagreements else [left, right]
        readbacks = [read_graph_facts(r["revision"]["revised_penman"], source["contract"])
                     for r in revisions]
        text_facts = chosen["text"]["facts"]
        raw_facts = chosen["raw_graph"]["facts"]
        revised_facts = chosen["revision"]["facts"]
        text_sig = actual_signature(text_facts)
        raw_sig = actual_signature(raw_facts)
        revised_sig = actual_signature(revised_facts)
        raw_readback = raw[case_id]
        raw_reader_matches = validation.facts_key(raw_readback["facts"]) == validation.facts_key(raw_facts)
        revised_reader_matches = [
            validation.facts_key(b["facts"]) == validation.facts_key(revised_facts) for b in readbacks]
        text_symbol = classify_facts(text_facts, source["contract"]).symbol if text_sig else None
        layers = []
        if text_sig is not None and raw_sig is not None and text_sig != raw_sig:
            layers.append("original_graph_representation")
        if raw_sig is None:
            layers.append("original_graph_semantics_unresolved")
        if not raw_reader_matches:
            layers.append("raw_graph_fact_reader")
        if raw_reader_matches and raw_readback["prediction"] != classify_facts(raw_facts, source["contract"]).symbol:
            layers.append("raw_ABCD_classifier")
        if text_sig is not None and revised_sig is not None and text_sig != revised_sig:
            layers.append("revision_fact_preservation")
        if not all(revised_reader_matches):
            layers.append("revised_graph_fact_reader")
        formal_complete = all(sig is not None for sig in (text_sig, raw_sig, revised_sig))
        eligible = source["structurally_eligible"]
        original_restored = bool(
            formal_complete and eligible and raw_sig == text_sig and raw_reader_matches
            and text_symbol is not None and raw_readback["prediction"] == text_symbol)
        revised_restored = bool(
            formal_complete and eligible and revised_sig == text_sig and all(revised_reader_matches)
            and text_symbol is not None and all(b["prediction"] == text_symbol for b in readbacks))
        result.update(
            status="LAYER_COMPARISON_COMPLETE" if formal_complete else "INDEPENDENT_EVIDENCE_VALID_LAYER_UNRESOLVED",
            independent_evidence_valid=True, formal_complete=formal_complete,
            original_restoration_verified=original_restored, revised_restoration_verified=revised_restored,
            original_human_graph_matches_human_text=raw_sig == text_sig if text_sig else None,
            revised_human_graph_matches_human_text=revised_sig == text_sig if text_sig else None,
            raw_reader_matches_human_raw_facts=raw_reader_matches,
            revised_reader_matches_human_revised_facts=revised_reader_matches,
            refined_layers=layers, revised_readbacks=readbacks,
            revision_reader_output_variation=len({json.dumps(
                {"facts": b["facts"], "prediction": b["prediction"]}, sort_keys=True)
                for b in readbacks}) > 1,
            annotation_basis="Real-human attestations, phase order, literal evidence and two-reviewer agreement checked; identity and semantic entailment not mechanically proven",
        )
        results.append(result)
    output.mkdir(parents=True, mode=0o700)
    validation.write_json(output / "ROWS.json", results)
    validation.write_json(output / "DISAGREEMENTS.json", disputes)
    validation.write_json(output / "adjudications.json", [
        {"case_id": d["case_id"], "reviewed": False, "rationale": "", "resolution": {}}
        for d in disputes])
    validation.write_json(output / "reviewer.json", {
        "reviewer_code": "", "completed_at": "", "human_attestation": False,
        "independent_attestation": False,
        "no_model_or_target_answers_used": False,
        "review_binding_sha256": freeze["review_binding_sha256"]})
    focused = [r for r in results if r["focus171"]]

    def counts(rows):
        return {
            "denominator": len(rows),
            "independent_evidence_valid": sum(r["independent_evidence_valid"] for r in rows),
            "formal_complete": sum(r["formal_complete"] for r in rows),
            "original_restoration_verified": sum(r["original_restoration_verified"] for r in rows),
            "revised_restoration_verified": sum(r["revised_restoration_verified"] for r in rows),
            "structural_exclusions": sum(not r["structurally_eligible"] for r in rows),
            "status_counts": dict(Counter(r["status"] for r in rows)),
        }

    summary = {
        "status": "FORMAL_COLLECTION_COMPLETE" if all(r["formal_complete"] for r in results) else "NOT_READY",
        "reader_version": VERSION, "all192": counts(results), "focus171": counts(focused),
        "valid_independent_review_rows": [len(p["checked"]) for p in packets],
        "distinct_reviewer_codes": distinct, "disagreement_rows": len(disputes),
        "full_denominator_preserved": True, "no_assistant_or_model_fill": True,
        "administrative_plan_or_assistant_reference_accessed": False,
        "freeze_sha256": validation.file_sha(experiment / "FREEZE.json"),
        "submission_sha256": {str(i + 1): {
            name: validation.file_sha(Path(folder) / name)
            for name in ("reviewer.json", "raw_graph_answers.json", "text_answers.json", "revision_answers.json")}
            for i, folder in enumerate((reviewer1, reviewer2))},
    }
    validation.write_json(output / "AUDIT.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--reviewer1", type=Path, required=True)
    parser.add_argument("--reviewer2", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adjudicator", type=Path)
    parser.add_argument("--dependency-path", type=Path)
    args = parser.parse_args()
    if args.dependency_path:
        sys.path.insert(0, str(args.dependency_path))
    print(json.dumps(collect(args.experiment, args.reviewer1, args.reviewer2, args.out, args.adjudicator),
                     ensure_ascii=False))
