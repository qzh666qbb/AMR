"""Collect attested human graph/text reviews without manufacturing annotations."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core.semantic_fact_carrier import classify_facts, validate_evidence
from src.core.semantic_fact_layers import FIELDS, actual_signature, compare_fact_layers
from src.core.semantic_graph_fact_reader import read_graph_facts


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def timestamp(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result if result.tzinfo else None
    except (ValueError, AttributeError):
        return None


def graph_triples(penman_text):
    import logging
    import penman
    from penman.models.amr import model
    messages = []

    class Capture(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    logger = logging.getLogger("penman")
    saved = (logger.level, logger.propagate)
    handler = Capture(logging.WARNING)
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    try:
        literal = "\n".join(l for l in penman_text.splitlines() if not l.lstrip().startswith("#"))
        graph = penman.decode(literal, model=model)
        roundtrip = penman.decode(penman.encode(graph, model=model), model=model)
        if graph.top != roundtrip.top or Counter(graph.triples) != Counter(roundtrip.triples):
            raise ValueError("Graph roundtrip differs")
        errors = model.errors(graph)
        if errors:
            raise ValueError("AMR structural errors: " + repr(errors))
        if messages:
            raise ValueError("PENMAN warning: " + "; ".join(messages))
        return {tuple(t) for t in graph.triples}
    finally:
        logger.removeHandler(handler)
        logger.setLevel(saved[0])
        logger.propagate = saved[1]


def facts_key(facts):
    """Keep unresolved fields and semantic qualifiers in disagreement checks."""
    keys = (*FIELDS, "readability", "explicit_status", "contradiction")
    return tuple(" ".join(v.split()).casefold() if isinstance(v := facts.get(k), str)
                 else v for k in keys)


def identity_errors(meta, adjudicator=False):
    errors = []
    required = ["human_attestation", "independent_attestation",
                "no_model_or_target_answers_used", "raw_graph_read_without_text_answers"]
    if adjudicator:
        required = ["human_attestation", "no_model_or_target_answers_used"]
    if any(meta.get(k) is not True for k in required):
        errors.append("missing_real_human_or_independence_attestations")
    if not isinstance(meta.get("reviewer_code"), str) or not meta["reviewer_code"].strip():
        errors.append("missing_reviewer_code")
    done = timestamp(meta.get("completed_at", ""))
    if not done or done > datetime.now(timezone.utc):
        errors.append("invalid_completion_time")
    if not adjudicator:
        raw = timestamp(meta.get("raw_graph_completed_at", ""))
        text = timestamp(meta.get("text_completed_at", ""))
        if not raw or not text or not done or not raw <= text <= done:
            errors.append("raw_graph_phase_must_precede_text_phase")
    return errors


def validate_answer(answer, source, phase):
    if answer.get("reviewed") is not True:
        return ["not_reviewed"]
    errors = []
    for key in ("case_id", "input_text_sha256", "original_penman_sha256", "contract_sha256"):
        if answer.get(key) != source[key]:
            errors.append("identity_or_source_hash_mismatch:" + key)
    facts = answer.get("facts", {})
    if any(not isinstance(facts.get(f), str) or not facts[f].strip() for f in FIELDS):
        errors.append("missing_fact_fields")
    if facts.get("readability") not in {"clear", "unclear"}:
        errors.append("missing_readability")
    if facts.get("explicit_status") not in {"yes", "no", "unknown"}:
        errors.append("missing_explicit_status")
    if type(facts.get("contradiction")) is not bool:
        errors.append("missing_contradiction_flag")
    graph = None
    if phase != "text":
        literal = source["graph_only"] if phase == "raw_graph" else answer.get("revised_penman", "")
        if not isinstance(literal, str) or not literal.strip():
            errors.append("missing_revised_graph")
        else:
            try:
                graph = graph_triples(literal)
            except Exception as exc:
                # Bound warnings in original graphs were already excluded. They
                # remain reviewable, but a corrected graph must be warning-free.
                if phase == "raw_graph" and not source["structurally_eligible"]:
                    graph = {tuple(t) for t in source["normalized_triples"]}
                else:
                    errors.append("invalid_graph:" + str(exc))
    evidence = answer.get("evidence", {})
    for field in FIELDS:
        proof = evidence.get(field, {})
        unknown = facts.get(field) == "unknown"
        if unknown and not str(proof.get("uncertainty_reason", "")).strip():
            errors.append("unknown_requires_reason:" + field)
        if phase == "text":
            if not validate_evidence(source["text"], proof.get("text_spans")):
                errors.append("invalid_literal_text_evidence:" + field)
        elif graph is not None:
            triples = proof.get("triples")
            if (not isinstance(triples, list) or not triples
                    or any(not isinstance(t, list) or len(t) != 3
                           or tuple(t) not in graph for t in triples)):
                errors.append("invalid_graph_triple_evidence:" + field)
    if phase == "revision":
        if not validate_evidence(source["text"], answer.get("revision_text_evidence")):
            errors.append("invalid_revision_text_evidence")
        if not str(answer.get("revision_note", "")).strip():
            errors.append("missing_revision_rationale")
    return errors


def read_packet(folder, sources):
    folder = Path(folder)
    meta = load(folder / "reviewer.json")
    errors = identity_errors(meta)
    answers = {}
    for phase in ("raw_graph", "text", "revision"):
        rows = load(folder / (phase + "_answers.json"))
        phase_rows = {r.get("case_id"): r for r in rows}
        if len(rows) != len(sources) or len(phase_rows) != len(rows) or set(phase_rows) != set(sources):
            errors.append("denominator_or_identity_mismatch:" + phase)
        answers[phase] = phase_rows
    checked, row_errors = {}, {}
    for case_id, source in sources.items():
        failures = list(errors)
        for phase in answers:
            failures += [phase + ":" + e for e in validate_answer(
                answers[phase].get(case_id, {}), source, phase)]
        if failures:
            row_errors[case_id] = failures
        else:
            checked[case_id] = {phase: answers[phase][case_id] for phase in answers}
    return {"meta": meta, "errors": errors, "checked": checked, "row_errors": row_errors}


def collect(experiment, reviewer1, reviewer2, output, adjudicator=None):
    experiment, output = Path(experiment), Path(output)
    if output.exists():
        raise RuntimeError("Use a new collection directory; never overwrite collected evidence")
    freeze = load(experiment / "FREEZE.json")
    for relative, expected in freeze["code_sha256"].items():
        if file_sha(ROOT / relative) != expected:
            raise RuntimeError("Frozen readout/collector code drift:" + relative)
    private = experiment / "private_review"
    canonical_path = private / "FORMAL_CANONICAL_INPUTS.json"
    if file_sha(canonical_path) != load(experiment / "REVIEW_PREPARATION.json")["canonical_sha256"]:
        raise RuntimeError("Canonical review inputs changed")
    sources = {r["case_id"]: r for r in load(canonical_path)}
    for source in sources.values():
        if digest(source["text"]) != source["input_text_sha256"]:
            raise RuntimeError("Canonical literal text SHA differs")
        if digest(source["original_penman"]) != source["original_penman_sha256"]:
            raise RuntimeError("Canonical original graph SHA differs")
        if digest(json.dumps(source["contract"], sort_keys=True)) != source["contract_sha256"]:
            raise RuntimeError("Canonical public contract SHA differs")
    packets = [read_packet(reviewer1, sources), read_packet(reviewer2, sources)]
    codes = [p["meta"].get("reviewer_code", "").strip() for p in packets]
    distinct = bool(all(codes) and codes[0] != codes[1])
    adjudications, adjudicator_errors = {}, []
    if adjudicator:
        folder = Path(adjudicator)
        meta = load(folder / "reviewer.json")
        adjudicator_errors = identity_errors(meta, adjudicator=True)
        if meta.get("reviewer_code", "").strip() in codes:
            adjudicator_errors.append("adjudicator_not_distinct")
        rows = load(folder / "adjudications.json")
        adjudications = {r.get("case_id"): r for r in rows}
        if len(adjudications) != len(rows):
            adjudicator_errors.append("duplicate_adjudication")
    dev_rows = {r["case_id"]: r for r in load(private / "ROWS.json")}
    admin = {r["case_id"]: r for r in load(private / "ADMIN_PLANNED_SYMBOLS.json")}
    if file_sha(private / "ADMIN_PLANNED_SYMBOLS.json") != freeze["admin_plans_sha256"]:
        raise RuntimeError("Frozen planned-symbol mapping changed")
    if file_sha(private / "ROWS.json") != load(experiment / "RESULT.json")["rows_sha256"]:
        raise RuntimeError("Frozen development predictions changed")
    results, disputes = [], []
    for case_id, source in sources.items():
        result = {"case_id": case_id, "condition": source["condition"],
                  "formal_complete": False, "structurally_eligible": source["structurally_eligible"]}
        left, right = [p["checked"].get(case_id) for p in packets]
        if not distinct or not left or not right:
            result.update(status="PENDING_INDEPENDENT_HUMAN_EVIDENCE",
                          missing_or_invalid_evidence={str(i + 1): p["row_errors"].get(case_id, [])
                                                       for i, p in enumerate(packets)})
            results.append(result)
            continue
        disagreements = [phase for phase in left
                         if facts_key(left[phase]["facts"]) != facts_key(right[phase]["facts"])]
        chosen = left
        if disagreements:
            adjudication = adjudications.get(case_id, {})
            invalid = list(adjudicator_errors)
            if (adjudication.get("reviewed") is not True
                    or not adjudication.get("rationale", "").strip()):
                invalid.append("missing_adjudication_and_rationale")
            final = adjudication.get("resolution", {})
            for phase in ("raw_graph", "text", "revision"):
                invalid += [phase + ":" + e for e in validate_answer(final.get(phase, {}), source, phase)]
            if invalid:
                result.update(status="PENDING_THIRD_HUMAN_ADJUDICATION", disagreement_phases=disagreements)
                disputes.append({"case_id": case_id, "disagreement_phases": disagreements,
                                 "reviewer1": left, "reviewer2": right,
                                 "adjudication_errors": invalid})
                results.append(result)
                continue
            chosen = final
        # Different but fact-equivalent revisions are both checked by the reader;
        # retain differing behavior rather than arbitrarily select the easy one.
        graph_readbacks = [read_graph_facts(p["revision"]["revised_penman"], source["contract"])
                           for p in (left, right)] if not disagreements else [
                               read_graph_facts(chosen["revision"]["revised_penman"], source["contract"])]
        predictions = {p["prediction"] for p in graph_readbacks}
        if len(predictions) > 1:
            result.update(status="REVISION_READER_OUTPUT_VARIATION", revised_predictions=list(predictions))
            results.append(result)
            continue
        raw_prediction = dev_rows[case_id]["development"]["graph_only_prediction"]
        comparison = compare_fact_layers(
            contract=source["contract"], planned_symbol=admin[case_id]["planned_symbol"],
            text_facts=chosen["text"]["facts"], raw_graph_facts=chosen["raw_graph"]["facts"],
            revised_graph_facts=chosen["revision"]["facts"],
            raw_prediction=raw_prediction, revised_prediction=graph_readbacks[0]["prediction"],
            text_human_confirmed=True, raw_graph_facts_independent_confirmed=True,
            revision_independent_confirmed=True)
        # Frozen function's "classifier" includes graph reading. Refine that
        # combined label using independent graph facts and pure ABCD mapping.
        raw_fact_agreement = facts_key(dev_rows[case_id]["graph_readback"]["facts"]) == facts_key(
            chosen["raw_graph"]["facts"])
        verified_raw_symbol = classify_facts(chosen["raw_graph"]["facts"], source["contract"]).symbol
        refined = []
        text_signature = actual_signature(chosen["text"]["facts"])
        if text_signature is not None and actual_signature(chosen["raw_graph"]["facts"]) == text_signature:
            if not raw_fact_agreement:
                refined.append("raw_graph_fact_reader")
            if raw_fact_agreement and raw_prediction != verified_raw_symbol:
                refined.append("raw_ABCD_classifier")
        revised_fact_agreement = facts_key(graph_readbacks[0]["facts"]) == facts_key(
            chosen["revision"]["facts"])
        if text_signature is not None and actual_signature(chosen["revision"]["facts"]) == text_signature:
            if not revised_fact_agreement:
                refined.append("revised_graph_fact_reader")
            if revised_fact_agreement and graph_readbacks[0]["prediction"] != classify_facts(
                    chosen["revision"]["facts"], source["contract"]).symbol:
                refined.append("revised_ABCD_classifier")
        result.update(
            status=comparison["status"],
            formal_complete=comparison["status"] == "LAYER_COMPARISON_COMPLETE",
            comparison=comparison, raw_reader_matches_human_raw_facts=raw_fact_agreement,
            revised_reader_matches_human_revised_facts=revised_fact_agreement,
            refined_reader_classifier_layers=refined,
            revised_readbacks=graph_readbacks,
            independent_annotation_basis="Attestations and literal evidence checked; human identity/semantic entailment not mechanically proven",
            primary_metric_eligible=source["structurally_eligible"])
        results.append(result)
    output.mkdir(parents=True, mode=0o700)
    write_json(output / "ROWS.json", results)
    write_json(output / "DISAGREEMENTS.json", disputes)
    # A third reviewer receives evidence/answers, never planned/model symbols.
    write_json(output / "adjudications.json", [
        {"case_id": d["case_id"], "reviewed": False, "rationale": "", "resolution": {}}
        for d in disputes])
    write_json(output / "reviewer.json", {"reviewer_code": "", "completed_at": "",
                                         "human_attestation": False, "no_model_or_target_answers_used": False})
    complete = sum(r["formal_complete"] for r in results)
    summary = {
        "status": "FORMAL_COLLECTION_COMPLETE" if complete == len(sources) else "NOT_READY",
        "rows": len(sources), "formal_complete": complete,
        "completed_primary_eligible": sum(r["formal_complete"] and r["structurally_eligible"] for r in results),
        "valid_independent_review_rows": [len(p["checked"]) for p in packets],
        "distinct_reviewer_codes": distinct, "disagreement_rows": len(disputes),
        "status_counts": dict(Counter(r["status"] for r in results)),
        "full_denominator_preserved": True, "rejections_count_as_unconfirmed": True,
        "no_model_fill_or_forged_attestations": True,
        "freeze_sha256": file_sha(experiment / "FREEZE.json"),
        "review_submission_sha256": {
            str(i + 1): {name: file_sha(Path(folder) / name)
                         for name in ("reviewer.json", "raw_graph_answers.json", "text_answers.json", "revision_answers.json")}
            for i, folder in enumerate((reviewer1, reviewer2))},
    }
    write_json(output / "AUDIT.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=Path, required=True)
    ap.add_argument("--reviewer1", type=Path, required=True)
    ap.add_argument("--reviewer2", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--adjudicator", type=Path)
    ap.add_argument("--dependency-path", type=Path)
    args = ap.parse_args()
    if args.dependency_path:
        sys.path.insert(0, str(args.dependency_path))
    collect(args.experiment, args.reviewer1, args.reviewer2, args.out, args.adjudicator)
