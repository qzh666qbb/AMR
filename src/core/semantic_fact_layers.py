"""Post-annotation failure attribution; no inference or target-guided graph edit."""
from __future__ import annotations

from typing import Mapping

from .semantic_fact_carrier import classify_facts

FIELDS = ("sender", "recipient", "object", "transfer_status")


def actual_signature(facts: Mapping):
    if facts.get("readability", "clear") != "clear":
        return None
    if facts.get("explicit_status", "yes") != "yes" or facts.get("contradiction") is True:
        return None
    values = [facts.get(field) for field in FIELDS]
    if any(not isinstance(value, str) or not value.strip() or value == "unknown" for value in values):
        return None
    return tuple(" ".join(value.split()).casefold() for value in values)


def compare_fact_layers(*, contract: Mapping, planned_symbol: str,
                        text_facts: Mapping, raw_graph_facts=None,
                        revised_graph_facts=None, raw_prediction=None,
                        revised_prediction=None, text_human_confirmed=False,
                        raw_graph_facts_independent_confirmed=False,
                        revision_independent_confirmed=False):
    """Compare against ACTUAL text facts; the plan only diagnoses writing.

    Missing graphs remain pending. Model candidates cannot certify a parser or
    classifier failure. A classifier failure requires verified graph facts equal
    to verified text facts, then an incorrect (or erased) symbol output.
    """
    text_sig = actual_signature(text_facts)
    text_symbol = classify_facts(text_facts, contract).symbol if text_sig else None
    result = {
        "text_symbol": text_symbol,
        "writing_matches_plan": text_symbol == planned_symbol if text_symbol else False,
        "text_human_confirmed": bool(text_human_confirmed),
        "raw_graph_facts_independent_confirmed": bool(raw_graph_facts_independent_confirmed),
        "revision_independent_confirmed": bool(revision_independent_confirmed),
        "raw_graph_available": raw_graph_facts is not None,
        "revised_graph_available": revised_graph_facts is not None,
        "raw_graph_matches_actual_text": None,
        "revised_graph_matches_actual_text": None,
        "raw_output_matches_actual_text": None,
        "revised_output_matches_actual_text": None,
        "supported_failure_layers": [],
        "status": "PENDING",
        "blind_message_recovery_claimed": False,
    }
    if not text_human_confirmed:
        result["status"] = "PENDING_TEXT_CONFIRMATION"
        return result
    if text_sig is None or text_symbol is None:
        result["status"] = "TEXT_UNRESOLVED"
        return result
    if text_symbol != planned_symbol:
        result["supported_failure_layers"].append("writing")
    if raw_graph_facts is None:
        result["status"] = "PENDING_RAW_GRAPH"
        return result
    raw_ok = actual_signature(raw_graph_facts) == text_sig
    result["raw_graph_matches_actual_text"] = raw_ok
    result["raw_output_matches_actual_text"] = raw_prediction == text_symbol
    if not raw_graph_facts_independent_confirmed:
        result["status"] = "PENDING_RAW_GRAPH_FACT_CONFIRMATION"
        return result
    if raw_ok and raw_prediction != text_symbol:
        result["supported_failure_layers"].append("raw_classifier")
    if revised_graph_facts is None or not revision_independent_confirmed:
        result["status"] = "PENDING_INDEPENDENT_REVISION"
        return result
    revision_ok = actual_signature(revised_graph_facts) == text_sig
    result["revised_graph_matches_actual_text"] = revision_ok
    result["revised_output_matches_actual_text"] = revised_prediction == text_symbol
    if not revision_ok:
        result["status"] = "REVISION_DISAGREES_WITH_ACTUAL_TEXT"
        return result
    if not raw_ok:
        result["supported_failure_layers"].append("parser_or_graph_representation")
    if revised_prediction != text_symbol:
        result["supported_failure_layers"].append("revised_classifier")
    result["status"] = "LAYER_COMPARISON_COMPLETE"
    return result
