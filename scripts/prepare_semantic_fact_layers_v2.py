"""Prepare target-free parser/revision inputs, without executing any inference."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_semantic_fact_carrier_pilot2 as v2
p = v2.runtime
OUT = v2.OUT
PRIVATE = OUT / "private_review/fact_layers"


def main():
    model = OUT / "private_review/model_review"
    seal = p.load(model / "ANNOTATION_SEAL.json")
    assert all(p.sha(model / name) == digest for name, digest in seal["files"].items())
    destination = PRIVATE / "INPUT_SEAL.json"
    if destination.exists():
        prior = p.load(destination)
        assert all(p.sha(PRIVATE / name) == digest for name, digest in prior["files"].items())
        print(json.dumps({"status": "ALREADY_PREPARED", "inputs": 192, "parser_calls": 0}))
        return
    original = p.load(model / "INPUT_NO_TARGET.json")["records"]
    parser_inputs, annotations, review_cases = [], [], []
    for row in original:
        for condition in ("original", "paraphrase"):
            alias = hashlib.sha256(
                f"factlayer-v2:{row['review_index']}:{condition}".encode()).hexdigest()[:20]
            text = row[condition]
            record = {
                "case_id": alias, "input_text": text,
                "input_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "window": "exact complete carrier paragraph; fixed location",
            }
            # Parser receives only literal text and its opaque id/hash.
            parser_inputs.append(record)
            annotations.append({
                "case_id": alias, "reference_source": "MODEL_DIAGNOSTIC_NOT_GOLD",
                "human_text_confirmed": False,
                "review_index": row["review_index"], "condition": condition,
                "model_fact_item_id": row[condition + "_fact_item_id"],
                "public_roles_object": {"role_one": row["role_one"], "role_two": row["role_two"],
                                        "object": row["public_object"]},
                "original_penman": None, "original_graph_facts": None,
                "independent_revised_penman": None, "independent_revised_facts": None,
                "independent_revision_attestation": False,
                "original_ABCD_prediction": None, "revised_ABCD_prediction": None,
                "status": "PENDING_FORMAL_CARRIER_GATE",
            })
            review_cases.append({
                "case_id": alias, "input_text": text,
                "input_text_sha256": record["input_text_sha256"],
                "original_penman": None,
                "facts_read_only_from_graph": dict.fromkeys(p.FIELDS, "unknown"),
                "graph_evidence_triples": dict.fromkeys(p.FIELDS, []),
                "revised_penman": None, "revision_reason_and_text_evidence": [],
                "reviewer_type": "", "human_attestation": False,
                "independent_attestation": False, "status": "",
            })
    p.atomic(PRIVATE / "PARSER_INPUTS_NO_TARGET.json", {
        "version": "v2-fact-layers-preparation:1",
        "intended_symbols_included": False, "reference_facts_included": False,
        "input_mode": "whole_paragraph_fixed_window",
        "multi_sentence_parser_semantics_not_yet_validated": True,
        "rows": parser_inputs})
    p.atomic(PRIVATE / "LAYER_TABLE_PENDING.json", annotations)
    p.atomic(PRIVATE / "GRAPH_REVIEW_PENDING_NO_TARGET.json", {
        "model_or_target_answers_included": False,
        "independent_review_required": True, "cases": review_cases})
    files = ["PARSER_INPUTS_NO_TARGET.json", "LAYER_TABLE_PENDING.json",
             "GRAPH_REVIEW_PENDING_NO_TARGET.json"]
    p.atomic(destination, {
        "at": p.now(), "status": "PREPARED_NOT_EXECUTED",
        "source_annotation_seal_sha256": p.sha(model / "ANNOTATION_SEAL.json"),
        "source_no_target_input_sha256": p.sha(model / "INPUT_NO_TARGET.json"),
        "files": {name: p.sha(PRIVATE / name) for name in files},
        "parser_inputs": 192, "review_cases": 192,
        "target_or_model_facts_given_to_parser": False,
        "target_or_model_facts_given_to_independent_graph_reviewer": False,
        "human_completed": 0, "parser_calls": 0,
        "independent_corrected_graphs": 0, "ABCD_graph_predictions": 0,
    })
    p.atomic(OUT / "FACT_LAYER_STATUS.json", {
        "at": p.now(), "status": "PREPARATION_COMPLETE_EXECUTION_PENDING_HUMAN_GATE",
        "input_seal_sha256": p.sha(destination), "parser_inputs": 192,
        "target_free_inputs": True, "fixed_full_paragraph_window": True,
        "full_penman_required": True, "new_parser_calls": 0,
        "independent_corrected_graphs": 0, "graph_ABCD_predictions": 0,
        "formal_human_gate": "NOT_EVALUATED", "blind_recovery_256": "NOT_RUN",
        "gpu_probe_is_not_inference_or_deployment_readiness": True,
    })
    print(json.dumps({"status": "PREPARED", "inputs": 192,
                      "independent_graph_review_cases": 192, "parser_calls": 0}))


if __name__ == "__main__":
    main()
