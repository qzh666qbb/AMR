"""Join actual raw AMR outputs, then call attribution on every sealed v2 row."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_semantic_fact_carrier_pilot2 as v2
from semantic_fact_raw_amr import audit_raw, replay_signature, validate_inputs
from src.core.semantic_fact_layers import compare_fact_layers

p = v2.runtime
DEFAULT_OUT = ROOT / "experiments/semantic_fact_layers_v3_parse1"


def run(out):
    if (out / "RESULT.json").exists():
        raise RuntimeError("Existing attribution release must not be overwritten")
    freeze = p.load(out / "FREEZE.json")
    freeze_sha = p.sha(out / "FREEZE.json")
    for name, expected in freeze["source_sha256"].items():
        if p.sha(ROOT / name) != expected or p.sha(out / "private/code" / name) != expected:
            raise RuntimeError("Frozen source or snapshot drift")
    historical = p.load(out / "private/HISTORICAL_SHA256.json")
    if p.sha(out / "private/HISTORICAL_SHA256.json") != freeze["historical_sha_manifest_sha256"]:
        raise RuntimeError("Historical SHA manifest changed")
    if any(p.sha(ROOT / name) != expected for name, expected in historical.items()):
        raise RuntimeError("Historical protected files changed")
    source = v2.OUT / "private_review/fact_layers"
    model = v2.OUT / "private_review/model_review"
    for folder, seal_name, key in (
        (source, "INPUT_SEAL.json", "source_input_seal_sha256"),
        (model, "ANNOTATION_SEAL.json", "source_annotation_seal_sha256"),
    ):
        if p.sha(folder / seal_name) != freeze[key]:
            raise RuntimeError("Source seal mismatch")
        for name, expected in p.load(folder / seal_name)["files"].items():
            if p.sha(folder / name) != expected:
                raise RuntimeError("Source sealed data drift")
    if p.sha(out / "private/PARSER_INPUTS_NO_TARGET.json") != freeze["input_sha256"]:
        raise RuntimeError("Prepared GPU input drift")
    remote = out / "private/remote_results"
    if (remote / "HALT.json").exists():
        raise RuntimeError("Halted parser run cannot be presented as completed")
    remote_sha = p.load(remote / "REMOTE_RESULT_SHA256.json")
    for name, expected in remote_sha.items():
        if p.sha(remote / name) != expected:
            raise RuntimeError("Downloaded parser result SHA mismatch")
    if p.sha(remote / "FREEZE.json") != freeze_sha:
        raise RuntimeError("Remote/local freeze differs")
    parser_result = p.load(remote / "PARSER_RESULT.json")
    if parser_result["status"] != "COMPLETE" or parser_result["primary_parser_calls"] != 192:
        raise RuntimeError("192 actual parser results required")
    rows = validate_inputs(p.load(out / "private/PARSER_INPUTS_NO_TARGET.json"))
    raw_files = list((remote / "private/raw_amr").glob("*.json"))
    if len(raw_files) != 192:
        raise RuntimeError("Raw output denominator differs")
    for row in rows[:2]:
        primary = p.load(remote / "private/raw_amr" / (row["case_id"] + ".json"))
        replay = p.load(remote / "private/replays" / (row["case_id"] + ".json"))
        if replay_signature(primary) != replay_signature(replay):
            raise RuntimeError("Local exact replay audit failed")
    layers = {r["case_id"]: r for r in p.load(source / "LAYER_TABLE_PENDING.json")}
    facts = {r["item_id"]: r for r in p.load(model / "facts.json")}
    if len(layers) != 192 or len(facts) != 192 or set(layers) != {r["case_id"] for r in rows}:
        raise RuntimeError("Input/reference identity mismatch")
    keys = p.load(v2.OUT / "private_review/admin/KEYS.json")["1"]["facts"]
    plan = {entry["row"]["item_id"]: entry["row"] for entry in p.load(v2.PRIVATE / "PLAN.json")}
    inputs_by_index = {r["review_index"]: r for r in p.load(model / "INPUT_NO_TARGET.json")["records"]}
    for reviewer in (1, 2):
        meta = p.load(v2.OUT / f"private_review/reviewer{reviewer}/reviewer.json")
        if meta["human_attestation"] or meta["independent_attestation"]:
            raise RuntimeError("New human data requires a separately frozen version")
    if (v2.OUT / "HUMAN_COLLECTION.json").exists():
        raise RuntimeError("New human collection needs separate version binding")
    review = p.load(source / "GRAPH_REVIEW_PENDING_NO_TARGET.json")
    reviews = {r["case_id"]: r for r in review["cases"]}
    private_review = out / "private_review"
    private_review.mkdir(mode=0o700)
    outputs, flat, layer_rows, review_rows, penman_entries = [], [], [], [], []
    for row in rows:
        case_id = row["case_id"]
        raw = p.load(remote / "private/raw_amr" / (case_id + ".json"))
        raw_audit = audit_raw(raw, row, freeze_sha)
        if raw_audit != raw["audit"]:
            raise RuntimeError("Recomputed structural audit differs")
        layer = dict(layers[case_id])
        label = facts[layer["model_fact_item_id"]]
        p.validate_fact(label)
        identity = keys[label["item_id"]]["identity"]
        contract = plan[identity["item"]]
        expected_text = inputs_by_index[layer["review_index"]][layer["condition"]]
        if expected_text != row["input_text"] or expected_text != label["text"]:
            raise RuntimeError("Literal input/reference text differs")
        if identity["condition"] != layer["condition"] or layer["public_roles_object"] != contract["public_contract"]:
            raise RuntimeError("Condition/public contract drift")
        if layer["human_text_confirmed"] or layer["independent_revision_attestation"]:
            raise RuntimeError("Model-only provenance changed")
        layer.update(original_penman=raw["amr"], original_penman_sha256=raw["amr_sha256"],
                     original_parse_audit=raw_audit,
                     status="RAW_AMR_GENERATED_FACT_REVIEW_PENDING")
        comparison = compare_fact_layers(
            contract=contract["public_contract"], planned_symbol=contract["intended_symbol"],
            text_facts=label, raw_graph_facts=layer["original_graph_facts"],
            revised_graph_facts=layer["independent_revised_facts"],
            raw_prediction=layer["original_ABCD_prediction"],
            revised_prediction=layer["revised_ABCD_prediction"],
            text_human_confirmed=False, raw_graph_facts_independent_confirmed=False,
            revision_independent_confirmed=False,
        )
        blocking = ["independent_human_text_confirmation", "raw_graph_fact_annotation",
                    "independent_graph_fact_confirmation", "independent_revised_graph_and_facts",
                    "raw_and_revised_ABCD_predictions"]
        if not raw_audit["structurally_eligible"]:
            blocking.insert(0, "structurally_eligible_raw_AMR")
        record = {
            "case_id": case_id, "condition": layer["condition"],
            "input_text_sha256": row["input_text_sha256"],
            "original_penman": raw["amr"], "original_penman_sha256": raw["amr_sha256"],
            "raw_parse": raw_audit, "graph_inventory": raw["graph_inventory"],
            "model_fact_item_id": label["item_id"],
            "text_reference_source": "SEALED_MODEL_ANNOTATION_NOT_HUMAN_GOLD",
            "comparison": comparison, "blocking_requirements": blocking,
            "raw_AMR_is_not_graph_fact_annotation": True,
        }
        outputs.append(record)
        flat.append({
            "case_id": case_id, "condition": layer["condition"],
            "raw_AMR_present": raw_audit["raw_penman_present"],
            "raw_AMR_sha256": raw["amr_sha256"],
            "parse_status": raw_audit["parse_status"],
            "structurally_eligible": raw_audit["structurally_eligible"],
            "exclusion_reasons": "|".join(raw_audit["exclusion_reasons"]),
            "token_length": raw["token_length"], "graph_instances": raw["graph_inventory"]["instances"],
            "graph_edges": raw["graph_inventory"]["edges"],
            "raw_graph_facts_present": layer["original_graph_facts"] is not None,
            "revised_AMR_present": bool(layer["independent_revised_penman"]),
            "attribution_status": comparison["status"],
            "model_text_reference_symbol": comparison["text_symbol"],
            "human_text_confirmed": False, "confirmed_failure_layers": "",
            "blocking_requirements": "|".join(blocking),
        })
        layer_rows.append(layer)
        graph_review = dict(reviews[case_id])
        graph_review.update(original_penman=raw["amr"], original_penman_sha256=raw["amr_sha256"],
                            original_parse_audit=raw_audit, status="RAW_AMR_AVAILABLE_REVIEW_PENDING")
        review_rows.append(graph_review)
        # Original parser bytes are retained in each per-case JSON. The collected
        # view adds id/hash comments only; it is not a replacement for raw output.
        penman_entries.append(f"# ::case_id {case_id}\n# ::input_sha256 {row['input_text_sha256']}\n{raw['amr'] or ''}")
    p.atomic(private_review / "ROWS.json", outputs)
    with (private_review / "ROWS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    p.atomic(private_review / "LAYER_TABLE_WITH_RAW_AMR.json", layer_rows)
    p.atomic(private_review / "GRAPH_REVIEW_WITH_RAW_AMR_NO_TARGET.json", {
        "model_or_target_answers_included": False, "independent_review_required": True,
        "cases": review_rows,
    })
    (private_review / "RAW_AMR_192.penman").write_text("\n\n".join(penman_entries) + "\n", encoding="utf-8")
    for path in private_review.iterdir():
        path.chmod(0o600)
    result = {
        "at": p.now(), "status": "RAW_AMR_COMPLETE_ATTRIBUTION_EXECUTED_CONFIRMATION_PENDING",
        "rows_received": 192, "rows_processed": len(outputs), "attribution_function_calls": len(outputs),
        "conditions": dict(Counter(r["condition"] for r in outputs)),
        "raw_graphs_available": sum(r["raw_parse"]["raw_penman_present"] for r in outputs),
        "structurally_eligible_raw_graphs": sum(r["raw_parse"]["structurally_eligible"] for r in outputs),
        "excluded_raw_graphs": sum(not r["raw_parse"]["structurally_eligible"] for r in outputs),
        "row_attribution_status_counts": dict(Counter(r["comparison"]["status"] for r in outputs)),
        "parse_status_counts": dict(Counter(r["raw_parse"]["parse_status"] for r in outputs)),
        "raw_graph_fact_annotations": 0, "independent_revised_graphs": 0,
        "ABCD_graph_predictions": 0, "human_completed": 0,
        "complete_fact_graph_classifier_comparisons": 0,
        "confirmed_failure_attributions": 0, "parser_loss_rate": None, "classifier_error_rate": None,
        "primary_parser_calls": parser_result["primary_parser_calls"],
        "replay_parser_calls": parser_result["replay_parser_calls"],
        "total_parser_calls": parser_result["total_parser_calls"],
        "replay_exact_match": True, "parser_elapsed_seconds": parser_result["elapsed_seconds"],
        "peak_total_vram_fraction": parser_result["peak_total_vram_fraction"],
        "multi_sentence_semantic_completeness_validated": False,
        "formal_human_gate": "NOT_EVALUATED", "blind_recovery_256": "NOT_RUN",
        "new_provider_detector_embedding_blind_decode_calls": 0, "freeze_sha256": freeze_sha,
    }
    p.atomic(out / "RESULT.json", result)
    p.atomic(out / "AUDIT.json", {
        "at": p.now(), "status": "PASS",
        "downloaded_remote_manifest_entries_verified": len(remote_sha),
        "all_192_literal_input_id_condition_and_output_SHA_verified": True,
        "all_192_original_AMR_slots_filled_from_actual_parser_records": True,
        "all_192_rows_called_actual_attribution_function": True,
        "exact_replays_locally_verified": 2,
        "no_graph_fact_or_human_attestation_fabrication": True,
        "v1_v2_prior_execution_protected_files_unchanged": len(historical),
        "sealed_sources_and_model_references_unchanged": True,
        "rows_json_sha256": p.sha(private_review / "ROWS.json"),
        "rows_csv_sha256": p.sha(private_review / "ROWS.csv"),
    })
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    run(args.out)
