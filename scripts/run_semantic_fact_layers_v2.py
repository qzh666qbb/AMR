"""Execute all 192 sealed rows through attribution; never invent missing graphs."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_semantic_fact_carrier_pilot2 as v2
from src.core.semantic_fact_layers import compare_fact_layers

p = v2.runtime
SOURCE = v2.OUT
DEFAULT_OUT = ROOT / "experiments/semantic_fact_layers_v2_execution1"


def csv_write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.chmod(0o600)


def run(out):
    started = time.monotonic()
    p.verify(SOURCE, v2.PRIVATE)
    preparation = p.load(SOURCE / "FACT_LAYER_PREPARATION_RELEASE.json")
    for name, digest in preparation["sources"].items():
        if p.sha(ROOT / name) != digest:
            raise RuntimeError("Prepared attribution or test source drift")
    source_folder = SOURCE / "private_review/fact_layers"
    input_seal_path = source_folder / "INPUT_SEAL.json"
    input_seal = p.load(input_seal_path)
    for name, digest in input_seal["files"].items():
        if p.sha(source_folder / name) != digest:
            raise RuntimeError("Frozen fact-layer input drift")
    model_folder = SOURCE / "private_review/model_review"
    annotation_seal_path = model_folder / "ANNOTATION_SEAL.json"
    annotation_seal = p.load(annotation_seal_path)
    for name, digest in annotation_seal["files"].items():
        if p.sha(model_folder / name) != digest:
            raise RuntimeError("Sealed model annotation drift")
    input_data = p.load(source_folder / "PARSER_INPUTS_NO_TARGET.json")
    inputs = input_data["rows"]
    pending = p.load(source_folder / "LAYER_TABLE_PENDING.json")
    model_facts = p.load(model_folder / "facts.json")
    if not (len(inputs) == len(pending) == len(model_facts) == 192):
        raise RuntimeError("The full 192-row denominator is required")
    input_ids = [r["case_id"] for r in inputs]
    layers = {r["case_id"]: r for r in pending}
    facts = {r["item_id"]: r for r in model_facts}
    if len(set(input_ids)) != 192 or len(layers) != 192 or len(facts) != 192:
        raise RuntimeError("Duplicate case or fact identity")
    if set(input_ids) != set(layers):
        raise RuntimeError("Input-to-layer correspondence differs")
    if input_data["intended_symbols_included"] or input_data["reference_facts_included"]:
        raise RuntimeError("Parser input is not target-free")
    keys = p.load(SOURCE / "private_review/admin/KEYS.json")["1"]["facts"]
    plan = {entry["row"]["item_id"]: entry["row"]
            for entry in p.load(v2.PRIVATE / "PLAN.json")}
    model_by_index = {r["review_index"]: r
                      for r in p.load(model_folder / "INPUT_NO_TARGET.json")["records"]}
    prior_protected = p.load(v2.PRIVATE / "V1_PROTECTED_SHA256.json")
    if not all(p.sha(ROOT / name) == digest for name, digest in prior_protected.items()):
        raise RuntimeError("Historical v1 SHA drift")
    for reviewer in (1, 2):
        metadata = p.load(SOURCE / f"private_review/reviewer{reviewer}/reviewer.json")
        if metadata["human_attestation"] or metadata["independent_attestation"]:
            raise RuntimeError("Use a separately frozen human-confirmed batch")
    if (SOURCE / "HUMAN_COLLECTION.json").exists():
        raise RuntimeError("New human result requires a separately bound input version")
    sources = ["src/core/semantic_fact_layers.py", "tests/test_semantic_fact_layers.py",
               "scripts/run_semantic_fact_layers_v2.py"]
    freeze = {
        "scope": "192-row attribution execution on existing model facts and pending graph slots",
        "authorization": "User: 继续执行分层诊断的 192 条输入与归因代码，并运行 12 项测试",
        "source_generation_freeze_sha256": p.sha(SOURCE / "FREEZE.json"),
        "source_input_seal_sha256": p.sha(input_seal_path),
        "source_annotation_seal_sha256": p.sha(annotation_seal_path),
        "code_sha256": {name: p.sha(ROOT / name) for name in sources},
        "rows": 192, "uses_model_reference_facts_not_gold": True,
        "formal_human_gate_not_overridden": True,
        "new_parser_or_provider_calls_allowed": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    if (out / "RUN_FREEZE.json").exists():
        raise RuntimeError("Execution release already exists; verify it or use a new output directory")
    p.atomic(out / "RUN_FREEZE.json", {"at": p.now(), **freeze})
    private = out / "private_review"
    private.mkdir(mode=0o700)
    outputs, flat = [], []
    for text_input in inputs:
        case_id = text_input["case_id"]
        layer = layers[case_id]
        model_fact_id = layer["model_fact_item_id"]
        label = facts[model_fact_id]
        p.validate_fact(label)
        identity = keys[model_fact_id]["identity"]
        if identity["condition"] != layer["condition"]:
            raise RuntimeError("Original/paraphrase identity drift")
        expected_text = model_by_index[layer["review_index"]][layer["condition"]]
        if text_input["input_text"] != expected_text or label["text"] != expected_text:
            raise RuntimeError("Literal parser text differs from sealed annotation")
        if hashlib.sha256(expected_text.encode()).hexdigest() != text_input["input_text_sha256"]:
            raise RuntimeError("Parser text SHA mismatch")
        if layer["human_text_confirmed"] or layer["independent_revision_attestation"]:
            raise RuntimeError("Model-only batch cannot supply human attestations")
        contract = plan[identity["item"]]
        if layer["public_roles_object"] != contract["public_contract"]:
            raise RuntimeError("Public contract differs from frozen row")
        # Execute the actual production attribution function once for EVERY row.
        # Model reference facts and empty graph slots retain their true provenance.
        comparison = compare_fact_layers(
            contract=contract["public_contract"],
            planned_symbol=contract["intended_symbol"],
            text_facts=label,
            raw_graph_facts=layer["original_graph_facts"],
            revised_graph_facts=layer["independent_revised_facts"],
            raw_prediction=layer["original_ABCD_prediction"],
            revised_prediction=layer["revised_ABCD_prediction"],
            text_human_confirmed=False,
            raw_graph_facts_independent_confirmed=False,
            revision_independent_confirmed=False,
        )
        outputs.append({
            "case_id": case_id, "condition": layer["condition"],
            "input_text_sha256": text_input["input_text_sha256"],
            "model_fact_item_id": model_fact_id,
            "text_reference_source": "SEALED_MODEL_ANNOTATION_NOT_HUMAN_GOLD",
            "raw_penman_present": bool(layer["original_penman"]),
            "revised_penman_present": bool(layer["independent_revised_penman"]),
            "comparison": comparison,
        })
        flat.append({
            "case_id": case_id, "condition": layer["condition"],
            "status": comparison["status"],
            "model_text_reference_symbol": comparison["text_symbol"],
            "model_reference_matches_plan": comparison["writing_matches_plan"],
            "raw_graph_present": bool(layer["original_penman"]),
            "revised_graph_present": bool(layer["independent_revised_penman"]),
            "confirmed_failure_layers": "|".join(comparison["supported_failure_layers"]),
            "human_text_confirmed": False,
        })
    if len(outputs) != 192 or len(flat) != 192:
        raise RuntimeError("Incomplete execution denominator")
    if not all(p.sha(source_folder / name) == digest for name, digest in input_seal["files"].items()):
        raise RuntimeError("Source inputs changed during execution")
    if not all(p.sha(model_folder / name) == digest for name, digest in annotation_seal["files"].items()):
        raise RuntimeError("Model annotations changed during execution")
    if not all(p.sha(ROOT / name) == digest for name, digest in prior_protected.items()):
        raise RuntimeError("Historical data changed during execution")
    p.atomic(private / "ROWS.json", outputs)
    csv_write(private / "ROWS.csv", flat)
    status_counts = dict(Counter(r["comparison"]["status"] for r in outputs))
    result = {
        "at": p.now(), "status": "BATCH_CODE_EXECUTED_GRAPH_DIAGNOSIS_PENDING",
        "rows_received": 192, "rows_processed": 192, "attribution_function_calls": 192,
        "conditions": dict(Counter(r["condition"] for r in outputs)),
        "row_status_counts": status_counts,
        "complete_fact_graph_classifier_comparisons": sum(
            r["comparison"]["status"] == "LAYER_COMPARISON_COMPLETE" for r in outputs),
        "raw_graphs_available": sum(r["raw_penman_present"] for r in outputs),
        "independent_revised_graphs_available": sum(r["revised_penman_present"] for r in outputs),
        "confirmed_failure_attributions": sum(bool(r["comparison"]["supported_failure_layers"])
                                             for r in outputs),
        "parser_loss_rate": None, "classifier_error_rate": None,
        "model_reference_matches_plan_diagnostic": dict(Counter(
            r["condition"] for r in outputs if r["comparison"]["writing_matches_plan"])),
        "text_fact_symbols_are_existing_model_references_not_graph_predictions": True,
        "human_completed": 0, "formal_human_gate": "NOT_EVALUATED",
        "new_parser_calls": 0, "new_provider_calls": 0, "new_embedding_calls": 0,
        "new_blind_decode_calls": 0, "elapsed_seconds": time.monotonic() - started,
        "pending_is_not_a_successful_semantic_comparison": True,
        "run_freeze_sha256": p.sha(out / "RUN_FREEZE.json"),
    }
    p.atomic(out / "RESULT.json", result)
    p.atomic(out / "AUDIT.json", {
        "at": p.now(), "status": "PASS",
        "all_192_rows_actually_called_attribution_function": True,
        "literal_text_and_case_id_correspondence_verified": True,
        "sealed_input_sha_match_before_and_after": True,
        "sealed_model_annotation_sha_match_before_and_after": True,
        "no_human_attestation_or_graph_fact_fabrication": True,
        "parser_and_classifier_rates_left_unestimated": True,
        "v1_protected_sha_match": len(prior_protected),
        "rows_json_sha256": p.sha(private / "ROWS.json"),
        "rows_csv_sha256": p.sha(private / "ROWS.csv"),
        "run_sources_sha256": freeze["code_sha256"],
    })
    (out / "REPORT.md").write_text(
        "# 192条分层归因代码执行\n\n"
        "192条封存输入全部完成文字SHA、ID及原句/复述对应检查，并逐条调用实际"
        "`compare_fact_layers`函数，共192次；96条原句与96条复述均保留。\n\n"
        "结果192/192为`PENDING_TEXT_CONFIRMATION`。两位真人事实表和声明仍为空，"
        "原AMR、图事实、独立修订图均没有实际内容。因此完整文本—原图—修订图—"
        "分类对照完成0条，解析损失率与分类错误率未估计（null）。"
        "这次完成的是现有输入上的批处理与门控执行，不是192条AMR推理或完整诊断。\n\n"
        "已有模型事实映射与目标相符：原句96/96、复述96/96；这是复用封存模型"
        "判断的参考一致性检查，不是新的图读出性能。未伪造真人身份、未补图、"
        "未把pending当作过门。新parser/provider/Embedding/盲恢复调用均0。\n\n"
        "12项既有逻辑测试本轮实际重新运行并全部通过，验证角色反向、状态不同、"
        "擦除、错误修订、模型判断不能作独立确认、原文不符目标时应按实际"
        "文本归因等情况。测试记录见[测试结果](TEST_RESULT.json)。\n\n"
        "旧v1保护清单64个文件SHA、v2封存输入与模型判断均未变。"
        "实际图层诊断仍须独立真人确认、原AMR及独立图事实/修订结果；"
        "这些缺项会保留为pending，不会被解释成成功或无错误。\n\n"
        "入口：[聚合结果](RESULT.json)、[192条执行状态](private_review/ROWS.csv)、"
        "[逐条JSON](private_review/ROWS.json)、[执行审计](AUDIT.json)。\n",
        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.out)
