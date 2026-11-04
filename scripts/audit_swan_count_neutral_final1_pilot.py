#!/usr/bin/env python3
"""Offline closure audit for the failed swan_count_neutral_final1 pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from swan_final1_budget import Journal
from swan_final1_common import OUT, digest, load, sha256, single_writer
from swan_final1_contract import check
from swan_final1_generation import load_slots, verify_generation
from swan_final1_workflow import cohort_gates, require_pilot, sources_for


ROOT = Path(__file__).resolve().parents[1]
CLOSURE = OUT / "pilot_closure"


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n").encode()


def markdown_bytes(value):
    return value.encode("utf-8")


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


def input_paths():
    fixed = [
        OUT / "manifest.json",
        OUT / "source_manifest.json",
        OUT / "environment_lock.json",
        OUT / "authorization_pilot.json",
        OUT / "private/pilot/journal.jsonl",
        OUT / "transforms/pilot.json",
        OUT / "quality/pilot/baseline/private_index.json",
        OUT / "quality/pilot/baseline/packet_A.json",
        OUT / "quality/pilot/baseline/packet_B.json",
        OUT / "quality/pilot/baseline/review_A.json",
        OUT / "quality/pilot/baseline/review_B.json",
        OUT / "quality/pilot/baseline/decision.json",
        OUT / "quality/pilot/pair/private_index.json",
        OUT / "quality/pilot/pair/packet_A.json",
        OUT / "quality/pilot/pair/packet_B.json",
        OUT / "quality/pilot/pair/review_A.json",
        OUT / "quality/pilot/pair/review_B.json",
        OUT / "quality/pilot/pair/decision.json",
        OUT / "auxiliary_reviews/baseline_round1/report.md",
        OUT / "PILOT_CONCLUSION.md",
        Path(__file__).resolve(),
    ]
    return fixed + sorted((OUT / "documents/pilot").glob("*.json"))


def relative(path):
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def build():
    contract = check(OUT, full_environment=True)
    sources = sources_for(OUT, "pilot")
    documents = verify_generation(OUT, "pilot", sources)
    gate = cohort_gates(OUT, "pilot")
    journal = Journal(
        OUT, "pilot", 1, None, sha256(OUT / "manifest.json"), allow_halted=True
    )
    accounting = journal.totals()

    rows = []
    for document in documents:
        slots, interrupted = load_slots(OUT, "pilot", document["allocation_id"])
        rows.append({
            "allocation_id": document["allocation_id"],
            "generation_complete": document["generation_complete"],
            "termination": document["termination"],
            "word_count": document["word_count"],
            "generated_slots": len(slots),
            "num_detection_units": document["num_detection_units"],
            "structural_pass": document["structural_pass"],
            "word_range_pass": document["word_range_pass"],
            "unit_count_pass": document["unit_count_pass"],
            "slot_document_segmentation_equal":
                document["slot_document_segmentation_equal"],
            "identity_pass": document["identity_pass"],
            "provider_truncation": document["provider_truncation"],
            "interrupted": interrupted is not None,
        })

    baseline = load(OUT / "quality/pilot/baseline/decision.json")
    pair = load(OUT / "quality/pilot/pair/decision.json")
    transforms = load(OUT / "transforms/pilot.json")
    auxiliary = (OUT / "auxiliary_reviews/baseline_round1/report.md").read_text(
        encoding="utf-8"
    )
    provenance_disclosed = (
        "评审来源为本会话的语言模型助手" in auxiliary
        and "不属于独立盲评" in auxiliary
    )

    main_blocked = False
    try:
        require_pilot(OUT)
    except PermissionError:
        main_blocked = True

    attrition = {
        "schema_version": 1,
        "experiment": "swan_count_neutral_final1",
        "phase": "pilot",
        "documents": rows,
        "counts": {
            "allocated": len(sources),
            "terminal_documents": len(documents),
            "generation_complete": sum(r["generation_complete"] for r in rows),
            "structural_pass": sum(r["structural_pass"] for r in rows),
            "structural_fail": sum(not r["structural_pass"] for r in rows),
            "baseline_reviewed": len(baseline["decisions"]),
            "baseline_quality_pass": baseline["quality_pass_count"],
            "transformed_allocations": len(transforms),
            "pair_reviewed": len(pair["decisions"]),
            "pair_quality_pass": pair["quality_pass_count"],
            "detection_units_evaluated": 0,
        },
    }

    hashes = {relative(path): sha256(path) for path in input_paths()}
    artifact_hashes = {
        "schema_version": 1,
        "experiment": "swan_count_neutral_final1",
        "phase": "pilot",
        "inputs": hashes,
        "stable_aggregate_sha256": digest(hashes),
    }

    checks = {
        "frozen_manifest_code_assets_and_environment_verified":
            contract["state"] == "frozen_hashes_verified"
            and contract["environment_verified"],
        "pilot_source_count_is_8": len(sources) == 8,
        "terminal_document_count_is_8": len(documents) == 8,
        "generation_provenance_replayed": len(rows) == 8
            and all(r["generation_complete"] and not r["interrupted"] for r in rows),
        "journal_chain_and_accounting_verified":
            accounting["calls"] == 3467
            and accounting["uncertain_or_invalid_calls"] == 0,
        "structural_attrition_is_6_of_8":
            attrition["counts"]["structural_pass"] == 6
            and attrition["counts"]["structural_fail"] == 2,
        "baseline_labels_rebound_and_replayed":
            gate["counts"]["baseline_quality_pass"] == 0
            and len(baseline["decisions"]) == 6,
        "empty_transform_replayed": gate["counts"]["C2_constructed"] == 0
            and len(transforms) == 0,
        "empty_pair_labels_replayed": gate["counts"]["C2_quality_pass"] == 0
            and len(pair["decisions"]) == 0,
        "pilot_gate_failed": gate["passed"] is False
            and gate["gates"] == {"pilot": False},
        "main_access_blocked": main_blocked,
        "no_detection_plan_or_results":
            not (OUT / "detection_plan.json").exists()
            and not (OUT / "results.json").exists(),
        "review_provenance_limitation_disclosed": provenance_disclosed,
        "two_independent_human_reviews_satisfied": False,
    }
    required = {k: v for k, v in checks.items()
                if k != "two_independent_human_reviews_satisfied"}
    if not all(required.values()):
        failed = [k for k, value in required.items() if not value]
        raise ValueError(f"Pilot closure checks failed: {failed}")

    attrition_raw = json_bytes(attrition)
    artifacts_raw = json_bytes(artifact_hashes)
    audit = {
        "schema_version": 1,
        "experiment": "swan_count_neutral_final1",
        "phase": "pilot",
        "status": "passed_with_disclosed_review_provenance_limitation",
        "decision": "branch_C_stop_generation_route",
        "pilot_passed": False,
        "failure_reason": "generated_text_quality",
        "checks": checks,
        "gate": gate,
        "accounting": accounting,
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "attrition_sha256": bytes_sha256(attrition_raw),
        "artifact_hashes_sha256": bytes_sha256(artifacts_raw),
        "new_api_calls": 0,
        "new_neural_parser_calls": 0,
        "new_s2match_calls": 0,
        "new_detector_calls": 0,
        "confirmatory_attack_tests_performed": False,
    }
    audit_raw = json_bytes(audit)

    report = """# final1 Pilot 收尾审计

状态：**审计通过，但保留已披露的评审来源限制。**

离线重放确认 8 个 pilot source 均有终态文档，3467 次调用全部具有确定
终态且无 uncertain/invalid 记录。6/8 文档通过结构门；当前冻结 baseline
标签通过数为 0/6，因此变换和 pair 集均为空，pilot gate 重算为
`passed=false`，main 访问保持阻断。

该结果把当前生成路线归入分支 C，失败原因为生成文本质量，不是检测或攻击
效果。本次收尾审计没有调用生成 API、神经 AMR parser、S2Match 或检测器，
也没有创建检测计划。

评审来源限制：辅助报告披露当前 A/B 标签来自语言模型辅助评审，不满足
协议要求的两名独立真人评审。因此这里只保留生成可行性负结果和停止决定，
不把标签升级为正式真人双盲证据。
"""
    report_raw = markdown_bytes(report)

    recovery = {
        "schema_version": 1,
        "status": "complete_no_compute_pilot_closure",
        "decision": "branch_C_stop_generation_route",
        "files_checked": len(hashes),
        "stable_aggregate_sha256": artifact_hashes["stable_aggregate_sha256"],
        "network_forbidden": True,
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
        help="require all closure artifacts to exist and match the offline replay",
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
