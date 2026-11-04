"""Run the versioned P1b development gate with hop penalty."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Sequence

from .bank_detector import add_runtime_arguments, build_detector
from .elastic_v2 import V2PathResult
from .elastic_v2_development import (
    ABSTAIN_CALIBRATION_STATISTIC,
    DEVELOPMENT_COHORTS,
    SOURCE,
    heartbeat,
    jsonl,
    read_json,
    read_rows,
    relative,
    summarize as summarize_v2,
)
from .elastic_v3 import (
    ElasticBoundaryProposerV3,
    InvalidTolerantAMRExtractor,
    SwanAMRBankScorerV3,
    V3EdgeEvidence,
)
from .elastic_v4 import SwanBankLatticeDetectorV4

from .formal_null_evaluation import (
    ROOT,
    _atomic_write,
    _audit_payload,
    _freeze,
    _json_bytes,
    sha256_bytes,
    sha256_file,
)
from .null_calibration import DetectionResult, EmpiricalNullCalibrator
from .parser_recovery_audit import attach_recovery_audit
from .positive_attack_evaluation import VERSION, freeze_runtime
from .sqlite_cache import normalize_text


DEFAULT_OUTPUT = ROOT / "experiments/swan_elastic_v4_development1"


def verify_inputs(output: Path) -> dict:
    manifest_path = output / "INPUT_MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing P1b manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    for name, item in manifest["outputs"].items():
        if sha256_file(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"P1b input SHA differs: {name}")
    return manifest


def prepare(output: Path) -> dict:
    calibration_path = output / "inputs/calibration_rows.jsonl"
    development_path = output / "inputs/development_rows.jsonl"
    if calibration_path.exists() or development_path.exists():
        raise FileExistsError("P1b inputs already exist")
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_source = read_rows(ROOT / SOURCE["calibration"])
    calibration_rows = []
    for row in calibration_source:
        if row["cohort"] != "c4_real_short_20_80_1k":
            continue
        calibration_rows.append(row)
    development_source = read_rows(ROOT / SOURCE["development"])
    development_rows = []
    for row in development_source:
        if row["cohort"] in DEVELOPMENT_COHORTS:
            development_rows.append(row)
    jsonl(calibration_path, calibration_rows)
    jsonl(development_path, development_rows)
    manifest = {
        "version": VERSION,
        "selection": {
            "calibration_start_index": 0,
            "v2_partial_reused": False,
        },
        "outputs": {
            "calibration": {
                "path": relative(calibration_path),
                "rows": len(calibration_rows),
                "sha256": sha256_file(calibration_path),
            },
            "development": {
                "path": relative(development_path),
                "rows": len(development_rows),
                "sha256": sha256_file(development_path),
            },
        },
        "claim_limits": {
            "calibration_cohort": "c4_real_short_20_80_1k",
            "development_cohorts": DEVELOPMENT_COHORTS,
        },
    }
    _freeze(output / "INPUT_MANIFEST.json", _json_bytes(manifest))
    return manifest


def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def build_v4_detector(args: argparse.Namespace) -> SwanBankLatticeDetectorV4:
    base = build_detector(args)
    base_extractor = base.scorer.extractor
    extractor = InvalidTolerantAMRExtractor(
        backend=base_extractor.backend,
        cache=base_extractor.cache,
        request_batch_size=args.request_batch_size,
    )
    scorer = SwanAMRBankScorerV3(
        extractor=extractor,
        bank_path=args.bank,
    )
    return SwanBankLatticeDetectorV4(
        builder=ElasticBoundaryProposerV3(
            max_boundary_hops=args.max_boundary_hops,
            max_span_words=args.max_span_words,
            capitalization_gap_words=args.capitalization_gap_words,
            long_gap_stride_words=args.long_gap_stride_words,
        ),
        scorer=scorer,
        score_center=args.score_center,
        green_threshold=args.green_threshold,
        p2=0.18,
        p3=0.18,
    )


def detection_payload(
    result: V2PathResult,
    lattice,
    evidence: Sequence[V3EdgeEvidence],
) -> dict:
    if len(evidence) != len(lattice.edges):
        raise RuntimeError("P1b edge evidence cardinality differs")
    valid_edges = {
        edge for edge, item in zip(lattice.edges, evidence) if item.valid
    }
    selected_invalid = sum(
        unit.edge not in valid_edges for unit in result.units
    )
    if selected_invalid:
        raise RuntimeError("v4 selected a structurally invalid edge")
    reasons = Counter(
        reason
        for item in evidence
        for reason in item.reasons
    )
    return {
        "status": result.status,
        "statistic": result.statistic,
        "raw_utility": result.raw_utility,
        "unit_count": result.unit_count,
        "green_count": result.green_count,
        "invalid_edge_count": result.invalid_edge_count,
        "invalid_edge_reasons": dict(sorted(reasons.items())),
        "abstain_reason": result.abstain_reason,
        "selected_invalid_edges": selected_invalid,
        "units": [
            {
                "start": unit.edge.start,
                "end": unit.edge.end,
                "boundary_hops": unit.edge.boundary_hops,
                "span_sha256": sha256_bytes(
                    normalize_text(unit.edge.span_text).encode("utf-8")
                ),
                "utility": unit.utility,
                "match": asdict(unit.match),
            }
            for unit in result.units
        ],
    }


def score_row(
    detector: SwanBankLatticeDetectorV4,
    recovery,
    source: dict,
    stage: str,
    calibrator: EmpiricalNullCalibrator = None,
) -> dict:
    text = source["window"] if stage == "calibration" else source["text"]
    lattice = detector.builder.build_lattice(text)
    result, evidence = detector.detect_lattice_with_evidence(
        lattice,
        recovery.records,
    )
    normalized_spans = {
        normalize_text(edge.span_text) for edge in lattice.edges
    }
    records = tuple(
        detector.scorer._records[span]  # pylint: disable=protected-access
        for span in sorted(normalized_spans)
    )
    audit = _audit_payload(records)
    if audit["span_records"] != len(normalized_spans):
        raise RuntimeError("P1b row has missing PENMAN evidence")
    detection = detection_payload(result, lattice, evidence)
    row = {
        "schema_version": 1,
        "stage": stage,
        "word_count": len(text.split()),
        "candidate_edges": len(lattice.edges),
        "unique_candidate_spans": len(normalized_spans),
        "detection": detection,
        "penman_audit": audit,
    }
    if stage == "calibration":
        row.update({
            "cohort": source["cohort"],
            "cohort_index": source["cohort_index"],
            "allocation_id": source["allocation_id"],
            "window_sha256": source["window_sha256"],
        })
    else:
        row.update({
            "input_index": source["input_index"],
            "cohort": source["cohort"],
            "cohort_index": source["cohort_index"],
            "row_id": source["row_id"],
            "group_id": source["group_id"],
            "variant_id": source["variant_id"],
            "text_sha256": source["text_sha256"],
        })
        if calibrator is None:
            raise ValueError("Development rows require v4 calibration")
        statistic = (
            result.statistic
            if not result.abstained
            else ABSTAIN_CALIBRATION_STATISTIC
        )
        calibrated = asdict(
            calibrator.calibrate(
                statistic=statistic,
                word_count=row["word_count"],
                alpha=0.01,
            )
        )
        if result.abstained:
            calibrated.update({
                "statistic": None,
                "p_value": 1.0,
                "is_watermarked": False,
                "scope": "abstain",
            })
        row["calibration"] = calibrated
    return row


def load_checkpoint(path: Path, inputs: Sequence[dict], stage: str) -> list:
    if not path.exists():
        return []
    rows = read_rows(path)
    if len(rows) > len(inputs):
        raise ValueError(f"P1b checkpoint has excess rows: {path}")
    identity = "window_sha256" if stage == "calibration" else "row_id"
    for index, row in enumerate(rows):
        if row.get(identity) != inputs[index].get(identity):
            raise ValueError(
                f"P1b checkpoint identity mismatch: {path}:{index + 1}"
            )
    return rows


def process(
    detector: SwanBankLatticeDetectorV4,
    recovery,
    inputs: Sequence[dict],
    path: Path,
    output: Path,
    stage: str,
    batch_size: int,
    calibrator: EmpiricalNullCalibrator = None,
) -> list:
    rows = load_checkpoint(path, inputs, stage)
    heartbeat(output, stage, len(rows), len(inputs))
    for start in range(len(rows), len(inputs), batch_size):
        batch = inputs[start:start + batch_size]
        spans = []
        for source in batch:
            text = source["window"] if stage == "calibration" else source["text"]
            spans.extend(
                edge.span_text
                for edge in detector.builder.build_lattice(text).edges
            )
        detector.scorer.prepare(spans)
        for source in batch:
            row = score_row(
                detector,
                recovery,
                source,
                stage,
                calibrator,
            )
            append(path, row)
            rows.append(row)
            heartbeat(output, stage, len(rows), len(inputs))
            print(
                json.dumps({
                    "stage": stage,
                    "completed": len(rows),
                    "total": len(inputs),
                    "status": row["detection"]["status"],
                    "invalid_edges": row["detection"]["invalid_edge_count"],
                    "statistic": row["detection"]["statistic"],
                }),
                flush=True,
            )
        detector.scorer.clear_prepared()
    return rows


def fit_calibration(
    output: Path,
    inputs: Sequence[dict],
    rows: Sequence[dict],
    runtime: dict,
) -> EmpiricalNullCalibrator:
    path = output / "results/CALIBRATION.json"
    provenance = {
        "version": VERSION,
        "input_manifest_sha256": sha256_file(output / "INPUT_MANIFEST.json"),
        "runtime_manifest_sha256": sha256_file(output / "RUNTIME_MANIFEST.json"),
        "calibration_rows_sha256": sha256_file(
            output / "results/calibration_rows.jsonl"
        ),
        "abstain_policy": {
            "retained_in_denominator": True,
            "statistic": ABSTAIN_CALIBRATION_STATISTIC,
        },
        "invalid_edge_policy": "exclude_before_dp",
    }
    if path.exists():
        calibrator = EmpiricalNullCalibrator.load(path)
        if calibrator.artifact.get("v4_provenance") != provenance:
            raise ValueError("P1b calibration provenance differs")
        return calibrator
    if len(inputs) != len(rows):
        raise ValueError("P1b calibration is incomplete")
    samples = []
    for source, row in zip(inputs, rows):
        if source["window_sha256"] != row["window_sha256"]:
            raise ValueError("P1b calibration source binding differs")
        statistic = row["detection"]["statistic"]
        if row["detection"]["status"] == "abstain":
            statistic = ABSTAIN_CALIBRATION_STATISTIC
        samples.append((
            source["window"],
            DetectionResult(
                statistic=statistic,
                raw_utility=statistic,
                unit_count=row["detection"]["unit_count"],
                green_count=row["detection"]["green_count"],
                units=(),
            ),
        ))
    calibrator = EmpiricalNullCalibrator.fit(
        samples=samples,
        detector_fingerprint=runtime["fingerprints"]["detector"],
        bank_fingerprint=runtime["fingerprints"]["bank"],
        bank_sha256=runtime["fingerprints"]["bank_sha256"],
        parser_fingerprint=runtime["fingerprints"]["parser"],
        min_bucket_size=20,
    )
    calibrator.artifact["v4_provenance"] = provenance
    calibrator.save(path)
    return calibrator


def summarize(
    calibration_rows: Sequence[dict],
    rows: Sequence[dict],
) -> dict:
    summary = summarize_v2(calibration_rows, rows)
    all_rows = list(calibration_rows) + list(rows)
    reasons = Counter()
    graph_invalid_records = 0
    roundtrip_invalid_records = 0
    warning_span_records = 0
    for row in all_rows:
        reasons.update(row["detection"].get("invalid_edge_reasons", {}))
        audit = row["penman_audit"]
        graph_invalid_records += (
            audit["span_records"] - audit["graph_valid_records"]
        )
        roundtrip_invalid_records += (
            audit["span_records"] - audit["roundtrip_valid_records"]
        )
        warning_span_records += audit["warning_span_records"]
    summary.update({
        "invalid_candidate_edge_reasons": dict(sorted(reasons.items())),
        "row_local_graph_invalid_span_records": graph_invalid_records,
        "row_local_roundtrip_invalid_span_records": roundtrip_invalid_records,
        "row_local_warning_span_records": warning_span_records,
    })
    return summary


def write_reports(output: Path, result: dict) -> None:
    summary = result["summary"]

    def cell(item: dict) -> str:
        return (
            f"{item['successes']}/{item['trials']} ({item['rate']:.1%})"
            if item["trials"]
            else "n/a"
        )

    lines = [
        "# SWAN 弹性检测 v4：P1b 开发验证", "",
        f"裁决：**{result['decision']}**。本裁决仅限开发门，不是正式 "
        "FPR/TPR 证据。", "",
        "| 指标 | 结果 |", "|---|---:|",
        f"| baseline 检出 | {cell(summary['baseline_detection'])} |",
        f"| 五变体全部条件存活 | "
        f"{cell(summary['survival_all_five_variants'])} |",
        f"| calibration abstain | {cell(summary['calibration']['abstain'])} |",
        f"| development abstain | {cell(summary['development_abstain'])} |",
        f"| 无效候选边 | {summary['invalid_candidate_edges']} |",
        f"| 选中无效边 | {summary['selected_invalid_edges']} |", "",
        "无效原因计数：`"
        + json.dumps(
            summary["invalid_candidate_edge_reasons"],
            sort_keys=True,
        )
        + "`。",
        "",
        "引入 Hop Penalty：2-hop/3-hop 及以上均惩罚 0.18，非法候选边在 DP 前排除；断路样本记 abstain 并保留在全分母。",
        "P1b Go 只授权准备新来源 P2。",
    ]
    _atomic_write(
        output / "REPORT.md",
        ("\n".join(lines) + "\n").encode("utf-8"),
    )
    decision = [
        f"# P1b 裁决：{result['decision']}", "",
        "主门：baseline 检出至少 80%，baseline 可检出条件下五变体全部"
        "存活至少 80%，且选中无效边为 0。", "",
        f"- baseline：{cell(summary['baseline_detection'])}",
        f"- 五变体全部存活："
        f"{cell(summary['survival_all_five_variants'])}",
        f"- 选中无效边：{summary['selected_invalid_edges']}", "",
        (
            "Go 只授权 P2 新来源准备。"
            if result["decision"] == "Go"
            else "No-Go：停止在当前开发资产上继续调规则，收口负结果。"
        ),
    ]
    _atomic_write(
        output / "DECISION.md",
        ("\n".join(decision) + "\n").encode("utf-8"),
    )


def run(args: argparse.Namespace) -> dict:
    output = args.output_root.resolve()
    if args.document_batch_size < 1:
        raise ValueError("document_batch_size must be positive")
    manifest = verify_inputs(output)
    detector = build_v4_detector(args)
    recovery = attach_recovery_audit(
        detector,
        output / "results/parser_recovery_rows.jsonl",
    )
    fingerprints = {
        "detector": detector.fingerprint,
        "bank": detector.scorer.fingerprint,
        "bank_sha256": detector.scorer.bank_sha256,
        "parser": detector.scorer.parser_fingerprint,
    }
    code = (
        "elastic_v2.py",
        "elastic_v2_development.py",
        "elastic_v3.py",
        "elastic_v3_development.py",
        "elastic_v4.py",
        "elastic_v4_development.py",
        "bank_detector.py",
        "bank_scorer.py",
        "amr_extractor.py",
        "sqlite_cache.py",
        "null_calibration.py",
        "models.py",
        "parser_recovery_audit.py",
    )
    runtime = freeze_runtime(
        output / "RUNTIME_MANIFEST.json",
        {
            "version": VERSION,
            "input_manifest_sha256": sha256_file(
                output / "INPUT_MANIFEST.json"
            ),
            "fingerprints": fingerprints,
            "code_sha256": {
                name: sha256_file(ROOT / "src/core" / name)
                for name in code
            },
            "configuration": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key != "command"
            },
        },
    )
    inputs = {
        name: read_rows(ROOT / item["path"])
        for name, item in manifest["outputs"].items()
    }
    calibration_rows = process(
        detector,
        recovery,
        inputs["calibration"],
        output / "results/calibration_rows.jsonl",
        output,
        "calibration",
        args.document_batch_size,
    )
    calibrator = fit_calibration(
        output,
        inputs["calibration"],
        calibration_rows,
        runtime,
    )
    development_rows = process(
        detector,
        recovery,
        inputs["development"],
        output / "results/development_rows.jsonl",
        output,
        "development",
        args.document_batch_size,
        calibrator,
    )
    summary = summarize(calibration_rows, development_rows)
    gates = {
        "baseline_detection_at_least_0_80": (
            summary["baseline_detection"]["rate"] >= 0.80
        ),
        "all_five_survival_at_least_0_80": (
            summary["survival_all_five_variants"]["rate"] is not None
            and summary["survival_all_five_variants"]["rate"] >= 0.80
        ),
        "selected_invalid_edges_zero": (
            summary["selected_invalid_edges"] == 0
        ),
        "all_rows_retained": summary["total_rows"] == 1210,
    }
    decision = "Go" if all(gates.values()) else "No-Go"
    result = {
        "version": VERSION,
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "decision_scope": "observed development assets only",
        "gates": gates,
        "summary": summary,
        "parser_recovery_audit": recovery.summary(),
        "extractor_stats": asdict(detector.scorer.extractor.stats),
        "fingerprints": fingerprints,
        "artifacts": {
            name: sha256_file(output / name)
            for name in (
                "INPUT_MANIFEST.json",
                "RUNTIME_MANIFEST.json",
                "results/CALIBRATION.json",
                "results/calibration_rows.jsonl",
                "results/development_rows.jsonl",
                "results/parser_recovery_rows.jsonl",
            )
        },
        "claim_limits": manifest["claim_limits"],
    }
    result_path = output / "RESULT.json"
    if result_path.exists():
        saved = read_json(result_path)
        comparable = lambda value: {
            key: item
            for key, item in value.items()
            if key != "completed_at"
        }
        if comparable(saved) != comparable(result):
            raise ValueError("P1b result differs from deterministic replay")
        result = saved
    else:
        _freeze(result_path, _json_bytes(result))
    write_reports(output, result)
    heartbeat(
        output,
        "complete",
        len(calibration_rows) + len(development_rows),
        len(calibration_rows) + len(development_rows),
        status="complete",
    )
    return result


def audit(output: Path) -> dict:
    manifest = verify_inputs(output)
    result = read_json(output / "RESULT.json")
    for name, digest in result["artifacts"].items():
        if sha256_file(output / name) != digest:
            raise ValueError(f"P1b artifact SHA differs: {name}")
    calibration_rows = read_rows(output / "results/calibration_rows.jsonl")
    development_rows = read_rows(output / "results/development_rows.jsonl")
    if len(calibration_rows) != manifest["outputs"]["calibration"]["rows"]:
        raise ValueError("P1b calibration result cardinality differs")
    if len(development_rows) != manifest["outputs"]["development"]["rows"]:
        raise ValueError("P1b development result cardinality differs")
    if summarize(calibration_rows, development_rows) != result["summary"]:
        raise ValueError("P1b aggregate replay differs")
    if result["summary"]["selected_invalid_edges"] != 0:
        raise ValueError("P1b selected an invalid edge")
    return {
        "status": "pass",
        "new_model_calls": 0,
        "decision": result["decision"],
        "result_sha256": sha256_file(output / "RESULT.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    run_parser.add_argument("--document-batch-size", type=int, default=16)
    run_parser.add_argument("--capitalization-gap-words", type=int, default=12)
    run_parser.add_argument("--long-gap-stride-words", type=int, default=40)
    add_runtime_arguments(run_parser)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()

    try:
        if args.command == "prepare":
            value = prepare(args.output_root.resolve())
        elif args.command == "audit":
            value = audit(args.output_root.resolve())
        else:
            value = run(args)
    except Exception as exc:
        if args.command == "run":
            _atomic_write(
                args.output_root.resolve() / "FAILURE.json",
                _json_bytes({
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "occurred_at": datetime.now(timezone.utc).isoformat(),
                }),
            )
        raise
    print(json.dumps(value, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
