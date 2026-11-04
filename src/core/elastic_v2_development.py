"""Run the frozen P1 development calibration and punctuation stress gate."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Sequence

from .bank_detector import add_runtime_arguments, build_detector
from .elastic_v2 import (
    ElasticBoundaryProposerV2,
    SwanBankLatticeDetectorV2,
    V2PathResult,
    recovery_edge_validity,
)
from .formal_null_evaluation import (
    ROOT,
    _atomic_write,
    _audit_payload,
    _freeze,
    _json_bytes,
    sha256_bytes,
    sha256_file,
)
from .models import DetectionResult
from .null_calibration import EmpiricalNullCalibrator
from .parser_recovery_audit import attach_recovery_audit
from .sqlite_cache import normalize_text


VERSION = "swan-elastic-v2-development:v1"
DEFAULT_OUTPUT = ROOT / "experiments/swan_elastic_v2_development1"
SOURCE = ROOT / "experiments/swan_elastic_length_boundary_audited2"
P0 = ROOT / "experiments/swan_elastic_v2_preflight1"
DEVELOPMENT_COHORTS = (
    "confirmation_baseline_positive",
    "confirmation_boundary_attack",
    "confirmation_punctuation_stress",
)
VARIANTS = (
    "semicolon_all",
    "semicolon_pairs",
    "comma_all",
    "newline_all",
    "space_all",
)
ABSTAIN_CALIBRATION_STATISTIC = -1.0e12


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def jsonl(rows: Sequence[dict]) -> bytes:
    return b"".join(
        (
            json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n"
        ).encode("utf-8")
        for row in rows
    )


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def verify_p0() -> dict:
    precheck = read_json(P0 / "PRECHECK.json")
    if precheck.get("decision") != "Go":
        raise ValueError("P0 precheck is not Go")
    expected = precheck["code_sha256"]["src/core/elastic_v2.py"]
    if sha256_file(ROOT / "src/core/elastic_v2.py") != expected:
        raise ValueError("v2 implementation differs from P0-bound code")
    return precheck


def prepare(output: Path) -> dict:
    verify_p0()
    calibration = read_rows(SOURCE / "inputs/calibration_windows.jsonl")
    source_stress = read_rows(SOURCE / "inputs/stress_rows.jsonl")
    development = [
        row for row in source_stress
        if row["cohort"] in DEVELOPMENT_COHORTS
    ]
    if len(calibration) != 1000 or len(development) != 210:
        raise ValueError("Unexpected P1 input cardinality")
    if len({row["window_sha256"] for row in calibration}) != 1000:
        raise ValueError("Duplicate calibration identity")
    if len({row["row_id"] for row in development}) != 210:
        raise ValueError("Duplicate development identity")

    outputs = {}
    for name, rows in (
        ("calibration", calibration),
        ("development", development),
    ):
        path = output / f"inputs/{name}_rows.jsonl"
        data = jsonl(rows)
        _freeze(path, data)
        outputs[name] = {
            "path": relative(path),
            "sha256": sha256_bytes(data),
            "bytes": len(data),
            "rows": len(rows),
        }

    dependencies = (
        SOURCE / "inputs/calibration_windows.jsonl",
        SOURCE / "inputs/stress_rows.jsonl",
        SOURCE / "INPUT_MANIFEST.json",
        P0 / "PRECHECK.json",
        P0 / "PROTOCOL.md",
        P0 / "DECISION.md",
    )
    manifest = {
        "version": VERSION,
        "status": "frozen_before_parser_execution",
        "outputs": outputs,
        "dependencies": {
            relative(path): sha256_file(path) for path in dependencies
        },
        "selection": {
            "calibration_rows": len(calibration),
            "development_rows": len(development),
            "development_cohorts": list(DEVELOPMENT_COHORTS),
            "test_used_for_fitting": False,
            "detector_score_used_for_selection": False,
        },
        "configuration": {
            "max_boundary_hops": 3,
            "max_span_words": 80,
            "capitalization_gap_words": 12,
            "long_gap_stride_words": 40,
            "score_center": 0.65,
            "green_threshold": 0.75,
            "alpha": 0.01,
            "abstain_calibration_statistic": ABSTAIN_CALIBRATION_STATISTIC,
        },
        "gates": {
            "minimum_baseline_detection": 0.80,
            "minimum_all_five_conditional_survival": 0.80,
            "selected_invalid_edges": 0,
            "missing_recovery_evidence": "stop",
        },
        "claim_limits": [
            "P1 inputs are observed development assets.",
            "P1 does not estimate independent FPR or formal TPR.",
            "P1 Go only authorizes fresh-source P2 preparation.",
        ],
    }
    _freeze(output / "INPUT_MANIFEST.json", _json_bytes(manifest))
    return manifest


def verify_inputs(output: Path) -> dict:
    verify_p0()
    manifest = read_json(output / "INPUT_MANIFEST.json")
    if (
        manifest.get("version") != VERSION
        or manifest.get("status") != "frozen_before_parser_execution"
    ):
        raise ValueError("P1 input manifest is not frozen")
    for name, digest in manifest["dependencies"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f"P1 dependency differs: {name}")
    for item in manifest["outputs"].values():
        path = ROOT / item["path"]
        if (
            sha256_file(path) != item["sha256"]
            or path.stat().st_size != item["bytes"]
            or len(read_rows(path)) != item["rows"]
        ):
            raise ValueError(f"P1 frozen input differs: {path}")
    return manifest


def freeze_runtime(path: Path, current: dict) -> dict:
    if path.exists():
        saved = read_json(path)
        comparable = {k: v for k, v in saved.items() if k != "created_at"}
        if comparable != current:
            raise ValueError("P1 runtime/configuration differs on resume")
        return saved
    value = {
        **current,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _freeze(path, _json_bytes(value))
    return value


def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def heartbeat(
    output: Path,
    stage: str,
    completed: int,
    total: int,
    status: str = "running",
) -> None:
    _atomic_write(
        output / "HEARTBEAT.json",
        _json_bytes({
            "stage": stage,
            "completed": completed,
            "total": total,
            "status": status,
            "pid": os.getpid(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }),
    )


def build_v2_detector(args: argparse.Namespace) -> SwanBankLatticeDetectorV2:
    base = build_detector(args)
    return SwanBankLatticeDetectorV2(
        builder=ElasticBoundaryProposerV2(
            max_boundary_hops=args.max_boundary_hops,
            max_span_words=args.max_span_words,
            capitalization_gap_words=args.capitalization_gap_words,
            long_gap_stride_words=args.long_gap_stride_words,
        ),
        scorer=base.scorer,
        score_center=args.score_center,
        green_threshold=args.green_threshold,
    )


def detection_payload(
    result: V2PathResult,
    lattice,
    validity: Sequence[bool],
) -> dict:
    valid_edges = {
        edge for edge, valid in zip(lattice.edges, validity) if valid
    }
    if any(unit.edge not in valid_edges for unit in result.units):
        raise RuntimeError("v2 selected an invalid recovery edge")
    return {
        "status": result.status,
        "statistic": result.statistic,
        "raw_utility": result.raw_utility,
        "unit_count": result.unit_count,
        "green_count": result.green_count,
        "invalid_edge_count": result.invalid_edge_count,
        "abstain_reason": result.abstain_reason,
        "selected_invalid_edges": 0,
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
    detector: SwanBankLatticeDetectorV2,
    recovery,
    source: dict,
    stage: str,
    calibrator: EmpiricalNullCalibrator = None,
) -> dict:
    text = source["window"] if stage == "calibration" else source["text"]
    lattice = detector.builder.build_lattice(text)
    result = detector.detect_lattice_with_recovery(
        lattice,
        recovery.records,
    )
    validity = recovery_edge_validity(lattice, recovery.records)
    normalized_spans = {
        normalize_text(edge.span_text) for edge in lattice.edges
    }
    records = tuple(
        detector.scorer._records[span]  # pylint: disable=protected-access
        for span in sorted(normalized_spans)
    )
    audit = _audit_payload(records)
    if not (
        audit["span_records"]
        == audit["graph_valid_records"]
        == audit["roundtrip_valid_records"]
        == len(normalized_spans)
    ):
        raise RuntimeError("P1 row has invalid or missing PENMAN evidence")
    detection = detection_payload(result, lattice, validity)
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
            raise ValueError("Development rows require v2 calibration")
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
        raise ValueError(f"P1 checkpoint has excess rows: {path}")
    identity = "window_sha256" if stage == "calibration" else "row_id"
    for index, row in enumerate(rows):
        if row.get(identity) != inputs[index].get(identity):
            raise ValueError(f"P1 checkpoint identity mismatch: {path}:{index + 1}")
    return rows


def process(
    detector: SwanBankLatticeDetectorV2,
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
    }
    if path.exists():
        calibrator = EmpiricalNullCalibrator.load(path)
        if calibrator.artifact.get("v2_provenance") != provenance:
            raise ValueError("P1 calibration provenance differs")
        return calibrator
    if len(inputs) != len(rows):
        raise ValueError("P1 calibration is incomplete")
    samples = []
    for source, row in zip(inputs, rows):
        if source["window_sha256"] != row["window_sha256"]:
            raise ValueError("P1 calibration source binding differs")
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
    calibrator.artifact["v2_provenance"] = provenance
    calibrator.save(path)
    return calibrator


def rate(successes: int, trials: int) -> dict:
    return {
        "successes": successes,
        "trials": trials,
        "rate": successes / trials if trials else None,
    }


def summarize(calibration_rows: Sequence[dict], rows: Sequence[dict]) -> dict:
    baseline = {
        row["group_id"]: row
        for row in rows
        if row["cohort"] == "confirmation_baseline_positive"
    }
    attacks = {
        row["group_id"]: row
        for row in rows
        if row["cohort"] == "confirmation_boundary_attack"
    }
    stress = [
        row for row in rows
        if row["cohort"] == "confirmation_punctuation_stress"
    ]
    detected = {
        group_id for group_id, row in baseline.items()
        if row["calibration"]["is_watermarked"]
    }
    attack_detected = {
        group_id for group_id, row in attacks.items()
        if row["calibration"]["is_watermarked"]
    }
    variant_summary = {}
    variant_survivors = {}
    for variant in VARIANTS:
        variant_rows = [
            row for row in stress if row["variant_id"] == variant
        ]
        survivors = {
            row["group_id"] for row in variant_rows
            if row["calibration"]["is_watermarked"]
        }
        variant_survivors[variant] = survivors
        variant_summary[variant] = {
            "detected": rate(len(survivors), len(variant_rows)),
            "survival_given_baseline_detected": rate(
                len(survivors & detected),
                len(detected),
            ),
        }
    all_five = set(detected)
    for survivors in variant_survivors.values():
        all_five &= survivors

    all_rows = list(calibration_rows) + list(rows)
    return {
        "calibration": {
            "rows": len(calibration_rows),
            "abstain": rate(
                sum(
                    row["detection"]["status"] == "abstain"
                    for row in calibration_rows
                ),
                len(calibration_rows),
            ),
        },
        "baseline_detection": rate(len(detected), len(baseline)),
        "historical_boundary_attack_detection": rate(
            len(attack_detected),
            len(attacks),
        ),
        "historical_boundary_survival": rate(
            len(attack_detected & detected),
            len(detected),
        ),
        "positive_punctuation": variant_summary,
        "survival_all_five_variants": rate(
            len(all_five),
            len(detected),
        ),
        "development_abstain": rate(
            sum(row["detection"]["status"] == "abstain" for row in rows),
            len(rows),
        ),
        "invalid_candidate_edges": sum(
            row["detection"]["invalid_edge_count"] for row in all_rows
        ),
        "selected_invalid_edges": sum(
            row["detection"]["selected_invalid_edges"] for row in all_rows
        ),
        "total_rows": len(all_rows),
    }


def write_reports(output: Path, result: dict) -> None:
    summary = result["summary"]

    def cell(item: dict) -> str:
        return (
            f"{item['successes']}/{item['trials']} ({item['rate']:.1%})"
            if item["trials"]
            else "n/a"
        )

    lines = [
        "# SWAN 弹性检测 v2：P1 开发验证", "",
        f"裁决：**{result['decision']}**。本裁决仅限开发门，不是正式 FPR/TPR 证据。", "",
        "| 指标 | 结果 |", "|---|---:|",
        f"| baseline 检出 | {cell(summary['baseline_detection'])} |",
        f"| 既有句界攻击条件存活 | {cell(summary['historical_boundary_survival'])} |",
        f"| 五变体全部条件存活 | {cell(summary['survival_all_five_variants'])} |",
        f"| calibration abstain | {cell(summary['calibration']['abstain'])} |",
        f"| development abstain | {cell(summary['development_abstain'])} |",
        f"| 无效候选边 | {summary['invalid_candidate_edges']} |",
        f"| 选中无效边 | {summary['selected_invalid_edges']} |", "",
        "| 变体 | 检出 | baseline 可检出条件下存活 |",
        "|---|---:|---:|",
    ]
    for variant in VARIANTS:
        item = summary["positive_punctuation"][variant]
        lines.append(
            f"| {variant} | {cell(item['detected'])} | "
            f"{cell(item['survival_given_baseline_detected'])} |"
        )
    recovery = result["parser_recovery_audit"]
    lines.extend([
        "", "## Recovery", "",
        f"绑定唯一 Span：{recovery['unique_spans']}；"
        f"选中 beam 含 ERROR 的 Span："
        f"{recovery['spans_with_selected_beam_errors']}。"
        "这些边在 DP 前已标为无效。",
        "", "## 证据边界", "",
        "全部正例与攻击均为已观察开发资产；development 未参与 calibration 拟合。",
        "本轮没有独立 Null test，因此不报告 FPR。",
        "P1 Go 只授权准备新来源 P2，不能升级正式防御 claim。",
    ])
    _atomic_write(
        output / "REPORT.md",
        ("\n".join(lines) + "\n").encode("utf-8"),
    )
    decision = [
        f"# P1 裁决：{result['decision']}", "",
        "主门：baseline 检出至少 80%，且 baseline 可检出条件下五变体全部存活至少 80%。",
        "",
        f"- baseline：{cell(summary['baseline_detection'])}",
        f"- 五变体全部存活：{cell(summary['survival_all_five_variants'])}",
        f"- 选中无效边：{summary['selected_invalid_edges']}",
        "",
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
    detector = build_v2_detector(args)
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
    recovery_summary = recovery.summary()
    result = {
        "version": VERSION,
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "decision_scope": "observed development assets only",
        "gates": gates,
        "summary": summary,
        "parser_recovery_audit": recovery_summary,
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
            raise ValueError("P1 result differs from deterministic replay")
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
            raise ValueError(f"P1 artifact SHA differs: {name}")
    calibration_rows = read_rows(output / "results/calibration_rows.jsonl")
    development_rows = read_rows(output / "results/development_rows.jsonl")
    if len(calibration_rows) != manifest["outputs"]["calibration"]["rows"]:
        raise ValueError("P1 calibration result cardinality differs")
    if len(development_rows) != manifest["outputs"]["development"]["rows"]:
        raise ValueError("P1 development result cardinality differs")
    if summarize(calibration_rows, development_rows) != result["summary"]:
        raise ValueError("P1 aggregate replay differs")
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
