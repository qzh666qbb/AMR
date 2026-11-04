from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Sequence

from .bank_detector import add_runtime_arguments, build_detector, detection_payload
from .formal_null_evaluation import (
    FORMAL_NULL_VERSION,
    _atomic_write,
    _audit_payload,
    _audit_summary,
    _clopper_pearson,
    _freeze,
    _json_bytes,
    sha256_bytes,
    sha256_file,
)
from .null_calibration import EmpiricalNullCalibrator
from .sqlite_cache import normalize_text


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_VERSION = "swan-elastic-positive-attack:v1"
DEFAULT_OUTPUT_ROOT = ROOT / "experiments/swan_elastic_positive_attack1"
DEFAULT_FORMAL_NULL_ROOT = ROOT / "experiments/swan_elastic_formal_null1"
CONFIRMATION_ROOT = ROOT / "experiments/confirmation_three_arm30"
NORMAL_LENGTH_ROOT = ROOT / "experiments/swan_normal_length_boundary_location1"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl_bytes(rows: Iterable[dict]) -> bytes:
    return b"".join(
        (
            json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _text_sha256(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def _base_row(
    cohort: str,
    cohort_index: int,
    row_id: str,
    group_id: str,
    variant_id: str,
    text: str,
    source_paths: Sequence[Path],
    metadata: dict,
) -> dict:
    return {
        "schema_version": 1,
        "cohort": cohort,
        "cohort_index": cohort_index,
        "row_id": row_id,
        "group_id": group_id,
        "variant_id": variant_id,
        "text": text,
        "text_sha256": _text_sha256(text),
        "normalized_text_sha256": sha256_bytes(
            normalize_text(text).encode("utf-8")
        ),
        "word_count": len(text.split()),
        "source_paths": [_relative(path) for path in source_paths],
        "source_sha256": {
            _relative(path): sha256_file(path) for path in source_paths
        },
        "metadata": metadata,
    }


def _confirmation_rows(source_root: Path) -> List[dict]:
    rows: List[dict] = []
    baselines = sorted((source_root / "baselines").glob("*.json"))
    if len(baselines) != 30:
        raise ValueError("Expected 30 confirmation baseline files")

    for index, path in enumerate(baselines):
        payload = _read_json(path)
        prompt_id = payload["prompt_id"]
        detection = payload["detection"]
        rows.append(
            _base_row(
                cohort="confirmation_baseline_positive",
                cohort_index=index,
                row_id=f"confirmation:{prompt_id}:baseline",
                group_id=prompt_id,
                variant_id="baseline",
                text=payload["text"],
                source_paths=(path,),
                metadata={
                    "historical_detector": {
                        "n": detection["n"],
                        "green": detection["green"],
                        "z": detection["z"],
                        "detected_at_1pct_theoretical": detection["z"] > 2.33,
                    },
                    "all_injection_accepted": payload.get(
                        "all_injection_accepted"
                    ),
                    "positive_denominator": "all_completed_baselines",
                },
            )
        )

    evaluated = sorted((source_root / "evaluated").glob("*-boundary_only.json"))
    if len(evaluated) != 30:
        raise ValueError("Expected 30 evaluated boundary_only files")

    for index, path in enumerate(evaluated):
        payload = _read_json(path)
        prompt_id = payload["prompt_id"]
        candidate = payload.get("no_feedback_candidate")
        rows_by_candidate = {row["candidate"]: row for row in payload["rows"]}
        if candidate not in rows_by_candidate:
            raise ValueError(f"Missing no-feedback candidate for {prompt_id}")
        selected = rows_by_candidate[candidate]
        if not selected.get("quality_valid"):
            raise ValueError(
                f"No-feedback boundary candidate is not quality-valid: {prompt_id}"
            )
        rows.append(
            _base_row(
                cohort="confirmation_boundary_attack",
                cohort_index=index,
                row_id=(
                    f"confirmation:{prompt_id}:boundary_only:"
                    f"candidate-{candidate}"
                ),
                group_id=prompt_id,
                variant_id=f"boundary_only_candidate_{candidate}",
                text=selected["attack"],
                source_paths=(path,),
                metadata={
                    "selection_rule": (
                        "no_feedback_candidate from frozen quality/evaluation "
                        "artifact; selected before this detector run"
                    ),
                    "candidate": candidate,
                    "quality_valid": selected.get("quality_valid"),
                    "legacy_no_feedback_success": payload.get(
                        "no_feedback_success"
                    ),
                    "legacy_feedback_success": payload.get("feedback_success"),
                    "source_text_sha256": selected.get("source_sha256"),
                    "legacy_detection": selected.get("detection", {}),
                },
            )
        )
    return rows


def _normal_length_rows(source_root: Path) -> List[dict]:
    path = source_root / "inputs.json"
    payload = _read_json(path)
    conditions = payload["conditions"]
    rows: List[dict] = []
    for index, condition in enumerate(("C0", "R0", "R1", "R4")):
        row = conditions[condition]
        role = {
            "C0": "baseline_positive",
            "R0": "boundary_location_attack",
            "R1": "boundary_location_attack",
            "R4": "restoration_control",
        }[condition]
        rows.append(
            _base_row(
                cohort="normal_length_boundary_case",
                cohort_index=index,
                row_id=f"normal_length_boundary:{condition}",
                group_id="normal-length-coherent-pilot1-00",
                variant_id=condition,
                text=row["text"],
                source_paths=(path,),
                metadata={
                    "condition_role": role,
                    "condition": condition,
                    "num_detection_units": row.get("num_detection_units"),
                    "legacy_text_sha256": row.get("text_sha256"),
                    "word_tokens_sha256": row.get("word_tokens_sha256"),
                },
            )
        )
    return rows


def prepare(
    output_root: Path,
    formal_null_root: Path = DEFAULT_FORMAL_NULL_ROOT,
    confirmation_root: Path = CONFIRMATION_ROOT,
    normal_length_root: Path = NORMAL_LENGTH_ROOT,
) -> dict:
    rows = _confirmation_rows(confirmation_root) + _normal_length_rows(
        normal_length_root
    )
    rows = [
        {**row, "input_index": index}
        for index, row in enumerate(rows)
    ]
    if len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate evaluation row_id")

    output_root = output_root.resolve()
    rows_path = output_root / "inputs/evaluation_rows.jsonl"
    rows_data = _jsonl_bytes(rows)
    _freeze(rows_path, rows_data)

    calibration_path = formal_null_root / "results/CALIBRATION.json"
    input_manifest_path = formal_null_root / "INPUT_MANIFEST.json"
    result_path = formal_null_root / "results/RESULT.json"
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)

    cohort_counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        cohort_counts[row["cohort"]] += 1

    manifest = {
        "version": EVALUATION_VERSION,
        "status": "frozen_before_parser_execution",
        "selection_rule": {
            "confirmation_baseline_positive": (
                "all 30 completed confirmation_three_arm30 baselines, "
                "sorted by prompt_id"
            ),
            "confirmation_boundary_attack": (
                "one frozen boundary_only no-feedback quality-valid candidate "
                "per prompt_id; no selection by this detector"
            ),
            "normal_length_boundary_case": (
                "C0/R0/R1/R4 from swan_normal_length_boundary_location1"
            ),
        },
        "formal_null": {
            "version": FORMAL_NULL_VERSION,
            "root": _relative(formal_null_root),
            "input_manifest_sha256": sha256_file(input_manifest_path),
            "result_sha256": sha256_file(result_path),
            "calibration_artifact_path": _relative(calibration_path),
            "calibration_artifact_sha256": sha256_file(calibration_path),
        },
        "sources": {
            "confirmation_three_arm30": {
                "path": _relative(confirmation_root),
                "final_analysis_sha256": sha256_file(
                    confirmation_root / "final_analysis.json"
                ),
                "audit_sha256": sha256_file(confirmation_root / "audit.json"),
            },
            "swan_normal_length_boundary_location1": {
                "path": _relative(normal_length_root),
                "inputs_sha256": sha256_file(normal_length_root / "inputs.json"),
                "detection_sha256": sha256_file(
                    normal_length_root / "detection.json"
                ),
            },
        },
        "outputs": {
            "evaluation_rows": {
                "path": _relative(rows_path),
                "sha256": sha256_bytes(rows_data),
                "bytes": len(rows_data),
                "rows": len(rows),
            },
        },
        "cohort_counts": dict(sorted(cohort_counts.items())),
    }
    _freeze(output_root / "INPUT_MANIFEST.json", _json_bytes(manifest))
    return manifest


def _load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL {path}:{line_number}") from exc
    return rows


def _load_manifest(output_root: Path) -> dict:
    path = output_root.resolve() / "INPUT_MANIFEST.json"
    manifest = _read_json(path)
    if manifest.get("version") != EVALUATION_VERSION:
        raise ValueError("Unsupported positive/attack manifest version")
    if manifest.get("status") != "frozen_before_parser_execution":
        raise ValueError("Evaluation input manifest is not frozen")
    rows_meta = manifest["outputs"]["evaluation_rows"]
    rows_path = ROOT / rows_meta["path"]
    if not rows_path.is_file():
        rows_path = Path(rows_meta["path"])
    if (
        sha256_file(rows_path) != rows_meta["sha256"]
        or rows_path.stat().st_size != rows_meta["bytes"]
    ):
        raise ValueError("Frozen evaluation rows changed")
    calibration = manifest["formal_null"]
    calibration_path = ROOT / calibration["calibration_artifact_path"]
    if not calibration_path.is_file():
        calibration_path = Path(calibration["calibration_artifact_path"])
    if sha256_file(calibration_path) != calibration["calibration_artifact_sha256"]:
        raise ValueError("Frozen calibration artifact changed")
    return manifest


def _input_rows_path(manifest: dict) -> Path:
    path = Path(manifest["outputs"]["evaluation_rows"]["path"])
    return path if path.is_absolute() else ROOT / path


def _calibration_path(manifest: dict) -> Path:
    path = Path(manifest["formal_null"]["calibration_artifact_path"])
    return path if path.is_absolute() else ROOT / path


def _load_checkpoint(path: Path, inputs: Sequence[dict]) -> List[dict]:
    if not path.exists():
        return []
    rows = _load_jsonl(path)
    if len(rows) > len(inputs):
        raise ValueError(f"Checkpoint has excess rows: {path}")
    for index, row in enumerate(rows):
        expected = inputs[index]
        if (
            row.get("input_index") != index
            or row.get("row_id") != expected["row_id"]
            or row.get("text_sha256") != expected["text_sha256"]
        ):
            raise ValueError(f"Checkpoint identity mismatch at {path}:{index + 1}")
    return rows


def _score_row(detector, input_row: dict, calibrator, alpha: float) -> dict:
    text = input_row["text"]
    lattice = detector.builder.build_lattice(text)
    result = detector.detect_lattice(lattice)
    normalized_spans = {normalize_text(edge.span_text) for edge in lattice.edges}
    records = tuple(
        detector.scorer._records[span]  # pylint: disable=protected-access
        for span in sorted(normalized_spans)
    )
    audit = _audit_payload(records)
    if (
        audit["span_records"] != len(normalized_spans)
        or audit["graph_valid_records"] != audit["span_records"]
        or audit["roundtrip_valid_records"] != audit["span_records"]
    ):
        raise RuntimeError("Document contains an unaudited or invalid AMR span")

    compact_detection = detection_payload(result)
    for unit in compact_detection["units"]:
        unit["span_sha256"] = _text_sha256(unit.pop("span_text"))

    calibration = asdict(
        calibrator.calibrate(
            statistic=result.statistic,
            word_count=input_row["word_count"],
            alpha=alpha,
        )
    )
    return {
        "schema_version": 1,
        "input_index": input_row["input_index"],
        "cohort": input_row["cohort"],
        "cohort_index": input_row["cohort_index"],
        "row_id": input_row["row_id"],
        "group_id": input_row["group_id"],
        "variant_id": input_row["variant_id"],
        "text_sha256": input_row["text_sha256"],
        "normalized_text_sha256": input_row["normalized_text_sha256"],
        "word_count": input_row["word_count"],
        "candidate_edges": len(lattice.edges),
        "unique_candidate_spans": len(normalized_spans),
        "metadata": input_row["metadata"],
        "detection": compact_detection,
        "calibration": calibration,
        "penman_audit": audit,
    }


def _runtime_manifest(args: argparse.Namespace, detector) -> dict:
    files = [
        ROOT / "src/core/positive_attack_evaluation.py",
        ROOT / "src/core/bank_detector.py",
        ROOT / "src/core/bank_scorer.py",
        ROOT / "src/core/amr_extractor.py",
        ROOT / "src/core/sqlite_cache.py",
        ROOT / "src/core/null_calibration.py",
        ROOT / "src/core/formal_null_evaluation.py",
    ]
    return {
        "version": EVALUATION_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_sha256": {
            _relative(path): sha256_file(path) for path in files
        },
        "parameters": {
            "alpha": args.alpha,
            "max_boundary_hops": args.max_boundary_hops,
            "max_span_words": args.max_span_words,
            "score_center": args.score_center,
            "green_threshold": args.green_threshold,
            "backend": args.backend,
            "model_batch_size": args.model_batch_size,
            "request_batch_size": args.request_batch_size,
            "document_batch_size": args.document_batch_size,
        },
        "fingerprints": {
            "detector": detector.fingerprint,
            "bank": detector.scorer.fingerprint,
            "bank_sha256": detector.scorer.bank_sha256,
            "parser": detector.scorer.parser_fingerprint,
        },
    }


def _rate(successes: int, trials: int) -> dict:
    return {
        "successes": successes,
        "trials": trials,
        "rate": successes / trials if trials else None,
        "confidence_interval": _clopper_pearson(successes, trials)
        if trials
        else None,
    }


def _cohort_summary(rows: Sequence[dict]) -> dict:
    detected = sum(row["calibration"]["is_watermarked"] for row in rows)
    stats = [row["detection"]["statistic"] for row in rows]
    p_values = [row["calibration"]["p_value"] for row in rows]
    return {
        "documents": len(rows),
        "detected": detected,
        "tpr_or_detection_rate": _rate(detected, len(rows)),
        "statistic": {
            "min": min(stats) if stats else None,
            "median": median(stats) if stats else None,
            "max": max(stats) if stats else None,
        },
        "p_value": {
            "min": min(p_values) if p_values else None,
            "median": median(p_values) if p_values else None,
            "max": max(p_values) if p_values else None,
        },
        "calibration_scopes": dict(
            sorted(
                {
                    scope: sum(row["calibration"]["scope"] == scope for row in rows)
                    for scope in {row["calibration"]["scope"] for row in rows}
                }.items()
            )
        ),
    }


def summarize(rows: Sequence[dict], alpha: float) -> dict:
    by_cohort: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_cohort[row["cohort"]].append(row)

    baseline = {
        row["group_id"]: row
        for row in by_cohort.get("confirmation_baseline_positive", [])
    }
    attack = {
        row["group_id"]: row
        for row in by_cohort.get("confirmation_boundary_attack", [])
    }
    paired_ids = sorted(set(baseline) & set(attack))
    baseline_detected_ids = [
        group_id
        for group_id in paired_ids
        if baseline[group_id]["calibration"]["is_watermarked"]
    ]
    attack_detected_ids = [
        group_id
        for group_id in paired_ids
        if attack[group_id]["calibration"]["is_watermarked"]
    ]
    robust_ids = [
        group_id
        for group_id in baseline_detected_ids
        if attack[group_id]["calibration"]["is_watermarked"]
    ]
    evaded_ids = [
        group_id
        for group_id in baseline_detected_ids
        if not attack[group_id]["calibration"]["is_watermarked"]
    ]
    historical_positive_ids = [
        group_id
        for group_id, row in baseline.items()
        if row["metadata"]["historical_detector"][
            "detected_at_1pct_theoretical"
        ]
    ]
    historical_positive_detected = [
        group_id
        for group_id in historical_positive_ids
        if baseline[group_id]["calibration"]["is_watermarked"]
    ]

    normal_case = {
        row["variant_id"]: {
            "is_watermarked": row["calibration"]["is_watermarked"],
            "p_value": row["calibration"]["p_value"],
            "statistic": row["detection"]["statistic"],
            "unit_count": row["detection"]["unit_count"],
            "scope": row["calibration"]["scope"],
        }
        for row in by_cohort.get("normal_length_boundary_case", [])
    }

    return {
        "alpha": alpha,
        "cohorts": {
            cohort: _cohort_summary(cohort_rows)
            for cohort, cohort_rows in sorted(by_cohort.items())
        },
        "confirmation_pairing": {
            "paired_documents": len(paired_ids),
            "baseline_detected": _rate(len(baseline_detected_ids), len(paired_ids)),
            "baseline_detected_among_historical_positive": _rate(
                len(historical_positive_detected),
                len(historical_positive_ids),
            ),
            "attack_detected": _rate(len(attack_detected_ids), len(paired_ids)),
            "robust_survival_given_baseline_detected": _rate(
                len(robust_ids),
                len(baseline_detected_ids),
            ),
            "evasion_given_baseline_detected": _rate(
                len(evaded_ids),
                len(baseline_detected_ids),
            ),
            "evaded_group_ids": evaded_ids,
        },
        "normal_length_case": normal_case,
        "penman_audit": {
            cohort: _audit_summary(cohort_rows)
            for cohort, cohort_rows in sorted(by_cohort.items())
        },
    }


def _write_report(path: Path, result: dict) -> None:
    pairing = result["summary"]["confirmation_pairing"]
    cohorts = result["summary"]["cohorts"]
    normal = result["summary"]["normal_length_case"]
    lines = [
        "# SWAN Elastic Positive And Boundary Attack Evaluation",
        "",
        f"Status: `{result['status']}`",
        "",
        "## Main 30-Document Confirmation Set",
        "",
        "| Metric | Value |",
        "|---|---:|",
        (
            "| Baseline positive detection | "
            f"{pairing['baseline_detected']['successes']}/"
            f"{pairing['baseline_detected']['trials']} |"
        ),
        (
            "| Baseline detection among historical positives | "
            f"{pairing['baseline_detected_among_historical_positive']['successes']}/"
            f"{pairing['baseline_detected_among_historical_positive']['trials']} |"
        ),
        (
            "| Boundary attack detected | "
            f"{pairing['attack_detected']['successes']}/"
            f"{pairing['attack_detected']['trials']} |"
        ),
        (
            "| Robust survival given baseline detected | "
            f"{pairing['robust_survival_given_baseline_detected']['successes']}/"
            f"{pairing['robust_survival_given_baseline_detected']['trials']} |"
        ),
        (
            "| Evasion given baseline detected | "
            f"{pairing['evasion_given_baseline_detected']['successes']}/"
            f"{pairing['evasion_given_baseline_detected']['trials']} |"
        ),
        "",
        "## Cohort Rates",
        "",
        "| Cohort | Detected | Rate | Median p | Median statistic | Scope |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for cohort, summary in cohorts.items():
        rate = summary["tpr_or_detection_rate"]
        scopes = ", ".join(
            f"{key}:{value}"
            for key, value in summary["calibration_scopes"].items()
        )
        lines.append(
            f"| {cohort} | {rate['successes']}/{rate['trials']} | "
            f"{rate['rate']:.6f} | {summary['p_value']['median']:.6f} | "
            f"{summary['statistic']['median']:.6f} | {scopes} |"
        )
    lines.extend(
        [
            "",
            "## Normal-Length Boundary Case",
            "",
            "| Condition | Detected | p-value | Statistic | Units | Scope |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for condition in ("C0", "R0", "R1", "R4"):
        row = normal.get(condition)
        if row is None:
            continue
        lines.append(
            f"| {condition} | {str(row['is_watermarked']).lower()} | "
            f"{row['p_value']:.6f} | {row['statistic']:.6f} | "
            f"{row['unit_count']} | {row['scope']} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This run uses the frozen formal calibration artifact only for inference. "
            "It does not refit the empirical Null or select attack candidates by "
            "the new detector result. The 30-document confirmation set is short "
            "text and therefore uses global fallback calibration because the "
            "formal Null has no 0-80 word bucket samples.",
            "",
            "## Artifacts",
            "",
        ]
    )
    for key, value in result["artifacts"].items():
        if key.endswith("_sha256"):
            lines.append(f"- {key}: `{value}`")
    _atomic_write(path, ("\n".join(lines) + "\n").encode("utf-8"))


def run(args: argparse.Namespace) -> dict:
    output_root = args.output_root.resolve()
    manifest = _load_manifest(output_root)
    input_rows = _load_jsonl(_input_rows_path(manifest))
    calibration_path = _calibration_path(manifest)

    detector = build_detector(args)
    calibrator = EmpiricalNullCalibrator.load(calibration_path)
    calibrator.assert_compatible(
        detector_fingerprint=detector.fingerprint,
        bank_fingerprint=detector.scorer.fingerprint,
        parser_fingerprint=detector.scorer.parser_fingerprint,
    )

    runtime = _runtime_manifest(args, detector)
    _freeze(output_root / "RUNTIME_MANIFEST.json", _json_bytes(runtime))

    rows_path = output_root / "results/evaluation_rows.jsonl"
    completed = _load_checkpoint(rows_path, input_rows)
    pending = input_rows[len(completed):]
    for start in range(0, len(pending), args.document_batch_size):
        batch = pending[start:start + args.document_batch_size]
        batch_spans = []
        for input_row in batch:
            lattice = detector.builder.build_lattice(input_row["text"])
            batch_spans.extend(edge.span_text for edge in lattice.edges)
        detector.scorer.prepare(batch_spans)
        for input_row in batch:
            row = _score_row(detector, input_row, calibrator, args.alpha)
            _atomic_append(rows_path, row)
            completed.append(row)
            print(
                json.dumps(
                    {
                        "completed": len(completed),
                        "total": len(input_rows),
                        "cohort": row["cohort"],
                        "row_id": row["row_id"],
                        "p_value": row["calibration"]["p_value"],
                        "is_watermarked": row["calibration"][
                            "is_watermarked"
                        ],
                        "warning_spans": row["penman_audit"][
                            "warning_span_records"
                        ],
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
        detector.scorer.clear_prepared()

    summary = summarize(completed, args.alpha)
    result = {
        "version": EVALUATION_VERSION,
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest_sha256": sha256_file(output_root / "INPUT_MANIFEST.json"),
        "runtime_manifest_sha256": sha256_file(
            output_root / "RUNTIME_MANIFEST.json"
        ),
        "summary": summary,
        "fingerprints": runtime["fingerprints"],
        "artifacts": {
            "input_manifest_sha256": sha256_file(
                output_root / "INPUT_MANIFEST.json"
            ),
            "runtime_manifest_sha256": sha256_file(
                output_root / "RUNTIME_MANIFEST.json"
            ),
            "evaluation_input_rows_sha256": manifest["outputs"][
                "evaluation_rows"
            ]["sha256"],
            "evaluation_result_rows_sha256": sha256_file(rows_path),
            "calibration_artifact_sha256": manifest["formal_null"][
                "calibration_artifact_sha256"
            ],
        },
    }
    _atomic_write(output_root / "results/RESULT.json", _json_bytes(result))
    _write_report(output_root / "results/REPORT.md", result)
    return result


def _atomic_append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        stream.flush()
        import os

        os.fsync(stream.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen SWAN positives and sentence-boundary attacks with "
            "the formal empirical calibration artifact"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    prepare_parser.add_argument(
        "--formal-null-root", type=Path, default=DEFAULT_FORMAL_NULL_ROOT
    )
    prepare_parser.add_argument(
        "--confirmation-root", type=Path, default=CONFIRMATION_ROOT
    )
    prepare_parser.add_argument(
        "--normal-length-root", type=Path, default=NORMAL_LENGTH_ROOT
    )

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run_parser.add_argument("--alpha", type=float, default=0.01)
    run_parser.add_argument("--document-batch-size", type=int, default=8)
    add_runtime_arguments(run_parser)

    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    all_parser.add_argument(
        "--formal-null-root", type=Path, default=DEFAULT_FORMAL_NULL_ROOT
    )
    all_parser.add_argument(
        "--confirmation-root", type=Path, default=CONFIRMATION_ROOT
    )
    all_parser.add_argument(
        "--normal-length-root", type=Path, default=NORMAL_LENGTH_ROOT
    )
    all_parser.add_argument("--alpha", type=float, default=0.01)
    all_parser.add_argument("--document-batch-size", type=int, default=8)
    add_runtime_arguments(all_parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        manifest = prepare(
            output_root=args.output_root,
            formal_null_root=args.formal_null_root,
            confirmation_root=args.confirmation_root,
            normal_length_root=args.normal_length_root,
        )
        print(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True))
    elif args.command == "all":
        prepare(
            output_root=args.output_root,
            formal_null_root=args.formal_null_root,
            confirmation_root=args.confirmation_root,
            normal_length_root=args.normal_length_root,
        )
        result = run(args)
        print(json.dumps(result["summary"], ensure_ascii=True, indent=2))
    else:
        result = run(args)
        print(json.dumps(result["summary"], ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
