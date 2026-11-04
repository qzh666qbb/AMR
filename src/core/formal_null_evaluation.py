from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .bank_detector import (
    add_runtime_arguments,
    build_detector,
    detection_payload,
)
from .lattice_builder import LatticeBuilder
from .models import DetectionResult
from .null_calibration import EmpiricalNullCalibrator
from .sqlite_cache import normalize_text


ROOT = Path(__file__).resolve().parents[2]
FORMAL_NULL_VERSION = "swan-elastic-formal-null:v1"
DATASET_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
DEFAULT_INPUT_ROOT = ROOT / "experiments/swan_count_neutral_final1/sealed"
DEFAULT_OUTPUT_ROOT = ROOT / "experiments/swan_elastic_formal_null1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=str(path.parent),
        prefix=path.name + ".",
        delete=False,
    ) as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _freeze(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Frozen artifact differs: {path}")
        return
    _atomic_write(path, data)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def select_prefix_window(text: str) -> Optional[dict]:
    normalized = text.strip()
    builder = LatticeBuilder(include_soft_boundaries=False)
    boundaries = builder.candidate_boundaries(normalized)
    for hard_units, boundary in enumerate(boundaries[1:], start=1):
        prefix = normalized[:boundary.char_offset].strip()
        word_count = len(prefix.split())
        if word_count > 220 or hard_units > 24:
            return None
        if word_count >= 100 and hard_units >= 8:
            return {
                "text": prefix,
                "text_sha256": sha256_bytes(prefix.encode("utf-8")),
                "word_count": word_count,
                "hard_unit_count": hard_units,
                "char_end": boundary.char_offset,
            }
    return None


def _prepare_cohort(
    source_path: Path,
    cohort: str,
    sample_count: int,
) -> Tuple[List[dict], dict]:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Source allocation is not a list: {source_path}")
    selected: List[dict] = []
    eligible = 0
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("Source allocation contains a non-object row")
        window = select_prefix_window(row["text"])
        if window is None:
            continue
        eligible += 1
        if len(selected) >= sample_count:
            continue
        selected.append(
            {
                "schema_version": 1,
                "cohort": cohort,
                "cohort_index": len(selected),
                "allocation_id": row["allocation_id"],
                "allocation_ordinal": int(row["ordinal"]),
                "locator": row["locator"],
                "normalized_source_sha256": row[
                    "normalized_source_sha256"
                ],
                "url_sha256": row["url_sha256"],
                "source_text_sha256": sha256_bytes(
                    row["text"].encode("utf-8")
                ),
                "window": window["text"],
                "window_sha256": window["text_sha256"],
                "word_count": window["word_count"],
                "hard_unit_count": window["hard_unit_count"],
                "char_end": window["char_end"],
            }
        )
    if len(selected) != sample_count:
        raise ValueError(
            f"{cohort} has {len(selected)} eligible rows, expected {sample_count}"
        )
    summary = {
        "allocated_rows": len(payload),
        "eligible_rows": eligible,
        "selected_rows": len(selected),
        "last_selected_allocation_ordinal": selected[-1][
            "allocation_ordinal"
        ],
        "mean_word_count": sum(row["word_count"] for row in selected)
        / len(selected),
        "mean_hard_unit_count": sum(
            row["hard_unit_count"] for row in selected
        )
        / len(selected),
    }
    return selected, summary


def _jsonl_bytes(rows: Iterable[dict]) -> bytes:
    return b"".join(
        (
            json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n"
        ).encode("utf-8")
        for row in rows
    )


def prepare(
    calibration_source: Path,
    test_source: Path,
    output_root: Path,
    sample_count: int = 1000,
) -> dict:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    sources = {
        "calibration": calibration_source.resolve(),
        "independent_test": test_source.resolve(),
    }
    cohorts: Dict[str, List[dict]] = {}
    summaries = {}
    for cohort, source in sources.items():
        cohorts[cohort], summaries[cohort] = _prepare_cohort(
            source, cohort, sample_count
        )

    identity_fields = (
        "allocation_id",
        "locator",
        "normalized_source_sha256",
        "source_text_sha256",
        "url_sha256",
        "window_sha256",
    )
    overlaps = {}
    for field in identity_fields:
        left = {row[field] for row in cohorts["calibration"]}
        right = {row[field] for row in cohorts["independent_test"]}
        overlaps[field] = len(left & right)
        if len(left) != sample_count or len(right) != sample_count:
            raise ValueError(f"Duplicate {field} within a formal cohort")
        if overlaps[field]:
            raise ValueError(f"Formal cohorts overlap on {field}")

    input_dir = output_root.resolve() / "inputs"
    output_metadata = {}
    for cohort, rows in cohorts.items():
        path = input_dir / f"{cohort}_windows.jsonl"
        data = _jsonl_bytes(rows)
        _freeze(path, data)
        output_metadata[cohort] = {
            "path": _relative(path),
            "sha256": sha256_bytes(data),
            "bytes": len(data),
            "rows": len(rows),
        }

    manifest = {
        "version": FORMAL_NULL_VERSION,
        "status": "frozen_before_parser_execution",
        "dataset": {
            "name": "allenai/c4 realnewslike",
            "revision": DATASET_REVISION,
            "sources": {
                cohort: {
                    "path": _relative(path),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "rows": 4000,
                }
                for cohort, path in sources.items()
            },
        },
        "selection_rule": {
            "order": "allocation ordinal",
            "prefix": "first hard-boundary prefix reaching both minima",
            "minimum_whitespace_words": 100,
            "minimum_hard_units": 8,
            "maximum_whitespace_words": 220,
            "maximum_hard_units": 24,
            "detector_or_parser_features_used": False,
            "replacement_after_selection": False,
        },
        "sample_count_per_cohort": sample_count,
        "summaries": summaries,
        "identity_overlap_counts": overlaps,
        "outputs": output_metadata,
    }
    manifest_path = output_root.resolve() / "INPUT_MANIFEST.json"
    _freeze(manifest_path, _json_bytes(manifest))
    return manifest


def _load_manifest(output_root: Path) -> dict:
    path = output_root.resolve() / "INPUT_MANIFEST.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != FORMAL_NULL_VERSION:
        raise ValueError("Unsupported formal Null manifest version")
    if manifest.get("status") != "frozen_before_parser_execution":
        raise ValueError("Formal Null input manifest is not frozen")
    for cohort, metadata in manifest["outputs"].items():
        input_path = ROOT / metadata["path"]
        if not input_path.is_file():
            input_path = Path(metadata["path"])
        if (
            sha256_file(input_path) != metadata["sha256"]
            or input_path.stat().st_size != metadata["bytes"]
        ):
            raise ValueError(f"Frozen {cohort} input changed")
    if any(manifest["identity_overlap_counts"].values()):
        raise ValueError("Calibration and test identities are not disjoint")
    return manifest


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL {path}:{line_number}") from exc
            rows.append(row)
    return rows


def _input_path(manifest: dict, cohort: str) -> Path:
    path = Path(manifest["outputs"][cohort]["path"])
    return path if path.is_absolute() else ROOT / path


def _load_checkpoint(
    path: Path,
    inputs: Sequence[dict],
    cohort: str,
) -> List[dict]:
    if not path.exists():
        return []
    rows = _read_jsonl(path)
    if len(rows) > len(inputs):
        raise ValueError(f"Checkpoint has excess rows: {path}")
    for index, row in enumerate(rows):
        expected = inputs[index]
        if (
            row.get("cohort") != cohort
            or row.get("cohort_index") != index
            or row.get("window_sha256") != expected["window_sha256"]
            or row.get("allocation_id") != expected["allocation_id"]
        ):
            raise ValueError(f"Checkpoint identity mismatch at {path}:{index + 1}")
    return rows


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _audit_payload(records: Sequence[object]) -> dict:
    warning_details = []
    for record in records:
        if record.penman_warnings:
            warning_details.append(
                {
                    "text_sha256": record.text_sha256,
                    "warnings": list(record.penman_warnings),
                }
            )
    return {
        "span_records": len(records),
        "graph_valid_records": sum(record.graph_valid for record in records),
        "roundtrip_valid_records": sum(
            record.roundtrip_valid for record in records
        ),
        "warning_span_records": len(warning_details),
        "warning_records": sum(
            len(row["warnings"]) for row in warning_details
        ),
        "warning_details": warning_details,
    }


def _score_row(detector, input_row: dict, calibrator=None, alpha=0.01) -> dict:
    text = input_row["window"]
    lattice = detector.builder.build_lattice(text)
    result = detector.detect_lattice(lattice)
    normalized_spans = {
        normalize_text(edge.span_text) for edge in lattice.edges
    }
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
        unit["span_sha256"] = sha256_bytes(
            unit.pop("span_text").encode("utf-8")
        )
    row = {
        "schema_version": 1,
        "cohort": input_row["cohort"],
        "cohort_index": input_row["cohort_index"],
        "allocation_id": input_row["allocation_id"],
        "allocation_ordinal": input_row["allocation_ordinal"],
        "window_sha256": input_row["window_sha256"],
        "word_count": input_row["word_count"],
        "hard_unit_count": input_row["hard_unit_count"],
        "candidate_edges": len(lattice.edges),
        "unique_candidate_spans": len(normalized_spans),
        "detection": compact_detection,
        "penman_audit": audit,
    }
    if calibrator is not None:
        row["calibration"] = asdict(
            calibrator.calibrate(
                statistic=result.statistic,
                word_count=input_row["word_count"],
                alpha=alpha,
            )
        )
    return row


def _process_cohort(
    detector,
    inputs: Sequence[dict],
    cohort: str,
    path: Path,
    calibrator=None,
    alpha: float = 0.01,
    document_batch_size: int = 16,
) -> List[dict]:
    if document_batch_size < 1:
        raise ValueError("document_batch_size must be positive")
    completed = _load_checkpoint(path, inputs, cohort)
    pending = inputs[len(completed):]
    for start in range(0, len(pending), document_batch_size):
        document_batch = pending[start:start + document_batch_size]
        batch_spans = []
        for input_row in document_batch:
            lattice = detector.builder.build_lattice(input_row["window"])
            batch_spans.extend(edge.span_text for edge in lattice.edges)
        detector.scorer.prepare(batch_spans)
        for input_row in document_batch:
            row = _score_row(detector, input_row, calibrator, alpha)
            _append_row(path, row)
            completed.append(row)
            print(
                json.dumps(
                    {
                        "cohort": cohort,
                        "completed": len(completed),
                        "total": len(inputs),
                        "statistic": row["detection"]["statistic"],
                        "warning_spans": row["penman_audit"][
                            "warning_span_records"
                        ],
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
        detector.scorer.clear_prepared()
    return completed


def _detection_from_row(row: dict) -> DetectionResult:
    detection = row["detection"]
    return DetectionResult(
        statistic=float(detection["statistic"]),
        raw_utility=float(detection["raw_utility"]),
        unit_count=int(detection["unit_count"]),
        green_count=int(detection["green_count"]),
        units=(),
    )


def _fit_or_load_calibrator(
    path: Path,
    detector,
    inputs: Sequence[dict],
    rows: Sequence[dict],
    manifest_path: Path,
    runtime_manifest_path: Path,
    rows_path: Path,
) -> EmpiricalNullCalibrator:
    provenance = {
        "formal_null_version": FORMAL_NULL_VERSION,
        "input_manifest_sha256": sha256_file(manifest_path),
        "runtime_manifest_sha256": sha256_file(runtime_manifest_path),
        "calibration_input_sha256": sha256_file(
            manifest_path.parent / "inputs/calibration_windows.jsonl"
        ),
        "calibration_rows_sha256": sha256_file(rows_path),
        "selection_frozen_before_parser_execution": True,
        "independent_test_used_for_fitting": False,
    }
    if path.exists():
        calibrator = EmpiricalNullCalibrator.load(path)
        if calibrator.artifact.get("formal_provenance") != provenance:
            raise ValueError("Frozen calibration provenance mismatch")
    else:
        samples = (
            (input_row["window"], _detection_from_row(row))
            for input_row, row in zip(inputs, rows)
        )
        calibrator = EmpiricalNullCalibrator.fit(
            samples=samples,
            detector_fingerprint=detector.fingerprint,
            bank_fingerprint=detector.scorer.fingerprint,
            bank_sha256=detector.scorer.bank_sha256,
            parser_fingerprint=detector.scorer.parser_fingerprint,
            min_bucket_size=20,
        )
        calibrator.artifact["formal_provenance"] = provenance
        calibrator.save(path)
    calibrator.assert_compatible(
        detector_fingerprint=detector.fingerprint,
        bank_fingerprint=detector.scorer.fingerprint,
        parser_fingerprint=detector.scorer.parser_fingerprint,
    )
    return calibrator


def _freeze_runtime_manifest(
    path: Path,
    input_manifest_path: Path,
    detector,
    args: argparse.Namespace,
) -> dict:
    code_paths = (
        Path(__file__),
        ROOT / "src/core/models.py",
        ROOT / "src/core/amr_extractor.py",
        ROOT / "src/core/sqlite_cache.py",
        ROOT / "src/core/lattice_builder.py",
        ROOT / "src/core/bank_scorer.py",
        ROOT / "src/core/bank_detector.py",
        ROOT / "src/core/null_calibration.py",
    )
    runtime = {
        "version": FORMAL_NULL_VERSION,
        "status": "frozen_before_parser_execution",
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "code": {
            _relative(code_path): {
                "sha256": sha256_file(code_path),
                "bytes": code_path.stat().st_size,
            }
            for code_path in code_paths
        },
        "configuration": {
            "alpha": args.alpha,
            "backend": args.backend,
            "cache_db": str(Path(args.cache_db).resolve()),
            "device": args.device,
            "document_batch_size": args.document_batch_size,
            "green_threshold": args.green_threshold,
            "max_boundary_hops": args.max_boundary_hops,
            "max_span_words": args.max_span_words,
            "model_batch_size": args.model_batch_size,
            "request_batch_size": args.request_batch_size,
            "score_center": args.score_center,
        },
        "fingerprints": {
            "detector": detector.fingerprint,
            "bank": detector.scorer.fingerprint,
            "bank_sha256": detector.scorer.bank_sha256,
            "parser": detector.scorer.parser_fingerprint,
        },
    }
    _freeze(path, _json_bytes(runtime))
    return runtime


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    maximum_iterations = 300
    epsilon = 3e-14
    minimum = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < minimum:
        d = minimum
    d = 1.0 / d
    value = d
    for iteration in range(1, maximum_iterations + 1):
        twice = 2 * iteration
        coefficient = (
            iteration * (b - iteration) * x
            / ((qam + twice) * (a + twice))
        )
        d = 1.0 + coefficient * d
        if abs(d) < minimum:
            d = minimum
        c = 1.0 + coefficient / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        value *= d * c

        coefficient = (
            -(a + iteration)
            * (qab + iteration)
            * x
            / ((a + twice) * (qap + twice))
        )
        d = 1.0 + coefficient * d
        if abs(d) < minimum:
            d = minimum
        c = 1.0 + coefficient / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        delta = d * c
        value *= delta
        if abs(delta - 1.0) <= epsilon:
            return value
    raise RuntimeError("Incomplete beta continued fraction did not converge")


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    factor = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * _beta_continued_fraction(a, b, x) / a
    return 1.0 - (
        factor * _beta_continued_fraction(b, a, 1.0 - x) / b
    )


def _beta_quantile(probability: float, a: float, b: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("Beta quantile probability must be in (0, 1)")
    lower = 0.0
    upper = 1.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if _regularized_beta(midpoint, a, b) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _clopper_pearson(successes: int, trials: int) -> dict:
    if not 0 <= successes <= trials or trials < 1:
        raise ValueError("Invalid binomial counts")
    lower = (
        0.0
        if successes == 0
        else _beta_quantile(
            0.025, successes, trials - successes + 1
        )
    )
    upper = (
        1.0
        if successes == trials
        else _beta_quantile(
            0.975, successes + 1, trials - successes
        )
    )
    one_sided_upper = (
        1.0
        if successes == trials
        else _beta_quantile(
            0.95, successes + 1, trials - successes
        )
    )
    return {
        "method": "clopper_pearson_exact",
        "confidence_level": 0.95,
        "lower": lower,
        "upper": upper,
        "one_sided_upper_95": one_sided_upper,
    }


def _audit_summary(rows: Sequence[dict]) -> dict:
    spans = sum(row["penman_audit"]["span_records"] for row in rows)
    warning_spans = sum(
        row["penman_audit"]["warning_span_records"] for row in rows
    )
    warnings = [
        warning
        for row in rows
        for detail in row["penman_audit"]["warning_details"]
        for warning in detail["warnings"]
    ]
    warning_documents = sum(
        row["penman_audit"]["warning_span_records"] > 0 for row in rows
    )
    return {
        "documents": len(rows),
        "candidate_span_records": spans,
        "graph_valid_records": sum(
            row["penman_audit"]["graph_valid_records"] for row in rows
        ),
        "roundtrip_valid_records": sum(
            row["penman_audit"]["roundtrip_valid_records"] for row in rows
        ),
        "documents_with_warnings": warning_documents,
        "document_warning_rate": warning_documents / len(rows),
        "spans_with_warnings": warning_spans,
        "span_warning_rate": warning_spans / spans,
        "warning_records": len(warnings),
        "warning_message_counts": dict(sorted(Counter(warnings).items())),
    }


def _write_report(path: Path, result: dict) -> None:
    fpr = result["independent_test_fpr"]
    calibration_audit = result["penman_audit"]["calibration"]
    test_audit = result["penman_audit"]["independent_test"]
    lines = [
        "# SWAN Elastic Formal 1% Null Calibration",
        "",
        f"Status: `{result['status']}`",
        "",
        "## Independent FPR",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Target alpha | {fpr['alpha']:.4f} |",
        f"| False positives | {fpr['false_positives']}/{fpr['trials']} |",
        f"| Observed FPR | {fpr['observed_fpr']:.6f} |",
        f"| Exact 95% CI lower | {fpr['confidence_interval']['lower']:.6f} |",
        f"| Exact 95% CI upper | {fpr['confidence_interval']['upper']:.6f} |",
        (
            "| Exact one-sided 95% upper | "
            f"{fpr['confidence_interval']['one_sided_upper_95']:.6f} |"
        ),
        "",
        "The independent test cohort was not used to fit or modify the "
        "calibration artifact.",
        "",
        "## PENMAN Audit",
        "",
        "| Cohort | Documents with warnings | Spans with warnings | "
        "Graph valid | Round-trip valid |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Calibration | {calibration_audit['documents_with_warnings']}/"
            f"{calibration_audit['documents']} | "
            f"{calibration_audit['spans_with_warnings']}/"
            f"{calibration_audit['candidate_span_records']} | "
            f"{calibration_audit['graph_valid_records']}/"
            f"{calibration_audit['candidate_span_records']} | "
            f"{calibration_audit['roundtrip_valid_records']}/"
            f"{calibration_audit['candidate_span_records']} |"
        ),
        (
            f"| Independent test | {test_audit['documents_with_warnings']}/"
            f"{test_audit['documents']} | "
            f"{test_audit['spans_with_warnings']}/"
            f"{test_audit['candidate_span_records']} | "
            f"{test_audit['graph_valid_records']}/"
            f"{test_audit['candidate_span_records']} | "
            f"{test_audit['roundtrip_valid_records']}/"
            f"{test_audit['candidate_span_records']} |"
        ),
        "",
        "Warnings are retained per AMR span. Any structurally invalid or "
        "non-round-trippable graph would stop the run before statistical "
        "finalization.",
        "",
        "## Reproducibility",
        "",
        f"- Input manifest SHA256: `{result['artifacts']['input_manifest_sha256']}`",
        f"- Calibration artifact SHA256: "
        f"`{result['artifacts']['calibration_artifact_sha256']}`",
        f"- Calibration rows SHA256: "
        f"`{result['artifacts']['calibration_rows_sha256']}`",
        f"- Independent test rows SHA256: "
        f"`{result['artifacts']['independent_test_rows_sha256']}`",
        "",
    ]
    _atomic_write(path, "\n".join(lines).encode("utf-8"))


def run(args: argparse.Namespace) -> dict:
    output_root = args.output_root.resolve()
    manifest_path = output_root / "INPUT_MANIFEST.json"
    manifest = _load_manifest(output_root)
    calibration_inputs = _read_jsonl(_input_path(manifest, "calibration"))
    test_inputs = _read_jsonl(_input_path(manifest, "independent_test"))
    expected = manifest["sample_count_per_cohort"]
    if len(calibration_inputs) != expected or len(test_inputs) != expected:
        raise ValueError("Formal input row count mismatch")

    detector = build_detector(args)
    results_dir = output_root / "results"
    runtime_manifest_path = output_root / "RUNTIME_MANIFEST.json"
    _freeze_runtime_manifest(
        runtime_manifest_path,
        manifest_path,
        detector,
        args,
    )
    calibration_rows_path = results_dir / "calibration_rows.jsonl"
    test_rows_path = results_dir / "independent_test_rows.jsonl"
    calibration_rows = _process_cohort(
        detector,
        calibration_inputs,
        "calibration",
        calibration_rows_path,
        document_batch_size=args.document_batch_size,
    )
    calibration_path = results_dir / "CALIBRATION.json"
    calibrator = _fit_or_load_calibrator(
        calibration_path,
        detector,
        calibration_inputs,
        calibration_rows,
        manifest_path,
        runtime_manifest_path,
        calibration_rows_path,
    )

    test_rows = _process_cohort(
        detector,
        test_inputs,
        "independent_test",
        test_rows_path,
        calibrator=calibrator,
        alpha=args.alpha,
        document_batch_size=args.document_batch_size,
    )
    false_positives = sum(
        row["calibration"]["is_watermarked"] for row in test_rows
    )
    interval = _clopper_pearson(false_positives, len(test_rows))
    audit = {
        "calibration": _audit_summary(calibration_rows),
        "independent_test": _audit_summary(test_rows),
    }
    all_graphs_valid = all(
        summary["graph_valid_records"] == summary["candidate_span_records"]
        and summary["roundtrip_valid_records"]
        == summary["candidate_span_records"]
        for summary in audit.values()
    )
    observed_fpr = false_positives / len(test_rows)
    result = {
        "version": FORMAL_NULL_VERSION,
        "status": "complete" if all_graphs_valid else "invalid",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "calibration": {
            "samples": len(calibration_rows),
            "minimum_resolvable_p": calibrator.artifact[
                "minimum_resolvable_p"
            ],
            "word_bucket_counts": {
                key: len(values)
                for key, values in calibrator.artifact[
                    "bucket_statistics"
                ].items()
            },
            "independent_test_used_for_fitting": False,
        },
        "independent_test_fpr": {
            "alpha": args.alpha,
            "false_positives": false_positives,
            "trials": len(test_rows),
            "observed_fpr": observed_fpr,
            "observed_fpr_at_or_below_target": observed_fpr <= args.alpha,
            "confidence_interval": interval,
            "confidence_upper_at_or_below_target": (
                interval["one_sided_upper_95"] is not None
                and interval["one_sided_upper_95"] <= args.alpha
            ),
        },
        "penman_audit": audit,
        "fingerprints": {
            "detector": detector.fingerprint,
            "bank": detector.scorer.fingerprint,
            "bank_sha256": detector.scorer.bank_sha256,
            "parser": detector.scorer.parser_fingerprint,
        },
        "artifacts": {
            "input_manifest_sha256": sha256_file(manifest_path),
            "runtime_manifest_sha256": sha256_file(runtime_manifest_path),
            "calibration_artifact_sha256": sha256_file(calibration_path),
            "calibration_rows_sha256": sha256_file(calibration_rows_path),
            "independent_test_rows_sha256": sha256_file(test_rows_path),
            "cache_db": str(Path(args.cache_db).resolve()),
        },
        "extractor_stats_this_process": asdict(detector.scorer.extractor.stats),
    }
    _atomic_write(results_dir / "RESULT.json", _json_bytes(result))
    _write_report(results_dir / "REPORT.md", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze and run formal independent 1% SWAN Null evaluation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--calibration-source",
        type=Path,
        default=DEFAULT_INPUT_ROOT / "calibration_sources.json",
    )
    prepare_parser.add_argument(
        "--test-source",
        type=Path,
        default=DEFAULT_INPUT_ROOT / "test_sources.json",
    )
    prepare_parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    prepare_parser.add_argument("--sample-count", type=int, default=1000)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    run_parser.add_argument("--alpha", type=float, default=0.01)
    run_parser.add_argument("--document-batch-size", type=int, default=16)
    add_runtime_arguments(run_parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        result = prepare(
            calibration_source=args.calibration_source,
            test_source=args.test_source,
            output_root=args.output_root,
            sample_count=args.sample_count,
        )
    else:
        if not 0.0 < args.alpha < 1.0:
            raise ValueError("alpha must be strictly between zero and one")
        result = run(args)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
