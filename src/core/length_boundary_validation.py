"""Prospective short-text Null extension and fixed punctuation stress evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Sequence

from .bank_detector import add_runtime_arguments, build_detector
from .formal_null_evaluation import (
    ROOT, _atomic_write, _audit_summary,
    _detection_from_row, _freeze, _json_bytes, _load_checkpoint,
    _score_row as score_null, sha256_bytes, sha256_file,
)
from .lattice_builder import LatticeBuilder
from .null_calibration import EmpiricalNullCalibrator
from .parser_recovery_audit import attach_recovery_audit
from .positive_attack_evaluation import (
    _base_row, _load_checkpoint as load_stress_checkpoint,
    _score_row as score_stress, _rate,
)
from .sqlite_cache import normalize_text


VERSION = "swan-elastic-length-boundary:v1"
DEFAULT_OUTPUT = ROOT / "experiments/swan_elastic_length_boundary_audited2"
FORMAL = ROOT / "experiments/swan_elastic_formal_null1"
POSITIVE = ROOT / "experiments/swan_elastic_positive_attack1"
SOURCES = ROOT / "experiments/swan_count_neutral_final1/sealed"
LENGTH_STRATA = ((17, 24), (25, 32), (33, 40), (41, 48))
VARIANTS = ("semicolon_all", "semicolon_pairs", "comma_all", "newline_all", "space_all")
IDENTITIES = (
    "allocation_id", "locator", "normalized_source_sha256",
    "source_text_sha256", "url_sha256", "window_sha256",
)
LEXICAL_RE = re.compile(r"\w+(?:['’-]\w+)*", re.UNICODE)
END_MARK_RE = re.compile(r'[.!?]+(?=["\')\]]*$)')


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
        (json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n").encode()
        for row in rows
    )


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def lexical_tokens(text: str) -> tuple:
    return tuple(LEXICAL_RE.findall(text))


def hard_units(text: str) -> list:
    text = text.strip()
    boundaries = LatticeBuilder(include_soft_boundaries=False).candidate_boundaries(text)
    return [
        text[a.char_offset:b.char_offset].strip()
        for a, b in zip(boundaries, boundaries[1:])
    ]


def punctuation_variant(text: str, variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError("Unknown punctuation variant")
    units = hard_units(text)
    result = ""
    for index, unit in enumerate(units):
        if index == len(units) - 1:
            result += unit
            break
        change = variant != "semicolon_pairs" or index % 2 == 0
        if not change:
            result += unit + " "
            continue
        separator = {
            "semicolon_all": "; ", "semicolon_pairs": "; ",
            "comma_all": ", ", "newline_all": "\n", "space_all": " ",
        }[variant]
        result += END_MARK_RE.sub("", unit) + separator
    if lexical_tokens(result) != lexical_tokens(text):
        raise ValueError("Punctuation transform changed lexical content or order")
    return result


def select_short_cohort(source: Path, cohort: str, per_stratum: int,
                        historical: Sequence[dict]) -> list:
    """One first eligible complete prefix per source, no detector features."""
    counts = Counter()
    used = {field: {r[field] for r in historical} for field in IDENTITIES}
    cutoff = max(r["allocation_ordinal"] for r in historical)
    selected = []
    for source_row in sorted(read_json(source), key=lambda r: r["ordinal"]):
        if source_row["ordinal"] <= cutoff:
            continue
        identities = {
            field: source_row[field]
            for field in ("allocation_id", "locator", "normalized_source_sha256", "url_sha256")
        }
        identities["source_text_sha256"] = sha256_bytes(source_row["text"].encode())
        if any(identities[field] in used[field] for field in identities):
            continue
        for low, high in LENGTH_STRATA:
            label = f"words-{low:02d}-{high:02d}"
            if counts[label] >= per_stratum:
                continue
            units = hard_units(source_row["text"])
            prefix = ""
            for count, unit in enumerate(units, 1):
                prefix = (prefix + " " + unit).strip()
                words = len(prefix.split())
                if words > high:
                    break
                if words < low or count < 2:
                    continue
                digest = sha256_bytes(prefix.encode())
                if digest in used["window_sha256"]:
                    break
                row = {
                    "schema_version": 1, "cohort": cohort,
                    "cohort_index": len(selected), **identities,
                    "allocation_ordinal": source_row["ordinal"],
                    "window": prefix, "window_sha256": digest,
                    "word_count": words, "hard_unit_count": count,
                    "length_stratum": label,
                }
                selected.append(row)
                counts[label] += 1
                for field in IDENTITIES:
                    used[field].add(row[field])
                break
            else:
                continue
            if selected and selected[-1]["allocation_id"] == source_row["allocation_id"]:
                break
        if all(counts[f"words-{lo:02d}-{hi:02d}"] == per_stratum
               for lo, hi in LENGTH_STRATA):
            break
    if len(selected) != per_stratum * len(LENGTH_STRATA):
        raise ValueError(f"Insufficient distinct short sources: {dict(counts)}")
    return selected


def prepare(output: Path, per_stratum: int = 250) -> dict:
    if per_stratum < 25:
        raise ValueError("At least 100 short Null documents per cohort are required")
    old_inputs = {
        cohort: read_rows(FORMAL / f"inputs/{cohort}_windows.jsonl")
        for cohort in ("calibration", "independent_test")
    }
    all_old = old_inputs["calibration"] + old_inputs["independent_test"]
    cal = select_short_cohort(
        SOURCES / "calibration_sources.json", "calibration", per_stratum,
        old_inputs["calibration"],
    )
    test = select_short_cohort(
        SOURCES / "test_sources.json", "independent_test", per_stratum,
        old_inputs["independent_test"],
    )
    overlap = {}
    for field in IDENTITIES:
        left, right = {r[field] for r in cal}, {r[field] for r in test}
        overlap[field] = {
            "between_short_cohorts": len(left & right),
            "with_original_formal_cohorts": len((left | right) & {r[field] for r in all_old}),
        }
        if len(left) != len(cal) or len(right) != len(test) or any(overlap[field].values()):
            raise ValueError(f"Null independence failed: {field}")
    positive_rows = read_rows(POSITIVE / "inputs/evaluation_rows.jsonl")
    short_hashes = {sha256_bytes(normalize_text(r["window"]).encode()) for r in cal + test}
    positive_overlap = short_hashes & {r["normalized_text_sha256"] for r in positive_rows}
    if positive_overlap:
        raise ValueError("Short Null overlaps positive text")
    stress = [dict(row) for row in positive_rows]
    for row in positive_rows:
        if row["cohort"] != "confirmation_baseline_positive":
            continue
        for variant in VARIANTS:
            stress.append(_base_row(
                cohort="confirmation_punctuation_stress", cohort_index=len(stress),
                row_id=f"punctuation:{row['group_id']}:{variant}",
                group_id=row["group_id"], variant_id=variant,
                text=punctuation_variant(row["text"], variant),
                source_paths=(POSITIVE / "inputs/evaluation_rows.jsonl",),
                metadata={"lexical_sequence_preserved": True,
                          "quality_valid": None, "source_row_id": row["row_id"],
                          "source_text_sha256": row["text_sha256"]},
            ))
    for row in test:
        for variant in VARIANTS:
            stress.append(_base_row(
                cohort="independent_null_punctuation_stress", cohort_index=len(stress),
                row_id=f"null:{row['allocation_id']}:{variant}",
                group_id=row["allocation_id"], variant_id=variant,
                text=punctuation_variant(row["window"], variant),
                source_paths=(),
                metadata={"lexical_sequence_preserved": True, "quality_valid": None,
                          "source_window_sha256": row["window_sha256"],
                          "length_stratum": row["length_stratum"]},
            ))
    stress = [{**row, "input_index": i} for i, row in enumerate(stress)]
    if len({r["row_id"] for r in stress}) != len(stress):
        raise ValueError("Duplicate stress row IDs")
    files = {
        "calibration": (output / "inputs/calibration_windows.jsonl", cal),
        "independent_test": (output / "inputs/independent_test_windows.jsonl", test),
        "stress": (output / "inputs/stress_rows.jsonl", stress),
    }
    outputs = {}
    for key, (path, rows) in files.items():
        data = jsonl(rows)
        _freeze(path, data)
        outputs[key] = {"path": relative(path), "sha256": sha256_bytes(data),
                        "bytes": len(data), "rows": len(rows)}
    dependencies = [
        FORMAL / "INPUT_MANIFEST.json", FORMAL / "RUNTIME_MANIFEST.json",
        FORMAL / "results/CALIBRATION.json", FORMAL / "results/calibration_rows.jsonl",
        FORMAL / "inputs/calibration_windows.jsonl",
        FORMAL / "results/RESULT.json", POSITIVE / "INPUT_MANIFEST.json",
        POSITIVE / "inputs/evaluation_rows.jsonl",
        POSITIVE / "results/evaluation_rows.jsonl",
        SOURCES / "calibration_sources.json", SOURCES / "test_sources.json",
    ]
    manifest = {
        "version": VERSION, "status": "frozen_before_parser_execution",
        "outputs": outputs,
        "dependencies": {relative(p): sha256_file(p) for p in dependencies},
        "identity_overlap_counts": overlap, "positive_text_overlap": 0,
        "selection": {
            "length_strata": LENGTH_STRATA, "per_stratum": per_stratum,
            "order": "allocation ordinal, ascending strata, first complete hard prefix",
            "minimum_hard_units": 2, "one_window_per_source": True,
            "detector_or_parser_selection": False,
            "independence_scope": "declared original formal cohorts and positive input",
            "historical_allocations_reused": True,
        },
        "variants": list(VARIANTS),
        "calibration_policy": {
            "original_artifact_immutable": True,
            "extend_with_short_calibration_only": True,
            "preserve_existing_length_bucket_statistics": True,
            "test_used_for_fitting": False, "alpha": 0.01,
        },
        "parser_recovery_policy": {
            "version": "amrlib-recovery-audit:v1",
            "original_outputs_and_beam_choice_preserved": True,
            "all_attempted_beams_logged_by_span": True,
            "missing_or_unbound_evidence": "stop",
            "selected_beam_error_action": "retain_all_rows_but_engineering_no_go",
            "old_long_calibration_upstream_audit_unavailable": True,
        },
        "gates": {
            "strict_fpr_one_sided_upper": 0.01,
            "engineering_fpr_one_sided_upper": 0.02,
            "minimum_fixed_attack_conditional_survival": 0.80,
            "failed_gate_action": "record failure, no tuning or replacement in this run",
        },
        "claim_limits": [
            "Reused 30 positives are retrospective development evidence.",
            "Lexical preservation does not certify semantics or grammatical quality.",
            "Five variants per document are correlated; report document denominators.",
            "Natural human Null does not match the SWAN generator or template distribution.",
            "No certificate, normal-length cohort, or independent quality claim.",
        ],
    }
    _freeze(output / "INPUT_MANIFEST.json", _json_bytes(manifest))
    return manifest


def verify_inputs(output: Path, runtime_dependencies: bool = False) -> dict:
    manifest = read_json(output / "INPUT_MANIFEST.json")
    if manifest["version"] != VERSION or manifest["status"] != "frozen_before_parser_execution":
        raise ValueError("Unfrozen or incompatible manifest")
    for item in manifest["outputs"].values():
        path = ROOT / item["path"]
        rows = read_rows(path)
        if (sha256_file(path) != item["sha256"] or path.stat().st_size != item["bytes"]
                or len(rows) != item["rows"]):
            raise ValueError(f"Frozen input mismatch: {path}")
    # Raw allocated sources are not required on the inference host.
    for name, digest in manifest["dependencies"].items():
        path = ROOT / name
        if runtime_dependencies and name.startswith(relative(SOURCES) + "/"):
            continue
        if sha256_file(path) != digest:
            raise ValueError(f"Frozen dependency mismatch: {name}")
    return manifest


def freeze_runtime(path: Path, current: dict) -> dict:
    if path.exists():
        old = read_json(path)
        if {k: v for k, v in old.items() if k != "created_at"} != current:
            raise ValueError("Resume runtime/code/configuration differs from frozen run")
        return old
    value = {**current, "created_at": datetime.now(timezone.utc).isoformat()}
    _freeze(path, _json_bytes(value))
    return value


def heartbeat(output: Path, stage: str, done: int, total: int, status: str = "running") -> None:
    _atomic_write(output / "HEARTBEAT.json", _json_bytes({
        "stage": stage, "completed": done, "total": total, "status": status,
        "pid": os.getpid(), "updated_at": datetime.now(timezone.utc).isoformat(),
    }))


def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def process(detector, inputs: list, path: Path, output: Path, stage: str,
            batch_size: int, calibrator=None, legacy=None, stress: bool = False) -> list:
    rows = (load_stress_checkpoint(path, inputs) if stress
            else _load_checkpoint(path, inputs, stage))
    for row in rows:
        audit = row["penman_audit"]
        if not (audit["span_records"] == audit["graph_valid_records"]
                == audit["roundtrip_valid_records"]):
            raise ValueError("Invalid resumed PENMAN evidence")
        for name, artifact in (("calibration", calibrator), ("legacy_calibration", legacy)):
            if artifact is not None:
                expected = asdict(artifact.calibrate(row["detection"]["statistic"],
                                                     row["word_count"], alpha=0.01))
                if row.get(name) != expected:
                    raise ValueError("Resumed calibration binding differs")
    heartbeat(output, stage, len(rows), len(inputs))
    for start in range(len(rows), len(inputs), batch_size):
        batch = inputs[start:start + batch_size]
        spans = [
            edge.span_text for row in batch
            for edge in detector.builder.build_lattice(
                row["text"] if stress else row["window"]
            ).edges
        ]
        detector.scorer.prepare(spans)
        for source in batch:
            row = (score_stress(detector, source, calibrator, 0.01) if stress
                   else score_null(detector, source, calibrator, 0.01))
            if legacy is not None:
                row["legacy_calibration"] = asdict(legacy.calibrate(
                    row["detection"]["statistic"], row["word_count"], alpha=0.01))
            append(path, row)
            rows.append(row)
            heartbeat(output, stage, len(rows), len(inputs))
            print(json.dumps({"stage": stage, "completed": len(rows), "total": len(inputs),
                              "statistic": row["detection"]["statistic"]}), flush=True)
        detector.scorer.clear_prepared()
    return rows


def extend_calibration(output: Path, short_inputs: list, short_rows: list,
                       legacy: EmpiricalNullCalibrator, runtime: dict) -> EmpiricalNullCalibrator:
    provenance = {
        "version": VERSION, "original_calibration_sha256":
        sha256_file(FORMAL / "results/CALIBRATION.json"),
        "input_manifest_sha256": sha256_file(output / "INPUT_MANIFEST.json"),
        "runtime_manifest_sha256": sha256_file(output / "RUNTIME_MANIFEST.json"),
        "short_calibration_rows_sha256": sha256_file(output / "results/calibration_rows.jsonl"),
        "test_used_for_fitting": False,
    }
    path = output / "results/CALIBRATION.json"
    if path.exists():
        extended = EmpiricalNullCalibrator.load(path)
        if extended.artifact.get("extension_provenance") != provenance:
            raise ValueError("Extended frozen calibration provenance differs")
    else:
        old_inputs = read_rows(FORMAL / "inputs/calibration_windows.jsonl")
        old_rows = read_rows(FORMAL / "results/calibration_rows.jsonl")
        if len(old_inputs) != len(old_rows) or len(short_inputs) != len(short_rows):
            raise ValueError("Calibration input/result length mismatch")
        samples = []
        for inputs, rows in ((old_inputs, old_rows), (short_inputs, short_rows)):
            for source, row in zip(inputs, rows):
                if source["window_sha256"] != row["window_sha256"]:
                    raise ValueError("Calibration source binding mismatch")
                samples.append((source["window"], _detection_from_row(row)))
        extended = EmpiricalNullCalibrator.fit(
            samples, runtime["fingerprints"]["detector"],
            runtime["fingerprints"]["bank"], runtime["fingerprints"]["bank_sha256"],
            runtime["fingerprints"]["parser"], min_bucket_size=legacy.artifact["min_bucket_size"],
            bins=legacy.bins,
        )
        for label, low, _ in legacy.bins:
            if low > 80 and (extended.artifact["bucket_statistics"][label]
                             != legacy.artifact["bucket_statistics"][label]):
                raise ValueError("Existing length bucket changed")
        extended.artifact["extension_provenance"] = provenance
        extended.save(path)
    if extended.artifact["bucket_statistics"]["words-000-080"] != sorted(
            r["detection"]["statistic"] for r in short_rows):
        raise ValueError("Short calibration bucket differs from frozen calibration rows")
    for label, low, _ in legacy.bins:
        if low > 80 and extended.artifact["bucket_statistics"][label] != legacy.artifact["bucket_statistics"][label]:
            raise ValueError("Existing length bucket differs from original calibration")
    extended.assert_compatible(
        runtime["fingerprints"]["detector"], runtime["fingerprints"]["bank"],
        runtime["fingerprints"]["parser"],
    )
    return extended


def false_positive_summary(rows: list, key: str) -> dict:
    rate = _rate(sum(r[key]["is_watermarked"] for r in rows), len(rows))
    upper = rate["confidence_interval"]["one_sided_upper_95"] if rows else None
    return {**rate, "strict_1pct_supported": upper is not None and upper <= 0.01,
            "engineering_2pct_gate": upper is not None and upper <= 0.02}


def summarize(test: list, stress: list) -> dict:
    result = {}
    for key in ("legacy_calibration", "calibration"):
        base = {r["group_id"]: r for r in stress
                if r["cohort"] == "confirmation_baseline_positive"}
        attacks = {r["group_id"]: r for r in stress
                   if r["cohort"] == "confirmation_boundary_attack"}
        positive = [r for r in stress if r["cohort"] == "confirmation_punctuation_stress"]
        null = [r for r in stress if r["cohort"] == "independent_null_punctuation_stress"]
        detected = {group for group, row in base.items() if row[key]["is_watermarked"]}
        positive_variants, null_variants = {}, {}
        for variant in VARIANTS:
            rows = [r for r in positive if r["variant_id"] == variant]
            survivors = {r["group_id"] for r in rows if r[key]["is_watermarked"]}
            positive_variants[variant] = {
                "detected": _rate(len(survivors), len(rows)),
                "survival_given_baseline_detected": _rate(len(survivors & detected), len(detected)),
                "evaded_groups": sorted(detected - survivors),
            }
            null_variants[variant] = false_positive_summary(
                [r for r in null if r["variant_id"] == variant], key)
        worst_survivors = {group for group in detected if all(
            r[key]["is_watermarked"] for r in positive if r["group_id"] == group)}
        null_groups = {r["group_id"] for r in null}
        null_false = {r["group_id"] for r in null if r[key]["is_watermarked"]}
        clean_false = {r["allocation_id"] for r in test if r[key]["is_watermarked"]}
        family_fpr = _rate(len(null_false), len(null_groups))
        detected_attacks = {group for group, row in attacks.items() if row[key]["is_watermarked"]}
        worst = _rate(len(worst_survivors), len(detected))
        clean = false_positive_summary(test, key)
        result[key] = {
            "clean_short_fpr": clean, "null_punctuation_fpr": null_variants,
            "any_of_five_null_false_positive_rate": family_fpr,
            "clean_or_any_attack_null_false_positive_rate": _rate(
                len(null_false | clean_false), len(null_groups)),
            "baseline_detection": _rate(len(detected), len(base)),
            "historical_boundary_attack_detection": _rate(len(detected_attacks), len(attacks)),
            "historical_boundary_survival": _rate(len(detected & detected_attacks), len(detected)),
            "positive_punctuation": positive_variants, "survival_all_five_variants": worst,
            "engineering_go": clean["engineering_2pct_gate"]
            and all(v["engineering_2pct_gate"] for v in null_variants.values())
            and worst["rate"] is not None and worst["rate"] >= 0.80,
            "length_strata_clean_fpr": {
                label: false_positive_summary(
                    [r for r in test if LENGTH_STRATA[i][0] <= r["word_count"]
                     <= LENGTH_STRATA[i][1]], key)
                for i, label in enumerate(("17-24", "25-32", "33-40", "41-48"))
            },
        }
    return result


def write_report(output: Path, result: dict) -> None:
    lines = [
        "# SWAN 短文本校准与标点压力验证", "",
        "状态：完成。原检测器与旧 calibration 保持冻结；新增制品只扩充短文本 Null。", "",
        "| 指标 | 原 calibration | 扩充后 calibration |",
        "|---|---:|---:|",
    ]
    def cell(rate):
        return f"{rate['successes']}/{rate['trials']} ({rate['rate']:.2%})" if rate["trials"] else "n/a"
    for name, label in (
        ("clean_short_fpr", "短文本独立 FPR"),
        ("baseline_detection", "既有 baseline 检出"),
        ("historical_boundary_attack_detection", "既有句界改写检出"),
        ("historical_boundary_survival", "既有改写条件存活"),
        ("survival_all_five_variants", "5 种标点变体全部存活"),
        ("any_of_five_null_false_positive_rate", "Null 任一标点变体误报"),
    ):
        lines.append(f"| {label} | {cell(result['summary']['legacy_calibration'][name])} | "
                     f"{cell(result['summary']['calibration'][name])} |")
    for key in ("legacy_calibration", "calibration"):
        summary = result["summary"][key]
        lines.extend(["", f"## {key}", "",
                      "| 变体 | 正例条件存活 | Null FPR | Null 单侧 95% 上界 |",
                      "|---|---:|---:|---:|"])
        for variant in VARIANTS:
            p = summary["positive_punctuation"][variant]["survival_given_baseline_detected"]
            n = summary["null_punctuation_fpr"][variant]
            lines.append(f"| {variant} | {cell(p)} | {cell(n)} | "
                         f"{n['confidence_interval']['one_sided_upper_95']:.2%} |")
        upper = summary["clean_short_fpr"]["confidence_interval"]["one_sided_upper_95"]
        lines.extend(["", f"干净短文本 FPR 单侧 95% 上界：{upper:.2%}。",
                      f"严格 1% 支撑：{summary['clean_short_fpr']['strict_1pct_supported']}；"
                      f"预声明工程 Go：{summary['engineering_go']}。"])
    lines.extend(["", "## 证据边界", "",
                  "30 篇正例是已观察的开发资产，部分注入有 fallback，不能当作新的质量合格 cohort。",
                  "标点变体保持词汇与词序；问句/感叹句合并可能改变语气，没有独立语义质量评审，"
                  "这里报告压力测试检出率，不报告 Valid ASR。",
                  "同一文档的五个变体相关，任一变体指标按独立来源文档计数。",
                  "自然人类 Null 与短句模板注入文本仍有生成方式、句长和语体差异。",
                  "本轮没有证明通用鲁棒性、正常长度推广或认证边界。", "",
                  "## PENMAN 审计", "",
                  "| 队列 | Span | graph-valid | round-trip-valid | warning Span |",
                  "|---|---:|---:|---:|---:|"])
    for name, audit in result["penman_audit"].items():
        lines.append(f"| {name} | {audit['candidate_span_records']} | {audit['graph_valid_records']} | "
                     f"{audit['roundtrip_valid_records']} | {audit['spans_with_warnings']} |")
    recovery = result["parser_recovery_audit"]
    lines.extend(["", "## amrlib上游图恢复", "",
                  f"逐Span绑定{recovery['unique_spans']}个唯一解析；"
                  f"{recovery['spans_with_events']}个Span有诊断，"
                  f"{recovery['spans_with_selected_beam_errors']}个选中beam有ERROR。",
                  "诊断可定位不代表恢复语义正确；选中beam有ERROR时工程门不通过。",
                  f"联合审计/统计工程Go：{result['overall_engineering_go']}。"])
    lines.extend(["", "## SHA256", ""])
    lines.extend(f"- {name}: `{digest}`" for name, digest in result["artifacts"].items())
    _atomic_write(output / "results/REPORT.md", ("\n".join(lines) + "\n").encode())


def run(args) -> dict:
    output = args.output_root.resolve()
    if args.document_batch_size < 1:
        raise ValueError("document_batch_size must be positive")
    manifest = verify_inputs(output, runtime_dependencies=True)
    legacy = EmpiricalNullCalibrator.load(FORMAL / "results/CALIBRATION.json")
    detector = build_detector(args)
    recovery = attach_recovery_audit(detector, output / "results/parser_recovery_rows.jsonl")
    fingerprints = {
        "detector": detector.fingerprint, "bank": detector.scorer.fingerprint,
        "bank_sha256": detector.scorer.bank_sha256, "parser": detector.scorer.parser_fingerprint,
    }
    legacy.assert_compatible(fingerprints["detector"], fingerprints["bank"], fingerprints["parser"])
    old_runtime = read_json(FORMAL / "RUNTIME_MANIFEST.json")
    for name, binding in old_runtime["code"].items():
        if sha256_file(ROOT / name) != binding["sha256"]:
            raise ValueError(f"Detector dependency changed since original formal run: {name}")
    code = (
        "length_boundary_validation.py", "positive_attack_evaluation.py",
        "formal_null_evaluation.py", "bank_detector.py", "lattice_builder.py",
        "bank_scorer.py", "amr_extractor.py", "sqlite_cache.py",
        "null_calibration.py", "models.py", "parser_recovery_audit.py",
    )
    runtime = freeze_runtime(output / "RUNTIME_MANIFEST.json", {
        "version": VERSION, "input_manifest_sha256": sha256_file(output / "INPUT_MANIFEST.json"),
        "fingerprints": fingerprints,
        "code_sha256": {name: sha256_file(ROOT / "src/core" / name) for name in code},
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                          if k not in ("command", "per_stratum")},
    })
    cohorts = {key: read_rows(ROOT / meta["path"]) for key, meta in manifest["outputs"].items()}
    cal = process(detector, cohorts["calibration"], output / "results/calibration_rows.jsonl",
                  output, "calibration", args.document_batch_size)
    extended = extend_calibration(output, cohorts["calibration"], cal, legacy, runtime)
    test = process(detector, cohorts["independent_test"], output / "results/independent_test_rows.jsonl",
                   output, "independent_test", args.document_batch_size, extended, legacy)
    stress = process(detector, cohorts["stress"], output / "results/stress_rows.jsonl",
                     output, "stress", args.document_batch_size, extended, legacy, stress=True)
    audits = {name: _audit_summary(rows)
              for name, rows in (("calibration", cal), ("independent_test", test), ("stress", stress))}
    if any(a["candidate_span_records"] != a["graph_valid_records"]
           or a["candidate_span_records"] != a["roundtrip_valid_records"] for a in audits.values()):
        raise ValueError("PENMAN finalization audit failed")
    summary = summarize(test, stress)
    recovery_summary = recovery.summary()
    result = {
        "version": VERSION, "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary, "penman_audit": audits, "fingerprints": fingerprints,
        "parser_recovery_audit": recovery_summary,
        "overall_engineering_go": summary["calibration"]["engineering_go"]
        and recovery_summary["spans_with_selected_beam_errors"] == 0,
        "artifacts": {name: sha256_file(output / name) for name in (
            "INPUT_MANIFEST.json", "RUNTIME_MANIFEST.json", "results/CALIBRATION.json",
            "results/calibration_rows.jsonl", "results/independent_test_rows.jsonl",
            "results/stress_rows.jsonl", "results/parser_recovery_rows.jsonl",
        )},
        "original_calibration_sha256": sha256_file(FORMAL / "results/CALIBRATION.json"),
        "claim_limits": manifest["claim_limits"],
    }
    result_path = output / "results/RESULT.json"
    if result_path.exists():
        frozen_result = read_json(result_path)
        comparable = lambda value: {k: v for k, v in value.items() if k != "completed_at"}
        if comparable(frozen_result) != comparable(result):
            raise ValueError("Completed result differs from deterministic replay")
        result = frozen_result
    else:
        _freeze(result_path, _json_bytes(result))
    write_report(output, result)
    heartbeat(output, "complete", len(cal) + len(test) + len(stress),
              len(cal) + len(test) + len(stress), status="complete")
    return result


def audit(output: Path) -> dict:
    verify_inputs(output, runtime_dependencies=True)
    result = read_json(output / "results/RESULT.json")
    for name, digest in result["artifacts"].items():
        if sha256_file(output / name) != digest:
            raise ValueError(f"Output artifact hash mismatch: {name}")
    test = read_rows(output / "results/independent_test_rows.jsonl")
    stress = read_rows(output / "results/stress_rows.jsonl")
    calibration_rows = read_rows(output / "results/calibration_rows.jsonl")
    recovery_rows = read_rows(output / "results/parser_recovery_rows.jsonl")
    recovery = {r["text_sha256"]: r for r in recovery_rows}
    manifest = read_json(output / "INPUT_MANIFEST.json")
    for key, rows in (("calibration", calibration_rows), ("independent_test", test), ("stress", stress)):
        inputs = read_rows(ROOT / manifest["outputs"][key]["path"])
        if len(rows) != len(inputs):
            raise ValueError("Incomplete result rows")
        for source, row in zip(inputs, rows):
            identity = "row_id" if key == "stress" else "window_sha256"
            if row[identity] != source[identity]:
                raise ValueError("Result row identity mismatch")
            text = source["text"] if key == "stress" else source["window"]
            for edge in LatticeBuilder().build_lattice(text).edges:
                digest = sha256_bytes(normalize_text(edge.span_text).encode())
                if digest not in recovery:
                    raise ValueError("Result Span lacks upstream parser recovery record")
            evidence = row["penman_audit"]
            if not (evidence["span_records"] == evidence["graph_valid_records"]
                    == evidence["roundtrip_valid_records"]):
                raise ValueError("Saved PENMAN audit is invalid")
    for name, calibrator in (
        ("calibration", EmpiricalNullCalibrator.load(output / "results/CALIBRATION.json")),
        ("legacy_calibration", EmpiricalNullCalibrator.load(FORMAL / "results/CALIBRATION.json")),
    ):
        for row in test + stress:
            if row[name] != asdict(calibrator.calibrate(
                    row["detection"]["statistic"], row["word_count"], alpha=0.01)):
                raise ValueError("Saved p-value differs from frozen calibration")
    if summarize(test, stress) != result["summary"]:
        raise ValueError("Aggregate recomputation mismatch")
    return {"status": "pass", "new_model_calls": 0, "test_rows": len(test),
            "stress_rows": len(stress), "result_sha256": sha256_file(output / "results/RESULT.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument("--per-stratum", type=int, default=250)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--document-batch-size", type=int, default=16)
    add_runtime_arguments(run_parser)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.output_root.resolve(), args.per_stratum)
    elif args.command == "audit":
        value = audit(args.output_root.resolve())
    else:
        try:
            value = run(args)
        except Exception as exc:
            _atomic_write(args.output_root / "FAILURE.json", _json_bytes({
                "type": type(exc).__name__, "message": str(exc),
                "occurred_at": datetime.now(timezone.utc).isoformat(),
            }))
            raise
    print(json.dumps(value, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
