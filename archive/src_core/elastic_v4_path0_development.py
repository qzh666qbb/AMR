"""Persisted, version-bound two-arm development comparison over shared AMRs."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal

from .bank_detector import add_runtime_arguments, build_detector
from .elastic_v3 import ElasticBoundaryProposerV3, InvalidTolerantAMRExtractor, classify_edge_evidence
from .elastic_v4 import SwanBankLatticeDetectorV4
from .elastic_v4_path0 import AllBankScorer, CheckedHopOptimizer, EmptyPathZeroScorer
from .elastic_v4_path0_inputs import (
    VARIANTS, file_sha, freeze, history_snapshot, now, prepare, sha, variants, write,
)
from .elastic_v4_path0_preflight import run_p0
from .models import DetectionResult
from .null_calibration import EmpiricalNullCalibrator
from .parser_recovery_audit import attach_recovery_audit
from .sqlite_cache import normalize_text, text_sha256


ROOT = Path(__file__).resolve().parents[2]
ALPHAS = (.001, .002, .005, .01, .02, .05)
ABSTAIN_STATISTIC = -1e12


def checkpoint(path: Path, binding: dict):
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    if row.get("binding") != binding:
        raise ValueError(f"Checkpoint input/runtime identity differs: {path.name}")
    evidence = path.parent.parent.parent / row["evidence_path"]
    if not evidence.exists() or file_sha(evidence) != row["evidence_sha256"]:
        raise ValueError(f"Checkpoint candidate evidence SHA differs: {path.name}")
    return row


def build_arms(args):
    if args.backend != "amrlib":
        raise ValueError("Development comparison requires the real amrlib backend")
    base = build_detector(args)
    extractor = InvalidTolerantAMRExtractor(
        base.scorer.extractor.backend, base.scorer.extractor.cache, args.request_batch_size
    )
    builder = ElasticBoundaryProposerV3(
        args.max_boundary_hops, args.max_span_words,
        args.capitalization_gap_words, args.long_gap_stride_words,
    )
    arms = {}
    for name, cls in (("control", AllBankScorer), ("path0", EmptyPathZeroScorer)):
        scorer = cls(extractor, args.bank)
        detector = SwanBankLatticeDetectorV4(builder, scorer, args.score_center, args.green_threshold)
        detector.optimizer = CheckedHopOptimizer(args.score_center, args.green_threshold)
        arms[name] = detector
    recovery = attach_recovery_audit(arms["control"], args.output_root / "sealed/parser_recovery.jsonl")
    return arms, recovery


def runtime_payload(args, arms):
    core = Path(__file__).parent
    model_paths = tuple(Path(args.model_dir).glob("*")) + tuple(Path(args.tokenizer_dir).glob("*"))
    return {
        "version": "path0-development-v1",
        "code_sha256": {str(path.relative_to(ROOT)): file_sha(path) for path in sorted(core.glob("*.py"))},
        "protocol_sha256": file_sha(args.output_root / "PROTOCOL.md"),
        "execution_amendment_sha256": (
            file_sha(args.output_root / "RUN_AMENDMENT.md")
            if (args.output_root / "RUN_AMENDMENT.md").exists() else None
        ),
        "input_manifest_sha256": file_sha(args.output_root / "INPUT_MANIFEST.json"),
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "versions": {
            "python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in ("torch", "amrlib", "penman", "transformers", "numpy", "scipy")},
        },
        "model_asset_sha256": {str(path): file_sha(path) for path in sorted(model_paths) if path.is_file()},
        "arms": {
            name: {"scorer": detector.scorer.fingerprint, "detector": detector.fingerprint,
                   "parser": detector.scorer.parser_fingerprint, "bank_sha256": detector.scorer.bank_sha256}
            for name, detector in arms.items()
        },
        "calibration": {"word_bins": "original_default_0_80_bucket", "alpha": .01, "curve_alphas": ALPHAS,
                        "abstain_statistic": ABSTAIN_STATISTIC, "abstain_p": 1, "full_denominator": True},
    }


def score_document(args, arms, recovery, source, variant, text, binding):
    lattice = arms["control"].builder.build_lattice(text)
    spans = sorted({normalize_text(edge.span_text) for edge in lattice.edges})
    arms["control"].scorer.prepare(spans)
    records = arms["control"].scorer._records
    arms["path0"].scorer._records = records
    evidence = classify_edge_evidence(lattice, recovery.records, records)
    span_payload = {}
    matches_by_arm = {name: {} for name in arms}
    template_changes = 0
    for span in spans:
        record = records[span]
        digest = text_sha256(span)
        recovery.assert_record(record)
        detail = {
            "record": asdict(record), "amr_sha256": sha(record.amr.encode()),
            "recovery": recovery.records[digest], "arms": {},
        }
        for name, detector in arms.items():
            if not record.graph_valid or not record.roundtrip_valid:
                matches_by_arm[name][span] = None
                detail["arms"][name] = {"all_bank_matches": [], "best": None}
            else:
                matches, summary = detector.scorer.audited_matches(record.amr)
                winner = max(matches, key=lambda item: item.score)
                matches_by_arm[name][span] = winner
                detail["arms"][name] = {
                    **summary, "all_bank_matches": [asdict(item) for item in matches],
                }
        left, right = matches_by_arm["control"][span], matches_by_arm["path0"][span]
        template_changes += bool(left and right and left.template_index != right.template_index)
        span_payload[digest] = detail
    detections = {}
    paths = {}
    for name, detector in arms.items():
        matches = tuple(matches_by_arm[name][normalize_text(edge.span_text)] for edge in lattice.edges)
        result = detector.optimizer.optimize(lattice, matches, tuple(item.valid for item in evidence))
        valid = {edge for edge, item in zip(lattice.edges, evidence) if item.valid}
        selected_invalid = sum(unit.edge not in valid for unit in result.units)
        if selected_invalid:
            raise RuntimeError("Selected invalid candidate edge")
        detections[name] = {
            "status": result.status, "statistic": result.statistic, "raw_utility": result.raw_utility,
            "unit_count": result.unit_count, "green_count": result.green_count,
            "invalid_edge_count": result.invalid_edge_count, "selected_invalid_edges": selected_invalid,
            "abstain_reason": result.abstain_reason,
            "units": [
                {"start": unit.edge.start, "end": unit.edge.end, "span_sha256": text_sha256(unit.edge.span_text),
                 "boundary_hops": unit.edge.boundary_hops, "utility": unit.utility, "match": asdict(unit.match)}
                for unit in result.units
            ],
        }
        paths[name] = [(unit.edge.start, unit.edge.end) for unit in result.units]
    row_id = f"{source['allocation_id']}__{variant}"
    evidence_path = args.output_root / f"sealed/candidates/{row_id}.json.gz"
    payload = {
        "binding": binding, "text_sha256": sha(text.encode()),
        "boundaries": [asdict(item) for item in lattice.boundaries],
        "edges": [
            {"start": edge.start, "end": edge.end, "boundary_hops": edge.boundary_hops,
             "span_sha256": text_sha256(edge.span_text), "evidence": asdict(item)}
            for edge, item in zip(lattice.edges, evidence)
        ],
        "spans": span_payload,
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = evidence_path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        stream.write(gzip.compress(json.dumps(payload, sort_keys=True, allow_nan=False).encode(), mtime=0))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, evidence_path)
    row = {
        "binding": binding, "stage": source["stage"], "row_id": row_id,
        "allocation_id": source["allocation_id"], "variant": variant,
        "word_count": len(text.split()), "length_bin": source["length_bin"],
        "arms": detections, "candidate_edges": len(lattice.edges), "unique_spans": len(spans),
        "template_changed_spans": template_changes, "path_changed": paths["control"] != paths["path0"],
        "evidence_path": str(evidence_path.relative_to(args.output_root)), "evidence_sha256": file_sha(evidence_path),
    }
    arms["control"].scorer.clear_prepared()
    return row


def rate(count, total):
    from scipy.stats import beta

    return {
        "successes": count, "trials": total, "rate": count / total if total else None,
        "ci95": [
            float(beta.ppf(.025, count, total - count + 1)) if count else 0.0,
            float(beta.ppf(.975, count + 1, total - count)) if count < total else 1.0,
        ] if total else None,
        "upper95_one_sided": (
            float(beta.ppf(.95, count + 1, total - count)) if count < total else 1.0
        ) if total else None,
    }


def fit(args, arm, detector, rows, inputs, runtime_sha):
    samples = []
    for row, source in zip(rows, inputs):
        result = row["arms"][arm]
        statistic = result["statistic"] if result["status"] != "abstain" else ABSTAIN_STATISTIC
        if not math.isfinite(statistic):
            raise ValueError("Non-finite calibration statistic")
        samples.append((source["text"], DetectionResult(statistic, statistic, result["unit_count"], result["green_count"], ())))
    artifact_path = args.output_root / f"results/CALIBRATION_{arm}.json"
    if artifact_path.exists():
        calibrator = EmpiricalNullCalibrator.load(artifact_path)
        if calibrator.artifact["binding_runtime_sha256"] != runtime_sha:
            raise ValueError("Calibration runtime SHA mismatch")
    else:
        calibrator = EmpiricalNullCalibrator.fit(
            samples, detector.fingerprint, detector.scorer.fingerprint,
            detector.scorer.bank_sha256, detector.scorer.parser_fingerprint,
        )
        calibrator.artifact["binding_runtime_sha256"] = runtime_sha
        calibrator.artifact["binding_input_sha256"] = file_sha(args.output_root / "INPUT_MANIFEST.json")
        calibrator.save(artifact_path)
    calibrator.assert_compatible(detector.fingerprint, detector.scorer.fingerprint, detector.scorer.parser_fingerprint)
    return calibrator


def summarize_null(rows, calibrators):
    summary = {}
    for arm in calibrators:
        curves = {}
        for alpha in ALPHAS:
            counts = Counter()
            unions = set()
            abstains = Counter()
            for row in rows:
                result = row["arms"][arm]
                hit = result["status"] != "abstain" and calibrators[arm].calibrate(
                    result["statistic"], row["word_count"], alpha
                ).is_watermarked
                counts[row["variant"]] += bool(hit)
                abstains[row["variant"]] += result["status"] == "abstain"
                if hit and row["variant"] != "baseline":
                    unions.add(row["allocation_id"])
            curves[str(alpha)] = {
                "conditions": {variant: rate(counts[variant], 200) for variant in ("baseline",) + VARIANTS},
                "any_stress_false_positive": rate(len(unions), 200),
                "abstain": dict(abstains),
            }
        summary[arm] = {"primary": curves["0.01"], "alpha_curve": curves}
    selected_invalid = sum(
        result["selected_invalid_edges"] for row in rows for result in row["arms"].values()
    )
    if selected_invalid:
        raise RuntimeError("Invalid selected edge in summary")
    return {
        "status": "null_complete_pending_generation", "decision": "incomplete_positive_comparison",
        "arms": summary, "null_source_count": 200, "null_condition_rows": len(rows),
        "positive_slots_frozen": 50, "positive_generated": 0,
        "selected_invalid_edges": selected_invalid,
        "path_changed_rows": sum(row["path_changed"] for row in rows),
        "template_changed_spans": sum(row["template_changed_spans"] for row in rows),
        "limitations": ["development_only", "positive_TPR_and_survival_pending", "normal_length_unconfirmed"],
    }


def execute(args):
    output = args.output_root
    write(output / "HEARTBEAT.json", {"status": "running", "stage": "p0", "updated_at": now()})
    p0 = run_p0(args.bank)
    p0["dependency_versions"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "amrlib", "penman", "transformers", "numpy", "scipy")
    }
    # The resume contract is executable, including a deliberately wrong identity.
    probe = output / "results/.p0_identity_probe.json"
    write(probe, {"binding": {"runtime": "wrong"}})
    try:
        checkpoint(probe, {"runtime": "expected"})
    except ValueError:
        p0["checkpoint_identity_rejection"] = True
    else:
        raise AssertionError("Checkpoint identity guard failed")
    finally:
        probe.unlink(missing_ok=True)
    p0["code_sha256"] = {
        name: file_sha(Path(__file__).parent / name)
        for name in ("elastic_v4_path0.py", "elastic_v4_path0_preflight.py", "elastic_v4_path0_inputs.py", "elastic_v4_path0_development.py")
    }
    freeze(output / "P0.json", p0)
    manifest = prepare(args.history_root, output)
    arms, recovery = build_arms(args)
    runtime = runtime_payload(args, arms)
    runtime = json.loads(json.dumps(runtime))
    freeze(output / "RUNTIME_MANIFEST.json", runtime)
    runtime_sha = file_sha(output / "RUNTIME_MANIFEST.json")
    inputs = {
        stage: [json.loads(line) for line in (output / item["path"]).read_text().splitlines()]
        for stage, item in manifest["outputs"].items()
    }
    calibration_rows = []
    calibrators = None
    null_rows = []
    for stage in ("calibration", "null"):
        sources = inputs[stage]
        total = len(sources) * (1 if stage == "calibration" else 6)
        done = 0
        for source in sources:
            texts = {"baseline": source["text"]} if stage == "calibration" else variants(source["text"])
            for variant, text in texts.items():
                binding = {
                    "runtime_sha256": runtime_sha, "input_manifest_sha256": runtime["input_manifest_sha256"],
                    "allocation_id": source["allocation_id"], "source_window_sha256": source["window_sha256"],
                    "variant": variant, "text_sha256": sha(text.encode()),
                }
                path = output / f"results/rows/{source['allocation_id']}__{variant}.json"
                row = checkpoint(path, binding)
                if row is None:
                    write(output / "HEARTBEAT.json", {
                        "status": "running", "stage": stage, "completed": done, "total": total,
                        "active_row": f"{source['allocation_id']}__{variant}", "updated_at": now(),
                    })
                    row = score_document(args, arms, recovery, source, variant, text, binding)
                    if stage == "null":
                        row["calibration"] = {}
                        for name, calibrator in calibrators.items():
                            result = row["arms"][name]
                            row["calibration"][name] = (
                                {"statistic": None, "p_value": 1.0, "is_watermarked": False, "scope": "abstain"}
                                if result["status"] == "abstain" else
                                asdict(calibrator.calibrate(result["statistic"], row["word_count"], .01))
                            )
                    write(path, row)
                (calibration_rows if stage == "calibration" else null_rows).append(row)
                done += 1
                write(output / "HEARTBEAT.json", {
                    "status": "running", "stage": stage, "completed": done, "total": total,
                    "updated_at": now(), "parser_unique_spans": len(recovery.records),
                })
                print(json.dumps({"stage": stage, "completed": done, "total": total,
                                  "statistics": {k: v["statistic"] for k, v in row["arms"].items()}}), flush=True)
        if stage == "calibration":
            calibrators = {
                name: fit(args, name, detector, calibration_rows, inputs["calibration"], runtime_sha)
                for name, detector in arms.items()
            }
    if len(calibration_rows) != 1000 or len(null_rows) != 1200:
        raise RuntimeError("Incomplete frozen denominator")
    result = summarize_null(null_rows, calibrators)
    result["binding_runtime_sha256"] = runtime_sha
    result["parser_recovery_summary"] = recovery.summary()
    write(output / "NULL_RESULT.json", result)
    lines = ["# SWAN v4 path0 开发比较：Null 阶段完成", "",
             "状态：50 个新正例槽位已冻结，待生成。当前没有 TPR 或鲁棒存活结果，不能裁为整体 Go。", "",
             "| 条件 | 对照误报 | path0 误报 |", "|---|---:|---:|"]
    for variant in ("baseline",) + VARIANTS:
        items = [result["arms"][arm]["primary"]["conditions"][variant] for arm in arms]
        lines.append(f"| {variant} | {items[0]['successes']}/200 | {items[1]['successes']}/200 |")
    lines.extend(["", "全部置信区间、原文级联合误报和预声明 alpha 曲线见 NULL_RESULT.json。", "",
                  "这批数据仅用于开发，正常长度与正式泛化尚未确认。"])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    write(output / "HEARTBEAT.json", {
        "status": "pending_generation", "stage": "null_complete", "completed": 1200, "total": 1200,
        "positive_slots": 50, "positive_generated": 0, "updated_at": now(),
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "p0", "history"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, default=ROOT)
    parser.add_argument("--capitalization-gap-words", type=int, default=12)
    parser.add_argument("--long-gap-stride-words", type=int, default=40)
    add_runtime_arguments(parser)
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    if args.command == "history":
        freeze(args.output_root / "LOCAL_HISTORY.json", history_snapshot(args.history_root, args.output_root))
        print(json.dumps({"status": "history_frozen", "path": str(args.output_root / "LOCAL_HISTORY.json")}))
        return
    if args.command == "p0":
        print(json.dumps(run_p0(args.bank), sort_keys=True))
        return

    def interrupted(signum, frame):
        raise RuntimeError(f"Process interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        execute(args)
    except BaseException as exc:
        write(args.output_root / "HEARTBEAT.json", {
            "status": "failed", "exception_type": type(exc).__name__, "updated_at": now(),
        })
        raise


if __name__ == "__main__":
    main()
