"""真实多视图开发流水线，prepare/precompute/develop/audit均不加载私有标签。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.amr_extractor import AmrlibBackend
from src.core.bank_scorer import SwanAMRBankScorer
from src.core.elastic_v3 import InvalidTolerantAMRExtractor
from src.core.multiview_candidates import candidate_windows, config_dict, select_budget
from src.core.multiview_detector import DetectorConfig, MultiviewDetector
from src.core.multiview_views import (
    ENCODER_ID, ENCODER_REVISION, SemanticEncoder, SurfaceAnalyzer,
)
from src.core.parser_recovery_audit import AuditedAmrlibBackend
from src.core.sqlite_cache import AMRCache, normalize_text


DEFAULT_INPUT = ROOT / "experiments/swan_multiview_development1/inputs/development_rows.jsonl"
DEFAULT_OUTPUT = ROOT / "experiments/swan_multiview_development1/pipeline1"
BANK = ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_100.json"
EXPECTED_BANK = "7e51393dacc5473ac9b3d96a12ba2a2591503432277ede174a125864e755a777"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_data(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(json_data(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def set_once(path, value):
    payload = json_data(value)
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"运行配置/输入已经变化，请另开目录：{path}")
    if not path.exists():
        atomic(path, value)


def setup(args):
    if sha(BANK) != EXPECTED_BANK:
        raise ValueError("Bank100 SHA不同")
    source_rows = load_jsonl(args.input)
    groups = sorted({row["group_id"] for row in source_rows},
                    key=lambda group: hashlib.sha256(group.encode()).hexdigest())
    if not 1 <= args.documents <= len(groups):
        raise ValueError("独立来源数量不足")
    selected = set(groups[:args.documents])
    rows = []
    for row in source_rows:
        if row["group_id"] not in selected:
            continue
        if hashlib.sha256(row["text"].encode()).hexdigest() != row["text_sha256"]:
            raise ValueError("输入SHA不一致")
        pool = candidate_windows(row["text"])
        rows.append({**row, "candidate_pool_count": len(pool["candidates"]),
                     "candidate_truncated_count": pool["truncated_count"],
                     "candidates": select_budget(pool["candidates"], args.budget)})
    model_dir = ROOT / "models/model_parse_xfm_bart_large-v0_1_0"
    tokenizer_dir = ROOT / "models/bart-large-tokenizer"
    assets = {str(path.relative_to(ROOT)): sha(path)
              for folder in (model_dir, tokenizer_dir)
              for path in sorted(folder.rglob("*")) if path.is_file()}
    encoder_snapshot = (args.encoder_cache / "models--sentence-transformers--all-MiniLM-L6-v2"
                        / "snapshots" / ENCODER_REVISION)
    if not (encoder_snapshot / "model.safetensors").is_file():
        raise FileNotFoundError("指定revision的本地编码器权重不完整")
    encoder_assets = {str(path.relative_to(encoder_snapshot)): sha(path)
                      for path in sorted(encoder_snapshot.rglob("*")) if path.is_file()}
    code_paths = [
        Path(__file__), ROOT / "src/core/multiview_candidates.py",
        ROOT / "src/core/multiview_views.py", ROOT / "src/core/multiview_alignment.py",
        ROOT / "src/core/multiview_detector.py", ROOT / "src/core/bank_scorer.py",
        ROOT / "src/core/amr_extractor.py", ROOT / "src/core/elastic_v3.py",
        ROOT / "src/core/parser_recovery_audit.py", ROOT / "src/core/sqlite_cache.py",
        ROOT / "src/core/models.py"]
    base = AmrlibBackend(str(model_dir), str(tokenizer_dir),
                        device=args.device, batch_size=args.batch_size)
    # 仅新流水线使用完整资产指纹，不改旧实验cache键及模型执行。
    partial_fingerprint = base.fingerprint
    identity = {"base": partial_fingerprint, "assets": assets}
    base._fingerprint = "amrlib-full:" + hashlib.sha256(json_data(identity)).hexdigest()
    journal = args.output / "private/parser_recovery.jsonl"
    imported_recovery = None
    if args.prior_run:
        source_journal = args.prior_run / "private/parser_recovery.jsonl"
        source_audit = AuditedAmrlibBackend(base, source_journal)
        imported_recovery = {"path": str(source_journal.resolve()),
                             "sha256": sha(source_journal), "records": len(source_audit.records)}
        if not journal.exists():
            journal.parent.mkdir(parents=True, exist_ok=True)
            with journal.open("xb") as stream:
                stream.write(source_journal.read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
    audited = AuditedAmrlibBackend(base, journal)
    cache = AMRCache(str(args.cache))
    extractor = InvalidTolerantAMRExtractor(audited, cache, request_batch_size=args.batch_size)
    scorer = SwanAMRBankScorer(extractor, str(BANK))
    manifest = {
        "status": "development_diagnostic_no_gold_no_null", "input": str(args.input),
        "input_sha256": sha(args.input), "group_ids": groups[:args.documents],
        "rows": len(rows), "candidate_config": config_dict(), "amr_budget": args.budget,
        "cache": str(args.cache.resolve()), "model_device": args.device,
        "model_batch_size": args.batch_size, "encoder_cache": str(args.encoder_cache),
        "parser_fingerprint": audited.fingerprint,
        "old_partial_parser_fingerprint": partial_fingerprint, "assets_sha256": assets,
        "encoder_id": ENCODER_ID, "encoder_revision": ENCODER_REVISION,
        "encoder_assets_sha256": encoder_assets,
        "imported_recovery": imported_recovery,
        "bank_sha256": sha(BANK), "bank_scorer": scorer.fingerprint,
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in code_paths},
        "dependencies": {package: importlib.metadata.version(package) for package in (
            "torch", "transformers", "sentence-transformers", "amrlib", "spacy", "penman", "scipy")},
        "detector_config": asdict(DetectorConfig()),
        "normal_length_confirmation": False, "gold_alignment_labels": 0,
        "new_fpr_tpr_measured": False,
    }
    set_once(args.output / "MANIFEST.json", manifest)
    set_once(args.output / "SELECTED_ROWS.json", rows)
    return rows, audited, extractor, scorer


def heartbeat(output, phase, completed, total, **extra):
    atomic(output / "HEARTBEAT.json", {
        "phase": phase, "completed": completed, "total": total,
        "updated_at": datetime.now(timezone.utc).isoformat(), **extra})


def texts_from(rows):
    return list(dict.fromkeys(normalize_text(candidate["text"])
                              for row in rows for candidate in row["candidates"]))


def require_cache(texts, backend, extractor):
    records = {}
    missing = []
    for text in texts:
        record = extractor.cache.get(text, backend.fingerprint)
        if record is None:
            missing.append(text)
        else:
            backend.assert_record(record)
            records[text] = record
    if missing:
        raise RuntimeError(f"cache-only 缺少{len(missing)}个片段；请单独运行precompute")
    return records


def precompute(args, rows, backend, extractor):
    texts = texts_from(rows)
    started = time.monotonic()
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start:start + args.batch_size]
        records = extractor.extract_many(batch)
        for record in records.values():
            backend.assert_record(record)
        heartbeat(args.output, "precompute", min(start + len(batch), len(texts)),
                  len(texts), stats=asdict(extractor.stats))
        print(json.dumps({"phase": "precompute", "completed": min(start + len(batch), len(texts)),
                          "total": len(texts), "elapsed_seconds": time.monotonic() - started}),
              flush=True)
    report = {"elapsed_seconds": time.monotonic() - started, "unique_texts": len(texts),
              "stats": asdict(extractor.stats), "recovery": backend.summary()}
    atomic(args.output / "PRECOMPUTE_LATEST.json", report)
    return report


def develop(args, rows, backend, extractor, scorer):
    records = require_cache(texts_from(rows), backend, extractor)
    encoder = SemanticEncoder(str(args.encoder_cache), device=args.device)
    surface = SurfaceAnalyzer()
    outputs, elapsed = [], 0.0
    for index, row in enumerate(rows):
        destination = args.output / "private/results" / (
            hashlib.sha256(row["row_id"].encode()).hexdigest() + ".json")
        started = time.monotonic()
        variants = {}
        for name, config in (
            ("predicted_rule", DetectorConfig()),
            ("predicted_fixed", DetectorConfig(weighting="fixed")),
            ("position_fixed", DetectorConfig(alignment_mode="position", weighting="fixed")),
            ("semantic_fixed", DetectorConfig(alignment_mode="semantic", weighting="fixed")),
            ("repaired_rule", DetectorConfig(warning_policy="include_repaired")),
        ):
            detector = MultiviewDetector(scorer, backend, encoder, surface, config)
            variants[name] = detector.detect(row["text"], row["candidates"], records)
        payload = {k: row[k] for k in ("row_id", "group_id", "variant_id", "text_sha256")}
        payload["variants"] = variants
        if destination.exists():
            if destination.read_bytes() != json_data(payload):
                raise RuntimeError("cache-only复放结果不同")
        else:
            atomic(destination, payload)
        elapsed += time.monotonic() - started
        outputs.append({"row_id": row["row_id"], "variant_id": row["variant_id"],
                        "result_sha256": sha(destination),
                        "variants": {name: {
                            k: value[k] for k in ("candidate_count", "usable_candidates",
                                                  "evidence_count", "statistic")}
                            | {"events": len(value["events"]),
                               "aligned_groups": len(value["alignment"]["selected"]),
                               "cross_candidate_matches": sum(
                                   edge["cross_candidate"] for edge in value["alignment"]["selected"]),
                               "rejected_pairs": len(value["alignment"]["rejected"])}
                            for name, value in variants.items()}})
        heartbeat(args.output, "develop_cache_only", index + 1, len(rows))
    report = {
        "status": "real_amr_development_diagnostic_no_performance_claim",
        "documents": len({row["group_id"] for row in rows}), "rows": len(rows),
        "candidate_occurrences": sum(len(row["candidates"]) for row in rows),
        "unique_texts": len(records), "cache_only": True, "parser_calls": 0,
        "elapsed_seconds_excluding_model_load": elapsed,
        "manual_alignment_labels": 0, "new_fpr_tpr_measured": False,
        "learned_reliability": False, "outputs": outputs,
    }
    summary = args.output / "DEVELOP_SUMMARY.json"
    if summary.exists():
        previous = json.loads(summary.read_text())
        if previous["outputs"] != outputs:
            raise RuntimeError("聚合结果复放不同")
        atomic(args.output / "REPLAY.json", {"identical_result_sha256": True,
                                            "rows": len(rows), "parser_calls": 0,
                                            "elapsed_seconds_excluding_model_load": elapsed})
    else:
        atomic(summary, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "precompute", "develop", "audit"))
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--documents", type=int, default=1)
    parser.add_argument("--budget", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache", type=Path, default=Path("/tmp/amr_multiview_20261020/pipeline1.db"))
    parser.add_argument("--encoder-cache", type=Path,
                        default=Path("/tmp/amr_multiview_20261020/encoder"))
    parser.add_argument("--prior-run", type=Path,
                        help="以完整parser指纹核验后，复制已有恢复审核；不改旧运行文件")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size必须为正")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output / ".writer.lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("运行目录已有写入锁；核实对应进程后再处理") from exc
    try:
        with os.fdopen(lock_fd, "w") as stream:
            stream.write(str(os.getpid()))
        rows, backend, extractor, scorer = setup(args)
        if args.phase == "precompute":
            report = precompute(args, rows, backend, extractor)
        elif args.phase == "develop":
            report = develop(args, rows, backend, extractor, scorer)
        elif args.phase == "audit":
            records = require_cache(texts_from(rows), backend, extractor)
            report = {"cache_only": True, "records": len(records),
                      "all_bound_recovery": True, "recovery": backend.summary()}
            atomic(args.output / "CACHE_AUDIT.json", report)
        else:
            report = {"selected_rows": len(rows), "unique_texts": len(texts_from(rows))}
        print(json.dumps({k: value for k, value in report.items() if k != "outputs"},
                         ensure_ascii=False, indent=2))
    except Exception as exc:
        heartbeat(args.output, "failed", 0, 0, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
