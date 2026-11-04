"""多视图开发前检：旧数据身份、候选窗口、待标注表及真实AMR烟测。

本脚本不提供新检测器，不运行正式误报/检出评价，不调用生成API。
旧30篇仅作为开发诊断；事实及攻击后的载体标签必须另行核验。
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.sqlite_cache import normalize_text

SOURCE = ROOT / "experiments/swan_elastic_v4_development1/inputs/development_rows.jsonl"
BANK = ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_100.json"
EXPECTED_BANK = "7e51393dacc5473ac9b3d96a12ba2a2591503432277ede174a125864e755a777"
DEFAULT_OUTPUT = ROOT / "experiments/swan_multiview_development1"
VERSION = "swan-multiview-preflight:v1"
WINDOW_CONFIG = {"lengths": [12, 24, 48], "stride": 6, "max_words": 80, "max_candidates": 128}
ENCODER_ID = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def freeze(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"现有输出与当前运行不一致：{path}")
        return
    with path.open("xb") as stream:
        stream.write(payload)


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def candidate_windows(text: str) -> dict:
    """只截取输入现有文字，保留精确字符偏移；窗口不依赖原句界真值。"""
    tokens = list(re.finditer(r"\S+", text))
    spans = {}

    def add(left, right, kind):
        fragment = text[left:right]
        trimmed = fragment.strip()
        if not trimmed or len(trimmed.split()) > WINDOW_CONFIG["max_words"]:
            return
        left += len(fragment) - len(fragment.lstrip())
        right = left + len(trimmed)
        spans.setdefault((left, right), set()).add(kind)

    add(0, len(text), "whole_text")
    start = 0
    for boundary in re.finditer(r"[.!?;]\s+|\n+", text):
        add(start, boundary.start() + 1, "surface_boundary")
        start = boundary.end()
    add(start, len(text), "surface_boundary")
    for size in WINDOW_CONFIG["lengths"]:
        if not tokens:
            continue
        starts = set(range(0, max(1, len(tokens) - size + 1), WINDOW_CONFIG["stride"]))
        starts.add(max(0, len(tokens) - size))
        for offset in sorted(starts):
            end = min(len(tokens), offset + size)
            add(tokens[offset].start(), tokens[end - 1].end(), f"word_window_{size}")
    ordered = sorted(spans, key=lambda key: (
        0 if "whole_text" in spans[key] else 1 if "surface_boundary" in spans[key] else 2,
        key[0], key[1],
    ))
    total = len(ordered)
    ordered = ordered[:WINDOW_CONFIG["max_candidates"]]
    windows = []
    for left, right in ordered:
        span = text[left:right]
        windows.append({
            "start_char": left, "end_char": right, "text": span,
            "span_sha256": digest_bytes(normalize_text(span).encode()),
            "word_count": len(span.split()), "kinds": sorted(spans[(left, right)]),
        })
    return {"candidates": windows, "enumerated_count": total, "truncated_count": total - len(windows)}


def prepare(output: Path, documents: int) -> dict:
    if not 1 <= documents <= 30:
        raise ValueError("开发文档数必须介于1和30")
    if digest(BANK) != EXPECTED_BANK:
        raise ValueError("Bank100哈希不一致")
    all_rows = load_rows(SOURCE)
    baselines = [r for r in all_rows if r["cohort"] == "confirmation_baseline_positive"]
    selected = sorted(baselines, key=lambda r: digest_bytes(r["group_id"].encode()))[:documents]
    if len(selected) != documents:
        raise ValueError("开发文档数量不足")
    group_ids = {r["group_id"] for r in selected}
    rows = [r for r in all_rows if r["group_id"] in group_ids]
    if len({r["row_id"] for r in rows}) != len(rows):
        raise ValueError("重复row_id")
    candidates, traces, references = [], [], {}
    for row in rows:
        if digest_bytes(row["text"].encode()) != row["text_sha256"]:
            raise ValueError(f"原文字节哈希不一致：{row['row_id']}")
        if digest_bytes(normalize_text(row["text"]).encode()) != row["normalized_text_sha256"]:
            raise ValueError(f"规范化文字哈希不一致：{row['row_id']}")
        for source, expected in row["source_sha256"].items():
            if digest(ROOT / source) != expected:
                raise ValueError(f"输入引用哈希不一致：{source}")
            references[source] = expected
        candidates.append({
            "row_id": row["row_id"], "group_id": row["group_id"],
            "variant_id": row["variant_id"], "text_sha256": row["text_sha256"],
            **candidate_windows(row["text"]),
        })
        if row["cohort"] != "confirmation_baseline_positive":
            continue
        raw = json.loads((ROOT / row["source_paths"][0]).read_text())
        if raw["text"] != row["text"] or raw["prompt_id"] != row["group_id"]:
            raise ValueError("生成记录与输入身份不同")
        cursor, slots = 0, []
        for index, slot in enumerate(raw["sentences"]):
            left = row["text"].find(slot["text"], cursor)
            if left < 0:
                raise ValueError("生成槽位无法在原文字中定位")
            right = left + len(slot["text"])
            cursor = right
            slots.append({
                "slot_id": f"{row['group_id']}:slot-{index:02d}",
                "start_char": left, "end_char": right, "text": slot["text"],
                "chosen_template": slot["chosen_template"],
                "generation_accepted": slot["accepted"], "generation_s2match": slot["s2match"],
                "fact_id": None, "human_verified_carrier": None,
            })
        traces.append({
            "group_id": row["group_id"], "baseline_text_sha256": row["text_sha256"],
            "all_injection_accepted": raw["all_injection_accepted"], "slots": slots,
            "label_status": "generation_trace_only_not_gold",
        })
    output.mkdir(parents=True, exist_ok=True)
    row_payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode()
    candidate_payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in candidates).encode()
    freeze(output / "inputs/development_rows.jsonl", row_payload)
    freeze(output / "inputs/candidate_rows.jsonl", candidate_payload)
    freeze(output / "private/generation_traces.json", json_bytes(traces))
    annotation = output / "private/evidence_review.tsv"
    if not annotation.exists():
        annotation.parent.mkdir(parents=True, exist_ok=True)
        with annotation.open("x", newline="") as stream:
            fields = ["group_id", "row_id", "variant_id", "text", "reviewer_id",
                      "fact_id", "support_spans_json", "fact_preserved", "carrier_preserved",
                      "quality_pass", "notes"]
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row[k] for k in fields if k in row})
    report = {
        "version": VERSION, "status": "prepared_development_diagnostic_not_confirmation",
        "selected_documents": len(group_ids), "rows": len(rows),
        "source": str(SOURCE.relative_to(ROOT)), "source_sha256": digest(SOURCE),
        "reference_sha256": references, "script_sha256": digest(Path(__file__)),
        "window_config": WINDOW_CONFIG, "bank_sha256": digest(BANK),
        "selection": "按完整group_id的SHA排序，未按检测结果筛选",
        "word_range_baseline": [min(r["word_count"] for r in selected), max(r["word_count"] for r in selected)],
        "baseline_generation_accepted_slots": sum(s["generation_accepted"] for t in traces for s in t["slots"]),
        "baseline_slots": sum(len(t["slots"]) for t in traces),
        "baseline_all_accepted_documents": sum(t["all_injection_accepted"] for t in traces),
        "candidate_occurrences": sum(len(r["candidates"]) for r in candidates),
        "unique_candidate_texts": len({c["span_sha256"] for r in candidates for c in r["candidates"]}),
        "truncated_rows": sum(r["truncated_count"] > 0 for r in candidates),
        "gold_alignment_labels": 0, "normal_length_qualified_documents": 0,
        "new_detector_implemented": False, "new_fpr_tpr_measured": False,
        "outputs_sha256": {
            "development_rows": digest(output / "inputs/development_rows.jsonl"),
            "candidate_rows": digest(output / "inputs/candidate_rows.jsonl"),
            "generation_traces": digest(output / "private/generation_traces.json"),
        },
    }
    if report["bank_sha256"] != EXPECTED_BANK:
        raise ValueError("Bank100哈希不一致")
    freeze(output / "PREFLIGHT.json", json_bytes(report))
    return report


def smoke(output: Path, cache: Path, limit: int) -> dict:
    from src.core.amr_extractor import AmrlibBackend, AMRExtractor
    from src.core.bank_scorer import SwanAMRBankScorer, template_features, _multiset_f1
    from src.core.sqlite_cache import AMRCache
    rows = load_rows(output / "inputs/development_rows.jsonl")
    selected = []
    for cohort in ["confirmation_baseline_positive", "confirmation_boundary_attack", "confirmation_punctuation_stress"]:
        row = next(r for r in rows if r["cohort"] == cohort)
        selected.append(candidate_windows(row["text"])["candidates"][0]["text"])
    selected = selected[:limit]
    backend = AmrlibBackend(
        str(ROOT / "models/model_parse_xfm_bart_large-v0_1_0"),
        str(ROOT / "models/bart-large-tokenizer"), device="cpu", batch_size=2,
    )
    extractor = AMRExtractor(backend, AMRCache(str(cache)), request_batch_size=2)
    scorer = SwanAMRBankScorer(extractor, str(BANK))
    logs = []

    class WarningRecorder(logging.Handler):
        def emit(self, record):
            logs.append({"logger": record.name, "level": record.levelname,
                         "message": record.getMessage()})

    handler = WarningRecorder(level=logging.WARNING)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    import amrlib
    amrlib_version = amrlib.__version__
    parser_logger = logging.getLogger("amrlib.models.parse_xfm.penman_serializer")
    parser_logger.addHandler(handler)
    started = time.monotonic()
    try:
        backend._load()
        records = extractor.extract_many(selected)
    finally:
        root_logger.removeHandler(handler)
        parser_logger.removeHandler(handler)
    logs = list({json.dumps(item, sort_keys=True): item for item in logs}.values())
    cold_seconds = time.monotonic() - started
    before = asdict(extractor.stats)
    started = time.monotonic()
    replay = extractor.extract_many(selected)
    hot_seconds = time.monotonic() - started
    observations = []
    for text, record in records.items():
        features = template_features(record.amr)
        scores = []
        for index, reference in enumerate(scorer._bank_features):
            components = [_multiset_f1(features.concepts, reference.concepts),
                          _multiset_f1(features.relations, reference.relations),
                          _multiset_f1(features.paths, reference.paths)]
            scores.append({"template_index": index, "components": components,
                           "score": sum(w * v for w, v in zip([.25, .5, .25], components))})
        scores.sort(key=lambda item: (-item["score"], item["template_index"]))
        observations.append({
            "text_sha256": record.text_sha256, "word_count": len(text.split()),
            "graph_valid": record.graph_valid, "roundtrip_valid": record.roundtrip_valid,
            "warnings": list(record.penman_warnings), "top2": scores[:2],
            "score_count": len(scores), "replay_amr_identical": replay[text].amr == record.amr,
        })
    report = {
        "version": VERSION, "status": "diagnostics_present_review_required" if logs else "real_amr_smoke_only",
        "requested_texts": len(selected), "unique_parsed_texts": len(records),
        "cold_seconds": cold_seconds, "hot_seconds": hot_seconds,
        "cold_stats": before, "final_stats": asdict(extractor.stats),
        "parser_fingerprint": backend.fingerprint, "bank_sha256": digest(BANK),
        "observations": observations, "new_detector_implemented": False,
        "new_fpr_tpr_measured": False,
        "backend_warning_logs": logs, "backend_warning_count": len(logs),
        "amrlib_version": amrlib_version,
        "script_sha256": digest(Path(__file__)),
    }
    freeze(output / f"AMR_AUDITED_SMOKE_{cache.stem}.json", json_bytes(report))
    return report


def encoder_smoke(output: Path, cache: Path) -> dict:
    """真实编码器可用性检查；向量相似度不作为水印判定。"""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    candidate_rows = load_rows(output / "inputs/candidate_rows.jsonl")
    texts = list(dict.fromkeys(
        normalize_text(c["text"]) for r in candidate_rows for c in r["candidates"]
    ))
    started = time.monotonic()
    model = SentenceTransformer(
        ENCODER_ID, revision=ENCODER_REVISION, device="cpu",
        cache_folder=str(cache.parent / "encoder"), token=False,
    )
    model_load_seconds = time.monotonic() - started
    started = time.monotonic()
    vectors = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    encode_seconds = time.monotonic() - started
    repeated = model.encode(texts[:3], batch_size=3, normalize_embeddings=True, show_progress_bar=False)
    report = {
        "version": VERSION, "status": "real_normalized_encoder_smoke_only",
        "model_id": ENCODER_ID, "revision": ENCODER_REVISION,
        "normalized_texts": len(texts), "dimension": int(vectors.shape[1]),
        "model_load_seconds": model_load_seconds, "encode_seconds": encode_seconds,
        "max_replay_absolute_error": float(np.max(np.abs(vectors[:3] - repeated))),
        "finite": bool(np.isfinite(vectors).all()), "script_sha256": digest(Path(__file__)),
        "new_alignment_evaluated": False, "new_fpr_tpr_measured": False,
    }
    freeze(output / "ENCODER_NORMALIZED_SMOKE.json", json_bytes(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "smoke", "encoder"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--documents", type=int, default=12)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/amr_multiview_20261020/amr_smoke.db"))
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    if args.phase == "prepare":
        result = prepare(args.output, args.documents)
    elif args.phase == "smoke":
        result = smoke(args.output, args.cache, args.limit)
    else:
        result = encoder_smoke(args.output, args.cache)
    print(json.dumps({k: v for k, v in result.items() if k not in (
        "reference_sha256", "outputs_sha256", "observations")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
