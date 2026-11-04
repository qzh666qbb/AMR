"""dataset4 完整流程：攻击变体 + 多视图检测 + 载体恢复率。

输入：experiments/swan_multiview_development1/dataset4/documents.json
输出：experiments/swan_multiview_development1/dataset4_attack1/

载体位置由构造已知（每个 slot 的 template_index + text），不需人工标注。
检测器使用现役 light-weight bank_scorer（multiset-F1），非官方 S2Match；
报告中明确标注此口径差异。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party/swan_s2match_runtime_711a231d"))

from src.core.amr_extractor import AmrlibBackend, AMRExtractor  # noqa: E402
from src.core.bank_scorer import SwanAMRBankScorer  # noqa: E402
from src.core.multiview_candidates import WindowConfig, candidate_windows, select_budget  # noqa: E402
from src.core.multiview_detector import DetectorConfig, MultiviewDetector  # noqa: E402
from src.core.multiview_views import SemanticEncoder, SurfaceAnalyzer  # noqa: E402
from src.core.parser_recovery_audit import AuditedAmrlibBackend  # noqa: E402
from src.core.sqlite_cache import AMRCache, normalize_text  # noqa: E402

from probe_swan_multitemplate_yield import strip_snt, normalize_to_template  # noqa: E402
import amr_py3  # noqa: E402
from s2match import compute_s2match_from_strings  # noqa: E402

if not hasattr(amr_py3, "log_helper"):
    class _SilentLog:
        def debug(self, *a): pass
        def error(self, *a): pass
    amr_py3.log_helper = _SilentLog()

GLOVE = ROOT / "vectors/glove.6B.100d.txt"

# 预加载 GloVe 向量（compute_s2match_from_strings 每次都会 load_vecs，缓存避免重复读 347MB）
import s2match as _s2match_mod  # noqa: E402
_S2MATCH_VECS = _s2match_mod.load_vecs(str(GLOVE))
_s2match_mod.load_vecs = lambda fp: _S2MATCH_VECS

BANK = ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_100.json"
DATASET4 = ROOT / "experiments/swan_multiview_development1/dataset4/documents.json"
DEFAULT_OUT = ROOT / "experiments/swan_multiview_development1/dataset4_attack1"

# 候选窗口：去掉 48 词窗口（CPU 上 24s/条），保留 12 词窗口 + 句界。
WINDOW = WindowConfig(lengths=(12,), stride=6, max_words=24, max_candidates=128)


# ───────────────────────── 攻击变体 ─────────────────────────

def _sentence_endings(text: str) -> list:
    return [(m.start(), m.end()) for m in re.finditer(r"[.!?]", text)]


def attack_baseline(slots: list) -> tuple:
    """原文：slot 文本用空格连接。"""
    parts = [s["text"] for s in slots]
    text = " ".join(parts)
    spans = _span_map(text, parts)
    return text, spans


def attack_semicolon(slots: list) -> tuple:
    parts = [re.sub(r"[.!?]$", ";", s["text"]) for s in slots]
    text = " ".join(parts)
    spans = _span_map(text, parts)
    return text, spans


def attack_comma(slots: list) -> tuple:
    parts = [re.sub(r"[.!?]$", ",", s["text"]) for s in slots]
    text = " ".join(parts)
    spans = _span_map(text, parts)
    return text, spans


def attack_newline(slots: list) -> tuple:
    parts = [s["text"] for s in slots]
    text = "\n".join(parts)
    spans = _span_map_newline(text, parts)
    return text, spans


def attack_space(slots: list) -> tuple:
    parts = [re.sub(r"[.!?]$", "", s["text"]) for s in slots]
    text = " ".join(parts)
    spans = _span_map(text, parts)
    return text, spans


def attack_split(slots: list) -> tuple:
    """偶数 slot 在中间词处拆成两句。"""
    parts = []
    for i, s in enumerate(slots):
        words = s["text"].split()
        if i % 2 == 0 and len(words) >= 4:
            mid = len(words) // 2
            first = " ".join(words[:mid]) + "."
            second = " ".join(words[mid:])
            if not second[-1] in ".!?":
                second += "."
            parts.append(first + " " + second)
        else:
            parts.append(s["text"])
    text = " ".join(parts)
    spans = _span_map(text, parts)
    return text, spans


def attack_merge(slots: list) -> tuple:
    """合并相邻 slot 对：把句号换成逗号 and。"""
    parts = []
    i = 0
    while i < len(slots):
        if i + 1 < len(slots):
            a = re.sub(r"[.!?]$", "", slots[i]["text"])
            b = slots[i + 1]["text"]
            parts.append(a + ", and " + b)
            i += 2
        else:
            parts.append(slots[i]["text"])
            i += 1
    text = " ".join(parts)
    spans = _span_map_merge(text, parts, slots)
    return text, spans


def _span_map(text: str, parts: list) -> list:
    """返回每个 part 在 text 中的 (start, end)。"""
    spans = []
    cursor = 0
    for part in parts:
        idx = text.find(part, cursor)
        if idx < 0:
            raise ValueError(f"part not found: {part[:40]!r}")
        spans.append((idx, idx + len(part)))
        cursor = idx + len(part)
    return spans


def _span_map_newline(text: str, parts: list) -> list:
    spans = []
    cursor = 0
    for part in parts:
        idx = text.find(part, cursor)
        spans.append((idx, idx + len(part)))
        cursor = idx + len(part) + 1  # +1 for newline
    return spans


def _span_map_merge(text: str, parts: list, slots: list) -> list:
    """合并后每个 part 对应两个原始 slot，返回 (slot_a_span, slot_b_span) 列表。"""
    spans = []
    cursor = 0
    si = 0
    for part in parts:
        idx = text.find(part, cursor)
        if si + 1 < len(slots) and ", and " in part:
            a_text = re.sub(r"[.!?]$", "", slots[si]["text"])
            b_text = slots[si + 1]["text"]
            a_idx = part.find(a_text)
            b_idx = part.find(b_text, a_idx + len(a_text))
            spans.append({
                "merged": True,
                "slot_a": (idx + a_idx, idx + a_idx + len(a_text)),
                "slot_b": (idx + b_idx, idx + b_idx + len(b_text)),
            })
            si += 2
        else:
            s_text = slots[si]["text"]
            s_idx = part.find(s_text)
            spans.append({
                "merged": False,
                "slot_a": (idx + s_idx, idx + s_idx + len(s_text)),
                "slot_b": None,
            })
            si += 1
        cursor = idx + len(part)
    return spans


ATTACKS = {
    "baseline": attack_baseline,
    "split": attack_split,
    "merge": attack_merge,
    "semicolon": attack_semicolon,
    "comma": attack_comma,
    "newline": attack_newline,
    "space": attack_space,
}


# ───────────────────────── 载体恢复率 ─────────────────────────

def _overlap(a: tuple, b: tuple) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def compute_recovery(slots: list, spans: list, evidence: list, candidates: list,
                     records: dict, bank_templates: list, attack_name: str) -> dict:
    """计算载体恢复率（双口径）。

    - light 口径：检测器证据中存在 template_index 匹配且 span 重叠的项。
      注意 light scorer 的 top-1 模板可能与注入时 S2Match 的模板不同。
    - s2match 口径：与 carrier span 重叠的候选中，S2Match(候选AMR, 载体模板)
      的最大值 >= 0.75。使用与注入时一致的官方 S2Match。
    """
    s2match_cache = {}

    def s2match_vs(amr: str, tpl_idx: int) -> float:
        key = (amr, tpl_idx)
        if key not in s2match_cache:
            try:
                norm = normalize_to_template(strip_snt(amr))
                _, _, score = compute_s2match_from_strings(
                    norm, bank_templates[tpl_idx], vectors_file=str(GLOVE))
                s2match_cache[key] = round(float(score), 4)
            except Exception:
                s2match_cache[key] = None
        return s2match_cache[key]

    results = []
    for si, slot in enumerate(slots):
        tpl = slot["template_index"]
        # 定位该 slot 在变体文本中的 span
        if attack_name == "merge":
            slot_span = None
            for sp in spans:
                if sp.get("_slot_a_idx") == si:
                    slot_span = sp["slot_a"]; break
                if sp.get("_slot_b_idx") == si:
                    slot_span = sp["slot_b"]; break
        else:
            slot_span = spans[si] if si < len(spans) else None

        if slot_span is None:
            results.append({"slot": si, "template_index": tpl,
                            "light_recovered": False, "s2match_recovered": False})
            continue

        # light 口径：检测器证据
        light_ok = False
        light_ev = None
        for ev in evidence:
            if ev["template_index"] != tpl:
                continue
            cand = candidates[ev["candidate_index"]]
            if _overlap(slot_span, (cand["start_char"], cand["end_char"])):
                light_ok = True
                light_ev = ev
                break

        # s2match 口径：所有重叠候选的 AMR vs 载体模板
        best_s2 = 0.0
        best_cand = None
        for ci, cand in enumerate(candidates):
            if not _overlap(slot_span, (cand["start_char"], cand["end_char"])):
                continue
            norm = normalize_text(cand["text"])
            rec = records.get(norm)
            if rec is None or not rec.graph_valid:
                continue
            s = s2match_vs(rec.amr, tpl)
            if s is not None and s > best_s2:
                best_s2 = s
                best_cand = ci

        results.append({
            "slot": si, "template_index": tpl,
            "slot_text": slot["text"],
            "light_recovered": light_ok,
            "light_evidence_score": round(light_ev["bank_score"], 4) if light_ev else None,
            "s2match_recovered": best_s2 >= 0.75,
            "s2match_best": best_s2,
            "s2match_best_candidate": best_cand,
        })
    total = len(results)
    light_rec = sum(r["light_recovered"] for r in results)
    s2_rec = sum(r["s2match_recovered"] for r in results)
    return {
        "attack": attack_name,
        "total_carriers": total,
        "light_recovered": light_rec,
        "light_recovery_rate": round(light_rec / total, 4) if total else 0.0,
        "s2match_recovered": s2_rec,
        "s2match_recovery_rate": round(s2_rec / total, 4) if total else 0.0,
        "per_slot": results,
    }


def _annotate_merge_spans(spans: list) -> None:
    """为 merge 的 span 标注原始 slot 索引。"""
    si = 0
    for sp in spans:
        if sp["merged"]:
            sp["_slot_a_idx"] = si
            sp["_slot_b_idx"] = si + 1
            si += 2
        else:
            sp["_slot_a_idx"] = si
            si += 1


# ───────────────────────── 主流程 ─────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache", type=Path,
                    default=Path("/tmp/amr_multiview_20261020/dataset4_test.db"))
    ap.add_argument("--encoder-cache", type=Path,
                    default=Path("/tmp/amr_multiview_20261020/encoder"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--attacks", nargs="+", default=list(ATTACKS.keys()),
                    choices=list(ATTACKS.keys()))
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    audit_path = Path("/tmp/amr_multiview_20261020/dataset4_test_recovery.jsonl")
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(DATASET4.read_text())
    docs = data["documents"]
    bank_data = json.loads(BANK.read_text())
    bank_templates = [row["normalized_template_amr"] for row in bank_data]
    print(f"loaded {len(docs)} documents from dataset4", flush=True)

    # 初始化 parser / scorer / detector
    backend = AmrlibBackend(
        str(ROOT / "models/model_parse_xfm_bart_large-v0_1_0"),
        str(ROOT / "models/bart-large-tokenizer"),
        device=args.device, batch_size=args.batch_size,
    )
    print(f"parser fingerprint: {backend.fingerprint}", flush=True)
    cache = AMRCache(str(args.cache))
    extractor = AMRExtractor(backend, cache, request_batch_size=args.batch_size)
    scorer = SwanAMRBankScorer(extractor, str(BANK))
    audited = AuditedAmrlibBackend(backend, audit_path)
    extractor.backend = audited

    encoder = SemanticEncoder(str(args.encoder_cache), device=args.device)
    surface = SurfaceAnalyzer()
    detector = MultiviewDetector(scorer, audited, encoder, surface, DetectorConfig())

    all_results = []
    all_detection_outputs = []

    for di, doc in enumerate(docs):
        slots = doc["slots"]
        doc_id = doc["document_id"]
        print(f"\n=== {doc_id} ({len(slots)} slots) ===", flush=True)

        for attack_name in args.attacks:
            attack_fn = ATTACKS[attack_name]
            if attack_name == "merge":
                text, spans = attack_fn(slots)
                _annotate_merge_spans(spans)
            else:
                text, spans = attack_fn(slots)

            # 候选窗口
            pool = candidate_windows(text, WINDOW)
            candidates = select_budget(pool["candidates"], 0)
            texts = list(dict.fromkeys(normalize_text(c["text"]) for c in candidates))

            # 解析 AMR（cache 优先）
            t0 = time.monotonic()
            records = extractor.extract_many(texts)
            parse_s = time.monotonic() - t0
            print(f"  [{attack_name}] {len(candidates)} cands, {len(texts)} unique, "
                  f"parse {parse_s:.1f}s, stats={vars(extractor.stats)}", flush=True)

            # 检测
            t0 = time.monotonic()
            result = detector.detect(text, candidates, records)
            detect_s = time.monotonic() - t0

            # 恢复率（双口径）
            recovery = compute_recovery(slots, spans, result["evidence"],
                                        candidates, records, bank_templates, attack_name)

            print(f"    statistic={result['statistic']:.4f} "
                  f"evidence={result['evidence_count']} "
                  f"usable={result['usable_candidates']} "
                  f"light={recovery['light_recovered']}/{recovery['total_carriers']} "
                  f"s2match={recovery['s2match_recovered']}/{recovery['total_carriers']}",
                  flush=True)

            all_results.append({
                "document_id": doc_id,
                "attack": attack_name,
                "word_count": len(text.split()),
                "candidate_count": len(candidates),
                "usable_candidates": result["usable_candidates"],
                "evidence_count": result["evidence_count"],
                "statistic": round(result["statistic"], 6),
                "parse_seconds": round(parse_s, 1),
                "detect_seconds": round(detect_s, 1),
                **{k: v for k, v in recovery.items() if k != "per_slot"},
            })
            all_detection_outputs.append({
                "document_id": doc_id,
                "attack": attack_name,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "statistic": result["statistic"],
                "evidence_count": result["evidence_count"],
                "evidence": result["evidence"],
                "recovery": recovery,
                "candidate_count": len(candidates),
                "candidates": [{"start_char": c["start_char"], "end_char": c["end_char"],
                                "text": c["text"], "word_count": c["word_count"]}
                               for c in candidates],
            })

    # 汇总
    by_attack = {}
    for r in all_results:
        by_attack.setdefault(r["attack"], []).append(r)
    summary = {}
    for attack, rows in by_attack.items():
        total = sum(r["total_carriers"] for r in rows)
        light_rec = sum(r["light_recovered"] for r in rows)
        s2_rec = sum(r["s2match_recovered"] for r in rows)
        summary[attack] = {
            "documents": len(rows),
            "total_carriers": total,
            "light_recovered": light_rec,
            "light_recovery_rate": round(light_rec / total, 4) if total else 0.0,
            "s2match_recovered": s2_rec,
            "s2match_recovery_rate": round(s2_rec / total, 4) if total else 0.0,
            "mean_statistic": round(sum(r["statistic"] for r in rows) / len(rows), 6),
            "mean_evidence": round(sum(r["evidence_count"] for r in rows) / len(rows), 2),
        }

    output = {
        "dataset": "dataset4",
        "dataset_sha256": hashlib.sha256(DATASET4.read_bytes()).hexdigest(),
        "bank_sha256": scorer.bank_sha256,
        "scorer": "lightweight_multiset_f1 (NOT official S2Match)",
        "window_config": {"lengths": list(WINDOW.lengths), "stride": WINDOW.stride,
                          "max_words": WINDOW.max_words},
        "detector_config": {"score_center": 0.65, "alignment_mode": "predicted",
                            "weighting": "rule"},
        "attacks": args.attacks,
        "documents": len(docs),
        "summary_by_attack": summary,
        "per_document_per_attack": all_results,
    }
    (args.output / "RECOVERY_SUMMARY.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2))
    (args.output / "detection_outputs.json").write_text(
        json.dumps(all_detection_outputs, ensure_ascii=False, indent=2))

    print("\n" + "=" * 70)
    print("RECOVERY RATE BY ATTACK (light scorer vs official S2Match)")
    print("=" * 70)
    print(f"{'attack':<12} {'light':>8} {'s2match':>8} {'total':>6} {'stat':>10}")
    print("-" * 70)
    for attack in args.attacks:
        s = summary[attack]
        print(f"{attack:<12} {s['light_recovery_rate']:>8.1%} "
              f"{s['s2match_recovery_rate']:>8.1%} "
              f"{s['total_carriers']:>6} "
              f"{s['mean_statistic']:>10.4f}")
    print(f"\nwritten: {args.output / 'RECOVERY_SUMMARY.json'}")


if __name__ == "__main__":
    main()
