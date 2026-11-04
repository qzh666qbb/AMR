"""P1 标点恢复基线（30 篇 / 471 槽版本）。

与早期 2 篇 / 32 槽版本的口径一致，但复用已有分数以避免重复解析：
- none：直接取 FUSION_RECOVERY 中各攻击的 vec_d2；
- oracle：恢复后等于原文，等价于 baseline 行；
- blind semicolon / comma / newline：若恢复文本与原文逐字相同，复用 baseline 行；
- blind space：必须重新解析（边界被删除，只能靠大小写启发式猜测）。

分母固定为总槽数 471：恢复后无法定位的载体计为失败。
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party/swan_s2match_runtime_711a231d"))

from src.core.amr_extractor import AmrlibBackend, AMRExtractor  # noqa: E402
from src.core.multiview_candidates import WindowConfig, candidate_windows, select_budget  # noqa: E402
from src.core.sqlite_cache import AMRCache, normalize_text  # noqa: E402

from run_dataset4_fusion_recovery import (  # noqa: E402
    BANK, S2Scorer, elementwise_max, event_subgraphs,
)
from run_dataset4_attack_recovery import ATTACKS  # noqa: E402

DATASET5 = ROOT / "experiments/swan_multiview_development1/dataset5/documents.json"
FUSION = ROOT / "experiments/swan_multiview_development1/dataset5_eval/FUSION_RECOVERY.json"
NULL = ROOT / "experiments/swan_multiview_development1/null_fp_audit1/PARTIAL_WINDOW_200.json"
OUT = ROOT / "experiments/swan_multiview_development1/null_fp_audit1/P1_PUNCT_30DOC.json"
WINDOW = WindowConfig(lengths=(12,), stride=6, max_words=24, max_candidates=128)
ATTACKS_ORDER = ("baseline", "semicolon", "comma", "space", "newline", "split", "merge")


def restore_blind(text: str, attack: str) -> str:
    if attack == "semicolon":
        return text.replace(";", ".")
    if attack == "comma":
        return text.replace(",", ".")
    if attack == "newline":
        return text
    if attack == "space":
        return re.sub(r"(?<=[a-z0-9])\s+(?=[A-Z])", ". ", text)
    return text


def span_map(text: str, slots: list) -> list:
    """定位每个 slot 的字符区间，并把句末标点一并计入（与攻击函数的 span 口径一致）。"""
    spans, cursor = [], 0
    for s in slots:
        body = re.sub(r"[.!?;,]$", "", s["text"])
        idx = text.find(body, cursor)
        if idx < 0:
            spans.append(None)
            continue
        end = idx + len(body)
        if end < len(text) and text[end] in ".!?;,":
            end += 1
        spans.append((idx, end))
        cursor = end
    return spans


def build_ecdf(units, key):
    n_tpl = len(units[0][key])
    cols = [[] for _ in range(n_tpl)]
    for u in units:
        for t, v in enumerate(u[key]):
            if v is not None:
                cols[t].append(v)
    return [sorted(c) for c in cols]


def p_emp(arr, score):
    n = len(arr)
    return 1.0 if n == 0 else (1 + n - bisect.bisect_left(arr, score)) / (1 + n)


def e_stat(d2, ecdf):
    best = None
    for t, v in enumerate(d2):
        if v is None:
            continue
        x = -2.0 * math.log(p_emp(ecdf[t], v))
        best = x if best is None or x > best else best
    return best


def tau_at(scores, q):
    s = sorted([x for x in scores if x is not None], reverse=True)
    k = max(1, math.ceil(q * len(s)))
    return s[min(k, len(s)) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET5)
    ap.add_argument("--fusion", type=Path, default=FUSION)
    ap.add_argument("--null", type=Path, default=NULL)
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cache", type=Path, default=Path("/tmp/p1_30doc.db"))
    ap.add_argument("--doc-offset", type=int, default=0)
    ap.add_argument("--limit-docs", type=int, default=0)
    args = ap.parse_args()

    null = [u for u in json.loads(args.null.read_text())["per_unit"] if u.get("vec_d2")]
    calib = [u for u in null if u["cohort"] == "calibration"]
    test = [u for u in null if u["cohort"] == "independent_test"]
    ecdf = build_ecdf(calib, "vec_d2")
    test_e = [e_stat(u["vec_d2"], ecdf) for u in test]
    taus = {q: tau_at(test_e, q) for q in (0.01, 0.05)}

    rows = json.loads(args.fusion.read_text())["per_instance"]
    by_key = {(r["doc"], r["attack"], r["slot"]): r["vec_d2"] for r in rows}
    docs = json.loads(args.dataset.read_text())["documents"]
    if args.doc_offset:
        docs = docs[args.doc_offset:]
    if args.limit_docs:
        docs = docs[: args.limit_docs]
    total_slots = sum(len(d["slots"]) for d in docs)

    bank = [r["normalized_template_amr"] for r in json.loads(BANK.read_text())]
    backend = AmrlibBackend(str(ROOT / "models/model_parse_xfm_bart_large-v0_1_0"),
                            str(ROOT / "models/bart-large-tokenizer"),
                            device=args.device, batch_size=args.batch_size)
    cache = AMRCache(str(args.cache))
    extractor = AMRExtractor(backend, cache, request_batch_size=args.batch_size)
    scorer = S2Scorer(bank)

    results = {a: {"none": [], "blind": [], "oracle": []} for a in ATTACKS_ORDER}
    reused = {"blind": 0, "computed": 0}

    for doc in docs:
        slots = doc["slots"]
        baseline_text = " ".join(s["text"] for s in slots)
        for attack in ATTACKS_ORDER:
            text_orig, spans_orig = ATTACKS[attack](slots)

            # none：直接复用已有分数
            for si in range(len(slots)):
                v = by_key.get((doc["document_id"], attack, si))
                results[attack]["none"].append(e_stat(v, ecdf) if v else None)
            # oracle：仅对"恢复有意义"的攻击给出上界（split/merge 不可恢复）
            if attack in ("baseline", "semicolon", "comma", "space", "newline"):
                for si in range(len(slots)):
                    v = by_key.get((doc["document_id"], "baseline", si))
                    results[attack]["oracle"].append(e_stat(v, ecdf) if v else None)
            else:
                results[attack]["oracle"] = [None] * len(slots)

            # blind
            if attack in ("semicolon", "comma", "newline"):
                text_b = restore_blind(text_orig, attack)
                if normalize_text(text_b) == normalize_text(baseline_text):
                    reused["blind"] += 1
                    for si in range(len(slots)):
                        v = by_key.get((doc["document_id"], "baseline", si))
                        results[attack]["blind"].append(e_stat(v, ecdf) if v else None)
                    continue
            elif attack == "space":
                text_b = restore_blind(text_orig, attack)
            else:
                results[attack]["blind"] = [None] * len(slots)
                continue

            reused["computed"] += 1
            spans = span_map(text_b, slots)
            cands = select_budget(candidate_windows(text_b, WINDOW)["candidates"], 0)
            texts = list(dict.fromkeys(normalize_text(c["text"]) for c in cands))
            extractor.extract_many(texts)
            for si in range(len(slots)):
                sp = spans[si] if si < len(spans) else None
                if sp is None:
                    results[attack]["blind"].append(None)
                    continue
                d2 = [None] * len(bank)
                for c in cands:
                    if not (sp[0] < c["end_char"] and c["start_char"] < sp[1]):
                        continue
                    rec = extractor.extract(normalize_text(c["text"]))
                    if rec is None or not rec.graph_valid:
                        continue
                    for sub in event_subgraphs(rec.amr):
                        d2 = elementwise_max(d2, scorer.scores(sub))
                results[attack]["blind"].append(e_stat(d2, ecdf))
        print(f"  {doc['document_id']} done", flush=True)

    def rate(vals, q):
        vals = [v for v in vals if v is not None]
        hits = sum(1 for v in vals if v >= taus[q])
        return hits / total_slots if total_slots else None, hits, len(vals)

    summary = {"note": "P1 标点恢复基线（30 篇 / 471 槽）；分母固定为总槽数",
               "slot_total": total_slots,
               "thresholds": {f"q{q}": round(taus[q], 4) for q in taus},
               "reused_blind_docs": reused,
               "rows": {}}
    print("\n" + "=" * 92)
    print(f"{'attack':<11}{'none@1%':>10}{'blind@1%':>10}{'oracle@1%':>11}"
          f"{'none@5%':>10}{'blind@5%':>10}{'oracle@5%':>11}{'可定位':>9}")
    for attack in ATTACKS_ORDER:
        cells = {}
        for kind in ("none", "blind", "oracle"):
            r1, h1, loc = rate(results[attack][kind], 0.01)
            r5, _, _ = rate(results[attack][kind], 0.05)
            cells[kind] = {"r1": r1, "r5": r5, "hits": h1, "locatable": loc}
        summary["rows"][attack] = cells
        loc = cells["blind"]["locatable"]
        print(f"{attack:<11}{cells['none']['r1']:>10.1%}{cells['blind']['r1']:>10.1%}"
              f"{cells['oracle']['r1']:>11.1%}{cells['none']['r5']:>10.1%}"
              f"{cells['blind']['r5']:>10.1%}{cells['oracle']['r5']:>11.1%}"
              f"{loc:>7}/{total_slots}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nS2Match calls: {scorer.calls}")
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
