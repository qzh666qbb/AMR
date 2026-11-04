"""口径消融：评分器 × 恢复判据 的 2×2 分析。

目的：分离两个被混在一起的变量。
  - 评分器 (scorer)：light multiset-F1  vs  官方 S2Match
  - 判据 (criterion)：
      argmax_identity   —— 该候选的 top-1 模板是否恰等于注入模板（原口径）
      target_conditioned —— 该候选对"注入模板自身"的分数是否 >= 0.75

只用缓存里的 AMR，不调用 parser、不调用 API。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party/swan_s2match_runtime_711a231d"))

from src.core.amr_extractor import AmrlibBackend  # noqa: E402
from src.core.bank_scorer import _multiset_f1, template_features  # noqa: E402
from src.core.multiview_candidates import WindowConfig, candidate_windows, select_budget  # noqa: E402
from src.core.sqlite_cache import AMRCache, normalize_text  # noqa: E402

from probe_swan_multitemplate_yield import normalize_to_template, strip_snt  # noqa: E402
import amr_py3  # noqa: E402
import s2match as s2match_mod  # noqa: E402
from s2match import compute_s2match_from_strings  # noqa: E402

if not hasattr(amr_py3, "log_helper"):
    class _SilentLog:
        def debug(self, *a): pass
        def error(self, *a): pass
    amr_py3.log_helper = _SilentLog()

ROOT_DIR = ROOT
GLOVE = ROOT / "vectors/glove.6B.100d.txt"
_S2MATCH_VECS = s2match_mod.load_vecs(str(GLOVE))
s2match_mod.load_vecs = lambda fp: _S2MATCH_VECS

BANK = ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_100.json"
DATASET4 = ROOT / "experiments/swan_multiview_development1/dataset4/documents.json"
DEFAULT_OUT = ROOT / "experiments/swan_multiview_development1/dataset4_attack1"
WINDOW = WindowConfig(lengths=(12,), stride=6, max_words=24, max_candidates=128)
CACHE = Path("/tmp/amr_multiview_20261020/dataset4_test.db")
THRESHOLD = 0.75
WEIGHTS = (0.25, 0.50, 0.25)

# 复用 run_dataset4_attack_recovery 的攻击定义
sys.path.insert(0, str(ROOT / "scripts"))
from run_dataset4_attack_recovery import ATTACKS, _annotate_merge_spans  # noqa: E402


def overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def slot_span_for(attack_name, spans, si):
    if attack_name == "merge":
        for sp in spans:
            if sp.get("_slot_a_idx") == si:
                return sp["slot_a"]
            if sp.get("_slot_b_idx") == si:
                return sp["slot_b"]
        return None
    return spans[si] if si < len(spans) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    docs = json.loads(DATASET4.read_text())["documents"]
    bank = json.loads(BANK.read_text())
    bank_amr = [row["normalized_template_amr"] for row in bank]

    backend = AmrlibBackend(str(ROOT / "models/model_parse_xfm_bart_large-v0_1_0"),
                            str(ROOT / "models/bart-large-tokenizer"), device=args.device)
    fp = backend.fingerprint
    cache = AMRCache(str(CACHE))

    # 预处理 bank 的 light 特征
    bank_features = [template_features(t) for t in bank_amr]

    def light_scores(amr):
        f = template_features(amr)
        out = []
        for ref in bank_features:
            c = _multiset_f1(f.concepts, ref.concepts)
            r = _multiset_f1(f.relations, ref.relations)
            p = _multiset_f1(f.paths, ref.paths)
            out.append(sum(w * v for w, v in zip(WEIGHTS, (c, r, p))))
        return out

    s2_cache = {}

    def s2_amr_vs(amr, tpl_idx):
        key = (amr, tpl_idx)
        if key not in s2_cache:
            try:
                norm = normalize_to_template(strip_snt(amr))
                _, _, sc = compute_s2match_from_strings(
                    norm, bank_amr[tpl_idx], vectors_file=str(GLOVE))
                s2_cache[key] = float(sc)
            except Exception:
                s2_cache[key] = None
            if len(s2_cache) % 2000 == 0:
                print(f"  [s2match calls: {len(s2_cache)}]", flush=True)
        return s2_cache[key]

    light_cache = {}

    def light_all(amr):
        if amr not in light_cache:
            light_cache[amr] = light_scores(amr)
        return light_cache[amr]

    # 统计（单位 = carrier 实例：doc × attack × slot）
    KEYS = ("light_argmax", "light_target", "s2_argmax", "s2_target")
    counts = {k: 0 for k in KEYS}
    totals = {k: 0 for k in KEYS}
    missing_amr = 0
    carrier_instances = 0
    # 诊断：argmax-identity 命中时，该候选的最高分究竟有多高？
    diag_light = []   # (argmax_hit, target_hit, 候选最高分, 注入模板自身分数)

    for doc in docs:
        slots = doc["slots"]
        for attack_name, fn in ATTACKS.items():
            if attack_name == "merge":
                text, spans = fn(slots)
                _annotate_merge_spans(spans)
            else:
                text, spans = fn(slots)
            candidates = select_budget(candidate_windows(text, WINDOW)["candidates"], 0)

            for si, slot in enumerate(slots):
                tpl = slot["template_index"]
                span = slot_span_for(attack_name, spans, si)
                if span is None:
                    continue
                hit = {k: False for k in KEYS}
                found_any = False
                light_best = 0.0
                light_target_best = 0.0
                for cand in candidates:
                    if not overlap(span, (cand["start_char"], cand["end_char"])):
                        continue
                    rec = cache.get(normalize_text(cand["text"]), fp)
                    if rec is None:
                        missing_amr += 1
                        continue
                    found_any = True
                    # light 口径
                    ls = light_all(rec.amr)
                    light_best = max(light_best, max(ls))
                    light_target_best = max(light_target_best, ls[tpl])
                    if int(max(range(len(ls)), key=lambda i: (ls[i], -i))) == tpl:
                        hit["light_argmax"] = True
                    if ls[tpl] >= THRESHOLD:
                        hit["light_target"] = True
                    # s2match 口径（每个候选做全库 argmax）
                    vals = [s2_amr_vs(rec.amr, t) for t in range(len(bank_amr))]
                    valid = [(v, t) for t, v in enumerate(vals) if v is not None]
                    if valid:
                        best_t = max(valid, key=lambda x: (x[0], -x[1]))[1]
                        if best_t == tpl:
                            hit["s2_argmax"] = True
                    st = s2_amr_vs(rec.amr, tpl)
                    if st is not None and st >= THRESHOLD:
                        hit["s2_target"] = True
                if not found_any:
                    continue
                carrier_instances += 1
                for k in KEYS:
                    totals[k] += 1
                    counts[k] += int(hit[k])
                diag_light.append({
                    "attack": attack_name, "doc": doc["document_id"], "slot": si,
                    "argmax_hit": hit["light_argmax"], "target_hit": hit["light_target"],
                    "light_best": round(light_best, 4),
                    "light_target": round(light_target_best, 4),
                })

    print("=" * 78)
    print("2x2 口径消融：评分器 × 恢复判据（统计单位 = carrier 实例）")
    print("=" * 78)
    print(f"{'组合':<34} {'命中':>6} {'总数':>6} {'比例':>8}")
    print("-" * 78)
    rows = [
        ("light  scorer + argmax-identity", "light_argmax"),
        ("light  scorer + target-conditioned", "light_target"),
        ("S2Match       + argmax-identity", "s2_argmax"),
        ("S2Match       + target-conditioned", "s2_target"),
    ]
    for label, key in rows:
        n, d = counts[key], totals[key]
        print(f"{label:<34} {n:>6} {d:>6} {n/d:>8.1%}" if d else f"{label:<34} {n:>6} {d:>6} {'n/a':>8}")
    print(f"\ncarrier 实例总数: {carrier_instances}")
    print()
    if missing_amr:
        print(f"⚠ 缓存缺失候选数：{missing_amr}（未计入）")

    # 诊断：argmax-identity 命中但 target 未命中时，最高分有多低？
    am_only = [d for d in diag_light if d["argmax_hit"] and not d["target_hit"]]
    both = [d for d in diag_light if d["argmax_hit"] and d["target_hit"]]
    print("─" * 78)
    print("诊断：argmax-identity 判据在 light 评分器下的行为")
    print("─" * 78)
    print(f"  两者都命中            : {len(both)}")
    print(f"  仅 argmax 命中(目标分<阈值): {len(am_only)}")
    if am_only:
        bests = sorted(d["light_best"] for d in am_only)
        targs = sorted(d["light_target"] for d in am_only)
        mid = len(bests) // 2
        print(f"    这些实例的候选最高分：中位 {bests[mid]:.3f}，"
              f"范围 [{bests[0]:.3f}, {bests[-1]:.3f}]")
        print(f"    注入模板自身分数    ：中位 {targs[mid]:.3f}，"
              f"范围 [{targs[0]:.3f}, {targs[-1]:.3f}]")
        low = sum(1 for d in am_only if d["light_best"] < THRESHOLD)
        print(f"    候选最高分 < {THRESHOLD} 的比例: {low}/{len(am_only)} "
              f"({low/len(am_only):.1%})")

    out = {
        "note": "口径消融，用于分离评分器与判据两个变量；不含负例，非FPR/TPR",
        "unit": "carrier instance (doc x attack x slot)",
        "criterion": {
            "argmax_identity": "该候选全库 top-1 模板 == 注入模板",
            "target_conditioned": "该候选对注入模板自身分数 >= 阈值",
        },
        "threshold": THRESHOLD,
        "carrier_instances": carrier_instances,
        "counts": counts, "totals": totals,
        "missing_amr": missing_amr,
        "s2match_values_cached": len(s2_cache),
        "diag_light": diag_light,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "METRIC_CRITERION_ABLATION.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))
    print(f"written: {args.output / 'METRIC_CRITERION_ABLATION.json'}")


if __name__ == "__main__":
    main()
