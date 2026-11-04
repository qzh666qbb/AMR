"""配对负例对照：同生成器、同文档位置、不同命题的负例 vs 正例。

输入：
- 正例 FUSION_RECOVERY.json（dataset5）
- 负例 FUSION_RECOVERY.json（dataset5_paired_neg，同一批 doc/slot，经过同样 7 类扰动）

输出：
- 每个攻击下，max_d1 / max_d2 区分正负例的 AUC 与 TPR
- 配对差值（同一 doc/slot/attack 的分数差）及聚类 bootstrap CI
- 负例上“配对模板”的排名，用于检验负例是否同样像模板

判读：若 AUC 接近 0.5，说明当前检测器主要响应“生成文本/模板相似度”，
而不是水印本身；若显著高于 0.5，水印信号在控制生成与领域后仍然存在。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_ed_ablation import auc, tpr  # noqa: E402

POS = ROOT / "experiments/swan_multiview_development1/dataset5_eval/FUSION_RECOVERY.json"
NEG = ROOT / "experiments/swan_multiview_development1/dataset5_paired_neg_eval/FUSION_RECOVERY.json"
OUT = ROOT / "experiments/swan_multiview_development1/dataset5_eval/PAIRED_CONTROL.json"
ATTACKS = ("baseline", "semicolon", "newline", "split", "merge", "comma", "space")


def rank_of(vec, target):
    if not vec or not isinstance(target, int) or target >= len(vec):
        return None
    s = vec[target]
    if s is None:
        return None
    return 1 + sum(1 for v in vec if v is not None and v > s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=Path, default=POS)
    ap.add_argument("--neg", type=Path, default=NEG)
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261022)
    args = ap.parse_args()

    pos = json.loads(args.pos.read_text())["per_instance"]
    neg = json.loads(args.neg.read_text())["per_instance"]
    P = {(r["doc"], r["slot"], r["attack"]): r for r in pos}
    N = {(r["doc"], r["slot"], r["attack"]): r for r in neg}
    keys = sorted(set(P) & set(N))
    print(f"配对：{len(keys)} / 正例 {len(P)} / 负例 {len(N)}")

    rng = np.random.default_rng(args.seed)
    docs = sorted({k[0] for k in keys})
    by_doc = defaultdict(list)
    for k in keys:
        by_doc[k[0]].append(k)

    def paired_diff(k, field):
        a, b = P[k].get(field), N[k].get(field)
        if a is None or b is None:
            return None
        return a - b

    def boot(fn, draws):
        vals = []
        for _ in range(draws):
            sample = []
            for i in rng.integers(0, len(docs), len(docs)):
                sample.extend(by_doc[docs[i]])
            v = fn(sample)
            if v is not None and np.isfinite(v):
                vals.append(v)
        if not vals:
            return [None, None]
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    result = {"paired": len(keys), "per_attack": {}}
    print(f"\n{'attack':10} {'n':>4} {'AUC d1':>7} {'AUC d2':>7} "
          f"{'TPR1 d1':>8} {'TPR1 d2':>8} {'Δmax d1':>8} {'负例模板Top1':>11} {'负例Top5':>8}")
    for a in ATTACKS:
        ks = [k for k in keys if k[2] == a]
        if not ks:
            continue
        p1 = [P[k]["max_d1"] for k in ks if P[k].get("max_d1") is not None]
        n1 = [N[k]["max_d1"] for k in ks if N[k].get("max_d1") is not None]
        p2 = [P[k]["max_d2"] for k in ks if P[k].get("max_d2") is not None]
        n2 = [N[k]["max_d2"] for k in ks if N[k].get("max_d2") is not None]
        diffs = [d for d in (paired_diff(k, "max_d1") for k in ks) if d is not None]
        ranks = [rank_of(N[k].get("vec_d1"), N[k].get("template_index")) for k in ks]
        ranks = [r for r in ranks if r is not None]
        a1, a2 = auc(p1, n1), auc(p2, n2)
        t1, t2 = tpr(p1, n1, 0.01), tpr(p2, n2, 0.01)
        top1 = sum(1 for r in ranks if r == 1) / len(ranks) if ranks else float("nan")
        top5 = sum(1 for r in ranks if r <= 5) / len(ranks) if ranks else float("nan")
        ci = boot(lambda sample: np.mean([paired_diff(k, "max_d1") for k in sample
                                          if paired_diff(k, "max_d1") is not None]), 500)
        result["per_attack"][a] = {
            "n": len(ks), "auc_d1": round(float(a1), 4), "auc_d2": round(float(a2), 4),
            "tpr1_d1": round(float(t1), 4), "tpr1_d2": round(float(t2), 4),
            "mean_diff_max_d1": round(float(np.mean(diffs)), 4) if diffs else None,
            "diff_ci": [None if v is None else round(v, 4) for v in ci],
            "neg_template_top1": round(top1, 4) if ranks else None,
            "neg_template_top5": round(top5, 4) if ranks else None,
        }
        print(f"{a:10} {len(ks):4d} {a1:7.4f} {a2:7.4f} {t1:8.1%} {t2:8.1%} "
              f"{np.mean(diffs):8.4f} {top1:11.1%} {top5:8.1%}")

    # 总体（不分攻击）
    p1 = [P[k]["max_d1"] for k in keys if P[k].get("max_d1") is not None]
    n1 = [N[k]["max_d1"] for k in keys if N[k].get("max_d1") is not None]
    result["overall"] = {
        "n": len(keys), "auc_d1": round(float(auc(p1, n1)), 4),
        "tpr1_d1": round(float(tpr(p1, n1, 0.01)), 4),
        "tpr5_d1": round(float(tpr(p1, n1, 0.05)), 4),
        "mean_diff": round(float(np.mean([P[k]["max_d1"] - N[k]["max_d1"]
                                          for k in keys
                                          if P[k].get("max_d1") is not None
                                          and N[k].get("max_d1") is not None])), 4),
    }
    print(f"\n总体：AUC(d1)={result['overall']['auc_d1']:.4f}  "
          f"TPR@1%={result['overall']['tpr1_d1']:.1%}  "
          f"TPR@5%={result['overall']['tpr5_d1']:.1%}  "
          f"Δmax={result['overall']['mean_diff']:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
