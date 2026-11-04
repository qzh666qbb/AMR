"""融合检测器的 TPR@FPR 对比（逐个检测器在各自零假设分布上校准阈值）。

为什么必须这样校准：D1/D2/D3 的分数是 S2Match 尺度，D4/D5 的分数是 RRF 尺度。
把同一个阈值（如 0.65）套在所有检测器上是**无效**的——RRF 分数上限约 0.18，
永远不可能达到 0.65。正确做法是对每个检测器分别在其零假设分布上取分位。

口径（两侧对称）：
  负例侧 green = max_over_bank >= τ        （虚假命中）
  正例侧 green = max_over_bank >= τ        （与负例同口径）
  另附目标条件化变体：正例侧 green = score(真实注入模板) >= τ（更保守）
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NULL = ROOT / "experiments/swan_multiview_development1/null_fp_audit1"
POS = ROOT / "experiments/swan_multiview_development1/dataset4_fusion1"
DETECTORS = ("d1", "d2", "d3", "d4", "d5")
LABELS = {"d1": "D1 整图", "d2": "D2 局部", "d3": "D3 朴max",
          "d4": "D4 秩融合", "d5": "D5 门控"}
TARGETS = (0.01, 0.02, 0.05, 0.10)


def tau_at(scores: list, q: float) -> float:
    """降序排列后取上侧分位，使超过阈值的比例约为 q。"""
    s = sorted(scores, reverse=True)
    k = max(1, math.ceil(q * len(s)))
    return s[min(k, len(s)) - 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", type=float, default=list(TARGETS))
    args = ap.parse_args()

    null_units = json.loads((NULL / "PARTIAL.json").read_text())["per_unit"]
    pos_rows = json.loads((POS / "FUSION_RECOVERY.json").read_text())["per_instance"]
    print(f"负例单元 {len(null_units)}，正例载体 {len(pos_rows)}")
    print()

    null_scores = {d: [u[f"top_{d}"] for u in null_units if u.get(f"top_{d}") is not None]
                   for d in DETECTORS}
    pos_max = {d: [r[f"max_{d}"] for r in pos_rows if r.get(f"max_{d}") is not None]
               for d in DETECTORS}
    pos_true = {d: [r[f"score_{d}_true"] for r in pos_rows
                    if r.get(f"score_{d}_true") is not None] for d in DETECTORS}

    def table_payload():
        return {
            "note": ("每个检测器在各自零假设分位上校准阈值；跨检测器不可用同一阈值。"
                     "max_over_bank 口径两侧对称；target 口径正例侧更保守。"),
            "null_units": len(null_units), "positive_carriers": len(pos_rows),
            "results": {},
        }

    payload = table_payload()

    for q in args.targets:
        print("=" * 100)
        print(f"目标空文虚假率 q = {q:.0%}")
        print("=" * 100)
        print(f"{'检测器':<10} {'τ':>9} {'空文实测':>9} {'正例(max)':>11} {'正例(target)':>13}")
        print("-" * 100)
        entry = {}
        for d in DETECTORS:
            ns = null_scores[d]
            if not ns:
                continue
            tau = tau_at(ns, q)
            null_rate = sum(1 for x in ns if x >= tau) / len(ns)
            rec_max = (sum(1 for x in pos_max[d] if x >= tau) / len(pos_max[d])
                       if pos_max[d] else None)
            rec_true = (sum(1 for x in pos_true[d] if x >= tau) / len(pos_true[d])
                        if pos_true[d] else None)
            entry[d] = {
                "tau": round(tau, 4), "null_rate": round(null_rate, 4),
                "positive_recovery_max": round(rec_max, 4) if rec_max is not None else None,
                "positive_recovery_target": round(rec_true, 4) if rec_true is not None else None,
            }
            print(f"{LABELS[d]:<10} {tau:>9.4f} {null_rate:>9.2%} "
                  f"{rec_max:>11.2%} {rec_true:>13.2%}")
        payload["results"][f"q_{q}"] = entry
        print()

    # 逐攻击明细（正例侧，max 口径），只报 D1 vs D5
    print("=" * 100)
    print("逐攻击明细（正例 max_over_bank 口径，D1 vs D5）")
    print("=" * 100)
    by_attack = {}
    for r in pos_rows:
        by_attack.setdefault(r["attack"], []).append(r)
    print(f"{'attack':<11} {'n':>4} | " + " | ".join(
        f"{LABELS[d]} @q={q:.0%}" for q in args.targets[:2] for d in ("d1", "d5")))
    print("-" * 100)
    detail = {}
    taus = {q: {d: tau_at(null_scores[d], q) for d in DETECTORS} for q in args.targets}
    for attack in sorted(by_attack):
        rows = by_attack[attack]
        cells = f"{attack:<11} {len(rows):>4} | "
        for q in args.targets[:2]:
            for d in ("d1", "d5"):
                rate = sum(1 for r in rows if r["max_" + d] >= taus[q][d]) / len(rows)
                cells += f"{rate:>12.1%} | "
        print(cells)
        detail[attack] = {
            "n": len(rows),
            **{f"q{q}_{d}": round(sum(1 for r in rows if r["max_" + d] >= taus[q][d]) / len(rows), 4)
               for q in args.targets[:2] for d in ("d1", "d5")},
        }
    payload["per_attack_max"] = detail

    (NULL / "TPR_FPR_TABLE.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nwritten: {NULL / 'TPR_FPR_TABLE.json'}")


if __name__ == "__main__":
    main()
