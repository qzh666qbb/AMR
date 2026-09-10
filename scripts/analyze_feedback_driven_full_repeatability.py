"""Repeat S2Match detection on all 120 candidates from the strict audit run.

This script measures how often each candidate and each paragraph stays below the
escape threshold when S2Match is recomputed repeatedly on the fixed parsed AMRs.
It is intended to support a stability curve / robustness figure for the paper.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt


THRESHOLD = 2.33
S2_THRESHOLD = 0.65
LMBDA = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("experiments/candidate_budget16_pilot30/feedback_driven_strict30"),
    )
    return parser.parse_args()


def score_doc(amrs, bank, compute_s2match_score):
    count_green = 0
    for amr in amrs:
        if not amr.strip():
            continue
        best = 0.0
        for bank_amr in bank:
            score = compute_s2match_score(amr, bank_amr)
            if score > best:
                best = score
        if best >= S2_THRESHOLD:
            count_green += 1
    n = len(amrs)
    return (count_green - LMBDA * n) / math.sqrt(n * LMBDA * (1 - LMBDA))


def main() -> None:
    args = parse_args()
    root = args.root
    experiment = root / args.experiment

    os.environ.setdefault(
        "GLOVE_VECTORS_PATH",
        str(root / "third_party" / "SWAN" / "vectors" / "glove.6B.100d.txt"),
    )

    import sys

    sys.path.insert(0, str(root / "third_party" / "SWAN"))
    from utils.amr_utils import compute_s2match_score, load_amr_bank

    audit = json.loads((experiment / "assistant_audit.json").read_text(encoding="utf-8"))
    rows = audit["rows"]
    parsed = json.loads((experiment / "dataset" / "parsed_amrs.json").read_text(encoding="utf-8"))
    bank = load_amr_bank(str(root / "third_party" / "SWAN" / "amr_bank" / "banks" / "amr_bank_50.json"))

    paragraph_to_positions = collections.defaultdict(list)
    for row_pos, row in enumerate(rows):
        paragraph_to_positions[row["index"]].append(row_pos)

    candidate_scores = [[] for _ in rows]
    paragraph_success = collections.Counter()

    for _ in range(args.repeats):
        scores = []
        for row_pos, row in enumerate(rows):
            z = score_doc(parsed[row_pos], bank, compute_s2match_score)
            scores.append(z)
            candidate_scores[row_pos].append(z)

        for paragraph_idx, positions in paragraph_to_positions.items():
            if any(scores[pos] <= THRESHOLD for pos in positions):
                paragraph_success[paragraph_idx] += 1

    candidate_stats = []
    for row_pos, row in enumerate(rows):
        values = candidate_scores[row_pos]
        counts = collections.Counter(values)
        candidate_stats.append(
            {
                "row_pos": row_pos,
                "index": row["index"],
                "candidate_index": row["candidate_index"],
                "strategy": row["strategy"],
                "base_z": row["z_score"],
                "text_changed": row.get("text_changed"),
                "escaped_once": row.get("escaped"),
                "unique_scores": {str(k): v for k, v in sorted(counts.items())},
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "escape_rate": sum(v <= THRESHOLD for v in values) / len(values),
                "flip_rate": int(0 < sum(v <= THRESHOLD for v in values) < len(values)),
            }
        )

    stable_candidates = sum(1 for item in candidate_stats if item["flip_rate"] == 0)
    unstable_candidates = len(candidate_stats) - stable_candidates
    candidates_with_any_escape = sum(1 for item in candidate_stats if item["escape_rate"] > 0)
    candidates_with_full_escape = sum(1 for item in candidate_stats if item["escape_rate"] == 1)
    candidates_with_mixed = sum(1 for item in candidate_stats if 0 < item["escape_rate"] < 1)

    summary = {
        "repeats": args.repeats,
        "total_candidates": len(rows),
        "total_paragraphs": len(paragraph_to_positions),
        "mean_candidate_escape_rate": sum(item["escape_rate"] for item in candidate_stats) / len(candidate_stats),
        "candidate_with_any_escape": candidates_with_any_escape,
        "candidate_with_full_escape": candidates_with_full_escape,
        "candidate_with_mixed_escape": candidates_with_mixed,
        "stable_candidates": stable_candidates,
        "unstable_candidates": unstable_candidates,
        "paragraph_success_freq": {str(k): v for k, v in sorted(paragraph_success.items())},
        "mean_paragraph_asr": sum(paragraph_success.values()) / (args.repeats * len(paragraph_to_positions)),
        "candidate_stats": candidate_stats,
    }

    out_json = experiment / f"full_repeatability_{args.repeats}.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    out_md = experiment / f"full_repeatability_{args.repeats}.md"
    top_flip = sorted(candidate_stats, key=lambda item: (item["flip_rate"], -item["escape_rate"]), reverse=True)[:10]
    lines = [
        "# 全量候选重复性审计",
        "",
        f"对全部 {len(rows)} 个候选做了 {args.repeats} 次独立重复 S2Match 计算。",
        "",
        f"- 候选总数: {len(rows)}",
        f"- 段落总数: {len(paragraph_to_positions)}",
        f"- 平均候选逃逸率: {summary['mean_candidate_escape_rate']:.3f}",
        f"- 全部段落平均ASR: {summary['mean_paragraph_asr']:.3f}",
        f"- 候选中有任意逃逸的数量: {candidates_with_any_escape}",
        f"- 候选中全程逃逸的数量: {candidates_with_full_escape}",
        f"- 候选中出现混合翻转的数量: {candidates_with_mixed}",
        "",
        "段落成功频率:",
        ", ".join(f"{k}={v}/{args.repeats}" for k, v in sorted(paragraph_success.items())),
        "",
        "最不稳定的候选（按是否翻转排序）:",
    ]
    for item in top_flip:
        lines.append(
            f"- paragraph {item['index']} / candidate {item['candidate_index']} / {item['strategy']} "
            f": escape_rate={item['escape_rate']:.3f}, base_z={item['base_z']:.3f}, unique={item['unique_scores']}"
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")

    # Plot: candidate escape rate against base z.
    xs = [item["base_z"] for item in candidate_stats]
    ys = [item["escape_rate"] for item in candidate_stats]
    colors = ["tab:green" if item["escape_rate"] == 1 else "tab:orange" if item["escape_rate"] > 0 else "tab:blue" for item in candidate_stats]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, c=colors, alpha=0.85, s=32, edgecolors="none")
    ax.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1, label="escape threshold")
    ax.set_xlabel("Base z score")
    ax.set_ylabel("Repeat escape rate")
    ax.set_title("Full repeatability across 120 candidates")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(experiment / f"full_repeatability_{args.repeats}.png", dpi=200)
    plt.close(fig)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
