"""Prepare sentence targets and oracle requirements for targeted SWAN attacks."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import sent_tokenize


MATCH_THRESHOLD = 0.65
Z_THRESHOLD = 2.33
LAMBDA = 0.25


def split_sentences(text: str) -> list[str]:
    return [piece.strip() for piece in sent_tokenize(text) if piece.strip()]


def z_score(green: int, total: int) -> float:
    return (green - LAMBDA * total) / math.sqrt(total * LAMBDA * (1 - LAMBDA))


def minimum_flips_to_escape(green: int, total: int) -> int:
    for flips in range(green + 1):
        if z_score(green - flips, total) <= Z_THRESHOLD:
            return flips
    return green


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    baseline = (
        args.root / "baselines" / "local" / "swan_acl2026_realnews" / "runs" / "formal_250x5"
    )
    match_docs = json.loads(
        (baseline / "detection" / "machine_amr_matches.json").read_text(encoding="utf-8")
    )["matched_amrs"]
    baseline_z = np.load(baseline / "detection" / "machine_z_scores.npy")
    candidate_rows = json.loads(
        (
            args.root
            / "experiments"
            / "planE_e2_joint250"
            / "candidates.json"
        ).read_text(encoding="utf-8")
    )["rows"]
    source_by_index = {}
    for row in candidate_rows:
        source_by_index[int(row["index"])] = row["source"]

    rng = np.random.default_rng(args.seed)
    output_rows = []
    mismatches = []
    for index, matches in enumerate(match_docs):
        sentences = split_sentences(source_by_index[index])
        if len(sentences) != len(matches):
            mismatches.append(
                {"index": index, "sentence_count": len(sentences), "match_count": len(matches)}
            )
            sentences = [f"sentence_{position}" for position in range(len(matches))]
        scored = []
        for position, (sentence, match) in enumerate(zip(sentences, matches)):
            score = float(match["score"])
            scored.append(
                {
                    "position": position,
                    "sentence": sentence,
                    "match_score": score,
                    "green": score >= MATCH_THRESHOLD,
                    "margin": score - MATCH_THRESHOLD,
                    "best_matching_amr": match["best_matching_amr"],
                }
            )
        green = sum(item["green"] for item in scored)
        required = minimum_flips_to_escape(green, len(scored))
        green_items = [item for item in scored if item["green"]]
        top_margin = sorted(green_items, key=lambda item: (-item["margin"], item["position"]))
        near_threshold = sorted(green_items, key=lambda item: (item["margin"], item["position"]))
        random_items = list(green_items)
        rng.shuffle(random_items)
        output_rows.append(
            {
                "index": index,
                "source": source_by_index[index],
                "n_sentences": len(scored),
                "green_sentences": green,
                "baseline_z_recorded": float(baseline_z[index]),
                "baseline_z_recomputed": z_score(green, len(scored)),
                "minimum_green_flips_to_escape": required,
                "top_margin_targets": top_margin[:required],
                "near_threshold_targets": near_threshold[:required],
                "random_green_targets": random_items[:required],
                "all_sentences": scored,
            }
        )

    requirements = np.asarray([row["minimum_green_flips_to_escape"] for row in output_rows])
    greens = np.asarray([row["green_sentences"] for row in output_rows])
    summary = {
        "paragraphs": len(output_rows),
        "match_threshold": MATCH_THRESHOLD,
        "z_threshold": Z_THRESHOLD,
        "lambda": LAMBDA,
        "sentence_match_mismatches": mismatches,
        "green_sentence_distribution": {
            str(value): int((greens == value).sum()) for value in sorted(set(greens.tolist()))
        },
        "minimum_flip_distribution": {
            str(value): int((requirements == value).sum())
            for value in sorted(set(requirements.tolist()))
        },
        "mean_required_flips": float(requirements.mean()),
    }

    out_dir = args.root / "experiments" / "targeted_sentence_attack"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "targets.json").write_text(
        json.dumps({"summary": summary, "rows": output_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    labels = sorted(summary["minimum_flip_distribution"], key=int)
    counts = [summary["minimum_flip_distribution"][label] for label in labels]
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.bar(labels, counts, color="#3b82f6")
    axis.set_xlabel("Minimum green-sentence flips required")
    axis.set_ylabel("Paragraphs")
    axis.set_title("Oracle sentence budget to cross z=2.33")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "minimum_flip_distribution.png", dpi=200)
    plt.close(fig)

    lines = [
        "# 高贡献句子靶向攻击：目标准备报告",
        "",
        "## 方法",
        "",
        "SWAN先将每句最大S2MATCH分数按0.65阈值二值化，再以绿色句数量计算段落z-score。"
        "因此本报告计算每段最少需要将多少个绿色句翻转为非绿色，才能使z≤2.33。",
        "",
        "## 结果",
        "",
        f"- 段落数：{summary['paragraphs']}；",
        f"- 句子与匹配记录数量不一致：{len(mismatches)}段；",
        f"- 平均最少翻转句数：{summary['mean_required_flips']:.3f}；",
        "- 最少翻转句数分布："
        + "，".join(f"{key}句={value}段" for key, value in summary["minimum_flip_distribution"].items())
        + "。",
        "",
        "## 后续对照",
        "",
        "每段准备三套等预算目标：最高匹配边际句、最接近0.65阈值句、随机绿色句。"
        "正式生成时必须使用相同提示词、候选数和API参数，并对完整段落重新解析检测。",
        "",
        "连续S2MATCH最高的句子未必最容易攻击；最接近阈值的绿色句可能只需较小结构变化即可翻转。"
        "因此最高边际与近阈值策略均应保留，不能预设Top-score一定最优。",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
