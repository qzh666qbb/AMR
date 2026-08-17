"""Analyze the controlled 30-paragraph targeted-sentence attack pilot."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score


MATCH_THRESHOLD = 0.65
ESCAPE_THRESHOLD = 2.33


def numbers(text: str) -> list[str]:
    return re.findall(r"\b\d+(?:[.,]\d+)*\b", text)


def negation_count(text: str) -> int:
    return len(re.findall(r"\b(?:no|not|never|neither|nor|without|n't)\b", text, flags=re.I))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    exp = root / "experiments" / "targeted_sentence_controlled30"
    payload = json.loads((exp / "candidates.json").read_text(encoding="utf-8"))
    rows = payload["rows"]
    z_scores = np.load(exp / "detection" / "machine_z_scores.npy")
    human_z = np.load(exp / "detection" / "human_z_scores.npy")
    matches = json.loads(
        (exp / "detection" / "machine_amr_matches.json").read_text(encoding="utf-8")
    )["matched_amrs"]
    baseline_z_all = np.load(
        root
        / "baselines"
        / "local"
        / "swan_acl2026_realnews"
        / "runs"
        / "formal_250x5"
        / "detection"
        / "machine_z_scores.npy"
    )
    if not (len(rows) == len(z_scores) == len(matches)):
        raise RuntimeError("Candidate, z-score, and match lengths differ")

    analyzed = []
    for row, attack_z, doc_matches in zip(rows, z_scores, matches):
        positions = [int(value) for value in row["target_positions"]]
        target_scores = [float(doc_matches[position]["score"]) for position in positions]
        flips = sum(value < MATCH_THRESHOLD for value in target_scores)
        analyzed.append(
            {
                "index": int(row["index"]),
                "strategy": row["strategy"],
                "baseline_z": float(baseline_z_all[int(row["index"])]),
                "attack_z": float(attack_z),
                "z_drop": float(baseline_z_all[int(row["index"])] - attack_z),
                "escaped": float(attack_z) <= ESCAPE_THRESHOLD,
                "target_count": len(positions),
                "target_flips": flips,
                "target_flip_rate": flips / len(positions),
                "oracle_flip_requirement_met": flips >= int(row["minimum_green_flips_to_escape"]),
                "numbers_preserved": numbers(row["source"]) == numbers(row["attack"]),
                "negation_preserved": negation_count(row["source"]) == negation_count(row["attack"]),
            }
        )

    by_index: dict[int, dict[str, dict]] = defaultdict(dict)
    for row in analyzed:
        by_index[row["index"]][row["strategy"]] = row
    strategies = payload["protocol"]["strategies"]
    paired_indices = sorted(
        index for index, values in by_index.items() if all(strategy in values for strategy in strategies)
    )
    paired = {
        strategy: [by_index[index][strategy] for index in paired_indices] for strategy in strategies
    }

    summary = {}
    for strategy, strategy_rows in paired.items():
        attack_values = np.asarray([row["attack_z"] for row in strategy_rows])
        labels = np.r_[np.zeros(len(human_z)), np.ones(len(attack_values))]
        scores = np.r_[human_z, attack_values]
        summary[strategy] = {
            "n": len(strategy_rows),
            "mean_attack_z": float(attack_values.mean()),
            "mean_z_drop": float(np.mean([row["z_drop"] for row in strategy_rows])),
            "escape_rate": float(np.mean([row["escaped"] for row in strategy_rows])),
            "target_flip_rate": float(np.mean([row["target_flip_rate"] for row in strategy_rows])),
            "oracle_requirement_rate": float(
                np.mean([row["oracle_flip_requirement_met"] for row in strategy_rows])
            ),
            "numbers_preserved": float(np.mean([row["numbers_preserved"] for row in strategy_rows])),
            "negation_preserved": float(np.mean([row["negation_preserved"] for row in strategy_rows])),
            "auroc": float(roc_auc_score(labels, scores)),
        }

    comparisons = {}
    pairs = [
        ("top_margin", "near_threshold"),
        ("top_margin", "random_green"),
        ("near_threshold", "random_green"),
    ]
    for left, right in pairs:
        left_z = np.asarray([row["attack_z"] for row in paired[left]])
        right_z = np.asarray([row["attack_z"] for row in paired[right]])
        try:
            statistic, p_value = wilcoxon(left_z, right_z, zero_method="zsplit")
        except ValueError:
            statistic, p_value = float("nan"), float("nan")
        comparisons[f"{left}_vs_{right}"] = {
            "mean_z_difference_left_minus_right": float((left_z - right_z).mean()),
            "wilcoxon_statistic": float(statistic),
            "p_value": float(p_value),
        }

    output = {
        "protocol": {
            "paired_paragraphs": len(paired_indices),
            "paired_indices": paired_indices,
            "match_threshold": MATCH_THRESHOLD,
            "escape_threshold": ESCAPE_THRESHOLD,
        },
        "summary": summary,
        "paired_comparisons": comparisons,
        "rows": analyzed,
    }
    (exp / "analysis.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    display = {
        "top_margin": "最高匹配边际句",
        "near_threshold": "近0.65阈值句",
        "random_green": "随机绿色句",
    }
    plot_display = {
        "top_margin": "High margin",
        "near_threshold": "Near threshold",
        "random_green": "Random green",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    x = np.arange(len(strategies))
    axes[0].bar(x, [summary[s]["escape_rate"] for s in strategies], color=["#2563eb", "#7c3aed", "#94a3b8"])
    axes[0].set_xticks(x, [plot_display[s] for s in strategies], rotation=15)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Escape rate")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, [summary[s]["target_flip_rate"] for s in strategies], color=["#2563eb", "#7c3aed", "#94a3b8"])
    axes[1].set_xticks(x, [plot_display[s] for s in strategies], rotation=15)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Target sentence flip rate")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(exp / "targeted_strategy_comparison.png", dpi=200)
    plt.close(fig)

    lines = [
        "# 靶向句子攻击先导实验",
        "",
        "## 协议",
        "",
        f"三种策略在{len(paired_indices)}个完整配对段落上比较。每个策略每段一次独立API调用，"
        "只重写预先指定的句子；非目标句逐字不变，句数保持不变。",
        "",
        "## 结果",
        "",
        "| 策略 | n | 平均攻击z | 平均z降幅 | 逃逸率 | 目标句翻转率 | 达到oracle翻转预算 | AUROC | 数字保持 | 否定保持 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in strategies:
        row = summary[strategy]
        lines.append(
            f"| {display[strategy]} | {row['n']} | {row['mean_attack_z']:.3f} | "
            f"{row['mean_z_drop']:.3f} | {row['escape_rate']:.1%} | "
            f"{row['target_flip_rate']:.1%} | {row['oracle_requirement_rate']:.1%} | "
            f"{row['auroc']:.3f} | {row['numbers_preserved']:.1%} | {row['negation_preserved']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## 配对比较",
            "",
            "| 比较 | 左减右平均z差 | Wilcoxon p值 |",
            "|---|---:|---:|",
        ]
    )
    for name, row in comparisons.items():
        lines.append(
            f"| {name} | {row['mean_z_difference_left_minus_right']:.3f} | {row['p_value']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 解释限制",
            "",
            "本实验严格控制了句数和非目标句，因此能够检验句子选择策略本身，但仍是29段先导实验。"
            "数字与否定检查是表面约束，最终Valid ASR需要按冻结人工标准盲评。",
        ]
    )
    (exp / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
