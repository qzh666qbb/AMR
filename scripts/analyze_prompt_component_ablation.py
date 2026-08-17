"""Analyze the equal-budget prompt-component ablation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score


ESCAPE_THRESHOLD = 2.33


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    exp = root / "experiments" / "prompt_component_ablation30"
    payload = json.loads((exp / "candidates.json").read_text(encoding="utf-8"))
    rows = payload["rows"]
    validation_rows = json.loads((exp / "validation.json").read_text(encoding="utf-8"))["rows"]
    validations = {(int(row["index"]), row["strategy"]): row for row in validation_rows}
    parsed = json.loads((exp / "dataset" / "parsed.json").read_text(encoding="utf-8"))
    z_scores = np.load(exp / "detection" / "machine_z_scores.npy")
    human_z = np.load(exp / "detection" / "human_z_scores.npy")
    baseline_z = np.load(
        root
        / "baselines"
        / "local"
        / "swan_acl2026_realnews"
        / "runs"
        / "formal_250x5"
        / "detection"
        / "machine_z_scores.npy"
    )
    if not (len(rows) == len(parsed) == len(z_scores)):
        raise RuntimeError("Row, parsed AMR, and z-score lengths differ")

    analyzed = []
    for row, doc_amrs, attack_z in zip(rows, parsed, z_scores):
        validation = validations[(int(row["index"]), row["strategy"])]
        analyzed.append(
            {
                "index": int(row["index"]),
                "strategy": row["strategy"],
                "baseline_z": float(baseline_z[int(row["index"])]),
                "attack_z": float(attack_z),
                "z_drop": float(baseline_z[int(row["index"])] - attack_z),
                "escaped": float(attack_z) <= ESCAPE_THRESHOLD,
                "sentence_delta": int(validation["sentence_delta"]),
                "numbers_preserved": bool(validation["numbers_preserved"]),
                "negation_preserved": bool(validation["negation_preserved"]),
                "amr_nonempty_rate": float(np.mean([bool(amr.strip()) for amr in doc_amrs])),
            }
        )

    strategies = payload["protocol"]["strategies"]
    grouped = {strategy: [row for row in analyzed if row["strategy"] == strategy] for strategy in strategies}
    summary = {}
    for strategy, selected in grouped.items():
        machine = np.asarray([row["attack_z"] for row in selected])
        summary[strategy] = {
            "n": len(selected),
            "mean_attack_z": float(machine.mean()),
            "mean_z_drop": float(np.mean([row["z_drop"] for row in selected])),
            "escape_rate": float(np.mean([row["escaped"] for row in selected])),
            "auroc": float(
                roc_auc_score(
                    np.r_[np.zeros(len(human_z)), np.ones(len(machine))], np.r_[human_z, machine]
                )
            ),
            "mean_sentence_delta": float(np.mean([row["sentence_delta"] for row in selected])),
            "fewer_sentence_rate": float(np.mean([row["sentence_delta"] < 0 for row in selected])),
            "numbers_preserved": float(np.mean([row["numbers_preserved"] for row in selected])),
            "negation_preserved": float(np.mean([row["negation_preserved"] for row in selected])),
            "amr_nonempty_rate": float(np.mean([row["amr_nonempty_rate"] for row in selected])),
        }

    actual_groups = {
        "fewer": [row for row in analyzed if row["sentence_delta"] < 0],
        "same": [row for row in analyzed if row["sentence_delta"] == 0],
        "more": [row for row in analyzed if row["sentence_delta"] > 0],
    }
    actual_summary = {
        name: {
            "n": len(selected),
            "mean_z_drop": float(np.mean([row["z_drop"] for row in selected])) if selected else None,
            "escape_rate": float(np.mean([row["escaped"] for row in selected])) if selected else None,
        }
        for name, selected in actual_groups.items()
    }

    by_strategy_index = {
        strategy: {row["index"]: row["attack_z"] for row in selected}
        for strategy, selected in grouped.items()
    }
    comparisons = {}
    for left in strategies:
        for right in strategies:
            if strategies.index(left) >= strategies.index(right):
                continue
            common = sorted(set(by_strategy_index[left]) & set(by_strategy_index[right]))
            left_z = np.asarray([by_strategy_index[left][index] for index in common])
            right_z = np.asarray([by_strategy_index[right][index] for index in common])
            statistic, p_value = wilcoxon(left_z, right_z, zero_method="zsplit")
            comparisons[f"{left}_vs_{right}"] = {
                "mean_z_difference_left_minus_right": float((left_z - right_z).mean()),
                "p_value": float(p_value),
                "statistic": float(statistic),
            }

    output = {
        "summary": summary,
        "actual_sentence_change": actual_summary,
        "paired_comparisons": comparisons,
        "rows": analyzed,
    }
    (exp / "analysis.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    display = {
        "weak": "Weak paraphrase",
        "no_boundary": "No boundary change",
        "boundary_only": "Boundary only",
        "full_e2": "Full E2",
    }
    x = np.arange(len(strategies))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(x, [summary[s]["mean_z_drop"] for s in strategies], color=["#94a3b8", "#14b8a6", "#2563eb", "#7c3aed"])
    axes[0].set_xticks(x, [display[s] for s in strategies], rotation=18)
    axes[0].set_ylabel("Mean z-score drop")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, [summary[s]["escape_rate"] for s in strategies], color=["#94a3b8", "#14b8a6", "#2563eb", "#7c3aed"])
    axes[1].set_xticks(x, [display[s] for s in strategies], rotation=18)
    axes[1].set_ylabel("Escape rate")
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(exp / "prompt_component_ablation.png", dpi=200)
    plt.close(fig)

    cn = {
        "weak": "弱同义改写",
        "no_boundary": "无句界谓词/论元改写",
        "boundary_only": "仅句界重组",
        "full_e2": "完整E2联合改写",
    }
    lines = [
        "# 提示组件消融先导实验",
        "",
        "## 结果",
        "",
        "| 条件 | n | 平均攻击z | 平均z降幅 | 逃逸率 | AUROC | 减少句数比例 | 数字保持 | 否定保持 | AMR非空率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in strategies:
        row = summary[strategy]
        lines.append(
            f"| {cn[strategy]} | {row['n']} | {row['mean_attack_z']:.3f} | {row['mean_z_drop']:.3f} | "
            f"{row['escape_rate']:.1%} | {row['auroc']:.3f} | {row['fewer_sentence_rate']:.1%} | "
            f"{row['numbers_preserved']:.1%} | {row['negation_preserved']:.1%} | {row['amr_nonempty_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## 按实际句数变化汇总",
            "",
            "| 实际变化 | n | 平均z降幅 | 逃逸率 |",
            "|---|---:|---:|---:|",
        ]
    )
    actual_cn = {"fewer": "减少句数", "same": "句数不变", "more": "增加句数"}
    for name, row in actual_summary.items():
        if row["n"]:
            lines.append(
                f"| {actual_cn[name]} | {row['n']} | {row['mean_z_drop']:.3f} | {row['escape_rate']:.1%} |"
            )
    lines.extend(
        [
            "",
            "## 配对显著性",
            "",
            "| 比较 | 左减右平均z差 | Wilcoxon p值 |",
            "|---|---:|---:|",
        ]
    )
    for name, row in comparisons.items():
        lines.append(
            f"| {name} | {row['mean_z_difference_left_minus_right']:.3f} | {row['p_value']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 解释限制",
            "",
            "这是30段等调用预算先导实验。部分弱提示和无句界提示未被模型完全遵守，因此组件结论必须结合实际句数变化。"
            "数字、否定和AMR非空只是自动约束，最终Valid ASR仍需盲法人工评价。",
        ]
    )
    (exp / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
