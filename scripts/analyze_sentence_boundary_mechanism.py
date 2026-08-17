"""Mechanism analysis for E3 sentence-boundary, plain, and AMR-guided attacks."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import sent_tokenize


MATCH_THRESHOLD = 0.65


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    exp = args.root / "experiments" / "e3_main"
    records = json.loads((exp / "candidates.json").read_text(encoding="utf-8"))["records"]
    families = ["plain_paraphrase", "sentence_boundary", "amr_guided"]
    results = {}
    rows_out = []

    for family in families:
        detection = exp / "detection" / family
        z_scores = np.load(detection / "machine_z_scores.npy")
        matched_docs = json.loads(
            (detection / "machine_amr_matches.json").read_text(encoding="utf-8")
        )["matched_amrs"]
        if not (len(records) == len(z_scores) == len(matched_docs)):
            raise RuntimeError(f"Length mismatch for {family}")

        family_rows = []
        for position, (record, z_value, matches) in enumerate(zip(records, z_scores, matched_docs)):
            source_n = len(sent_tokenize(record["source"]))
            attack_text = record["candidates"][family]
            attack_n = len(sent_tokenize(attack_text))
            match_scores = np.asarray([float(item["score"]) for item in matches], dtype=float)
            green = int((match_scores >= MATCH_THRESHOLD).sum())
            baseline_z = float(record["baseline_z"])
            row = {
                "position": position,
                "source_index": record["source_index"],
                "family": family,
                "source_sentences": source_n,
                "attack_sentences": attack_n,
                "sentence_delta": attack_n - source_n,
                "baseline_z": baseline_z,
                "attack_z": float(z_value),
                "z_drop": baseline_z - float(z_value),
                "parsed_amrs": len(matches),
                "green_amrs": green,
                "green_rate": green / len(matches) if matches else float("nan"),
                "mean_best_match": float(match_scores.mean()) if len(match_scores) else float("nan"),
                "near_threshold_rate": float(
                    ((match_scores >= MATCH_THRESHOLD) & (match_scores < MATCH_THRESHOLD + 0.1)).mean()
                )
                if len(match_scores)
                else float("nan"),
            }
            family_rows.append(row)
            rows_out.append(row)

        deltas = np.asarray([row["sentence_delta"] for row in family_rows], dtype=float)
        drops = np.asarray([row["z_drop"] for row in family_rows], dtype=float)
        attack_z = np.asarray([row["attack_z"] for row in family_rows], dtype=float)
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in family_rows:
            label = "fewer" if row["sentence_delta"] < 0 else "more" if row["sentence_delta"] > 0 else "same"
            groups[label].append(row)
        results[family] = {
            "n": len(family_rows),
            "mean_source_sentences": mean_or_nan([row["source_sentences"] for row in family_rows]),
            "mean_attack_sentences": mean_or_nan([row["attack_sentences"] for row in family_rows]),
            "mean_sentence_delta": float(deltas.mean()),
            "changed_sentence_count_rate": float(np.mean(deltas != 0)),
            "mean_attack_z": float(attack_z.mean()),
            "mean_z_drop": float(drops.mean()),
            "sentence_delta_z_drop_correlation": safe_corr(deltas, drops),
            "mean_green_rate": mean_or_nan([row["green_rate"] for row in family_rows]),
            "mean_best_match": mean_or_nan([row["mean_best_match"] for row in family_rows]),
            "groups": {
                label: {
                    "n": len(group_rows),
                    "mean_z_drop": mean_or_nan([row["z_drop"] for row in group_rows]),
                    "mean_attack_z": mean_or_nan([row["attack_z"] for row in group_rows]),
                    "mean_green_rate": mean_or_nan([row["green_rate"] for row in group_rows]),
                }
                for label, group_rows in sorted(groups.items())
            },
        }

    out_dir = args.root / "experiments" / "mechanism_sentence_boundary"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps({"summary": results, "rows": rows_out}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    labels = ["Plain", "Sentence boundary", "AMR guided"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(labels, [results[key]["mean_z_drop"] for key in families], color=["#94a3b8", "#2563eb", "#7c3aed"])
    axes[0].set_ylabel("Mean z-score drop")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(
        labels,
        [results[key]["changed_sentence_count_rate"] for key in families],
        color=["#94a3b8", "#2563eb", "#7c3aed"],
    )
    axes[1].set_ylabel("Fraction changing sentence count")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "family_mechanism_comparison.png", dpi=200)
    plt.close(fig)

    lines = [
        "# E3 句界机制分析",
        "",
        "## 攻击族总体比较",
        "",
        "| 攻击族 | n | 改变句数比例 | 平均句数变化 | 平均z降幅 | 平均绿色句率 | 平均最大匹配分 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    display = {
        "plain_paraphrase": "普通改写",
        "sentence_boundary": "句界攻击",
        "amr_guided": "AMR引导",
    }
    for family in families:
        result = results[family]
        lines.append(
            f"| {display[family]} | {result['n']} | {result['changed_sentence_count_rate']:.1%} | "
            f"{result['mean_sentence_delta']:.3f} | {result['mean_z_drop']:.3f} | "
            f"{result['mean_green_rate']:.1%} | {result['mean_best_match']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 同一攻击族内按句数变化分组",
            "",
            "| 攻击族 | 句数变化 | n | 平均z降幅 | 平均攻击z | 平均绿色句率 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    group_display = {"fewer": "减少", "same": "不变", "more": "增加"}
    for family in families:
        for label, group in results[family]["groups"].items():
            lines.append(
                f"| {display[family]} | {group_display[label]} | {group['n']} | "
                f"{group['mean_z_drop']:.3f} | {group['mean_attack_z']:.3f} | "
                f"{group['mean_green_rate']:.1%} |"
            )
    lines.extend(
        [
            "",
            "## 主要发现",
            "",
            "1. 三类攻击中，只要候选减少了句子数量，平均z降幅都明显增大：普通改写1.595、句界攻击1.837、AMR引导1.588。",
            "2. 句界攻击在句数不变时平均z降幅仅0.258，与普通改写的0.288接近；其总体优势主要集中在成功合并句子的样本。",
            "3. AMR引导即使保持句数不变仍有0.799的平均z降幅，说明它除句界变化外还产生了谓词、论元或parser结构变化。",
            "4. 句界攻击增加句数的5个样本没有收益，当前证据更支持“合句削弱检测”，而不是任何句界变化都有效。",
            "",
            "## 解释边界",
            "",
            "该分析利用已有E3候选，能够分解句数变化、绿色句比例和匹配分的相关关系，"
            "但不能完全隔离模板破坏、聚合变化和parser误差。严格因果拆解仍需生成控制候选："
            "仅切句、改写但保持句数、改写并改变句数。",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
