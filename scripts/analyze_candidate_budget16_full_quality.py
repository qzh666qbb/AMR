"""Analyze candidate budgets after reviewing all 480 candidates."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


K_VALUES = (1, 2, 4, 8, 16)
REPEATS = 1000
SEED = 20260818
THRESHOLD = 2.33

root = Path(__file__).resolve().parents[1]
exp = root / "experiments" / "candidate_budget16_pilot30"
surface_rows = json.loads((exp / "validation.json").read_text(encoding="utf-8"))["rows"]
judgments = json.loads((exp / "all_candidate_judgments.json").read_text(encoding="utf-8"))
small_group_judgments = json.loads((exp / "blind_judgments.json").read_text(encoding="utf-8"))
z = np.concatenate(
    [np.load(exp / "chunks" / f"chunk{i}" / "detection" / "machine_z_scores.npy") for i in range(4)]
).reshape(30, 16)
surface = np.asarray(
    [row["numbers_preserved"] and row["negation_preserved"] for row in surface_rows], dtype=bool
).reshape(30, 16)

semantic = np.zeros((30, 16), dtype=bool)
scores = np.zeros((30, 16, 3), dtype=int)
for group in judgments["groups"]:
    paragraph = int(group["source_index"])
    if len(group["reviews"]) != 16:
        raise RuntimeError(f"Paragraph {paragraph} does not have 16 reviews")
    for candidate, review in enumerate(group["reviews"]):
        score = [
            int(review["meaning_score"]),
            int(review["factual_score"]),
            int(review["readability_score"]),
        ]
        scores[paragraph, candidate] = score
        semantic[paragraph, candidate] = (
            bool(review["meaning_preserved"])
            and bool(review["facts_preserved"])
            and bool(review["readable"])
            and not bool(review["major_error"])
            and min(score) >= 4
        )
quality = surface & semantic


def strict_review(review: dict) -> bool:
    return (
        bool(review["meaning_preserved"])
        and bool(review["facts_preserved"])
        and bool(review["readable"])
        and not bool(review["major_error"])
        and min(
            int(review["meaning_score"]),
            int(review["factual_score"]),
            int(review["readability_score"]),
        )
        >= 4
    )


def review_map(payload: dict) -> dict[tuple[int, str], bool]:
    return {
        (int(group["source_index"]), candidate): strict_review(review)
        for group in payload["groups"]
        for candidate, review in zip(group["candidates"], group["reviews"])
    }


small_map = review_map(small_group_judgments)
full_map = review_map(judgments)
overlap = sorted(set(small_map) & set(full_map))
agreement = sum(small_map[key] == full_map[key] for key in overlap)
stability = {
    "overlapping_pairs": len(overlap),
    "strict_label_agreements": agreement,
    "strict_label_agreement_rate": agreement / len(overlap),
    "small_group_valid": sum(small_map[key] for key in overlap),
    "sixteen_candidate_group_valid": sum(full_map[key] for key in overlap),
    "valid_in_both": sum(small_map[key] and full_map[key] for key in overlap),
}

rng = np.random.default_rng(SEED)
summary = {}
for k in K_VALUES:
    run_rows = []
    for _ in range(REPEATS):
        chosen = np.stack([rng.choice(16, size=k, replace=False) for _ in range(30)])
        subset_z = np.take_along_axis(z, chosen, axis=1)
        subset_quality = np.take_along_axis(quality, chosen, axis=1)
        available = np.any(subset_quality, axis=1)
        selected_z = np.full(30, np.nan)
        for paragraph in range(30):
            allowed = np.flatnonzero(subset_quality[paragraph])
            if len(allowed):
                selected_z[paragraph] = subset_z[paragraph, allowed].min()
        valid_success = available & (np.nan_to_num(selected_z, nan=np.inf) <= THRESHOLD)
        run_rows.append(
            {
                "quality_coverage": float(available.mean()),
                "valid_asr": float(valid_success.mean()),
                "escape_given_quality": float(valid_success.sum() / max(1, available.sum())),
                "mean_selected_z": float(np.nanmean(selected_z)) if available.any() else None,
            }
        )

    summary[str(k)] = {
        name: {
            "mean": float(np.mean([row[name] for row in run_rows])),
            "p025": float(np.quantile([row[name] for row in run_rows], 0.025)),
            "p975": float(np.quantile([row[name] for row in run_rows], 0.975)),
        }
        for name in run_rows[0]
    }

per_paragraph_valid_counts = quality.sum(axis=1)
output = {
    "protocol": {
        "paragraphs": 30,
        "candidates_per_paragraph": 16,
        "random_subsampling_repeats": REPEATS,
        "seed": SEED,
        "quality_rule": "surface constraints AND strict blinded semantic judgment",
    },
    "pool_quality": {
        "valid_candidates": int(quality.sum()),
        "valid_candidate_rate": float(quality.mean()),
        "paragraphs_with_at_least_one_valid": int(np.sum(per_paragraph_valid_counts > 0)),
        "paragraphs_with_no_valid_candidate": int(np.sum(per_paragraph_valid_counts == 0)),
        "paragraph_indices_with_no_valid_candidate": np.flatnonzero(
            per_paragraph_valid_counts == 0
        ).astype(int).tolist(),
        "mean_valid_candidates_per_paragraph": float(per_paragraph_valid_counts.mean()),
        "valid_count_distribution": {
            str(int(value)): int(np.sum(per_paragraph_valid_counts == value))
            for value in np.unique(per_paragraph_valid_counts)
        },
    },
    "evaluator_stability": stability,
    "summary": summary,
}
(exp / "full_quality_analysis.json").write_text(
    json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
)

lines = [
    "# 全部480候选质量评审后的预算消融",
    "",
    "质量有效要求同时通过数字/否定词约束，以及盲法语义、事实、可读性严格评审。"
    "每个K从16候选中随机无放回抽取，重复1000次，再在质量合格候选中选择最低z。",
    "",
    "| K | 质量候选覆盖率 | Valid ASR | 合格后逃逸率 | 合格候选平均z |",
    "|---:|---:|---:|---:|---:|",
]
for k in K_VALUES:
    row = summary[str(k)]
    lines.append(
        f"| {k} | {row['quality_coverage']['mean']:.1%} | {row['valid_asr']['mean']:.1%} | "
        f"{row['escape_given_quality']['mean']:.1%} | {row['mean_selected_z']['mean']:.3f} |"
    )
pool = output["pool_quality"]
lines.extend(
    [
        "",
        "## 候选池诊断",
        "",
        f"- 严格质量合格候选：{pool['valid_candidates']}/480（{pool['valid_candidate_rate']:.1%}）。",
        f"- 16个候选中至少有一个合格候选的段落：{pool['paragraphs_with_at_least_one_valid']}/30。",
        f"- 即使K=16仍没有合格候选的段落：{pool['paragraphs_with_no_valid_candidate']}/30。",
        f"- 每段平均合格候选数：{pool['mean_valid_candidates_per_paragraph']:.2f}。",
        f"- 48个重复评审候选的严格标签一致率只有{stability['strict_label_agreement_rate']:.1%}"
        f"（小组评审判为合格{stability['small_group_valid']}个，16候选同组评审判为合格{stability['sixteen_candidate_group_valid']}个）。",
        "",
        "## 结论边界",
        "",
        "增加K主要提高质量合格候选的覆盖率；如果某段16个候选全部存在语义问题，继续做同分布抽样的收益有限，"
        "需要事实约束更强的生成提示或自适应补救。重复评审一致率较低，说明当前模型质量标签受候选组规模或上下文影响；"
        "因此57.9%–76.7%只能作为模型评审下的探索性估计，正式结果必须由独立人工复核或多评审器多数表决。",
    ]
)
(exp / "full_quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

x = np.asarray(K_VALUES)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(x, [summary[str(k)]["quality_coverage"]["mean"] for k in K_VALUES], "o-", label="Quality coverage")
ax.plot(x, [summary[str(k)]["valid_asr"]["mean"] for k in K_VALUES], "o-", label="Valid ASR")
ax.set(xlabel="Candidate budget K", ylabel="Rate", xticks=x, ylim=(0, 1.03))
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(exp / "full_quality_budget_curve.png", dpi=200)
plt.close(fig)
print(json.dumps(output, ensure_ascii=False, indent=2))
