"""Combine blinded quality judgments with detector outcomes by budget."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parents[1]
exp = root / "experiments" / "candidate_budget16_pilot30"
rows = json.loads((exp / "rows.json").read_text(encoding="utf-8"))
surface_rows = json.loads((exp / "validation.json").read_text(encoding="utf-8"))["rows"]
judgments = json.loads((exp / "blind_judgments.json").read_text(encoding="utf-8"))
z = np.concatenate(
    [np.load(exp / "chunks" / f"chunk{i}" / "detection" / "machine_z_scores.npy") for i in range(4)]
).reshape(30, 16)
surface = np.asarray(
    [row["numbers_preserved"] and row["negation_preserved"] for row in surface_rows], dtype=bool
).reshape(30, 16)
by_key = {(row["index"], row["candidate_index"]): row for row in rows}
review_map = {}
for group in judgments["groups"]:
    for candidate, review in zip(group["candidates"], group["reviews"]):
        review_map[(int(group["source_index"]), candidate)] = review

selected_rows = []
for k in (1, 2, 4, 8, 16):
    for paragraph in range(30):
        allowed = np.flatnonzero(surface[paragraph, :k])
        pool = allowed if len(allowed) else np.arange(k)
        candidate_index = int(pool[np.argmin(z[paragraph, pool])])
        row = by_key[(paragraph, candidate_index)]
        review = review_map[(paragraph, row["attack"])]
        semantic_valid = (
            bool(review["meaning_preserved"])
            and bool(review["facts_preserved"])
            and bool(review["readable"])
            and not bool(review["major_error"])
            and int(review["meaning_score"]) >= 4
            and int(review["factual_score"]) >= 4
            and int(review["readability_score"]) >= 4
        )
        surface_valid = bool(surface[paragraph, candidate_index])
        escaped = bool(z[paragraph, candidate_index] <= 2.33)
        selected_rows.append(
            {
                "budget_k": k,
                "source_index": paragraph,
                "candidate_index": candidate_index,
                "z_score": float(z[paragraph, candidate_index]),
                "escaped": escaped,
                "surface_valid": surface_valid,
                "semantic_valid": semantic_valid,
                "valid": surface_valid and semantic_valid,
                "valid_asr": escaped and surface_valid and semantic_valid,
                "review": review,
            }
        )

summary = {}
for k in (1, 2, 4, 8, 16):
    group = [row for row in selected_rows if row["budget_k"] == k]
    summary[str(k)] = {
        "n": len(group),
        "escape_rate": float(np.mean([row["escaped"] for row in group])),
        "surface_valid_rate": float(np.mean([row["surface_valid"] for row in group])),
        "semantic_valid_rate": float(np.mean([row["semantic_valid"] for row in group])),
        "combined_quality_valid_rate": float(np.mean([row["valid"] for row in group])),
        "valid_asr": float(np.mean([row["valid_asr"] for row in group])),
        "mean_z": float(np.mean([row["z_score"] for row in group])),
        "unique_selected_candidates": len({(row["source_index"], row["candidate_index"]) for row in group}),
    }

failure_reasons = {}
for row in selected_rows:
    if not row["semantic_valid"]:
        reason = str(row["review"].get("reason", "unspecified"))
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

output = {
    "validity_rule": {
        "surface": "numbers and negation preserved",
        "semantic": "meaning/facts/readability true, major_error false, and all three scores >=4",
        "success": "surface valid AND semantic valid AND z<=2.33",
    },
    "summary": summary,
    "rows": selected_rows,
    "semantic_failure_reasons": failure_reasons,
}
(exp / "judged_analysis.json").write_text(
    json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
)

review_table = list(csv.DictReader((exp / "blind_human_review.csv").open(encoding="utf-8-sig")))
for row in review_table:
    review = review_map[(int(row["source_index"]), row["candidate"])]
    row["meaning_preserved_0_or_1"] = int(bool(review["meaning_preserved"]))
    row["facts_preserved_0_or_1"] = int(bool(review["facts_preserved"]))
    row["readable_0_or_1"] = int(bool(review["readable"]))
    row["reviewer_notes"] = review.get("reason", "")
with (exp / "blind_review_completed.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=review_table[0].keys())
    writer.writeheader()
    writer.writerows(review_table)

lines = [
    "# 候选预算盲法质量评审结果",
    "",
    "评审时未提供SWAN分数、是否逃逸、候选排名或预算信息。严格有效要求：数字与否定词自动约束通过；"
    "语义、事实和可读性均通过；三项评分均不低于4；无重大错误。",
    "",
    "| K | 逃逸率 | 表面约束通过率 | 语义质量通过率 | 联合质量通过率 | Valid ASR | 平均z |",
    "|---:|---:|---:|---:|---:|---:|---:|",
]
for k in (1, 2, 4, 8, 16):
    row = summary[str(k)]
    lines.append(
        f"| {k} | {row['escape_rate']:.1%} | {row['surface_valid_rate']:.1%} | "
        f"{row['semantic_valid_rate']:.1%} | {row['combined_quality_valid_rate']:.1%} | "
        f"{row['valid_asr']:.1%} | {row['mean_z']:.3f} |"
    )
lines.extend(
    [
        "",
        "## 解释限制",
        "",
        "这是30段先导实验，质量评审由独立模型评审完成而非多名人工标注者；正式论文应抽样复核并报告一致性。"
        "由于本表采用确定性的前K候选，而检测预算曲线采用1000次随机子采样，两张表的逃逸率不应逐项视作同一统计量。",
    ]
)
(exp / "judged_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

x = np.asarray([1, 2, 4, 8, 16])
fig, ax = plt.subplots(figsize=(6.3, 4.2))
ax.plot(x, [summary[str(k)]["escape_rate"] for k in x], "o-", label="Escape rate")
ax.plot(x, [summary[str(k)]["valid_asr"] for k in x], "o-", label="Judged Valid ASR")
ax.set(xlabel="Candidate budget K", ylabel="Rate", xticks=x, ylim=(0, 1.03))
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(exp / "judged_valid_asr_curve.png", dpi=200)
plt.close(fig)
print(json.dumps(summary, ensure_ascii=False, indent=2))
