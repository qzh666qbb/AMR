"""Analyze strategy controls for the final two hard cases."""

import json
from pathlib import Path

import numpy as np


root = Path(__file__).resolve().parents[1]
out = root / "experiments" / "candidate_budget16_pilot30" / "final_hardcase_control"
rows = json.loads((out / "rows.json").read_text(encoding="utf-8"))
judgments = json.loads((out / "judgments.json").read_text(encoding="utf-8"))
parsed = json.loads((out / "dataset" / "parsed_amrs.json").read_text(encoding="utf-8"))
z = np.load(out / "detection" / "machine_z_scores.npy")
review_map = {}
for group in judgments["groups"]:
    selected = [row for row in rows if row["index"] == int(group["source_index"])]
    for row, review in zip(selected, group["reviews"]):
        review_map[(row["index"], row["strategy"], row["candidate_index"])] = review

evaluated = []
for position, row in enumerate(rows):
    review = review_map[(row["index"], row["strategy"], row["candidate_index"])]
    semantic = (
        bool(review["meaning_preserved"])
        and bool(review["facts_preserved"])
        and bool(review["readable"])
        and not bool(review["major_error"])
        and min(int(review[key]) for key in ("meaning_score", "factual_score", "readability_score")) >= 4
    )
    evaluated.append(
        {
            **row,
            "z_score": float(z[position]),
            "escaped": bool(z[position] <= 2.33),
            "semantic_valid": semantic,
            "valid_success": bool(semantic and z[position] <= 2.33),
            "amr_nonempty_rate": float(np.mean([bool(amr.strip()) for amr in parsed[position]])),
            "review": review,
        }
    )

summary = {}
for index in (6, 16):
    summary[str(index)] = {}
    for strategy in ("literal_boundary_only", "constrained_structural"):
        group = [row for row in evaluated if row["index"] == index and row["strategy"] == strategy]
        valid = [row for row in group if row["semantic_valid"]]
        summary[str(index)][strategy] = {
            "n": len(group),
            "semantic_valid": len(valid),
            "escaped": sum(row["escaped"] for row in group),
            "valid_success": sum(row["valid_success"] for row in group),
            "mean_z": float(np.mean([row["z_score"] for row in group])),
            "best_valid_z": min((row["z_score"] for row in valid), default=None),
            "amr_nonempty_rate": float(np.mean([row["amr_nonempty_rate"] for row in group])),
        }

output = {"summary": summary, "rows": evaluated}
(out / "analysis.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
lines = [
    "# 最终难例策略控制实验",
    "",
    "| 段落 | 策略 | 语义合格/8 | 逃逸/8 | Valid成功/8 | 平均z | 最低合格z |",
    "|---:|---|---:|---:|---:|---:|---:|",
]
names = {"literal_boundary_only": "字面句界重组", "constrained_structural": "受限结构改写"}
for index in (6, 16):
    for strategy in names:
        row = summary[str(index)][strategy]
        best = "—" if row["best_valid_z"] is None else f"{row['best_valid_z']:.3f}"
        lines.append(
            f"| {index} | {names[strategy]} | {row['semantic_valid']}/8 | {row['escaped']}/8 | "
            f"{row['valid_success']}/8 | {row['mean_z']:.3f} | {best} |"
        )
lines.extend(
    [
        "",
        "该实验用于区分句界操作与深层结构操作在极端模糊源文上的质量—规避权衡。质量标签仍来自单一模型评审，需结合原文逐例复核。",
    ]
)
(out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
