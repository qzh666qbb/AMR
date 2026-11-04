"""Evaluate fact-constrained rescue candidates and combined pilot coverage."""

import json
from pathlib import Path

import numpy as np


THRESHOLD = 2.33
root = Path(__file__).resolve().parents[1]
exp = root / "experiments" / "candidate_budget16_pilot30"
out = exp / "fact_constrained_rescue"
rows = json.loads((out / "rows.json").read_text(encoding="utf-8"))
judgments = json.loads((out / "judgments.json").read_text(encoding="utf-8"))
parsed = json.loads((out / "dataset" / "parsed_amrs.json").read_text(encoding="utf-8"))
z = np.load(out / "detection" / "machine_z_scores.npy")
review_map = {
    (int(group["source_index"]), candidate_index): review
    for group in judgments["groups"]
    for candidate_index, review in enumerate(group["reviews"])
}

evaluated = []
for position, row in enumerate(rows):
    review = review_map[(row["index"], row["candidate_index"])]
    semantic_valid = (
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
    evaluated.append(
        {
            **row,
            "z_score": float(z[position]),
            "escaped": bool(z[position] <= THRESHOLD),
            "semantic_valid": semantic_valid,
            "valid_success": bool(
                row["numbers_preserved"]
                and row["negation_preserved"]
                and semantic_valid
                and z[position] <= THRESHOLD
            ),
            "amr_nonempty_rate": float(np.mean([bool(amr.strip()) for amr in parsed[position]])),
            "review": review,
        }
    )

summary = {}
rescued = []
for index in (6, 14, 16, 20):
    group = [row for row in evaluated if row["index"] == index]
    valid = [row for row in group if row["semantic_valid"]]
    successful = [row for row in group if row["valid_success"]]
    if successful:
        rescued.append(index)
    summary[str(index)] = {
        "candidates": len(group),
        "semantic_valid": len(valid),
        "escaped": sum(row["escaped"] for row in group),
        "valid_successes": len(successful),
        "best_valid_z": min((row["z_score"] for row in valid), default=None),
        "mean_amr_nonempty_rate": float(np.mean([row["amr_nonempty_rate"] for row in group])),
    }

result = {
    "summary": summary,
    "rescued_indices": rescued,
    "rescue_rate": len(rescued) / 4,
    "combined_quality_covered_paragraphs": 26 + len(rescued),
    "combined_valid_asr_upper_selection": (26 + len(rescued)) / 30,
    "rows": evaluated,
}
(out / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

lines = [
    "# 事实约束自适应补救实验",
    "",
    "对普通16候选池中没有严格合格候选的4个段落，每段生成8个事实约束候选，并独立进行AMR检测和盲法质量评审。",
    "",
    "| 段落 | 候选数 | 语义合格 | 检测逃逸 | Valid成功 | 最低合格z | AMR非空率 |",
    "|---:|---:|---:|---:|---:|---:|---:|",
]
for index in (6, 14, 16, 20):
    row = summary[str(index)]
    best = "—" if row["best_valid_z"] is None else f"{row['best_valid_z']:.3f}"
    lines.append(
        f"| {index} | {row['candidates']} | {row['semantic_valid']} | {row['escaped']} | "
        f"{row['valid_successes']} | {best} | {row['mean_amr_nonempty_rate']:.1%} |"
    )
lines.extend(
    [
        "",
        f"补救成功{len(rescued)}/4段；与原16候选池合并后，至少存在一个质量合格且逃逸候选的段落为"
        f"{26 + len(rescued)}/30（{(26 + len(rescued))/30:.1%}）。",
        "",
        "该合并比例是候选池存在性上限，前提是可以用质量评审进行选优；质量评审稳定性问题仍然适用。",
    ]
)
(out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(result | {"rows": f"{len(evaluated)} rows"}, ensure_ascii=False, indent=2))
