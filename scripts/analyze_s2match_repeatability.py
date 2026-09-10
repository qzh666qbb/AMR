"""Summarize repeated S2Match results for one fixed AMR document."""

import json
from collections import Counter
from pathlib import Path

import numpy as np


root = Path(__file__).resolve().parents[1]
out = root / "experiments" / "parser_repeatability30"
scores = np.load(out / "s2match_repeat100_detection" / "machine_z_scores.npy")
counts = Counter(float(score) for score in scores)
summary = {
    "source_index": 23,
    "fixed_amr_repeats": len(scores),
    "unique_z_scores": {str(score): count for score, count in sorted(counts.items())},
    "modal_z": counts.most_common(1)[0][0],
    "nonmodal_rate": 1 - counts.most_common(1)[0][1] / len(scores),
    "escape_rate_at_2_33": float(np.mean(scores <= 2.33)),
}
(out / "s2match_repeatability.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
with (out / "report.md").open("a", encoding="utf-8") as handle:
    handle.write(
        "\n## 固定AMR的S2Match重复计算\n\n"
        f"对索引23的完全相同AMR重复检测{len(scores)}次，得到"
        f"{len(counts)}种z值；众数为{summary['modal_z']:.3f}，"
        f"非众数比例为{summary['nonmodal_rate']:.1%}。\n"
    )
print(json.dumps(summary, ensure_ascii=False, indent=2))
