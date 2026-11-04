"""Summarize repeated S2Match scores for the threshold-edge candidate."""

import json
from collections import Counter
from pathlib import Path

import numpy as np


root = Path(__file__).resolve().parents[1]
out = root / "experiments" / "candidate_budget16_pilot30" / "final_hardcase_control"
meta = json.loads((out / "threshold_repeat_meta.json").read_text(encoding="utf-8"))
scores = np.load(out / "threshold_repeat_detection" / "machine_z_scores.npy")
summary = {
    **meta,
    "observations": len(scores),
    "unique_scores": {str(score): count for score, count in Counter(scores.tolist()).items()},
    "mean": float(scores.mean()),
    "minimum": float(scores.min()),
    "maximum": float(scores.max()),
    "below_or_equal_2_33": int(np.sum(scores <= 2.33)),
    "stable_escape_rate": float(np.mean(scores <= 2.33)),
}
(out / "threshold_repeatability.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
with (out / "report.md").open("a", encoding="utf-8") as handle:
    handle.write(
        "\n## 阈值边缘重复性\n\n"
        f"索引16唯一Valid候选的同一解析AMR重复计算{len(scores)}次，"
        f"z范围为{scores.min():.3f}–{scores.max():.3f}，"
        f"其中{np.sum(scores <= 2.33)}/{len(scores)}次位于2.33阈值以下。\n"
    )
print(json.dumps(summary, ensure_ascii=False, indent=2))
