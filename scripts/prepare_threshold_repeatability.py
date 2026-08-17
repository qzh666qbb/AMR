"""Duplicate the threshold-edge parsed document for S2Match repeatability auditing."""

import json
from pathlib import Path

import numpy as np


root = Path(__file__).resolve().parents[1]
out = root / "experiments" / "candidate_budget16_pilot30" / "final_hardcase_control"
analysis = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
parsed = json.loads((out / "dataset" / "parsed_amrs.json").read_text(encoding="utf-8"))
z = np.load(out / "detection" / "machine_z_scores.npy")
winners = [
    (position, row)
    for position, row in enumerate(analysis["rows"])
    if row["index"] == 16 and row["valid_success"]
]
if len(winners) != 1:
    raise RuntimeError(f"Expected one edge winner, found {len(winners)}")
position, row = winners[0]
if abs(float(z[position]) - float(row["z_score"])) > 1e-12:
    raise RuntimeError("Analysis and detector score differ")
repeats = 40
(out / "threshold_repeat_parsed.json").write_text(
    json.dumps([parsed[position] for _ in range(repeats)], indent=2), encoding="utf-8"
)
(out / "threshold_repeat_meta.json").write_text(
    json.dumps(
        {
            "repeats": repeats,
            "source_index": row["index"],
            "strategy": row["strategy"],
            "candidate_index": row["candidate_index"],
            "original_z": row["z_score"],
            "attack": row["attack"],
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print({"position": position, "repeats": repeats, "original_z": row["z_score"]})
