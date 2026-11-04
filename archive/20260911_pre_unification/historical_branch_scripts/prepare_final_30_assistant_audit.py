"""Assemble one final candidate per paragraph for independent assistant audit."""

import json
from pathlib import Path

import numpy as np


root = Path(__file__).resolve().parents[1]
exp = root / "experiments" / "candidate_budget16_pilot30"
base_rows = json.loads((exp / "rows.json").read_text(encoding="utf-8"))
full = json.loads((exp / "full_quality_analysis.json").read_text(encoding="utf-8"))
z = np.concatenate(
    [np.load(exp / "chunks" / f"chunk{i}" / "detection" / "machine_z_scores.npy") for i in range(4)]
)

# Reconstruct strict full-pool labels from the saved all-candidate judgments.
judgments = json.loads((exp / "all_candidate_judgments.json").read_text(encoding="utf-8"))
semantic = {}
for group in judgments["groups"]:
    for candidate_index, review in enumerate(group["reviews"]):
        semantic[(int(group["source_index"]), candidate_index)] = (
            bool(review["meaning_preserved"])
            and bool(review["facts_preserved"])
            and bool(review["readable"])
            and not bool(review["major_error"])
            and min(int(review[key]) for key in ("meaning_score", "factual_score", "readability_score")) >= 4
        )

selected = []
for index in range(30):
    candidates = []
    for position, row in enumerate(base_rows):
        if row["index"] == index and semantic[(index, row["candidate_index"])] and z[position] <= 2.33:
            candidates.append((float(z[position]), row, "base16"))
    if candidates:
        score, row, pool = min(candidates, key=lambda item: item[0])
        selected.append({**row, "z_score": score, "pool": pool})

for name in ("fact_constrained_rescue", "final_hardcase_control"):
    analysis = json.loads((exp / name / "analysis.json").read_text(encoding="utf-8"))
    for row in analysis["rows"]:
        if not row["valid_success"]:
            continue
        index = int(row["index"])
        current = next((item for item in selected if item["index"] == index), None)
        candidate = {
            "index": index,
            "candidate_index": row["candidate_index"],
            "source": row["source"],
            "attack": row["attack"],
            "z_score": row["z_score"],
            "pool": name,
            "strategy": row.get("strategy"),
        }
        if current is None or candidate["z_score"] < current["z_score"]:
            selected = [item for item in selected if item["index"] != index]
            selected.append(candidate)

selected.sort(key=lambda row: row["index"])
if len(selected) != 30:
    raise RuntimeError(f"Expected 30 final candidates, found {len(selected)}")
for row in selected:
    row.update(
        {
            "assistant_meaning_preserved": None,
            "assistant_facts_preserved": None,
            "assistant_readable": None,
            "assistant_major_error": None,
            "assistant_reason": "",
        }
    )
(exp / "final_30_assistant_audit.json").write_text(
    json.dumps(
        {
            "protocol": {
                "reviewer": "OpenAI Codex assistant, designated by the researcher as an independent audit",
                "blinded_to_previous_review_reasons": True,
                "note": "This is an AI-assistant audit and must not be represented as a human-subject annotation.",
            },
            "rows": selected,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print({"selected": len(selected), "pools": {name: sum(r["pool"] == name for r in selected) for name in {r["pool"] for r in selected}}})
