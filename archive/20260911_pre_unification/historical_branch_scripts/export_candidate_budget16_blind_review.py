"""Export blinded human-review rows for deterministic prefix-budget selections."""

import csv
import json
import random
from pathlib import Path

import numpy as np


root = Path(__file__).resolve().parents[1]
exp = root / "experiments" / "candidate_budget16_pilot30"
rows = json.loads((exp / "rows.json").read_text(encoding="utf-8"))
validation = json.loads((exp / "validation.json").read_text(encoding="utf-8"))["rows"]
z = np.concatenate(
    [np.load(exp / "chunks" / f"chunk{i}" / "detection" / "machine_z_scores.npy") for i in range(4)]
).reshape(30, 16)
quality = np.asarray(
    [row["numbers_preserved"] and row["negation_preserved"] for row in validation], dtype=bool
).reshape(30, 16)
by_key = {(row["index"], row["candidate_index"]): row for row in rows}

review = []
for k in (1, 2, 4, 8, 16):
    for paragraph in range(30):
        allowed = np.flatnonzero(quality[paragraph, :k])
        pool = allowed if len(allowed) else np.arange(k)
        candidate_index = int(pool[np.argmin(z[paragraph, pool])])
        selected = by_key[(paragraph, candidate_index)]
        review.append(
            {
                "budget_k": k,
                "source_index": paragraph,
                "source": selected["source"],
                "candidate": selected["attack"],
                "meaning_preserved_0_or_1": "",
                "facts_preserved_0_or_1": "",
                "readable_0_or_1": "",
                "reviewer_notes": "",
            }
        )

random.Random(20260818).shuffle(review)
for review_id, row in enumerate(review, 1):
    row["review_id"] = review_id
columns = [
    "review_id",
    "budget_k",
    "source_index",
    "source",
    "candidate",
    "meaning_preserved_0_or_1",
    "facts_preserved_0_or_1",
    "readable_0_or_1",
    "reviewer_notes",
]
with (exp / "blind_human_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writeheader()
    writer.writerows(review)
print(f"exported={len(review)}")
