"""Create a randomized blind human-review sheet for prompt ablation candidates."""

import csv
import json
import random
from pathlib import Path


root = Path(__file__).resolve().parents[1]
exp = root / "experiments" / "prompt_component_ablation30"
rows = json.loads((exp / "candidates.json").read_text(encoding="utf-8"))["rows"]
random.Random(20260818).shuffle(rows)
public = []
private = []
for review_id, row in enumerate(rows, start=1):
    public.append(
        {
            "review_id": review_id,
            "source": row["source"],
            "candidate": row["attack"],
            "core_semantics_preserved_0_1": "",
            "entities_numbers_preserved_0_1": "",
            "negation_modality_preserved_0_1": "",
            "event_roles_preserved_0_1": "",
            "no_added_or_missing_facts_0_1": "",
            "natural_readable_0_1": "",
            "overall_valid_0_1": "",
            "notes": "",
        }
    )
    private.append(
        {"review_id": review_id, "index": row["index"], "strategy": row["strategy"]}
    )
with (exp / "human_review_blind.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(public[0]))
    writer.writeheader()
    writer.writerows(public)
(exp / "human_review_private_key.json").write_text(
    json.dumps(private, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"prepared {len(public)} blinded rows")
