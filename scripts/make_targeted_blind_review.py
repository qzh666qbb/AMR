"""Create a blinded, randomized human-review sheet for the targeted pilot."""

import csv
import json
import random
from pathlib import Path


root = Path(__file__).resolve().parents[1]
experiment = root / "experiments" / "targeted_sentence_controlled30"
payload = json.loads((experiment / "candidates.json").read_text(encoding="utf-8"))
analysis = json.loads((experiment / "analysis.json").read_text(encoding="utf-8"))
paired = set(analysis["protocol"]["paired_indices"])
rows = [row for row in payload["rows"] if int(row["index"]) in paired]
random.Random(20260817).shuffle(rows)

private_key = []
public_rows = []
for review_id, row in enumerate(rows, start=1):
    private_key.append(
        {
            "review_id": review_id,
            "index": row["index"],
            "strategy": row["strategy"],
        }
    )
    public_rows.append(
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

with (experiment / "human_review_blind.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(public_rows[0]))
    writer.writeheader()
    writer.writerows(public_rows)
(experiment / "human_review_private_key.json").write_text(
    json.dumps(private_key, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"prepared {len(public_rows)} blinded rows")
