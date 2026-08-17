"""Prepare the final hard-case strategy-control dataset."""

import json
import re
from pathlib import Path

from datasets import Dataset


root = Path(__file__).resolve().parents[1]
out = root / "experiments" / "candidate_budget16_pilot30" / "final_hardcase_control"
payload = json.loads((out / "candidates.json").read_text(encoding="utf-8"))
rows = []
for record in payload["records"]:
    source_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)*\b", record["source"]))
    source_negation = bool(re.search(r"\b(?:no|not|never|neither|nor|without|n't)\b", record["source"], re.I))
    for strategy in ("literal_boundary_only", "constrained_structural"):
        for candidate_index, attack in enumerate(record[strategy]):
            rows.append(
                {
                    "index": int(record["index"]),
                    "strategy": strategy,
                    "candidate_index": candidate_index,
                    "source": record["source"],
                    "attack": attack.strip(),
                    "numbers_preserved": source_numbers
                    == set(re.findall(r"\b\d+(?:[.,]\d+)*\b", attack)),
                    "negation_preserved": source_negation
                    == bool(re.search(r"\b(?:no|not|never|neither|nor|without|n't)\b", attack, re.I)),
                }
            )
(out / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
Dataset.from_dict({"text": [row["attack"] for row in rows]}).save_to_disk(str(out / "dataset"))
print({"rows": len(rows), "surface_valid": sum(r["numbers_preserved"] and r["negation_preserved"] for r in rows)})
