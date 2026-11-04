"""Flatten and prepare the fact-constrained rescue candidates."""

import json
import re
from pathlib import Path

from datasets import Dataset


root = Path(__file__).resolve().parents[1]
out = root / "experiments" / "candidate_budget16_pilot30" / "fact_constrained_rescue"
payload = json.loads((out / "candidates.json").read_text(encoding="utf-8"))
rows = []
for record in payload["records"]:
    if len(record["candidates"]) != 8:
        raise RuntimeError(f"Index {record['index']} does not have eight candidates")
    source_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)*\b", record["source"]))
    source_negation = bool(re.search(r"\b(?:no|not|never|neither|nor|without|n't)\b", record["source"], re.I))
    for candidate_index, attack in enumerate(record["candidates"]):
        rows.append(
            {
                "index": int(record["index"]),
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
print(
    {
        "rows": len(rows),
        "numbers_preserved": sum(row["numbers_preserved"] for row in rows),
        "negation_preserved": sum(row["negation_preserved"] for row in rows),
    }
)
