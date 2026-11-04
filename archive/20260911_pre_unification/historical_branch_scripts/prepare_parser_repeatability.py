"""Build a shuffled repeated-text dataset for AMR parser repeatability checks."""

import json
import random
from pathlib import Path

from datasets import Dataset


root = Path(__file__).resolve().parents[1]
source_file = root / "experiments" / "candidate_budget16_pilot30" / "candidates.json"
out = root / "experiments" / "parser_repeatability30"
out.mkdir(parents=True, exist_ok=True)

records = json.loads(source_file.read_text(encoding="utf-8"))["records"]
rows = [
    {"source_index": int(record["index"]), "repeat": repeat, "text": record["source"]}
    for repeat in range(5)
    for record in records
]
random.Random(20260818).shuffle(rows)

(out / "manifest.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
)
Dataset.from_dict({"text": [row["text"] for row in rows]}).save_to_disk(str(out / "dataset"))
print({"unique_texts": len(records), "repeats": 5, "rows": len(rows)})
