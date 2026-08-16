import json
from pathlib import Path

from datasets import Dataset

root = Path(__file__).resolve().parents[1]
experiment = root / "experiments/planE_e2_joint250"
rows = json.load((experiment / "candidates.json").open(encoding="utf-8"))["rows"]
new_rows = rows[400:]
chunks = experiment / "chunks"
chunks.mkdir(parents=True, exist_ok=True)
for start in range(0, len(new_rows), 100):
    chunk_id = start // 100
    target = chunks / f"chunk_{chunk_id:02d}"
    Dataset.from_dict({"text": [row["attack"] for row in new_rows[start:start + 100]]}).save_to_disk(str(target))
print(f"prepared {len(new_rows)} rows in {(len(new_rows) + 99) // 100} chunks")
