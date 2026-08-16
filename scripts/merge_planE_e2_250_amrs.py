import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
experiment = root / "experiments/planE_e2_joint250"
parsed = json.load((root / "experiments/planE_e2_joint100/parsed.json").open(encoding="utf-8"))
for chunk in sorted((experiment / "chunks").glob("chunk_*/parsed.json")):
    parsed.extend(json.load(chunk.open(encoding="utf-8")))
if len(parsed) != 1000:
    raise RuntimeError(f"Expected 1000 parsed candidates, got {len(parsed)}")
json.dump(parsed, (experiment / "parsed.json").open("w", encoding="utf-8"), ensure_ascii=False)
print(f"merged {len(parsed)} parsed candidates")
