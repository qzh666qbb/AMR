import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E3 = ROOT / "experiments/e3_main"
OUT = E3 / "human_eval_sample_100.csv"

with (E3 / "candidates.json").open(encoding="utf-8") as handle:
    payload = json.load(handle)

families = ["plain_paraphrase", "sentence_boundary", "amr_guided"]
rows = []
for i, record in enumerate(payload["records"][:100]):
    family = families[i % len(families)]
    rows.append(
        {
            "pair_id": i + 1,
            "source_text": record["source"],
            "candidate_method_blinded": f"Method-{(i % len(families)) + 1}",
            "candidate_text": record["candidates"][family],
            "meaning_preserved_1to5": "",
            "factual_consistency_1to5": "",
            "fluency_1to5": "",
            "coherent_1to5": "",
            "major_error": "",
            "notes": "",
        }
    )

with OUT.open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(OUT)
