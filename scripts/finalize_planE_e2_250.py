import json
import re
from pathlib import Path

import numpy as np
import penman

root = Path(__file__).resolve().parents[1]
experiment = root / "experiments/planE_e2_joint250"
rows = json.load((experiment / "rows.json").open(encoding="utf-8"))
parsed = json.load((experiment / "parsed.json").open(encoding="utf-8"))
scores = np.load(experiment / "detection/machine_z_scores.npy")
if not (len(rows) == len(parsed) == len(scores) == 1000):
    raise RuntimeError(f"Length mismatch: rows={len(rows)}, parsed={len(parsed)}, scores={len(scores)}")
for position, (row, score) in enumerate(zip(rows, scores)):
    row["position"] = position
    row["z_score"] = float(score)
best = []
for index in range(250):
    best.append(min((row for row in rows if row["index"] == index), key=lambda row: row["z_score"]))
quality = []
for row in best:
    source, attack = row["source"], row["attack"]
    numbers = set(re.findall(r"\b\d[\d,.:/%-]*\b", source))
    candidate_numbers = set(re.findall(r"\b\d[\d,.:/%-]*\b", attack))
    negation = re.compile(r"\b(?:no|not|never|without|cannot|n't)\b", re.I)
    ratio = len(attack.split()) / max(1, len(source.split()))
    valid_graphs = 0
    graphs = parsed[row["position"]]
    for graph in graphs:
        try:
            penman.decode(graph)
            valid_graphs += 1
        except Exception:
            pass
    quality.append({
        "index": row["index"],
        "candidate": row["candidate"],
        "position": row["position"],
        "z_score": row["z_score"],
        "length_ratio": ratio,
        "length_ok": 0.7 <= ratio <= 1.6,
        "numbers_ok": numbers.issubset(candidate_numbers),
        "negation_ok": bool(negation.search(source)) == bool(negation.search(attack)),
        "amr_all_valid": len(graphs) > 0 and valid_graphs == len(graphs),
    })
json.dump(rows, (experiment / "scored_candidates.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(best, (experiment / "best_by_paragraph.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(quality, (experiment / "quality_checks.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
best_scores = np.array([row["z_score"] for row in best])
print(f"all_mean={scores.mean():.3f} all_low={(scores <= 2.33).sum()}/{len(scores)}")
print(f"best_mean={best_scores.mean():.3f} best_low={(best_scores <= 2.33).sum()}/{len(best_scores)}")
for field in ("length_ok", "numbers_ok", "negation_ok", "amr_all_valid"):
    print(f"{field}={sum(row[field] for row in quality)}/250")
review_path = experiment / "llm_review.json"
if review_path.exists():
    reviews = json.load(review_path.open(encoding="utf-8"))
    quality_by_index = {row["index"]: row for row in quality}
    strict = []
    for row in reviews:
        gate = quality_by_index[row["index"]]
        review = row["review"]
        if (
            review.get("score", 0) >= 4
            and not review.get("major_error", True)
            and gate["length_ok"]
            and gate["numbers_ok"]
            and gate["negation_ok"]
            and gate["amr_all_valid"]
        ):
            strict.append(row)
    json.dump(strict, (experiment / "strict_valid.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
    strict_scores = np.array([row["z_score"] for row in strict])
    print(f"strict={len(strict)}/250 strict_mean={strict_scores.mean():.3f} strict_low={(strict_scores <= 2.33).sum()}/{len(strict_scores)}")
