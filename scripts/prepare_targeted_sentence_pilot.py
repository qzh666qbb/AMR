"""Validate generated targeted candidates and build a Hugging Face dataset."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
from nltk.tokenize import sent_tokenize


root = Path(__file__).resolve().parents[1]
experiment = root / "experiments" / "targeted_sentence_pilot30"
payload = json.loads((experiment / "candidates.json").read_text(encoding="utf-8"))
strategies = payload["protocol"]["strategies"]
rows = []
for record in payload["records"]:
    source_sentences = sent_tokenize(record["source"])
    for strategy in strategies:
        attack = record["candidates"][strategy]
        attack_sentences = sent_tokenize(attack)
        target_positions = {int(item["position"]) for item in record["targets"][strategy]}
        same_sentence_count = len(source_sentences) == len(attack_sentences)
        non_targets_verbatim = same_sentence_count and all(
            source_sentences[position] == attack_sentences[position]
            for position in range(len(source_sentences))
            if position not in target_positions
        )
        target_changed = same_sentence_count and all(
            source_sentences[position] != attack_sentences[position]
            for position in target_positions
        )
        rows.append(
            {
                "index": int(record["index"]),
                "strategy": strategy,
                "source": record["source"],
                "text": attack,
                "target_positions": sorted(target_positions),
                "source_sentence_count": len(source_sentences),
                "attack_sentence_count": len(attack_sentences),
                "same_sentence_count": same_sentence_count,
                "non_targets_verbatim": non_targets_verbatim,
                "target_changed": target_changed,
                "minimum_green_flips_to_escape": int(record["minimum_green_flips_to_escape"]),
            }
        )

(experiment / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
Dataset.from_list(rows).save_to_disk(str(experiment / "dataset"))
summary = {
    "rows": len(rows),
    "same_sentence_count": sum(row["same_sentence_count"] for row in rows),
    "non_targets_verbatim": sum(row["non_targets_verbatim"] for row in rows),
    "target_changed": sum(row["target_changed"] for row in rows),
}
(experiment / "validation.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(summary))
