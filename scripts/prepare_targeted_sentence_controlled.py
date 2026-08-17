"""Build the controlled targeted-attack dataset and verify reconstruction constraints."""

import json
from pathlib import Path

from datasets import Dataset
from nltk.tokenize import sent_tokenize


root = Path(__file__).resolve().parents[1]
experiment = root / "experiments" / "targeted_sentence_controlled30"
payload = json.loads((experiment / "candidates.json").read_text(encoding="utf-8"))
rows = payload["rows"]
validation = []
for row in rows:
    source = sent_tokenize(row["source"])
    attack = sent_tokenize(row["attack"])
    targets = set(row["target_positions"])
    same_count = len(source) == len(attack)
    non_targets_verbatim = same_count and all(
        source[position] == attack[position]
        for position in range(len(source))
        if position not in targets
    )
    targets_changed = same_count and all(
        source[position] != attack[position] for position in targets
    )
    validation.append(
        {
            "index": row["index"],
            "strategy": row["strategy"],
            "same_sentence_count": same_count,
            "non_targets_verbatim": non_targets_verbatim,
            "targets_changed": targets_changed,
        }
    )

summary = {
    "rows": len(rows),
    "same_sentence_count": sum(row["same_sentence_count"] for row in validation),
    "non_targets_verbatim": sum(row["non_targets_verbatim"] for row in validation),
    "targets_changed": sum(row["targets_changed"] for row in validation),
}
if min(summary.values()) != len(rows):
    raise RuntimeError(f"Controlled reconstruction validation failed: {summary}")
(experiment / "validation.json").write_text(
    json.dumps({"summary": summary, "rows": validation}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
Dataset.from_dict({"text": [row["attack"] for row in rows]}).save_to_disk(
    str(experiment / "dataset")
)
print(json.dumps(summary))
