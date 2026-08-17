"""Validate, flatten, and chunk the 16-candidate budget pilot."""

import json
import re
from collections import Counter
from pathlib import Path

from datasets import Dataset
from nltk.tokenize import sent_tokenize


def numbers(text: str) -> list[str]:
    return re.findall(r"\b\d+(?:[.,]\d+)*\b", text)


def negations(text: str) -> int:
    return len(re.findall(r"\b(?:no|not|never|neither|nor|without|n't)\b", text, flags=re.I))


root = Path(__file__).resolve().parents[1]
exp = root / "experiments" / "candidate_budget16_pilot30"
payload = json.loads((exp / "candidates.json").read_text(encoding="utf-8"))
rows = []
validation = []
for record in sorted(payload["records"], key=lambda row: int(row["index"])):
    candidates = record["candidates"]
    if len(candidates) != 16:
        raise RuntimeError(f"Paragraph {record['index']} has {len(candidates)} candidates")
    source = record["source"].strip()
    source_sentences = len(sent_tokenize(source))
    normalized = [" ".join(candidate.split()).casefold() for candidate in candidates]
    for candidate_index, attack in enumerate(candidates):
        attack = attack.strip()
        if not attack:
            raise RuntimeError(f"Empty candidate at {record['index']}:{candidate_index}")
        row = {
            "index": int(record["index"]),
            "candidate_index": candidate_index,
            "source": source,
            "attack": attack,
        }
        rows.append(row)
        validation.append(
            {
                "index": row["index"],
                "candidate_index": candidate_index,
                "source_sentences": source_sentences,
                "attack_sentences": len(sent_tokenize(attack)),
                "numbers_preserved": numbers(source) == numbers(attack),
                "negation_preserved": negations(source) == negations(attack),
                "duplicate_within_paragraph": normalized.count(normalized[candidate_index]) > 1,
            }
        )

(exp / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
summary = {
    "paragraphs": len(payload["records"]),
    "candidates": len(rows),
    "numbers_preserved": sum(row["numbers_preserved"] for row in validation),
    "negation_preserved": sum(row["negation_preserved"] for row in validation),
    "duplicate_candidates": sum(row["duplicate_within_paragraph"] for row in validation),
    "sentence_delta_distribution": dict(
        Counter(row["attack_sentences"] - row["source_sentences"] for row in validation)
    ),
}
(exp / "validation.json").write_text(
    json.dumps({"summary": summary, "rows": validation}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
for chunk_index, start in enumerate(range(0, len(rows), 120)):
    chunk = rows[start : start + 120]
    destination = exp / "chunks" / f"chunk{chunk_index}"
    Dataset.from_dict({"text": [row["attack"] for row in chunk]}).save_to_disk(str(destination))
print(json.dumps(summary, ensure_ascii=False))
