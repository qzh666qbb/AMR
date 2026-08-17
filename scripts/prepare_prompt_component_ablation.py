"""Validate and prepare the prompt-component ablation dataset."""

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
exp = root / "experiments" / "prompt_component_ablation30"
payload = json.loads((exp / "candidates.json").read_text(encoding="utf-8"))
rows = payload["rows"]
validation = []
for row in rows:
    source_n = len(sent_tokenize(row["source"]))
    attack_n = len(sent_tokenize(row["attack"]))
    validation.append(
        {
            "index": row["index"],
            "strategy": row["strategy"],
            "source_sentences": source_n,
            "attack_sentences": attack_n,
            "sentence_delta": attack_n - source_n,
            "numbers_preserved": numbers(row["source"]) == numbers(row["attack"]),
            "negation_preserved": negations(row["source"]) == negations(row["attack"]),
        }
    )
summary = {}
for strategy in payload["protocol"]["strategies"]:
    selected = [row for row in validation if row["strategy"] == strategy]
    summary[strategy] = {
        "n": len(selected),
        "same_sentence_count": sum(row["sentence_delta"] == 0 for row in selected),
        "fewer_sentences": sum(row["sentence_delta"] < 0 for row in selected),
        "more_sentences": sum(row["sentence_delta"] > 0 for row in selected),
        "numbers_preserved": sum(row["numbers_preserved"] for row in selected),
        "negation_preserved": sum(row["negation_preserved"] for row in selected),
        "sentence_delta_distribution": dict(Counter(row["sentence_delta"] for row in selected)),
    }
(exp / "validation.json").write_text(
    json.dumps({"summary": summary, "rows": validation}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
Dataset.from_dict({"text": [row["attack"] for row in rows]}).save_to_disk(str(exp / "dataset"))
print(json.dumps(summary))
