"""Export E2 candidate families as Hugging Face datasets for SWAN parsing."""

import json
import argparse
from pathlib import Path

from datasets import Dataset


ROOT = Path(__file__).resolve().parents[1]
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not payload.get("complete"):
        raise RuntimeError("Candidate generation is incomplete")
    args.output.mkdir(parents=True, exist_ok=True)
    for family in ("plain_paraphrase", "sentence_boundary", "amr_guided"):
        records = payload["records"]
        dataset = Dataset.from_dict(
            {
                "text": [record["candidates"][family] for record in records],
                "source_index": [record["source_index"] for record in records],
            }
        )
        destination = args.output / family
        dataset.save_to_disk(str(destination))
        print(f"{family}: {len(dataset)} -> {destination}")


if __name__ == "__main__":
    main()
