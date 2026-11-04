"""Generate sixteen same-prompt E2 candidates per paragraph for budget ablation."""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


API_URL = "https://api.deepseek.com/chat/completions"


def load_dotenv(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.lstrip().startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip())


def extract_candidates(text: str) -> list[str]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("No JSON object returned")
    payload = json.loads(match.group(0))
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 16:
        raise ValueError("Expected exactly sixteen candidates")
    if not all(isinstance(item, str) and item.strip() for item in candidates):
        raise ValueError("Candidates must be nonempty strings")
    if len(set(item.strip() for item in candidates)) != 16:
        raise ValueError("Candidates must be distinct")
    return [item.strip() for item in candidates]


def request_one(index: int, source: str, key: str, model: str, timeout: int) -> dict:
    prompt = f"""Generate exactly 16 distinct whole-paragraph rewrites for a controlled robustness experiment.

SOURCE:
{source}

For every candidate:
1. Preserve every fact, entity, number, date, location, negation, modality, attribution, event, and semantic role.
2. Produce natural English with no added or omitted information.
3. Strongly vary whole-paragraph structure using different combinations of sentence merging, sentence splitting, predicate packaging, nominalization, argument realization, clause structure, and coreference.
4. Make the sixteen candidates structurally diverse rather than minor lexical variants.
5. Do not mention watermarking, detection, AMR, or these instructions.

Return only valid JSON: {{"candidates":["...", "...", ...]}}
"""
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.9,
            "max_tokens": 7000,
            "thinking": {"type": "disabled"},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "index": index,
        "source": source,
        "candidates": extract_candidates(payload["choices"][0]["message"]["content"]),
        "model_returned": payload.get("model"),
        "usage": payload.get("usage", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--model", default="deepseek-v4-flash")
    args = parser.parse_args()
    load_dotenv(args.root / ".env")
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing")

    existing = json.loads(
        (args.root / "experiments" / "planE_e2_joint250" / "candidates.json").read_text(
            encoding="utf-8"
        )
    )["rows"]
    source_by_index = {}
    for row in existing:
        source_by_index[int(row["index"])] = row["source"]
    indices = sorted(source_by_index)[: args.limit]

    out_dir = args.root / "experiments" / "candidate_budget16_pilot30"
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / "generation.partial.jsonl"
    completed = {}
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                completed[int(item["index"])] = item
    pending = [index for index in indices if index not in completed]
    failures = []
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                request_one, index, source_by_index[index], key, args.model, args.timeout
            ): index
            for index in pending
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                item = future.result()
                completed[index] = item
                with lock, partial.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            except Exception as exc:
                failures.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})

    records = [completed[index] for index in indices if index in completed]
    output = {
        "protocol": {
            "model_requested": args.model,
            "temperature": 0.9,
            "paragraphs": len(indices),
            "candidates_per_paragraph": 16,
            "one_call_per_paragraph": True,
        },
        "complete": len(records) == len(indices),
        "records": records,
        "failures": failures,
    }
    (out_dir / "candidates.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"completed={len(records)} expected={len(indices)} failures={len(failures)}")
    if failures:
        raise RuntimeError("Some generations failed; rerun to resume")


if __name__ == "__main__":
    main()
