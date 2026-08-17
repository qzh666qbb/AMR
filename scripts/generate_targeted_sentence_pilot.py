"""Generate a 30-paragraph targeted-sentence attack pilot with equal budgets."""

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
STRATEGIES = ["top_margin", "near_threshold", "random_green"]
TARGET_KEYS = {
    "top_margin": "top_margin_targets",
    "near_threshold": "near_threshold_targets",
    "random_green": "random_green_targets",
}


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip())


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("Response did not contain a JSON object")
    return json.loads(match.group(0))


def target_description(row: dict, strategy: str) -> str:
    targets = row[TARGET_KEYS[strategy]]
    return "; ".join(
        f"sentence {int(item['position']) + 1}: {item['sentence']}" for item in targets
    )


def build_prompt(row: dict) -> str:
    return f"""You are preparing controlled paraphrase candidates for a watermark robustness study.

SOURCE PARAGRAPH:
{row['source']}

Create exactly one full-paragraph candidate for each target policy below.

top_margin targets: {target_description(row, 'top_margin')}
near_threshold targets: {target_description(row, 'near_threshold')}
random_green targets: {target_description(row, 'random_green')}

For each policy:
1. Rewrite only the listed target sentence(s). Copy every non-target sentence verbatim and in the same order.
2. Keep exactly the same number of sentences. Do not merge, split, delete, or add sentences.
3. Preserve every fact, entity, number, time, location, negation, modality, attribution, event, and semantic role.
4. In target sentences, substantially change predicate packaging or grammatical realization (for example active/passive, nominalization, subordinate-clause packaging, or argument realization) while remaining natural.
5. Do not mention watermarking, AMR, detection, or these instructions.

Return only valid JSON with exactly these string fields:
{{"top_margin":"...", "near_threshold":"...", "random_green":"..."}}
"""


def request_one(row: dict, key: str, model: str, timeout: int) -> dict:
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": build_prompt(row)}],
            "temperature": 0.7,
            "max_tokens": 2400,
            "thinking": {"type": "disabled"},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    parsed = extract_json(payload["choices"][0]["message"]["content"])
    if set(parsed) != set(STRATEGIES) or not all(isinstance(parsed[key], str) for key in STRATEGIES):
        raise ValueError("Response JSON fields did not match the frozen schema")
    return {
        "index": row["index"],
        "source": row["source"],
        "minimum_green_flips_to_escape": row["minimum_green_flips_to_escape"],
        "targets": {strategy: row[TARGET_KEYS[strategy]] for strategy in STRATEGIES},
        "candidates": parsed,
        "usage": payload.get("usage", {}),
        "model_returned": payload.get("model"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--model", default="deepseek-v4-flash")
    args = parser.parse_args()

    load_dotenv(args.root / ".env")
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")

    targets_path = args.root / "experiments" / "targeted_sentence_attack" / "targets.json"
    target_rows = json.loads(targets_path.read_text(encoding="utf-8"))["rows"]
    eligible = [row for row in target_rows if row["minimum_green_flips_to_escape"] > 0][: args.limit]
    out_dir = args.root / "experiments" / "targeted_sentence_pilot30"
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / "generation.partial.jsonl"
    completed = {}
    if partial_path.exists():
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                completed[int(item["index"])] = item

    lock = threading.Lock()
    failures = []
    pending = [row for row in eligible if int(row["index"]) not in completed]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(request_one, row, key, args.model, args.timeout): row for row in pending
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
                completed[int(result["index"])] = result
                with lock, partial_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            except Exception as exc:
                failures.append({"index": row["index"], "error": f"{type(exc).__name__}: {exc}"})

    records = [completed[int(row["index"])] for row in eligible if int(row["index"]) in completed]
    output = {
        "protocol": {
            "model_requested": args.model,
            "temperature": 0.7,
            "limit": args.limit,
            "strategies": STRATEGIES,
            "same_sentence_count_required": True,
            "non_target_sentences_verbatim_required": True,
        },
        "complete": len(records) == len(eligible),
        "records": records,
        "failures": failures,
    }
    (out_dir / "candidates.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"completed={len(records)} eligible={len(eligible)} failures={len(failures)}")
    if failures:
        raise RuntimeError("Some API generations failed; rerun to resume from checkpoint")


if __name__ == "__main__":
    main()
