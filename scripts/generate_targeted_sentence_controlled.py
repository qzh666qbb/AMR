"""Generate controlled target-sentence rewrites and reconstruct paragraphs locally."""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from nltk.tokenize import sent_tokenize


API_URL = "https://api.deepseek.com/chat/completions"
STRATEGIES = ["top_margin", "near_threshold", "random_green"]
TARGET_KEYS = {
    "top_margin": "top_margin_targets",
    "near_threshold": "near_threshold_targets",
    "random_green": "random_green_targets",
}


def load_dotenv(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.lstrip().startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip())


def extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("No JSON object in response")
    return json.loads(match.group(0))


def request_rewrites(row: dict, strategy: str, key: str, model: str, timeout: int) -> dict:
    targets = row[TARGET_KEYS[strategy]]
    target_texts = [item["sentence"] for item in targets]
    prompt = f"""Rewrite the target sentence(s) below for a controlled robustness experiment.

TARGET SENTENCES:
{json.dumps(target_texts, ensure_ascii=False)}

Requirements:
1. Return exactly {len(target_texts)} rewritten sentence(s), one replacement for each input in the same order.
2. Each replacement must remain exactly one sentence. Do not merge or split sentences.
3. Preserve every fact, entity, number, time, location, negation, modality, attribution, event, and semantic role.
4. Substantially change predicate packaging or grammatical realization while remaining natural.
5. Do not add commentary and do not mention watermarking, detection, AMR, or these instructions.

Return only valid JSON: {{"rewrites":["..."]}}
"""
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1000,
            "thinking": {"type": "disabled"},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    parsed = extract_json(payload["choices"][0]["message"]["content"])
    rewrites = parsed.get("rewrites")
    if not isinstance(rewrites, list) or len(rewrites) != len(targets):
        raise ValueError("Rewrite count mismatch")
    if not all(isinstance(item, str) and len(sent_tokenize(item.strip())) == 1 for item in rewrites):
        raise ValueError("Every rewrite must be exactly one sentence")
    if any(rewrite.strip() == original.strip() for rewrite, original in zip(rewrites, target_texts)):
        raise ValueError("At least one target sentence was returned unchanged")

    source_sentences = sent_tokenize(row["source"])
    reconstructed = list(source_sentences)
    for target, rewrite in zip(targets, rewrites):
        position = int(target["position"])
        if reconstructed[position] != target["sentence"]:
            raise ValueError("Target text does not match source sentence at recorded position")
        reconstructed[position] = rewrite.strip()
    return {
        "index": int(row["index"]),
        "strategy": strategy,
        "source": row["source"],
        "target_positions": [int(item["position"]) for item in targets],
        "original_targets": target_texts,
        "rewrites": rewrites,
        "attack": " ".join(reconstructed),
        "minimum_green_flips_to_escape": int(row["minimum_green_flips_to_escape"]),
        "model_returned": payload.get("model"),
        "usage": payload.get("usage", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--model", default="deepseek-v4-flash")
    args = parser.parse_args()
    load_dotenv(args.root / ".env")
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing")

    source = json.loads(
        (args.root / "experiments" / "targeted_sentence_attack" / "targets.json").read_text(
            encoding="utf-8"
        )
    )["rows"]
    eligible = [row for row in source if row["minimum_green_flips_to_escape"] > 0][: args.limit]
    out_dir = args.root / "experiments" / "targeted_sentence_controlled30"
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / "generation.partial.jsonl"
    completed = {}
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if all(
                    rewrite.strip() != original.strip()
                    for rewrite, original in zip(item["rewrites"], item["original_targets"])
                ):
                    completed[(int(item["index"]), item["strategy"])] = item

    tasks = [
        (row, strategy)
        for row in eligible
        for strategy in STRATEGIES
        if (int(row["index"]), strategy) not in completed
    ]
    lock = threading.Lock()
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(request_rewrites, row, strategy, key, args.model, args.timeout): (
                row,
                strategy,
            )
            for row, strategy in tasks
        }
        for future in as_completed(futures):
            row, strategy = futures[future]
            try:
                result = future.result()
                completed[(result["index"], strategy)] = result
                with lock, partial.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            except Exception as exc:
                failures.append(
                    {"index": row["index"], "strategy": strategy, "error": f"{type(exc).__name__}: {exc}"}
                )

    rows = [
        completed[(int(row["index"]), strategy)]
        for row in eligible
        for strategy in STRATEGIES
        if (int(row["index"]), strategy) in completed
    ]
    output = {
        "protocol": {
            "model_requested": args.model,
            "temperature": 0.7,
            "paragraphs": len(eligible),
            "strategies": STRATEGIES,
            "one_api_call_per_paragraph_strategy": True,
            "paragraph_reconstructed_locally": True,
        },
        "complete": len(rows) == len(eligible) * len(STRATEGIES),
        "rows": rows,
        "failures": failures,
    }
    (out_dir / "candidates.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"completed={len(rows)} expected={len(eligible) * len(STRATEGIES)} failures={len(failures)}")
    if failures:
        raise RuntimeError("Some generations failed; rerun to resume")


if __name__ == "__main__":
    main()
