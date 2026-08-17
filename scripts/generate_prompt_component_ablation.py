"""Generate an equal-budget 30-paragraph prompt-component ablation."""

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
STRATEGIES = ["weak", "no_boundary", "boundary_only", "full_e2"]

INSTRUCTIONS = {
    "weak": "Paraphrase the paragraph naturally using synonymous wording. Preserve the meaning and facts.",
    "no_boundary": (
        "Substantially rewrite predicate packaging and argument realization using active/passive alternation, "
        "nominalization, subordinate clauses, or grammatical-role changes. Keep exactly the same number of "
        "sentences and do not merge or split sentences."
    ),
    "boundary_only": (
        "Change sentence boundaries by naturally merging related sentences or, where appropriate, splitting a "
        "sentence. Keep vocabulary, predicates, entities, facts, and event roles as close to the source as possible; "
        "the intended structural change is sentence segmentation."
    ),
    "full_e2": (
        "Perform a strong whole-paragraph structural rewrite. Jointly vary sentence boundaries, predicate packaging, "
        "nominalization, argument realization, clause structure, and coreference while preserving the complete meaning."
    ),
}


def load_dotenv(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.lstrip().startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip())


def clean_response(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:text)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    for prefix in ["Paraphrase:", "Rewritten paragraph:", "Rewrite:"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
    if not text:
        raise ValueError("Empty candidate")
    return text


def request_one(source: str, index: int, strategy: str, key: str, model: str, timeout: int) -> dict:
    prompt = f"""Rewrite the source paragraph for a controlled robustness experiment.

SOURCE:
{source}

ATTACK CONDITION:
{INSTRUCTIONS[strategy]}

For every condition, preserve all entities, numbers, dates, locations, negation, modality, attribution, events, and semantic roles. Do not add or omit facts. Produce natural English. Do not mention watermarking, detection, AMR, or these instructions.

Return only the rewritten paragraph with no label or commentary.
"""
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1600,
            "thinking": {"type": "disabled"},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "index": index,
        "strategy": strategy,
        "source": source,
        "attack": clean_response(payload["choices"][0]["message"]["content"]),
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

    candidates = json.loads(
        (args.root / "experiments" / "planE_e2_joint250" / "candidates.json").read_text(
            encoding="utf-8"
        )
    )["rows"]
    source_by_index = {}
    for row in candidates:
        source_by_index[int(row["index"])] = row["source"]
    indices = sorted(source_by_index)[: args.limit]

    out_dir = args.root / "experiments" / "prompt_component_ablation30"
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / "generation.partial.jsonl"
    completed = {}
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                completed[(int(item["index"]), item["strategy"])] = item

    tasks = [
        (index, strategy)
        for index in indices
        for strategy in STRATEGIES
        if (index, strategy) not in completed
    ]
    lock = threading.Lock()
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                request_one, source_by_index[index], index, strategy, key, args.model, args.timeout
            ): (index, strategy)
            for index, strategy in tasks
        }
        for future in as_completed(futures):
            index, strategy = futures[future]
            try:
                result = future.result()
                completed[(index, strategy)] = result
                with lock, partial.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            except Exception as exc:
                failures.append(
                    {"index": index, "strategy": strategy, "error": f"{type(exc).__name__}: {exc}"}
                )

    rows = [
        completed[(index, strategy)]
        for index in indices
        for strategy in STRATEGIES
        if (index, strategy) in completed
    ]
    output = {
        "protocol": {
            "model_requested": args.model,
            "temperature": 0.7,
            "paragraphs": len(indices),
            "strategies": STRATEGIES,
            "one_call_per_paragraph_strategy": True,
            "instructions": INSTRUCTIONS,
        },
        "complete": len(rows) == len(indices) * len(STRATEGIES),
        "rows": rows,
        "failures": failures,
    }
    (out_dir / "candidates.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"completed={len(rows)} expected={len(indices) * len(STRATEGIES)} failures={len(failures)}")
    if failures:
        raise RuntimeError("Some generations failed; rerun to resume")


if __name__ == "__main__":
    main()
