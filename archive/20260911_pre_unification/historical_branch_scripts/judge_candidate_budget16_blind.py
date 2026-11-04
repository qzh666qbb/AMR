"""Blind semantic review for the deterministic candidate-budget selections."""

from __future__ import annotations

import argparse
import csv
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


def parse_json(text: str) -> list[dict]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    match = re.search(r"\[.*\]", text, flags=re.S)
    if not match:
        raise ValueError("No JSON array returned")
    result = json.loads(match.group(0))
    if not isinstance(result, list):
        raise ValueError("Judge response is not a list")
    return result


def judge_source(source_index: int, source: str, candidates: list[str], key: str, model: str) -> dict:
    rendered = "\n\n".join(f"CANDIDATE {i}:\n{text}" for i, text in enumerate(candidates))
    prompt = f"""Independently evaluate candidate rewrites against the source. You are not given and must not infer any detector score or candidate-selection method.

SOURCE:
{source}

{rendered}

For each candidate, return:
- meaning_preserved: true only if all source propositions and semantic roles remain and no new proposition is added;
- facts_preserved: true only if entities, numbers, dates, locations, polarity, modality, attribution, events, and relations are unchanged;
- readable: true only if the result is grammatical, coherent natural English rather than an awkward concatenation;
- major_error: true if there is any material omission, addition, contradiction, role swap, or unsupported claim;
- scores meaning, factual, readability as integers 1 to 5;
- one concise reason.

Be strict. Return only a JSON array in candidate order with objects containing candidate, meaning_preserved, facts_preserved, readable, major_error, meaning_score, factual_score, readability_score, reason."""
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 1800,
            "thinking": {"type": "disabled"},
        },
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()
    judged = parse_json(payload["choices"][0]["message"]["content"])
    if len(judged) != len(candidates):
        raise ValueError(f"Expected {len(candidates)} reviews, got {len(judged)}")
    mapped = {int(item["candidate"]): item for item in judged}
    if set(mapped) != set(range(len(candidates))):
        raise ValueError("Candidate identifiers are incomplete")
    return {
        "source_index": source_index,
        "source": source,
        "candidates": candidates,
        "reviews": [mapped[i] for i in range(len(candidates))],
        "model_returned": payload.get("model"),
        "usage": payload.get("usage", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--all-candidates", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    exp = root / "experiments" / "candidate_budget16_pilot30"
    load_dotenv(root / ".env")
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing")

    groups = {}
    if args.all_candidates:
        candidate_rows = json.loads((exp / "rows.json").read_text(encoding="utf-8"))
        for row in candidate_rows:
            source_index = int(row["index"])
            group = groups.setdefault(source_index, {"source": row["source"], "candidates": []})
            group["candidates"].append(row["attack"])
    else:
        table = list(csv.DictReader((exp / "blind_human_review.csv").open(encoding="utf-8-sig")))
        for row in table:
            source_index = int(row["source_index"])
            group = groups.setdefault(source_index, {"source": row["source"], "candidates": []})
            if row["candidate"] not in group["candidates"]:
                group["candidates"].append(row["candidate"])

    stem = "all_candidate_judgments" if args.all_candidates else "blind_judgments"
    partial = exp / f"{stem}.partial.jsonl"
    completed = {}
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                completed[int(item["source_index"])] = item
    pending = [index for index in sorted(groups) if index not in completed]
    lock = threading.Lock()
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                judge_source,
                index,
                groups[index]["source"],
                groups[index]["candidates"],
                key,
                args.model,
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
                failures.append({"source_index": index, "error": f"{type(exc).__name__}: {exc}"})
    output = {
        "protocol": {
            "model_requested": args.model,
            "temperature": 0.0,
            "blinded_to_detector_scores": True,
            "source_groups": len(groups),
            "unique_source_candidate_pairs": sum(len(group["candidates"]) for group in groups.values()),
        },
        "complete": len(completed) == len(groups),
        "groups": [completed[index] for index in sorted(completed)],
        "failures": failures,
    }
    (exp / f"{stem}.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"completed={len(completed)}/{len(groups)} unique_pairs={output['protocol']['unique_source_candidate_pairs']} failures={len(failures)}"
    )
    if failures:
        raise RuntimeError("Some judgments failed; rerun to resume")


if __name__ == "__main__":
    main()
