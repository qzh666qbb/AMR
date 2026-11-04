"""Generate strict repair candidates using independent audit feedback for all 30 paragraphs."""

from __future__ import annotations

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


def extract(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("No JSON object returned")
    payload = json.loads(match.group(0))
    if not isinstance(payload.get("atomic_units"), list):
        raise ValueError("atomic_units missing")
    if not isinstance(payload.get("candidates"), list) or len(payload["candidates"]) != 4:
        raise ValueError("Expected four candidates")
    if not all(isinstance(candidate, str) and candidate.strip() for candidate in payload["candidates"]):
        raise ValueError("Candidates must be nonempty strings")
    return payload


def request_one(row: dict, key: str) -> dict:
    prompt = f"""Repair a failed rewrite for a strict semantic-preservation experiment.

SOURCE:
{row['source']}

PREVIOUS FAILED CANDIDATE:
{row['attack']}

INDEPENDENT AUDIT FAILURE:
{row['assistant_reason']}

Generate exactly four new candidates. The source may be repetitive, contradictory, vague, imperative, or unnatural. Do not improve its logic and do not normalize it with world knowledge.

Required procedure:
1. List every distinct atomic unit in SOURCE, including speech-act type (question, command, or statement), tense, modality, polarity, participant roles, vague placeholders, time and location.
2. Exact duplicate units may appear once. Units that differ in tense, role, object, location, specificity, modality, or speech act are not duplicates and must remain separately represented.
3. Preserve placeholders such as "someone", "something", "a location", and "an entity" without forcing coreference between separate occurrences.
4. Never infer an industry, identity, intention, causal link, emotional reaction, audience, responsibility, transaction, or real-world fact.
5. Candidate 1 must be a minimal boundary/coordination edit with nearly all original content words.
6. Candidate 2 may reorder clauses but must preserve each unit's speech act.
7. Candidate 3 may use active/passive alternation only where participant roles remain identical.
8. Candidate 4 may use cautious nominalization but must preserve every unit.
9. Before returning, verify each candidate unit-by-unit against the list. Naturalness is secondary to exact semantic fidelity.

Return JSON only: {{"atomic_units":["..."],"candidates":["...","...","...", "..."]}}"""
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.15,
            "max_tokens": 4200,
            "thinking": {"type": "disabled"},
        },
        timeout=300,
    )
    response.raise_for_status()
    body = response.json()
    parsed = extract(body["choices"][0]["message"]["content"])
    return {
        "index": int(row["index"]),
        "source": row["source"],
        "previous_candidate": row["attack"],
        "audit_failure": row["assistant_reason"],
        "atomic_units": parsed["atomic_units"],
        "candidates": [candidate.strip() for candidate in parsed["candidates"]],
        "model_returned": body.get("model"),
        "usage": body.get("usage", {}),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    exp = root / "experiments" / "candidate_budget16_pilot30"
    out = exp / "feedback_driven_strict30"
    out.mkdir(parents=True, exist_ok=True)
    load_dotenv(root / ".env")
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing")
    audit = json.loads((exp / "final_30_assistant_audit.json").read_text(encoding="utf-8"))["rows"]
    partial = out / "generation.partial.jsonl"
    completed = {}
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                completed[int(item["index"])] = item
    pending = [row for row in audit if int(row["index"]) not in completed]
    failures = []
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(request_one, row, key): int(row["index"]) for row in pending}
        for future in as_completed(futures):
            index = futures[future]
            try:
                item = future.result()
                completed[index] = item
                with lock, partial.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                print(f"completed={len(completed)}/30", flush=True)
            except Exception as exc:
                failures.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
    payload = {
        "protocol": {
            "paragraphs": 30,
            "candidates_per_paragraph": 4,
            "temperature": 0.15,
            "uses_independent_audit_feedback": True,
        },
        "complete": len(completed) == 30,
        "records": [completed[index] for index in sorted(completed)],
        "failures": failures,
    }
    (out / "candidates.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"completed={len(completed)}/30 failures={len(failures)}")
    if failures:
        raise RuntimeError("Some generations failed; rerun to resume")


if __name__ == "__main__":
    main()
