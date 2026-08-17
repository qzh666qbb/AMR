"""Generate fact-constrained rescue candidates for the four true hard cases."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import requests


API_URL = "https://api.deepseek.com/chat/completions"
HARD_INDICES = (6, 14, 16, 20)


def load_dotenv(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.lstrip().startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip())


def extract(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("No JSON object returned")
    payload = json.loads(match.group(0))
    if not isinstance(payload.get("atomic_claims"), list):
        raise ValueError("atomic_claims missing")
    if not isinstance(payload.get("candidates"), list) or len(payload["candidates"]) != 8:
        raise ValueError("Expected eight candidates")
    return payload


def request(source: str, key: str) -> tuple[dict, dict]:
    prompt = f"""Create eight conservative rewrites for a semantic-preservation robustness study.

SOURCE:
{source}

First list every distinct atomic claim explicitly stated by the source. Then write exactly eight candidates.

Hard constraints:
1. Every candidate must cover every distinct source claim, event, participant role, relation, attribution, modality, polarity, entity, number and date.
2. Do not use world knowledge. Do not infer what an organization does, what a role means, who "New York" denotes, or what "something" refers to.
3. Preserve deliberately vague expressions such as "something", "an idea", "one Aylesbury", "define as something", and "is something about" without making them more specific.
4. Repeated source sentences may be represented once if they express exactly the same proposition; non-identical variants must remain represented.
5. Do not add emotional reactions, intentions, causal links, responsibility, accountability, commitments, business domains, audiences, or locations absent from the source.
6. Prefer conservative clause reordering, coordination, passive/active alternation, and careful sentence merging. Natural English is required.
7. After drafting, silently verify each candidate against the atomic-claim list and discard any candidate with an omission or addition.

Return only JSON: {{"atomic_claims":["..."],"candidates":["..." eight strings]}}"""
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 4500,
            "thinking": {"type": "disabled"},
        },
        timeout=300,
    )
    response.raise_for_status()
    body = response.json()
    return extract(body["choices"][0]["message"]["content"]), body


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    exp = root / "experiments" / "candidate_budget16_pilot30"
    out = exp / "fact_constrained_rescue"
    out.mkdir(parents=True, exist_ok=True)
    load_dotenv(root / ".env")
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing")
    records = json.loads((exp / "candidates.json").read_text(encoding="utf-8"))["records"]
    sources = {int(record["index"]): record["source"] for record in records}
    result = []
    for index in HARD_INDICES:
        payload, body = request(sources[index], key)
        result.append(
            {
                "index": index,
                "source": sources[index],
                "atomic_claims": payload["atomic_claims"],
                "candidates": payload["candidates"],
                "model_returned": body.get("model"),
                "usage": body.get("usage", {}),
            }
        )
        print(f"completed index={index}", flush=True)
    (out / "candidates.json").write_text(
        json.dumps(
            {
                "protocol": {"hard_indices": HARD_INDICES, "candidates_per_paragraph": 8},
                "records": result,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
