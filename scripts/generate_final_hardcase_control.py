"""Generate two controlled rescue strategies for the final hard cases 6 and 16."""

import json
import os
import re
from pathlib import Path

import requests


API_URL = "https://api.deepseek.com/chat/completions"
INDICES = (6, 16)


def load_dotenv(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def request(source: str, key: str) -> tuple[dict, dict]:
    prompt = f"""Produce two controlled sets of rewrites for the source below.

SOURCE:
{source}

The source is intentionally vague and awkward. Its wording does not license any world-knowledge inference. Preserve every distinct proposition and every vague placeholder literally enough that its vagueness remains unchanged.

SET A — literal_boundary_only, exactly 8 candidates:
- Keep the original content words and predicates as much as possible.
- Only change punctuation, coordination, clause order, sentence boundaries, and repeated-subject realization.
- Do not replace vague phrases with interpretations.
- Do not add adjectives, explanations, emotions, industries, responsibilities, intentions, audiences, or causal relations.

SET B — constrained_structural, exactly 8 candidates:
- Allow active/passive alternation, nominalization, predicate packaging, and coreference.
- Still preserve every proposition and vague phrase without interpreting it.
- Do not add or omit information.

Repeated sentences expressing the identical proposition may be represented once. Similar but non-identical propositions must each remain represented. Return JSON only:
{{"literal_boundary_only":["8 strings"],"constrained_structural":["8 strings"]}}"""
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.25,
            "max_tokens": 5000,
            "thinking": {"type": "disabled"},
        },
        timeout=300,
    )
    response.raise_for_status()
    body = response.json()
    text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        body["choices"][0]["message"]["content"].strip(),
        flags=re.I | re.S,
    )
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("No JSON returned")
    payload = json.loads(match.group(0))
    for strategy in ("literal_boundary_only", "constrained_structural"):
        if not isinstance(payload.get(strategy), list) or len(payload[strategy]) != 8:
            raise ValueError(f"{strategy} must contain eight candidates")
    return payload, body


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    exp = root / "experiments" / "candidate_budget16_pilot30"
    out = exp / "final_hardcase_control"
    out.mkdir(parents=True, exist_ok=True)
    load_dotenv(root / ".env")
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing")
    records = json.loads((exp / "candidates.json").read_text(encoding="utf-8"))["records"]
    sources = {int(record["index"]): record["source"] for record in records}
    output = []
    for index in INDICES:
        payload, body = request(sources[index], key)
        output.append(
            {
                "index": index,
                "source": sources[index],
                **payload,
                "model_returned": body.get("model"),
                "usage": body.get("usage", {}),
            }
        )
        print(f"generated index={index}", flush=True)
    (out / "candidates.json").write_text(
        json.dumps(
            {"protocol": {"indices": INDICES, "candidates_per_strategy": 8}, "records": output},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
