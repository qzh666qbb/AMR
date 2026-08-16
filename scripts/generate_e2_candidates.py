"""Generate checkpointed candidates for the 50-paragraph E2 attack pilot."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk


ROOT = Path(__file__).resolve().parents[1]
SWAN = ROOT / "third_party" / "SWAN"
sys.path.insert(0, str(SWAN))

from detection.detection_utils import find_fpr_thresholds_amr  # noqa: E402
from utils.bedrock_utils import BedrockManager  # noqa: E402


SYSTEM = """You are generating controlled research candidates for a robustness
audit. Preserve every fact, entity, number, date, polarity, modality, and
attribution. Do not add explanations. Return one valid JSON object with exactly
the keys plain_paraphrase, sentence_boundary, and amr_guided."""


def extract_json(text: str) -> dict[str, str]:
    text = text.strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object returned")
    payload = json.loads(match.group(0))
    keys = {"plain_paraphrase", "sentence_boundary", "amr_guided"}
    if set(payload) != keys or not all(isinstance(payload[k], str) for k in keys):
        raise ValueError("Candidate JSON has the wrong schema")
    return {key: payload[key].strip() for key in keys}


def surface_checks(source: str, candidate: str) -> dict[str, bool]:
    numbers = set(re.findall(r"\b\d[\d,.:/%-]*\b", source))
    source_negation = bool(re.search(r"\b(?:no|not|never|without|n't)\b", source, re.I))
    candidate_negation = bool(
        re.search(r"\b(?:no|not|never|without|n't)\b", candidate, re.I)
    )
    return {
        "nonempty": bool(candidate.strip()),
        "numbers_preserved": numbers.issubset(set(re.findall(r"\b\d[\d,.:/%-]*\b", candidate))),
        "negation_preserved": source_negation == candidate_negation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "baselines/local/swan_acl2026_realnews/runs/formal_250x5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/e2_attack_pilot/candidates.json",
    )
    args = parser.parse_args()

    dataset = load_from_disk(str(args.run_dir))
    texts = dataset["text"]
    machine_z = np.load(args.run_dir / "detection/machine_z_scores.npy")
    human_z = np.load(args.run_dir / "detection/human_z_scores.npy")
    threshold_1, _ = find_fpr_thresholds_amr(human_z)
    selected = [i for i, score in enumerate(machine_z) if score > threshold_1][
        : args.limit
    ]
    if len(selected) < args.limit:
        raise RuntimeError(f"Only {len(selected)} baseline-positive paragraphs")

    with (args.run_dir / "parsed_amrs.json").open(encoding="utf-8") as handle:
        parsed = json.load(handle)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if args.output.exists():
        with args.output.open(encoding="utf-8") as handle:
            records = json.load(handle)["records"]
    completed = {record["source_index"] for record in records}
    manager = BedrockManager(region="us-east-1", model_id=args.model)

    for index in selected:
        if index in completed:
            continue
        source = texts[index]
        amr_excerpt = "\n\n".join(parsed[index])
        prompt = f"""Rewrite the source paragraph in three controlled ways.

SOURCE:
{source}

PARSER AMRS:
{amr_excerpt}

Requirements:
1. plain_paraphrase: ordinary meaning-preserving paraphrase; do not use AMR.
2. sentence_boundary: preserve all content but split, merge, or redistribute
   modifiers across sentence boundaries.
3. amr_guided: use the AMRs to change predicate packaging, clause attachment,
   nominalization/verbalization, control/raising, or referential structure
   while preserving meaning. Aim for a fresh parser structure.
4. Keep approximately the same length and output only the JSON object."""
        candidates = None
        format_error = None
        for format_attempt in range(3):
            raw = manager.generate(
                user_text=prompt,
                system_text=SYSTEM,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=0.9,
            )
            try:
                candidates = extract_json(raw)
                break
            except (ValueError, json.JSONDecodeError) as exc:
                format_error = exc
                prompt += (
                    "\n\nYour previous response violated the JSON-only schema. "
                    "Return only one valid JSON object with the three required keys."
                )
        if candidates is None:
            raise RuntimeError(
                f"Invalid JSON after 3 format attempts for source {index}: {format_error}"
            )
        records.append(
            {
                "source_index": index,
                "baseline_z": float(machine_z[index]),
                "source": source,
                "candidates": candidates,
                "surface_checks": {
                    name: surface_checks(source, candidate)
                    for name, candidate in candidates.items()
                },
            }
        )
        temporary = args.output.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "model": args.model,
                    "temperature": args.temperature,
                    "selection_threshold_1pct_fpr": threshold_1,
                    "limit": args.limit,
                    "complete": False,
                    "records": records,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        temporary.replace(args.output)
        print(f"checkpoint {len(records)}/{args.limit}: source {index}", flush=True)

    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": args.model,
                "temperature": args.temperature,
                "selection_threshold_1pct_fpr": threshold_1,
                "limit": args.limit,
                "complete": True,
                "records": records,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
