"""Prepare independent prompt inputs and a three-arm protocol, never generate text.

Raw C4 articles are generation prompts, not watermarked attack inputs.
The API model identity and watermark generation protocol remain to be verified.
"""

import argparse
import ast
import gzip
import json
import os
from pathlib import Path
import sys
import unicodedata

from run_segmentation_intervention import REVISION, ROOT, sha, text_sha, write_json

sys.path.insert(0, str(ROOT / "third_party/SWAN"))
from utils.text_utils import extract_prompt_from_text

SHARD = "realnewslike/c4-validation.00000-of-00001.json.gz"
CHANGE = "Change syntax broadly (sentence boundary, voice, nominalization, clause attachment, reference)."
ARMS = {
    "plain_paraphrase": "Paraphrase naturally using synonymous wording; no deliberate sentence-boundary restructuring is required.",
    "boundary_only": (
        "Change sentence boundaries by naturally merging related sentences or splitting a sentence. "
        "Keep vocabulary, predicates, entities, facts, and event roles as close to the source as possible; "
        "the intended structural change is sentence segmentation."
    ),
    "full_e2": CHANGE,
}


def normalized_hash(text):
    return text_sha(" ".join(unicodedata.normalize("NFKC", text).split()).casefold())


def historical_template():
    path = ROOT / "scripts/run_planE_e2_candidates30.py"
    tree = ast.parse(path.read_text())
    # Do not import the historical script: import would trigger generation.
    value = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "prompt_tpl" for t in node.targets))
    if value.count(CHANGE) != 1 or value.count("{}") != 1:
        raise ValueError("Historical prompt changed; recheck comparison protocol")
    return value


def attack_templates():
    full = historical_template()
    return {arm: full.replace(CHANGE, instruction) + "\nCandidate variant: {candidate}"
            for arm, instruction in ARMS.items()}


def choose_prompts(records, count, excluded_texts):
    seen_text = {normalized_hash(t) for t in excluded_texts}
    seen_prompt = {normalized_hash(extract_prompt_from_text(t, 32)) for t in excluded_texts}
    seen_url = set()
    chosen, rejected = [], []
    for row_index, row in enumerate(records):
        text = row["text"]
        prompt = extract_prompt_from_text(text, 32)
        text_hash, prompt_hash = normalized_hash(text), normalized_hash(prompt)
        url = row.get("url", "")
        duplicate = text_hash in seen_text or prompt_hash in seen_prompt or bool(url and url in seen_url)
        # The first 250 validation records belong to the historical prompt pool.
        if row_index >= 250 and (duplicate or not text.strip() or not prompt.strip()):
            rejected.append({"shard_row": row_index, "reason": "duplicate_or_empty"})
        elif row_index >= 250:
            chosen.append({
                "prompt_id": f"c4-validation-{row_index}", "shard_row": row_index,
                "text": text, "generation_prompt": prompt, "url": url,
                "raw_sha256": text_sha(text), "normalized_sha256": text_hash,
                "generation_prompt_sha256": text_sha(prompt),
                "role": "unwatermarked_generation_prompt_only",
            })
        seen_text.add(text_hash)
        seen_prompt.add(prompt_hash)
        if url:
            seen_url.add(url)
        if len(chosen) == count:
            return chosen, rejected, row_index + 1
    raise ValueError("Not enough independent prompts")


def stream_validation():
    import requests

    url = f"https://huggingface.co/datasets/allenai/c4/resolve/{REVISION}/{SHARD}"
    with requests.get(url, stream=True, timeout=(20, 60)) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        with gzip.GzipFile(fileobj=response.raw) as handle:
            for line in handle:
                yield json.loads(line)


def check(folder):
    path = folder / "preparation.json"
    data = json.loads(path.read_text())
    if data["attack"]["templates"] != attack_templates():
        raise ValueError("Prompt instructions differ from preparation")
    for relative, expected in data["input_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Preparation dependency changed: {relative}")
    records = data["prompts"]
    if len({r["prompt_id"] for r in records}) != len(records):
        raise ValueError("Duplicate prompt identity")
    for row in records:
        if row["role"] != "unwatermarked_generation_prompt_only" or row["shard_row"] < 250:
            raise ValueError("Incorrect input role or historical split overlap")
        if text_sha(row["text"]) != row["raw_sha256"] or normalized_hash(row["text"]) != row["normalized_sha256"]:
            raise ValueError("Prompt content modified")
        prompt = extract_prompt_from_text(row["text"], 32)
        if prompt != row["generation_prompt"] or text_sha(prompt) != row["generation_prompt_sha256"]:
            raise ValueError("Extracted prompt mismatch")
    return {
        "preparation_sha256": sha(path), "prompt_count": len(records),
        "planned_attack_candidates": len(records) * 3 * 4,
        "api_key_present_in_environment": bool(os.environ.get("DEEPSEEK_API_KEY", "").strip()),
        "generation_started": False,
        "ready_for_attack_generation": False,
        "pending": [
            "Verify generation API access and returned model identity",
            "Freeze watermark injection settings; generate or supply provenance-linked SWAN paragraphs",
            "Reparse watermarked baselines and record injection failures without dropping prompts",
            "Freeze watermarked text identities, semantic review implementation, and generation runtime before attacks",
        ],
    }


def prepare(folder, count):
    path = folder / "preparation.json"
    if path.exists():
        raise FileExistsError("Frozen preparation already exists; use check")
    history = ROOT / "experiments/planE_e2_joint250/candidates.json"
    human = ROOT / "experiments/segmentation_intervention/manifest.json"
    excluded = {r["source"] for r in json.loads(history.read_text())["rows"]}
    excluded.update(r["text"] for r in json.loads(human.read_text())["records"])
    stream = stream_validation()
    try:
        prompts, rejected, scanned = choose_prompts(stream, count, excluded)
    finally:
        stream.close()
    dependencies = [
        history, human, Path(__file__), ROOT / "scripts/run_planE_e2_candidates30.py",
        ROOT / "scripts/quality_aware_select_e2_250.py",
        ROOT / "third_party/SWAN/utils/text_utils.py",
        ROOT / "third_party/SWAN/amr_bank/banks/amr_bank_50.json",
    ]
    payload = {
        "status": "prompt_inputs_prepared_not_watermarked_not_fully_frozen",
        "dataset": {"name": "allenai/c4", "revision": REVISION, "shard": SHARD,
                    "split": "validation", "historical_rows_excluded": 250,
                    "scanned_rows": scanned, "selection": "first eligible rows after historical 250"},
        "exclusion": {"checks": ["NFKC whitespace/case-normalized text", "extracted 32-word prompt", "URL within validation stream"],
                      "historical_source_paragraphs": 250,
                      "note": "Identity checks do not prove topic-level or approximate semantic independence",
                      "rejected": rejected},
        "prompts": prompts,
        "attack": {
            "arms": list(ARMS), "k": 4, "templates": attack_templates(),
            "model_requested": "deepseek-v4-flash", "temperature": .7, "max_tokens": 1400,
            "thinking": {"type": "disabled"}, "planned_candidate_calls": count * 3 * 4,
            "planned_max_output_tokens": count * 3 * 4 * 1400,
            "budget_note": "Generation only; watermark rejection sampling and review calls budgeted separately",
            "failures": "Keep failed slots and original denominator; no automatic semantic retries",
            "schedule": "Interleave arms within each paragraph and candidate slot to limit temporal API drift",
            "log_fields": ["request_id", "prompt_sha256", "model_returned", "usage", "finish_reason",
                           "elapsed_seconds", "transport_attempts", "error_type"],
        },
        "evaluation": {
            "quality": "Same score>=4, no major error, number/negation/length/AMR gates in all arms",
            "review_reference": "scripts/quality_aware_select_e2_250.py",
            "review_blinding": "Do not expose arm identity or detector scores; freeze review job mapping before detection",
            "primary": "All prompts denominator, quality then minimum z<=2.33",
            "no_feedback": "First quality-valid candidate by preassigned index; choose before reading z",
            "no_valid_candidate": "Failure; never exclude paragraph from denominator",
            "secondary": "Conditional success on initially detected baselines; report denominator separately",
            "quality_and_detection_cost": "Report token usage, review calls, detector queries and elapsed separately",
            "adaptive_rescue": False,
        },
        "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in dependencies},
    }
    write_json(path, payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "check"))
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/confirmation_three_arm30")
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    if args.stage == "prepare":
        prepare(args.output, args.count)
    status = check(args.output)
    write_json(args.output / "readiness.json", status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
