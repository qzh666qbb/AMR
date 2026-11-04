"""Offline, descriptive diagnostics of already-inspected watermark experiments."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import statistics

from run_segmentation_intervention import ROOT, sha, write_json


def lexical_tokens(text):
    return re.findall(r"\w+(?:['\u2019]\w+)*", text.casefold())


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, token in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (token != other)))
        previous = current
    return previous[-1]


def ngram_retention(source, attack, n):
    def counts(tokens):
        return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    before, after = counts(source), counts(attack)
    total = sum(before.values())
    return sum((before & after).values()) / total if total else None


def text_metrics(source, attack, tokenize):
    left, right = lexical_tokens(source), lexical_tokens(attack)
    before, after = tokenize(source), tokenize(attack)
    return {
        "source_words": len(source.split()), "attack_words": len(attack.split()),
        "source_sentences": len(before), "attack_sentences": len(after),
        "sentence_delta": len(after) - len(before),
        "sentence_count_changed": len(after) != len(before),
        "lexical_edit_fraction": edit_distance(left, right) / max(1, len(left), len(right)),
        "lexical_sequence_identical": left == right,
        **{f"lexical_{n}gram_retention": ngram_retention(left, right, n) for n in (2, 3, 4)},
    }


def distribution(values):
    values = [v for v in values if v is not None]
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)} if values else {"n": 0}


def validate_group(group):
    rows = sorted(group["rows"], key=lambda r: r["candidate"])
    if [r["candidate"] for r in rows] != list(range(4)):
        raise ValueError("Missing or duplicate K=4 candidate slot")
    if any(r["prompt_id"] != group["prompt_id"] for r in rows):
        raise ValueError("Candidate prompt identity mismatch")
    selected = next((r["candidate"] for r in rows if r["quality_valid"]), None)
    if selected != group["no_feedback_candidate"]:
        raise ValueError("Saved no-feedback selection is not first quality-valid candidate")


def summarize(rows):
    fields = ("source_words", "attack_words", "source_sentences", "attack_sentences",
              "sentence_delta", "lexical_edit_fraction", "lexical_2gram_retention",
              "lexical_3gram_retention", "lexical_4gram_retention", "model_4gram_retention",
              "baseline_margin", "attack_margin", "score_delta")
    return {
        "n": len(rows), "quality_valid": sum(r["quality_valid"] for r in rows),
        "sentence_count_changed": sum(r["sentence_count_changed"] for r in rows),
        "sentence_count_reduced": sum(r["sentence_delta"] < 0 for r in rows),
        "lexical_sequence_identical": sum(r["lexical_sequence_identical"] for r in rows),
        "quality_valid_escape": sum(r["quality_valid"] and r["attack_margin"] <= 0 for r in rows),
        "swan_ceiling_below_threshold": sum(r.get("ceiling_below_threshold", False) for r in rows),
        "distributions": {field: distribution([r.get(field) for r in rows]) for field in fields},
    }


def analyze(output):
    import nltk
    from transformers import AutoTokenizer

    inputs = {}

    def load(path):
        inputs[str(path.relative_to(ROOT))] = sha(path)
        return json.loads(path.read_text())

    tokenizer_dir = ROOT / "models/qwen2.5-0.5b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    for path in sorted(tokenizer_dir.glob("*")):
        if path.is_file() and ("token" in path.name or path.suffix == ".json"):
            inputs[str(path.relative_to(ROOT))] = sha(path)
    original = ROOT / "experiments/confirmation_three_arm30"
    transfer = ROOT / "experiments/cross_watermark_transfer30"
    expected_ids = {r["prompt_id"] for r in load(original / "preparation.json")["prompts"]}
    rows = []
    seen = set()
    coverage = Counter()
    planned = Counter()
    for cohort, folder, glob in (
        ("swan", original, "evaluated/*.json"),
        ("sira_adapted", ROOT / "experiments/sira_adapted30", "evaluated/*.json"),
        ("kgw", transfer, "evaluated/kgw/*.json"),
        ("synthid", transfer, "evaluated/synthid/*.json"),
    ):
        for path in sorted(folder.glob(glob)):
            group = load(path)
            validate_group(group)
            pid = group["prompt_id"]
            watermark = "swan" if cohort in ("swan", "sira_adapted") else cohort
            arm = group.get("arm", "sira_adapted" if cohort == "sira_adapted" else "boundary_only")
            label = f"{watermark}/{arm}"
            if (label, pid) in seen or pid not in expected_ids:
                raise ValueError("Duplicate or unexpected evaluated group")
            seen.add((label, pid))
            baseline_path = ((original / "baselines" / f"{pid}.json") if watermark == "swan"
                             else (transfer / "baselines" / watermark / f"{pid}.json"))
            baseline = load(baseline_path)
            threshold = 2.33 if watermark == "swan" else group["threshold"]
            base_score = baseline["detection"]["z" if watermark == "swan" else "score"]
            planned[label] += 1
            coverage[label] += group["no_feedback_candidate"] is not None
            for row in group["rows"]:
                if row["source"] != baseline["text"]:
                    raise ValueError("Candidate source differs from baseline")
                metrics = text_metrics(row["source"], row["attack"], nltk.sent_tokenize)
                score = (row["detection"]["z"] if watermark == "swan"
                         else row["target_detection"]["score"])
                if watermark == "swan" and metrics["attack_sentences"] != row["detection"]["n"]:
                    raise ValueError("Saved detector and current NLTK segmentation differ")
                value = {
                    "watermark": watermark, "arm": arm, "group": label, "prompt_id": pid,
                    "candidate": row["candidate"], "quality_valid": row["quality_valid"],
                    "no_feedback_selected": row["candidate"] == group["no_feedback_candidate"],
                    "threshold": threshold, "baseline_score": base_score, "attack_score": score,
                    "baseline_margin": base_score - threshold, "attack_margin": score - threshold,
                    "score_delta": score - base_score, **metrics,
                }
                if watermark == "swan":
                    value["ceiling_below_threshold"] = math.sqrt(3 * row["detection"]["n"]) <= threshold
                else:
                    left = tokenizer.encode(row["source"], add_special_tokens=False)
                    right = tokenizer.encode(row["attack"], add_special_tokens=False)
                    value["model_4gram_retention"] = ngram_retention(left, right, 4)
                rows.append(value)
    groups = {}
    for label in sorted(planned):
        pool = [r for r in rows if r["group"] == label]
        chosen = [r for r in pool if r["no_feedback_selected"]]
        if (len(pool) != planned[label] * 4 or len(chosen) != coverage[label]
                or {r["prompt_id"] for r in pool} != expected_ids):
            raise ValueError("Missing candidate or selection identity")
        groups[label] = {
            "planned_paragraphs": planned[label], "quality_coverage": coverage[label],
            "all_candidates": summarize(pool), "no_feedback_selected": summarize(chosen),
            "all_sample_no_feedback_valid_asr": sum(
                r["quality_valid"] and r["attack_margin"] <= 0 for r in chosen) / planned[label],
        }
    expected_groups = {"swan/plain_paraphrase", "swan/boundary_only", "swan/full_e2",
                       "swan/sira_adapted", "kgw/boundary_only", "synthid/boundary_only"}
    if set(groups) != expected_groups:
        raise ValueError("Missing or unexpected experimental arm")
    inputs[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    result = {
        "status": "completed_exploratory_reanalysis", "new_api_calls": 0,
        "new_detector_queries": 0, "rows": rows, "groups": groups, "input_sha256": inputs,
        "definitions": {
            "lexical_tokens": "Lowercase Unicode word tokens, retaining internal apostrophes; punctuation excluded",
            "lexical_edit_fraction": "Levenshtein distance / max(source tokens, attack tokens, 1)",
            "ngram_retention": "Multiset intersection / source ngram count; null if source too short",
            "model_4gram_retention": "Local Qwen tokenizer IDs including punctuation; KGW/SynthID only",
            "margin": "score minus own frozen threshold; positive means detected",
            "sentence_count_changed": "Count change, not a full alignment of sentence boundaries",
        },
        "limitations": [
            "Already-inspected cohorts; descriptive exploration, not an independent confirmation.",
            "Selected summaries exclude paragraphs with no quality-valid candidate; ASR denominator does not.",
            "Lexical ngrams are proxies, not the watermark hash contexts or a causal attribution.",
            "Raw score margins have different units across detectors and must not be compared in magnitude.",
            "Generation model, lengths, and watermark mechanisms are confounded across cohorts.",
            "Quality labels retain their saved provenance; API review is not new independent human review.",
        ],
    }
    write_json(output / "diagnostics.json", result)
    print(json.dumps({"rows": len(rows), "groups": groups}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/feedback2_diagnostics")
    args = parser.parse_args()
    os.environ.setdefault("NLTK_DATA", str(ROOT / ".nltk_data"))
    analyze(args.output)


if __name__ == "__main__":
    main()
