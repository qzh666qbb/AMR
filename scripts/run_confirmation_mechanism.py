"""Prepare the frozen resegmentation and punctuation-only mechanism cohort."""

import argparse
import json
import os
from pathlib import Path
import re

from prepare_confirmation_experiment import ARMS
from run_segmentation_intervention import ROOT, partition, segment, sha, text_sha, write_json


def lexical_tokens(text):
    return re.findall(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)*", text)


def remove_terminal_punctuation(sentence):
    match = re.search(r'([.!?]+)(["\')\]]*)$', sentence)
    if not match:
        raise ValueError("Sentence has no terminal punctuation")
    return sentence[:match.start()] + match.group(2)


def punctuation_merge(sentences):
    if len(sentences) < 2:
        raise ValueError("Punctuation control requires multiple sentences")
    merged = "; ".join(remove_terminal_punctuation(s) for s in sentences[:-1])
    merged += "; " + sentences[-1]
    if lexical_tokens(merged) != lexical_tokens(" ".join(sentences)):
        raise ValueError("Punctuation merge changed lexical tokens")
    return merged


def freeze(path, payload):
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f"Frozen artifact differs: {path}")
    else:
        write_json(path, payload)


def prepare(output):
    import nltk
    import spacy

    source = ROOT / "experiments/confirmation_three_arm30"
    final_path = source / "final_analysis.json"
    final = json.loads(final_path.read_text())
    if final["status"] != "complete_independently_verified":
        raise ValueError("Confirmation cohort is not final")
    calibration_path = ROOT / "experiments/segmentation_intervention/summary.json"
    calibration = json.loads(calibration_path.read_text())
    if calibration["status"] != "complete_calibrated":
        raise ValueError("Segmentation calibration is incomplete")
    thresholds = {
        mode: calibration["calibrated"][mode]["threshold"]
        for mode in ("sentence", "clause")
    }
    nlp = spacy.load("en_core_web_sm")
    records = []
    dependencies = {
        "confirmation_final_analysis": sha(final_path),
        "confirmation_runtime": sha(source / "runtime.json"),
        "segmentation_summary": sha(calibration_path),
        "segmentation_manifest": sha(ROOT / "experiments/segmentation_intervention/manifest.json"),
        "prepare_script": sha(Path(__file__)),
    }

    for path in sorted((source / "baselines").glob("*.json")):
        baseline = json.loads(path.read_text())
        pid = baseline["prompt_id"]
        dependencies[str(path.relative_to(ROOT))] = sha(path)
        original = [row["text"] for row in baseline["detection"]["details"]]
        if len(original) != 5:
            raise ValueError("Expected five-sentence confirmation baseline")
        merged = punctuation_merge(original)
        if len(nltk.sent_tokenize(merged)) != 1:
            raise ValueError("Punctuation merge did not produce one NLTK sentence")
        merged_segments = segment(merged, nltk.sent_tokenize, nlp)
        exact_clause = partition(
            merged, [match.end() for match in re.finditer(";", merged)], min_words=1)
        if len(exact_clause) != len(original):
            raise ValueError("Exact semicolon resegmentation did not restore five units")
        if any(lexical_tokens(text) != lexical_tokens(" ".join(original))
               for text in (baseline["text"], merged, " ".join(merged_segments["clause"]),
                            " ".join(exact_clause))):
            raise ValueError("Punctuation control changed lexical content")
        records.append({
            "id": f"punctuation-{pid}", "cohort": "attack",
            "kind": "punctuation_control", "prompt_id": pid,
            "text": merged, "text_sha256": text_sha(merged),
            "baseline_text_sha256": baseline["text_sha256"],
            "lexical_sha256": text_sha(json.dumps(lexical_tokens(merged))),
            "historical_baseline": {k: baseline["detection"][k] for k in ("n", "green", "z")},
            "segments": {
                "original": original,
                "merged_sentence": [merged],
                "merged_rule_clause": merged_segments["clause"],
                "merged_exact_clause": exact_clause,
            },
        })

    for arm in ARMS:
        for path in sorted((source / "evaluated").glob(f"*-{arm}.json")):
            group = json.loads(path.read_text())
            dependencies[str(path.relative_to(ROOT))] = sha(path)
            selected = {}
            for role in ("feedback", "no_feedback"):
                candidate = group[f"{role}_candidate"]
                if candidate is not None:
                    selected.setdefault(candidate, []).append(role)
            for candidate, roles in sorted(selected.items()):
                row = group["rows"][candidate]
                if not row["quality_valid"]:
                    raise ValueError("Selected candidate is not quality-valid")
                variants = segment(row["attack"], nltk.sent_tokenize, nlp)
                if variants["sentence"] != [d["text"] for d in row["detection"]["details"]]:
                    raise ValueError("Fresh sentence segmentation differs from saved detection")
                records.append({
                    "id": f"selected-{group['prompt_id']}-{arm}-{candidate}",
                    "cohort": "attack", "kind": "selected_attack",
                    "prompt_id": group["prompt_id"], "arm": arm,
                    "candidate": candidate, "roles": sorted(roles),
                    "text": row["attack"], "text_sha256": text_sha(row["attack"]),
                    "source_sha256": row["source_sha256"],
                    "historical_detection": {
                        k: row["detection"][k] for k in ("n", "green", "z")
                    },
                    "segments": variants,
                })

    counts = {
        "punctuation_control": sum(r["kind"] == "punctuation_control" for r in records),
        "selected_attack": sum(r["kind"] == "selected_attack" for r in records),
    }
    if counts != {"punctuation_control": 30, "selected_attack": 104}:
        raise ValueError(f"Unexpected frozen cohort counts: {counts}")
    manifest = {
        "protocol": {
            "version": 1,
            "purpose": "Separate sentence-count mechanics from AMR match-rate changes",
            "punctuation_control": (
                "Replace only terminal punctuation between each baseline's five "
                "sentences with semicolons; preserve lexical token sequence"
            ),
            "candidate_selection": (
                "Union of frozen feedback and no-feedback quality-valid selections; "
                "no new candidate generation or reselection"
            ),
            "segmentation": (
                "NLTK punkt_tab sentences; clause mode splits semicolons and verbal "
                "conjuncts with explicit subjects; minimum four words"
            ),
            "exact_semicolon_control": (
                "Split at every inserted semicolon with minimum one word; this restores "
                "five units even when the deployable four-word rule does not"
            ),
            "fixed_threshold": 2.33,
            "calibrated_thresholds": thresholds,
            "calibration_source": (
                "Existing independent 250-human calibration and 250-human test "
                "cohorts from segmentation_intervention"
            ),
            "s2match_randomness": (
                "Fresh sentence/clause scores are paired within this run; saved "
                "historical score mismatches are reported, not overwritten"
            ),
            "spacy": spacy.__version__, "spacy_model": nlp.meta["version"],
            "nltk": nltk.__version__, "record_counts": counts,
        },
        "records": records,
        "dependencies": dependencies,
    }
    freeze(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output / "manifest.json"),
                      "records": len(records), **counts}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments/confirmation_mechanism30")
    args = parser.parse_args()
    os.environ.setdefault("NLTK_DATA", str(ROOT / ".nltk_data"))
    prepare(args.output)


if __name__ == "__main__":
    main()
