"""Prepare and optionally detect count-neutral SWAN boundary exchanges."""

import argparse
import json
import os
from pathlib import Path
import re

from run_segmentation_intervention import ROOT, sha, text_sha, write_json


SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "expl"}
COORDINATORS = {"and", "but", "or", "yet", "so"}
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)*")


def lexical_sequence(text):
    """Return case-insensitive lexical tokens while ignoring punctuation."""
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def capitalize_initial(text):
    for index, character in enumerate(text):
        if character.isalpha():
            return text[:index] + character.upper() + text[index + 1:]
    return text


def clause_profile(text, nlp):
    doc = nlp(text)
    predicates = [
        token for token in doc
        if token.pos_ in {"VERB", "AUX"} and token.dep_ == "ROOT"
    ]
    subjects = [
        token for token in doc
        if token.dep_ in SUBJECT_DEPS
    ]
    content = {
        token.lemma_.casefold()
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
        and token.pos_ not in {"DET", "PRON"}
    }
    return {
        "finite_root": bool(predicates),
        "explicit_subject": bool(subjects),
        "content_lemmas": sorted(content),
    }


def split_candidate(sentence, kind, separator_start, right_start, nlp, tokenize):
    left = sentence[:separator_start].rstrip(" ,;")
    right = capitalize_initial(sentence[right_start:].lstrip())
    if len(lexical_sequence(left)) < 3 or len(lexical_sequence(right)) < 3:
        return None
    parts = [left + ".", right]
    transformed = " ".join(parts)
    if lexical_sequence(transformed) != lexical_sequence(sentence):
        raise ValueError("Clause split changed lexical sequence")
    if tokenize(transformed) != parts:
        return None
    profiles = [clause_profile(part, nlp) for part in parts]
    if not all(profile["finite_root"] and profile["explicit_subject"]
               for profile in profiles):
        return None
    return {
        "kind": kind,
        "separator_start": separator_start,
        "right_start": right_start,
        "parts": parts,
        "text": transformed,
    }


def find_split_candidates(sentence, nlp, tokenize):
    """Find punctuation-only splits at explicit independent clauses."""
    if tokenize(sentence) != [sentence]:
        raise ValueError("Expected one source sentence")
    doc = nlp(sentence)
    candidates = []

    for token in doc:
        if token.text == ";":
            candidate = split_candidate(
                sentence, "semicolon", token.idx, token.idx + len(token.text),
                nlp, tokenize,
            )
            if candidate is not None:
                candidates.append(candidate)

        if token.dep_ != "conj" or token.pos_ not in {"VERB", "AUX"}:
            continue
        if not any(child.dep_ in SUBJECT_DEPS for child in token.children):
            continue
        coordinators = [
            candidate for candidate in doc
            if candidate.dep_ == "cc"
            and candidate.text.casefold() in COORDINATORS
            and candidate.i < token.i
            and candidate.head in {token, token.head}
        ]
        coordinators = sorted(coordinators, key=lambda row: row.i)[-1:]
        for coordinator in coordinators:
            separator_start = coordinator.idx
            prefix = sentence[:separator_start].rstrip()
            if prefix.endswith(","):
                separator_start = len(prefix) - 1
            candidate = split_candidate(
                sentence, "explicit_subject_coordination",
                separator_start, coordinator.idx, nlp, tokenize,
            )
            if candidate is not None:
                candidates.append(candidate)

    unique = {}
    for candidate in candidates:
        unique.setdefault(candidate["text"], candidate)
    return sorted(
        unique.values(),
        key=lambda row: (row["separator_start"], row["kind"], row["text"]),
    )


def find_merge_candidates(sentences, nlp):
    """Find adjacent related declaratives that can be joined by a semicolon."""
    profiles = [clause_profile(sentence, nlp) for sentence in sentences]
    candidates = []
    for index, (left, right) in enumerate(zip(sentences, sentences[1:])):
        if not left.endswith(".") or not right.endswith("."):
            continue
        if len(lexical_sequence(left)) < 3 or len(lexical_sequence(right)) < 3:
            continue
        left_profile, right_profile = profiles[index:index + 2]
        if not (
            left_profile["finite_root"] and left_profile["explicit_subject"]
            and right_profile["finite_root"] and right_profile["explicit_subject"]
        ):
            continue
        shared = sorted(
            set(left_profile["content_lemmas"])
            & set(right_profile["content_lemmas"])
        )
        if not shared:
            continue
        merged = left[:-1] + "; " + right
        if lexical_sequence(merged) != lexical_sequence(left + " " + right):
            raise ValueError("Clause merge changed lexical sequence")
        candidates.append({
            "boundary_index": index,
            "shared_content_lemmas": shared,
            "text": merged,
        })
    return candidates


def build_exchange(sentences, merge, split, tokenize):
    merge_index = merge["boundary_index"]
    split_index = split["sentence_index"]
    if split_index in {merge_index, merge_index + 1}:
        raise ValueError("Merge and split must affect different source sentences")

    units = []
    index = 0
    while index < len(sentences):
        if index == merge_index:
            units.append(merge["text"])
            index += 2
            continue
        if index == split_index:
            units.extend(split["parts"])
        else:
            units.append(sentences[index])
        index += 1

    text = " ".join(units)
    original_text = " ".join(sentences)
    if lexical_sequence(text) != lexical_sequence(original_text):
        raise ValueError("Boundary exchange changed lexical sequence")
    if len(units) != len(sentences):
        raise ValueError("Boundary exchange changed the unit count")
    if tokenize(text) != units:
        raise ValueError("Frozen tokenizer does not recover exchange units")
    return {
        "merge_boundary_index": merge_index,
        "split_sentence_index": split_index,
        "split_kind": split["kind"],
        "units": units,
        "text": text,
        "text_sha256": text_sha(text),
        "lexical_sha256": text_sha(json.dumps(lexical_sequence(text))),
    }


def construct_exchanges(sentences, nlp, tokenize):
    splits = []
    for sentence_index, sentence in enumerate(sentences):
        for candidate in find_split_candidates(sentence, nlp, tokenize):
            splits.append({"sentence_index": sentence_index, **candidate})
    merges = find_merge_candidates(sentences, nlp)
    exchanges = []
    for split in splits:
        for merge in merges:
            if split["sentence_index"] in {
                merge["boundary_index"], merge["boundary_index"] + 1,
            }:
                continue
            exchanges.append(build_exchange(sentences, merge, split, tokenize))
    exchanges.sort(
        key=lambda row: (
            row["split_sentence_index"],
            row["merge_boundary_index"],
            row["text"],
        )
    )
    return splits, merges, exchanges


def historical_asset_inventory():
    expected_release_paths = [
        ROOT / "third_party/SWAN/output/watermarked",
        ROOT / "third_party/SWAN/data/c4-val",
        ROOT / "third_party/SWAN/data/c4-human",
    ]
    prior = {}
    for name in (
        "swan_natural_mixed_windows1",
        "swan_natural_narrative_windows2",
        "swan_c4_train_narrative_windows3",
        "swan_quality_first_windows4",
        "swan_known_positive1",
    ):
        path = ROOT / "experiments" / name / "summary.json"
        payload = json.loads(path.read_text())
        prior[name] = {
            "summary_sha256": sha(path),
            "status": payload["status"],
            "decision": payload.get("decision"),
        }
    return {
        "official_expected_paths": {
            str(path.relative_to(ROOT)): path.exists()
            for path in expected_release_paths
        },
        "official_watermarked_release_present": any(
            path.exists() for path in expected_release_paths[:1]
        ),
        "prior_searches": prior,
        "interpretation": (
            "The official checkout contains generation code and AMR banks but no "
            "released watermarked-text dataset. Prior normal-length searches are "
            "not eligible watermarked positives."
        ),
    }


def build_manifest(nlp, tokenize):
    source = ROOT / "experiments/confirmation_three_arm30"
    baseline_paths = sorted((source / "baselines").glob("*.json"))
    if len(baseline_paths) != 30:
        raise ValueError("Expected the frozen 30-document baseline cohort")

    documents = []
    records = []
    dependencies = {
        "official_readme": sha(ROOT / "third_party/SWAN/README.md"),
        "confirmation_final_analysis": sha(source / "final_analysis.json"),
        "script": sha(Path(__file__)),
    }
    for path in baseline_paths:
        baseline = json.loads(path.read_text())
        dependencies[str(path.relative_to(ROOT))] = sha(path)
        sentences = [row["text"] for row in baseline["detection"]["details"]]
        if len(sentences) != baseline["detection"]["n"] or len(sentences) != 5:
            raise ValueError("Expected five saved detector units")
        original_text = " ".join(sentences)
        if tokenize(original_text) != sentences:
            raise ValueError("Fresh tokenizer differs from saved detector units")
        if lexical_sequence(original_text) != lexical_sequence(baseline["text"]):
            raise ValueError("Saved baseline text differs from detector details")

        splits, merges, exchanges = construct_exchanges(
            sentences, nlp, tokenize,
        )
        detected = baseline["detection"]["z"] > 2.33
        eligible = bool(exchanges) and detected
        selected = exchanges[0] if eligible else None
        document = {
            "prompt_id": baseline["prompt_id"],
            "source": str(path.relative_to(ROOT)),
            "source_sha256": sha(path),
            "original_text": original_text,
            "original_text_sha256": text_sha(original_text),
            "lexical_sha256": text_sha(
                json.dumps(lexical_sequence(original_text))
            ),
            "words": len(lexical_sequence(original_text)),
            "units": len(sentences),
            "historical_detection": {
                "n": baseline["detection"]["n"],
                "green": baseline["detection"]["green"],
                "green_rate": (
                    baseline["detection"]["green"]
                    / baseline["detection"]["n"]
                ),
                "z": baseline["detection"]["z"],
            },
            "historically_detected_at_2_33": detected,
            "split_candidates": splits,
            "merge_candidates": merges,
            "exchange_candidate_count": len(exchanges),
            "selected_exchange": selected,
            "eligible": eligible,
        }
        documents.append(document)
        if selected is not None:
            records.append({
                "id": f'count-neutral-{baseline["prompt_id"]}',
                "cohort": "attack",
                "prompt_id": baseline["prompt_id"],
                "text": selected["text"],
                "text_sha256": selected["text_sha256"],
                "texts": {
                    "original": original_text,
                    "exchange": selected["text"],
                    "restored": original_text,
                },
                "segments": {
                    "original": sentences,
                    "exchange": selected["units"],
                    "restored": sentences,
                },
                "construction": selected,
            })

    coverage = {
        "documents": len(documents),
        "historically_detected": sum(
            row["historically_detected_at_2_33"] for row in documents
        ),
        "documents_with_split_candidate": sum(
            bool(row["split_candidates"]) for row in documents
        ),
        "documents_with_merge_candidate": sum(
            bool(row["merge_candidates"]) for row in documents
        ),
        "documents_with_any_exchange": sum(
            row["exchange_candidate_count"] > 0 for row in documents
        ),
        "eligible_detected_exchanges": len(records),
    }
    ready = bool(records)
    return {
        "status": (
            "ready_for_detection" if ready
            else "complete_construction_gate_failed"
        ),
        "decision": (
            "run_local_detection" if ready
            else "stop_before_amr_parser_and_s2match"
        ),
        "protocol": {
            "version": 1,
            "source_cohort": "confirmation_three_arm30 frozen baselines",
            "selection": (
                "Earliest explicit independent-clause split, then earliest "
                "non-overlapping related declarative merge; no detector score "
                "enters construction or selection"
            ),
            "split_rule": (
                "Existing semicolon or coordinating VERB/AUX conjunct with an "
                "explicit subject; both resulting clauses need explicit subjects, "
                "finite roots, and at least three lexical tokens"
            ),
            "merge_rule": (
                "Adjacent period-final explicit-subject finite clauses sharing at "
                "least one non-stopword content lemma"
            ),
            "invariants": [
                "case-insensitive lexical sequence and order are unchanged",
                "NLTK sentence count is unchanged",
                "merge and split affect different source sentences",
                "restored text and units equal the frozen original exactly",
            ],
            "original_positive_rule": "historical z > 2.33",
            "gate": (
                "At least one historically detected document must admit an "
                "exchange before AMR parsing or S2Match"
            ),
            "network_calls": 0,
        },
        "coverage": coverage,
        "asset_inventory": historical_asset_inventory(),
        "documents": documents,
        "records": records,
        "dependencies": dependencies,
    }


def summarize(manifest):
    coverage = manifest["coverage"]
    return {
        "status": manifest["status"],
        "decision": manifest["decision"],
        "coverage": coverage,
        "negative_result": (
            coverage["documents_with_split_candidate"] == 0
        ),
        "parser_calls": 0,
        "s2match_calls": 0,
        "api_calls": 0,
        "claim": (
            "The frozen 30-document short-text cohort contains no source "
            "sentence satisfying the conservative natural split rule, so a "
            "count-neutral boundary exchange cannot be constructed on this "
            "cohort without relaxing the text-validity constraint."
            if coverage["eligible_detected_exchanges"] == 0 else
            "The construction gate passed; detection remains pending."
        ),
    }


def render_report(summary):
    coverage = summary["coverage"]
    return "\n".join([
        "# SWAN count-neutral boundary exchange",
        "",
        f"Status: `{summary['status']}`.",
        "",
        f"- Frozen documents screened: **{coverage['documents']}**.",
        f"- Historically detected at `z > 2.33`: "
        f"**{coverage['historically_detected']}**.",
        f"- Documents with a conservative split point: "
        f"**{coverage['documents_with_split_candidate']}**.",
        f"- Documents with a related adjacent merge: "
        f"**{coverage['documents_with_merge_candidate']}**.",
        f"- Documents admitting a count-neutral exchange: "
        f"**{coverage['documents_with_any_exchange']}**.",
        f"- Eligible detected exchanges: "
        f"**{coverage['eligible_detected_exchanges']}**.",
        "",
        summary["claim"],
        "",
        "No AMR parser, S2Match, generation API, attack detection, or "
        "post-hoc rule relaxation was run.",
        "",
    ])


def prepare(output):
    import nltk
    import spacy

    nlp = spacy.load("en_core_web_sm")
    manifest = build_manifest(nlp, nltk.sent_tokenize)
    summary = summarize(manifest)
    manifest_path = output / "manifest.json"
    summary_path = output / "summary.json"
    report_path = output / "report.md"
    if any(path.exists() for path in (manifest_path, summary_path, report_path)):
        raise FileExistsError(f"Refusing to overwrite frozen output: {output}")
    write_json(manifest_path, manifest)
    write_json(summary_path, summary)
    report_path.write_text(render_report(summary))
    print(json.dumps(summary, indent=2), flush=True)


def detect(args):
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest["records"]:
        print(json.dumps({
            "status": "stopped_at_construction_gate",
            "reason": "No eligible count-neutral boundary exchanges",
            "parser_loaded": False,
            "s2match_loaded": False,
        }, indent=2), flush=True)
        return
    from run_segmentation_intervention import detect as detect_records
    detect_records(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "detect"))
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "experiments/swan_count_neutral_boundary1",
    )
    parser.add_argument("--cohort", default="attack")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument(
        "--model-dir", type=Path,
        default=ROOT / "models/model_parse_xfm_bart_large-v0_1_0",
    )
    parser.add_argument(
        "--tokenizer-dir", type=Path,
        default=ROOT / "models/bart-large-tokenizer",
    )
    args = parser.parse_args()
    os.environ.setdefault("NLTK_DATA", str(ROOT / ".nltk_data"))
    os.environ.setdefault(
        "GLOVE_VECTORS_PATH", str(ROOT / "vectors/glove.6B.100d.txt")
    )
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
    (prepare(args.output) if args.stage == "prepare" else detect(args))


if __name__ == "__main__":
    main()
