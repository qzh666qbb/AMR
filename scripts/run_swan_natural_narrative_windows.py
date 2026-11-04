"""Find natural short-slot windows after deterministic narrative filtering."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import unicodedata

from feedback2_runtime import Heartbeat, claim
from prepare_confirmation_experiment import normalized_hash, stream_validation
from run_confirmation_experiment import digest, freeze, request_body
from run_segmentation_intervention import ROOT, sha, text_sha, write_json
import run_swan_natural_mixed_windows as mixed
from supervise_confirmation_experiment import read_credential


OUTPUT = ROOT / "experiments/swan_natural_narrative_windows2"
INPUT = ROOT / "experiments/swan_natural_mixed_windows1"
SOURCE_START = 4300
SOURCE_END = 54300
DOCUMENTS = 5
WINDOW_SENTENCES = 12
MIN_TOTAL_WORDS = 100
MAX_TOTAL_WORDS = 220
MIN_SENTENCE_WORDS = 5
MAX_SENTENCE_WORDS = 35
MIN_SHORT_SLOTS = 8
MIN_SHORT_WORDS = 7
MAX_SHORT_WORDS = 10
MAX_SOURCE_LINES = 8
MIN_MULTI_SENTENCE_LINES = 3
MIN_VERBAL_ROOTS = 10
MIN_UNIQUE_PREFIX2 = 8
MAX_PREFIX3_FREQUENCY = 2
MAX_PAIR_JACCARD = 0.65
MIN_ADJACENT_OVERLAP = 4
API_CAP = 6
MAX_ACQUISITION_ATTEMPTS = 3
LIST_MARKER = re.compile(r"^\s*(?:[-*#]|\d+[.)])\s")


def configure_environment():
    mixed.configure_environment()


def canonical(text):
    return " ".join(text.split())


def identity_text(text):
    return " ".join(
        unicodedata.normalize("NFKC", text)
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .split()
    ).casefold()


def lexical_tokens(text):
    return re.findall(r"[a-z0-9']+", identity_text(text))


def content_lemmas(doc):
    return {
        token.lemma_.casefold()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and token.pos_ not in {"DET", "PRON", "AUX"}
    }


def jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def sentence_rows(text, tokenize):
    rows = []
    for line_index, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        line_sentences = tokenize(line)
        line_count = len(line_sentences)
        for sentence in line_sentences:
            value = canonical(sentence)
            words = len(value.split())
            rows.append(
                {
                    "text": value,
                    "text_sha256": text_sha(value),
                    "identity_sha256": text_sha(identity_text(value)),
                    "words": words,
                    "short_slot": MIN_SHORT_WORDS <= words <= MAX_SHORT_WORDS,
                    "line_index": line_index,
                    "line_sentence_count": line_count,
                    "surface_pass": bool(
                        MIN_SENTENCE_WORDS <= words <= MAX_SENTENCE_WORDS
                        and re.search(r"[.?!][\"')\]]?$", value)
                        and not LIST_MARKER.match(value)
                    ),
                }
            )
    return rows


def narrative_metrics(window, nlp):
    docs = list(nlp.pipe(row["text"] for row in window))
    roots = [list(doc.sents)[0].root for doc in docs]
    verbal_roots = sum(root.pos_ in {"VERB", "AUX"} for root in roots)
    prefix2 = [tuple(lexical_tokens(row["text"])[:2]) for row in window]
    prefix3 = [tuple(lexical_tokens(row["text"])[:3]) for row in window]
    prefix3_counts = Counter(prefix3)
    contents = [content_lemmas(doc) for doc in docs]
    pair_scores = [
        jaccard(contents[left], contents[right])
        for left in range(len(contents))
        for right in range(left + 1, len(contents))
    ]
    adjacent_overlap = sum(
        bool(contents[index] & contents[index + 1])
        for index in range(len(contents) - 1)
    )
    line_counts = Counter(row["line_index"] for row in window)
    checks = {
        "maximum_source_lines": len(line_counts) <= MAX_SOURCE_LINES,
        "minimum_multi_sentence_lines": sum(
            count >= 2 for count in line_counts.values()
        ) >= MIN_MULTI_SENTENCE_LINES,
        "minimum_verbal_roots": verbal_roots >= MIN_VERBAL_ROOTS,
        "minimum_unique_prefix2": len(set(prefix2)) >= MIN_UNIQUE_PREFIX2,
        "maximum_prefix3_frequency": (
            max(prefix3_counts.values(), default=0)
            <= MAX_PREFIX3_FREQUENCY
        ),
        "maximum_pair_jaccard": max(pair_scores, default=0.0) <= MAX_PAIR_JACCARD,
        "minimum_adjacent_overlap": adjacent_overlap >= MIN_ADJACENT_OVERLAP,
    }
    return {
        "source_lines": len(line_counts),
        "multi_sentence_lines": sum(
            count >= 2 for count in line_counts.values()
        ),
        "verbal_roots": verbal_roots,
        "unique_prefix2": len(set(prefix2)),
        "maximum_prefix3_frequency": max(
            prefix3_counts.values(),
            default=0,
        ),
        "maximum_pair_jaccard": max(pair_scores, default=0.0),
        "adjacent_overlap_pairs": adjacent_overlap,
        "root_pos": [root.pos_ for root in roots],
        "checks": checks,
        "pass": all(checks.values()),
    }


def find_window(text, tokenize, nlp):
    sentences = sentence_rows(text, tokenize)
    for start in range(max(0, len(sentences) - WINDOW_SENTENCES + 1)):
        window = sentences[start:start + WINDOW_SENTENCES]
        total_words = sum(row["words"] for row in window)
        short_indices = [
            index for index, row in enumerate(window)
            if row["short_slot"]
        ]
        base_checks = {
            "exact_sentence_count": len(window) == WINDOW_SENTENCES,
            "sentence_surface": all(row["surface_pass"] for row in window),
            "total_word_range": MIN_TOTAL_WORDS <= total_words <= MAX_TOTAL_WORDS,
            "minimum_short_slots": len(short_indices) >= MIN_SHORT_SLOTS,
            "unique_sentences": len({
                row["identity_sha256"] for row in window
            }) == len(window),
        }
        if not all(base_checks.values()):
            continue
        narrative = narrative_metrics(window, nlp)
        if not narrative["pass"]:
            continue
        passage = " ".join(row["text"] for row in window)
        checks = dict(base_checks)
        checks.update(narrative["checks"])
        return {
            "window_start": start,
            "window_end_exclusive": start + WINDOW_SENTENCES,
            "sentences": window,
            "short_slot_indices": short_indices,
            "short_slot_count": len(short_indices),
            "total_words": total_words,
            "passage": passage,
            "passage_sha256": text_sha(passage),
            "narrative": narrative,
            "checks": checks,
            "pass": all(checks.values()),
        }
    return None


def source_attempt(folder):
    path = folder / "source_acquisition.json"
    journal = (
        json.loads(path.read_text())
        if path.exists()
        else {"attempts": []}
    )
    if len(journal["attempts"]) >= MAX_ACQUISITION_ATTEMPTS:
        raise RuntimeError("Frozen source acquisition attempt cap reached")
    attempt = {
        "ordinal": len(journal["attempts"]) + 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "in_flight",
    }
    journal["attempts"].append(attempt)
    write_json(path, journal)
    return path, journal, attempt


def acquire_sources(folder, tokenize, nlp):
    path, journal, attempt = source_attempt(folder)
    selected = []
    seen_text = set()
    seen_url = set()
    screening = {
        "eligible_rows_scanned": 0,
        "duplicate_rows": 0,
        "base_window_rows": 0,
        "selected_documents": 0,
    }
    stream = None
    scanned_through = SOURCE_START - 1
    try:
        stream = stream_validation()
        for shard_row, row in enumerate(stream):
            text = row.get("text", "")
            url = row.get("url", "")
            identity = normalized_hash(text)
            duplicate = identity in seen_text or bool(url and url in seen_url)
            seen_text.add(identity)
            if url:
                seen_url.add(url)
            if shard_row < SOURCE_START:
                continue
            if shard_row > SOURCE_END:
                break
            screening["eligible_rows_scanned"] += 1
            if duplicate:
                screening["duplicate_rows"] += 1
                continue
            base_exists = mixed.find_window(text, tokenize) is not None
            screening["base_window_rows"] += bool(base_exists)
            if not base_exists:
                continue
            window = find_window(text, tokenize, nlp)
            if window is None:
                continue
            selected.append(
                {
                    "prompt_id": f"c4-validation-{shard_row}",
                    "shard_row": shard_row,
                    "url": url,
                    "url_sha256": text_sha(url),
                    "source": text,
                    "source_sha256": text_sha(text),
                    "normalized_source_sha256": identity,
                    "window": window,
                }
            )
            if len(selected) == DOCUMENTS:
                break
        scanned_through = shard_row
    except Exception as exc:
        attempt.update(
            status="failed",
            error_type=type(exc).__name__,
            finished_at=datetime.now(timezone.utc).isoformat(),
            manifest_written=False,
        )
        write_json(path, journal)
        raise
    finally:
        if stream is not None:
            stream.close()
    screening["selected_documents"] = len(selected)
    attempt.update(
        status="complete",
        finished_at=datetime.now(timezone.utc).isoformat(),
        scanned_through_row=scanned_through,
        selected_documents=len(selected),
    )
    write_json(path, journal)
    return selected, screening, scanned_through


def package_versions():
    values = mixed.package_versions()
    values["en_core_web_sm"] = importlib.metadata.version("en_core_web_sm")
    return values


def dependency_paths():
    paths = set(mixed.dependency_paths())
    paths.update(
        {
            Path(__file__),
            ROOT / "scripts/audit_swan_natural_narrative_windows.py",
            OUTPUT / "protocol.md",
            INPUT / "manifest.json",
            INPUT / "summary.json",
            INPUT / "audit.json",
            INPUT / "recovery_verification.json",
            ROOT / "scripts/run_swan_natural_mixed_windows.py",
        }
    )
    return sorted(paths)


def dependencies():
    return {
        str(path.relative_to(ROOT)): sha(path)
        for path in dependency_paths()
    }


def prepare(folder=OUTPUT):
    configure_environment()
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        return check(folder)
    import nltk
    import spacy

    nlp = spacy.load("en_core_web_sm")
    sources, screening, scanned_through = acquire_sources(
        folder,
        nltk.sent_tokenize,
        nlp,
    )
    acquisition_path = folder / "source_acquisition.json"
    manifest = {
        "status": "frozen_after_narrative_scan_before_api_or_amr",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "allenai/c4",
            "first_eligible_row": SOURCE_START,
            "maximum_row": SOURCE_END,
            "scanned_through_row": scanned_through,
            "selection": "first five frozen narrative mixed windows",
            "source_acquisition_sha256": sha(acquisition_path),
            "dataset_stream_http_requests": len(
                json.loads(acquisition_path.read_text())["attempts"]
            ),
        },
        "screening": screening,
        "sources": sources,
        "protocol": {
            "workflow": "swan_natural_narrative_window_preflight",
            "target_documents": DOCUMENTS,
            "window_sentences": WINDOW_SENTENCES,
            "window_total_word_range": [MIN_TOTAL_WORDS, MAX_TOTAL_WORDS],
            "short_slot_word_range": [MIN_SHORT_WORDS, MAX_SHORT_WORDS],
            "minimum_short_slots": MIN_SHORT_SLOTS,
            "narrative_filters": {
                "maximum_source_lines": MAX_SOURCE_LINES,
                "minimum_multi_sentence_lines": MIN_MULTI_SENTENCE_LINES,
                "minimum_verbal_roots": MIN_VERBAL_ROOTS,
                "minimum_unique_prefix2": MIN_UNIQUE_PREFIX2,
                "maximum_prefix3_frequency": MAX_PREFIX3_FREQUENCY,
                "maximum_pair_jaccard": MAX_PAIR_JACCARD,
                "minimum_adjacent_overlap": MIN_ADJACENT_OVERLAP,
            },
            "quality": (
                "deterministic narrative window pass and all style dimensions >=4"
            ),
            "graph_scale": {
                "minimum_compatible_slots": mixed.MIN_SCALE_COMPATIBLE,
                "minimum_compatible_fraction": mixed.MIN_SCALE_FRACTION,
                "median_triples_range": [
                    mixed.MIN_MEDIAN_TRIPLES,
                    mixed.MAX_MEDIAN_TRIPLES,
                ],
                "bank_sha256": sha(mixed.BANK_PATH),
                "s2match_optimization": False,
            },
            "continuation_gate": {
                "minimum_joint_documents": mixed.MIN_JOINT_DOCUMENTS,
                "denominator": DOCUMENTS,
            },
            "api_cap": API_CAP,
            "new_generation_calls": 0,
            "no_watermark_injection_detection_or_attack": True,
        },
        "dependencies": dependencies(),
        "packages": package_versions(),
    }
    freeze(manifest_path, manifest)
    return manifest


def check(folder=OUTPUT):
    configure_environment()
    import nltk
    import spacy

    manifest = json.loads((folder / "manifest.json").read_text())
    if (
        manifest["protocol"]["workflow"]
        != "swan_natural_narrative_window_preflight"
    ):
        raise ValueError("Unexpected workflow")
    if sha(folder / "source_acquisition.json") != manifest["dataset"][
        "source_acquisition_sha256"
    ]:
        raise ValueError("Source acquisition journal changed")
    for relative, expected in manifest["dependencies"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Frozen dependency changed: {relative}")
    if package_versions() != manifest["packages"]:
        raise ValueError("Frozen package versions changed")
    nlp = spacy.load("en_core_web_sm")
    for source in manifest["sources"]:
        if text_sha(source["source"]) != source["source_sha256"]:
            raise ValueError("Frozen source changed")
        rebuilt = find_window(source["source"], nltk.sent_tokenize, nlp)
        if rebuilt != source["window"]:
            raise ValueError("Frozen narrative window changed")
    return manifest


def summarize(folder=OUTPUT):
    manifest = json.loads((folder / "manifest.json").read_text())
    selected = len(manifest["sources"])
    if selected < DOCUMENTS:
        result = {
            "status": "complete_corpus_gate_failed",
            "selected_documents": selected,
            "target_documents": DOCUMENTS,
            "screening": manifest["screening"],
            "quality_pass": 0,
            "graph_pass": 0,
            "joint_pass": 0,
            "continuation_gate": False,
            "decision": "stop_natural_narrative_window_route",
            "api_attempts": len(list((folder / "calls").glob("*.json"))),
            "neural_parser_sentences": 0,
            "new_s2match_calls": 0,
            "new_watermark_injection_detection_or_attack_calls": 0,
            "manifest_sha256": sha(folder / "manifest.json"),
        }
        write_json(folder / "summary.json", result)
        write_report(folder, result)
        return result
    rows = []
    for source in manifest["sources"]:
        generated = json.loads(
            (folder / "generated" / f"{source['prompt_id']}.json").read_text()
        )
        review = json.loads(
            (folder / "reviews" / f"{source['prompt_id']}.json").read_text()
        )
        parser_path = folder / "parser" / f"{source['prompt_id']}.json"
        graph = (
            mixed.graph_document(json.loads(parser_path.read_text()))
            if parser_path.exists()
            else None
        )
        rows.append(
            {
                "prompt_id": source["prompt_id"],
                "shard_row": source["shard_row"],
                "words": generated["structure"]["words"],
                "short_slots": generated["structure"]["short_slots"],
                "narrative": source["window"]["narrative"],
                "quality_gates": review["gates"],
                "quality_pass": review["quality_pass"],
                "graph": graph,
                "joint_pass": bool(
                    review["quality_pass"] and graph and graph["pass"]
                ),
            }
        )
    quality_pass = sum(row["quality_pass"] for row in rows)
    parser_docs = sum(row["graph"] is not None for row in rows)
    complete = len(rows) == DOCUMENTS and parser_docs == quality_pass
    joint = sum(row["joint_pass"] for row in rows)
    result = {
        "status": "complete" if complete else "incomplete",
        "selected_documents": selected,
        "target_documents": DOCUMENTS,
        "screening": manifest["screening"],
        "documents": rows,
        "quality_pass": quality_pass,
        "graph_pass": sum(
            bool(row["graph"] and row["graph"]["pass"]) for row in rows
        ),
        "joint_pass": joint,
        "continuation_gate": bool(
            complete and joint >= mixed.MIN_JOINT_DOCUMENTS
        ),
        "decision": (
            "natural_narrative_windows_support_partial_injection_preflight"
            if complete and joint >= mixed.MIN_JOINT_DOCUMENTS
            else "stop_natural_narrative_window_route"
            if complete
            else "pending"
        ),
        "api_attempts": len(list((folder / "calls").glob("*.json"))),
        "neural_parser_sentences": sum(
            len(json.loads(path.read_text())["sentences"])
            for path in (folder / "parser").glob("*.json")
        ),
        "new_s2match_calls": 0,
        "new_watermark_injection_detection_or_attack_calls": 0,
        "manifest_sha256": sha(folder / "manifest.json"),
    }
    write_json(folder / "summary.json", result)
    if complete:
        write_report(folder, result)
    return result


def write_report(folder, result):
    lines = [
        "# SWAN natural narrative mixed-window preflight",
        "",
    ]
    if result["selected_documents"] < DOCUMENTS:
        lines.extend(
            [
                f"Narrative corpus gate: **{result['selected_documents']}/"
                f"{DOCUMENTS}** documents found.",
                "",
                f"- Rows scanned: "
                f"**{result['screening']['eligible_rows_scanned']}**.",
                f"- Base-window rows before narrative filters: "
                f"**{result['screening']['base_window_rows']}**.",
                "- No quality API, AMR parser, S2Match, injection, detection, "
                "or attack call was run.",
                f"- Decision: `{result['decision']}`.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"Quality-valid windows: **{result['quality_pass']}/{DOCUMENTS}**; "
                f"joint quality and graph pass: **{result['joint_pass']}/"
                f"{DOCUMENTS}**.",
                "",
                "| Document | Words | Short slots | Verbal roots | Adjacent overlap "
                "| Quality | Scale-compatible | Joint |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in result["documents"]:
            graph = row["graph"]
            lines.append(
                f"| {row['prompt_id']} | {row['words']} | "
                f"{row['short_slots']} | "
                f"{row['narrative']['verbal_roots']} | "
                f"{row['narrative']['adjacent_overlap_pairs']} | "
                f"{'yes' if row['quality_pass'] else 'no'} | "
                f"{graph['scale_compatible'] if graph else 'n/a'} | "
                f"{'yes' if row['joint_pass'] else 'no'} |"
            )
        lines.extend(
            [
                "",
                f"- API calls: **{result['api_attempts']}**; local AMR parser "
                f"sentences: **{result['neural_parser_sentences']}**; "
                "S2Match calls: **0**.",
                "- No injection, detection, attack, TPR, FPR, or ASR was run.",
                f"- Decision: `{result['decision']}`.",
                "",
            ]
        )
    (folder / "report.md").write_text("\n".join(lines))


def run(folder, key):
    manifest = check(folder)
    mixed.freeze_documents(folder, manifest)
    if len(manifest["sources"]) < DOCUMENTS:
        return summarize(folder)
    with Heartbeat(folder) as heartbeat:
        api = mixed.Journal(folder, key, heartbeat)
        probe = api.request(
            "probe",
            request_body("Reply exactly OK.", 0, 16),
        )
        if probe["finish_reason"] != "stop" or probe["content"].strip() != "OK":
            raise ValueError("Unexpected connectivity response")
        api.model_returned = probe["model"]
        runtime = freeze(
            mixed.runtime_path(folder),
            {
                "manifest_sha256": sha(folder / "manifest.json"),
                "model_returned": probe["model"],
                "probe_id": probe["id"],
            },
        )
        mixed.review_all(folder, manifest, api, runtime, heartbeat)
        parser_calls = mixed.parse_quality_documents(
            folder,
            manifest,
            runtime,
            heartbeat,
        )
        heartbeat.update(
            stage="complete",
            new_api_calls=api.sent,
            new_neural_parser_calls=parser_calls,
            new_s2match_calls=0,
        )
    return summarize(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "summarize"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    configure_environment()
    with claim(args.output / ".runner.lock"):
        if args.stage == "prepare":
            manifest = prepare(args.output)
            result = {
                "status": "prepared",
                "selected_documents": len(manifest["sources"]),
                "scanned_through_row": manifest["dataset"][
                    "scanned_through_row"
                ],
                "manifest_sha256": sha(args.output / "manifest.json"),
            }
        elif args.stage == "run":
            key = (
                read_credential(args.key_file)
                if args.key_file
                else os.environ.get("DEEPSEEK_API_KEY", "")
            )
            try:
                result = run(args.output, key)
            finally:
                if (args.output / "manifest.json").exists():
                    print(json.dumps(summarize(args.output), indent=2), flush=True)
        else:
            result = summarize(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
