"""Run a slot-blind, quality-first natural-window SWAN preflight."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import statistics
import time
import unicodedata

from feedback2_runtime import Heartbeat, claim
from prepare_confirmation_experiment import normalized_hash
from run_confirmation_experiment import (
    CallJournal,
    digest,
    freeze,
    request_body,
)
from run_segmentation_intervention import REVISION, ROOT, sha, text_sha, write_json
import run_swan_c4_train_narrative_windows as train
import run_swan_natural_mixed_windows as mixed
import run_swan_natural_narrative_windows as narrative
from supervise_confirmation_experiment import read_credential


OUTPUT = ROOT / "experiments/swan_quality_first_windows4"
PRIOR_FOLDERS = (
    ROOT / "experiments/swan_natural_mixed_windows1",
    ROOT / "experiments/swan_natural_narrative_windows2",
    ROOT / "experiments/swan_c4_train_narrative_windows3",
)
SHARD_INDICES = tuple(range(2, 6))
MAX_ELIGIBLE_ROWS = 100000
DOCUMENTS = 30
API_CAP = 31
MIN_JOINT_COVERAGE_DOCUMENTS = 3
MAX_ACQUISITION_ATTEMPTS = 3
LIST_MARKER = re.compile(r"^\s*(?:[-*#]|\d+[.)])\s")


def configure_environment():
    narrative.configure_environment()


def identity_text(text):
    return " ".join(
        unicodedata.normalize("NFKC", text)
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .split()
    ).casefold()


def quality_sentence_rows(text, tokenize):
    rows = []
    for line_index, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        line_sentences = tokenize(line)
        line_count = len(line_sentences)
        for sentence in line_sentences:
            value = narrative.canonical(sentence)
            words = len(value.split())
            rows.append(
                {
                    "text": value,
                    "text_sha256": text_sha(value),
                    "identity_sha256": text_sha(identity_text(value)),
                    "words": words,
                    "line_index": line_index,
                    "line_sentence_count": line_count,
                    "surface_pass": bool(
                        narrative.MIN_SENTENCE_WORDS
                        <= words
                        <= narrative.MAX_SENTENCE_WORDS
                        and re.search(r"[.?!][\"')\]]?$", value)
                        and not LIST_MARKER.match(value)
                    ),
                }
            )
    return rows


def find_quality_window(text, tokenize, nlp):
    sentences = quality_sentence_rows(text, tokenize)
    for start in range(
        max(0, len(sentences) - narrative.WINDOW_SENTENCES + 1)
    ):
        window = sentences[start:start + narrative.WINDOW_SENTENCES]
        total_words = sum(row["words"] for row in window)
        base_checks = {
            "exact_sentence_count": (
                len(window) == narrative.WINDOW_SENTENCES
            ),
            "sentence_surface": all(row["surface_pass"] for row in window),
            "total_word_range": (
                narrative.MIN_TOTAL_WORDS
                <= total_words
                <= narrative.MAX_TOTAL_WORDS
            ),
            "unique_sentences": len({
                row["identity_sha256"] for row in window
            }) == len(window),
        }
        if not all(base_checks.values()):
            continue
        narrative_metrics = narrative.narrative_metrics(window, nlp)
        if not narrative_metrics["pass"]:
            continue
        passage = " ".join(row["text"] for row in window)
        checks = dict(base_checks)
        checks.update(narrative_metrics["checks"])
        return {
            "window_start": start,
            "window_end_exclusive": start + narrative.WINDOW_SENTENCES,
            "sentences": window,
            "total_words": total_words,
            "passage": passage,
            "passage_sha256": text_sha(passage),
            "narrative": narrative_metrics,
            "checks": checks,
            "pass": all(checks.values()),
        }
    return None


def prior_identities():
    texts = set()
    urls = set()
    for folder in PRIOR_FOLDERS:
        manifest = json.loads((folder / "manifest.json").read_text())
        for source in manifest["sources"]:
            texts.add(source["normalized_source_sha256"])
            if source["url"]:
                urls.add(source["url"])
    return texts, urls


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
        "shards_opened": 0,
    }
    journal["attempts"].append(attempt)
    write_json(path, journal)
    return path, journal, attempt


def acquire_sources(folder, tokenize, nlp):
    path, journal, attempt = source_attempt(folder)
    seen_text, seen_url = prior_identities()
    selected = []
    screening = {
        "eligible_rows_scanned": 0,
        "duplicate_rows": 0,
        "quality_rule_window_rows": 0,
        "selected_documents": 0,
        "short_slot_predicate_evaluations": 0,
    }
    scanned_through = {}
    reached_limit = False
    stream = None
    try:
        for shard_index in SHARD_INDICES:
            stream = train.stream_train_shard(shard_index)
            attempt["shards_opened"] += 1
            for row_index, row in enumerate(stream):
                if screening["eligible_rows_scanned"] >= MAX_ELIGIBLE_ROWS:
                    reached_limit = True
                    break
                scanned_through[str(shard_index)] = row_index
                screening["eligible_rows_scanned"] += 1
                text = row.get("text", "")
                url = row.get("url", "")
                identity = normalized_hash(text)
                duplicate = (
                    identity in seen_text
                    or bool(url and url in seen_url)
                )
                seen_text.add(identity)
                if url:
                    seen_url.add(url)
                if duplicate:
                    screening["duplicate_rows"] += 1
                    continue
                window = find_quality_window(text, tokenize, nlp)
                if window is None:
                    continue
                screening["quality_rule_window_rows"] += 1
                selected.append(
                    {
                        "prompt_id": (
                            f"c4-train-{shard_index:05d}-{row_index:08d}"
                        ),
                        "shard_index": shard_index,
                        "shard_row": row_index,
                        "url": url,
                        "url_sha256": text_sha(url),
                        "source": text,
                        "source_sha256": text_sha(text),
                        "normalized_source_sha256": identity,
                        "window": window,
                    }
                )
                if len(selected) == DOCUMENTS:
                    reached_limit = True
                    break
            stream.close()
            stream = None
            if reached_limit:
                break
    except Exception as exc:
        attempt.update(
            status="failed",
            error_type=type(exc).__name__,
            finished_at=datetime.now(timezone.utc).isoformat(),
            scanned_through=scanned_through,
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
        scanned_through=scanned_through,
        eligible_rows_scanned=screening["eligible_rows_scanned"],
        selected_documents=len(selected),
        short_slot_predicate_evaluations=0,
    )
    write_json(path, journal)
    return selected, screening, scanned_through


def package_versions():
    values = narrative.package_versions()
    values["en_core_web_sm"] = importlib.metadata.version("en_core_web_sm")
    return values


def dependency_paths():
    paths = set(narrative.dependency_paths())
    paths.update(
        {
            Path(__file__),
            ROOT / "scripts/audit_swan_quality_first_windows.py",
            OUTPUT / "protocol.md",
            ROOT / "scripts/run_swan_c4_train_narrative_windows.py",
        }
    )
    for folder in PRIOR_FOLDERS:
        paths.update(
            {
                folder / "manifest.json",
                folder / "summary.json",
                folder / "audit.json",
                folder / "recovery_verification.json",
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
    attempts = json.loads(acquisition_path.read_text())["attempts"]
    manifest = {
        "status": "frozen_quality_first_before_api_or_slot_reveal",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "allenai/c4",
            "configuration": "realnewslike",
            "revision": REVISION,
            "shards": [train.shard_name(index) for index in SHARD_INDICES],
            "maximum_eligible_rows": MAX_ELIGIBLE_ROWS,
            "scanned_through": scanned_through,
            "selection": "first thirty slot-blind quality-rule windows",
            "source_acquisition_sha256": sha(acquisition_path),
            "dataset_stream_http_requests": sum(
                row["shards_opened"] for row in attempts
            ),
        },
        "screening": screening,
        "sources": sources,
        "protocol": {
            "workflow": "swan_quality_first_natural_window_preflight",
            "target_documents": DOCUMENTS,
            "selection_short_slot_predicate_evaluations": 0,
            "window_sentences": narrative.WINDOW_SENTENCES,
            "window_total_word_range": [
                narrative.MIN_TOTAL_WORDS,
                narrative.MAX_TOTAL_WORDS,
            ],
            "sentence_word_range": [
                narrative.MIN_SENTENCE_WORDS,
                narrative.MAX_SENTENCE_WORDS,
            ],
            "narrative_filters": {
                "maximum_source_lines": narrative.MAX_SOURCE_LINES,
                "minimum_multi_sentence_lines": (
                    narrative.MIN_MULTI_SENTENCE_LINES
                ),
                "minimum_verbal_roots": narrative.MIN_VERBAL_ROOTS,
                "minimum_unique_prefix2": narrative.MIN_UNIQUE_PREFIX2,
                "maximum_prefix3_frequency": (
                    narrative.MAX_PREFIX3_FREQUENCY
                ),
                "maximum_pair_jaccard": narrative.MAX_PAIR_JACCARD,
                "minimum_adjacent_overlap": (
                    narrative.MIN_ADJACENT_OVERLAP
                ),
            },
            "quality": (
                "slot-blind review; all style dimensions >=4 and no severe issue"
            ),
            "coverage_reveal": {
                "word_range": [
                    narrative.MIN_SHORT_WORDS,
                    narrative.MAX_SHORT_WORDS,
                ],
                "minimum_slots": narrative.MIN_SHORT_SLOTS,
                "minimum_joint_quality_coverage_documents": (
                    MIN_JOINT_COVERAGE_DOCUMENTS
                ),
            },
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
                "minimum_graph_documents": mixed.MIN_JOINT_DOCUMENTS,
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
        != "swan_quality_first_natural_window_preflight"
        or manifest["screening"]["short_slot_predicate_evaluations"] != 0
        or manifest["protocol"][
            "selection_short_slot_predicate_evaluations"
        ] != 0
    ):
        raise ValueError("Unexpected or non-blind workflow")
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
        if any(
            "short" in key
            for key in source["window"]
        ) or any(
            "short" in key
            for row in source["window"]["sentences"]
            for key in row
        ):
            raise ValueError("Frozen selection leaked short-slot metadata")
        if (
            source["shard_index"] not in SHARD_INDICES
            or text_sha(source["source"]) != source["source_sha256"]
            or normalized_hash(source["source"])
            != source["normalized_source_sha256"]
        ):
            raise ValueError("Frozen source identity changed")
        rebuilt = find_quality_window(
            source["source"],
            nltk.sent_tokenize,
            nlp,
        )
        if rebuilt != source["window"]:
            raise ValueError("Frozen quality-first window changed")
    return manifest


def build_quality_document(source):
    window = source["window"]
    return {
        "prompt_id": source["prompt_id"],
        "source_sha256": source["source_sha256"],
        "shard_index": source["shard_index"],
        "shard_row": source["shard_row"],
        "passage": window["passage"],
        "passage_sha256": window["passage_sha256"],
        "sentences": [row["text"] for row in window["sentences"]],
        "sentence_words": [row["words"] for row in window["sentences"]],
        "structure": {
            "sentences": narrative.WINDOW_SENTENCES,
            "words": window["total_words"],
            "checks": window["checks"],
            "pass": window["pass"],
        },
    }


def freeze_quality_documents(folder, manifest):
    for source in manifest["sources"]:
        freeze(
            folder / "generated" / f"{source['prompt_id']}.json",
            build_quality_document(source),
        )


def build_coverage(generated):
    short_indices = [
        index
        for index, words in enumerate(generated["sentence_words"])
        if narrative.MIN_SHORT_WORDS <= words <= narrative.MAX_SHORT_WORDS
    ]
    facts = [generated["sentences"][index] for index in short_indices]
    return {
        "prompt_id": generated["prompt_id"],
        "passage_sha256": generated["passage_sha256"],
        "word_range": [
            narrative.MIN_SHORT_WORDS,
            narrative.MAX_SHORT_WORDS,
        ],
        "slot_source_indices": short_indices,
        "slots": facts,
        "slot_count": len(facts),
        "coverage_pass": len(facts) >= narrative.MIN_SHORT_SLOTS,
    }


def reveal_coverage(folder, manifest):
    for source in manifest["sources"]:
        generated = json.loads(
            (folder / "generated" / f"{source['prompt_id']}.json").read_text()
        )
        freeze(
            folder / "coverage" / f"{source['prompt_id']}.json",
            build_coverage(generated),
        )


class Journal(CallJournal):
    def __init__(self, folder, key, heartbeat):
        super().__init__(folder, key, API_CAP)
        self.prior = len(list((folder / "calls").glob("*.json")))
        self.heartbeat = heartbeat

    def request(self, scope, body):
        ordinal = self.counters.get(scope, 0)
        path = self.folder / "calls" / f"{scope}-{ordinal:03d}.json"
        if not path.exists() and self.prior + self.sent >= API_CAP:
            raise RuntimeError("Frozen cumulative API cap reached")
        self.heartbeat.update(
            api_scope=scope,
            api_attempts=self.prior + self.sent,
        )
        return super().request(scope, body)


def review_all(folder, manifest, api, runtime, heartbeat):
    for index, source in enumerate(manifest["sources"]):
        generated = json.loads(
            (folder / "generated" / f"{source['prompt_id']}.json").read_text()
        )
        body = mixed.style_body(generated["passage"])
        response = api.request(f"style-{source['prompt_id']}", body)
        errors = {}
        try:
            style = mixed.decode_baseline_review(response["content"])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            style = None
            errors["style"] = type(exc).__name__
        gates = mixed.style_gate(generated, style)
        freeze(
            folder / "reviews" / f"{source['prompt_id']}.json",
            {
                "prompt_id": source["prompt_id"],
                "source_sha256": source["source_sha256"],
                "passage_sha256": generated["passage_sha256"],
                "runtime_sha256": digest(runtime),
                "request_sha256": digest(body),
                "request_exposes_coverage": False,
                "response": response,
                "style": style,
                "decode_errors": errors,
                "gates": gates,
                "quality_pass": all(gates.values()),
            },
        )
        heartbeat.update(
            stage="quality_review",
            completed_documents=index + 1,
            total_documents=DOCUMENTS,
        )


def joint_quality_coverage_sources(folder, manifest):
    eligible = []
    for source in manifest["sources"]:
        review = json.loads(
            (folder / "reviews" / f"{source['prompt_id']}.json").read_text()
        )
        coverage = json.loads(
            (folder / "coverage" / f"{source['prompt_id']}.json").read_text()
        )
        if review["quality_pass"] and coverage["coverage_pass"]:
            eligible.append(source)
    return eligible


def parse_eligible_documents(folder, manifest, runtime, heartbeat):
    eligible = joint_quality_coverage_sources(folder, manifest)
    if len(eligible) < MIN_JOINT_COVERAGE_DOCUMENTS:
        return 0
    pending = [
        source
        for source in eligible
        if not (
            folder / "parser" / f"{source['prompt_id']}.json"
        ).exists()
    ]
    if not pending:
        return 0
    import amrlib
    import torch

    utils = mixed.load_utils()
    bank = json.loads(mixed.BANK_PATH.read_text())
    bank_counts = [
        mixed.s2_counts(row["normalized_template_amr"])
        for row in bank
    ]
    torch.set_num_threads(4)
    parser = amrlib.load_stog_model(
        model_dir=str(mixed.MODEL_DIR),
        tok_name_or_path=str(mixed.TOKENIZER_DIR),
        device="cpu",
        batch_size=4,
    )
    calls = 0
    for source in pending:
        generated = json.loads(
            (folder / "generated" / f"{source['prompt_id']}.json").read_text()
        )
        coverage = json.loads(
            (folder / "coverage" / f"{source['prompt_id']}.json").read_text()
        )
        sentences = coverage["slots"]
        token_lengths = [
            len(parser.tokenizer(sentence)["input_ids"])
            for sentence in sentences
        ]
        if any(length > parser.max_sent_len for length in token_lengths):
            raise ValueError("Tokenizer limit exceeded")
        heartbeat.update(
            stage="local_amr_parser",
            prompt_id=source["prompt_id"],
            new_neural_parser_calls=calls,
        )
        started = time.monotonic()
        outputs = parser.parse_sents(sentences)
        if len(outputs) != len(sentences) or any(not row for row in outputs):
            raise ValueError("Incomplete parser output")
        rows = []
        for index, (sentence, tokens, raw_output) in enumerate(
            zip(sentences, token_lengths, outputs)
        ):
            identity = {
                "sentence_index": index,
                "source_sentence_index": coverage[
                    "slot_source_indices"
                ][index],
                "text": sentence,
                "text_sha256": text_sha(sentence),
                "words": len(sentence.split()),
            }
            row = mixed.analyze_parser_output(
                identity,
                raw_output,
                utils,
                bank_counts,
            )
            row["input_tokens"] = tokens
            rows.append(row)
        freeze(
            folder / "parser" / f"{source['prompt_id']}.json",
            {
                "prompt_id": source["prompt_id"],
                "source_sha256": source["source_sha256"],
                "passage_sha256": generated["passage_sha256"],
                "coverage_sha256": digest(coverage),
                "runtime_sha256": digest(runtime),
                "sentences": rows,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        calls += len(rows)
    return calls


def coverage_distribution(values):
    return {
        "counts": values,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "median": statistics.median(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "histogram": {
            str(value): values.count(value)
            for value in sorted(set(values))
        },
    }


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
            "coverage_pass": 0,
            "quality_coverage_pass": 0,
            "graph_pass": 0,
            "continuation_gate": False,
            "decision": "stop_quality_first_window_route",
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
        coverage = json.loads(
            (folder / "coverage" / f"{source['prompt_id']}.json").read_text()
        )
        parser_path = folder / "parser" / f"{source['prompt_id']}.json"
        graph = (
            mixed.graph_document(json.loads(parser_path.read_text()))
            if parser_path.exists()
            else None
        )
        joint = bool(
            review["quality_pass"] and coverage["coverage_pass"]
        )
        rows.append(
            {
                "prompt_id": source["prompt_id"],
                "shard_index": source["shard_index"],
                "shard_row": source["shard_row"],
                "words": generated["structure"]["words"],
                "quality_gates": review["gates"],
                "quality_pass": review["quality_pass"],
                "short_slots": coverage["slot_count"],
                "coverage_pass": coverage["coverage_pass"],
                "quality_coverage_pass": joint,
                "graph": graph,
                "graph_pass": bool(graph and graph["pass"]),
            }
        )
    quality_pass = sum(row["quality_pass"] for row in rows)
    coverage_pass = sum(row["coverage_pass"] for row in rows)
    joint_coverage = sum(row["quality_coverage_pass"] for row in rows)
    expected_parser_docs = (
        joint_coverage
        if joint_coverage >= MIN_JOINT_COVERAGE_DOCUMENTS
        else 0
    )
    parser_docs = sum(row["graph"] is not None for row in rows)
    complete = len(rows) == DOCUMENTS and parser_docs == expected_parser_docs
    graph_pass = sum(row["graph_pass"] for row in rows)
    quality_slot_counts = [
        row["short_slots"] for row in rows if row["quality_pass"]
    ]
    result = {
        "status": "complete" if complete else "incomplete",
        "selected_documents": selected,
        "target_documents": DOCUMENTS,
        "screening": manifest["screening"],
        "documents": rows,
        "quality_pass": quality_pass,
        "coverage_pass": coverage_pass,
        "quality_coverage_pass": joint_coverage,
        "quality_conditioned_slot_distribution": coverage_distribution(
            quality_slot_counts
        ),
        "graph_pass": graph_pass,
        "continuation_gate": bool(
            complete and graph_pass >= mixed.MIN_JOINT_DOCUMENTS
        ),
        "decision": (
            "quality_first_windows_support_partial_injection_preflight"
            if complete and graph_pass >= mixed.MIN_JOINT_DOCUMENTS
            else "stop_quality_first_window_route"
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
    lines = ["# SWAN quality-first natural-window preflight", ""]
    if result["selected_documents"] < DOCUMENTS:
        lines.extend(
            [
                f"Corpus gate: **{result['selected_documents']}/{DOCUMENTS}**.",
                "",
                f"- Rows scanned: "
                f"**{result['screening']['eligible_rows_scanned']}**.",
                "- No API, coverage reveal, AMR, S2Match, injection, "
                "detection, or attack call was run.",
                f"- Decision: `{result['decision']}`.",
                "",
            ]
        )
    else:
        distribution = result["quality_conditioned_slot_distribution"]
        lines.extend(
            [
                f"Quality pass: **{result['quality_pass']}/{DOCUMENTS}**; "
                f"coverage pass: **{result['coverage_pass']}/{DOCUMENTS}**; "
                f"joint quality and coverage: "
                f"**{result['quality_coverage_pass']}/{DOCUMENTS}**.",
                "",
                f"Quality-conditioned short-slot counts: "
                f"`{distribution['counts']}`.",
                "",
                "| Document | Words | Quality | Short slots | Coverage | "
                "Scale-compatible | Graph |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in result["documents"]:
            graph = row["graph"]
            lines.append(
                f"| {row['prompt_id']} | {row['words']} | "
                f"{'yes' if row['quality_pass'] else 'no'} | "
                f"{row['short_slots']} | "
                f"{'yes' if row['coverage_pass'] else 'no'} | "
                f"{graph['scale_compatible'] if graph else 'n/a'} | "
                f"{'yes' if row['graph_pass'] else 'no'} |"
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
    freeze_quality_documents(folder, manifest)
    if len(manifest["sources"]) < DOCUMENTS:
        return summarize(folder)
    with Heartbeat(folder) as heartbeat:
        api = Journal(folder, key, heartbeat)
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
        review_all(folder, manifest, api, runtime, heartbeat)
        reveal_coverage(folder, manifest)
        parser_calls = parse_eligible_documents(
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
                "scanned_through": manifest["dataset"]["scanned_through"],
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
