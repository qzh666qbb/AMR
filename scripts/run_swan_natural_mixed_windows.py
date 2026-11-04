"""Find and preflight natural C4 passages with many short sentence slots."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import statistics
import time

from feedback2_runtime import Heartbeat, claim
from prepare_confirmation_experiment import (
    normalized_hash,
    stream_validation,
)
from run_confirmation_experiment import (
    CallJournal,
    digest,
    freeze,
    request_body,
)
from run_segmentation_intervention import ROOT, sha, text_sha, write_json
from run_swan_atomic_fact_preflight import (
    MODEL_DIR,
    TOKENIZER_DIR,
    VECTORS_PATH,
    analyze_parser_output,
    s2_counts,
)
from run_swan_bank100_microfact_null import BANK_PATH
from run_swan_known_positive_audit import load_utils
from run_swan_sentence_growth import decode_baseline_review
from run_swan_constrained_microfact import style_body
from supervise_confirmation_experiment import read_credential


OUTPUT = ROOT / "experiments/swan_natural_mixed_windows1"
SOURCE_START = 1652
SOURCE_END = 51651
DOCUMENTS = 5
WINDOW_SENTENCES = 12
MIN_TOTAL_WORDS = 100
MAX_TOTAL_WORDS = 220
MIN_SENTENCE_WORDS = 5
MAX_SENTENCE_WORDS = 35
MIN_SHORT_SLOTS = 8
MIN_SHORT_WORDS = 7
MAX_SHORT_WORDS = 10
MIN_SCALE_COMPATIBLE = 7
MIN_SCALE_FRACTION = 0.75
MIN_MEDIAN_TRIPLES = 8
MAX_MEDIAN_TRIPLES = 16
MIN_JOINT_DOCUMENTS = 3
API_CAP = 6
MAX_ACQUISITION_ATTEMPTS = 3
LIST_MARKER = re.compile(r"^\s*(?:[-*#]|\d+[.)])\s")


def configure_environment():
    os.environ.setdefault("NLTK_DATA", str(ROOT / ".nltk_data"))
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
    os.environ.setdefault("GLOVE_VECTORS_PATH", str(VECTORS_PATH))


def canonical(text):
    return " ".join(text.split())


def sentence_metrics(sentence):
    text = canonical(sentence)
    words = len(text.split())
    checks = {
        "word_range": MIN_SENTENCE_WORDS <= words <= MAX_SENTENCE_WORDS,
        "terminal_punctuation": bool(re.search(r"[.?!][\"')\]]?$", text)),
        "not_list_marker": not LIST_MARKER.match(text),
    }
    return {
        "text": text,
        "text_sha256": text_sha(text),
        "words": words,
        "short_slot": MIN_SHORT_WORDS <= words <= MAX_SHORT_WORDS,
        "checks": checks,
        "pass": all(checks.values()),
    }


def find_window(text, tokenize):
    sentences = [sentence_metrics(row) for row in tokenize(text)]
    for start in range(max(0, len(sentences) - WINDOW_SENTENCES + 1)):
        window = sentences[start:start + WINDOW_SENTENCES]
        total_words = sum(row["words"] for row in window)
        short_indices = [
            index for index, row in enumerate(window)
            if row["short_slot"]
        ]
        checks = {
            "exact_sentence_count": len(window) == WINDOW_SENTENCES,
            "sentence_surface": all(row["pass"] for row in window),
            "total_word_range": MIN_TOTAL_WORDS <= total_words <= MAX_TOTAL_WORDS,
            "minimum_short_slots": len(short_indices) >= MIN_SHORT_SLOTS,
            "unique_sentences": len({
                normalized_hash(row["text"]) for row in window
            }) == len(window),
        }
        if all(checks.values()):
            passage = " ".join(row["text"] for row in window)
            return {
                "window_start": start,
                "window_end_exclusive": start + WINDOW_SENTENCES,
                "sentences": window,
                "short_slot_indices": short_indices,
                "short_slot_count": len(short_indices),
                "total_words": total_words,
                "passage": passage,
                "passage_sha256": text_sha(passage),
                "checks": checks,
                "pass": True,
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


def acquire_sources(folder, tokenize):
    path, journal, attempt = source_attempt(folder)
    selected = []
    seen_text = set()
    seen_url = set()
    screening = {
        "eligible_rows_scanned": 0,
        "duplicate_rows": 0,
        "rows_with_eligible_window": 0,
        "selected_documents": 0,
    }
    stream = None
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
            window = find_window(text, tokenize)
            if window is None:
                continue
            screening["rows_with_eligible_window"] += 1
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
    return {
        name: importlib.metadata.version(name)
        for name in (
            "amrlib",
            "nltk",
            "numpy",
            "penman",
            "requests",
            "spacy",
            "torch",
            "transformers",
        )
    }


def dependency_paths():
    paths = [
        Path(__file__),
        ROOT / "scripts/audit_swan_natural_mixed_windows.py",
        OUTPUT / "protocol.md",
        ROOT / "scripts/feedback2_runtime.py",
        ROOT / "scripts/prepare_confirmation_experiment.py",
        ROOT / "scripts/run_confirmation_experiment.py",
        ROOT / "scripts/run_segmentation_intervention.py",
        ROOT / "scripts/run_swan_atomic_fact_preflight.py",
        ROOT / "scripts/run_swan_known_positive_audit.py",
        ROOT / "scripts/run_swan_sentence_growth.py",
        ROOT / "scripts/run_swan_constrained_microfact.py",
        ROOT / "scripts/supervise_confirmation_experiment.py",
        BANK_PATH,
        ROOT / "third_party/SWAN/utils/amr_utils.py",
        ROOT / "third_party/SWAN/utils/s2match_patch.py",
        MODEL_DIR / "amrlib_meta.json",
        MODEL_DIR / "model_parse_xfm_bart_large.json",
        MODEL_DIR / "config.json",
        MODEL_DIR / "pytorch_model.bin",
        TOKENIZER_DIR / "tokenizer.json",
        TOKENIZER_DIR / "tokenizer_config.json",
        VECTORS_PATH,
    ]
    paths.extend(
        sorted((ROOT / ".nltk_data/tokenizers/punkt_tab/english").glob("*"))
    )
    return paths


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

    sources, screening, scanned_through = acquire_sources(
        folder,
        nltk.sent_tokenize,
    )
    acquisition_path = folder / "source_acquisition.json"
    manifest = {
        "status": "frozen_after_corpus_scan_before_api_or_parser",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "allenai/c4",
            "first_eligible_row": SOURCE_START,
            "maximum_row": SOURCE_END,
            "scanned_through_row": scanned_through,
            "selection": "first five eligible natural mixed windows",
            "source_acquisition_sha256": sha(acquisition_path),
            "dataset_stream_http_requests": len(
                json.loads(acquisition_path.read_text())["attempts"]
            ),
        },
        "screening": screening,
        "sources": sources,
        "protocol": {
            "workflow": "swan_natural_mixed_window_preflight",
            "target_documents": DOCUMENTS,
            "window_sentences": WINDOW_SENTENCES,
            "window_total_word_range": [
                MIN_TOTAL_WORDS,
                MAX_TOTAL_WORDS,
            ],
            "sentence_word_range": [
                MIN_SENTENCE_WORDS,
                MAX_SENTENCE_WORDS,
            ],
            "short_slot_word_range": [
                MIN_SHORT_WORDS,
                MAX_SHORT_WORDS,
            ],
            "minimum_short_slots": MIN_SHORT_SLOTS,
            "quality": (
                "deterministic window pass and style/coherence dimensions >=4"
            ),
            "graph_scale": {
                "minimum_compatible_slots": MIN_SCALE_COMPATIBLE,
                "minimum_compatible_fraction": MIN_SCALE_FRACTION,
                "median_triples_range": [
                    MIN_MEDIAN_TRIPLES,
                    MAX_MEDIAN_TRIPLES,
                ],
                "bank_sha256": sha(BANK_PATH),
                "s2match_optimization": False,
            },
            "continuation_gate": {
                "minimum_joint_documents": MIN_JOINT_DOCUMENTS,
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

    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["protocol"]["workflow"] != "swan_natural_mixed_window_preflight":
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
    for source in manifest["sources"]:
        if text_sha(source["source"]) != source["source_sha256"]:
            raise ValueError("Frozen source changed")
        rebuilt = find_window(source["source"], nltk.sent_tokenize)
        if rebuilt != source["window"]:
            raise ValueError("Frozen natural window changed")
    return manifest


def build_document(source):
    window = source["window"]
    short_indices = window["short_slot_indices"]
    facts = [
        window["sentences"][index]["text"]
        for index in short_indices
    ]
    return {
        "prompt_id": source["prompt_id"],
        "source_sha256": source["source_sha256"],
        "shard_row": source["shard_row"],
        "passage": window["passage"],
        "passage_sha256": window["passage_sha256"],
        "sentences": [
            row["text"] for row in window["sentences"]
        ],
        "sentence_words": [
            row["words"] for row in window["sentences"]
        ],
        "facts": facts,
        "fact_source_indices": short_indices,
        "structure": {
            "sentences": WINDOW_SENTENCES,
            "words": window["total_words"],
            "short_slots": len(facts),
            "checks": window["checks"],
            "pass": window["pass"],
        },
    }


def freeze_documents(folder, manifest):
    for source in manifest["sources"]:
        freeze(
            folder / "generated" / f"{source['prompt_id']}.json",
            build_document(source),
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


def style_gate(generated, style):
    return {
        "structure": generated["structure"]["pass"],
        "style": bool(
            style
            and min(
                style["fluency"],
                style["coherence"],
                style["nonredundancy"],
            ) >= 4
            and not style["severe_problem"]
        ),
    }


def review_all(folder, manifest, api, runtime, heartbeat):
    for index, source in enumerate(manifest["sources"]):
        generated = json.loads(
            (folder / "generated" / f"{source['prompt_id']}.json").read_text()
        )
        body = style_body(generated["passage"])
        response = api.request(f"style-{source['prompt_id']}", body)
        errors = {}
        try:
            style = decode_baseline_review(response["content"])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            style = None
            errors["style"] = type(exc).__name__
        gates = style_gate(generated, style)
        freeze(
            folder / "reviews" / f"{source['prompt_id']}.json",
            {
                "prompt_id": source["prompt_id"],
                "source_sha256": source["source_sha256"],
                "passage_sha256": generated["passage_sha256"],
                "runtime_sha256": digest(runtime),
                "request_sha256": digest(body),
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


def parse_quality_documents(folder, manifest, runtime, heartbeat):
    pending = []
    for source in manifest["sources"]:
        review = json.loads(
            (folder / "reviews" / f"{source['prompt_id']}.json").read_text()
        )
        path = folder / "parser" / f"{source['prompt_id']}.json"
        if review["quality_pass"] and not path.exists():
            pending.append(source)
    if not pending:
        return 0
    import amrlib
    import torch

    utils = load_utils()
    bank = json.loads(BANK_PATH.read_text())
    bank_counts = [
        s2_counts(row["normalized_template_amr"])
        for row in bank
    ]
    torch.set_num_threads(4)
    parser = amrlib.load_stog_model(
        model_dir=str(MODEL_DIR),
        tok_name_or_path=str(TOKENIZER_DIR),
        device="cpu",
        batch_size=4,
    )
    calls = 0
    for source in pending:
        generated = json.loads(
            (folder / "generated" / f"{source['prompt_id']}.json").read_text()
        )
        sentences = generated["facts"]
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
                "source_sentence_index": generated[
                    "fact_source_indices"
                ][index],
                "text": sentence,
                "text_sha256": text_sha(sentence),
                "words": len(sentence.split()),
            }
            row = analyze_parser_output(
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
                "runtime_sha256": digest(runtime),
                "sentences": rows,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        calls += len(rows)
    return calls


def graph_document(parsed):
    rows = parsed["sentences"]
    compatible = sum(row["scale_compatible"] for row in rows)
    fraction = compatible / len(rows) if rows else None
    median = statistics.median(
        row["template_triples"] for row in rows
    ) if rows else None
    required = max(
        MIN_SCALE_COMPATIBLE,
        math.ceil(MIN_SCALE_FRACTION * len(rows)),
    )
    return {
        "candidate_slots": len(rows),
        "scale_compatible": compatible,
        "scale_compatible_fraction": fraction,
        "required_compatible": required,
        "template_triples_median": median,
        "pass": bool(
            len(rows) >= MIN_SHORT_SLOTS
            and compatible >= required
            and MIN_MEDIAN_TRIPLES <= median <= MAX_MEDIAN_TRIPLES
        ),
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
            "graph_pass": 0,
            "joint_pass": 0,
            "continuation_gate": False,
            "decision": "stop_natural_mixed_window_route",
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
            graph_document(json.loads(parser_path.read_text()))
            if parser_path.exists()
            else None
        )
        rows.append(
            {
                "prompt_id": source["prompt_id"],
                "shard_row": source["shard_row"],
                "sentences": generated["structure"]["sentences"],
                "words": generated["structure"]["words"],
                "short_slots": generated["structure"]["short_slots"],
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
            complete and joint >= MIN_JOINT_DOCUMENTS
        ),
        "decision": (
            "natural_mixed_windows_support_partial_injection_preflight"
            if complete and joint >= MIN_JOINT_DOCUMENTS
            else "stop_natural_mixed_window_route"
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
        "# SWAN natural mixed-window feasibility preflight",
        "",
    ]
    if result["selected_documents"] < DOCUMENTS:
        lines.extend(
            [
                f"Natural window corpus gate: **{result['selected_documents']}/"
                f"{DOCUMENTS}** documents found.",
                "",
                f"- Eligible rows scanned: "
                f"**{result['screening']['eligible_rows_scanned']}**.",
                "- No API, AMR parser, S2Match, injection, detection, or attack "
                "call was run.",
                f"- Decision: `{result['decision']}`.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"Quality-valid windows: **{result['quality_pass']}/{DOCUMENTS}**; "
                f"quality plus graph-scale pass: **{result['joint_pass']}/"
                f"{DOCUMENTS}**.",
                "",
                "| Document | Words | Short slots | Quality | Scale-compatible | "
                "Graph | Joint |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in result["documents"]:
            graph = row["graph"]
            lines.append(
                f"| {row['prompt_id']} | {row['words']} | "
                f"{row['short_slots']} | "
                f"{'yes' if row['quality_pass'] else 'no'} | "
                f"{graph['scale_compatible'] if graph else 'n/a'} | "
                f"{'yes' if graph and graph['pass'] else 'no'} | "
                f"{'yes' if row['joint_pass'] else 'no'} |"
            )
        lines.extend(
            [
                "",
                f"- API calls: **{result['api_attempts']}**; local parser "
                f"sentences: **{result['neural_parser_sentences']}**; "
                "S2Match calls: **0**.",
                "- No watermark injection, detection, attack, TPR, FPR, or ASR "
                "was run.",
                f"- Decision: `{result['decision']}`.",
                "",
            ]
        )
    (folder / "report.md").write_text("\n".join(lines))


def runtime_path(folder):
    return folder / "runtime.json"


def run(folder, key):
    manifest = check(folder)
    freeze_documents(folder, manifest)
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
            runtime_path(folder),
            {
                "manifest_sha256": sha(folder / "manifest.json"),
                "model_returned": probe["model"],
                "probe_id": probe["id"],
            },
        )
        review_all(folder, manifest, api, runtime, heartbeat)
        parser_calls = parse_quality_documents(
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
