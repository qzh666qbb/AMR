"""Freeze and run a no-generation SWAN atomic-fact scale preflight."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time

from feedback2_runtime import Heartbeat, claim
from prepare_confirmation_experiment import SHARD, normalized_hash, stream_validation
from run_confirmation_experiment import freeze
from run_segmentation_intervention import REVISION, ROOT, amr_body, sha, text_sha, write_json
from run_swan_known_positive_audit import (
    BANK_PATH,
    MODEL_DIR,
    TOKENIZER_DIR,
    VECTORS_PATH,
    load_utils,
    pipeline,
    triple_counts,
)

OUTPUT = ROOT / "experiments/swan_atomic_fact_preflight1"
START_ROW = 1195
SCAN_LIMIT = 5000
DOCUMENTS = 5
MIN_FACTS = 10
MAX_FACTS = 14
MIN_TOTAL_WORDS = 100
MAX_TOTAL_WORDS = 180
MIN_FACT_WORDS = 8
MAX_FACT_WORDS = 18
MATCH_THRESHOLD = 0.75
MIN_COMPATIBLE_TEMPLATES = 10
MIN_COMPATIBLE_FRACTION = 0.8
MIN_TEMPLATE_TRIPLES = 6
MAX_TEMPLATE_TRIPLES = 20
MIN_DOCUMENT_MEDIAN = 8
MAX_DOCUMENT_MEDIAN = 16
MATCHER_PATH = (
    ROOT / "third_party/SWAN/amr-metric-suite/py3-Smatch-and-S2match/smatch"
)
BAD_START = {
    "and", "but", "can", "could", "did", "do", "does", "he", "her", "here",
    "his", "i", "if", "it", "its", "let", "may", "might", "my", "our", "she",
    "should", "so", "that", "their", "then", "there", "these", "they", "this",
    "those", "we", "will", "would", "yet", "you", "your",
}
FIRST_SECOND_PERSON = {"i", "me", "mine", "my", "our", "ours", "us", "we",
                       "you", "your", "yours"}


def configure_environment():
    os.environ.setdefault("NLTK_DATA", str(ROOT / ".nltk_data"))
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
    os.environ.setdefault("GLOVE_VECTORS_PATH", str(VECTORS_PATH))


def clean_token(token):
    return re.sub(r"[^A-Za-z]", "", token).lower()


def surface_fact_candidates(text, tokenize):
    rows = []
    for sentence_index, sentence in enumerate(tokenize(" ".join(text.split()))):
        words = sentence.split()
        normalized_tokens = {clean_token(token) for token in words}
        first = clean_token(words[0]) if words else ""
        reasons = []
        if not MIN_FACT_WORDS <= len(words) <= MAX_FACT_WORDS:
            reasons.append("words_outside_8_18")
        if first in BAD_START:
            reasons.append("excluded_initial_token")
        if FIRST_SECOND_PERSON & normalized_tokens:
            reasons.append("first_or_second_person")
        if sentence.count(",") > 1:
            reasons.append("multiple_commas")
        if any(character in sentence for character in ';?!:"\u201c\u201d'):
            reasons.append("excluded_punctuation")
        if re.search(r"\d", sentence):
            reasons.append("contains_digit")
        if re.search(r"&[A-Za-z#0-9]+;", sentence):
            reasons.append("html_entity")
        if "\ufffd" in sentence or "http" in sentence.lower() or "@" in sentence:
            reasons.append("malformed_or_link")
        if re.match(r"^\s*(?:[-*#]|\d+[.)])\s", sentence):
            reasons.append("list_prefix")
        if not reasons:
            rows.append({
                "sentence_index": sentence_index,
                "text": sentence,
                "text_sha256": text_sha(sentence),
                "normalized_sha256": normalized_hash(sentence),
                "words": len(words),
            })
    return rows


def syntax_accepts(doc):
    roots = [token for token in doc if token.dep_ == "ROOT" and token.pos_ in {"VERB", "AUX"}]
    if len(roots) != 1:
        return False
    root = roots[0]
    return any(
        token.dep_ in {"nsubj", "nsubjpass", "expl"} and token.head == root
        for token in doc
    )


def syntactic_fact_candidates(rows, nlp):
    if not rows:
        return []
    docs = nlp.pipe([row["text"] for row in rows], batch_size=64)
    accepted = []
    seen = set()
    for row, doc in zip(rows, docs):
        if syntax_accepts(doc) and row["normalized_sha256"] not in seen:
            accepted.append(row)
            seen.add(row["normalized_sha256"])
    return accepted


def choose_fact_prefix(candidates):
    chosen = []
    total_words = 0
    for row in candidates:
        if len(chosen) == MAX_FACTS:
            break
        chosen.append(row)
        total_words += row["words"]
        if len(chosen) >= MIN_FACTS and total_words >= MIN_TOTAL_WORDS:
            break
    eligible = (
        MIN_FACTS <= len(chosen) <= MAX_FACTS
        and MIN_TOTAL_WORDS <= total_words <= MAX_TOTAL_WORDS
    )
    return chosen, total_words, eligible


def screen_document(text, tokenize, nlp):
    normalized = " ".join(text.split())
    words = len(normalized.split())
    tail = normalized.rstrip(" \t\n\"'\u2019\u201d)]}")
    document_reasons = []
    if not 180 <= words <= 800:
        document_reasons.append("document_words_outside_180_800")
    if not tail.endswith((".", "!", "?")) or tail.endswith(("..", "\u2026")):
        document_reasons.append("incomplete_terminal")
    surface = surface_fact_candidates(normalized, tokenize)
    syntactic = syntactic_fact_candidates(surface, nlp)
    facts, fact_words, enough = choose_fact_prefix(syntactic)
    if not enough:
        document_reasons.append("insufficient_atomic_fact_prefix")
    return {
        "eligible": not document_reasons,
        "reasons": document_reasons,
        "words": words,
        "surface_candidates": len(surface),
        "syntactic_candidates": len(syntactic),
        "selected_fact_count": len(facts),
        "selected_fact_words": fact_words,
        "facts": facts,
    }


def select_sources(records, tokenize, nlp):
    selected = []
    rejected = []
    seen_text = set()
    seen_urls = set()
    scanned_to = None
    for row_index, row in enumerate(records):
        normalized = " ".join(row["text"].split())
        identity = normalized_hash(normalized)
        url = row.get("url", "")
        duplicate = identity in seen_text or bool(url and url in seen_urls)
        seen_text.add(identity)
        if url:
            seen_urls.add(url)
        if row_index < START_ROW:
            continue
        if row_index >= START_ROW + SCAN_LIMIT:
            break
        screening = screen_document(normalized, tokenize, nlp)
        reasons = list(screening["reasons"])
        if duplicate:
            reasons.append("duplicate_prior_row")
        identity_row = {
            "prompt_id": f"c4-validation-{row_index}",
            "shard_row": row_index,
            "source_sha256": text_sha(normalized),
            "normalized_sha256": identity,
            "url": url,
            "screening": {key: value for key, value in screening.items() if key != "facts"},
        }
        if reasons:
            rejected.append({**identity_row, "reasons": reasons})
        else:
            selected.append({
                **identity_row,
                "source": normalized,
                "facts": screening["facts"],
            })
        scanned_to = row_index
        if len(selected) == DOCUMENTS:
            break
    if len(selected) != DOCUMENTS:
        raise ValueError(
            f"Found {len(selected)}/{DOCUMENTS} eligible documents through row {scanned_to}"
        )
    return selected, rejected, scanned_to


def dependency_hashes():
    paths = [
        Path(__file__),
        OUTPUT / "protocol.md",
        BANK_PATH,
        ROOT / "third_party/SWAN/utils/amr_utils.py",
        ROOT / "third_party/SWAN/utils/s2match_patch.py",
        MATCHER_PATH / "amr_py3.py",
        MODEL_DIR / "amrlib_meta.json",
        MODEL_DIR / "model_parse_xfm_bart_large.json",
        MODEL_DIR / "config.json",
        MODEL_DIR / "pytorch_model.bin",
        TOKENIZER_DIR / "tokenizer.json",
        TOKENIZER_DIR / "tokenizer_config.json",
        VECTORS_PATH,
        ROOT / ".nltk_data/tokenizers/punkt_tab/english/abbrev_types.txt",
        ROOT / ".nltk_data/tokenizers/punkt_tab/english/collocations.tab",
        ROOT / ".nltk_data/tokenizers/punkt_tab/english/ortho_context.tab",
        ROOT / ".nltk_data/tokenizers/punkt_tab/english/sent_starters.txt",
    ]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def package_versions():
    return {
        name: importlib.metadata.version(name)
        for name in (
            "amrlib", "en_core_web_sm", "nltk", "numpy", "penman", "spacy",
            "torch", "transformers",
        )
    }


def prepare(folder):
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        return check(folder)
    import nltk
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    records = stream_validation()
    try:
        sources, rejected, scanned_to = select_sources(records, nltk.sent_tokenize, nlp)
    finally:
        records.close()
    manifest = {
        "status": "frozen_before_amr_preflight",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "allenai/c4",
            "revision": REVISION,
            "shard": SHARD,
            "development_exposure_through_row": START_ROW - 1,
            "first_eligible_row": START_ROW,
            "scan_limit": SCAN_LIMIT,
            "scanned_through_row": scanned_to,
            "dataset_stream_http_requests": 1,
        },
        "protocol": {
            "workflow": "swan_atomic_fact_scale_preflight",
            "documents": DOCUMENTS,
            "fact_count_range": [MIN_FACTS, MAX_FACTS],
            "document_word_range": [180, 800],
            "fact_word_range": [MIN_FACT_WORDS, MAX_FACT_WORDS],
            "selected_fact_total_word_range": [MIN_TOTAL_WORDS, MAX_TOTAL_WORDS],
            "match_threshold": MATCH_THRESHOLD,
            "single_fact_gate": {
                "template_triples": [MIN_TEMPLATE_TRIPLES, MAX_TEMPLATE_TRIPLES],
                "minimum_count_compatible_bank_templates": MIN_COMPATIBLE_TEMPLATES,
            },
            "document_gate": {
                "minimum_compatible_fact_fraction": MIN_COMPATIBLE_FRACTION,
                "template_triple_median": [MIN_DOCUMENT_MEDIAN, MAX_DOCUMENT_MEDIAN],
                "all_parser_outputs_required": True,
            },
            "continuation_gate": "At least 3/5 documents pass before a paid injection pilot may be designed",
            "network_generation_api_calls": 0,
            "s2match_optimization_calls": 0,
            "no_attack_asr_or_population_fpr": True,
        },
        "sources": sources,
        "screened_rejections": rejected,
        "dependencies": dependency_hashes(),
        "packages": package_versions(),
    }
    freeze(manifest_path, manifest)
    return manifest


def check(folder):
    import nltk
    import spacy

    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["protocol"]["workflow"] != "swan_atomic_fact_scale_preflight":
        raise ValueError("Unexpected workflow")
    for relative, expected in manifest["dependencies"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Frozen dependency changed: {relative}")
    if package_versions() != manifest["packages"]:
        raise ValueError("Frozen package versions changed")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    if len(manifest["sources"]) != DOCUMENTS:
        raise ValueError("Wrong source count")
    for source in manifest["sources"]:
        screening = screen_document(source["source"], nltk.sent_tokenize, nlp)
        expected = {key: value for key, value in screening.items() if key != "facts"}
        if (
            not screening["eligible"]
            or source["screening"] != expected
            or source["facts"] != screening["facts"]
            or text_sha(source["source"]) != source["source_sha256"]
            or normalized_hash(source["source"]) != source["normalized_sha256"]
            or source["shard_row"] < START_ROW
        ):
            raise ValueError(f"Frozen source mismatch: {source['prompt_id']}")
    return manifest


def s2_counts(amr):
    if str(MATCHER_PATH) not in sys.path:
        sys.path.insert(0, str(MATCHER_PATH))
    from amr_py3 import AMR

    graph = AMR.parse_AMR_line(amr, False)
    if graph is None:
        raise ValueError("S2Match parser returned no graph")
    return [len(items) for items in graph.get_triples()]


def count_upper_bound(left, right):
    total = sum(left) + sum(right)
    if not total:
        return 0.0
    return 2 * sum(min(a, b) for a, b in zip(left, right)) / total


def analyze_parser_output(fact, raw_output, utils, bank_counts):
    graph = amr_body(raw_output)
    transformed = pipeline(graph, utils)
    counts = s2_counts(transformed["final_amr"])
    bounds = [count_upper_bound(counts, target) for target in bank_counts]
    compatible = sum(value >= MATCH_THRESHOLD for value in bounds)
    total = sum(counts)
    return {
        **fact,
        "raw_parser_output": raw_output,
        "graph_amr": graph,
        "graph_sha256": text_sha(graph),
        "template_amr": transformed["final_amr"],
        "template_sha256": transformed["final_sha256"],
        "penman_counts": triple_counts(transformed["final_amr"]),
        "s2match_counts": counts,
        "template_triples": total,
        "count_upper_bounds": bounds,
        "maximum_count_upper_bound": max(bounds),
        "compatible_bank_templates": compatible,
        "scale_compatible": (
            MIN_TEMPLATE_TRIPLES <= total <= MAX_TEMPLATE_TRIPLES
            and compatible >= MIN_COMPATIBLE_TEMPLATES
        ),
    }


def run_parser(folder):
    manifest = check(folder)
    pending = [
        source for source in manifest["sources"]
        if not (folder / "parser" / f"{source['prompt_id']}.json").exists()
    ]
    if not pending:
        return summarize(folder)

    import amrlib
    import torch

    utils = load_utils()
    bank = json.loads(BANK_PATH.read_text())
    bank_counts = [s2_counts(row["normalized_template_amr"]) for row in bank]
    torch.set_num_threads(4)
    parser = amrlib.load_stog_model(
        model_dir=str(MODEL_DIR),
        tok_name_or_path=str(TOKENIZER_DIR),
        device="cpu",
        batch_size=4,
    )
    started = time.monotonic()
    parsed_facts = 0
    with Heartbeat(folder) as heartbeat:
        for document_index, source in enumerate(manifest["sources"]):
            path = folder / "parser" / f"{source['prompt_id']}.json"
            if path.exists():
                continue
            sentences = [fact["text"] for fact in source["facts"]]
            token_lengths = [len(parser.tokenizer(sentence)["input_ids"]) for sentence in sentences]
            if any(length > parser.max_sent_len for length in token_lengths):
                raise ValueError(f"Tokenizer limit exceeded: {source['prompt_id']}")
            heartbeat.update(
                stage="local_amr_parser",
                prompt_id=source["prompt_id"],
                completed_documents=document_index,
                completed_facts=parsed_facts,
            )
            document_started = time.monotonic()
            outputs = parser.parse_sents(sentences)
            if len(outputs) != len(sentences) or any(not output for output in outputs):
                raise ValueError(f"Incomplete parser output: {source['prompt_id']}")
            rows = []
            for fact, tokens, raw_output in zip(source["facts"], token_lengths, outputs):
                row = analyze_parser_output(fact, raw_output, utils, bank_counts)
                row["input_tokens"] = tokens
                rows.append(row)
            parsed_facts += len(rows)
            freeze(path, {
                "prompt_id": source["prompt_id"],
                "source_sha256": source["source_sha256"],
                "facts": rows,
                "elapsed_seconds": time.monotonic() - document_started,
            })
            print(json.dumps({
                "stage": "local_amr_parser",
                "completed_documents": document_index + 1,
                "total_documents": DOCUMENTS,
                "prompt_id": source["prompt_id"],
                "facts": len(rows),
                "scale_compatible": sum(row["scale_compatible"] for row in rows),
            }), flush=True)
        heartbeat.update(
            stage="complete",
            completed_documents=DOCUMENTS,
            completed_facts=sum(len(source["facts"]) for source in manifest["sources"]),
            new_generation_api_calls=0,
            new_s2match_optimization_calls=0,
        )
    write_json(folder / "runtime.json", {
        "elapsed_seconds": time.monotonic() - started,
        "new_neural_parser_calls": parsed_facts,
        "new_generation_api_calls": 0,
        "new_s2match_optimization_calls": 0,
    })
    return summarize(folder)


def document_summary(source, parsed):
    rows = parsed["facts"]
    triples = [row["template_triples"] for row in rows]
    compatible = sum(row["scale_compatible"] for row in rows)
    fraction = compatible / len(rows)
    median = statistics.median(triples)
    passed = (
        len(rows) == len(source["facts"])
        and fraction >= MIN_COMPATIBLE_FRACTION
        and MIN_DOCUMENT_MEDIAN <= median <= MAX_DOCUMENT_MEDIAN
    )
    return {
        "prompt_id": source["prompt_id"],
        "shard_row": source["shard_row"],
        "facts": len(rows),
        "fact_words": sum(row["words"] for row in rows),
        "parsed": len(rows),
        "scale_compatible_facts": compatible,
        "scale_compatible_fraction": fraction,
        "template_triples": {
            "mean": statistics.mean(triples),
            "median": median,
            "min": min(triples),
            "max": max(triples),
        },
        "compatible_bank_templates": {
            "mean": statistics.mean(row["compatible_bank_templates"] for row in rows),
            "median": statistics.median(row["compatible_bank_templates"] for row in rows),
            "min": min(row["compatible_bank_templates"] for row in rows),
            "max": max(row["compatible_bank_templates"] for row in rows),
        },
        "document_gate": passed,
    }


def write_report(folder, result):
    lines = [
        "# SWAN 原子事实正常长度基线离线预检",
        "",
        "日期：2026-09-15。状态："
        + ("完成并达到继续门。" if result["status"] == "complete" and result["continuation_gate"] else "完成但未达到继续门。"),
        "",
        "本轮只检查未见源文档能否提供足量短事实句，以及这些句子的本地 AMR 图规模是否与 bank50 兼容。"
        "没有生成 API、S2Match 优化、攻击、ASR 或总体 FPR。",
        "",
        "| 文档 | 原文行 | 事实句 | 事实词数 | 兼容句 | 模板图三元组中位数 | 兼容模板数中位数 | 文档门 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["documents"]:
        lines.append(
            f"| {row['prompt_id']} | {row['shard_row']} | {row['facts']} | {row['fact_words']} | "
            f"{row['scale_compatible_facts']}/{row['facts']} | "
            f"{row['template_triples']['median']:.1f} | "
            f"{row['compatible_bank_templates']['median']:.1f} | "
            f"{'通过' if row['document_gate'] else '未通过'} |"
        )
    lines += [
        "",
        f"- 文档门通过：**{result['documents_passing']}/{DOCUMENTS}**。",
        f"- 继续门：**{'通过' if result['continuation_gate'] else '未通过'}**；"
        + ("可以另行冻结小规模原子事实注入 pilot。" if result["continuation_gate"]
           else "停止当前事实筛选/表示，不发起付费注入。"),
        f"- 新增本地神经 parser 调用：**{result['new_neural_parser_calls']}**；"
        "新增生成 API 与 S2Match 优化调用均为 **0**。",
        "",
        "计数上界是必要条件，不是实际匹配分数。即使通过，本轮仍没有建立成功注入且可检测的正常长度水印基线。",
        "数据入口：[manifest.json](manifest.json)、[summary.json](summary.json)、[audit.json](audit.json)。",
        "",
    ]
    (folder / "report.md").write_text("\n".join(lines))


def summarize(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    documents = []
    for source in manifest["sources"]:
        path = folder / "parser" / f"{source['prompt_id']}.json"
        if path.exists():
            documents.append(document_summary(source, json.loads(path.read_text())))
    complete = len(documents) == DOCUMENTS
    passing = sum(row["document_gate"] for row in documents)
    runtime_path = folder / "runtime.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    result = {
        "status": "complete" if complete else "incomplete",
        "documents": documents,
        "documents_passing": passing,
        "continuation_gate": bool(complete and passing >= 3),
        "decision": (
            "prepare_bounded_atomic_injection_pilot"
            if complete and passing >= 3
            else "stop_current_atomic_representation"
            if complete
            else "pending"
        ),
        "new_neural_parser_calls": runtime.get("new_neural_parser_calls", 0),
        "new_generation_api_calls": 0,
        "new_s2match_optimization_calls": 0,
        "no_attack_asr_or_population_fpr": True,
        "manifest_sha256": sha(folder / "manifest.json"),
    }
    write_json(folder / "summary.json", result)
    if complete:
        write_report(folder, result)
    return result


def audit(folder):
    manifest = check(folder)
    utils = load_utils()
    bank = json.loads(BANK_PATH.read_text())
    bank_counts = [s2_counts(row["normalized_template_amr"]) for row in bank]
    checked = 0
    for source in manifest["sources"]:
        parsed = json.loads((folder / "parser" / f"{source['prompt_id']}.json").read_text())
        if (
            parsed["prompt_id"] != source["prompt_id"]
            or parsed["source_sha256"] != source["source_sha256"]
            or len(parsed["facts"]) != len(source["facts"])
        ):
            raise ValueError("Parser checkpoint identity mismatch")
        for expected, row in zip(source["facts"], parsed["facts"]):
            reconstructed = analyze_parser_output(expected, row["raw_parser_output"], utils, bank_counts)
            reconstructed["input_tokens"] = row["input_tokens"]
            if reconstructed != {key: value for key, value in row.items()}:
                raise ValueError(f"Parser-derived row mismatch: {source['prompt_id']}")
            checked += 1
    result = summarize(folder)
    if result["status"] != "complete":
        raise ValueError("Cannot audit incomplete preflight")
    audit_result = {
        "status": "passed",
        "mismatches": 0,
        "documents_checked": DOCUMENTS,
        "facts_checked": checked,
        "new_dataset_http_requests": 0,
        "new_generation_api_calls": 0,
        "new_neural_parser_calls": 0,
        "new_s2match_optimization_calls": 0,
        "summary_sha256": sha(folder / "summary.json"),
        "limitation": "Saved neural parser outputs were transformed and recounted; the neural parser was not rerun.",
    }
    write_json(folder / "audit.json", audit_result)
    return audit_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "summarize", "audit"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    configure_environment()
    with claim(args.output / ".runner.lock"):
        if args.stage == "prepare":
            result = prepare(args.output)
            output = {
                "status": "prepared",
                "sources": [source["prompt_id"] for source in result["sources"]],
                "facts": sum(len(source["facts"]) for source in result["sources"]),
                "scanned_through_row": result["dataset"]["scanned_through_row"],
                "manifest_sha256": sha(args.output / "manifest.json"),
                "new_generation_api_calls": 0,
            }
        elif args.stage == "run":
            output = run_parser(args.output)
        elif args.stage == "audit":
            output = audit(args.output)
        else:
            output = summarize(args.output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
