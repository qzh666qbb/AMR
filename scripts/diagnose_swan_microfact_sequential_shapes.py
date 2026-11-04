"""Post-hoc AMR graph-scale diagnostic for sequential microfact slots."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import statistics
import time

from feedback2_runtime import Heartbeat, claim
from run_confirmation_experiment import freeze
from run_segmentation_intervention import ROOT, sha, text_sha, write_json
from run_swan_atomic_fact_preflight import (
    MODEL_DIR,
    TOKENIZER_DIR,
    VECTORS_PATH,
    analyze_parser_output,
    configure_environment,
    s2_counts,
)
from run_swan_bank100_microfact_null import BANK_PATH
from run_swan_known_positive_audit import load_utils

OUTPUT = ROOT / "experiments/swan_microfact_sequential2"
MANIFEST = OUTPUT / "exploratory_scale_manifest.json"
SUMMARY = OUTPUT / "exploratory_scale_summary.json"
REPORT = OUTPUT / "exploratory_scale_report.md"
PARSER_DIR = OUTPUT / "exploratory_parser"
FACTS = 15


def dependencies():
    paths = [
        Path(__file__),
        OUTPUT / "exploratory_scale_protocol.md",
        OUTPUT / "manifest.json",
        OUTPUT / "summary.json",
        OUTPUT / "audit.json",
        ROOT / "scripts/run_swan_microfact_sequential.py",
        ROOT / "scripts/run_swan_atomic_fact_preflight.py",
        ROOT / "scripts/run_swan_known_positive_audit.py",
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
    paths += sorted((OUTPUT / "generated").glob("*.json"))
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def package_versions():
    return {
        name: importlib.metadata.version(name)
        for name in ("amrlib", "numpy", "penman", "torch", "transformers")
    }


def source_rows():
    parent = json.loads((OUTPUT / "manifest.json").read_text())
    rows = []
    for source in parent["sources"]:
        path = OUTPUT / "generated" / f"{source['prompt_id']}.json"
        generated = json.loads(path.read_text())
        facts = generated["facts"]
        if len(facts) != FACTS:
            raise ValueError("Expected exactly 15 frozen slot outputs")
        rows.append({
            "prompt_id": source["prompt_id"],
            "source_sha256": source["source_sha256"],
            "generated_sha256": sha(path),
            "facts": [
                {
                    "fact_index": index,
                    "text": fact,
                    "text_sha256": text_sha(fact),
                    "words": len(fact.split()),
                }
                for index, fact in enumerate(facts)
            ],
        })
    return rows


def prepare(folder=OUTPUT):
    if folder != OUTPUT:
        raise ValueError("Diagnostic is bound to its parent experiment")
    if MANIFEST.exists():
        return check(folder)
    rows = source_rows()
    manifest = {
        "status": "frozen_before_exploratory_parser",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Post-hoc diagnostic; parent 0/5 and stop decision cannot change",
        "documents": rows,
        "protocol": {
            "facts": sum(len(row["facts"]) for row in rows),
            "word_strata": ["under_7", "7_10", "over_10"],
            "single_fact_scale_gate": (
                "6..20 transformed triples and at least 10 bank100 "
                "count-compatible templates at U>=0.75"
            ),
            "exploratory_gate": {
                "strict_word_scale_compatible_fraction": 0.7,
                "documents_with_at_least_10_of_15_scale_compatible": 3,
            },
            "new_generation_review_s2match_calls": 0,
        },
        "dependencies": dependencies(),
        "packages": package_versions(),
    }
    freeze(MANIFEST, manifest)
    return manifest


def check(folder=OUTPUT):
    if folder != OUTPUT:
        raise ValueError("Diagnostic is bound to its parent experiment")
    manifest = json.loads(MANIFEST.read_text())
    if manifest["documents"] != source_rows():
        raise ValueError("Frozen sequential facts changed")
    for relative, expected in manifest["dependencies"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Frozen dependency changed: {relative}")
    if package_versions() != manifest["packages"]:
        raise ValueError("Frozen package versions changed")
    return manifest


def parse_document(source, parser, utils, bank_counts):
    path = PARSER_DIR / f"{source['prompt_id']}.json"
    if path.exists():
        return json.loads(path.read_text()), 0
    facts = source["facts"]
    token_lengths = [
        len(parser.tokenizer(row["text"])["input_ids"])
        for row in facts
    ]
    if any(length > parser.max_sent_len for length in token_lengths):
        raise ValueError(f"Tokenizer limit exceeded: {source['prompt_id']}")
    started = time.monotonic()
    outputs = parser.parse_sents([row["text"] for row in facts])
    if len(outputs) != FACTS or any(not output for output in outputs):
        raise ValueError(f"Incomplete parser output: {source['prompt_id']}")
    rows = []
    for fact, tokens, raw_output in zip(facts, token_lengths, outputs):
        row = analyze_parser_output(fact, raw_output, utils, bank_counts)
        row["input_tokens"] = tokens
        rows.append(row)
    result = {
        "prompt_id": source["prompt_id"],
        "source_sha256": source["source_sha256"],
        "generated_sha256": source["generated_sha256"],
        "facts": rows,
        "elapsed_seconds": time.monotonic() - started,
    }
    freeze(path, result)
    return result, len(rows)


def describe(rows):
    if not rows:
        return {
            "facts": 0,
            "scale_compatible": 0,
            "scale_compatible_fraction": None,
            "count_feasible": 0,
            "template_triples_mean": None,
            "template_triples_median": None,
            "compatible_bank_templates_mean": None,
        }
    compatible = sum(row["scale_compatible"] for row in rows)
    return {
        "facts": len(rows),
        "scale_compatible": compatible,
        "scale_compatible_fraction": compatible / len(rows),
        "count_feasible": sum(
            row["compatible_bank_templates"] > 0 for row in rows
        ),
        "template_triples_mean": statistics.mean(
            row["template_triples"] for row in rows
        ),
        "template_triples_median": statistics.median(
            row["template_triples"] for row in rows
        ),
        "compatible_bank_templates_mean": statistics.mean(
            row["compatible_bank_templates"] for row in rows
        ),
    }


def summarize(folder=OUTPUT):
    if folder != OUTPUT:
        raise ValueError("Diagnostic is bound to its parent experiment")
    manifest = json.loads(MANIFEST.read_text())
    documents = []
    all_rows = []
    for source in manifest["documents"]:
        path = PARSER_DIR / f"{source['prompt_id']}.json"
        if not path.exists():
            continue
        parsed = json.loads(path.read_text())
        all_rows.extend(parsed["facts"])
        metrics = describe(parsed["facts"])
        documents.append({
            "prompt_id": source["prompt_id"],
            **metrics,
            "at_least_10_of_15_scale_compatible": (
                metrics["scale_compatible"] >= 10
            ),
        })
    strict = [row for row in all_rows if 7 <= row["words"] <= 10]
    under = [row for row in all_rows if row["words"] < 7]
    over = [row for row in all_rows if row["words"] > 10]
    strict_metrics = describe(strict)
    document_gate_count = sum(
        row["at_least_10_of_15_scale_compatible"]
        for row in documents
    )
    complete = len(documents) == len(manifest["documents"])
    exploratory_gate = bool(
        complete
        and strict_metrics["scale_compatible_fraction"] is not None
        and strict_metrics["scale_compatible_fraction"] >= 0.7
        and document_gate_count >= 3
    )
    runtime_path = folder / "exploratory_scale_runtime.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    result = {
        "status": "complete_exploratory_posthoc" if complete else "incomplete",
        "parent_decision_unchanged": True,
        "documents": documents,
        "overall": describe(all_rows),
        "under_7_words": describe(under),
        "strict_7_10_words": strict_metrics,
        "over_10_words": describe(over),
        "documents_with_at_least_10_of_15_scale_compatible": document_gate_count,
        "exploratory_gate": exploratory_gate,
        "decision": (
            "graph_scale_supports_separate_deterministic_shortening_study"
            if exploratory_gate
            else "graph_scale_does_not_support_advancement"
            if complete
            else "pending"
        ),
        "new_neural_parser_calls": runtime.get("new_neural_parser_calls", 0),
        "new_generation_review_s2match_calls": 0,
        "manifest_sha256": sha(MANIFEST),
    }
    write_json(SUMMARY, result)
    if complete:
        write_report(result)
    return result


def metric_line(label, metrics):
    fraction = metrics["scale_compatible_fraction"]
    fraction_text = "n/a" if fraction is None else f"{fraction:.1%}"
    return (
        f"- {label}: **{metrics['scale_compatible']}/{metrics['facts']}** "
        f"({fraction_text}) scale-compatible."
    )


def write_report(result):
    lines = [
        "# Sequential microfact post-hoc AMR graph-scale diagnostic",
        "",
        "This analysis does not change the parent experiment's 0/5 quality result "
        "or stop decision.",
        "",
        "| Document | Scale-compatible | Median triples | Compatible templates mean | 10/15 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in result["documents"]:
        lines.append(
            f"| {row['prompt_id']} | {row['scale_compatible']}/"
            f"{row['facts']} | {row['template_triples_median']:.1f} | "
            f"{row['compatible_bank_templates_mean']:.3f} | "
            f"{'yes' if row['at_least_10_of_15_scale_compatible'] else 'no'} |"
        )
    lines += [
        "",
        metric_line("Under 7 words", result["under_7_words"]),
        metric_line("Strict 7--10 words", result["strict_7_10_words"]),
        metric_line("Over 10 words", result["over_10_words"]),
        f"- Documents reaching 10/15: "
        f"**{result['documents_with_at_least_10_of_15_scale_compatible']}/5**.",
        f"- Exploratory gate: "
        f"**{'passed' if result['exploratory_gate'] else 'failed'}**.",
        f"- New local parser calls: **{result['new_neural_parser_calls']}**; "
        "new generation, review, S2Match, watermark, detector, and attack calls: "
        "**0**.",
        f"- Decision: `{result['decision']}`.",
        "",
    ]
    REPORT.write_text("\n".join(lines))


def run(folder=OUTPUT):
    manifest = check(folder)
    pending = [
        source
        for source in manifest["documents"]
        if not (PARSER_DIR / f"{source['prompt_id']}.json").exists()
    ]
    if not pending:
        return summarize(folder)
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
    parser_calls = 0
    started = time.monotonic()
    with Heartbeat(folder) as heartbeat:
        for index, source in enumerate(manifest["documents"]):
            heartbeat.update(
                stage="exploratory_local_amr_parser",
                prompt_id=source["prompt_id"],
                completed_documents=index,
                parser_calls=parser_calls,
            )
            parsed, added = parse_document(
                source,
                parser,
                utils,
                bank_counts,
            )
            parser_calls += added
            print(json.dumps({
                "stage": "exploratory_local_amr_parser",
                "completed": index + 1,
                "total": len(manifest["documents"]),
                "prompt_id": source["prompt_id"],
                "facts": len(parsed["facts"]),
                "scale_compatible": sum(
                    row["scale_compatible"] for row in parsed["facts"]
                ),
            }), flush=True)
        heartbeat.update(
            stage="exploratory_complete",
            new_neural_parser_calls=parser_calls,
            new_generation_review_s2match_calls=0,
        )
    total_parser_facts = sum(
        len(json.loads(path.read_text())["facts"])
        for path in PARSER_DIR.glob("*.json")
    )
    write_json(folder / "exploratory_scale_runtime.json", {
        "elapsed_seconds": time.monotonic() - started,
        "new_neural_parser_calls": total_parser_facts,
        "new_generation_review_s2match_calls": 0,
    })
    return summarize(folder)


def audit(folder=OUTPUT):
    manifest = check(folder)
    utils = load_utils()
    bank = json.loads(BANK_PATH.read_text())
    bank_counts = [
        s2_counts(row["normalized_template_amr"])
        for row in bank
    ]
    checked = 0
    for source in manifest["documents"]:
        saved = json.loads(
            (PARSER_DIR / f"{source['prompt_id']}.json").read_text()
        )
        rebuilt = []
        for fact, row in zip(source["facts"], saved["facts"]):
            derived = analyze_parser_output(
                fact,
                row["raw_parser_output"],
                utils,
                bank_counts,
            )
            derived["input_tokens"] = row["input_tokens"]
            rebuilt.append(derived)
        if (
            rebuilt != saved["facts"]
            or len(rebuilt) != FACTS
            or saved["generated_sha256"] != source["generated_sha256"]
        ):
            raise ValueError("Exploratory parser checkpoint mismatch")
        checked += len(rebuilt)
    result = summarize(folder)
    if (
        result["status"] != "complete_exploratory_posthoc"
        or checked != manifest["protocol"]["facts"]
    ):
        raise ValueError("Exploratory summary mismatch")
    audit_result = {
        "status": "passed",
        "mismatches": 0,
        "parser_sentences_reconstructed": checked,
        "new_api_calls": 0,
        "new_neural_parser_calls": 0,
        "new_s2match_calls": 0,
        "summary_sha256": sha(SUMMARY),
    }
    write_json(folder / "exploratory_scale_audit.json", audit_result)
    return audit_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "summarize", "audit"))
    args = parser.parse_args()
    configure_environment()
    with claim(OUTPUT / ".exploratory_scale.lock"):
        if args.stage == "prepare":
            result = prepare()
        elif args.stage == "run":
            result = run()
        elif args.stage == "audit":
            result = audit()
        else:
            result = summarize()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
