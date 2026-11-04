"""Post-hoc AMR shape diagnostic for saved microfact generations."""

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
    BANK_PATH,
    MODEL_DIR,
    TOKENIZER_DIR,
    VECTORS_PATH,
    analyze_parser_output,
    configure_environment,
    s2_counts,
)
from run_swan_known_positive_audit import load_utils
from run_swan_microfact_generation import FACTS, OUTPUT, decode_microfacts, strip_fence

MANIFEST = OUTPUT / "exploratory_scale_manifest.json"
SUMMARY = OUTPUT / "exploratory_scale_summary.json"
REPORT = OUTPUT / "exploratory_scale_report.md"
PARSER_DIR = OUTPUT / "exploratory_parser"


def recover_microfacts(content):
    try:
        return decode_microfacts(content), "none"
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        stripped = strip_fence(content)
        if not stripped.startswith('{"facts":[') or not stripped.endswith("}"):
            raise ValueError("Output is not eligible for the one-character repair") from None
        repaired = stripped[:-1] + "]}"
        facts = decode_microfacts(repaired)
        return facts, "insert_array_close_before_final_object_close"


def dependencies():
    paths = [
        Path(__file__),
        OUTPUT / "exploratory_scale_protocol.md",
        OUTPUT / "manifest.json",
        OUTPUT / "summary.json",
        OUTPUT / "audit.json",
        ROOT / "scripts/run_swan_microfact_generation.py",
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
    experiment_manifest = json.loads((OUTPUT / "manifest.json").read_text())
    rows = []
    for source in experiment_manifest["sources"]:
        path = OUTPUT / "generated" / f"{source['prompt_id']}.json"
        generated = json.loads(path.read_text())
        facts, repair = recover_microfacts(generated["response"]["content"])
        if len(facts) != FACTS:
            raise ValueError("Exploratory recovery did not yield exactly 15 facts")
        rows.append({
            "prompt_id": source["prompt_id"],
            "source_sha256": source["source_sha256"],
            "generated_sha256": sha(path),
            "response_content_sha256": text_sha(generated["response"]["content"]),
            "repair": repair,
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


def document_fact_count(rows):
    return sum(len(row["facts"]) for row in rows)


def prepare(folder=OUTPUT):
    if folder != OUTPUT:
        raise ValueError("This diagnostic is bound to its frozen parent directory")
    if MANIFEST.exists():
        return check(folder)
    rows = source_rows()
    manifest = {
        "status": "frozen_before_exploratory_parser",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Post-hoc diagnostic; cannot alter the parent experiment's 0/5 decision",
        "documents": rows,
        "protocol": {
            "facts": document_fact_count(rows),
            "single_fact_scale_gate": "6..20 template triples and >=10 count-compatible bank templates",
            "strict_word_range": [7, 10],
            "hypothesis_gate": {
                "strict_word_scale_compatible_fraction": 0.7,
                "documents_with_at_least_12_of_15_scale_compatible": 3,
            },
            "new_generation_api_calls": 0,
            "new_s2match_optimization_calls": 0,
        },
        "dependencies": dependencies(),
        "packages": package_versions(),
    }
    freeze(MANIFEST, manifest)
    return manifest


def check(folder=OUTPUT):
    if folder != OUTPUT:
        raise ValueError("This diagnostic is bound to its frozen parent directory")
    manifest = json.loads(MANIFEST.read_text())
    if manifest["documents"] != source_rows():
        raise ValueError("Frozen repaired facts changed")
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
    token_lengths = [len(parser.tokenizer(row["text"])["input_ids"]) for row in facts]
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
        "repair": source["repair"],
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
            "template_triples_mean": None,
            "template_triples_median": None,
            "compatible_bank_templates_mean": None,
        }
    compatible = sum(row["scale_compatible"] for row in rows)
    return {
        "facts": len(rows),
        "scale_compatible": compatible,
        "scale_compatible_fraction": compatible / len(rows),
        "template_triples_mean": statistics.mean(row["template_triples"] for row in rows),
        "template_triples_median": statistics.median(row["template_triples"] for row in rows),
        "compatible_bank_templates_mean": statistics.mean(
            row["compatible_bank_templates"] for row in rows
        ),
    }


def summarize(folder=OUTPUT):
    if folder != OUTPUT:
        raise ValueError("This diagnostic is bound to its frozen parent directory")
    manifest = json.loads(MANIFEST.read_text())
    document_rows = []
    all_rows = []
    for source in manifest["documents"]:
        path = PARSER_DIR / f"{source['prompt_id']}.json"
        if not path.exists():
            continue
        parsed = json.loads(path.read_text())
        all_rows.extend(parsed["facts"])
        metrics = describe(parsed["facts"])
        document_rows.append({
            "prompt_id": source["prompt_id"],
            "repair": source["repair"],
            **metrics,
            "at_least_12_of_15_scale_compatible": metrics["scale_compatible"] >= 12,
        })
    strict = [row for row in all_rows if 7 <= row["words"] <= 10]
    over = [row for row in all_rows if row["words"] > 10]
    strict_metrics = describe(strict)
    document_gate_count = sum(
        row["at_least_12_of_15_scale_compatible"] for row in document_rows
    )
    complete = len(document_rows) == len(manifest["documents"])
    hypothesis_gate = bool(
        complete
        and strict_metrics["scale_compatible_fraction"] is not None
        and strict_metrics["scale_compatible_fraction"] >= 0.7
        and document_gate_count >= 3
    )
    runtime_path = folder / "exploratory_scale_runtime.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    result = {
        "status": "complete_exploratory_posthoc" if complete else "incomplete",
        "parent_confirmatory_decision_unchanged": True,
        "documents": document_rows,
        "overall": describe(all_rows),
        "strict_7_10_words": strict_metrics,
        "over_10_words": describe(over),
        "documents_with_at_least_12_of_15_scale_compatible": document_gate_count,
        "hypothesis_gate": hypothesis_gate,
        "decision": (
            "sequential_structure_only_retry_worth_pre_registration"
            if hypothesis_gate
            else "do_not_advance_to_sequential_microfact_generation"
            if complete
            else "pending"
        ),
        "new_neural_parser_calls": runtime.get("new_neural_parser_calls", 0),
        "new_generation_api_calls": 0,
        "new_s2match_optimization_calls": 0,
        "manifest_sha256": sha(MANIFEST),
    }
    write_json(SUMMARY, result)
    if complete:
        write_report(result)
    return result


def write_report(result):
    lines = [
        "# Microfact 生成输出的后验图规模诊断",
        "",
        "该分析不修改父实验的结构失败与 `0/5` 联合通过结论。289、292 只在诊断副本中"
        "补入缺失的数组闭合字符，原始响应保持不变。",
        "",
        "| 文档 | JSON 修复 | 规模兼容 | 三元组中位数 | 兼容模板数均值 | 12/15 门 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in result["documents"]:
        lines.append(
            f"| {row['prompt_id']} | {row['repair']} | "
            f"{row['scale_compatible']}/{row['facts']} | "
            f"{row['template_triples_median']:.1f} | "
            f"{row['compatible_bank_templates_mean']:.3f} | "
            f"{'是' if row['at_least_12_of_15_scale_compatible'] else '否'} |"
        )
    strict = result["strict_7_10_words"]
    over = result["over_10_words"]
    lines += [
        "",
        f"- 严格 7--10 词句：**{strict['scale_compatible']}/{strict['facts']}** "
        f"({strict['scale_compatible_fraction']:.1%}) 规模兼容。",
        f"- 超过 10 词句：**{over['scale_compatible']}/{over['facts']}** "
        f"({over['scale_compatible_fraction']:.1%}) 规模兼容。",
        f"- 达到 12/15 规模门的文档："
        f"**{result['documents_with_at_least_12_of_15_scale_compatible']}/5**。",
        f"- 新假设门：**{'通过' if result['hypothesis_gate'] else '未通过'}**。",
        f"- 新增本地神经 parser 调用：**{result['new_neural_parser_calls']}**；"
        "生成 API 与 S2Match 优化调用均为 **0**。",
        "",
        "该结果只说明保存句子的图规模，不证明实际模板匹配、注入成功、质量合格或可检测。",
        "",
    ]
    REPORT.write_text("\n".join(lines))


def run(folder=OUTPUT):
    manifest = check(folder)
    pending = [
        source for source in manifest["documents"]
        if not (PARSER_DIR / f"{source['prompt_id']}.json").exists()
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
            parsed, added = parse_document(source, parser, utils, bank_counts)
            parser_calls += added
            print(json.dumps({
                "stage": "exploratory_local_amr_parser",
                "completed": index + 1,
                "total": len(manifest["documents"]),
                "prompt_id": source["prompt_id"],
                "facts": len(parsed["facts"]),
                "scale_compatible": sum(row["scale_compatible"] for row in parsed["facts"]),
            }), flush=True)
        heartbeat.update(
            stage="exploratory_complete",
            new_neural_parser_calls=parser_calls,
            new_generation_api_calls=0,
            new_s2match_optimization_calls=0,
        )
    total_parser_facts = sum(
        len(json.loads(path.read_text())["facts"])
        for path in PARSER_DIR.glob("*.json")
    )
    write_json(folder / "exploratory_scale_runtime.json", {
        "elapsed_seconds": time.monotonic() - started,
        "new_neural_parser_calls": total_parser_facts,
        "new_generation_api_calls": 0,
        "new_s2match_optimization_calls": 0,
    })
    return summarize(folder)


def audit(folder=OUTPUT):
    manifest = check(folder)
    utils = load_utils()
    bank = json.loads(BANK_PATH.read_text())
    bank_counts = [s2_counts(row["normalized_template_amr"]) for row in bank]
    facts_checked = 0
    for source in manifest["documents"]:
        parsed = json.loads((PARSER_DIR / f"{source['prompt_id']}.json").read_text())
        if (
            parsed["prompt_id"] != source["prompt_id"]
            or parsed["source_sha256"] != source["source_sha256"]
            or parsed["generated_sha256"] != source["generated_sha256"]
            or parsed["repair"] != source["repair"]
        ):
            raise ValueError("Exploratory parser checkpoint identity mismatch")
        rows = []
        for fact, saved in zip(source["facts"], parsed["facts"]):
            row = analyze_parser_output(
                fact, saved["raw_parser_output"], utils, bank_counts
            )
            row["input_tokens"] = saved["input_tokens"]
            rows.append(row)
        if rows != parsed["facts"] or len(rows) != FACTS:
            raise ValueError("Exploratory parser-derived rows changed")
        facts_checked += len(rows)
    summary = summarize(folder)
    if summary["status"] != "complete_exploratory_posthoc":
        raise ValueError("Cannot audit incomplete diagnostic")
    result = {
        "status": "passed",
        "mismatches": 0,
        "documents_checked": len(manifest["documents"]),
        "facts_checked": facts_checked,
        "new_api_calls": 0,
        "new_neural_parser_calls": 0,
        "new_s2match_optimization_calls": 0,
        "summary_sha256": sha(SUMMARY),
        "limitation": "Saved neural parser outputs were transformed and recounted; neural inference was not rerun.",
    }
    write_json(folder / "exploratory_scale_audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "summarize", "audit"))
    args = parser.parse_args()
    configure_environment()
    with claim(OUTPUT / ".exploratory_scale.lock"):
        if args.stage == "prepare":
            result = prepare()
            output = {
                "status": "prepared",
                "documents": len(result["documents"]),
                "facts": result["protocol"]["facts"],
                "manifest_sha256": sha(MANIFEST),
            }
        elif args.stage == "run":
            output = run()
        elif args.stage == "audit":
            output = audit()
        else:
            output = summarize()
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
