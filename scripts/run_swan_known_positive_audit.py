"""Layered offline known-positive audit for the local SWAN pipeline."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

from run_confirmation_experiment import freeze
from run_segmentation_intervention import amr_body, ROOT, sha, text_sha, write_json

OUTPUT = ROOT / "experiments/swan_known_positive1"
SWAN = ROOT / "third_party/SWAN"
BANK_PATH = SWAN / "amr_bank/banks/amr_bank_50.json"
GROUPS_PATH = SWAN / "amr_bank/artifacts/normalized_template_groups.json"
EXAMPLES_PATH = SWAN / "amr_bank/artifacts/normalized_amr_examples.json"
MODEL_DIR = ROOT / "models/model_parse_xfm_bart_large-v0_1_0"
TOKENIZER_DIR = ROOT / "models/bart-large-tokenizer"
VECTORS_PATH = ROOT / "vectors/glove.6B.100d.txt"
SELF_REPEATS = 5


def configure_environment():
    os.environ.setdefault("NLTK_DATA", str(ROOT / ".nltk_data"))
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
    os.environ.setdefault("GLOVE_VECTORS_PATH", str(VECTORS_PATH))


def dependencies():
    paths = [
        Path(__file__),
        OUTPUT / "protocol.md",
        BANK_PATH,
        GROUPS_PATH,
        EXAMPLES_PATH,
        SWAN / "utils/amr_utils.py",
        SWAN / "utils/s2match_patch.py",
        SWAN / "amr-metric-suite/py3-Smatch-and-S2match/smatch/s2match.py",
        SWAN / "amr-metric-suite/py3-Smatch-and-S2match/smatch/amr_py3.py",
        MODEL_DIR / "amrlib_meta.json",
        MODEL_DIR / "model_parse_xfm_bart_large.json",
        MODEL_DIR / "config.json",
        MODEL_DIR / "pytorch_model.bin",
        TOKENIZER_DIR / "tokenizer.json",
        TOKENIZER_DIR / "tokenizer_config.json",
        VECTORS_PATH,
    ]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def package_versions():
    return {name: importlib.metadata.version(name) for name in (
        "torch", "amrlib", "transformers", "penman", "nltk", "spacy", "en_core_web_sm", "scipy", "numpy"
    )}


def load_sources():
    bank = json.loads(BANK_PATH.read_text())
    groups = json.loads(GROUPS_PATH.read_text())
    examples = json.loads(EXAMPLES_PATH.read_text())
    if len(bank) != 50 or len(examples) != 3:
        raise ValueError("Expected official bank50 and three prompt examples")
    group_map = {row["normalized_template_amr"]: row for row in groups}
    if len(group_map) != len(groups) or any(row["normalized_template_amr"] not in group_map for row in bank):
        raise ValueError("Bank templates do not map uniquely to official groups")
    positives = sum(len(group_map[row["normalized_template_amr"]]["examples"]) for row in bank)
    if positives != 387:
        raise ValueError(f"Expected 387 bank raw-AMR positives, got {positives}")
    return bank, group_map, examples


def prepare(folder):
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        return check(folder)
    bank, groups, examples = load_sources()
    manifest = {
        "status": "frozen_before_execution",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "workflow": "swan_known_positive_audit",
            "network_and_generation_api_calls": 0,
            "bank_templates": 50,
            "bank_raw_amr_positives": sum(len(groups[row["normalized_template_amr"]]["examples"]) for row in bank),
            "prompt_examples": 3,
            "self_match_repeats": SELF_REPEATS,
            "graph_gate": "All 387 bank raw AMRs and 3 prompt raw AMRs reconstruct exact saved target templates",
            "matcher_gate": "All positive and self scores >= 1-1e-9; each bank self score equals its all-bank maximum",
            "parser_layer": "Two same-process CPU parses per official prompt sentence; diagnostic, not a population estimate",
            "decision": {
                "base_failure": "Stop generation and isolate graph/matcher implementation",
                "parser_mismatch": "Local graph/matcher pass but sentence-to-AMR reproduction remains a compatibility risk",
                "all_pass": "Only then consider generation-model/task compatibility as the next bottleneck",
            },
            "limits": [
                "The three prompt examples are not bank50 members",
                "MASSIVE-AMR utterances for the 50 bank groups are not released in this checkout",
                "Same-process parser repetition is not an independent model replication",
                "S2Match reseeds internally; observed repeats are retained",
                "No attack ASR or population FPR is estimated",
            ],
        },
        "source_inventory": {
            "bank_template_sha256": [text_sha(row["normalized_template_amr"]) for row in bank],
            "bank_group_example_counts": [len(groups[row["normalized_template_amr"]]["examples"]) for row in bank],
            "prompt_sentence_sha256": [text_sha(row["sentence"]) for row in examples],
            "prompt_raw_amr_sha256": [text_sha(row["raw_amr"]) for row in examples],
        },
        "dependencies": dependencies(),
        "packages": package_versions(),
    }
    freeze(manifest_path, manifest)
    return manifest


def check(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["protocol"]["workflow"] != "swan_known_positive_audit":
        raise ValueError("Unexpected manifest workflow")
    for relative, expected in manifest["dependencies"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Frozen dependency changed: {relative}")
    if package_versions() != manifest["packages"]:
        raise ValueError("Frozen package versions changed")
    bank, groups, examples = load_sources()
    inventory = manifest["source_inventory"]
    if inventory != {
        "bank_template_sha256": [text_sha(row["normalized_template_amr"]) for row in bank],
        "bank_group_example_counts": [len(groups[row["normalized_template_amr"]]["examples"]) for row in bank],
        "prompt_sentence_sha256": [text_sha(row["sentence"]) for row in examples],
        "prompt_raw_amr_sha256": [text_sha(row["raw_amr"]) for row in examples],
    }:
        raise ValueError("Source inventory changed")
    return manifest


def load_utils():
    configure_environment()
    if str(SWAN) not in sys.path:
        sys.path.insert(0, str(SWAN))
    from utils import amr_utils
    return amr_utils


def pipeline(raw, utils):
    import penman
    clean = utils.remove_amr_comments(raw)
    penman.decode(clean)
    normalized = utils.normalize_amr_variables(clean)
    templated = utils.generate_template_amr(normalized)
    final = utils.normalize_amr_variables(templated)
    penman.decode(final)
    return {
        "clean_amr": clean,
        "normalized_amr": normalized,
        "template_amr": templated,
        "final_amr": final,
        "raw_sha256": text_sha(raw),
        "final_sha256": text_sha(final),
    }


def triple_counts(amr):
    import penman
    graph = penman.decode(amr)
    variables = {source for source, role, _ in graph.triples if role == ":instance"}
    return {
        "instances": sum(role == ":instance" for _, role, _ in graph.triples),
        "attributes": sum(role != ":instance" and target not in variables
                          for _, role, target in graph.triples),
        "relations": sum(role != ":instance" and target in variables
                         for _, role, target in graph.triples),
        "triples": len(graph.triples),
    }


def graph_reconstruction(folder, utils):
    path = folder / "graph_reconstruction.json"
    if path.exists():
        return json.loads(path.read_text())
    bank, groups, examples = load_sources()
    rows = []
    for bank_index, item in enumerate(bank):
        target = item["normalized_template_amr"]
        for example_index, raw in enumerate(groups[target]["examples"]):
            transformed = pipeline(raw, utils)
            rows.append({
                "kind": "bank_group", "bank_index": bank_index, "example_index": example_index,
                "target_sha256": text_sha(target), "exact": transformed["final_amr"].strip() == target.strip(),
                **transformed,
            })
    prompt_rows = []
    for index, item in enumerate(examples):
        transformed = pipeline(item["raw_amr"], utils)
        prompt_rows.append({
            "kind": "prompt_example", "example_index": index, "sentence": item["sentence"],
            "sentence_sha256": text_sha(item["sentence"]),
            "target_sha256": text_sha(item["normalized_template_amr"]),
            "stored_template_exact": item["template_amr"].strip() == item["normalized_template_amr"].strip(),
            "exact": transformed["final_amr"].strip() == item["normalized_template_amr"].strip(),
            **transformed,
        })
    result = {
        "bank_rows": rows,
        "prompt_rows": prompt_rows,
        "bank_exact": sum(row["exact"] for row in rows),
        "bank_total": len(rows),
        "prompt_exact": sum(row["exact"] for row in prompt_rows),
        "prompt_total": len(prompt_rows),
    }
    freeze(path, result)
    return result


def valid_score(score):
    return math.isfinite(score) and -1e-12 <= score <= 1 + 1e-12


def matcher_checks(folder, utils):
    bank, groups, _ = load_sources()
    checkpoint_dir = folder / "matcher"
    for bank_index, item in enumerate(bank):
        path = checkpoint_dir / f"bank_{bank_index:02d}.json"
        if path.exists():
            continue
        target = item["normalized_template_amr"]
        started = time.monotonic()
        self_scores = [float(utils.compute_s2match_score(target, target)) for _ in range(SELF_REPEATS)]
        bank_scores = [float(utils.compute_s2match_score(target, other["normalized_template_amr"])) for other in bank]
        raw_positive_scores = []
        for example_index, raw in enumerate(groups[target]["examples"]):
            final = pipeline(raw, utils)["final_amr"]
            raw_positive_scores.append({
                "example_index": example_index,
                "raw_sha256": text_sha(raw),
                "final_sha256": text_sha(final),
                "score": float(utils.compute_s2match_score(final, target)),
            })
        scores = self_scores + bank_scores + [row["score"] for row in raw_positive_scores]
        if not all(valid_score(score) for score in scores):
            raise ValueError("Invalid S2Match score")
        result = {
            "bank_index": bank_index,
            "target_sha256": text_sha(target),
            "target_counts": triple_counts(target),
            "self_scores": self_scores,
            "bank_scores": bank_scores,
            "own_bank_score": bank_scores[bank_index],
            "bank_max": max(bank_scores),
            "bank_max_indices": [i for i, score in enumerate(bank_scores) if abs(score - max(bank_scores)) <= 1e-9],
            "raw_positive_scores": raw_positive_scores,
            "elapsed_seconds": time.monotonic() - started,
        }
        freeze(path, result)
        print(json.dumps({
            "stage": "matcher", "completed": bank_index + 1, "total": len(bank),
            "self_min": min(self_scores), "own_bank_score": bank_scores[bank_index],
            "raw_positives": len(raw_positive_scores),
        }), flush=True)
    return [json.loads((checkpoint_dir / f"bank_{i:02d}.json").read_text()) for i in range(len(bank))]


def parser_checks(folder, utils):
    path = folder / "parser_examples.json"
    if path.exists():
        return json.loads(path.read_text())
    import amrlib
    import penman
    import torch
    bank, _, examples = load_sources()
    torch.set_num_threads(4)
    started = time.monotonic()
    parser = amrlib.load_stog_model(
        model_dir=str(MODEL_DIR), tok_name_or_path=str(TOKENIZER_DIR), device="cpu", batch_size=1)
    sentences = [row["sentence"] for row in examples]
    token_lengths = [len(parser.tokenizer(sentence)["input_ids"]) for sentence in sentences]
    if any(length > parser.max_sent_len for length in token_lengths):
        raise ValueError("Prompt example would be truncated")
    passes = []
    for _ in range(2):
        raw_outputs = parser.parse_sents(sentences)
        if len(raw_outputs) != len(sentences) or any(not raw for raw in raw_outputs):
            raise ValueError("AMR parser returned incomplete output")
        pass_rows = []
        for index, (item, raw) in enumerate(zip(examples, raw_outputs)):
            graph = amr_body(raw)
            penman.decode(graph)
            transformed = pipeline(graph, utils)
            target = item["normalized_template_amr"]
            bank_scores = [float(utils.compute_s2match_score(
                transformed["final_amr"], bank_item["normalized_template_amr"])) for bank_item in bank]
            row = {
                "example_index": index,
                "sentence": item["sentence"],
                "input_tokens": token_lengths[index],
                "raw_parser_output": raw,
                "graph_amr": graph,
                "graph_sha256": text_sha(graph),
                "expected_raw_sha256": text_sha(item["raw_amr"]),
                "expected_template_sha256": text_sha(target),
                "predicted_template": transformed["final_amr"],
                "predicted_template_sha256": transformed["final_sha256"],
                "raw_s2match_to_expected": float(utils.compute_s2match_score(graph, item["raw_amr"])),
                "template_s2match_to_expected": float(utils.compute_s2match_score(transformed["final_amr"], target)),
                "template_exact": transformed["final_amr"].strip() == target.strip(),
                "predicted_counts": triple_counts(transformed["final_amr"]),
                "expected_counts": triple_counts(target),
                "bank_max_score": max(bank_scores),
                "bank_max_indices": [i for i, score in enumerate(bank_scores) if abs(score - max(bank_scores)) <= 1e-9],
            }
            if not all(valid_score(row[key]) for key in (
                    "raw_s2match_to_expected", "template_s2match_to_expected", "bank_max_score")):
                raise ValueError("Invalid parser diagnostic score")
            pass_rows.append(row)
        passes.append(pass_rows)
    repeats = []
    for index in range(len(examples)):
        repeats.append({
            "example_index": index,
            "raw_exact": passes[0][index]["graph_amr"] == passes[1][index]["graph_amr"],
            "template_exact": passes[0][index]["predicted_template"] == passes[1][index]["predicted_template"],
        })
    result = {
        "passes": passes,
        "repeat_consistency": repeats,
        "elapsed_seconds": time.monotonic() - started,
        "same_process_same_model_instance": True,
    }
    freeze(path, result)
    return result


def summarize(folder):
    graph = json.loads((folder / "graph_reconstruction.json").read_text())
    matcher_paths = sorted((folder / "matcher").glob("bank_*.json"))
    parser_path = folder / "parser_examples.json"
    if len(matcher_paths) != 50 or not parser_path.exists():
        result = {
            "status": "incomplete",
            "completed_matcher_templates": len(matcher_paths),
            "parser_complete": parser_path.exists(),
            "new_api_calls": 0,
        }
        write_json(folder / "summary.json", result)
        return result
    matcher = [json.loads(path.read_text()) for path in matcher_paths]
    parser = json.loads(parser_path.read_text())
    self_scores = [score for row in matcher for score in row["self_scores"]]
    positive_scores = [item["score"] for row in matcher for item in row["raw_positive_scores"]]
    graph_gate = graph["bank_exact"] == graph["bank_total"] and graph["prompt_exact"] == graph["prompt_total"]
    matcher_gate = (
        min(self_scores) >= 1 - 1e-9
        and min(positive_scores) >= 1 - 1e-9
        and all(abs(row["own_bank_score"] - row["bank_max"]) <= 1e-9 for row in matcher)
    )
    first_pass = parser["passes"][0]
    parser_exact = sum(row["template_exact"] for row in first_pass)
    parser_repeat = sum(row["raw_exact"] and row["template_exact"] for row in parser["repeat_consistency"])
    if not graph_gate or not matcher_gate:
        decision = "stop_base_chain_failure"
    elif parser_exact < len(first_pass):
        decision = "base_chain_pass_parser_mismatch"
    else:
        decision = "base_chain_and_parser_examples_pass"
    result = {
        "status": "complete",
        "decision": decision,
        "graph_gate": graph_gate,
        "matcher_gate": matcher_gate,
        "graph_reconstruction": {
            "bank_exact": graph["bank_exact"], "bank_total": graph["bank_total"],
            "prompt_exact": graph["prompt_exact"], "prompt_total": graph["prompt_total"],
        },
        "matcher": {
            "self_comparisons": len(self_scores), "self_min": min(self_scores),
            "self_max_variation": max(max(row["self_scores"]) - min(row["self_scores"]) for row in matcher),
            "raw_positive_comparisons": len(positive_scores), "raw_positive_min": min(positive_scores),
            "own_equals_bank_max": sum(abs(row["own_bank_score"] - row["bank_max"]) <= 1e-9 for row in matcher),
            "templates_with_tied_bank_max": sum(len(row["bank_max_indices"]) > 1 for row in matcher),
        },
        "parser": {
            "examples": len(first_pass), "template_exact": parser_exact,
            "repeat_consistent": parser_repeat,
            "template_s2match": [row["template_s2match_to_expected"] for row in first_pass],
            "raw_s2match": [row["raw_s2match_to_expected"] for row in first_pass],
            "bank_max_score": [row["bank_max_score"] for row in first_pass],
            "elapsed_seconds": parser["elapsed_seconds"],
        },
        "new_api_calls": 0,
        "manifest_sha256": sha(folder / "manifest.json"),
    }
    write_json(folder / "summary.json", result)
    return result


def run(folder):
    check(folder)
    utils = load_utils()
    graph_reconstruction(folder, utils)
    matcher_checks(folder, utils)
    parser_checks(folder, utils)
    return summarize(folder)


def audit(folder):
    check(folder)
    utils = load_utils()
    bank, groups, examples = load_sources()
    graph = json.loads((folder / "graph_reconstruction.json").read_text())
    expected_rows = sum(len(groups[row["normalized_template_amr"]]["examples"]) for row in bank)
    if len(graph["bank_rows"]) != expected_rows or len(graph["prompt_rows"]) != len(examples):
        raise ValueError("Graph reconstruction inventory mismatch")
    for row in graph["bank_rows"]:
        target = bank[row["bank_index"]]["normalized_template_amr"]
        raw = groups[target]["examples"][row["example_index"]]
        actual = pipeline(raw, utils)
        expected = {
            "kind": "bank_group", "bank_index": row["bank_index"], "example_index": row["example_index"],
            "target_sha256": text_sha(target), "exact": actual["final_amr"].strip() == target.strip(), **actual,
        }
        if row != expected:
            raise ValueError("Bank graph reconstruction differs")
    for row in graph["prompt_rows"]:
        item = examples[row["example_index"]]
        actual = pipeline(item["raw_amr"], utils)
        expected = {
            "kind": "prompt_example", "example_index": row["example_index"], "sentence": item["sentence"],
            "sentence_sha256": text_sha(item["sentence"]),
            "target_sha256": text_sha(item["normalized_template_amr"]),
            "stored_template_exact": item["template_amr"].strip() == item["normalized_template_amr"].strip(),
            "exact": actual["final_amr"].strip() == item["normalized_template_amr"].strip(), **actual,
        }
        if row != expected:
            raise ValueError("Prompt graph reconstruction differs")
    matcher = [json.loads((folder / "matcher" / f"bank_{i:02d}.json").read_text()) for i in range(50)]
    for index, row in enumerate(matcher):
        target = bank[index]["normalized_template_amr"]
        if row["bank_index"] != index or row["target_sha256"] != text_sha(target):
            raise ValueError("Matcher checkpoint identity mismatch")
        if len(row["self_scores"]) != SELF_REPEATS or len(row["bank_scores"]) != 50:
            raise ValueError("Matcher score inventory mismatch")
        if len(row["raw_positive_scores"]) != len(groups[target]["examples"]):
            raise ValueError("Raw positive score inventory mismatch")
        if (row["own_bank_score"] != row["bank_scores"][index]
                or row["bank_max"] != max(row["bank_scores"])
                or row["bank_max_indices"] != [
                    i for i, score in enumerate(row["bank_scores"])
                    if abs(score - max(row["bank_scores"])) <= 1e-9
                ]):
            raise ValueError("Matcher ranking summary mismatch")
        if not all(valid_score(v) for v in row["self_scores"] + row["bank_scores"]
                           + [r["score"] for r in row["raw_positive_scores"]]):
            raise ValueError("Invalid saved matcher score")
    parser = json.loads((folder / "parser_examples.json").read_text())
    if len(parser["passes"]) != 2 or any(len(rows) != 3 for rows in parser["passes"]):
        raise ValueError("Parser result inventory mismatch")
    for pass_rows in parser["passes"]:
        for row in pass_rows:
            item = examples[row["example_index"]]
            penman_result = pipeline(row["graph_amr"], utils)
            if (row["sentence"] != item["sentence"]
                    or row["predicted_template"] != penman_result["final_amr"]
                    or row["graph_sha256"] != text_sha(row["graph_amr"])):
                raise ValueError("Saved parser transformation mismatch")
    summarize(folder)
    result = {
        "status": "passed", "mismatches": 0,
        "graph_rows_reconstructed": expected_rows + len(examples),
        "matcher_checkpoints_checked": len(matcher),
        "parser_outputs_checked": 6,
        "new_api_calls": 0, "new_neural_parser_calls": 0, "new_s2match_calls": 0,
        "summary_sha256": sha(folder / "summary.json"),
        "limitation": "Saved S2Match scores and parser outputs were structurally audited; stochastic matching and neural parsing were not rerun",
    }
    write_json(folder / "audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "audit", "summarize"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    configure_environment()
    if args.stage == "prepare":
        result = prepare(args.output)
        print(json.dumps({"status": "prepared", "manifest_sha256": sha(args.output / "manifest.json"),
                          "raw_positives": result["protocol"]["bank_raw_amr_positives"]}, indent=2))
    elif args.stage == "run":
        print(json.dumps(run(args.output), indent=2))
    elif args.stage == "audit":
        print(json.dumps(audit(args.output), indent=2))
    else:
        print(json.dumps(summarize(args.output), indent=2))


if __name__ == "__main__":
    main()
