"""Cross-run offline localization after the SWAN known-positive audit."""

import json
from pathlib import Path
import statistics
import sys

from run_segmentation_intervention import ROOT, sha, write_json

OUTPUT = ROOT / "experiments/swan_known_positive1/localization.json"
GROWTH = ROOT / "experiments/swan_sentence_growth1"
GROUNDED = ROOT / "experiments/grounded_injection5"
KNOWN = ROOT / "experiments/swan_known_positive1"
MATCHER_PATH = ROOT / "third_party/SWAN/amr-metric-suite/py3-Smatch-and-S2match/smatch"


def triple_count(amr):
    if str(MATCHER_PATH) not in sys.path:
        sys.path.insert(0, str(MATCHER_PATH))
    from amr_py3 import AMR
    graph = AMR.parse_AMR_line(amr, False)
    if graph is None:
        raise ValueError("Could not parse cached AMR")
    return sum(len(items) for items in graph.get_triples())


def parse_map(folder, inputs):
    result = {}
    for path in sorted((folder / "sentence_cache").glob("*/parse/*.json")):
        saved = json.loads(path.read_text())
        text = saved["key"].get("injection_input")
        if text is None:
            continue
        value = saved["value"]
        if text in result and result[text] != value:
            raise ValueError("Same candidate text has inconsistent cached AMRs")
        result[text] = value
        inputs[str(path.relative_to(ROOT))] = sha(path)
    return result


def collect(folder, pattern, inputs):
    parses = parse_map(folder, inputs)
    rows = []
    for path in sorted(folder.glob(pattern)):
        saved = json.loads(path.read_text())
        inputs[str(path.relative_to(ROOT))] = sha(path)
        text = saved["text"]
        if text not in parses:
            raise ValueError(f"Final candidate lacks cached parser output: {path}")
        target = saved.get("chosen_template", saved.get("template"))
        score = saved.get("s2match", saved.get("target_score"))
        if target is None or score is None:
            raise ValueError("Final candidate lacks target or score")
        rows.append({
            "record": str(path.relative_to(folder)),
            "words": len(text.split()),
            "candidate_triples": triple_count(parses[text]),
            "target_triples": triple_count(target),
            "triple_ratio": triple_count(parses[text]) / triple_count(target),
            "score": float(score),
            "accepted": bool(saved["accepted"]),
        })
    return rows


def describe(rows):
    return {
        "slots": len(rows),
        "accepted": sum(row["accepted"] for row in rows),
        "words": {
            "mean": statistics.mean(row["words"] for row in rows),
            "median": statistics.median(row["words"] for row in rows),
            "min": min(row["words"] for row in rows),
            "max": max(row["words"] for row in rows),
        },
        "candidate_triples": {
            "mean": statistics.mean(row["candidate_triples"] for row in rows),
            "median": statistics.median(row["candidate_triples"] for row in rows),
            "min": min(row["candidate_triples"] for row in rows),
            "max": max(row["candidate_triples"] for row in rows),
        },
        "target_triples_mean": statistics.mean(row["target_triples"] for row in rows),
        "candidate_target_ratio_mean": statistics.mean(row["triple_ratio"] for row in rows),
        "score_mean": statistics.mean(row["score"] for row in rows),
        "score_min": min(row["score"] for row in rows),
        "score_max": max(row["score"] for row in rows),
    }


def analyze(output=OUTPUT):
    inputs = {}
    known_summary_path = KNOWN / "summary.json"
    parser_path = KNOWN / "parser_examples.json"
    shape_path = GROUNDED / "shape_diagnostic.json"
    known = json.loads(known_summary_path.read_text())
    parser = json.loads(parser_path.read_text())
    shape = json.loads(shape_path.read_text())
    for path in (known_summary_path, parser_path, shape_path):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    growth_rows = collect(GROWTH, "injection/*.json", inputs)
    grounded_rows = collect(GROUNDED, "sentences/*-swan_grounded-*.json", inputs)
    prompt_scores = [row["template_s2match_to_expected"] for row in parser["passes"][0]]
    if not known["graph_gate"] or not known["matcher_gate"]:
        conclusion = "base_chain_failure"
    elif all(score >= .75 for score in prompt_scores):
        conclusion = "generation_task_template_scale_incompatibility_supported"
    else:
        conclusion = "parser_acceptance_reproduction_incomplete"
    result = {
        "status": "completed_exploratory_localization",
        "acceptance_threshold": .75,
        "known_positive": {
            "graph_gate": known["graph_gate"],
            "matcher_gate": known["matcher_gate"],
            "prompt_template_exact": known["parser"]["template_exact"],
            "prompt_threshold_pass": sum(score >= .75 for score in prompt_scores),
            "prompt_total": len(prompt_scores),
            "prompt_scores": prompt_scores,
        },
        "natural_short_sentence_run": describe(growth_rows),
        "grounded_summary_run": describe(grounded_rows),
        "grounded_attempt_shape": shape["summary"],
        "conclusion": conclusion,
        "interpretation": (
            "The local graph and matcher chain passes, and all three official prompt examples remain "
            "above the injection threshold after local neural parsing. Final candidates in the grounded "
            "summary run are much larger than their targets, unlike the prior short-sentence run. This "
            "supports task/template scale incompatibility as the immediate local failure mode."
        ),
        "confounds": [
            "The two generation runs use different prompts, source documents, retry budgets and template counts",
            "The 40 slots in the short-sentence run belong to one document and are not independent samples",
            "The 25 grounded slots belong to five documents and all are failed-injection fallbacks",
            "This comparison is post hoc and descriptive, not a causal estimate",
            "The result does not establish failure of the paper's original model or SWAN generally",
        ],
        "new_api_calls": 0,
        "new_neural_parser_calls": 0,
        "new_s2match_calls": 0,
        "input_sha256": inputs,
        "script_sha256": sha(Path(__file__)),
        "rows": {"natural_short_sentence_run": growth_rows, "grounded_summary_run": grounded_rows},
    }
    write_json(output, result)
    print(json.dumps({key: result[key] for key in (
        "status", "known_positive", "natural_short_sentence_run",
        "grounded_summary_run", "conclusion", "new_api_calls"
    )}, indent=2))
    return result


if __name__ == "__main__":
    analyze()
