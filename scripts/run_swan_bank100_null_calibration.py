"""Build and validate a bank100 null calibration on independent human and machine text."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import time

from feedback2_runtime import Heartbeat, claim
from prepare_confirmation_experiment import normalized_hash, stream_validation
from run_confirmation_experiment import CallJournal, MODEL, digest, freeze, request_body
from run_grounded_generation_comparison import (
    fact_body,
    generation_body,
    decode_fact_review,
)
from run_grounded_injection_pilot import screen_source
from run_segmentation_intervention import REVISION, ROOT, sha, text_sha, write_json
from run_swan_atomic_fact_preflight import (
    MODEL_DIR,
    TOKENIZER_DIR,
    VECTORS_PATH,
    analyze_parser_output,
    configure_environment,
    count_upper_bound,
    s2_counts,
)
from run_swan_bank_scale_pilot import HUMAN_PATH, SCORE_THRESHOLD
from run_swan_known_positive_audit import load_utils
from run_swan_sentence_growth import BASELINE_REVIEW, decode_baseline_review
from supervise_confirmation_experiment import read_credential

OUTPUT = ROOT / "experiments/swan_bank100_null_calibration1"
BANK_PATH = ROOT / "third_party/SWAN/amr_bank/banks/amr_bank_100.json"
SOURCE_START = 1486
SOURCE_SCAN_LIMIT = 3000
MACHINE_DOCUMENTS = 30
MIN_MACHINE_PASS = 18
API_CAP = 91
HUMAN_CALIBRATION = range(50, 150)
HUMAN_TEST = range(150, 250)
WORKERS = 4
TARGET_FPR = 0.05
FIXED_NULL = 0.25
FIXED_Z = 2.33
_WORKER_BANK = None
_WORKER_COUNTS = None
_WORKER_SCORE = None


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
        self.heartbeat.update(api_scope=scope, api_attempts=self.prior + self.sent)
        return super().request(scope, body)


def select_sources(records, tokenize):
    selected = []
    rejected = []
    seen_text = set()
    seen_urls = set()
    scanned_to = None
    for index, row in enumerate(records):
        text = " ".join(row["text"].split())
        identity = normalized_hash(text)
        url = row.get("url", "")
        duplicate = identity in seen_text or bool(url and url in seen_urls)
        seen_text.add(identity)
        if url:
            seen_urls.add(url)
        if index < SOURCE_START:
            continue
        if index >= SOURCE_START + SOURCE_SCAN_LIMIT:
            break
        screening = screen_source(text, tokenize)
        reasons = list(screening["reasons"])
        if duplicate:
            reasons.append("duplicate_prior_row")
        item = {
            "prompt_id": f"c4-validation-{index}",
            "shard_row": index,
            "source": text,
            "source_sha256": text_sha(text),
            "url": url,
            "screening": screening,
        }
        if reasons:
            rejected.append({
                key: value for key, value in {**item, "reasons": reasons}.items()
                if key != "source"
            })
        else:
            selected.append(item)
        scanned_to = index
        if len(selected) == MACHINE_DOCUMENTS:
            break
    if len(selected) != MACHINE_DOCUMENTS:
        raise ValueError(
            f"Found {len(selected)}/{MACHINE_DOCUMENTS} sources through row {scanned_to}"
        )
    return selected, rejected, scanned_to


def dependency_hashes():
    paths = [
        Path(__file__),
        OUTPUT / "protocol.md",
        BANK_PATH,
        HUMAN_PATH,
        ROOT / "experiments/swan_bank_scale_pilot1/manifest.json",
        ROOT / "experiments/swan_bank_scale_pilot1/summary.json",
        ROOT / "experiments/swan_bank_scale_pilot1/audit.json",
        ROOT / "scripts/feedback2_runtime.py",
        ROOT / "scripts/prepare_confirmation_experiment.py",
        ROOT / "scripts/run_confirmation_experiment.py",
        ROOT / "scripts/run_grounded_generation_comparison.py",
        ROOT / "scripts/run_grounded_injection_pilot.py",
        ROOT / "scripts/run_swan_atomic_fact_preflight.py",
        ROOT / "scripts/run_swan_bank_scale_pilot.py",
        ROOT / "scripts/run_swan_known_positive_audit.py",
        ROOT / "scripts/run_swan_sentence_growth.py",
        ROOT / "scripts/supervise_confirmation_experiment.py",
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
    paths += sorted((ROOT / ".nltk_data/tokenizers/punkt_tab/english").glob("*"))
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def package_versions():
    return {
        name: importlib.metadata.version(name)
        for name in (
            "amrlib", "nltk", "numpy", "penman", "requests", "spacy", "torch",
            "transformers",
        )
    }


def human_inventory():
    documents = json.loads(HUMAN_PATH.read_text())
    result = {}
    for name, indices in (
        ("calibration", HUMAN_CALIBRATION),
        ("test", HUMAN_TEST),
    ):
        result[name] = [
            {
                "document_index": index,
                "sentences": len(documents[index]),
                "amr_sha256": [text_sha(amr) for amr in documents[index]],
            }
            for index in indices
        ]
    return result


def prepare(folder):
    path = folder / "manifest.json"
    if path.exists():
        return check(folder)
    import nltk

    stream = stream_validation()
    try:
        sources, rejected, scanned_to = select_sources(stream, nltk.sent_tokenize)
    finally:
        stream.close()
    manifest = {
        "status": "frozen_before_generation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "allenai/c4",
            "revision": REVISION,
            "first_eligible_row": SOURCE_START,
            "scan_limit": SOURCE_SCAN_LIMIT,
            "scanned_through_row": scanned_to,
            "dataset_stream_http_requests": 1,
        },
        "machine_sources": sources,
        "screened_rejections": rejected,
        "human_inventory": human_inventory(),
        "protocol": {
            "workflow": "swan_bank100_null_calibration",
            "bank_size": 100,
            "bank_sha256": sha(BANK_PATH),
            "machine_documents": MACHINE_DOCUMENTS,
            "minimum_quality_pass": MIN_MACHINE_PASS,
            "generation": {
                "model": MODEL,
                "arm": "onepass",
                "prompt": generation_body("FROZEN_SOURCE_PLACEHOLDER", "onepass", 0, []),
                "attempts_per_document": 1,
            },
            "quality": {
                "style": "all dimensions >=4 and no severe problem",
                "facts": "faithfulness >=4, no unsupported claim, contradiction or issue",
                "format": "exactly 5 sentences and 100..160 whitespace words",
            },
            "human_calibration_indices": [50, 149],
            "human_test_indices": [150, 249],
            "sentence_green_threshold": SCORE_THRESHOLD,
            "calibration": (
                "Estimate lambda on human calibration sentences; use strict > empirical "
                "95th percentile calibration-document z threshold"
            ),
            "fixed_reference": {"lambda": FIXED_NULL, "z_threshold": FIXED_Z},
            "continuation_gate": {
                "machine_quality_pass_min": MIN_MACHINE_PASS,
                "human_test_fpr_max": 0.05,
                "machine_test_fpr_max": 0.10,
            },
            "api_cap": API_CAP,
            "workers": WORKERS,
            "no_watermark_injection_or_attack": True,
        },
        "dependencies": dependency_hashes(),
        "packages": package_versions(),
    }
    freeze(path, manifest)
    return manifest


def check(folder):
    import nltk

    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["protocol"]["workflow"] != "swan_bank100_null_calibration":
        raise ValueError("Unexpected workflow")
    for relative, expected in manifest["dependencies"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Frozen dependency changed: {relative}")
    if package_versions() != manifest["packages"]:
        raise ValueError("Frozen package versions changed")
    if manifest["human_inventory"] != human_inventory():
        raise ValueError("Human split changed")
    for source in manifest["machine_sources"]:
        if (
            text_sha(source["source"]) != source["source_sha256"]
            or screen_source(source["source"], nltk.sent_tokenize) != source["screening"]
            or not source["screening"]["eligible"]
            or source["shard_row"] < SOURCE_START
        ):
            raise ValueError(f"Machine source changed: {source['prompt_id']}")
    return manifest


def style_pass(review):
    return bool(
        review
        and all(review[key] >= 4 for key in ("fluency", "coherence", "nonredundancy"))
        and not review["severe_problem"]
    )


def factual_pass(review):
    return bool(
        review
        and review["faithfulness"] >= 4
        and not review["unsupported_claim"]
        and not review["contradiction"]
        and not review["issues"]
    )


def structural_metrics(text, finish_reason, tokenize):
    sentences = tokenize(text)
    words = len(text.split())
    checks = {
        "finish": finish_reason == "stop",
        "sentences": len(sentences) == 5,
        "words": 100 <= words <= 160,
        "single_paragraph": len([line for line in text.splitlines() if line.strip()]) == 1,
    }
    return {
        "words": words,
        "sentences": len(sentences),
        "sentence_words": [len(sentence.split()) for sentence in sentences],
        "checks": checks,
        "pass": all(checks.values()),
    }


def generate_and_review(folder, manifest, api, runtime, heartbeat):
    import nltk

    for index, source in enumerate(manifest["machine_sources"]):
        body = generation_body(source["source"], "onepass", 0, [])
        response = api.request(f"generate-{source['prompt_id']}", body)
        text = response["content"].strip()
        structure = structural_metrics(text, response["finish_reason"], nltk.sent_tokenize)
        generated = {
            "prompt_id": source["prompt_id"],
            "source_sha256": source["source_sha256"],
            "runtime_sha256": digest(runtime),
            "request_sha256": digest(body),
            "response": response,
            "text": text,
            "text_sha256": text_sha(text),
            "structure": structure,
        }
        freeze(folder / "generated" / f"{source['prompt_id']}.json", generated)
        heartbeat.update(stage="generation", completed_documents=index + 1,
                         prompt_id=source["prompt_id"])
        print(json.dumps({
            "stage": "generation",
            "completed": index + 1,
            "total": MACHINE_DOCUMENTS,
            "prompt_id": source["prompt_id"],
            "words": structure["words"],
            "sentences": structure["sentences"],
        }), flush=True)
    for index, source in enumerate(manifest["machine_sources"]):
        generated = json.loads(
            (folder / "generated" / f"{source['prompt_id']}.json").read_text()
        )
        text = generated["text"]
        style_body = request_body(BASELINE_REVIEW + text, 0, 700)
        source_fact_body = fact_body(source["source"], text)
        style_response = api.request(f"style-{source['prompt_id']}", style_body)
        fact_response = api.request(f"facts-{source['prompt_id']}", source_fact_body)
        errors = {}
        try:
            style = decode_baseline_review(style_response["content"])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            style, errors["style"] = None, type(exc).__name__
        try:
            facts = decode_fact_review(fact_response["content"])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            facts, errors["facts"] = None, type(exc).__name__
        gates = {
            "structure": generated["structure"]["pass"],
            "style": style_pass(style),
            "facts": factual_pass(facts),
        }
        reviewed = {
            "prompt_id": source["prompt_id"],
            "source_sha256": source["source_sha256"],
            "text_sha256": generated["text_sha256"],
            "runtime_sha256": digest(runtime),
            "style_request_sha256": digest(style_body),
            "fact_request_sha256": digest(source_fact_body),
            "style_response": style_response,
            "fact_response": fact_response,
            "style": style,
            "facts": facts,
            "decode_errors": errors,
            "gates": gates,
            "quality_pass": all(gates.values()),
        }
        freeze(folder / "reviews" / f"{source['prompt_id']}.json", reviewed)
        heartbeat.update(stage="review", completed_documents=index + 1,
                         prompt_id=source["prompt_id"])
        print(json.dumps({
            "stage": "review",
            "completed": index + 1,
            "total": MACHINE_DOCUMENTS,
            "prompt_id": source["prompt_id"],
            "gates": gates,
        }), flush=True)


def parse_quality_machine(folder, manifest, runtime, heartbeat):
    pending = []
    for source in manifest["machine_sources"]:
        reviewed = json.loads(
            (folder / "reviews" / f"{source['prompt_id']}.json").read_text()
        )
        path = folder / "parser" / f"{source['prompt_id']}.json"
        if reviewed["quality_pass"] and not path.exists():
            pending.append(source)
    if not pending:
        return 0
    import amrlib
    import nltk
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
    calls = 0
    for source in manifest["machine_sources"]:
        reviewed = json.loads(
            (folder / "reviews" / f"{source['prompt_id']}.json").read_text()
        )
        if not reviewed["quality_pass"]:
            continue
        path = folder / "parser" / f"{source['prompt_id']}.json"
        if path.exists():
            continue
        generated = json.loads(
            (folder / "generated" / f"{source['prompt_id']}.json").read_text()
        )
        sentences = nltk.sent_tokenize(generated["text"])
        token_lengths = [len(parser.tokenizer(sentence)["input_ids"]) for sentence in sentences]
        if any(length > parser.max_sent_len for length in token_lengths):
            raise ValueError(f"Tokenizer limit exceeded: {source['prompt_id']}")
        heartbeat.update(stage="local_amr_parser", prompt_id=source["prompt_id"],
                         new_neural_parser_calls=calls)
        started = time.monotonic()
        outputs = parser.parse_sents(sentences)
        if len(outputs) != 5 or any(not output for output in outputs):
            raise ValueError(f"Incomplete parser output: {source['prompt_id']}")
        rows = []
        for sentence_index, (sentence, tokens, raw_output) in enumerate(
            zip(sentences, token_lengths, outputs)
        ):
            identity = {
                "sentence_index": sentence_index,
                "text": sentence,
                "text_sha256": text_sha(sentence),
                "words": len(sentence.split()),
            }
            row = analyze_parser_output(identity, raw_output, utils, bank_counts)
            row["input_tokens"] = tokens
            rows.append(row)
        freeze(path, {
            "prompt_id": source["prompt_id"],
            "source_sha256": source["source_sha256"],
            "text_sha256": generated["text_sha256"],
            "runtime_sha256": digest(runtime),
            "sentences": rows,
            "elapsed_seconds": time.monotonic() - started,
        })
        calls += len(rows)
        print(json.dumps({
            "stage": "local_amr_parser",
            "prompt_id": source["prompt_id"],
            "sentences": len(rows),
        }), flush=True)
    return calls


def score_worker_init(bank):
    global _WORKER_BANK, _WORKER_COUNTS, _WORKER_SCORE
    configure_environment()
    utils = load_utils()
    _WORKER_BANK = bank
    _WORKER_COUNTS = {
        text_sha(template): s2_counts(template)
        for template in bank
    }
    _WORKER_SCORE = utils.compute_s2match_score


def score_bank100_sentence(amr, bank, counts, scorer):
    candidate_counts = s2_counts(amr)
    observations = []
    witness = None
    feasible = []
    for template_index, template in enumerate(bank):
        template_id = text_sha(template)
        bound = count_upper_bound(candidate_counts, counts[template_id])
        if bound >= SCORE_THRESHOLD:
            feasible.append((template_index, template, template_id, bound))
    for template_index, template, template_id, bound in feasible:
        score = float(scorer(amr, template))
        if not math.isfinite(score) or not 0 <= score <= 1 + 1e-9:
            raise ValueError("Invalid S2Match score")
        if score > bound + 1e-6:
            raise ValueError("S2Match score exceeds count upper bound")
        observations.append({
            "template_index": template_index,
            "template_sha256": template_id,
            "count_upper_bound": bound,
            "score": score,
        })
        if score >= SCORE_THRESHOLD:
            witness = observations[-1]
            break
    return {
        "amr_sha256": text_sha(amr),
        "candidate_counts": candidate_counts,
        "candidate_triples": sum(candidate_counts),
        "count_compatible_templates": len(feasible),
        "green": witness is not None,
        "witness": witness,
        "score_observations": observations,
        "s2match_calls": len(observations),
    }


def score_document(task):
    cohort, document_id, amrs = task
    started = time.monotonic()
    rows = [
        score_bank100_sentence(amr, _WORKER_BANK, _WORKER_COUNTS, _WORKER_SCORE)
        for amr in amrs
    ]
    return {
        "cohort": cohort,
        "document_id": document_id,
        "amr_sha256": [text_sha(amr) for amr in amrs],
        "sentences": rows,
        "s2match_calls": sum(row["s2match_calls"] for row in rows),
        "elapsed_seconds": time.monotonic() - started,
    }


def scoring_documents(folder, manifest):
    humans = json.loads(HUMAN_PATH.read_text())
    rows = {"human_calibration": [], "human_test": [], "machine": []}
    for index in HUMAN_CALIBRATION:
        rows["human_calibration"].append((f"human-{index:03d}", humans[index]))
    for index in HUMAN_TEST:
        rows["human_test"].append((f"human-{index:03d}", humans[index]))
    for source in manifest["machine_sources"]:
        review = json.loads(
            (folder / "reviews" / f"{source['prompt_id']}.json").read_text()
        )
        if review["quality_pass"]:
            parsed = json.loads(
                (folder / "parser" / f"{source['prompt_id']}.json").read_text()
            )
            rows["machine"].append((
                source["prompt_id"],
                [sentence["template_amr"] for sentence in parsed["sentences"]],
            ))
    return rows


def score_path(folder, cohort, document_id):
    return folder / "scores" / f"{cohort}-{document_id}.json"


def score_all(folder, manifest, heartbeat):
    cohorts = scoring_documents(folder, manifest)
    tasks = []
    for cohort, rows in cohorts.items():
        for document_id, amrs in rows:
            if not score_path(folder, cohort, document_id).exists():
                tasks.append((cohort, document_id, amrs))
    if not tasks:
        return 0
    bank = [row["normalized_template_amr"] for row in json.loads(BANK_PATH.read_text())]
    new_calls = 0
    completed = sum(len(rows) for rows in cohorts.values()) - len(tasks)
    total = sum(len(rows) for rows in cohorts.values())
    heartbeat.update(stage="loading_s2match_workers", completed_documents=completed,
                     total_documents=total)
    with ProcessPoolExecutor(
        max_workers=WORKERS,
        initializer=score_worker_init,
        initargs=(bank,),
    ) as executor:
        futures = {executor.submit(score_document, task): task[:2] for task in tasks}
        for future in as_completed(futures):
            cohort, document_id = futures[future]
            result = future.result()
            if result["cohort"] != cohort or result["document_id"] != document_id:
                raise ValueError("Scoring identity mismatch")
            freeze(score_path(folder, cohort, document_id), result)
            new_calls += result["s2match_calls"]
            completed += 1
            heartbeat.update(stage="s2match", completed_documents=completed,
                             total_documents=total, new_s2match_calls=new_calls)
            print(json.dumps({
                "stage": "s2match",
                "completed": completed,
                "total": total,
                "cohort": cohort,
                "document_id": document_id,
                "new_s2match_calls": result["s2match_calls"],
            }), flush=True)
    return new_calls


def proportion_z(green, total, null):
    denominator = math.sqrt(total * null * (1 - null))
    return (green - null * total) / denominator if denominator else None


def empirical_threshold(values, target_fpr):
    ordered = sorted(values)
    index = max(0, math.ceil((1 - target_fpr) * len(ordered)) - 1)
    return ordered[index]


def document_scores(documents, null, threshold):
    rows = []
    for document in documents:
        green = sum(sentence["green"] for sentence in document["sentences"])
        value = proportion_z(green, len(document["sentences"]), null)
        rows.append({
            "document_id": document["document_id"],
            "sentences": len(document["sentences"]),
            "green": green,
            "z": value,
            "detected": value > threshold,
        })
    return rows


def result_metrics(rows):
    detected = sum(row["detected"] for row in rows)
    return {
        "documents": len(rows),
        "detected": detected,
        "rate": detected / len(rows) if rows else None,
        "z_mean": statistics.mean(row["z"] for row in rows) if rows else None,
        "z_median": statistics.median(row["z"] for row in rows) if rows else None,
        "z_min": min((row["z"] for row in rows), default=None),
        "z_max": max((row["z"] for row in rows), default=None),
        "document_scores": rows,
    }


def summarize(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    reviews = []
    for source in manifest["machine_sources"]:
        path = folder / "reviews" / f"{source['prompt_id']}.json"
        if path.exists():
            reviews.append(json.loads(path.read_text()))
    quality_pass = [row for row in reviews if row["quality_pass"]]
    cohorts = {"human_calibration": [], "human_test": [], "machine": []}
    expected = {
        "human_calibration": 100,
        "human_test": 100,
        "machine": len(quality_pass),
    }
    for cohort in cohorts:
        for path in sorted((folder / "scores").glob(f"{cohort}-*.json")):
            cohorts[cohort].append(json.loads(path.read_text()))
    generation_complete = len(reviews) == MACHINE_DOCUMENTS
    scoring_complete = all(len(cohorts[name]) == expected[name] for name in cohorts)
    complete = generation_complete and scoring_complete
    calibration = {}
    gates = {
        "machine_quality_sample": len(quality_pass) >= MIN_MACHINE_PASS,
        "human_test_fpr": False,
        "machine_test_fpr": False,
    }
    if complete:
        calibration_sentences = [
            sentence for document in cohorts["human_calibration"]
            for sentence in document["sentences"]
        ]
        lambda_hat = sum(row["green"] for row in calibration_sentences) / len(calibration_sentences)
        if not 0 < lambda_hat < 1:
            raise ValueError("Degenerate calibration sentence rate")
        calibration_z = document_scores(cohorts["human_calibration"], lambda_hat, math.inf)
        threshold = empirical_threshold([row["z"] for row in calibration_z], TARGET_FPR)
        calibrated = {
            name: result_metrics(document_scores(rows, lambda_hat, threshold))
            for name, rows in cohorts.items()
        }
        fixed = {
            name: result_metrics(document_scores(rows, FIXED_NULL, FIXED_Z))
            for name, rows in cohorts.items()
        }
        calibration = {
            "lambda_hat": lambda_hat,
            "sentence_count": len(calibration_sentences),
            "green_sentences": sum(row["green"] for row in calibration_sentences),
            "target_fpr": TARGET_FPR,
            "document_z_threshold": threshold,
            "strict_greater_than": True,
            "calibrated": calibrated,
            "fixed_reference": fixed,
        }
        gates["human_test_fpr"] = calibrated["human_test"]["rate"] <= 0.05
        gates["machine_test_fpr"] = (
            calibrated["machine"]["rate"] is not None
            and calibrated["machine"]["rate"] <= 0.10
        )
    result = {
        "status": "complete" if complete else "incomplete",
        "allocated_machine_documents": MACHINE_DOCUMENTS,
        "machine_quality_pass": len(quality_pass),
        "machine_gate_pass_counts": {
            key: sum(row["gates"][key] for row in reviews)
            for key in ("structure", "style", "facts")
        },
        "cohort_completion": {key: len(value) for key, value in cohorts.items()},
        "calibration": calibration,
        "gates": gates,
        "joint_gate": bool(complete and all(gates.values())),
        "decision": (
            "bank100_ready_for_separate_injection_pilot"
            if complete and all(gates.values())
            else "stop_bank100_injection_route"
            if complete
            else "pending"
        ),
        "api_attempts": len(list((folder / "calls").glob("*.json"))),
        "neural_parser_sentences": sum(
            len(json.loads(path.read_text())["sentences"])
            for path in (folder / "parser").glob("*.json")
        ),
        "s2match_calls": sum(
            json.loads(path.read_text())["s2match_calls"]
            for path in (folder / "scores").glob("*.json")
        ),
        "new_watermark_injection_or_attack_calls": 0,
        "manifest_sha256": sha(folder / "manifest.json"),
    }
    write_json(folder / "summary.json", result)
    if complete:
        write_report(folder, result)
    return result


def write_report(folder, result):
    calibration = result["calibration"]
    lines = [
        "# SWAN bank100 双分布零假设独立校准",
        "",
        f"机器质量门：**{result['machine_quality_pass']}/{MACHINE_DOCUMENTS}**；"
        f"联合继续门：**{'通过' if result['joint_gate'] else '未通过'}**。",
        "",
        f"人类校准集估计 `lambda_hat={calibration['lambda_hat']:.6f}`，"
        f"经验 5% 文档阈值为 `z>{calibration['document_z_threshold']:.6f}`。",
        "",
        "| 规则 | 人类校准 | 人类保留 | 质量合格机器文本 |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (("校准规则", "calibrated"), ("固定 λ=.25, z>2.33", "fixed_reference")):
        values = calibration[key]
        lines.append(
            f"| {label} | {values['human_calibration']['detected']}/"
            f"{values['human_calibration']['documents']} | "
            f"{values['human_test']['detected']}/{values['human_test']['documents']} | "
            f"{values['machine']['detected']}/{values['machine']['documents']} |"
        )
    lines += [
        "",
        f"- API 请求：**{result['api_attempts']}**；本地 parser 句数："
        f"**{result['neural_parser_sentences']}**；S2Match 调用：**{result['s2match_calls']}**。",
        "- 本轮无水印注入、攻击、ASR 或 TPR。",
        f"- 决策：`{result['decision']}`。",
        "",
    ]
    (folder / "report.md").write_text("\n".join(lines))


def run(folder, key):
    manifest = check(folder)
    with Heartbeat(folder) as heartbeat:
        api = Journal(folder, key, heartbeat)
        probe = api.request("probe", request_body("Reply exactly OK.", 0, 16))
        if probe["finish_reason"] != "stop" or probe["content"].strip() != "OK":
            raise ValueError("Unexpected connectivity response")
        api.model_returned = probe["model"]
        runtime = freeze(folder / "runtime.json", {
            "manifest_sha256": sha(folder / "manifest.json"),
            "model_returned": probe["model"],
            "probe_id": probe["id"],
        })
        generate_and_review(folder, manifest, api, runtime, heartbeat)
        parser_calls = parse_quality_machine(folder, manifest, runtime, heartbeat)
        s2match_calls = score_all(folder, manifest, heartbeat)
        heartbeat.update(
            stage="complete",
            new_api_calls=api.sent,
            new_neural_parser_calls=parser_calls,
            new_s2match_calls=s2match_calls,
        )
    return summarize(folder)


def audit(folder):
    import nltk

    manifest = check(folder)
    runtime = json.loads((folder / "runtime.json").read_text())
    calls = [json.loads(path.read_text()) for path in sorted((folder / "calls").glob("*.json"))]
    if (
        len(calls) > API_CAP
        or any(
            call["status"] != "complete"
            or digest(call["request"]) != call["request_sha256"]
            or call["response"]["model"] != runtime["model_returned"]
            for call in calls
        )
    ):
        raise ValueError("API log inventory mismatch")

    class Replay(CallJournal):
        def request(self, scope, body):
            ordinal = self.counters.get(scope, 0)
            path = folder / "calls" / f"{scope}-{ordinal:03d}.json"
            if not path.exists():
                raise ValueError("Audit replay would require an API call")
            return super().request(scope, body)

    api = Replay(folder, "", 0)
    api.model_returned = runtime["model_returned"]
    api.request("probe", request_body("Reply exactly OK.", 0, 16))
    for source in manifest["machine_sources"]:
        body = generation_body(source["source"], "onepass", 0, [])
        response = api.request(f"generate-{source['prompt_id']}", body)
        text = response["content"].strip()
        expected = {
            "prompt_id": source["prompt_id"],
            "source_sha256": source["source_sha256"],
            "runtime_sha256": digest(runtime),
            "request_sha256": digest(body),
            "response": response,
            "text": text,
            "text_sha256": text_sha(text),
            "structure": structural_metrics(text, response["finish_reason"], nltk.sent_tokenize),
        }
        saved = json.loads(
            (folder / "generated" / f"{source['prompt_id']}.json").read_text()
        )
        if saved != expected:
            raise ValueError("Generated checkpoint mismatch")
    for source in manifest["machine_sources"]:
        generated = json.loads(
            (folder / "generated" / f"{source['prompt_id']}.json").read_text()
        )
        style_body = request_body(BASELINE_REVIEW + generated["text"], 0, 700)
        source_fact_body = fact_body(source["source"], generated["text"])
        style_response = api.request(f"style-{source['prompt_id']}", style_body)
        fact_response = api.request(f"facts-{source['prompt_id']}", source_fact_body)
        errors = {}
        try:
            style = decode_baseline_review(style_response["content"])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            style, errors["style"] = None, type(exc).__name__
        try:
            facts = decode_fact_review(fact_response["content"])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            facts, errors["facts"] = None, type(exc).__name__
        gates = {
            "structure": generated["structure"]["pass"],
            "style": style_pass(style),
            "facts": factual_pass(facts),
        }
        expected = {
            "prompt_id": source["prompt_id"],
            "source_sha256": source["source_sha256"],
            "text_sha256": generated["text_sha256"],
            "runtime_sha256": digest(runtime),
            "style_request_sha256": digest(style_body),
            "fact_request_sha256": digest(source_fact_body),
            "style_response": style_response,
            "fact_response": fact_response,
            "style": style,
            "facts": facts,
            "decode_errors": errors,
            "gates": gates,
            "quality_pass": all(gates.values()),
        }
        saved = json.loads(
            (folder / "reviews" / f"{source['prompt_id']}.json").read_text()
        )
        if saved != expected:
            raise ValueError("Review checkpoint mismatch")

    utils = load_utils()
    bank_rows = json.loads(BANK_PATH.read_text())
    bank = [row["normalized_template_amr"] for row in bank_rows]
    bank_counts = [s2_counts(template) for template in bank]
    parser_sentences = 0
    for source in manifest["machine_sources"]:
        review = json.loads(
            (folder / "reviews" / f"{source['prompt_id']}.json").read_text()
        )
        path = folder / "parser" / f"{source['prompt_id']}.json"
        if not review["quality_pass"]:
            if path.exists():
                raise ValueError("Parser output exists for quality-failed document")
            continue
        generated = json.loads(
            (folder / "generated" / f"{source['prompt_id']}.json").read_text()
        )
        parsed = json.loads(path.read_text())
        sentences = nltk.sent_tokenize(generated["text"])
        rebuilt = []
        for index, (sentence, saved) in enumerate(zip(sentences, parsed["sentences"])):
            identity = {
                "sentence_index": index,
                "text": sentence,
                "text_sha256": text_sha(sentence),
                "words": len(sentence.split()),
            }
            row = analyze_parser_output(
                identity, saved["raw_parser_output"], utils, bank_counts
            )
            row["input_tokens"] = saved["input_tokens"]
            rebuilt.append(row)
        if rebuilt != parsed["sentences"] or len(rebuilt) != 5:
            raise ValueError("Parser-derived output mismatch")
        parser_sentences += len(rebuilt)

    bank_count_map = {text_sha(template): s2_counts(template) for template in bank}
    scoring = scoring_documents(folder, manifest)
    scored_documents = 0
    scored_sentences = 0
    scored_calls = 0
    for cohort, rows in scoring.items():
        for document_id, amrs in rows:
            saved = json.loads(score_path(folder, cohort, document_id).read_text())
            if (
                saved["cohort"] != cohort
                or saved["document_id"] != document_id
                or saved["amr_sha256"] != [text_sha(amr) for amr in amrs]
                or len(saved["sentences"]) != len(amrs)
            ):
                raise ValueError("Score checkpoint identity mismatch")
            for sentence, amr in zip(saved["sentences"], amrs):
                candidate_counts = s2_counts(amr)
                feasible = [
                    text_sha(template) for template in bank
                    if count_upper_bound(candidate_counts, bank_count_map[text_sha(template)])
                    >= SCORE_THRESHOLD
                ]
                observations = sentence["score_observations"]
                if (
                    sentence["amr_sha256"] != text_sha(amr)
                    or sentence["candidate_counts"] != candidate_counts
                    or sentence["count_compatible_templates"] != len(feasible)
                    or sentence["s2match_calls"] != len(observations)
                ):
                    raise ValueError("Sentence score identity mismatch")
                for observation in observations:
                    expected_bound = count_upper_bound(
                        candidate_counts,
                        bank_count_map[observation["template_sha256"]],
                    )
                    if (
                        abs(expected_bound - observation["count_upper_bound"]) > 1e-12
                        or observation["score"] > expected_bound + 1e-6
                    ):
                        raise ValueError("Saved score bound mismatch")
                observed_ids = {row["template_sha256"] for row in observations}
                if sentence["green"]:
                    witness = sentence["witness"]
                    if not witness:
                        raise ValueError("Green witness missing")
                    template_id = text_sha(bank[witness["template_index"]])
                    if (
                        witness["score"] < SCORE_THRESHOLD
                        or witness["template_sha256"] != template_id
                        or template_id not in observed_ids
                        or witness["score"] != next(
                            row["score"] for row in observations
                            if row["template_sha256"] == template_id
                        )
                    ):
                        raise ValueError("Green witness missing")
                elif any(template_id not in observed_ids for template_id in feasible):
                    raise ValueError("Non-green sentence omitted feasible template")
                scored_sentences += 1
                scored_calls += len(observations)
            scored_documents += 1
    summary = summarize(folder)
    if (
        summary["status"] != "complete"
        or summary["api_attempts"] != len(calls)
        or sum(api.counters.values()) != len(calls)
        or summary["neural_parser_sentences"] != parser_sentences
        or summary["s2match_calls"] != scored_calls
    ):
        raise ValueError("Summary accounting mismatch")
    result = {
        "status": "passed",
        "mismatches": 0,
        "api_calls_replayed": len(calls),
        "parser_sentences_reconstructed": parser_sentences,
        "score_documents_checked": scored_documents,
        "score_sentences_checked": scored_sentences,
        "saved_s2match_observations_checked": scored_calls,
        "new_api_calls": api.sent,
        "new_neural_parser_calls": 0,
        "new_s2match_calls": 0,
        "summary_sha256": sha(folder / "summary.json"),
    }
    write_json(folder / "audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "summarize", "audit"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    configure_environment()
    with claim(args.output / ".runner.lock"):
        if args.stage == "prepare":
            result = prepare(args.output)
            output = {
                "status": "prepared",
                "machine_sources": len(result["machine_sources"]),
                "human_calibration_documents": len(result["human_inventory"]["calibration"]),
                "human_test_documents": len(result["human_inventory"]["test"]),
                "api_cap": result["protocol"]["api_cap"],
                "manifest_sha256": sha(args.output / "manifest.json"),
            }
        elif args.stage == "run":
            key = read_credential(args.key_file) if args.key_file else os.environ.get("DEEPSEEK_API_KEY", "")
            if not key or any(character.isspace() for character in key):
                raise ValueError("Expected one nonempty API credential")
            try:
                output = run(args.output, key)
            finally:
                if (args.output / "manifest.json").exists():
                    print(json.dumps(summarize(args.output), indent=2), flush=True)
        elif args.stage == "audit":
            output = audit(args.output)
        else:
            output = summarize(args.output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
