"""Run the frozen SWAN oracle-composition detector diagnostic."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import time

from feedback2_runtime import Heartbeat, claim
from run_confirmation_experiment import Detector, digest, freeze
from run_confirmation_mechanism import lexical_tokens, punctuation_merge
from run_segmentation_intervention import ROOT, sha, text_sha, write_json


OUTPUT = ROOT / "experiments/swan_oracle_composition1"
SOURCE = ROOT / "experiments/confirmation_three_arm30"
MECHANISM = ROOT / "experiments/confirmation_mechanism30"
BANK_PATH = ROOT / "third_party/SWAN/amr_bank/banks/amr_bank_50.json"
MODEL_DIR = ROOT / "models/model_parse_xfm_bart_large-v0_1_0"
TOKENIZER_DIR = ROOT / "models/bart-large-tokenizer"
VECTORS_PATH = ROOT / "vectors/glove.6B.100d.txt"
EXPECTED_IDS = [f"c4-validation-{index}" for index in range(250, 280)]
LAMBDA = 0.25
GREEN_THRESHOLD = 0.65
DETECTION_THRESHOLD = 2.33
MIN_BATCH_WORDS = 100
MAX_BATCH_WORDS = 220


def configure_environment():
    os.environ.setdefault("NLTK_DATA", str(ROOT / ".nltk_data"))
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
    os.environ.setdefault("GLOVE_VECTORS_PATH", str(VECTORS_PATH))


def z_score(green, n, null_rate=LAMBDA):
    if n <= 0 or not 0 < null_rate < 1:
        raise ValueError("Invalid z-score inputs")
    return (green - null_rate * n) / math.sqrt(
        n * null_rate * (1 - null_rate)
    )


def count_only_z(original_green, original_n, target_n):
    if original_n <= 0 or target_n <= 0:
        raise ValueError("Invalid count-only inputs")
    rate = original_green / original_n
    return (rate - LAMBDA) * math.sqrt(target_n / (LAMBDA * (1 - LAMBDA)))


def validate_detail(detail):
    scores = detail.get("bank_scores")
    if (
        not isinstance(scores, list)
        or len(scores) != 50
        or not all(
            isinstance(value, (int, float))
            and math.isfinite(value)
            and 0 <= value <= 1
            for value in scores
        )
    ):
        raise ValueError("Invalid bank50 score vector")
    if not math.isclose(detail["score"], max(scores), abs_tol=1e-12):
        raise ValueError("Saved detail maximum differs from bank scores")
    return detail["score"] >= GREEN_THRESHOLD


def detection_from_details(details):
    if not details:
        raise ValueError("Detection requires at least one unit")
    green = sum(validate_detail(detail) for detail in details)
    n = len(details)
    return {
        "n": n,
        "green": green,
        "green_rate": green / n,
        "z": z_score(green, n),
        "detected": z_score(green, n) > DETECTION_THRESHOLD,
        "z_ceiling": math.sqrt(3 * n),
    }


def validate_detection(detection, expected_texts):
    details = detection.get("details")
    if [detail.get("text") for detail in details] != expected_texts:
        raise ValueError("Saved detector units differ from expected text")
    rebuilt = detection_from_details(details)
    for key in ("n", "green"):
        if detection[key] != rebuilt[key]:
            raise ValueError(f"Saved detection {key} is inconsistent")
    if not math.isclose(detection["z"], rebuilt["z"], abs_tol=1e-12):
        raise ValueError("Saved detection z is inconsistent")
    return rebuilt


def pair_intervention(sentences):
    if len(sentences) != 5:
        raise ValueError("Expected exactly five source sentences")
    units = [
        punctuation_merge(sentences[0:2]),
        punctuation_merge(sentences[2:4]),
        sentences[4],
    ]
    if lexical_tokens(" ".join(units)) != lexical_tokens(" ".join(sentences)):
        raise ValueError("Pair intervention changed lexical tokens")
    return units


def batch_ids():
    return [EXPECTED_IDS[start:start + 5] for start in range(0, 30, 5)]


def primary_decision(groups, integrity_pass):
    pair_count_only_above = all(
        group["conditions"]["pair"]["count_only_z"] > DETECTION_THRESHOLD
        for group in groups
    )
    pair_not_detected = sum(
        not group["conditions"]["pair"]["detected"] for group in groups
    )
    pair_below_count_only = all(
        group["conditions"]["pair"]["z"]
        < group["conditions"]["pair"]["count_only_z"]
        for group in groups
    )
    support = bool(
        integrity_pass
        and pair_count_only_above
        and pair_not_detected >= 5
        and pair_below_count_only
    )
    return {
        "pair_count_only_above_threshold_batches": sum(
            group["conditions"]["pair"]["count_only_z"]
            > DETECTION_THRESHOLD
            for group in groups
        ),
        "pair_observed_at_or_below_threshold_batches": pair_not_detected,
        "pair_observed_below_count_only_batches": sum(
            group["conditions"]["pair"]["z"]
            < group["conditions"]["pair"]["count_only_z"]
            for group in groups
        ),
        "supports_boundary_sensitive_detector_mechanics": support,
        "decision": (
            "supports_oracle_boundary_sensitivity"
            if support
            else "pair_result_inconclusive"
        ),
    }


def package_versions():
    names = (
        "torch",
        "amrlib",
        "transformers",
        "penman",
        "nltk",
        "numpy",
        "scipy",
    )
    return {name: importlib.metadata.version(name) for name in names}


def boundary_inventory():
    rows = []
    for path in sorted((SOURCE / "attacks").glob("*-boundary_only-*.json")):
        value = json.loads(path.read_text())
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha(path),
                "text_sha256": text_sha(value["attack"]),
            }
        )
    if len(rows) != 120:
        raise ValueError(f"Expected 120 historical boundary candidates, got {len(rows)}")
    return {
        "count": len(rows),
        "inventory_sha256": digest(rows),
        "text_sha256": sorted(row["text_sha256"] for row in rows),
    }


def source_rows():
    import nltk

    rows = []
    for prompt_id in EXPECTED_IDS:
        baseline_path = SOURCE / "baselines" / f"{prompt_id}.json"
        full_path = MECHANISM / "detected" / f"punctuation-{prompt_id}.json"
        baseline = json.loads(baseline_path.read_text())
        full = json.loads(full_path.read_text())
        if baseline["prompt_id"] != prompt_id or not baseline["complete"]:
            raise ValueError(f"Incomplete or misidentified baseline: {prompt_id}")
        sentences = [row["text"] for row in baseline["sentences"]]
        if len(sentences) != 5:
            raise ValueError(f"Expected five source slots: {prompt_id}")
        validate_detection(baseline["detection"], sentences)
        record = full["record"]
        if (
            record["prompt_id"] != prompt_id
            or record["segments"]["original"] != sentences
            or record["text"] != punctuation_merge(sentences)
        ):
            raise ValueError(f"Historical full-merge identity mismatch: {prompt_id}")
        full_variant = full["variants"]["merged_sentence"]
        validate_detection(full_variant, [record["text"]])
        pair_units = pair_intervention(sentences)
        pair_text = " ".join(pair_units)
        if len(nltk.sent_tokenize(pair_text)) != 3:
            raise ValueError(f"Pair intervention did not produce three units: {prompt_id}")
        if len(nltk.sent_tokenize(record["text"])) != 1:
            raise ValueError(f"Full intervention did not produce one unit: {prompt_id}")
        rows.append(
            {
                "prompt_id": prompt_id,
                "baseline_path": str(baseline_path.relative_to(ROOT)),
                "baseline_sha256": sha(baseline_path),
                "full_path": str(full_path.relative_to(ROOT)),
                "full_sha256": sha(full_path),
                "original_text_sha256": baseline["text_sha256"],
                "full_text_sha256": record["text_sha256"],
                "pair_text": pair_text,
                "pair_text_sha256": text_sha(pair_text),
                "pair_units": [
                    {
                        "pair_index": index,
                        "source_slots": [index * 2, index * 2 + 1],
                        "text": pair_units[index],
                        "text_sha256": text_sha(pair_units[index]),
                    }
                    for index in range(2)
                ],
                "tail_text_sha256": text_sha(pair_units[2]),
                "words": len(baseline["text"].split()),
                "accepted_slots": sum(row["accepted"] for row in baseline["sentences"]),
                "fallback_slots": sum(
                    not row["accepted"] for row in baseline["sentences"]
                ),
            }
        )
    if [row["prompt_id"] for row in rows] != EXPECTED_IDS:
        raise ValueError("Source order differs from the frozen cohort")
    return rows


def dependency_paths():
    paths = [
        Path(__file__),
        ROOT / "scripts/audit_swan_oracle_composition.py",
        OUTPUT / "protocol.md",
        SOURCE / "final_analysis.json",
        SOURCE / "runtime.json",
        MECHANISM / "manifest.json",
        MECHANISM / "summary.json",
        ROOT / "scripts/feedback2_runtime.py",
        ROOT / "scripts/run_confirmation_experiment.py",
        ROOT / "scripts/run_confirmation_mechanism.py",
        ROOT / "scripts/run_segmentation_intervention.py",
        BANK_PATH,
        ROOT / "third_party/SWAN/utils/amr_utils.py",
        ROOT / "third_party/SWAN/utils/s2match_patch.py",
        ROOT
        / "third_party/SWAN/amr-metric-suite/py3-Smatch-and-S2match/smatch/s2match.py",
        ROOT
        / "third_party/SWAN/amr-metric-suite/py3-Smatch-and-S2match/smatch/amr_py3.py",
        MODEL_DIR / "amrlib_meta.json",
        MODEL_DIR / "model_parse_xfm_bart_large.json",
        MODEL_DIR / "config.json",
        MODEL_DIR / "pytorch_model.bin",
        TOKENIZER_DIR / "tokenizer.json",
        TOKENIZER_DIR / "tokenizer_config.json",
        TOKENIZER_DIR / "config.json",
        TOKENIZER_DIR / "merges.txt",
        TOKENIZER_DIR / "vocab.json",
        VECTORS_PATH,
    ]
    paths.extend(SOURCE / "baselines" / f"{prompt_id}.json" for prompt_id in EXPECTED_IDS)
    paths.extend(
        MECHANISM / "detected" / f"punctuation-{prompt_id}.json"
        for prompt_id in EXPECTED_IDS
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
    final = json.loads((SOURCE / "final_analysis.json").read_text())
    mechanism = json.loads((MECHANISM / "summary.json").read_text())
    if final["status"] != "complete_independently_verified":
        raise ValueError("Source confirmation experiment is not final")
    if mechanism["status"] != "complete_verified":
        raise ValueError("Historical mechanism experiment is not final")
    rows = source_rows()
    inventory = boundary_inventory()
    historical_hashes = set(inventory["text_sha256"])
    overlaps = [
        row["prompt_id"]
        for row in rows
        if row["pair_text_sha256"] in historical_hashes
    ]
    groups = []
    lookup = {row["prompt_id"]: row for row in rows}
    for index, ids in enumerate(batch_ids()):
        groups.append(
            {
                "batch_id": f"batch-{index:02d}",
                "prompt_ids": ids,
                "words": sum(lookup[prompt_id]["words"] for prompt_id in ids),
                "original_units": 25,
                "pair_units": 15,
                "full_units": 5,
            }
        )
    manifest = {
        "status": "frozen_before_pair_parser",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Retrospective oracle detector-mechanics diagnostic; only the pair "
            "condition is previously unobserved"
        ),
        "sources": rows,
        "groups": groups,
        "historical_boundary_candidates": {
            "count": inventory["count"],
            "inventory_sha256": inventory["inventory_sha256"],
            "pair_text_exact_overlap_prompt_ids": overlaps,
        },
        "protocol": {
            "documents": 30,
            "groups": 6,
            "grouping": "Ascending row id, five consecutive documents per batch",
            "conditions": {
                "original": "Five saved units per document; 25 per batch",
                "pair": "Merge slots 0+1 and 2+3; retain slot 4; 15 per batch",
                "full": "Historical five-to-one punctuation control; 5 per batch",
            },
            "fresh_local_parser_units": 60,
            "fresh_s2match_observations": 3000,
            "generation_or_review_api_calls": 0,
            "bank": "official bank50",
            "green_threshold": GREEN_THRESHOLD,
            "lambda": LAMBDA,
            "detection_rule": "z > 2.33",
            "normal_batch_word_range": [MIN_BATCH_WORDS, MAX_BATCH_WORDS],
            "primary_decision": {
                "all_integrity_gates": True,
                "all_pair_count_only_z_above": DETECTION_THRESHOLD,
                "pair_batches_at_or_below_threshold": 5,
                "all_pair_z_below_count_only": True,
            },
            "reporting_boundary": (
                "Not natural-text ASR, TPR, FPR, quality, or independent confirmation"
            ),
        },
        "dependencies": dependencies(),
        "packages": package_versions(),
    }
    freeze(manifest_path, manifest)
    return manifest


def check(folder=OUTPUT):
    configure_environment()
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["status"] != "frozen_before_pair_parser":
        raise ValueError("Unexpected oracle manifest status")
    for relative, expected in manifest["dependencies"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Frozen dependency changed: {relative}")
    if package_versions() != manifest["packages"]:
        raise ValueError("Frozen package versions changed")
    if manifest["sources"] != source_rows():
        raise ValueError("Frozen source inventory changed")
    inventory = boundary_inventory()
    frozen_inventory = manifest["historical_boundary_candidates"]
    if (
        inventory["count"] != frozen_inventory["count"]
        or inventory["inventory_sha256"] != frozen_inventory["inventory_sha256"]
    ):
        raise ValueError("Historical boundary candidate inventory changed")
    overlaps = [
        row["prompt_id"]
        for row in manifest["sources"]
        if row["pair_text_sha256"] in set(inventory["text_sha256"])
    ]
    if overlaps != frozen_inventory["pair_text_exact_overlap_prompt_ids"]:
        raise ValueError("Historical pair-text overlap changed")
    return manifest


def runtime_payload(folder):
    return {
        "manifest_sha256": sha(folder / "manifest.json"),
        "device": "cpu",
        "batch_size": 1,
        "num_beams": 4,
        "bank_sha256": sha(BANK_PATH),
        "vectors_sha256": sha(VECTORS_PATH),
        "packages": package_versions(),
        "model_files": {
            str(path.relative_to(MODEL_DIR)): sha(path)
            for path in (
                MODEL_DIR / "amrlib_meta.json",
                MODEL_DIR / "model_parse_xfm_bart_large.json",
                MODEL_DIR / "config.json",
                MODEL_DIR / "pytorch_model.bin",
            )
        },
        "green_threshold": GREEN_THRESHOLD,
        "lambda": LAMBDA,
        "detection_threshold": DETECTION_THRESHOLD,
        "network_api_calls": 0,
        "s2match_policy": (
            "Checkpoint first pair-unit observation; never rerun based on value"
        ),
    }


def parser_path(folder, prompt_id, pair_index):
    return folder / "pair_parser" / f"{prompt_id}-{pair_index}.json"


def score_path(folder, prompt_id, pair_index):
    return folder / "pair_scores" / f"{prompt_id}-{pair_index}.json"


def run_pair_unit(folder, source, unit, detector, runtime, heartbeat):
    prompt_id = source["prompt_id"]
    pair_index = unit["pair_index"]
    identity = {
        "prompt_id": prompt_id,
        "pair_index": pair_index,
        "source_slots": unit["source_slots"],
        "text": unit["text"],
        "text_sha256": unit["text_sha256"],
        "runtime_sha256": digest(runtime),
    }
    parsed_path = parser_path(folder, prompt_id, pair_index)
    if parsed_path.exists():
        parsed = json.loads(parsed_path.read_text())
        if any(parsed[key] != value for key, value in identity.items()):
            raise ValueError("Pair parser checkpoint identity changed")
    else:
        heartbeat.update(
            stage="parser", prompt_id=prompt_id, pair_index=pair_index
        )
        started = time.monotonic()
        details = detector.parse(unit["text"])
        if len(details) != 1 or details[0]["text"] != unit["text"]:
            raise ValueError("Pair text did not parse as one detector unit")
        parsed = {
            **identity,
            "detail": details[0],
            "elapsed_seconds": time.monotonic() - started,
        }
        freeze(parsed_path, parsed)
    scored_path = score_path(folder, prompt_id, pair_index)
    score_identity = {
        "prompt_id": prompt_id,
        "pair_index": pair_index,
        "text_sha256": unit["text_sha256"],
        "parser_sha256": sha(parsed_path),
        "runtime_sha256": digest(runtime),
    }
    if scored_path.exists():
        scored = json.loads(scored_path.read_text())
        if any(scored[key] != value for key, value in score_identity.items()):
            raise ValueError("Pair score checkpoint identity changed")
    else:
        heartbeat.update(
            stage="s2match", prompt_id=prompt_id, pair_index=pair_index
        )
        started = time.monotonic()
        detection = detector.score([parsed["detail"]])
        validate_detection(detection, [unit["text"]])
        scored = {
            **score_identity,
            "detail": detection["details"][0],
            "green": bool(detection["green"]),
            "elapsed_seconds": time.monotonic() - started,
        }
        freeze(scored_path, scored)
    validate_detail(scored["detail"])
    return scored


def load_source_artifacts(source):
    baseline = json.loads((ROOT / source["baseline_path"]).read_text())
    full = json.loads((ROOT / source["full_path"]).read_text())
    return baseline, full


def build_summary(folder=OUTPUT):
    manifest = json.loads((folder / "manifest.json").read_text())
    documents = []
    condition_details = {}
    accepted_total = 0
    fallback_total = 0
    lexical_checks = []
    for source in manifest["sources"]:
        prompt_id = source["prompt_id"]
        baseline, full = load_source_artifacts(source)
        original_details = baseline["detection"]["details"]
        pair_details = [
            json.loads(score_path(folder, prompt_id, index).read_text())["detail"]
            for index in range(2)
        ]
        pair_details.append(original_details[4])
        full_details = full["variants"]["merged_sentence"]["details"]
        original = detection_from_details(original_details)
        pair = detection_from_details(pair_details)
        full_metrics = detection_from_details(full_details)
        pair_texts = [detail["text"] for detail in pair_details]
        lexical_ok = (
            lexical_tokens(" ".join(pair_texts))
            == lexical_tokens(baseline["text"])
            == lexical_tokens(full["record"]["text"])
        )
        lexical_checks.append(lexical_ok)
        accepted_total += source["accepted_slots"]
        fallback_total += source["fallback_slots"]
        documents.append(
            {
                "prompt_id": prompt_id,
                "words": source["words"],
                "accepted_slots": source["accepted_slots"],
                "fallback_slots": source["fallback_slots"],
                "lexical_sequence_identical": lexical_ok,
                "original": original,
                "pair": pair,
                "full": full_metrics,
            }
        )
        condition_details[prompt_id] = {
            "original": original_details,
            "pair": pair_details,
            "full": full_details,
        }
    document_lookup = {row["prompt_id"]: row for row in documents}
    groups = []
    for group in manifest["groups"]:
        ids = group["prompt_ids"]
        conditions = {}
        for name in ("original", "pair", "full"):
            details = [
                detail
                for prompt_id in ids
                for detail in condition_details[prompt_id][name]
            ]
            conditions[name] = detection_from_details(details)
        original = conditions["original"]
        for name in ("pair", "full"):
            conditions[name]["count_only_z"] = count_only_z(
                original["green"], original["n"], conditions[name]["n"]
            )
            conditions[name]["observed_minus_count_only_z"] = (
                conditions[name]["z"] - conditions[name]["count_only_z"]
            )
        original_sentences = [
            detail["text"]
            for prompt_id in ids
            for detail in condition_details[prompt_id]["original"]
        ]
        groups.append(
            {
                **group,
                "accepted_slots": sum(
                    document_lookup[prompt_id]["accepted_slots"]
                    for prompt_id in ids
                ),
                "fallback_slots": sum(
                    document_lookup[prompt_id]["fallback_slots"]
                    for prompt_id in ids
                ),
                "unique_original_sentences": len(set(original_sentences)),
                "exact_duplicate_original_sentences": (
                    len(original_sentences) - len(set(original_sentences))
                ),
                "conditions": conditions,
            }
        )
    pair_files = list((folder / "pair_scores").glob("*.json"))
    parser_files = list((folder / "pair_parser").glob("*.json"))
    integrity = {
        "documents_30": len(documents) == 30,
        "groups_6": len(groups) == 6,
        "original_units_150": sum(
            row["original"]["n"] for row in documents
        )
        == 150,
        "pair_units_90": sum(row["pair"]["n"] for row in documents) == 90,
        "full_units_30": sum(row["full"]["n"] for row in documents) == 30,
        "new_pair_parser_checkpoints_60": len(parser_files) == 60,
        "new_pair_score_checkpoints_60": len(pair_files) == 60,
        "accepted_slots_137": accepted_total == 137,
        "fallback_slots_13": fallback_total == 13,
        "lexical_sequence_identical": all(lexical_checks),
        "normal_batch_words": all(
            MIN_BATCH_WORDS <= group["words"] <= MAX_BATCH_WORDS
            for group in groups
        ),
        "fixed_grouping": [group["prompt_ids"] for group in groups]
        == batch_ids(),
        "no_historical_pair_text_exact_overlap": not manifest[
            "historical_boundary_candidates"
        ]["pair_text_exact_overlap_prompt_ids"],
    }
    integrity_pass = all(integrity.values())
    primary = primary_decision(groups, integrity_pass)
    result = {
        "status": "complete_oracle_diagnostic",
        "scope": manifest["scope"],
        "integrity": integrity,
        "integrity_pass": integrity_pass,
        "documents": documents,
        "groups": groups,
        "primary": primary,
        "historical_context": {
            "original_detected_batches": sum(
                group["conditions"]["original"]["detected"] for group in groups
            ),
            "full_detected_batches": sum(
                group["conditions"]["full"]["detected"] for group in groups
            ),
            "full_endpoint_known_before_protocol": True,
        },
        "resource_counts": {
            "generation_review_api_calls": 0,
            "new_local_parser_units": len(parser_files),
            "new_s2match_observations": 50 * len(pair_files),
            "reused_original_units": 150,
            "reused_full_units": 30,
        },
        "aggregate_words": {
            "min": min(group["words"] for group in groups),
            "median": statistics.median(group["words"] for group in groups),
            "mean": statistics.mean(group["words"] for group in groups),
            "max": max(group["words"] for group in groups),
        },
        "limitations": [
            "Previously observed source cohort; only the pair midpoint was unobserved.",
            "Original and full endpoints were known before protocol freeze.",
            "Batches combine unrelated topics and are statistical containers only.",
            "Source prose is repetitive and not a natural-quality baseline.",
            "No new FPR calibration; bank50 fixed lambda and threshold are reused.",
            "All 13 injection fallback slots remain in the denominator.",
            "Not natural-text ASR, TPR, FPR, quality, or domain generalization evidence.",
        ],
        "manifest_sha256": sha(folder / "manifest.json"),
        "runtime_sha256": sha(folder / "runtime.json"),
    }
    return result


def report_text(summary):
    lines = [
        "# SWAN oracle composition diagnostic",
        "",
        "This is a retrospective detector-mechanics diagnostic. The batches are "
        "not coherent natural documents, and the result is not ASR, TPR, or FPR.",
        "",
        "| Batch | Words | Accepted | Original n/g/z | Pair n/g/z | "
        "Pair count-only z | Full n/g/z |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in summary["groups"]:
        original = group["conditions"]["original"]
        pair = group["conditions"]["pair"]
        full = group["conditions"]["full"]
        lines.append(
            f"| {group['batch_id']} | {group['words']} | "
            f"{group['accepted_slots']}/25 | "
            f"{original['n']}/{original['green']}/{original['z']:.3f} | "
            f"{pair['n']}/{pair['green']}/{pair['z']:.3f} | "
            f"{pair['count_only_z']:.3f} | "
            f"{full['n']}/{full['green']}/{full['z']:.3f} |"
        )
    primary = summary["primary"]
    lines.extend(
        [
            "",
            f"- Batch words: {summary['aggregate_words']['min']}--"
            f"{summary['aggregate_words']['max']}.",
            "- Pair batches at or below z=2.33: "
            f"**{primary['pair_observed_at_or_below_threshold_batches']}/6**.",
            "- Pair count-only counterfactuals above z=2.33: "
            f"**{primary['pair_count_only_above_threshold_batches']}/6**.",
            "- Pair observed z below count-only counterfactual: "
            f"**{primary['pair_observed_below_count_only_batches']}/6**.",
            f"- Integrity gate: **{'passed' if summary['integrity_pass'] else 'failed'}**.",
            f"- Frozen decision: `{primary['decision']}`.",
            "- New resources: "
            f"{summary['resource_counts']['new_local_parser_units']} local parser units, "
            f"{summary['resource_counts']['new_s2match_observations']} saved S2Match "
            "observations, 0 API calls.",
            "",
            "The historical full-merge endpoint is descriptive only. Source quality "
            "and prior observation prevent natural-text or independent-confirmation claims.",
            "",
        ]
    )
    return "\n".join(lines)


def write_summary(folder=OUTPUT):
    summary = build_summary(folder)
    write_json(folder / "summary.json", summary)
    (folder / "report.md").write_text(report_text(summary))
    return summary


def run(folder=OUTPUT):
    manifest = check(folder)
    runtime = freeze(folder / "runtime.json", runtime_payload(folder))
    missing = [
        (source, unit)
        for source in manifest["sources"]
        for unit in source["pair_units"]
        if not parser_path(
            folder, source["prompt_id"], unit["pair_index"]
        ).exists()
        or not score_path(
            folder, source["prompt_id"], unit["pair_index"]
        ).exists()
    ]
    if not missing:
        return write_summary(folder)
    with claim(folder / "run.lock"), Heartbeat(folder) as heartbeat:
        heartbeat.update(stage="loading_detector", remaining_units=len(missing))
        detector = Detector()
        if len(detector.bank) != 50:
            raise ValueError("Expected official bank50")
        completed = 0
        for source in manifest["sources"]:
            for unit in source["pair_units"]:
                run_pair_unit(
                    folder, source, unit, detector, runtime, heartbeat
                )
                completed += 1
            heartbeat.update(
                stage="pair_units",
                completed_units=completed,
                total_units=60,
                prompt_id=source["prompt_id"],
            )
            print(
                json.dumps(
                    {
                        "stage": "pair_units",
                        "prompt_id": source["prompt_id"],
                        "completed": completed,
                        "total": 60,
                    }
                ),
                flush=True,
            )
        heartbeat.update(stage="summary")
        return write_summary(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=OUTPUT
    )
    parser.add_argument(
        "--prepare-only", action="store_true"
    )
    args = parser.parse_args()
    configure_environment()
    prepare(args.output)
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "status": "frozen",
                    "manifest": str(args.output / "manifest.json"),
                }
            )
        )
        return
    summary = run(args.output)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["primary"]["decision"],
                "pair_not_detected": summary["primary"][
                    "pair_observed_at_or_below_threshold_batches"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
