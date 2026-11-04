"""Detect the strict 126-word coherent SWAN baseline."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import run_swan_coherent_baseline_cached as base
from swan_feasibility2_common import ROOT, digest, freeze, load, now, sha256
from swan_final1_contract import environment_lock
from swan_final1_transform import sentence_spans, text_hash, unit_texts

OUT = ROOT / "experiments/swan_normal_length_coherent_baseline1"
SOURCE = ROOT / "experiments/swan_normal_length_coherent_pilot1"
EXPECTED_SOURCE_MANIFEST = (
    "bd02fce51992f9f7fcecf4586505de26ce1c030eca9c97345fb61bee1624cb3f"
)
EXPECTED_SOURCE_RESULT = (
    "d71fe9a9946c1c4feff85ef49122267b5a1f1c724294c1b35b1a45b30d042587"
)
EXPECTED_DOCUMENT = (
    "fd1f5d7536d3e9a401005c4b89de153cd9e1875d48a6af5ac321c68bb70b7f31"
)
EQUIVALENCE_PAIRS = ((0, 0), (9, 25), (19, 47))
EQUIVALENCE_TOLERANCE = 1e-12
COUNT = 20
MIN_WORDS = 100
MAX_WORDS = 180


def diagnostics_empty(row):
    diagnostics = row.get("candidate_diagnostics") or {}
    return (
        not diagnostics.get("logged_warnings")
        and not diagnostics.get("python_warnings")
        and not diagnostics.get("stderr")
        and row.get("parser_or_scorer_error_type") is None
    )


def source_inputs():
    manifest_path = SOURCE / "manifest.json"
    result_path = SOURCE / "result.json"
    if sha256(manifest_path) != EXPECTED_SOURCE_MANIFEST \
            or sha256(result_path) != EXPECTED_SOURCE_RESULT:
        raise ValueError("Frozen normal-length source changed")
    result = load(result_path)
    document = result["document"]
    valid_result = (
        result["status"] == "normal_length_baseline_ready"
        and result["accepted_slots"] == COUNT
        and result["fallback_slots"] == 0
        and result["auxiliary_quality_gate_pass"] is True
        and len(result["auxiliary_reviews"]) == 2
        and all(row["pass"] for row in result["auxiliary_reviews"])
        and document["text_sha256"] == EXPECTED_DOCUMENT
        and MIN_WORDS <= document["word_count"] <= MAX_WORDS
        and len(document["segmentation_spans"]) == COUNT
    )
    if not valid_result:
        raise ValueError("Source did not pass frozen normal-length quality gate")

    selected = []
    seen = set()
    for source in result["selected"]:
        slot = source["slot"]
        slot_path = SOURCE / "private/pilot/generation_slots" / (
            f"coherent-slot-{slot:02d}.json"
        )
        saved = load(slot_path)
        trace_path = SOURCE / "private/pilot/generation_attempts" / (
            f"coherent-slot-{slot:02d}-attempt-{saved['terminal_attempt']:02d}.json"
        )
        trace = load(trace_path)
        valid = (
            sha256(slot_path) == source["slot_sha256"]
            == result["slot_sha256"][str(slot)]
            and saved["attempt_trace_sha256"][-1] == sha256(trace_path)
            and saved["accepted"] is True
            and saved["fallback"] is False
            and saved["text"] == source["text"] == trace["candidate"]
            and saved["text_sha256"] == source["text_sha256"]
            == trace["candidate_sha256"] == text_hash(source["text"])
            and saved["score"] == source["score"] == trace["score"]
            and math.isfinite(saved["score"])
            and saved["score"] >= 0.75
            and trace["selected"] is True
            and trace["status"] == "accepted"
            and trace["surface"]["pass"] is True
            and trace["score_computed"] is True
            and trace["official_score_accepted"] is True
            and diagnostics_empty(trace)
        )
        if not valid or source["text_sha256"] in seen:
            raise ValueError("Source slot is not a unique strict acceptance")
        seen.add(source["text_sha256"])
        selected.append({
            "slot": slot,
            "text": source["text"],
            "text_sha256": source["text_sha256"],
            "target_template_index": source["target_template_index"],
            "target_score": source["score"],
            "slot_path": str(slot_path.relative_to(ROOT)),
            "slot_sha256": sha256(slot_path),
            "trace_path": str(trace_path.relative_to(ROOT)),
            "trace_sha256": sha256(trace_path),
        })
    if [row["slot"] for row in selected] != list(range(COUNT)):
        raise ValueError("Source selection is not the exact twenty-slot sequence")
    text = " ".join(row["text"] for row in selected)
    spans = sentence_spans(text)
    if text != document["text"] \
            or text_hash(text) != EXPECTED_DOCUMENT \
            or spans != document["segmentation_spans"] \
            or unit_texts(text, spans) != [row["text"] for row in selected]:
        raise ValueError("Source document text or segmentation changed")
    reviews = []
    for review in result["auxiliary_reviews"]:
        path = ROOT / review["path"]
        if sha256(path) != review["sha256"]:
            raise ValueError("Source quality review changed")
        reviews.append({
            "label": review["label"],
            "path": review["path"],
            "sha256": review["sha256"],
            "parsed": review["parsed"],
            "pass": review["pass"],
        })
    return {
        "source_manifest_path": str(manifest_path.relative_to(ROOT)),
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST,
        "source_result_path": str(result_path.relative_to(ROOT)),
        "source_result_sha256": EXPECTED_SOURCE_RESULT,
        "quality_summary": {
            "model_assisted_gate_pass": True,
            "reviews": reviews,
            "human_quality_reviewed": False,
            "normal_length_words": document["word_count"],
        },
        "selected": selected,
        "document": document,
    }


def dependencies():
    paths = [
        Path(__file__),
        OUT / "protocol.md",
        base.BANK,
        base.GLOVE,
        SOURCE / "manifest.json",
        SOURCE / "result.json",
        ROOT / "scripts/run_swan_normal_length_coherent_pilot.py",
        ROOT / "scripts/run_swan_coherent_baseline_cached.py",
        ROOT / "scripts/swan_cached_s2match.py",
        ROOT / "scripts/run_swan_injection_only_pilot.py",
        ROOT / "scripts/run_swan_positive_collection.py",
        ROOT / "scripts/swan_feasibility2_common.py",
        ROOT / "scripts/swan_final1_contract.py",
        ROOT / "scripts/swan_final1_runtime.py",
        ROOT / "scripts/swan_final1_transform.py",
        ROOT / "scripts/test_swan_normal_length_coherent_baseline.py",
    ]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def prepare():
    if (OUT / "manifest.json").exists():
        return check()
    if sha256(base.BANK) != base.EXPECTED_BANK:
        raise ValueError("Bank50 changed")
    freeze(OUT / "inputs.json", source_inputs())
    freeze(OUT / "environment_lock.json", environment_lock())
    manifest = freeze(OUT / "manifest.json", {
        "schema_version": 1,
        "experiment": "swan_normal_length_coherent_baseline1",
        "frozen_at": now(),
        "status": "frozen_before_detection",
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST,
        "source_result_sha256": EXPECTED_SOURCE_RESULT,
        "document_text_sha256": EXPECTED_DOCUMENT,
        "inputs_sha256": sha256(OUT / "inputs.json"),
        "environment_sha256": sha256(OUT / "environment_lock.json"),
        "bank_sha256": base.EXPECTED_BANK,
        "glove_sha256": sha256(base.GLOVE),
        "dependencies": dependencies(),
        "equivalence_pairs": [list(pair) for pair in EQUIVALENCE_PAIRS],
        "equivalence_tolerance": EQUIVALENCE_TOLERANCE,
        "green_threshold": base.GREEN_THRESHOLD,
        "null_rate": base.NULL_RATE,
        "minimum_document_words": MIN_WORDS,
        "maximum_document_words": MAX_WORDS,
        "new_generation_calls": 0,
        "provider_calls": 0,
        "human_quality_claim": False,
        "source_anchoring_claim": False,
        "empirical_calibration_claim": False,
        "attack_authorized": False,
        "sealed_source_access": False,
    })
    return {
        "state": manifest["status"],
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "inputs_sha256": sha256(OUT / "inputs.json"),
    }


def check(full_environment=True):
    manifest = load(OUT / "manifest.json")
    valid = (
        manifest["experiment"] == "swan_normal_length_coherent_baseline1"
        and manifest["source_manifest_sha256"] == EXPECTED_SOURCE_MANIFEST
        and manifest["source_result_sha256"] == EXPECTED_SOURCE_RESULT
        and manifest["document_text_sha256"] == EXPECTED_DOCUMENT
        and manifest["equivalence_pairs"] == [
            list(pair) for pair in EQUIVALENCE_PAIRS
        ]
        and manifest["equivalence_tolerance"] == EQUIVALENCE_TOLERANCE
        and manifest["green_threshold"] == base.GREEN_THRESHOLD
        and manifest["null_rate"] == base.NULL_RATE
        and manifest["minimum_document_words"] == MIN_WORDS
        and manifest["maximum_document_words"] == MAX_WORDS
        and manifest["new_generation_calls"] == 0
        and manifest["provider_calls"] == 0
        and not manifest["human_quality_claim"]
        and not manifest["source_anchoring_claim"]
        and not manifest["empirical_calibration_claim"]
        and not manifest["attack_authorized"]
        and not manifest["sealed_source_access"]
    )
    if not valid:
        raise ValueError("Normal-length baseline identity changed")
    if sha256(SOURCE / "manifest.json") != EXPECTED_SOURCE_MANIFEST \
            or sha256(SOURCE / "result.json") != EXPECTED_SOURCE_RESULT \
            or sha256(base.BANK) != base.EXPECTED_BANK \
            or sha256(base.GLOVE) != manifest["glove_sha256"] \
            or manifest["inputs_sha256"] != sha256(OUT / "inputs.json") \
            or manifest["environment_sha256"] != sha256(OUT / "environment_lock.json"):
        raise ValueError("Frozen normal-length baseline artifact changed")
    if load(OUT / "inputs.json") != source_inputs():
        raise ValueError("Frozen normal-length baseline inputs changed")
    for relative, expected in manifest["dependencies"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Dependency changed: {relative}")
    if full_environment and environment_lock() != load(OUT / "environment_lock.json"):
        raise ValueError("Execution environment changed")
    return {
        "state": "frozen_hashes_verified",
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "environment_verified": full_environment,
    }


def seed(*parts):
    return int(digest(["swan_normal_length_coherent_baseline1", *parts])[:8], 16)


def detect():
    check()
    result_path = OUT / "detection.json"
    if result_path.exists():
        return load(result_path)
    inputs = load(OUT / "inputs.json")
    document = inputs["document"]
    bank = [row["normalized_template_amr"] for row in load(base.BANK)]
    units = []
    with base.official_runtime(OUT, error_folder=OUT / "errors") as module:
        cached = base.CachedS2Matcher(base.GLOVE)
        equivalence = base.validate_equivalence(cached, bank, inputs)
        for index, source in enumerate(inputs["selected"]):
            units.append(base.score_unit(
                module,
                cached,
                bank,
                document["document_id"],
                index,
                source,
            ))
    green = sum(row["green"] for row in units)
    n = len(units)
    z = (green - base.NULL_RATE * n) / math.sqrt(
        n * base.NULL_RATE * (1 - base.NULL_RATE)
    )
    return freeze(result_path, {
        "completed_at": now(),
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "inputs_sha256": sha256(OUT / "inputs.json"),
        "equivalence_check_sha256": sha256(OUT / "equivalence_check.json"),
        "equivalence_pass": equivalence["pass"],
        "document_id": document["document_id"],
        "document_text_sha256": document["text_sha256"],
        "document_word_count": document["word_count"],
        "N": n,
        "green": green,
        "green_rate": green / n,
        "z": z,
        "detected_at_1pct_theoretical": z > 2.33,
        "detected_at_5pct_theoretical": z > 1.645,
        "independent_empirical_calibration_pending": True,
        "new_generation_calls": 0,
        "provider_calls": 0,
        "model_assisted_quality_gate_pass": True,
        "human_quality_reviewed": False,
        "source_anchored": False,
        "units": units,
    })


def status():
    result = {
        "experiment": "swan_normal_length_coherent_baseline1",
        "manifest_present": (OUT / "manifest.json").exists(),
        "equivalence_present": (OUT / "equivalence_check.json").exists(),
        "detection_present": (OUT / "detection.json").exists(),
    }
    if (OUT / "manifest.json").exists():
        result["manifest_sha256"] = sha256(OUT / "manifest.json")
    if (OUT / "equivalence_check.json").exists():
        result["equivalence_sha256"] = sha256(OUT / "equivalence_check.json")
        result["equivalence_pass"] = load(
            OUT / "equivalence_check.json"
        )["pass"]
    if (OUT / "detection.json").exists():
        detection = load(OUT / "detection.json")
        result["detection_sha256"] = sha256(OUT / "detection.json")
        result["detection"] = {
            key: detection[key]
            for key in (
                "document_word_count",
                "N",
                "green",
                "green_rate",
                "z",
                "detected_at_1pct_theoretical",
                "detected_at_5pct_theoretical",
                "independent_empirical_calibration_pending",
                "provider_calls",
            )
        }
    return result


def configure_base():
    base.OUT = OUT
    base.EQUIVALENCE_PAIRS = EQUIVALENCE_PAIRS
    base.EQUIVALENCE_TOLERANCE = EQUIVALENCE_TOLERANCE
    base.source_inputs = source_inputs
    base.dependencies = dependencies
    base.prepare = prepare
    base.check = check
    base.seed = seed
    base.detect = detect
    base.status = status


def main():
    configure_base()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "detect", "status"))
    args = parser.parse_args()
    result = {
        "prepare": prepare,
        "check": check,
        "detect": detect,
        "status": status,
    }[args.command]()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
