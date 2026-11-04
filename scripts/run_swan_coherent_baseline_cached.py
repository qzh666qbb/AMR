"""Detect the coherent SWAN baseline with an equivalence-checked vector cache."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import run_swan_injection_only_pilot as injection
from swan_cached_s2match import CachedS2Matcher
from swan_feasibility2_common import ROOT, digest, freeze, load, now, sha256
from swan_final1_contract import environment_lock
from swan_final1_runtime import official_runtime, score_triple
from swan_final1_transform import sentence_spans, text_hash, unit_texts

OUT = ROOT / "experiments/swan_coherent_baseline_cached1"
SOURCE = ROOT / "experiments/swan_coherent_baseline1"
BANK = ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_50.json"
GLOVE = ROOT / "vectors/glove.6B.100d.txt"
EXPECTED_BANK = "f0a88c1af73d12078586d10ef68804c12ed5b79cabc07c9f6c2a34bb084ef5a4"
EXPECTED_SOURCE_MANIFEST = (
    "b3eaa46a0c1c32e21d960b4bd1cc400cd708dc264f47c23b1fcdb4e61e7820b8"
)
EXPECTED_SOURCE_INPUTS = (
    "fd0b17762106a604c0625be1ee8eeb6e37929432b7aa34bd08718659b0bb6fbe"
)
EXPECTED_SOURCE_QUALITY = (
    "2a7a7b0bc31a8d875c6e7ae2cc44b2ccfd17d135310d3d3c0a0b9291ba082a99"
)
GREEN_THRESHOLD = 0.65
NULL_RATE = 0.25
EQUIVALENCE_TOLERANCE = 1e-12
EQUIVALENCE_PAIRS = ((0, 0), (4, 25), (9, 47))


def source_inputs():
    paths = {
        "manifest": SOURCE / "manifest.json",
        "inputs": SOURCE / "inputs.json",
        "quality": SOURCE / "quality_result.json",
    }
    expected = {
        "manifest": EXPECTED_SOURCE_MANIFEST,
        "inputs": EXPECTED_SOURCE_INPUTS,
        "quality": EXPECTED_SOURCE_QUALITY,
    }
    for key, path in paths.items():
        if sha256(path) != expected[key]:
            raise ValueError(f"Frozen coherent baseline {key} changed")
    inputs = load(paths["inputs"])
    quality = load(paths["quality"])
    if not quality["qualified_for_detection"] \
            or quality["review_passes"] < quality["review_passes_required"] \
            or quality["manifest_sha256"] != EXPECTED_SOURCE_MANIFEST \
            or quality["inputs_sha256"] != EXPECTED_SOURCE_INPUTS:
        raise ValueError("Source quality gate is not valid")
    text = inputs["document"]["text"]
    spans = sentence_spans(text)
    if spans != inputs["document"]["segmentation_spans"] \
            or unit_texts(text, spans) != [
                row["text"] for row in inputs["selected"]
            ]:
        raise ValueError("Source document segmentation changed")
    return {
        "source_manifest_path": str(paths["manifest"].relative_to(ROOT)),
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST,
        "source_inputs_path": str(paths["inputs"].relative_to(ROOT)),
        "source_inputs_sha256": EXPECTED_SOURCE_INPUTS,
        "source_quality_path": str(paths["quality"].relative_to(ROOT)),
        "source_quality_sha256": EXPECTED_SOURCE_QUALITY,
        "quality_summary": {
            "qualified_for_detection": True,
            "review_passes": quality["review_passes"],
            "review_passes_required": quality["review_passes_required"],
            "human_quality_reviewed": quality["human_quality_reviewed"],
            "posthoc_genre_adjudication": quality["posthoc_genre_adjudication"],
        },
        "selected": inputs["selected"],
        "document": inputs["document"],
    }


def dependencies():
    paths = [
        Path(__file__),
        OUT / "protocol.md",
        BANK,
        GLOVE,
        SOURCE / "manifest.json",
        SOURCE / "inputs.json",
        SOURCE / "quality_result.json",
        ROOT / "scripts/swan_cached_s2match.py",
        ROOT / "scripts/run_swan_coherent_baseline.py",
        ROOT / "scripts/run_swan_injection_only_pilot.py",
        ROOT / "scripts/run_swan_positive_collection.py",
        ROOT / "scripts/swan_feasibility2_common.py",
        ROOT / "scripts/swan_final1_contract.py",
        ROOT / "scripts/swan_final1_runtime.py",
        ROOT / "scripts/swan_final1_transform.py",
        ROOT / "scripts/test_swan_coherent_baseline_cached.py",
    ]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def prepare():
    if (OUT / "manifest.json").exists():
        return check()
    if sha256(BANK) != EXPECTED_BANK:
        raise ValueError("Bank50 changed")
    freeze(OUT / "inputs.json", source_inputs())
    freeze(OUT / "environment_lock.json", environment_lock())
    manifest = freeze(OUT / "manifest.json", {
        "schema_version": 1,
        "experiment": "swan_coherent_baseline_cached1",
        "frozen_at": now(),
        "status": "frozen_before_detection",
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST,
        "source_inputs_sha256": EXPECTED_SOURCE_INPUTS,
        "source_quality_sha256": EXPECTED_SOURCE_QUALITY,
        "inputs_sha256": sha256(OUT / "inputs.json"),
        "environment_sha256": sha256(OUT / "environment_lock.json"),
        "bank_sha256": EXPECTED_BANK,
        "glove_sha256": sha256(GLOVE),
        "dependencies": dependencies(),
        "equivalence_pairs": [list(pair) for pair in EQUIVALENCE_PAIRS],
        "equivalence_tolerance": EQUIVALENCE_TOLERANCE,
        "green_threshold": GREEN_THRESHOLD,
        "null_rate": NULL_RATE,
        "new_generation_calls": 0,
        "provider_calls": 0,
        "human_quality_claim": False,
        "normal_length_claim": False,
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
        manifest["experiment"] == "swan_coherent_baseline_cached1"
        and manifest["source_manifest_sha256"] == EXPECTED_SOURCE_MANIFEST
        and manifest["source_inputs_sha256"] == EXPECTED_SOURCE_INPUTS
        and manifest["source_quality_sha256"] == EXPECTED_SOURCE_QUALITY
        and manifest["equivalence_pairs"] == [
            list(pair) for pair in EQUIVALENCE_PAIRS
        ]
        and manifest["equivalence_tolerance"] == EQUIVALENCE_TOLERANCE
        and manifest["green_threshold"] == GREEN_THRESHOLD
        and manifest["null_rate"] == NULL_RATE
        and manifest["new_generation_calls"] == 0
        and manifest["provider_calls"] == 0
        and not manifest["human_quality_claim"]
        and not manifest["normal_length_claim"]
        and not manifest["empirical_calibration_claim"]
        and not manifest["attack_authorized"]
        and not manifest["sealed_source_access"]
    )
    if not valid:
        raise ValueError("Cached baseline identity or claim boundary changed")
    if sha256(BANK) != EXPECTED_BANK \
            or sha256(GLOVE) != manifest["glove_sha256"] \
            or manifest["inputs_sha256"] != sha256(OUT / "inputs.json") \
            or manifest["environment_sha256"] != sha256(OUT / "environment_lock.json"):
        raise ValueError("Frozen cached baseline artifact changed")
    if load(OUT / "inputs.json") != source_inputs():
        raise ValueError("Frozen cached baseline inputs changed")
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
    return int(digest(["swan_coherent_baseline_cached1", *parts])[:8], 16)


def candidate_graph(module, source):
    raw = module.parse_amr(source["text"])
    normalized = module.normalize_amr_variables(raw)
    templated = module.generate_template_amr(normalized)
    candidate = module.normalize_amr_variables(templated)
    return raw, candidate


def validate_equivalence(cached, bank, inputs):
    rows = []
    with injection.capture_candidate_diagnostics() as diagnostics:
        for unit_index, bank_index in EQUIVALENCE_PAIRS:
            trace = load(ROOT / inputs["selected"][unit_index]["trace_path"])
            candidate = trace["candidate_template_amr"]
            score_seed = seed("equivalence", unit_index, bank_index)
            official = tuple(float(value) for value in score_triple(
                candidate,
                bank[bank_index],
                score_seed,
            ))
            cached_value = tuple(float(value) for value in cached.score(
                candidate,
                bank[bank_index],
                score_seed,
            ))
            differences = [
                abs(left - right)
                for left, right in zip(official, cached_value)
            ]
            rows.append({
                "unit_index": unit_index,
                "bank_index": bank_index,
                "score_seed": score_seed,
                "candidate_template_sha256": text_hash(candidate),
                "bank_template_sha256": text_hash(bank[bank_index]),
                "official": {
                    "precision": official[0],
                    "recall": official[1],
                    "f1": official[2],
                },
                "cached": {
                    "precision": cached_value[0],
                    "recall": cached_value[1],
                    "f1": cached_value[2],
                },
                "absolute_differences": differences,
                "pass": all(value <= EQUIVALENCE_TOLERANCE for value in differences),
            })
    diagnostics_present = bool(
        diagnostics["logged_warnings"]
        or diagnostics["python_warnings"]
        or diagnostics["stderr"]
    )
    result = {
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "inputs_sha256": sha256(OUT / "inputs.json"),
        "tolerance": EQUIVALENCE_TOLERANCE,
        "diagnostics": diagnostics,
        "diagnostics_present": diagnostics_present,
        "pairs": rows,
        "pass": not diagnostics_present and all(row["pass"] for row in rows),
    }
    freeze(OUT / "equivalence_check.json", result)
    if not result["pass"]:
        raise RuntimeError("Cached S2Match equivalence check failed")
    return result


def score_unit(module, cached, bank, document_id, index, source):
    diagnostics = None
    try:
        with injection.capture_candidate_diagnostics() as diagnostics:
            raw, candidate = candidate_graph(module, source)
            scores = []
            for bank_index, graph in enumerate(bank):
                precision, recall, f1 = [
                    float(value)
                    for value in cached.score(
                        candidate,
                        graph,
                        seed("detect", document_id, index, bank_index),
                    )
                ]
                if any(
                    not math.isfinite(value) or not 0 <= value <= 1
                    for value in (precision, recall, f1)
                ):
                    raise ValueError("Invalid S2Match score")
                scores.append({
                    "bank_index": bank_index,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                })
    except Exception as exc:
        failure = {
            "document_id": document_id,
            "unit_index": index,
            "text_sha256": source["text_sha256"],
            "diagnostics": diagnostics,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        freeze(OUT / f"detection_failure_unit_{index:02d}.json", failure)
        raise
    if diagnostics["logged_warnings"] \
            or diagnostics["python_warnings"] \
            or diagnostics["stderr"]:
        failure = {
            "document_id": document_id,
            "unit_index": index,
            "text_sha256": source["text_sha256"],
            "diagnostics": diagnostics,
            "error_type": "DetectionDiagnostic",
            "error_message": "Detection emitted a prohibited diagnostic",
        }
        freeze(OUT / f"detection_failure_unit_{index:02d}.json", failure)
        raise RuntimeError(f"Detection diagnostic in unit {index}")
    maximum = max(row["f1"] for row in scores)
    return {
        "unit_index": index,
        "text": source["text"],
        "text_sha256": source["text_sha256"],
        "source_slot": source["slot"],
        "source_target_template_index": source["target_template_index"],
        "source_target_score": source["target_score"],
        "raw_amr": raw,
        "raw_amr_sha256": text_hash(raw),
        "candidate_template_amr": candidate,
        "candidate_template_amr_sha256": text_hash(candidate),
        "diagnostics": diagnostics,
        "scores": scores,
        "max_score": maximum,
        "max_bank_indices": [
            row["bank_index"] for row in scores if row["f1"] == maximum
        ],
        "green": maximum >= GREEN_THRESHOLD,
    }


def detect():
    check()
    result_path = OUT / "detection.json"
    if result_path.exists():
        return load(result_path)
    inputs = load(OUT / "inputs.json")
    document = inputs["document"]
    bank = [row["normalized_template_amr"] for row in load(BANK)]
    units = []
    with official_runtime(OUT, error_folder=OUT / "errors") as module:
        cached = CachedS2Matcher(GLOVE)
        equivalence = validate_equivalence(cached, bank, inputs)
        for index, source in enumerate(inputs["selected"]):
            units.append(score_unit(
                module,
                cached,
                bank,
                document["document_id"],
                index,
                source,
            ))
    green = sum(row["green"] for row in units)
    n = len(units)
    z = (green - NULL_RATE * n) / math.sqrt(
        n * NULL_RATE * (1 - NULL_RATE)
    )
    result = {
        "completed_at": now(),
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "inputs_sha256": sha256(OUT / "inputs.json"),
        "equivalence_check_sha256": sha256(OUT / "equivalence_check.json"),
        "equivalence_pass": equivalence["pass"],
        "document_id": document["document_id"],
        "document_text_sha256": document["text_sha256"],
        "N": n,
        "green": green,
        "green_rate": green / n,
        "z": z,
        "detected_at_1pct_theoretical": z > 2.33,
        "detected_at_5pct_theoretical": z > 1.645,
        "independent_empirical_calibration_pending": True,
        "new_generation_calls": 0,
        "provider_calls": 0,
        "human_quality_reviewed": False,
        "posthoc_genre_adjudication": True,
        "units": units,
    }
    return freeze(result_path, result)


def status():
    result = {
        "experiment": "swan_coherent_baseline_cached1",
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


def main():
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
