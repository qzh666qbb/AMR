"""Compose and locally detect the injection-only SWAN-positive document."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import run_swan_injection_only_pilot as injection
from swan_feasibility2_common import ROOT, digest, freeze, load, now, sha256
from swan_final1_contract import environment_lock
from swan_final1_runtime import official_runtime, score_triple
from swan_final1_transform import sentence_spans, text_hash, unit_texts

OUT = ROOT / "experiments/swan_injection_only_baseline1"
SOURCE = ROOT / "experiments/swan_injection_only_pilot1"
BANK = ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_50.json"
EXPECTED_BANK = "f0a88c1af73d12078586d10ef68804c12ed5b79cabc07c9f6c2a34bb084ef5a4"
EXPECTED_SOURCE_MANIFEST = (
    "a41efaae1b26de0f4ab60e2d76d50aae7ac59a85d3db3669edf221afe2136ce2"
)
EXPECTED_SOURCE_RESULT = (
    "cb481c602074756928abcd7e84ad2bdb815b95852e18b354096a14cd56563f2a"
)
COUNT = 10
GREEN_THRESHOLD = 0.65
NULL_RATE = 0.25


def seed(*parts):
    return int(digest(["swan_injection_only_baseline1", *parts])[:8], 16)


def diagnostics_empty(row):
    diagnostics = row.get("candidate_diagnostics") or {}
    return (
        not diagnostics.get("logged_warnings")
        and not diagnostics.get("python_warnings")
        and not diagnostics.get("stderr")
        and row.get("parser_or_scorer_error_type") is None
    )


def source_inputs():
    result_path = SOURCE / "result.json"
    manifest_path = SOURCE / "manifest.json"
    if sha256(manifest_path) != EXPECTED_SOURCE_MANIFEST \
            or sha256(result_path) != EXPECTED_SOURCE_RESULT:
        raise ValueError("Frozen injection-only source changed")
    result = load(result_path)
    if result["status"] != "target_reached" \
            or result["accepted_slots"] != COUNT \
            or result["fallback_slots"] != 0 \
            or len(result["selected"]) != COUNT:
        raise ValueError("Source did not reach the strict injection target")
    selected = []
    seen = set()
    for source in result["selected"]:
        slot = source["slot"]
        slot_path = injection.slot_path(slot)
        if sha256(slot_path) != source["slot_sha256"] \
                or sha256(slot_path) != result["slot_sha256"][str(slot)]:
            raise ValueError("Source slot hash changed")
        saved = load(slot_path)
        trace_path = injection.attempt_path(slot, saved["terminal_attempt"])
        trace = load(trace_path)
        if saved["attempt_trace_sha256"][-1] != sha256(trace_path):
            raise ValueError("Terminal source trace hash changed")
        valid = (
            saved["accepted"] is True
            and saved["fallback"] is False
            and saved["text"] == source["text"] == trace["candidate"]
            and saved["text_sha256"] == source["text_sha256"]
            == trace["candidate_sha256"] == text_hash(source["text"])
            and saved["score"] == source["score"] == trace["score"]
            and math.isfinite(saved["score"])
            and saved["score"] >= 0.75
            and trace["selected"] is True
            and trace["status"] == "accepted"
            and trace["stop_reason"] == "stop"
            and trace["surface"]["pass"] is True
            and trace["score_computed"] is True
            and trace["official_score_accepted"] is True
            and diagnostics_empty(trace)
        )
        if not valid or saved["text_sha256"] in seen:
            raise ValueError("Source slot is not a unique strict acceptance")
        seen.add(saved["text_sha256"])
        selected.append({
            "slot": slot,
            "text": saved["text"],
            "text_sha256": saved["text_sha256"],
            "target_template_index": saved["target_template_index"],
            "target_score": saved["score"],
            "slot_path": str(slot_path.relative_to(ROOT)),
            "slot_sha256": sha256(slot_path),
            "trace_path": str(trace_path.relative_to(ROOT)),
            "trace_sha256": sha256(trace_path),
        })
    if [row["slot"] for row in selected] != sorted(row["slot"] for row in selected):
        raise ValueError("Source selection is not in slot order")
    text = " ".join(row["text"] for row in selected)
    spans = sentence_spans(text)
    if len(spans) != COUNT or unit_texts(text, spans) != [
        row["text"] for row in selected
    ]:
        raise ValueError("Concatenation does not preserve ten exact detector units")
    return {
        "source_manifest_path": str(manifest_path.relative_to(ROOT)),
        "source_manifest_sha256": sha256(manifest_path),
        "source_result_path": str(result_path.relative_to(ROOT)),
        "source_result_sha256": sha256(result_path),
        "selected": selected,
        "document": {
            "document_id": "injection-only-baseline-00",
            "text": text,
            "text_sha256": text_hash(text),
            "segmentation_spans": spans,
            "sentence_text_sha256": [row["text_sha256"] for row in selected],
        },
    }


def dependencies():
    paths = [
        Path(__file__),
        OUT / "protocol.md",
        BANK,
        ROOT / "scripts/run_swan_injection_only_pilot.py",
        ROOT / "scripts/run_swan_positive_collection.py",
        ROOT / "scripts/swan_feasibility2_common.py",
        ROOT / "scripts/swan_final1_contract.py",
        ROOT / "scripts/swan_final1_runtime.py",
        ROOT / "scripts/swan_final1_transform.py",
        ROOT / "scripts/test_swan_injection_only_baseline.py",
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
        "experiment": "swan_injection_only_baseline1",
        "frozen_at": now(),
        "status": "frozen_before_detection",
        "selected_sentences": COUNT,
        "documents": 1,
        "sentences_per_document": COUNT,
        "composition": "slot_order_ascii_space_no_edits",
        "green_threshold": GREEN_THRESHOLD,
        "null_rate": NULL_RATE,
        "bank_sha256": EXPECTED_BANK,
        "inputs_sha256": sha256(OUT / "inputs.json"),
        "environment_sha256": sha256(OUT / "environment_lock.json"),
        "dependencies": dependencies(),
        "new_generation_calls": 0,
        "provider_calls": 0,
        "coherence_claim": False,
        "empirical_calibration_claim": False,
        "sealed_source_access": False,
    })
    return {
        "state": manifest["status"],
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "inputs_sha256": sha256(OUT / "inputs.json"),
    }


def validate_inputs(inputs):
    if inputs != source_inputs():
        raise ValueError("Frozen accepted-only inputs changed")
    document = inputs["document"]
    spans = sentence_spans(document["text"])
    if spans != document["segmentation_spans"] \
            or unit_texts(document["text"], spans) != [
                row["text"] for row in inputs["selected"]
            ]:
        raise ValueError("Frozen document segmentation changed")


def check(full_environment=True):
    manifest = load(OUT / "manifest.json")
    valid = (
        manifest["experiment"] == "swan_injection_only_baseline1"
        and manifest["selected_sentences"] == COUNT
        and manifest["documents"] == 1
        and manifest["sentences_per_document"] == COUNT
        and manifest["composition"] == "slot_order_ascii_space_no_edits"
        and manifest["green_threshold"] == GREEN_THRESHOLD
        and manifest["null_rate"] == NULL_RATE
        and manifest["new_generation_calls"] == 0
        and manifest["provider_calls"] == 0
        and manifest["coherence_claim"] is False
        and manifest["empirical_calibration_claim"] is False
        and manifest["sealed_source_access"] is False
    )
    if not valid:
        raise ValueError("Baseline identity or claim boundary changed")
    if manifest["bank_sha256"] != sha256(BANK) \
            or manifest["inputs_sha256"] != sha256(OUT / "inputs.json") \
            or manifest["environment_sha256"] != sha256(OUT / "environment_lock.json"):
        raise ValueError("Frozen baseline artifact changed")
    for relative, expected in manifest["dependencies"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Dependency changed: {relative}")
    if full_environment and environment_lock() != load(OUT / "environment_lock.json"):
        raise ValueError("Execution environment changed")
    validate_inputs(load(OUT / "inputs.json"))
    return {
        "state": "frozen_hashes_verified",
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "environment_verified": full_environment,
    }


def score_unit(module, bank, document_id, index, source):
    diagnostics = None
    try:
        with injection.capture_candidate_diagnostics() as diagnostics:
            raw = module.parse_amr(source["text"])
            normalized = module.normalize_amr_variables(raw)
            templated = module.generate_template_amr(normalized)
            candidate = module.normalize_amr_variables(templated)
            scores = []
            for bank_index, graph in enumerate(bank):
                precision, recall, f1 = [
                    float(value)
                    for value in score_triple(
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
    if (OUT / "detection.json").exists():
        return load(OUT / "detection.json")
    inputs = load(OUT / "inputs.json")
    document = inputs["document"]
    bank = [row["normalized_template_amr"] for row in load(BANK)]
    units = []
    with official_runtime(OUT, error_folder=OUT / "errors") as module:
        for index, source in enumerate(inputs["selected"]):
            units.append(score_unit(
                module,
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
        "units": units,
    }
    freeze(OUT / "detection.json", result)
    return result


def status():
    result = {
        "experiment": "swan_injection_only_baseline1",
        "manifest_present": (OUT / "manifest.json").exists(),
        "detection_present": (OUT / "detection.json").exists(),
    }
    if (OUT / "manifest.json").exists():
        result["manifest_sha256"] = sha256(OUT / "manifest.json")
    if (OUT / "detection.json").exists():
        detection = load(OUT / "detection.json")
        result["detection_sha256"] = sha256(OUT / "detection.json")
        result["summary"] = {
            key: detection[key]
            for key in (
                "N",
                "green",
                "green_rate",
                "z",
                "detected_at_1pct_theoretical",
                "detected_at_5pct_theoretical",
                "independent_empirical_calibration_pending",
                "new_generation_calls",
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
