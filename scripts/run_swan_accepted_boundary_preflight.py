"""Construct and detect accepted-only SWAN boundary preflight conditions."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
import math
from pathlib import Path

from run_swan_accepted_baseline_preflight import (
    FailOnWarning,
    score_sentence,
)
from swan_feasibility2_common import ROOT, digest, freeze, load, now, sha256
from swan_final1_runtime import official_runtime
from swan_final1_transform import (
    FrozenSyntax,
    _candidates,
    apply_edits,
    punctuation,
    sentence_spans,
    text_hash,
    transform,
    unit_texts,
    word_tokens,
)

OUT = ROOT / "experiments/swan_accepted_boundary_preflight1"
BASE = ROOT / "experiments/swan_accepted_baseline_preflight1"
BANK = ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_50.json"
GREEN_THRESHOLD = 0.65
NULL_RATE = 0.25
GROUP = 10
UPSTREAM_FILES = (
    "manifest.json",
    "inputs.json",
    "detection.json",
    "protocol.md",
)


def record(text):
    spans = sentence_spans(text)
    return {
        "text": text,
        "document_sha256": text_hash(text),
        "word_tokens_sha256": digest(word_tokens(text)),
        "segmentation_spans": spans,
        "num_detection_units": len(spans),
        "unit_texts": unit_texts(text, spans),
    }


def merged_condition(text, merge):
    edits = [{
        "position": merge["position"],
        "before": ".",
        "after": ";",
        "kind": "merge",
    }]
    changed = apply_edits(text, edits)
    row = {**record(changed), "punctuation_edits": edits}
    original = record(text)
    if word_tokens(changed) != word_tokens(text) or len(changed) != len(text):
        raise ValueError("Merge changed words, order, or character length")
    if row["num_detection_units"] != original["num_detection_units"] - 1:
        raise ValueError("Merge did not remove exactly one detector boundary")
    expected = {end for _, end in original["segmentation_spans"]}
    expected.remove(merge["position"] + 1)
    if {end for _, end in row["segmentation_spans"]} != expected:
        raise ValueError("Merge changed an unexpected detector boundary")
    return row


def accepted_rows():
    inputs = load(BASE / "inputs.json")
    rows = inputs["selected"]
    if len(rows) != 20 or len({row["text_sha256"] for row in rows}) != 20:
        raise ValueError("Upstream accepted sentence set changed")
    return rows, inputs["documents"]


def standard_construction(syntax, documents):
    rows = []
    for document in documents:
        replay = transform(document["text"], syntax)
        c0 = replay["conditions"]["C0"]
        merges, _ = _candidates(
            document["text"], c0["segmentation_spans"], syntax
        )
        if not merges:
            raise ValueError("Expected at least one independent C1 merge")
        c1 = merged_condition(document["text"], merges[0])
        rows.append({
            "document_id": document["document_id"],
            "C0_reference": {
                "manifest_sha256": sha256(BASE / "manifest.json"),
                "detection_sha256": sha256(BASE / "detection.json"),
                "document_sha256": c0["document_sha256"],
                "N": c0["num_detection_units"],
            },
            "C1": c1,
            "standard_transform": replay,
        })
    return rows


def ordered_anchor(group, syntax):
    complete = [row for row in group if syntax(row["text"])["complete"]]
    incomplete = [row for row in group if not syntax(row["text"])["complete"]]
    key = lambda row: row["slot"]
    return sorted(complete, key=key) + sorted(incomplete, key=key), len(complete)


def relocation_construction(syntax, selected):
    groups = defaultdict(list)
    for row in selected:
        groups[row["allocation_id"]].append(row)
    anchors = []
    for allocation_id, group in sorted(groups.items()):
        ordered, complete_count = ordered_anchor(group, syntax)
        if complete_count >= 3:
            anchors.append((allocation_id, ordered, complete_count))
    if len(anchors) != 2:
        raise ValueError("Expected exactly two accepted anchor allocations")

    anchor_ids = {allocation_id for allocation_id, _, _ in anchors}
    fillers = sorted(
        (row for row in selected if row["allocation_id"] not in anchor_ids),
        key=lambda row: (row["allocation_id"], row["slot"]),
    )
    cursor = 0
    documents = []
    used = []
    for index, (allocation_id, anchor, complete_count) in enumerate(anchors):
        need = GROUP - len(anchor)
        if need < 0 or cursor + need > len(fillers):
            raise ValueError("Invalid deterministic filler allocation")
        ordered = anchor + fillers[cursor:cursor + need]
        cursor += need
        used.extend(row["text_sha256"] for row in ordered)
        text = " ".join(row["text"] for row in ordered)
        base = record(text)
        if base["num_detection_units"] != GROUP \
                or base["unit_texts"] != [row["text"] for row in ordered]:
            raise ValueError("Relocation composition changed source units")
        merges, _ = _candidates(text, base["segmentation_spans"], syntax)
        eligible = [
            merge for merge in merges
            if merge["unit_index"] < complete_count - 1
        ]
        if len(eligible) < 2:
            raise ValueError("Anchor lacks two fixed relocation boundaries")
        conditions = {}
        for condition, merge in zip(("R0", "R1", "R2"), eligible[:3]):
            conditions[condition] = merged_condition(text, merge)
        reference = conditions["R0"]
        for condition, row in conditions.items():
            if row["num_detection_units"] != GROUP - 1 \
                    or row["word_tokens_sha256"] != reference["word_tokens_sha256"] \
                    or len(row["text"]) != len(reference["text"]) \
                    or punctuation(row["text"]) != punctuation(reference["text"]):
                raise ValueError(f"{condition} violates count-neutral invariants")
        restored = apply_edits(
            reference["text"], reference["punctuation_edits"], reverse=True
        )
        if restored != text:
            raise ValueError("R4 exact restoration failed")
        conditions["R4"] = {
            **record(restored),
            "restored_from": "R0",
            "oracle_uses_saved_edit": True,
        }
        documents.append({
            "document_id": f"accepted-relocation-{index:02d}",
            "anchor_allocation_id": allocation_id,
            "anchor_complete_sentences": complete_count,
            "eligible_anchor_merges": eligible,
            "sentence_refs": [{
                "allocation_id": row["allocation_id"],
                "slot": row["slot"],
                "text_sha256": row["text_sha256"],
            } for row in ordered],
            "conditions": conditions,
        })
    if cursor != len(fillers) or sorted(used) != sorted(
            row["text_sha256"] for row in selected
    ):
        raise ValueError("Relocation documents do not partition accepted inputs")
    return documents


def build_inputs():
    selected, documents = accepted_rows()
    syntax = FrozenSyntax()
    return {
        "standard": standard_construction(syntax, documents),
        "relocation": relocation_construction(syntax, selected),
    }


def dependency_hashes():
    paths = [
        Path(__file__),
        OUT / "protocol.md",
        BANK,
        ROOT / "scripts/run_swan_accepted_baseline_preflight.py",
        ROOT / "scripts/swan_final1_runtime.py",
        ROOT / "scripts/swan_final1_transform.py",
        ROOT / "experiments/swan_count_neutral_final1/environment_lock.json",
    ]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def upstream_hashes():
    return {
        str((BASE / name).relative_to(ROOT)): sha256(BASE / name)
        for name in UPSTREAM_FILES
    }


def prepare():
    from run_swan_accepted_baseline_preflight import check as baseline_check

    baseline_check()
    if (OUT / "manifest.json").exists():
        return check()
    inputs = build_inputs()
    freeze(OUT / "inputs.json", inputs)
    standard_c2 = sum(
        "C2" in row["standard_transform"]["conditions"]
        for row in inputs["standard"]
    )
    manifest = freeze(OUT / "manifest.json", {
        "experiment": "swan_accepted_boundary_preflight1",
        "frozen_at": now(),
        "status": "frozen_before_attack_detection",
        "accepted_sentences": 20,
        "standard_documents": 2,
        "standard_C1_constructed": sum("C1" in row for row in inputs["standard"]),
        "standard_C2_constructed": standard_c2,
        "relocation_documents": len(inputs["relocation"]),
        "relocation_primary": ["R0", "R1"],
        "green_threshold": GREEN_THRESHOLD,
        "null_rate": NULL_RATE,
        "new_generation_calls": 0,
        "sealed_source_access": False,
        "inputs_sha256": sha256(OUT / "inputs.json"),
        "upstream": upstream_hashes(),
        "dependencies": dependency_hashes(),
    })
    return {
        "state": manifest["status"],
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "standard_C2_constructed": standard_c2,
    }


def check():
    from run_swan_accepted_baseline_preflight import check as baseline_check

    baseline_check()
    manifest = load(OUT / "manifest.json")
    if manifest["experiment"] != "swan_accepted_boundary_preflight1" \
            or manifest["new_generation_calls"] != 0 \
            or manifest["sealed_source_access"] \
            or manifest["standard_C1_constructed"] != 2 \
            or manifest["standard_C2_constructed"] != 0 \
            or manifest["inputs_sha256"] != sha256(OUT / "inputs.json") \
            or manifest["green_threshold"] != GREEN_THRESHOLD \
            or manifest["null_rate"] != NULL_RATE:
        raise ValueError("Boundary preflight contract changed")
    if manifest["upstream"] != upstream_hashes():
        raise ValueError("Upstream baseline evidence changed")
    for relative, expected in manifest["dependencies"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Dependency changed: {relative}")
    if build_inputs() != load(OUT / "inputs.json"):
        raise ValueError("Detector-blind construction replay changed")
    return {
        "state": "frozen_hashes_and_construction_verified",
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "standard_C2_constructed": manifest["standard_C2_constructed"],
    }


def baseline_evidence():
    detection = load(BASE / "detection.json")
    manifest_hash = sha256(BASE / "manifest.json")
    if detection["manifest_sha256"] != manifest_hash:
        raise ValueError("Upstream detection does not bind current manifest")
    evidence = {}
    for document in detection["documents"]:
        for index, unit in enumerate(document["units"]):
            key = unit["text_sha256"]
            if key in evidence:
                raise ValueError("Upstream unit text is not unique")
            evidence[key] = {
                **unit,
                "evidence_origin": "upstream_frozen_detection",
                "upstream_document_id": document["document_id"],
                "upstream_unit_index": index,
                "upstream_detection_sha256": sha256(BASE / "detection.json"),
            }
    if len(evidence) != 20:
        raise ValueError("Expected 20 upstream unit records")
    return evidence


def document_score(document_id, condition, row, get_evidence):
    units = []
    for index, text in enumerate(row["unit_texts"]):
        key, evidence = get_evidence(text)
        units.append({
            "index": index,
            "text": text,
            "unit_sha256": key,
            "max_score": evidence["max_score"],
            "max_bank_indices": evidence["max_bank_indices"],
            "green": evidence["green"],
            "evidence_origin": evidence["evidence_origin"],
        })
    green = sum(unit["green"] for unit in units)
    n = len(units)
    z = (green - NULL_RATE * n) / math.sqrt(n * NULL_RATE * (1 - NULL_RATE))
    return {
        "document_id": document_id,
        "condition": condition,
        "N": n,
        "green": green,
        "green_rate": green / n,
        "z": z,
        "detected_at_1pct_theoretical": z > 2.33,
        "detected_at_5pct_theoretical": z > 1.645,
        "units": units,
    }


def summarize(rows):
    return {
        "documents": len(rows),
        "TPR_at_1pct_theoretical": sum(
            row["detected_at_1pct_theoretical"] for row in rows
        ) / len(rows),
        "TPR_at_5pct_theoretical": sum(
            row["detected_at_5pct_theoretical"] for row in rows
        ) / len(rows),
        "mean_green_rate": sum(row["green_rate"] for row in rows) / len(rows),
        "mean_z": sum(row["z"] for row in rows) / len(rows),
    }


def detect():
    check()
    if (OUT / "detection.json").exists():
        return load(OUT / "detection.json")["summary"]
    inputs = load(OUT / "inputs.json")
    bank = [row["normalized_template_amr"] for row in load(BANK)]
    evidence = baseline_evidence()
    warning_handler = FailOnWarning(level=logging.WARNING)
    parser_logger = logging.getLogger("amrlib.models.parse_xfm.penman_serializer")
    parser_logger.addHandler(warning_handler)
    try:
        with official_runtime(OUT, error_folder=OUT / "errors") as module:
            def get_evidence(text):
                key = digest(text)
                if key not in evidence:
                    scored = score_sentence(
                        module, bank, f"merged-{key}", 0, text
                    )
                    if scored["text_sha256"] != key:
                        raise ValueError("Merged unit hash mismatch")
                    evidence[key] = {
                        **scored,
                        "evidence_origin": "new_merged_unit_detection",
                    }
                return key, evidence[key]

            standard_c1 = [
                document_score(
                    row["document_id"], "C1", row["C1"], get_evidence
                )
                for row in inputs["standard"]
            ]
            relocation = []
            for row in inputs["relocation"]:
                conditions = {}
                for condition in ("R0", "R1", "R2"):
                    if condition in row["conditions"]:
                        conditions[condition] = document_score(
                            row["document_id"],
                            condition,
                            row["conditions"][condition],
                            get_evidence,
                        )
                relocation.append({
                    "document_id": row["document_id"],
                    "conditions": conditions,
                })
    finally:
        parser_logger.removeHandler(warning_handler)

    upstream = load(BASE / "detection.json")
    c0_rows = [{
        key: row[key] for key in (
            "document_id",
            "N",
            "green",
            "z",
            "detected_at_1pct_theoretical",
            "detected_at_5pct_theoretical",
        )
    } for row in upstream["documents"]]
    c0_summary = {
        "documents": len(c0_rows),
        "TPR_at_1pct_theoretical": upstream["TPR_at_1pct_theoretical"],
        "TPR_at_5pct_theoretical": upstream["TPR_at_5pct_theoretical"],
        "mean_green_rate": sum(row["green"] / row["N"] for row in c0_rows)
        / len(c0_rows),
        "mean_z": sum(row["z"] for row in c0_rows) / len(c0_rows),
    }
    relocation_by_condition = {}
    for condition in ("R0", "R1", "R2"):
        rows = [
            row["conditions"][condition]
            for row in relocation
            if condition in row["conditions"]
        ]
        if rows:
            relocation_by_condition[condition] = summarize(rows)
    paired = []
    for row in relocation:
        r0, r1 = row["conditions"]["R0"], row["conditions"]["R1"]
        if r0["N"] != r1["N"]:
            raise ValueError("R0/R1 detector-unit counts differ")
        paired.append({
            "document_id": row["document_id"],
            "N": r0["N"],
            "R0_minus_R1_green": r0["green"] - r1["green"],
            "R0_minus_R1_green_rate": r0["green_rate"] - r1["green_rate"],
            "R0_minus_R1_z": r0["z"] - r1["z"],
            "R0_detected_R1_missed_at_1pct": (
                r0["detected_at_1pct_theoretical"]
                and not r1["detected_at_1pct_theoretical"]
            ),
            "R0_missed_R1_detected_at_1pct": (
                not r0["detected_at_1pct_theoretical"]
                and r1["detected_at_1pct_theoretical"]
            ),
        })
    result = {
        "completed_at": now(),
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "inputs_sha256": sha256(OUT / "inputs.json"),
        "upstream_detection_sha256": sha256(BASE / "detection.json"),
        "standard": {
            "C0": {
                "evidence_origin": "upstream_frozen_detection",
                "documents": c0_rows,
                "summary": c0_summary,
            },
            "C1": {
                "documents": standard_c1,
                "summary": summarize(standard_c1),
            },
            "C2": {
                "constructed": False,
                "constructed_documents": 0,
                "eligible_documents": 2,
                "reason": "no_eligible_swap_no_natural_split",
                "detection_not_run": True,
            },
        },
        "relocation": {
            "documents": relocation,
            "summary_by_condition": relocation_by_condition,
            "paired_R0_minus_R1": paired,
            "mean_R0_minus_R1_green_rate": sum(
                row["R0_minus_R1_green_rate"] for row in paired
            ) / len(paired),
            "mean_R0_minus_R1_z": sum(
                row["R0_minus_R1_z"] for row in paired
            ) / len(paired),
            "interpretation_limit": (
                "Two-document mechanism preflight; not a calibrated "
                "population-level or standard C2 estimate."
            ),
        },
        "unit_evidence": evidence,
        "new_generation_calls": 0,
        "independent_empirical_calibration_pending": True,
    }
    summary = {
        "standard_C0": result["standard"]["C0"]["summary"],
        "standard_C1": result["standard"]["C1"]["summary"],
        "standard_C2": result["standard"]["C2"],
        "relocation": result["relocation"]["summary_by_condition"],
        "paired_R0_minus_R1": paired,
    }
    result["summary"] = summary
    freeze(OUT / "detection.json", result)
    return summary


def status():
    manifest = OUT / "manifest.json"
    detection = OUT / "detection.json"
    return {
        "prepared": manifest.exists(),
        "detected": detection.exists(),
        "manifest_sha256": sha256(manifest) if manifest.exists() else None,
        "detection_sha256": sha256(detection) if detection.exists() else None,
        "summary": load(detection)["summary"] if detection.exists() else None,
    }


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
