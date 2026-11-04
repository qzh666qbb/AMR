"""Blind quality and transformation gates for source-anchored documents."""
from __future__ import annotations

from pathlib import Path

from swan_anchor1_common import digest, freeze, load, sha256
from swan_anchor1_generation import verify_generation
from swan_anchor1_sources import COHORTS
from swan_final1_quality import export_packets, finalize, verify_decision
from swan_final1_transform import (
    FrozenSyntax,
    apply_edits,
    punctuation,
    restore_original,
    sentence_spans,
    text_hash,
    transform,
    word_tokens,
)
from swan_final1_workflow import verify_inputs


def sources_for(folder, phase):
    if phase not in dict(COHORTS):
        raise ValueError("Unknown phase")
    if phase != "pilot":
        require_pilot(folder)
    return load(Path(folder) / "sealed" / f"{phase}_sources.json")


def documents_for(folder, phase):
    if phase not in {"pilot", "main"}:
        raise ValueError("Only positive generation phases have documents")
    return verify_generation(folder, phase, sources_for(folder, phase))


def quality_folder(folder, phase, kind):
    return Path(folder) / "quality" / phase / kind


def baseline_records(documents):
    return [
        {
            "allocation_id": row["allocation_id"],
            "source_excerpt": row["source_excerpt"],
            "text": row["text"],
        }
        for row in documents
        if row["structural_pass"]
    ]


def decision_for(folder, phase, kind, records):
    path = quality_folder(folder, phase, kind)
    verify_inputs(path, records)
    return verify_decision(path, sha256(Path(folder) / "manifest.json"))


def baseline_cohort(folder, phase, documents=None):
    documents = documents if documents is not None else documents_for(folder, phase)
    records = baseline_records(documents)
    decision = decision_for(folder, phase, "baseline", records)
    passed = {
        row["allocation_id"]
        for row in decision["decisions"]
        if row["quality_pass"]
    }
    return [
        row
        for row in documents
        if row["structural_pass"] and row["allocation_id"] in passed
    ]


def transform_path(folder, phase):
    return Path(folder) / "transforms" / f"{phase}.json"


def create_transforms(folder, phase):
    cohort = baseline_cohort(folder, phase)
    path = transform_path(folder, phase)
    if path.exists():
        return verified_transforms(folder, phase, cohort)
    syntax = FrozenSyntax()
    rows = []
    for document in cohort:
        evidence = {}

        def recorded(text):
            evidence[text] = syntax(text)
            return evidence[text]

        rows.append({
            "allocation_id": document["allocation_id"],
            "baseline_sha256": digest(document),
            **transform(document["text"], recorded),
            "syntax_evidence": evidence,
        })
    freeze(path, rows)
    return rows


def verified_transforms(folder, phase, cohort):
    rows = load(transform_path(folder, phase))
    if len(rows) != len(cohort):
        raise ValueError("Transformation cohort differs")
    for row, document in zip(rows, cohort):
        evidence = row["syntax_evidence"]
        expected = {
            "allocation_id": document["allocation_id"],
            "baseline_sha256": digest(document),
            **transform(document["text"], lambda text: evidence[text]),
            "syntax_evidence": evidence,
        }
        if row != expected:
            raise ValueError("Transformation cannot be reproduced")
        if row["input_sha256"] != text_hash(document["text"]) or row["detector_feedback"]:
            raise ValueError("Transformation input or blindness changed")
        c0 = row["conditions"]["C0"]
        for condition in ("C1", "C2", "C3"):
            if condition not in row["conditions"]:
                if condition not in row["failures"]:
                    raise ValueError("Transformation attrition is unaccounted")
                continue
            changed = row["conditions"][condition]
            if apply_edits(document["text"], changed["punctuation_edits"]) != changed["text"] \
                    or word_tokens(changed["text"]) != word_tokens(document["text"]) \
                    or sentence_spans(changed["text"]) != changed["segmentation_spans"]:
                raise ValueError("Boundary or token invariant changed")
        if "C2" in row["conditions"] and "C3" in row["conditions"] \
                and punctuation(row["conditions"]["C2"]["text"]) != \
                punctuation(row["conditions"]["C3"]["text"]):
            raise ValueError("C2/C3 punctuation control changed")
        if "C4" in row["conditions"]:
            restored = restore_original(
                row["conditions"]["C2"],
                c0["segmentation_spans"],
                c0["document_sha256"],
            )
            if restored != row["conditions"]["C4"]:
                raise ValueError("C4 restoration changed")
    return rows


def pair_records(transforms):
    return [
        {
            "allocation_id": row["allocation_id"],
            "condition_id": condition,
            "original_text": row["conditions"]["C0"]["text"],
            "transformed_text": row["conditions"][condition]["text"],
        }
        for row in transforms
        for condition in ("C1", "C2", "C3")
        if condition in row["conditions"]
    ]


def quality_action(folder, phase, kind, finalize_labels=False):
    documents = documents_for(folder, phase)
    if kind == "baseline":
        records = baseline_records(documents)
    else:
        cohort = baseline_cohort(folder, phase, documents)
        records = pair_records(verified_transforms(folder, phase, cohort))
    destination = quality_folder(folder, phase, kind)
    contract = sha256(Path(folder) / "manifest.json")
    if finalize_labels:
        verify_inputs(destination, records)
        result = finalize(destination, contract, folder)
        return {
            "quality_pass_count": result["quality_pass_count"],
            "items": len(records),
        }
    if (destination / "private_index.json").exists():
        verify_inputs(destination, records)
        return {"items_per_reviewer": len(records), "state": "already_exported"}
    return export_packets(records, destination, kind, contract)


def cohort_gates(folder, phase):
    if phase not in {"pilot", "main"}:
        raise ValueError("Only positive phases have cohort gates")
    documents = documents_for(folder, phase)
    structural = sum(row["structural_pass"] for row in documents)
    minimum = 6 if phase == "pilot" else 320
    if structural < minimum:
        return {
            "counts": {
                "allocated": len(documents),
                "structure_pass": structural,
                "baseline_quality_pass": None,
                "C2_constructed": None,
                "C2_quality_pass": None,
                "common_quality_pairs": None,
            },
            "gates": {"structural_upper_bound": False},
            "passed": False,
            "reason": "Too few structurally eligible documents",
            "branch": "C",
            "unperformed_quality_is_pending_not_failed": True,
        }
    cohort = baseline_cohort(folder, phase, documents)
    transforms = verified_transforms(folder, phase, cohort)
    pairs = decision_for(folder, phase, "pair", pair_records(transforms))
    passing = {
        (row["allocation_id"], row["condition_id"])
        for row in pairs["decisions"]
        if row["quality_pass"]
    }
    constructed = sum("C2" in row["conditions"] for row in transforms)
    quality_c2 = sum(
        (row["allocation_id"], "C2") in passing for row in transforms
    )
    common = [
        row["allocation_id"]
        for row in transforms
        if "C4" in row["conditions"]
        and all((row["allocation_id"], condition) in passing
                for condition in ("C1", "C2", "C3"))
    ]
    counts = {
        "allocated": len(documents),
        "structure_pass": structural,
        "baseline_quality_pass": len(cohort),
        "C2_constructed": constructed,
        "C2_quality_pass": quality_c2,
        "common_quality_pairs": len(common),
    }
    if phase == "pilot":
        gates = {
            "pilot": len(documents) == 8
            and structural >= 6
            and len(cohort) >= 6
            and quality_c2 >= 1
        }
    else:
        gates = {
            "A": len(documents) == 500 and len(cohort) >= 320,
            "C": bool(cohort) and constructed * 5 >= len(cohort) * 4,
            "D": constructed > 0
            and quality_c2 * 5 >= constructed * 4
            and len(common) >= 200,
        }
    return {
        "counts": counts,
        "gates": gates,
        "passed": all(gates.values()),
        "baseline_ids": [row["allocation_id"] for row in cohort],
        "common_ids": common,
        "manifest_sha256": sha256(Path(folder) / "manifest.json"),
    }


def require_pilot(folder):
    result = cohort_gates(Path(folder), "pilot")
    if not result["passed"]:
        raise PermissionError(
            "Pilot gate has not passed; main and human sealed content remain unavailable"
        )
    return result
