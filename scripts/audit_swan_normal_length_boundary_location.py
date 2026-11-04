"""Offline audit of the frozen normal-length boundary-location pilot."""
from __future__ import annotations

import json
import math

import run_swan_normal_length_boundary_location as runner
from swan_feasibility2_common import ROOT, load, now, sha256
from swan_final1_transform import punctuation, sentence_spans, word_tokens

OUT = ROOT / "experiments/swan_normal_length_boundary_location1"
SOURCE = ROOT / "experiments/swan_normal_length_coherent_baseline1/detection.json"


def build_audit():
    contract = runner.check(full_environment=False)
    inputs = load(OUT / "inputs.json")
    quality = load(OUT / "quality_result.json")
    equivalence = load(OUT / "equivalence_check.json")
    detection = load(OUT / "detection.json")
    source = load(SOURCE)
    checks = {}

    checks["manifest_verified"] = (
        contract["manifest_sha256"] == detection["manifest_sha256"]
    )
    checks["quality_gate_pass"] = (
        quality["qualified_for_detection"]
        and not quality["human_quality_reviewed"]
        and len(quality["reviews"]) == 2
        and all(row["pass"] for row in quality["reviews"])
    )
    checks["quality_accounting_exact"] = (
        quality["accounting"]["calls"] == 2
        and quality["accounting"]["uncertain_or_invalid_calls"] == 0
        and quality["accounting"]["actual_peak_nanodollars"] == 345000
    )
    checks["equivalence_pass"] = (
        equivalence["pass"]
        and max(equivalence["absolute_differences"]) <= 1e-12
    )
    checks["standard_c2_honestly_failed"] = (
        not detection["standard_C2_constructed"]
        and detection["standard_C2_failure"] == "no_eligible_internal_split"
        and inputs["construction"]["split_candidate_count"] == 0
    )
    c0_input = inputs["conditions"]["C0"]
    r0_input = inputs["conditions"]["R0"]
    r1_input = inputs["conditions"]["R1"]
    r4_input = inputs["conditions"]["R4"]
    checks["word_order_exact"] = (
        word_tokens(c0_input["text"])
        == word_tokens(r0_input["text"])
        == word_tokens(r1_input["text"])
        == word_tokens(r4_input["text"])
    )
    checks["location_punctuation_matched"] = (
        punctuation(r0_input["text"]) == punctuation(r1_input["text"])
        and r0_input["punctuation"] == r1_input["punctuation"]
    )
    checks["unit_count_exact"] = (
        len(sentence_spans(c0_input["text"])) == 20
        and len(sentence_spans(r0_input["text"])) == 19
        and len(sentence_spans(r1_input["text"])) == 19
        and len(sentence_spans(r4_input["text"])) == 20
    )
    checks["restoration_byte_exact"] = c0_input["text"] == r4_input["text"]

    source_by_hash = {row["text_sha256"]: row for row in source["units"]}
    condition_summaries = {}
    all_unit_checks = True
    reuse_checks = True
    for condition, row in detection["conditions"].items():
        green = 0
        merged = 0
        for unit in row["units"]:
            scores = unit["scores"]
            maximum = max(score["f1"] for score in scores)
            valid = (
                len(scores) == 50
                and all(score["bank_index"] == index
                        for index, score in enumerate(scores))
                and math.isclose(maximum, unit["max_score"], abs_tol=1e-15)
                and unit["green"] == (maximum >= runner.GREEN_THRESHOLD)
                and not unit["diagnostics"]["logged_warnings"]
                and not unit["diagnostics"]["python_warnings"]
                and not unit["diagnostics"]["stderr"]
            )
            all_unit_checks = all_unit_checks and valid
            green += unit["green"]
            if unit["reused_source_unit"]:
                original = source_by_hash.get(unit["text_sha256"])
                reuse_checks = reuse_checks and original is not None \
                    and original["scores"] == unit["scores"] \
                    and original["candidate_template_amr_sha256"] \
                    == unit["candidate_template_amr_sha256"]
            else:
                merged += 1
        n = len(row["units"])
        z = (green - runner.NULL_RATE * n) / math.sqrt(
            n * runner.NULL_RATE * (1 - runner.NULL_RATE)
        )
        all_unit_checks = all_unit_checks and (
            n == row["N"]
            and green == row["green"]
            and math.isclose(green / n, row["green_rate"], abs_tol=1e-15)
            and math.isclose(z, row["z"], abs_tol=1e-15)
            and merged == row["merged_units_scored"]
        )
        condition_summaries[condition] = {
            "N": n,
            "green": green,
            "green_rate": green / n,
            "z": z,
            "merged_units": merged,
        }
    checks["all_score_vectors_recomputed"] = all_unit_checks
    checks["all_source_reuse_exact"] = reuse_checks
    checks["count_neutral_effect_recomputed"] = (
        condition_summaries["R0"]["N"] == condition_summaries["R1"]["N"] == 19
        and condition_summaries["R1"]["green"]
        - condition_summaries["R0"]["green"] == -1
        and math.isclose(
            condition_summaries["R1"]["z"]
            - condition_summaries["R0"]["z"],
            detection["paired_difference_R1_minus_R0"]["z"],
            abs_tol=1e-15,
        )
    )
    merged = {
        condition: [{
            "unit_index": row["unit_index"],
            "text_sha256": row["text_sha256"],
            "max_score": row["max_score"],
            "max_bank_indices": row["max_bank_indices"],
            "target_47_score": row["target_47_score"],
            "green": row["green"],
            "raw_amr_sha256": row["raw_amr_sha256"],
            "candidate_template_amr_sha256":
                row["candidate_template_amr_sha256"],
        } for row in detection["conditions"][condition]["units"]
        if not row["reused_source_unit"]]
        for condition in ("R0", "R1")
    }
    checks["exactly_one_merged_unit_per_location"] = all(
        len(rows) == 1 for rows in merged.values()
    )
    ok = all(checks.values())
    return {
        "experiment": "swan_normal_length_boundary_location1",
        "audited_at": now(),
        "status": "pass" if ok else "fail",
        "ok": ok,
        "checks": checks,
        "condition_summaries": condition_summaries,
        "merged_units": merged,
        "paired_difference_R1_minus_R0":
            detection["paired_difference_R1_minus_R0"],
        "artifact_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                OUT / "manifest.json",
                OUT / "inputs.json",
                OUT / "quality_result.json",
                OUT / "equivalence_check.json",
                OUT / "detection.json",
            )
        },
        "new_generation_calls": 0,
        "quality_review_calls": quality["accounting"]["calls"],
        "parser_calls_during_audit": 0,
        "s2match_calls_during_audit": 0,
        "detector_calls_during_audit": 0,
        "claim_boundary": (
            "One exploratory document supports a boundary-location sensitivity "
            "observation at constant N; it does not support a population attack, "
            "formal standard-C2, calibrated-FPR, or human-quality claim."
        ),
    }


def main():
    audit = build_audit()
    path = OUT / "audit.json"
    rendered = json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        previous = load(path)
        previous.pop("audited_at", None)
        comparable = dict(audit)
        comparable.pop("audited_at", None)
        if previous != comparable:
            raise ValueError("Offline audit changed")
    else:
        path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
