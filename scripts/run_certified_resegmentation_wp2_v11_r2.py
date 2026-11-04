"""Run the frozen WP2-v11 R2 detector-reproduction diagnostics."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import struct

from datasets import load_from_disk
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
import torch

from certified_resegmentation_calibration import (
    clopper_pearson_upper,
    split_conformal_calibrate,
)
from certified_resegmentation_wp2_adapter import hash_raw_value
from run_certified_resegmentation_wp2_detector_equivalence import (
    package_versions,
    semstamp_core,
)
from run_certified_resegmentation_wp2_v11_r1 import (
    EXPECTED_SEMSTAMP_EXCLUSION,
    R0_SHA256,
    PROTOCOL_SHA256,
    atomic_write_json,
    canonical_json_bytes,
    float_sequence_sha256,
    identity_fields,
    materialize_and_validate,
    semstamp_direct_score,
    sha256_bytes,
    sha256_file,
    validate_common_inputs,
)


EXPERIMENT = "certified_resegmentation_wp2_v11_detector_reproduction1"
R1_SHA256 = {
    "semstamp": (
        "4517c64cda80a034048ad7b315bb4657a9a3e3988937c5068fda17641c911023"
    ),
    "simmark": (
        "03226fe166826e498c34c825e97ae8cb35848da3bbabb31e59e63c56350e665d"
    ),
}
V9_GAMMA_SHA256 = (
    "30193a829b1f0ed24bbde2ff386d11d0f79a470e8c2ae6614182aa50bb84d436"
)
V9_SUMMARY_SHA256 = (
    "c8e74d75d2a5bdc7526399ea949c7764188699962a70a864c8549671c474e4b4"
)
V10_P0_SUMMARY_SHA256 = (
    "c2aa47f6c4dbfda6ea03f1d9b9966582d1c931a9a30d5a3c81520b2d00ed8b34"
)
SIMMARK_INSTRUCTION = "Represent the sentence for cosine similarity:"
SIMMARK_INTERVAL = (0.68, 0.76)
SIMMARK_K = 250
TARGET_FPR = 0.05


def load_json_checked(path, expected_sha256, label):
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA256 mismatch")
    return json.loads(Path(path).read_text(encoding="ascii"))


def finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("expected a finite float")
    return result


def quantile(values, probability):
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("distribution requires finite nonempty values")
    return float(np.quantile(array, probability, method="linear"))


def distribution(values):
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("distribution requires finite nonempty values")
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p01": quantile(array, 0.01),
        "p05": quantile(array, 0.05),
        "p25": quantile(array, 0.25),
        "p50": quantile(array, 0.50),
        "p75": quantile(array, 0.75),
        "p95": quantile(array, 0.95),
        "p99": quantile(array, 0.99),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def clopper_pearson_lower(successes, trials, confidence=0.95):
    if not 0 <= successes <= trials or trials < 1:
        raise ValueError("require 0 <= successes <= trials and trials >= 1")
    if successes == 0:
        return 0.0
    return 1.0 - clopper_pearson_upper(
        trials - successes,
        trials,
        confidence,
    )


def binary_rate(scores, threshold, confidence=0.95):
    values = tuple(float(value) for value in scores)
    successes = sum(value > threshold for value in values)
    trials = len(values)
    return {
        "trials": trials,
        "strict_exceedances": successes,
        "rate": successes / trials,
        "one_sided_cp_confidence": confidence,
        "one_sided_cp_lower": clopper_pearson_lower(
            successes, trials, confidence
        ),
        "one_sided_cp_upper": clopper_pearson_upper(
            successes, trials, confidence
        ),
    }


def auroc(null_scores, positive_scores):
    null = np.asarray(tuple(null_scores), dtype=np.float64)
    positive = np.asarray(tuple(positive_scores), dtype=np.float64)
    if (
        null.size == 0
        or positive.size == 0
        or not np.all(np.isfinite(null))
        or not np.all(np.isfinite(positive))
    ):
        raise ValueError("AUROC requires finite nonempty score arrays")
    labels = np.concatenate((
        np.zeros(null.size, dtype=np.uint8),
        np.ones(positive.size, dtype=np.uint8),
    ))
    scores = np.concatenate((null, positive))
    return float(roc_auc_score(labels, scores))


def sequence_digest(values):
    return float_sequence_sha256(float(value) for value in values)


def index_sequence_digest(values):
    payload = b"".join(struct.pack("<I", int(value)) for value in values)
    return sha256_bytes(payload)


def gamma_diagnostics(distances):
    values = np.asarray(tuple(distances), dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("gamma distances must be finite and nonempty")
    counts, edges = np.histogram(values, bins=1000)
    centers = (edges[:-1] + edges[1:]) / 2
    histogram_count = int(np.sum(
        counts[
            (centers >= SIMMARK_INTERVAL[0])
            & (centers <= SIMMARK_INTERVAL[1])
        ]
    ))
    exact_count = int(np.sum(
        (values >= SIMMARK_INTERVAL[0])
        & (values <= SIMMARK_INTERVAL[1])
    ))
    histogram_gamma = histogram_count / int(values.size)
    exact_gamma = exact_count / int(values.size)
    return {
        "pair_count": int(values.size),
        "histogram_bins": 1000,
        "histogram_interval_count": histogram_count,
        "histogram_gamma": histogram_gamma,
        "exact_interval_count": exact_count,
        "exact_gamma": exact_gamma,
        "histogram_minus_exact": histogram_gamma - exact_gamma,
        "distance_sha256": sequence_digest(values),
        "histogram_counts_sha256": sha256_bytes(
            counts.astype("<u8", copy=False).tobytes()
        ),
        "histogram_edges_sha256": sha256_bytes(
            edges.astype("<f8", copy=False).tobytes()
        ),
    }


def distance_summary(distances):
    values = tuple(float(value) for value in distances)
    interval_count = sum(
        SIMMARK_INTERVAL[0] <= value <= SIMMARK_INTERVAL[1]
        for value in values
    )
    return {
        "distribution": distribution(values),
        "interval_count": interval_count,
        "interval_rate": interval_count / len(values),
        "sha256": sequence_digest(values),
    }


def score_from_soft_sum(soft_sum, edge_count, gamma):
    if edge_count < 1 or not 0 <= gamma <= 1:
        raise ValueError("invalid SimMark score parameters")
    return float(
        (float(soft_sum) - gamma * edge_count)
        / math.sqrt(edge_count * gamma * (1 - gamma) + 1e-12)
    )


def official_simmark_thresholds(null_scores):
    values = np.asarray(tuple(null_scores), dtype=np.float64)
    fpr_5_threshold = 0.0
    fpr_1_threshold = 0.0
    for threshold in np.arange(10, -10, -0.001):
        false_positive_rate = float(np.mean(values > threshold))
        if 0.0095 <= false_positive_rate <= 0.0104:
            fpr_1_threshold = float(threshold)
        elif 0.045 <= false_positive_rate <= 0.054:
            fpr_5_threshold = float(threshold)
    if fpr_1_threshold == 0:
        fpr_1_threshold = 2.33
    return {
        "one_percent": fpr_1_threshold,
        "five_percent": fpr_5_threshold,
        "source_semantics": "SimMark detection.py:evaluate_z_scores",
    }


def official_semstamp_thresholds(null_scores):
    values = np.asarray(tuple(null_scores), dtype=np.float64)
    fpr_5_threshold = 0.0
    fpr_1_threshold = 0.0
    for threshold in np.arange(0, 6, 0.005):
        false_positive_rate = float(np.mean(values > threshold))
        if 0.0095 <= false_positive_rate <= 0.0104:
            fpr_1_threshold = float(threshold)
        elif 0.045 <= false_positive_rate <= 0.054:
            fpr_5_threshold = float(threshold)
    if fpr_1_threshold == 0:
        fpr_1_threshold = 2.33
    return {
        "one_percent": fpr_1_threshold,
        "five_percent": fpr_5_threshold,
        "source_semantics": "SemStamp detection_utils.py:evaluate_z_scores",
        "five_percent_zero_has_no_official_fallback": True,
    }


def threshold_report(null_scores, positive_scores, thresholds):
    return {
        name: {
            "threshold": threshold,
            "null": binary_rate(null_scores, threshold),
            "positive": binary_rate(positive_scores, threshold),
        }
        for name, threshold in thresholds.items()
        if name in {"one_percent", "five_percent"}
    }


def split_report(scores):
    calibration_scores = scores[200:600]
    test_scores = scores[600:1000]
    positive_scores = scores[1000:2000]
    calibration = split_conformal_calibrate(
        calibration_scores, TARGET_FPR
    )
    threshold = calibration.threshold
    return {
        "calibration": calibration.to_dict(),
        "test": binary_rate(test_scores, threshold),
        "positive": binary_rate(positive_scores, threshold),
    }


def set_pooling_include_prompt(model, value):
    if not hasattr(model, "set_pooling_include_prompt"):
        raise ValueError("SentenceTransformer lacks pooling prompt control")
    model.set_pooling_include_prompt(bool(value))
    states = [
        bool(module.include_prompt)
        for module in model
        if hasattr(module, "include_prompt")
    ]
    if states != [bool(value)]:
        raise ValueError(
            f"pooling include_prompt did not become {bool(value)}: {states}"
        )


def prepare_simmark_documents(items, sources):
    groups = []
    for index, (item, source) in enumerate(zip(items, sources)):
        document = materialize_and_validate(item, source, index)
        official_units = tuple(sent_tokenize(item["raw_value"]))
        if official_units != document.original_units:
            raise AssertionError(
                f"SimMark official units changed at manifest index {index}"
            )
        if len(official_units) < 2:
            raise ValueError(
                f"SimMark document has fewer than two units at index {index}"
            )
        groups.append(official_units)
    return groups


def encode_simmark_variant(model, groups, fix_space, include_prompt):
    set_pooling_include_prompt(model, include_prompt)
    flat = []
    offsets = [0]
    for group in groups:
        flat.extend(
            unit if (index == 0 or not fix_space) else " " + unit
            for index, unit in enumerate(group)
        )
        offsets.append(len(flat))
    with torch.inference_mode():
        embeddings = np.asarray(model.encode(
            flat,
            prompt=SIMMARK_INSTRUCTION,
            batch_size=256,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ), dtype=np.float32)
    if embeddings.shape != (len(flat), 768):
        raise ValueError(f"unexpected SimMark embedding shape: {embeddings.shape}")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("SimMark returned non-finite embeddings")

    rows = []
    all_distances = []
    for index, group in enumerate(groups):
        selected = embeddings[offsets[index]:offsets[index + 1]]
        distances = [
            float(cosine_similarity(
                selected[edge].reshape(1, -1),
                selected[edge + 1].reshape(1, -1),
            )[0, 0])
            for edge in range(len(group) - 1)
        ]
        soft_counts = [
            float(np.exp(
                -SIMMARK_K
                * (
                    0.0
                    if SIMMARK_INTERVAL[0] <= distance <= SIMMARK_INTERVAL[1]
                    else min(
                        abs(distance - SIMMARK_INTERVAL[0]),
                        abs(distance - SIMMARK_INTERVAL[1]),
                    )
                )
            ))
            for distance in distances
        ]
        rows.append({
            "edge_count": len(distances),
            "soft_sum": float(sum(soft_counts)),
            "distance_sha256": sequence_digest(distances),
            "soft_count_sha256": sequence_digest(soft_counts),
            "distances": distances,
        })
        all_distances.extend(distances)
    del embeddings
    torch.cuda.empty_cache()
    return {
        "fix_space": fix_space,
        "include_prompt": include_prompt,
        "sentence_count": len(flat),
        "pair_count": len(all_distances),
        "all_distance_sha256": sequence_digest(all_distances),
        "rows": rows,
    }


def simmark_score_rows(encoded, gamma, sources):
    rows = []
    for index, (encoded_row, source) in enumerate(
        zip(encoded["rows"], sources)
    ):
        score = score_from_soft_sum(
            encoded_row["soft_sum"],
            encoded_row["edge_count"],
            gamma,
        )
        rows.append({
            "manifest_index": index,
            **identity_fields(source),
            "edge_count": encoded_row["edge_count"],
            "raw_soft_sum": encoded_row["soft_sum"],
            "score": score,
            "contains_raw_text": False,
        })
    return rows


def simmark_arm(
    arm_id,
    encoded,
    gamma_scope,
    gamma_kind,
    sources,
    s1_scores=None,
):
    if gamma_scope == "all_null":
        fit_rows = encoded["rows"][:1000]
    elif gamma_scope == "fit_200":
        fit_rows = encoded["rows"][:200]
    else:
        raise ValueError(f"unknown gamma scope: {gamma_scope}")
    fit_distances = [
        value for row in fit_rows for value in row["distances"]
    ]
    gamma_report = gamma_diagnostics(fit_distances)
    gamma = gamma_report[
        "exact_gamma" if gamma_kind == "exact" else "histogram_gamma"
    ]
    rows = simmark_score_rows(encoded, gamma, sources)
    scores = [row["score"] for row in rows]
    null_scores = scores[:1000]
    positive_scores = scores[1000:]
    null_distances = [
        value for row in encoded["rows"][:1000] for value in row["distances"]
    ]
    positive_distances = [
        value for row in encoded["rows"][1000:] for value in row["distances"]
    ]
    thresholds = official_simmark_thresholds(null_scores)
    delta = None
    if s1_scores is not None:
        delta = {
            "all_documents": distribution(
                score - reference
                for score, reference in zip(scores, s1_scores)
            ),
            "null": distribution(
                score - reference
                for score, reference in zip(
                    null_scores, s1_scores[:1000]
                )
            ),
            "positive": distribution(
                score - reference
                for score, reference in zip(
                    positive_scores, s1_scores[1000:]
                )
            ),
        }
    return {
        "arm_id": arm_id,
        "configuration": {
            "fix_space": encoded["fix_space"],
            "include_prompt": encoded["include_prompt"],
            "gamma_scope": gamma_scope,
            "gamma_kind_used": gamma_kind,
            "gamma_used": gamma,
            "interval": list(SIMMARK_INTERVAL),
            "K": SIMMARK_K,
        },
        "counts": {
            "null_documents": 1000,
            "positive_documents": 1000,
            "null_sentences": sum(
                source["unit_count"] for source in sources[:1000]
            ),
            "positive_sentences": sum(
                source["unit_count"] for source in sources[1000:]
            ),
            "null_pairs": len(null_distances),
            "positive_pairs": len(positive_distances),
        },
        "gamma": gamma_report,
        "cosine": {
            "null": distance_summary(null_distances),
            "positive": distance_summary(positive_distances),
        },
        "scores": {
            "null": distribution(null_scores),
            "positive": distribution(positive_scores),
            "null_sha256": sequence_digest(null_scores),
            "positive_sha256": sequence_digest(positive_scores),
            "non_finite_documents": sum(
                not math.isfinite(value) for value in scores
            ),
        },
        "official_same_sample": {
            "thresholds": thresholds,
            "performance": threshold_report(
                null_scores, positive_scores, thresholds
            ),
            "auroc": auroc(null_scores, positive_scores),
        },
        "v9_split": split_report(scores),
        "score_delta_from_S1": delta,
        "rows": rows,
    }


def classify_simmark(arms):
    criteria = {
        "gamma_nondegenerate": "0 < gamma < 1",
        "restored_separation": (
            "AUROC >= 0.80 and official same-sample 5% TPR >= 0.80"
        ),
        "s0_inseparable": (
            "AUROC <= 0.55 and official same-sample 5% TPR <= 0.20"
        ),
    }

    def restored(arm):
        gamma = arm["configuration"]["gamma_used"]
        return (
            0 < gamma < 1
            and arm["official_same_sample"]["auroc"] >= 0.80
            and arm["official_same_sample"]["performance"][
                "five_percent"
            ]["positive"]["rate"] >= 0.80
        )

    restored_arms = [
        arm_id for arm_id in ("S2_EXACT_GAMMA", "S3_INCLUDE_PROMPT_FALSE",
                              "S4_NO_LEADING_SPACE")
        if restored(arms[arm_id])
    ]
    s0 = arms["S0_OFFICIAL_ALL_NULL"]
    s0_gamma = s0["configuration"]["gamma_used"]
    s0_bad = (
        s0_gamma in {0.0, 1.0}
        and s0["official_same_sample"]["auroc"] <= 0.55
        and s0["official_same_sample"]["performance"][
            "five_percent"
        ]["positive"]["rate"] <= 0.20
    )
    if restored_arms == ["S2_EXACT_GAMMA"]:
        classification = "histogram_gamma_estimator"
    elif restored_arms == ["S3_INCLUDE_PROMPT_FALSE"]:
        classification = "instructor_prompt_pooling_compatibility"
    elif restored_arms == ["S4_NO_LEADING_SPACE"]:
        classification = "leading_space_preprocessing"
    elif s0_bad and not restored_arms:
        classification = (
            "released_pipeline_incompatibility_at_pinned_environment"
        )
    else:
        classification = "unresolved_or_interacting_causes"
    return {
        "classification": classification,
        "operational_criteria_frozen_before_r2_results": criteria,
        "restored_arms": restored_arms,
        "s0_degenerate_and_inseparable": s0_bad,
    }


def run_simmark(args, identity, items, sources, r1):
    from run_certified_resegmentation_wp2_v6_provider import load_simmark

    v9_gamma = load_json_checked(
        args.v9_gamma, V9_GAMMA_SHA256, "WP2-v9 gamma"
    )
    v9_summary = load_json_checked(
        args.v9_summary, V9_SUMMARY_SHA256, "WP2-v9 summary"
    )
    groups = prepare_simmark_documents(items, sources)
    model = load_simmark(args.model_dir)
    print(json.dumps({"stage": "simmark_encode", "variant": "baseline"}),
          flush=True)
    baseline = encode_simmark_variant(
        model, groups, fix_space=True, include_prompt=True
    )
    print(json.dumps({"stage": "simmark_encode", "variant": "prompt_false"}),
          flush=True)
    prompt_false = encode_simmark_variant(
        model, groups, fix_space=True, include_prompt=False
    )
    print(json.dumps({"stage": "simmark_encode", "variant": "no_leading_space"}),
          flush=True)
    no_leading_space = encode_simmark_variant(
        model, groups, fix_space=False, include_prompt=True
    )

    s1 = simmark_arm(
        "S1_V9_SPLIT", baseline, "fit_200", "histogram", sources
    )
    s1_scores = [row["score"] for row in s1["rows"]]
    arms = {
        "S0_OFFICIAL_ALL_NULL": simmark_arm(
            "S0_OFFICIAL_ALL_NULL",
            baseline,
            "all_null",
            "histogram",
            sources,
            s1_scores,
        ),
        "S1_V9_SPLIT": s1,
        "S2_EXACT_GAMMA": simmark_arm(
            "S2_EXACT_GAMMA",
            baseline,
            "fit_200",
            "exact",
            sources,
            s1_scores,
        ),
        "S3_INCLUDE_PROMPT_FALSE": simmark_arm(
            "S3_INCLUDE_PROMPT_FALSE",
            prompt_false,
            "fit_200",
            "histogram",
            sources,
            s1_scores,
        ),
        "S4_NO_LEADING_SPACE": simmark_arm(
            "S4_NO_LEADING_SPACE",
            no_leading_space,
            "fit_200",
            "histogram",
            sources,
            s1_scores,
        ),
    }
    arms["S1_V9_SPLIT"]["score_delta_from_S1"] = {
        "all_documents": distribution([0.0] * 2000),
        "null": distribution([0.0] * 1000),
        "positive": distribution([0.0] * 1000),
    }
    observed = arms["S1_V9_SPLIT"]["v9_split"]
    expected = {
        "gamma": v9_gamma["gamma"],
        "threshold": v9_summary["calibration"]["0"]["threshold"],
        "test_exceedances": v9_summary["test"]["0"]["strict_exceedances"],
        "positive_exceedances": (
            v9_summary["positive"]["0"]["strict_exceedances"]
        ),
    }
    replay = {
        "expected": expected,
        "observed": {
            "gamma": arms["S1_V9_SPLIT"]["configuration"]["gamma_used"],
            "threshold": observed["calibration"]["threshold"],
            "test_exceedances": observed["test"]["strict_exceedances"],
            "positive_exceedances": observed["positive"]["strict_exceedances"],
        },
    }
    replay["comparison"] = {
        "gamma_exact": replay["observed"]["gamma"] == expected["gamma"],
        "threshold_abs_error": abs(
            replay["observed"]["threshold"] - expected["threshold"]
        ),
        "test_exceedances_exact": (
            replay["observed"]["test_exceedances"]
            == expected["test_exceedances"]
        ),
        "positive_exceedances_exact": (
            replay["observed"]["positive_exceedances"]
            == expected["positive_exceedances"]
        ),
    }
    return {
        "schema_version": 1,
        "created_on": "2026-09-25",
        "experiment": EXPERIMENT,
        "stage": "R2_frozen_diagnostic_matrix",
        "system": "simmark",
        "status": "complete_exploratory",
        "protocol_sha256": PROTOCOL_SHA256,
        "r0_sha256": R0_SHA256,
        "r1_sha256": R1_SHA256["simmark"],
        "identity": identity,
        "environment": {
            **package_versions(),
            "platform": platform.platform(),
            "cuda_device": torch.cuda.get_device_name(0),
            "encoding_batch_size": 256,
        },
        "r1_gate": {
            "status": r1["status"],
            "aggregate_rows_sha256": r1["aggregate_rows_sha256"],
            "maximum_errors": r1["execution"]["maximum_errors"],
        },
        "arms": arms,
        "v9_replay": replay,
        "root_cause": classify_simmark(arms),
        "r4": {"status": "not_run_capacity_stop"},
        "attestation": {
            "historically_observed_release_data": True,
            "exploratory_only": True,
            "performance_metrics_computed": True,
            "watermark_generation_calls": 0,
            "provider_api_calls": 0,
            "certificate_calls": 0,
            "bertscore_calls": 0,
            "contains_raw_text": False,
            "network_access": False,
        },
    }


def validate_unit_list(value, label):
    if (
        not isinstance(value, list)
        or len(value) < 2
        or not all(isinstance(unit, str) for unit in value)
    ):
        raise ValueError(f"{label} is not an official list with >=2 units")
    return tuple(value)


def encode_semstamp_groups(model, relation, groups):
    flat = []
    offsets = [0]
    for group in groups:
        flat.extend(group)
        offsets.append(len(flat))
    with torch.inference_mode():
        regions = np.asarray(model.get_hash(flat), dtype=np.uint8)
    if regions.shape != (len(flat),):
        raise ValueError(f"unexpected SemStamp region shape: {regions.shape}")
    scores = []
    rows = []
    for index, group in enumerate(groups):
        selected = regions[offsets[index]:offsets[index + 1]]
        hits, score = semstamp_direct_score(selected, relation)
        scores.append(score)
        rows.append({
            "unit_count": len(group),
            "edge_count": len(group) - 1,
            "hit_count": int(sum(hits)),
            "regions_sha256": sha256_bytes(selected.tobytes(order="C")),
            "score": score,
        })
    return scores, rows


def official_semstamp_report(human_scores, clean_scores, para_scores):
    thresholds = official_semstamp_thresholds(human_scores)
    return {
        "score_distributions": {
            "human": distribution(human_scores),
            "clean": distribution(clean_scores),
            "para": distribution(para_scores),
        },
        "score_sha256": {
            "human": sequence_digest(human_scores),
            "clean": sequence_digest(clean_scores),
            "para": sequence_digest(para_scores),
        },
        "auroc": {
            "official_primary_para_vs_human": auroc(
                human_scores, para_scores
            ),
            "supplemental_clean_vs_human": auroc(
                human_scores, clean_scores
            ),
        },
        "thresholds": thresholds,
        "performance": {
            "clean": threshold_report(
                human_scores, clean_scores, thresholds
            ),
            "para": threshold_report(
                human_scores, para_scores, thresholds
            ),
        },
        "official_primary_positive": "para_text",
        "bertscore_skipped_by_protocol": True,
    }


def semstamp_m1(rows_by_index, v10_summary):
    calibration_scores = [
        rows_by_index[index]["score"] for index in range(200, 600)
    ]
    test_scores = [
        rows_by_index[index]["score"] for index in range(600, 1000)
    ]
    positive_scores = [
        rows_by_index[index]["score"]
        for index in range(1000, 2000)
        if index != EXPECTED_SEMSTAMP_EXCLUSION
    ]
    calibration = split_conformal_calibrate(
        calibration_scores, TARGET_FPR
    )
    threshold = calibration.threshold
    result = {
        "calibration": calibration.to_dict(),
        "test": binary_rate(test_scores, threshold),
        "positive": binary_rate(positive_scores, threshold),
    }
    result["v10_exact_replay"] = {
        "threshold_exact": (
            threshold == v10_summary["calibration"]["threshold"]
        ),
        "calibration_exceedances_exact": (
            result["calibration"]["strict_exceedances"]
            == v10_summary["calibration"]["strict_exceedances"]
        ),
        "test_exceedances_exact": (
            result["test"]["strict_exceedances"]
            == v10_summary["test"]["strict_exceedances"]
        ),
        "positive_exceedances_exact": (
            result["positive"]["strict_exceedances"]
            == v10_summary["positive"]["strict_exceedances"]
        ),
    }
    if not all(result["v10_exact_replay"].values()):
        raise AssertionError("SemStamp M1 did not exactly replay WP2-v10")
    return result


def semstamp_m2(null_rows):
    rows = []
    for split_index in range(100):
        prefix = f"WP2-v11-semstamp-split-{split_index}"
        ordered = sorted(
            null_rows,
            key=lambda row: hashlib.sha256(
                (prefix + row["raw_text_sha256"]).encode("ascii")
            ).digest(),
        )
        calibration_rows = ordered[:400]
        test_rows = ordered[400:800]
        calibration = split_conformal_calibrate(
            [row["score"] for row in calibration_rows],
            TARGET_FPR,
        )
        threshold = calibration.threshold
        test = binary_rate(
            [row["score"] for row in test_rows], threshold
        )
        rows.append({
            "split_index": split_index,
            "calibration_manifest_indices_sha256": index_sequence_digest(
                row["manifest_index"] for row in calibration_rows
            ),
            "test_manifest_indices_sha256": index_sequence_digest(
                row["manifest_index"] for row in test_rows
            ),
            "unused_manifest_indices_sha256": index_sequence_digest(
                row["manifest_index"] for row in ordered[800:]
            ),
            "threshold": threshold,
            "calibration_strict_exceedances": (
                calibration.strict_exceedances
            ),
            "test_strict_exceedances": test["strict_exceedances"],
            "test_fpr": test["rate"],
            "test_cp95_upper": test["one_sided_cp_upper"],
            "raw_order_sha256": sha256_bytes(
                "".join(row["raw_text_sha256"] for row in ordered).encode(
                    "ascii"
                )
            ),
        })
    fprs = [row["test_fpr"] for row in rows]
    cp_upper = [row["test_cp95_upper"] for row in rows]
    thresholds = [row["threshold"] for row in rows]
    return {
        "splits": rows,
        "summary": {
            "threshold": distribution(thresholds),
            "test_fpr": distribution(fprs),
            "test_cp95_upper": distribution(cp_upper),
            "fpr_gt_0_05_count": sum(value > 0.05 for value in fprs),
            "cp95_upper_gt_0_08_count": sum(
                value > 0.08 for value in cp_upper
            ),
            "both_fpr_and_cp_gate_fail_count": sum(
                fpr > 0.05 and upper > 0.08
                for fpr, upper in zip(fprs, cp_upper)
            ),
        },
    }


def semstamp_length_deciles(null_rows, threshold):
    ordered = sorted(
        null_rows,
        key=lambda row: (row["unit_count"], row["raw_text_sha256"]),
    )
    output = []
    direction_count = 0
    comparable = 0
    for decile in range(10):
        selected = ordered[decile * 100:(decile + 1) * 100]
        cohorts = {}
        for cohort in (
            "detector_fit",
            "certificate_calibration",
            "independent_test",
        ):
            cohort_rows = [row for row in selected if row["cohort"] == cohort]
            scores = [row["score"] for row in cohort_rows]
            cohorts[cohort] = (
                {
                    "trials": len(scores),
                    "strict_exceedances": sum(
                        score > threshold for score in scores
                    ),
                    "rate": (
                        sum(score > threshold for score in scores) / len(scores)
                        if scores else None
                    ),
                }
            )
        calibration_rate = cohorts["certificate_calibration"]["rate"]
        test_rate = cohorts["independent_test"]["rate"]
        if calibration_rate is not None and test_rate is not None:
            comparable += 1
            direction_count += int(test_rate > calibration_rate)
        output.append({
            "decile": decile + 1,
            "trials": len(selected),
            "unit_count_min": min(row["unit_count"] for row in selected),
            "unit_count_max": max(row["unit_count"] for row in selected),
            "unit_count_mean": float(np.mean([
                row["unit_count"] for row in selected
            ])),
            "all_null": binary_rate(
                [row["score"] for row in selected], threshold
            ),
            "cohorts": cohorts,
        })
    return {
        "construction": (
            "rank all 1000 null rows by (unit_count, raw_text_sha256), "
            "then take ten consecutive groups of 100"
        ),
        "deciles": output,
        "test_rate_gt_calibration_rate_deciles": direction_count,
        "comparable_deciles": comparable,
        "same_direction_majority": (
            comparable > 0 and direction_count >= math.ceil(0.6 * comparable)
        ),
        "same_direction_rule": (
            "test exceedance rate is greater than calibration rate in at "
            "least 60% of comparable fixed rank-deciles"
        ),
    }


def semstamp_m3(human_scores, clean_scores, para_scores):
    return {
        str(threshold): {
            "threshold": threshold,
            "human": binary_rate(human_scores, threshold),
            "clean": binary_rate(clean_scores, threshold),
            "para": binary_rate(para_scores, threshold),
        }
        for threshold in (1.645, 2.33)
    }


def classify_semstamp(m0, m1, m2, length_deciles):
    m0_normal = (
        m0["score_distributions"]["human"]["count"] == 1000
        and m0["score_distributions"]["clean"]["count"] == 999
        and m0["score_distributions"]["para"]["count"] == 999
        and m0["auroc"]["supplemental_clean_vs_human"] > 0.5
        and m0["auroc"]["official_primary_para_vs_human"] > 0.5
    )
    m1_fpr = m1["test"]["rate"]
    p05 = m2["summary"]["test_fpr"]["p05"]
    p95 = m2["summary"]["test_fpr"]["p95"]
    in_central_range = p05 <= m1_fpr <= p95
    above_p95 = m1_fpr > p95
    majority_persistent = (
        m2["summary"]["both_fpr_and_cp_gate_fail_count"] > 50
    )
    if m0_normal and in_central_range:
        classification = (
            "external_generalization_gate_not_official_reproduction_failure"
        )
    elif above_p95 and length_deciles["same_direction_majority"]:
        classification = (
            "deterministic_split_instability_with_length_association"
        )
    elif majority_persistent:
        classification = "persistent_null_calibration_mismatch"
    else:
        classification = "unresolved_or_mixed_statistical_cause"
    return {
        "classification": classification,
        "m0_operationally_normal": m0_normal,
        "m0_normal_rule": (
            "all direct-list scores finite at frozen denominators and both "
            "clean/para AUROC versus human exceed 0.5; no published numeric "
            "oracle exists"
        ),
        "m1_test_fpr": m1_fpr,
        "m2_test_fpr_p05": p05,
        "m2_test_fpr_p95": p95,
        "m1_in_m2_p05_p95": in_central_range,
        "m1_above_m2_p95": above_p95,
        "length_same_direction_majority": (
            length_deciles["same_direction_majority"]
        ),
        "m2_majority_both_gates_fail": majority_persistent,
    }


def run_semstamp(args, identity, items, sources, r1):
    v10_summary = load_json_checked(
        args.v10_summary,
        V10_P0_SUMMARY_SHA256,
        "WP2-v10 P0 summary",
    )
    r1_rows = r1["execution"]["rows"]
    rows_by_index = {
        row["manifest_index"]: {
            **row,
            "score": finite_float(row["direct_score"]),
        }
        for row in r1_rows
    }
    expected_indices = set(range(2000)) - {EXPECTED_SEMSTAMP_EXCLUSION}
    if set(rows_by_index) != expected_indices:
        raise ValueError("SemStamp R1 row identity set changed")

    positive_dataset = load_from_disk(str(
        args.root
        / "repos/semstamp-data/c4-semstamp-pegasus-parrot/"
        "semstamp-pegasus-bigram=False"
    ))
    if positive_dataset.column_names != ["text", "para_text"]:
        raise ValueError("SemStamp positive columns changed")
    positive_items = items[1000:2000]
    positive_sources = sources[1000:2000]
    excluded_position = EXPECTED_SEMSTAMP_EXCLUSION - 1000
    clean_excluded = validate_unit_list(
        positive_items[excluded_position]["raw_value"],
        "SemStamp excluded official clean row",
    )
    para_groups = []
    para_hashes = []
    for source in positive_sources:
        row = positive_dataset[source["row_index"]]
        if hash_raw_value(row["text"]) != source["raw_text_sha256"]:
            raise ValueError("SemStamp positive row identity changed")
        para_groups.append(validate_unit_list(
            row["para_text"], "SemStamp para_text"
        ))
        para_hashes.append(hash_raw_value(row["para_text"]))

    model, get_mask, _official_detect = semstamp_core(
        args.root / "repos/SemStamp", args.model_dir
    )
    model.batch_size = 64
    relation = bytes(
        int(current in {
            int(value)
            for value in get_mask(
                3, 0.25, previous
            ).detach().cpu().tolist()
        })
        for previous in range(8)
        for current in range(8)
    )
    print(json.dumps({
        "stage": "semstamp_encode",
        "groups": 1 + len(para_groups),
    }), flush=True)
    encoded_scores, encoded_rows = encode_semstamp_groups(
        model, relation, [clean_excluded, *para_groups]
    )
    excluded_clean_score = encoded_scores[0]
    para_scores_all = encoded_scores[1:]
    para_encoded_rows = encoded_rows[1:]

    human_rows = [rows_by_index[index] for index in range(1000)]
    human_scores = [row["score"] for row in human_rows]
    clean_scores_direct = [
        rows_by_index[index]["score"]
        for index in range(1000, 2000)
        if index != EXPECTED_SEMSTAMP_EXCLUSION
    ]
    para_scores_direct = [
        score
        for position, score in enumerate(para_scores_all)
        if position != excluded_position
    ]
    clean_scores_official = []
    for index in range(1000, 2000):
        clean_scores_official.append(
            excluded_clean_score
            if index == EXPECTED_SEMSTAMP_EXCLUSION
            else rows_by_index[index]["score"]
        )

    m0_direct = official_semstamp_report(
        human_scores, clean_scores_direct, para_scores_direct
    )
    m0_official = official_semstamp_report(
        human_scores, clean_scores_official, para_scores_all
    )
    m1 = semstamp_m1(rows_by_index, v10_summary)
    m2 = semstamp_m2(human_rows)
    m1_fpr = m1["test"]["rate"]
    m2_fprs = [row["test_fpr"] for row in m2["splits"]]
    m2_cp = [row["test_cp95_upper"] for row in m2["splits"]]
    m1["m2_empirical_percentile"] = {
        "definition": "100 * count(rotated_value <= M1_value) / 100",
        "test_fpr": 100.0 * sum(
            value <= m1_fpr for value in m2_fprs
        ) / len(m2_fprs),
        "test_cp95_upper": 100.0 * sum(
            value <= m1["test"]["one_sided_cp_upper"] for value in m2_cp
        ) / len(m2_cp),
    }
    length_deciles = semstamp_length_deciles(
        human_rows, m1["calibration"]["threshold"]
    )
    m3 = semstamp_m3(
        human_scores, clean_scores_direct, para_scores_direct
    )

    para_rows = [
        {
            "manifest_index": 1000 + position,
            "row_index": source["row_index"],
            "clean_raw_text_sha256": source["raw_text_sha256"],
            "para_text_sha256": para_hash,
            **encoded,
            "included_in_direct_list_aggregate": (
                1000 + position != EXPECTED_SEMSTAMP_EXCLUSION
            ),
            "contains_raw_text": False,
        }
        for position, (source, para_hash, encoded) in enumerate(zip(
            positive_sources, para_hashes, para_encoded_rows
        ))
    ]
    return {
        "schema_version": 1,
        "created_on": "2026-09-25",
        "experiment": EXPERIMENT,
        "stage": "R2_frozen_diagnostic_matrix",
        "system": "semstamp",
        "status": "complete_exploratory",
        "protocol_sha256": PROTOCOL_SHA256,
        "r0_sha256": R0_SHA256,
        "r1_sha256": R1_SHA256["semstamp"],
        "identity": identity,
        "environment": {
            **package_versions(),
            "platform": platform.platform(),
            "cuda_device": torch.cuda.get_device_name(0),
            "encoding_batch_size": 64,
        },
        "r1_gate": {
            "status": r1["status"],
            "aggregate_rows_sha256": r1["aggregate_rows_sha256"],
            "score_max_abs_error": r1["execution"]["score_max_abs_error"],
            "checkpoint_comparison": (
                r1["execution"]["checkpoint_comparison"]
            ),
        },
        "M0_OFFICIAL_ALL_NULL": {
            "official_raw_list_1000": m0_official,
            "frozen_direct_list_999": m0_direct,
            "excluded_manifest_index": EXPECTED_SEMSTAMP_EXCLUSION,
            "excluded_official_clean_produced_output": True,
            "excluded_official_clean_score": excluded_clean_score,
            "denominator_policy": (
                "Official raw-list statistics are reported separately at "
                "1000; all frozen direct-list inference and classification "
                "retain the preregistered 999 denominator."
            ),
        },
        "M1_V10_SPLIT": m1,
        "M2_ROTATED_SPLITS": m2,
        "M3_FIXED_Z": m3,
        "null_length_deciles_at_m1_threshold": length_deciles,
        "para_rows": para_rows,
        "root_cause": classify_semstamp(
            m0_direct, m1, m2, length_deciles
        ),
        "r4": {"status": "not_run_capacity_stop"},
        "attestation": {
            "historically_observed_release_data": True,
            "exploratory_only": True,
            "performance_metrics_computed": True,
            "watermark_generation_calls": 0,
            "provider_api_calls": 0,
            "certificate_calls": 0,
            "bertscore_calls": 0,
            "contains_raw_text": False,
            "network_access": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--system", choices=("semstamp", "simmark"), required=True
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--r0", type=Path, required=True)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--r1", type=Path, required=True)
    parser.add_argument("--v9-gamma", type=Path)
    parser.add_argument("--v9-summary", type=Path)
    parser.add_argument("--v10-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "root", "model_dir", "nltk_data", "protocol", "r0",
        "identity_manifest", "r1", "output",
    ):
        setattr(args, name, getattr(args, name).resolve())
    for name in ("v9_gamma", "v9_summary", "v10_summary"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.system == "simmark" and (
        args.v9_gamma is None or args.v9_summary is None
    ):
        parser.error("SimMark requires --v9-gamma and --v9-summary")
    if args.system == "semstamp" and args.v10_summary is None:
        parser.error("SemStamp requires --v10-summary")

    identity, items, sources = validate_common_inputs(args)
    r1 = load_json_checked(
        args.r1, R1_SHA256[args.system], f"{args.system} R1"
    )
    if (
        r1.get("system") != args.system
        or r1.get("status") != "pass"
        or r1.get("protocol_sha256") != PROTOCOL_SHA256
        or r1.get("r0_sha256") != R0_SHA256
    ):
        raise ValueError(f"{args.system} R1 gate is not valid")
    recomputed_rows_sha = sha256_bytes(
        canonical_json_bytes(r1["execution"]["rows"])
    )
    if recomputed_rows_sha != r1["aggregate_rows_sha256"]:
        raise ValueError(f"{args.system} R1 aggregate row SHA mismatch")

    torch.cuda.reset_peak_memory_stats()
    if args.system == "simmark":
        output = run_simmark(args, identity, items, sources, r1)
    else:
        output = run_semstamp(args, identity, items, sources, r1)
    torch.cuda.synchronize()
    output["result_payload_sha256"] = sha256_bytes(canonical_json_bytes({
        key: value
        for key, value in output.items()
        if key != "result_payload_sha256"
    }))
    atomic_write_json(args.output, output)
    print(json.dumps({
        "stage": "complete",
        "system": args.system,
        "status": output["status"],
        "root_cause": output["root_cause"]["classification"],
        "result_payload_sha256": output["result_payload_sha256"],
        "output": str(args.output),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
