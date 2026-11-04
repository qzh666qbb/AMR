"""Preflight template-specific background calibration for SWAN evidence."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path

from feedback2_runtime import Heartbeat, claim
from run_confirmation_experiment import digest, freeze
from run_segmentation_intervention import ROOT, sha, write_json
import run_swan_local_multi_instance as local


OUTPUT = ROOT / "experiments/swan_template_specificity1"
SOURCE = local.OUTPUT
ATTACK_SOURCE = ROOT / "experiments/segmentation_intervention"
FEATURES = ("local_f1", "coverage", "balanced_min")
TOP_KS = (1, 2, 4)
GRID = [
    {"feature": feature, "top_k": top_k}
    for feature in FEATURES
    for top_k in TOP_KS
]
CALIBRATION_DOCUMENTS = 250
CALIBRATION_MAX_FALSE_POSITIVES = 2
DEVELOPMENT_MIN_DETECTIONS = 8
TEST_MAX_FALSE_POSITIVES = 5
HOLDOUT_MIN_DETECTIONS = 12
ATTACK_MIN_DETECTIONS = 18
ATTACK_REPRESENTATIONS = ("sentence", "clause")
FEATURE_PREFERENCE = {
    "coverage": 0,
    "local_f1": 1,
    "balanced_min": 2,
}


def dependency_paths():
    return [
        Path(__file__),
        ROOT / "scripts/audit_swan_template_specificity.py",
        OUTPUT / "protocol.md",
        ROOT / "scripts/run_swan_local_multi_instance.py",
        ROOT / "scripts/audit_swan_local_multi_instance.py",
        SOURCE / "manifest.json",
        SOURCE / "summary.json",
        SOURCE / "audit.json",
        SOURCE / "recovery_verification.json",
        ATTACK_SOURCE / "manifest.json",
        ATTACK_SOURCE / "summary.json",
        local.BANK_PATH,
        local.VECTORS_PATH,
        ROOT / "third_party/SWAN/utils/s2match_patch.py",
        local.S2MATCH_DIR / "s2match.py",
        local.S2MATCH_DIR / "amr_py3.py",
    ]


def dependencies():
    return {
        str(path.relative_to(ROOT)): sha(path)
        for path in dependency_paths()
    }


def package_versions():
    return {
        name: importlib.metadata.version(name)
        for name in ("numpy", "scipy")
    }


def reused_payloads():
    return (
        local.human_payloads("human_calibration")
        + local.human_payloads("human_test")
        + local.positive_payloads(
            local.DEVELOPMENT_IDS + local.HOLDOUT_IDS
        )
    )


def source_checkpoint_inventory():
    groups = {}
    for payload in reused_payloads():
        key = f"{payload['cohort']}:{payload['representation']}"
        path = local.checkpoint_path(SOURCE, payload)
        if not path.exists():
            raise ValueError(f"Missing source checkpoint: {path}")
        groups.setdefault(key, []).append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha(path),
            }
        )
    return {
        key: {
            "count": len(rows),
            "aggregate_sha256": digest(sorted(rows, key=lambda row: row["path"])),
        }
        for key, rows in sorted(groups.items())
    }


def attack_inventory():
    manifest = json.loads((ATTACK_SOURCE / "manifest.json").read_text())
    summary = json.loads((ATTACK_SOURCE / "summary.json").read_text())
    records = {
        row["id"]: row
        for row in manifest["records"]
        if row["cohort"] == "attack"
    }
    pairs = {row["id"]: row for row in summary["pairs"]}
    threshold = summary["calibrated"]["sentence"]["threshold"]
    inventory = []
    for document_id in sorted(records):
        path = ATTACK_SOURCE / "detected" / f"{document_id}.json"
        saved = json.loads(path.read_text())
        record = records[document_id]
        pair = pairs[document_id]
        if saved["record"] != record:
            raise ValueError("Attack manifest/checkpoint identity mismatch")
        inventory.append(
            {
                "document_id": document_id,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha(path),
                "quality_valid": record["quality_valid"],
                "historical_sentence_z": pair["sentence_z"],
                "historically_successful": (
                    pair["sentence_z"] <= threshold
                ),
            }
        )
    if (
        len(inventory) != 30
        or sum(row["quality_valid"] for row in inventory) != 24
        or sum(
            row["quality_valid"] and row["historically_successful"]
            for row in inventory
        )
        != 23
    ):
        raise ValueError("Unexpected attack challenge inventory")
    return inventory


def source_guard():
    local.check(SOURCE)
    audit = json.loads((SOURCE / "audit.json").read_text())
    recovery = json.loads(
        (SOURCE / "recovery_verification.json").read_text()
    )
    if audit["status"] != "passed" or recovery["status"] != "passed":
        raise ValueError("Source local-alignment audit is not passed")


def prepare(folder=OUTPUT):
    path = folder / "manifest.json"
    if path.exists():
        return check(folder)
    source_guard()
    manifest = {
        "status": "frozen_before_new_statistic_development",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "workflow": "swan_template_specificity_preflight",
            "features": list(FEATURES),
            "top_k": list(TOP_KS),
            "grid": GRID,
            "template_count": 50,
            "calibration_documents": CALIBRATION_DOCUMENTS,
            "tail_probability": (
                "min(1, (1 + count(reference >= score)) / 250)"
            ),
            "calibration_scoring": "leave_one_out",
            "statistic": "sum_top_k_negative_natural_log_tail_p",
            "detection_comparison": "strict_greater_than",
            "calibration_max_false_positives": (
                CALIBRATION_MAX_FALSE_POSITIVES
            ),
            "development_min_detections_original_and_reparse": (
                DEVELOPMENT_MIN_DETECTIONS
            ),
            "test_max_false_positives": TEST_MAX_FALSE_POSITIVES,
            "holdout_min_detections_original_and_reparse": (
                HOLDOUT_MIN_DETECTIONS
            ),
            "quality_valid_attack_sentence_min_detections": (
                ATTACK_MIN_DETECTIONS
            ),
            "attack_representations": list(ATTACK_REPRESENTATIONS),
            "new_api_calls": 0,
            "new_parser_calls": 0,
            "new_s2match_calls_before_lock": 0,
        },
        "development_ids": local.DEVELOPMENT_IDS,
        "holdout_ids": local.HOLDOUT_IDS,
        "source_checkpoint_inventory": source_checkpoint_inventory(),
        "attack_inventory": attack_inventory(),
        "dependencies": dependencies(),
        "packages": package_versions(),
    }
    freeze(path, manifest)
    return manifest


def check(folder=OUTPUT):
    manifest = json.loads((folder / "manifest.json").read_text())
    protocol = manifest["protocol"]
    if (
        protocol["workflow"]
        != "swan_template_specificity_preflight"
        or protocol["grid"] != GRID
        or manifest["development_ids"] != local.DEVELOPMENT_IDS
        or manifest["holdout_ids"] != local.HOLDOUT_IDS
    ):
        raise ValueError("Unexpected frozen workflow")
    source_guard()
    if manifest["source_checkpoint_inventory"] != (
        source_checkpoint_inventory()
    ):
        raise ValueError("Source checkpoint inventory changed")
    if manifest["attack_inventory"] != attack_inventory():
        raise ValueError("Attack inventory changed")
    for relative, expected in manifest["dependencies"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Frozen dependency changed: {relative}")
    if package_versions() != manifest["packages"]:
        raise ValueError("Frozen package versions changed")
    return manifest


def load_reused(cohort, representations):
    if cohort.startswith("human_"):
        payloads = local.human_payloads(cohort)
    else:
        document_ids = (
            local.DEVELOPMENT_IDS
            if cohort == "positive_development"
            else local.HOLDOUT_IDS
        )
        payloads = [
            payload
            for payload in local.positive_payloads(document_ids)
            if payload["representation"] in representations
        ]
    checkpoints = local.load_checkpoints(SOURCE, payloads)
    return sorted(
        checkpoints,
        key=lambda row: (row["representation"], row["document_id"]),
    )


def feature_value(candidate, feature):
    if feature == "balanced_min":
        return min(candidate["coverage"], candidate["local_precision"])
    if feature not in ("local_f1", "coverage"):
        raise ValueError(f"Unknown feature: {feature}")
    return candidate[feature]


def document_template_scores(checkpoint, feature):
    scores = [0.0] * 50
    for graph in checkpoint["graphs"]:
        for candidate in graph["candidates"]:
            bank_index = candidate["bank_index"]
            scores[bank_index] = max(
                scores[bank_index],
                feature_value(candidate, feature),
            )
    return scores


def calibration_matrix(checkpoints, feature):
    ordered = sorted(checkpoints, key=lambda row: row["document_id"])
    if (
        len(ordered) != CALIBRATION_DOCUMENTS
        or len({row["document_id"] for row in ordered})
        != CALIBRATION_DOCUMENTS
        or any(
            row["cohort"] != "human_calibration"
            or row["representation"] != "sentence"
            for row in ordered
        )
    ):
        raise ValueError("Calibration checkpoint inventory mismatch")
    return ordered, [
        document_template_scores(checkpoint, feature)
        for checkpoint in ordered
    ]


def document_statistic(scores, matrix, top_k, leave_out=None):
    if len(scores) != 50 or len(matrix) != CALIBRATION_DOCUMENTS:
        raise ValueError("Statistic inventory mismatch")
    evidence = []
    for bank_index, score in enumerate(scores):
        count = sum(
            row[bank_index] >= score
            for index, row in enumerate(matrix)
            if index != leave_out
        )
        probability = min(
            1.0,
            (1 + count) / CALIBRATION_DOCUMENTS,
        )
        evidence.append(
            {
                "bank_index": bank_index,
                "feature_score": score,
                "tail_probability": probability,
                "surprisal": -math.log(probability),
            }
        )
    evidence.sort(
        key=lambda row: (-row["surprisal"], row["bank_index"])
    )
    selected = evidence[:top_k]
    return {
        "statistic": sum(row["surprisal"] for row in selected),
        "top_evidence": selected,
    }


def calibrated_threshold(statistics):
    if len(statistics) != CALIBRATION_DOCUMENTS:
        raise ValueError("Calibration statistic count mismatch")
    ordered = sorted(statistics, reverse=True)
    threshold = ordered[CALIBRATION_MAX_FALSE_POSITIVES]
    false_positives = sum(value > threshold for value in statistics)
    if false_positives > CALIBRATION_MAX_FALSE_POSITIVES:
        raise AssertionError("Calibration false-positive budget exceeded")
    return threshold, false_positives


def rule_id(rule):
    return f"{rule['feature']}-top{rule['top_k']}"


def score_checkpoint(checkpoint, feature, top_k, matrix, threshold):
    scores = document_template_scores(checkpoint, feature)
    result = document_statistic(scores, matrix, top_k)
    top_indices = {
        row["bank_index"] for row in result["top_evidence"]
    }
    known = [
        bank_index
        for graph in checkpoint["graphs"]
        for bank_index in graph["known_target_template_indices"]
    ]
    return {
        "document_id": checkpoint["document_id"],
        "statistic": result["statistic"],
        "threshold_margin": result["statistic"] - threshold,
        "detected": result["statistic"] > threshold,
        "top_evidence": result["top_evidence"],
        "known_target_instances": len(known),
        "known_target_instances_in_top_k": sum(
            bank_index in top_indices for bank_index in known
        ),
    }


def condition_result(checkpoints, feature, top_k, matrix, threshold):
    rows = [
        score_checkpoint(
            checkpoint,
            feature,
            top_k,
            matrix,
            threshold,
        )
        for checkpoint in sorted(
            checkpoints,
            key=lambda row: row["document_id"],
        )
    ]
    return {
        "documents": len(rows),
        "detected": sum(row["detected"] for row in rows),
        "known_target_instances": sum(
            row["known_target_instances"] for row in rows
        ),
        "known_target_instances_in_top_k": sum(
            row["known_target_instances_in_top_k"] for row in rows
        ),
        "rows": rows,
    }


def development_rank(row):
    positive = row["positive_development"]
    original = positive["original"]["detected"]
    reparse = positive["reparse"]["detected"]
    return (
        min(original, reparse),
        reparse,
        original,
        positive["composition"]["detected"],
        -row["calibration_false_positives"],
        FEATURE_PREFERENCE[row["rule"]["feature"]],
        -row["rule"]["top_k"],
    )


def calibration_statistics(matrix, top_k):
    return [
        document_statistic(
            scores,
            matrix,
            top_k,
            leave_out=index,
        )["statistic"]
        for index, scores in enumerate(matrix)
    ]


def build_development(calibration, positive):
    by_condition = {
        representation: [
            row
            for row in positive
            if row["representation"] == representation
        ]
        for representation in ("original", "composition", "reparse")
    }
    grid_rows = []
    for rule in GRID:
        _, matrix = calibration_matrix(calibration, rule["feature"])
        statistics = calibration_statistics(matrix, rule["top_k"])
        threshold, false_positives = calibrated_threshold(statistics)
        result = {
            representation: condition_result(
                checkpoints,
                rule["feature"],
                rule["top_k"],
                matrix,
                threshold,
            )
            for representation, checkpoints in by_condition.items()
        }
        grid_rows.append(
            {
                "rule_id": rule_id(rule),
                "rule": rule,
                "threshold": threshold,
                "calibration_false_positives": false_positives,
                "calibration_statistics": statistics,
                "positive_development": result,
            }
        )
    selected = max(grid_rows, key=development_rank)
    return grid_rows, selected


def write_report(folder, summary):
    selected_positive = summary["selected_rule"]["positive_development"]
    lines = [
        "# SWAN template-specific background calibration preflight",
        "",
        f"Decision: `{summary['decision']}`.",
        "",
        "This is a feasibility preflight on previously observed positive "
        "and attack assets, not a fully blind detector estimate.",
        "",
        "## Development",
        "",
        f"- Selected rule: `{summary['selected_rule']['rule_id']}`",
        (
            "- Positive development original/reparse: "
            f"{selected_positive['original']['detected']}"
            "/15 and "
            f"{selected_positive['reparse']['detected']}"
            "/15"
        ),
    ]
    if "locked_evaluation" in summary:
        final = summary["locked_evaluation"]
        attack_sentence = final["external_attack"]["sentence"]
        lines.extend(
            [
                "",
                "## Locked evaluation",
                "",
                "| Endpoint | Result | Gate |",
                "|---|---:|---:|",
                (
                    "| Human test false positives | "
                    f"{final['human_test']['detected']}/250 | <=5/250 |"
                ),
                (
                    "| Positive holdout original | "
                    f"{final['positive_holdout']['original']['detected']}/15 "
                    "| >=12/15 |"
                ),
                (
                    "| Positive holdout merged reparse | "
                    f"{final['positive_holdout']['reparse']['detected']}/15 "
                    "| >=12/15 |"
                ),
                (
                    "| Quality-valid attacks, sentence | "
                    f"{attack_sentence['quality_valid']['detected']}"
                    "/24 | >=18/24 |"
                ),
            ]
        )
    else:
        lines.extend(
            [
                "",
                "The development gate failed. Human test, positive holdout, "
                "and external attack statistics were not evaluated.",
            ]
        )
    lines.extend(
        [
            "",
            "No generation API or neural parser calls were made.",
            "",
        ]
    )
    write_json(folder / "summary.json", summary)
    content = "\n".join(lines)
    path = folder / "report.md"
    if path.exists() and path.read_text() != content:
        raise ValueError("Frozen report differs")
    if not path.exists():
        path.write_text(content)


def develop(folder=OUTPUT):
    check(folder)
    if (folder / "locked_rule.json").exists():
        return json.loads((folder / "locked_rule.json").read_text())
    if (folder / "stop_decision.json").exists():
        return json.loads((folder / "stop_decision.json").read_text())
    calibration = load_reused("human_calibration", ("sentence",))
    positive = load_reused(
        "positive_development",
        ("original", "composition", "reparse"),
    )
    grid_rows, selected = build_development(calibration, positive)
    gate_passed = (
        selected["positive_development"]["original"]["detected"]
        >= DEVELOPMENT_MIN_DETECTIONS
        and selected["positive_development"]["reparse"]["detected"]
        >= DEVELOPMENT_MIN_DETECTIONS
    )
    development = {
        "status": "development_complete",
        "manifest_sha256": sha(folder / "manifest.json"),
        "grid_rows": grid_rows,
        "selected_rule_id": selected["rule_id"],
        "development_gate": {
            "minimum_original_and_reparse": (
                DEVELOPMENT_MIN_DETECTIONS
            ),
            "passed": gate_passed,
        },
        "integrity": {
            "human_calibration_documents": len(calibration),
            "positive_development_checkpoints": len(positive),
            "new_api_calls": 0,
            "new_parser_calls": 0,
            "new_s2match_calls": 0,
        },
    }
    freeze(folder / "development.json", development)
    common = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha(folder / "manifest.json"),
        "development_sha256": sha(folder / "development.json"),
        "rule_id": selected["rule_id"],
        "rule": selected["rule"],
        "threshold": selected["threshold"],
        "calibration_false_positives": selected[
            "calibration_false_positives"
        ],
        "positive_development": selected["positive_development"],
    }
    if gate_passed:
        locked = {
            "status": "locked_before_any_heldout_or_attack_statistic",
            **common,
        }
        freeze(folder / "locked_rule.json", locked)
        return locked
    stopped = {
        "status": "stopped_at_development_gate",
        "decision": "stop_template_specificity_preflight",
        **common,
    }
    freeze(folder / "stop_decision.json", stopped)
    summary = {
        "status": "complete_development_stop",
        "decision": stopped["decision"],
        "manifest_sha256": sha(folder / "manifest.json"),
        "development_sha256": sha(folder / "development.json"),
        "stop_decision_sha256": sha(folder / "stop_decision.json"),
        "selected_rule": stopped,
        "gates": {"development_pass": False, "all_pass": False},
        "integrity": development["integrity"],
        "limitations": [
            "Stopped before all held-out and external-challenge statistics.",
            "Positive development assets were observed previously.",
        ],
    }
    write_report(folder, summary)
    return stopped


def attack_payloads():
    payloads = []
    for item in attack_inventory():
        path = ROOT / item["path"]
        saved = json.loads(path.read_text())
        for representation in ATTACK_REPRESENTATIONS:
            graphs = [
                {
                    "graph_id": f"{representation}-{index}",
                    "amr": detail["amr"],
                    "bank_scores": detail["bank_scores"],
                    "known_target_template_indices": [],
                }
                for index, detail in enumerate(
                    saved["variants"][representation]["details"]
                )
            ]
            payloads.append(
                {
                    "document_id": item["document_id"],
                    "cohort": "external_attack",
                    "representation": representation,
                    "source_files": {item["path"]: item["sha256"]},
                    "graphs": graphs,
                }
            )
    return payloads


def attack_condition_result(
    checkpoints,
    feature,
    top_k,
    matrix,
    threshold,
    inventory,
):
    result = condition_result(
        checkpoints,
        feature,
        top_k,
        matrix,
        threshold,
    )
    metadata = {row["document_id"]: row for row in inventory}
    for row in result["rows"]:
        item = metadata[row["document_id"]]
        row["quality_valid"] = item["quality_valid"]
        row["historically_successful"] = item[
            "historically_successful"
        ]

    def subgroup(predicate):
        rows = [row for row in result["rows"] if predicate(row)]
        return {
            "documents": len(rows),
            "detected": sum(row["detected"] for row in rows),
        }

    result["all"] = subgroup(lambda _: True)
    result["quality_valid"] = subgroup(
        lambda row: row["quality_valid"]
    )
    result["historically_successful_quality_valid"] = subgroup(
        lambda row: (
            row["quality_valid"] and row["historically_successful"]
        )
    )
    return result


def evaluate(folder=OUTPUT, workers=8):
    check(folder)
    if not (folder / "locked_rule.json").exists():
        raise ValueError("Development gate did not produce a locked rule")
    locked = json.loads((folder / "locked_rule.json").read_text())
    if (
        locked["manifest_sha256"] != sha(folder / "manifest.json")
        or locked["development_sha256"]
        != sha(folder / "development.json")
    ):
        raise ValueError("Locked rule identity mismatch")
    calibration = load_reused("human_calibration", ("sentence",))
    _, matrix = calibration_matrix(calibration, locked["rule"]["feature"])
    human_test = load_reused("human_test", ("sentence",))
    holdout = load_reused(
        "positive_holdout",
        ("original", "composition", "reparse"),
    )
    payloads = attack_payloads()
    with claim(folder / "run.lock"), Heartbeat(folder) as heartbeat:
        attacks = local.run_payloads(
            folder,
            payloads,
            workers,
            heartbeat,
        )
    feature = locked["rule"]["feature"]
    top_k = locked["rule"]["top_k"]
    threshold = locked["threshold"]
    human_result = condition_result(
        human_test,
        feature,
        top_k,
        matrix,
        threshold,
    )
    positive_result = {
        representation: condition_result(
            [
                row
                for row in holdout
                if row["representation"] == representation
            ],
            feature,
            top_k,
            matrix,
            threshold,
        )
        for representation in ("original", "composition", "reparse")
    }
    inventory = attack_inventory()
    attack_result = {
        representation: attack_condition_result(
            [
                row
                for row in attacks
                if row["representation"] == representation
            ],
            feature,
            top_k,
            matrix,
            threshold,
            inventory,
        )
        for representation in ATTACK_REPRESENTATIONS
    }
    gates = {
        "development_pass": True,
        "human_test_pass": (
            human_result["detected"] <= TEST_MAX_FALSE_POSITIVES
        ),
        "original_holdout_pass": (
            positive_result["original"]["detected"]
            >= HOLDOUT_MIN_DETECTIONS
        ),
        "reparse_holdout_pass": (
            positive_result["reparse"]["detected"]
            >= HOLDOUT_MIN_DETECTIONS
        ),
        "quality_valid_attack_sentence_pass": (
            attack_result["sentence"]["quality_valid"]["detected"]
            >= ATTACK_MIN_DETECTIONS
        ),
    }
    gates["all_pass"] = all(gates.values())
    summary = {
        "status": "complete_template_specificity_preflight",
        "decision": (
            "template_specificity_feasible"
            if gates["all_pass"]
            else "stop_template_specificity_rule"
        ),
        "manifest_sha256": sha(folder / "manifest.json"),
        "development_sha256": sha(folder / "development.json"),
        "locked_rule_sha256": sha(folder / "locked_rule.json"),
        "selected_rule": locked,
        "locked_evaluation": {
            "human_test": human_result,
            "positive_holdout": positive_result,
            "external_attack": attack_result,
        },
        "gates": gates,
        "integrity": {
            "human_test_documents": len(human_test),
            "positive_holdout_checkpoints": len(holdout),
            "external_attack_checkpoints": len(attacks),
            "external_attack_candidate_alignments": sum(
                len(graph["candidates"])
                for checkpoint in attacks
                for graph in checkpoint["graphs"]
            ),
            "new_api_calls": 0,
            "new_parser_calls": 0,
        },
        "limitations": [
            "The positive and external attack assets were observed previously.",
            "The empirical template tails use only 250 calibration documents.",
            "The saved-score prefilter is an offline computational shortcut.",
            "This is a feasibility preflight, not production detector TPR.",
        ],
    }
    write_report(folder, summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("prepare", "develop", "evaluate", "all"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "develop":
        result = develop(args.output)
    elif args.command == "evaluate":
        result = evaluate(args.output, args.workers)
    else:
        prepare(args.output)
        result = develop(args.output)
        if (args.output / "locked_rule.json").exists():
            result = evaluate(args.output, args.workers)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
