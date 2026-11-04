"""Preflight a local multi-instance AMR detector for SWAN."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import sys
import time

from feedback2_runtime import Heartbeat, claim
from run_confirmation_experiment import digest, freeze
from run_segmentation_intervention import ROOT, sha, text_sha, write_json


OUTPUT = ROOT / "experiments/swan_local_multi_instance2"
HUMAN = ROOT / "experiments/segmentation_intervention"
POSITIVE = ROOT / "experiments/swan_amr_composition_decomposition1"
BANK_PATH = ROOT / "third_party/SWAN/amr_bank/banks/amr_bank_50.json"
VECTORS_PATH = ROOT / "vectors/glove.6B.100d.txt"
S2MATCH_DIR = (
    ROOT
    / "third_party/SWAN/amr-metric-suite/py3-Smatch-and-S2match/smatch"
)
DEVELOPMENT_IDS = [f"c4-validation-{index}" for index in range(250, 265)]
HOLDOUT_IDS = [f"c4-validation-{index}" for index in range(265, 280)]
PREFILTER_RECALL = 0.50
MAX_OCCURRENCES = 4
CALIBRATION_MAX_FALSE_POSITIVES = 2
TEST_MAX_FALSE_POSITIVES = 5
POSITIVE_MIN_RECOVERY = 12
GRID = [
    {
        "coverage": coverage,
        "local_precision": precision,
        "min_active_nodes": nodes,
        "max_overlap": overlap,
    }
    for coverage, precision, nodes, overlap in itertools.product(
        (0.70, 0.80, 0.90),
        (0.70, 0.80, 0.90),
        (2, 3),
        (0.0, 0.5),
    )
]
_RUNTIME = {}


def configure_environment():
    os.environ.setdefault("GLOVE_VECTORS_PATH", str(VECTORS_PATH))


def load_s2match():
    configure_environment()
    if str(S2MATCH_DIR) not in sys.path:
        sys.path.insert(0, str(S2MATCH_DIR))
    import s2match

    return s2match


def dependency_paths():
    return [
        Path(__file__),
        ROOT / "scripts/audit_swan_local_multi_instance.py",
        OUTPUT / "protocol.md",
        BANK_PATH,
        VECTORS_PATH,
        ROOT / "third_party/SWAN/utils/s2match_patch.py",
        S2MATCH_DIR / "s2match.py",
        S2MATCH_DIR / "amr_py3.py",
        HUMAN / "manifest.json",
        HUMAN / "summary.json",
        POSITIVE / "manifest.json",
        POSITIVE / "summary.json",
        POSITIVE / "audit.json",
        POSITIVE / "recovery_verification.json",
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


def bank_templates():
    rows = json.loads(BANK_PATH.read_text())
    templates = [row["normalized_template_amr"] for row in rows]
    if len(templates) != 50 or len(set(templates)) != 50:
        raise ValueError("Expected 50 unique bank templates")
    return templates


def parse_graph(amr_string, prefix, s2match):
    graph = s2match.amr.AMR.parse_AMR_line(amr_string, False)
    if graph is None:
        raise ValueError("Invalid AMR")
    graph.rename_node(prefix)
    instances, attributes, relations = graph.get_triples()
    return {
        "instances": instances,
        "attributes": attributes,
        "relations": relations,
        "triples": len(instances) + len(attributes) + len(relations),
    }


def initialize_runtime():
    if _RUNTIME:
        return
    s2match = load_s2match()
    templates = bank_templates()
    vectors = s2match.load_vecs(str(VECTORS_PATH))
    _RUNTIME.update(
        {
            "s2match": s2match,
            "vectors": vectors,
            "simfun": s2match.get_sim_fun("cosine"),
            "templates": [
                parse_graph(template, "b", s2match)
                for template in templates
            ],
        }
    )


def graph_inventory(graph):
    return {
        "graph_id": graph["graph_id"],
        "amr_sha256": text_sha(graph["amr"]),
        "bank_scores_sha256": digest(graph["bank_scores"]),
        "known_target_template_indices": graph[
            "known_target_template_indices"
        ],
    }


def human_payloads(cohort):
    paths = sorted((HUMAN / "detected").glob(f"{cohort}-*.json"))
    payloads = []
    for path in paths:
        saved = json.loads(path.read_text())
        record = saved["record"]
        if record["cohort"] != cohort or record["id"] != path.stem:
            raise ValueError("Human checkpoint identity mismatch")
        graphs = []
        for index, detail in enumerate(
            saved["variants"]["sentence"]["details"]
        ):
            graphs.append(
                {
                    "graph_id": f"sentence-{index}",
                    "amr": detail["amr"],
                    "bank_scores": detail["bank_scores"],
                    "known_target_template_indices": [],
                }
            )
        payloads.append(
            {
                "document_id": record["id"],
                "cohort": cohort,
                "representation": "sentence",
                "source_files": {
                    str(path.relative_to(ROOT)): sha(path),
                },
                "graphs": graphs,
            }
        )
    if len(payloads) != 250:
        raise ValueError(f"Expected 250 records for {cohort}")
    return payloads


def positive_payloads(document_ids):
    manifest = json.loads((POSITIVE / "manifest.json").read_text())
    by_document = {document_id: [] for document_id in document_ids}
    for source in manifest["sources"]:
        if source["prompt_id"] in by_document:
            by_document[source["prompt_id"]].append(source)
    payloads = []
    for document_id in document_ids:
        sources = sorted(
            by_document[document_id],
            key=lambda row: row["pair_index"],
        )
        if len(sources) != 2:
            raise ValueError("Positive pair inventory mismatch")
        condition_graphs = {
            "original": [],
            "composition": [],
            "reparse": [],
        }
        source_files = {}
        for source in sources:
            baseline_path = ROOT / source["baseline_path"]
            pair_score_path = ROOT / source["pair_score_path"]
            composition_path = (
                POSITIVE
                / "scores"
                / f"{document_id}-{source['pair_index']}.json"
            )
            for path in (
                baseline_path,
                pair_score_path,
                composition_path,
            ):
                source_files[str(path.relative_to(ROOT))] = sha(path)
            baseline = json.loads(baseline_path.read_text())
            pair_score = json.loads(pair_score_path.read_text())
            composition = json.loads(composition_path.read_text())
            for component, slot in zip(
                source["components"],
                source["source_slots"],
            ):
                detail = baseline["detection"]["details"][slot]
                condition_graphs["original"].append(
                    {
                        "graph_id": (
                            f"pair-{source['pair_index']}-slot-{slot}"
                        ),
                        "amr": detail["amr"],
                        "bank_scores": detail["bank_scores"],
                        "known_target_template_indices": (
                            [component["chosen_template_index"]]
                            if component["accepted"]
                            else []
                        ),
                    }
                )
            accepted = sorted(
                component["chosen_template_index"]
                for component in source["components"]
                if component["accepted"]
            )
            condition_graphs["composition"].append(
                {
                    "graph_id": f"pair-{source['pair_index']}",
                    "amr": composition["composed_amr"],
                    "bank_scores": composition["bank"][
                        "composition_scores"
                    ],
                    "known_target_template_indices": accepted,
                }
            )
            detail = pair_score["detail"]
            condition_graphs["reparse"].append(
                {
                    "graph_id": f"pair-{source['pair_index']}",
                    "amr": detail["amr"],
                    "bank_scores": detail["bank_scores"],
                    "known_target_template_indices": accepted,
                }
            )
        cohort = (
            "positive_development"
            if document_id in DEVELOPMENT_IDS
            else "positive_holdout"
        )
        for representation, graphs in condition_graphs.items():
            payloads.append(
                {
                    "document_id": document_id,
                    "cohort": cohort,
                    "representation": representation,
                    "source_files": source_files,
                    "graphs": graphs,
                }
            )
    return payloads


def payload_inventory(payload):
    return {
        "document_id": payload["document_id"],
        "cohort": payload["cohort"],
        "representation": payload["representation"],
        "source_files": payload["source_files"],
        "graphs": [graph_inventory(graph) for graph in payload["graphs"]],
    }


def source_inventory():
    payloads = (
        human_payloads("human_calibration")
        + human_payloads("human_test")
        + positive_payloads(DEVELOPMENT_IDS + HOLDOUT_IDS)
    )
    counts = {}
    for payload in payloads:
        key = f"{payload['cohort']}:{payload['representation']}"
        counts[key] = counts.get(key, 0) + 1
    expected = {
        "human_calibration:sentence": 250,
        "human_test:sentence": 250,
        "positive_development:original": 15,
        "positive_development:composition": 15,
        "positive_development:reparse": 15,
        "positive_holdout:original": 15,
        "positive_holdout:composition": 15,
        "positive_holdout:reparse": 15,
    }
    if counts != expected:
        raise ValueError("Source cohort inventory mismatch")
    return [payload_inventory(payload) for payload in payloads]


def prepare(folder=OUTPUT):
    path = folder / "manifest.json"
    if path.exists():
        return check(folder)
    manifest = {
        "status": "frozen_before_alignment_extraction",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "workflow": "swan_local_multi_instance_preflight",
            "bank_templates": 50,
            "prefilter_template_recall": PREFILTER_RECALL,
            "max_disjoint_occurrences_per_graph_template": (
                MAX_OCCURRENCES
            ),
            "grid": GRID,
            "calibration_documents": 250,
            "calibration_max_false_positives": (
                CALIBRATION_MAX_FALSE_POSITIVES
            ),
            "positive_development_documents": 15,
            "human_test_documents": 250,
            "human_test_max_false_positives": (
                TEST_MAX_FALSE_POSITIVES
            ),
            "positive_holdout_documents": 15,
            "positive_min_recovery": POSITIVE_MIN_RECOVERY,
            "positive_representations": [
                "original",
                "composition",
                "reparse",
            ],
            "new_api_calls": 0,
            "new_parser_calls": 0,
        },
        "bank_template_sha256": [
            text_sha(template) for template in bank_templates()
        ],
        "development_ids": DEVELOPMENT_IDS,
        "holdout_ids": HOLDOUT_IDS,
        "sources": source_inventory(),
        "dependencies": dependencies(),
        "packages": package_versions(),
    }
    freeze(path, manifest)
    return manifest


def check(folder=OUTPUT):
    manifest = json.loads((folder / "manifest.json").read_text())
    protocol = manifest["protocol"]
    if (
        protocol["workflow"] != "swan_local_multi_instance_preflight"
        or protocol["grid"] != GRID
        or manifest["development_ids"] != DEVELOPMENT_IDS
        or manifest["holdout_ids"] != HOLDOUT_IDS
    ):
        raise ValueError("Unexpected frozen workflow")
    for relative, expected in manifest["dependencies"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Frozen dependency changed: {relative}")
    if package_versions() != manifest["packages"]:
        raise ValueError("Frozen package versions changed")
    if manifest["bank_template_sha256"] != [
        text_sha(template) for template in bank_templates()
    ]:
        raise ValueError("Bank inventory changed")
    if manifest["sources"] != source_inventory():
        raise ValueError("Source inventory changed")
    return manifest


def checkpoint_path(folder, payload):
    return (
        folder
        / "checkpoints"
        / payload["cohort"]
        / payload["representation"]
        / f"{payload['document_id']}.json"
    )


def payload_identity(payload):
    return digest(payload_inventory(payload))


def node_index(node, prefix):
    if not node.startswith(prefix):
        raise ValueError("Unexpected renamed node")
    return int(node[len(prefix):])


def best_match_with_pool(
    s2match,
    candidate_mappings,
    weight_dict,
    instance1,
    instance2,
):
    s2match.match_triple_dict = {}
    best_mapping = [-1] * len(instance1)
    best_match_num = 0.0
    for iteration in range(s2match.iteration_num):
        if iteration == 0:
            current = s2match.smart_init_mapping(
                candidate_mappings,
                instance1,
                instance2,
            )
        else:
            current = s2match.random_init_mapping(candidate_mappings)
        match_num = s2match.compute_match(current, weight_dict)
        while True:
            gain, new_mapping = s2match.get_best_gain(
                current,
                candidate_mappings,
                weight_dict,
                len(instance2),
                match_num,
            )
            if gain <= 1e-10:
                break
            match_num += gain
            current = new_mapping[:]
        if match_num > best_match_num:
            best_mapping = current[:]
            best_match_num = match_num
    s2match.match_triple_dict.clear()
    return best_mapping, best_match_num


def active_nodes(mapping, weight_dict):
    active = set()
    for candidate_index, template_index in enumerate(mapping):
        if template_index < 0:
            continue
        pair = (candidate_index, template_index)
        weights = weight_dict.get(pair, {})
        if weights.get(-1, 0) > 1e-12:
            active.add(candidate_index)
        for other, value in weights.items():
            if (
                other != -1
                and value > 1e-12
                and mapping[other[0]] == other[1]
            ):
                active.add(candidate_index)
                active.add(other[0])
    return active


def mask_weight_dict(weight_dict, blocked):
    masked = {}
    for pair, weights in weight_dict.items():
        if pair[0] in blocked:
            continue
        retained = {
            key: value
            for key, value in weights.items()
            if key == -1 or key[0] not in blocked
        }
        if retained:
            masked[pair] = retained
    return masked


def local_triple_count(parsed, active):
    return (
        sum(
            node_index(node, "a") in active
            for _, node, _ in parsed["instances"]
        )
        + sum(
            node_index(node, "a") in active
            for _, node, _ in parsed["attributes"]
        )
        + sum(
            node_index(source, "a") in active
            and node_index(target, "a") in active
            for _, source, target in parsed["relations"]
        )
    )


def harmonic(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_graph(graph):
    initialize_runtime()
    s2match = _RUNTIME["s2match"]
    parsed = parse_graph(graph["amr"], "a", s2match)
    scores = graph["bank_scores"]
    if (
        len(scores) != 50
        or not all(
            isinstance(value, (int, float))
            and math.isfinite(value)
            and 0 <= value <= 1
            for value in scores
        )
    ):
        raise ValueError("Invalid saved bank scores")
    candidates = []
    screened_templates = 0
    hill_climbs = 0
    for bank_index, (saved_f1, template) in enumerate(
        zip(scores, _RUNTIME["templates"])
    ):
        saved_recall = (
            saved_f1 * (parsed["triples"] + template["triples"])
            / (2 * template["triples"])
        )
        if saved_recall + 1e-12 < PREFILTER_RECALL:
            continue
        screened_templates += 1
        candidate_mappings, weight_dict = s2match.compute_pool(
            parsed["instances"],
            parsed["attributes"],
            parsed["relations"],
            template["instances"],
            template["attributes"],
            template["relations"],
            "a",
            "b",
            _RUNTIME["vectors"],
            0.5,
            0.5,
            _RUNTIME["simfun"],
            "split",
        )
        blocked = set()
        for occurrence in range(MAX_OCCURRENCES):
            remaining = [
                set() if index in blocked else set(values)
                for index, values in enumerate(candidate_mappings)
            ]
            remaining_weights = mask_weight_dict(weight_dict, blocked)
            seed_text = (
                f"{text_sha(graph['amr'])}:{bank_index}:{occurrence}"
            )
            random.seed(
                int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)
            )
            mapping, match_num = best_match_with_pool(
                s2match,
                remaining,
                remaining_weights,
                parsed["instances"],
                template["instances"],
            )
            hill_climbs += 1
            active = active_nodes(mapping, remaining_weights)
            if not active or match_num <= 1e-12:
                break
            local_count = local_triple_count(parsed, active)
            if local_count <= 0:
                raise ValueError("Positive match has no local triples")
            coverage = match_num / template["triples"]
            precision = match_num / local_count
            if (
                not math.isfinite(coverage)
                or not math.isfinite(precision)
                or coverage < 0
                or precision < 0
                or coverage > 1 + 1e-9
                or precision > 1 + 1e-9
            ):
                raise ValueError("Invalid local alignment metric")
            if coverage + 1e-12 >= PREFILTER_RECALL:
                candidates.append(
                    {
                        "bank_index": bank_index,
                        "occurrence": occurrence,
                        "soft_match": match_num,
                        "template_triples": template["triples"],
                        "local_candidate_triples": local_count,
                        "coverage": coverage,
                        "local_precision": precision,
                        "local_f1": harmonic(precision, coverage),
                        "active_nodes": sorted(active),
                    }
                )
            blocked.update(active)
    return {
        "graph_id": graph["graph_id"],
        "amr_sha256": text_sha(graph["amr"]),
        "bank_scores_sha256": digest(scores),
        "candidate_triples": parsed["triples"],
        "known_target_template_indices": graph[
            "known_target_template_indices"
        ],
        "screened_templates": screened_templates,
        "hill_climbs": hill_climbs,
        "candidates": candidates,
    }


def score_payload(payload):
    started = time.monotonic()
    graphs = [score_graph(graph) for graph in payload["graphs"]]
    return {
        "document_id": payload["document_id"],
        "cohort": payload["cohort"],
        "representation": payload["representation"],
        "source_identity": payload_identity(payload),
        "graphs": graphs,
        "screened_templates": sum(
            graph["screened_templates"] for graph in graphs
        ),
        "hill_climbs": sum(graph["hill_climbs"] for graph in graphs),
        "elapsed_seconds": time.monotonic() - started,
    }


def run_payloads(folder, payloads, workers, heartbeat):
    completed = []
    missing = []
    for payload in payloads:
        path = checkpoint_path(folder, payload)
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["source_identity"] != payload_identity(payload):
                raise ValueError("Checkpoint source identity changed")
            completed.append(saved)
        else:
            missing.append(payload)
    if not missing:
        return completed
    initialize_runtime()
    worker_count = max(1, min(workers, len(missing)))
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
    ) as pool:
        futures = {
            pool.submit(score_payload, payload): payload
            for payload in missing
        }
        done = len(completed)
        for future in as_completed(futures):
            payload = futures[future]
            result = future.result()
            freeze(checkpoint_path(folder, payload), result)
            completed.append(result)
            done += 1
            heartbeat.update(
                stage="alignment_extraction",
                cohort=payload["cohort"],
                representation=payload["representation"],
                completed_documents=done,
                total_documents=len(payloads),
            )
    return completed


def rule_id(rule):
    return (
        f"c{rule['coverage']:.2f}-p{rule['local_precision']:.2f}"
        f"-n{rule['min_active_nodes']}-o{rule['max_overlap']:.2f}"
    )


def overlap_coefficient(first, second):
    first, second = set(first), set(second)
    if not first or not second:
        return 0.0
    return len(first & second) / min(len(first), len(second))


def select_graph_evidence(graph, rule):
    candidates = [
        row
        for row in graph["candidates"]
        if (
            row["coverage"] + 1e-12 >= rule["coverage"]
            and row["local_precision"] + 1e-12
            >= rule["local_precision"]
            and len(row["active_nodes"]) >= rule["min_active_nodes"]
        )
    ]
    candidates.sort(
        key=lambda row: (
            -row["local_f1"],
            -row["coverage"],
            -row["local_precision"],
            -len(row["active_nodes"]),
            row["bank_index"],
            row["occurrence"],
        )
    )
    selected = []
    for row in candidates:
        if all(
            overlap_coefficient(
                row["active_nodes"],
                prior["active_nodes"],
            )
            <= rule["max_overlap"] + 1e-12
            for prior in selected
        ):
            selected.append(row)
    return selected


def document_evidence(checkpoint, rule):
    selected = []
    targets = 0
    recovered = 0
    for graph in checkpoint["graphs"]:
        rows = select_graph_evidence(graph, rule)
        selected.extend(
            {"graph_id": graph["graph_id"], **row}
            for row in rows
        )
        known = graph["known_target_template_indices"]
        targets += len(known)
        chosen_indices = [row["bank_index"] for row in rows]
        recovered += sum(index in chosen_indices for index in known)
    return {
        "count": len(selected),
        "selected": selected,
        "known_targets": targets,
        "recovered_target_templates": recovered,
    }


def calibrated_threshold(counts):
    for threshold in range(1, max(counts, default=0) + 2):
        false_positives = sum(count >= threshold for count in counts)
        if false_positives <= CALIBRATION_MAX_FALSE_POSITIVES:
            return threshold, false_positives
    raise AssertionError("Unreachable calibration threshold")


def condition_result(checkpoints, rule, threshold):
    rows = []
    for checkpoint in sorted(
        checkpoints,
        key=lambda row: row["document_id"],
    ):
        evidence = document_evidence(checkpoint, rule)
        rows.append(
            {
                "document_id": checkpoint["document_id"],
                "evidence_count": evidence["count"],
                "detected": evidence["count"] >= threshold,
                "known_targets": evidence["known_targets"],
                "recovered_target_templates": evidence[
                    "recovered_target_templates"
                ],
            }
        )
    return {
        "documents": len(rows),
        "detected": sum(row["detected"] for row in rows),
        "evidence_counts": [row["evidence_count"] for row in rows],
        "known_targets": sum(row["known_targets"] for row in rows),
        "recovered_target_templates": sum(
            row["recovered_target_templates"] for row in rows
        ),
        "rows": rows,
    }


def development_rank(row):
    recoveries = row["positive_development"]
    return (
        min(recoveries["original"], recoveries["reparse"]),
        recoveries["reparse"],
        recoveries["original"],
        recoveries["composition"],
        -row["calibration_false_positives"],
        row["rule"]["coverage"],
        row["rule"]["local_precision"],
        row["rule"]["min_active_nodes"],
        -row["rule"]["max_overlap"],
    )


def load_checkpoints(folder, payloads):
    checkpoints = []
    for payload in payloads:
        path = checkpoint_path(folder, payload)
        if not path.exists():
            raise ValueError(f"Missing checkpoint: {path}")
        saved = json.loads(path.read_text())
        if saved["source_identity"] != payload_identity(payload):
            raise ValueError("Checkpoint source identity changed")
        checkpoints.append(saved)
    return checkpoints


def develop(folder=OUTPUT, workers=8):
    check(folder)
    if (
        (folder / "development.json").exists()
        and (folder / "locked_rule.json").exists()
    ):
        locked = json.loads((folder / "locked_rule.json").read_text())
        if (
            locked["manifest_sha256"] != sha(folder / "manifest.json")
            or locked["development_sha256"]
            != sha(folder / "development.json")
        ):
            raise ValueError("Existing locked rule identity mismatch")
        return locked
    human_payload = human_payloads("human_calibration")
    positive_payload = positive_payloads(DEVELOPMENT_IDS)
    with claim(folder / "run.lock"), Heartbeat(folder) as heartbeat:
        human = run_payloads(
            folder,
            human_payload,
            workers,
            heartbeat,
        )
        positive = run_payloads(
            folder,
            positive_payload,
            workers,
            heartbeat,
        )
    by_condition = {
        condition: [
            row
            for row in positive
            if row["representation"] == condition
        ]
        for condition in ("original", "composition", "reparse")
    }
    grid_rows = []
    for rule in GRID:
        counts = [
            document_evidence(checkpoint, rule)["count"]
            for checkpoint in human
        ]
        threshold, false_positives = calibrated_threshold(counts)
        recoveries = {
            condition: condition_result(
                by_condition[condition],
                rule,
                threshold,
            )["detected"]
            for condition in by_condition
        }
        grid_rows.append(
            {
                "rule_id": rule_id(rule),
                "rule": rule,
                "document_threshold": threshold,
                "calibration_false_positives": false_positives,
                "positive_development": recoveries,
            }
        )
    selected = max(grid_rows, key=development_rank)
    development_result = {
        "status": "development_complete",
        "manifest_sha256": sha(folder / "manifest.json"),
        "grid_rows": grid_rows,
        "selected_rule_id": selected["rule_id"],
        "integrity": {
            "human_calibration_documents": len(human),
            "positive_development_checkpoints": len(positive),
            "screened_templates": sum(
                row["screened_templates"] for row in human + positive
            ),
            "hill_climbs": sum(
                row["hill_climbs"] for row in human + positive
            ),
            "new_api_calls": 0,
            "new_parser_calls": 0,
        },
    }
    freeze(folder / "development.json", development_result)
    locked = {
        "status": "locked_before_holdout_evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha(folder / "manifest.json"),
        "development_sha256": sha(folder / "development.json"),
        "rule_id": selected["rule_id"],
        "rule": selected["rule"],
        "document_threshold": selected["document_threshold"],
        "calibration_false_positives": selected[
            "calibration_false_positives"
        ],
        "positive_development": selected["positive_development"],
    }
    freeze(folder / "locked_rule.json", locked)
    return locked


def write_report(folder, summary):
    final = summary["locked_evaluation"]
    lines = [
        "# SWAN local multi-instance evidence preflight",
        "",
        f"Decision: `{summary['decision']}`.",
        "",
        "This is a defense feasibility preflight on previously observed, "
        "low-quality positive assets, not a production TPR estimate.",
        "",
        "## Locked rule",
        "",
        f"- Rule: `{summary['locked_rule']['rule_id']}`",
        f"- Document evidence threshold: "
        f"{summary['locked_rule']['document_threshold']}",
        "",
        "## Locked endpoints",
        "",
        "| Endpoint | Result | Gate |",
        "|---|---:|---:|",
        (
            "| Human test false positives | "
            f"{final['human_test']['detected']}/250 | <=5/250 |"
        ),
        (
            "| Original four-graph recovery | "
            f"{final['positive_holdout']['original']['detected']}/15 "
            "| >=12/15 |"
        ),
        (
            "| Merged-reparse two-graph recovery | "
            f"{final['positive_holdout']['reparse']['detected']}/15 "
            "| >=12/15 |"
        ),
        (
            "| Deterministic-composition recovery | "
            f"{final['positive_holdout']['composition']['detected']}/15 "
            "| diagnostic |"
        ),
        "",
        "No generation API, neural parser, injection, or attack calls were "
        "made.",
        "",
    ]
    write_json(folder / "summary.json", summary)
    path = folder / "report.md"
    content = "\n".join(lines)
    if path.exists() and path.read_text() != content:
        raise ValueError("Frozen report differs")
    if not path.exists():
        path.write_text(content)


def evaluate(folder=OUTPUT, workers=8):
    check(folder)
    locked = json.loads((folder / "locked_rule.json").read_text())
    if (
        locked["manifest_sha256"] != sha(folder / "manifest.json")
        or locked["development_sha256"]
        != sha(folder / "development.json")
    ):
        raise ValueError("Locked rule identity mismatch")
    human_payload = human_payloads("human_test")
    positive_payload = positive_payloads(HOLDOUT_IDS)
    with claim(folder / "run.lock"), Heartbeat(folder) as heartbeat:
        human = run_payloads(
            folder,
            human_payload,
            workers,
            heartbeat,
        )
        positive = run_payloads(
            folder,
            positive_payload,
            workers,
            heartbeat,
        )
    rule = locked["rule"]
    threshold = locked["document_threshold"]
    human_result = condition_result(human, rule, threshold)
    positive_result = {
        condition: condition_result(
            [
                row
                for row in positive
                if row["representation"] == condition
            ],
            rule,
            threshold,
        )
        for condition in ("original", "composition", "reparse")
    }
    passed = (
        human_result["detected"] <= TEST_MAX_FALSE_POSITIVES
        and positive_result["original"]["detected"]
        >= POSITIVE_MIN_RECOVERY
        and positive_result["reparse"]["detected"]
        >= POSITIVE_MIN_RECOVERY
    )
    summary = {
        "status": "complete_local_multi_instance_preflight",
        "decision": (
            "local_multi_instance_feasible"
            if passed
            else "stop_current_local_multi_instance_rule"
        ),
        "manifest_sha256": sha(folder / "manifest.json"),
        "development_sha256": sha(folder / "development.json"),
        "locked_rule_sha256": sha(folder / "locked_rule.json"),
        "locked_rule": locked,
        "locked_evaluation": {
            "human_test": human_result,
            "positive_holdout": positive_result,
        },
        "gates": {
            "human_test_pass": (
                human_result["detected"] <= TEST_MAX_FALSE_POSITIVES
            ),
            "original_recovery_pass": (
                positive_result["original"]["detected"]
                >= POSITIVE_MIN_RECOVERY
            ),
            "reparse_recovery_pass": (
                positive_result["reparse"]["detected"]
                >= POSITIVE_MIN_RECOVERY
            ),
            "all_pass": passed,
        },
        "integrity": {
            "human_test_documents": len(human),
            "positive_holdout_checkpoints": len(positive),
            "screened_templates": sum(
                row["screened_templates"] for row in human + positive
            ),
            "hill_climbs": sum(
                row["hill_climbs"] for row in human + positive
            ),
            "new_api_calls": 0,
            "new_parser_calls": 0,
        },
        "limitations": [
            "Positive assets are low quality and were observed previously.",
            "The holdout locks local-detector tuning but is not a fresh corpus.",
            "The saved-score prefilter is an offline computational shortcut.",
            "Results are a feasibility preflight, not production detector TPR.",
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
        result = develop(args.output, args.workers)
    elif args.command == "evaluate":
        result = evaluate(args.output, args.workers)
    else:
        prepare(args.output)
        develop(args.output, args.workers)
        result = evaluate(args.output, args.workers)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
