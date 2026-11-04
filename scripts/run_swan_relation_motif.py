"""Evaluate direct relation-anchored motif matching for SWAN templates."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib
import itertools
import json
import math
import multiprocessing
from pathlib import Path
import sys
import time

from feedback2_runtime import Heartbeat, claim
from run_confirmation_experiment import digest, freeze
from run_segmentation_intervention import ROOT, sha, text_sha, write_json
import run_swan_local_multi_instance as local


OUTPUT = ROOT / "experiments/swan_relation_motif1"
BANK_PATH = ROOT / "third_party/SWAN/amr_bank/banks/amr_bank_50.json"
AMR_DIR = (
    ROOT
    / "third_party/SWAN/amr-metric-suite/py3-Smatch-and-S2match/smatch"
)
ATTACK_SOURCE = ROOT / "experiments/segmentation_intervention"
FEATURES = (
    "relation_coverage",
    "motif_coverage",
    "idf_motif_coverage",
)
FEATURE_PREFERENCE = {
    "relation_coverage": 0,
    "motif_coverage": 1,
    "idf_motif_coverage": 2,
}
CANDIDATE_TOP_K = 5
CANDIDATE_MIN_ORIGINAL = 42
CANDIDATE_MIN_REPARSE = 34
DOCUMENT_TOP_KS = (1, 2, 4)
CALIBRATION_DOCUMENTS = 250
CALIBRATION_MAX_FALSE_POSITIVES = 2
DEVELOPMENT_MIN_DETECTIONS = 8
TEST_MAX_FALSE_POSITIVES = 5
HOLDOUT_MIN_DETECTIONS = 12
ATTACK_MIN_DETECTIONS = 18
REPRESENTATIONS = ("original", "composition", "reparse")
ATTACK_REPRESENTATIONS = ("sentence", "clause")
_RUNTIME = {}


def dependency_paths():
    return [
        Path(__file__),
        ROOT / "scripts/audit_swan_relation_motif.py",
        OUTPUT / "protocol.md",
        ROOT / "scripts/run_swan_local_multi_instance.py",
        BANK_PATH,
        AMR_DIR / "amr_py3.py",
        AMR_DIR / "helpers.py",
        local.HUMAN / "manifest.json",
        local.HUMAN / "summary.json",
        local.POSITIVE / "manifest.json",
        local.POSITIVE / "summary.json",
        local.POSITIVE / "audit.json",
    ]


def dependencies():
    return {
        str(path.relative_to(ROOT)): sha(path)
        for path in dependency_paths()
    }


def load_amr_parser():
    if str(AMR_DIR) not in sys.path:
        sys.path.insert(0, str(AMR_DIR))
    return importlib.import_module("amr_py3")


def bank_templates():
    rows = json.loads(BANK_PATH.read_text())
    templates = [row["normalized_template_amr"] for row in rows]
    if len(templates) != 50 or len(set(templates)) != 50:
        raise ValueError("Expected 50 unique bank templates")
    return templates


def parse_graph(amr_string):
    parser = load_amr_parser()
    graph = parser.AMR.parse_AMR_line(amr_string, False)
    if graph is None or not graph.nodes:
        raise ValueError("Invalid or empty AMR")
    index = {node: ordinal for ordinal, node in enumerate(graph.nodes)}
    concepts = [value.strip() for value in graph.node_values]
    relations = []
    attributes = []
    _, raw_attributes, raw_relations = graph.get_triples()
    for role, source, target in raw_relations:
        relations.append((role.strip(), index[source], index[target]))
    for role, node, value in raw_attributes:
        role = role.strip()
        if role != "TOP":
            attributes.append((role, index[node], value.strip()))
    relation_set = set(relations)
    attribute_set = set(attributes)
    if len(relation_set) != len(relations):
        raise ValueError("Duplicate semantic relation in parsed AMR")
    return {
        "root": index[graph.root],
        "concepts": concepts,
        "relations": relations,
        "attributes": attributes,
        "relation_set": relation_set,
        "attribute_set": attribute_set,
    }


def concept_compatible(template_concept, candidate_concept):
    return (
        template_concept == "X"
        or template_concept == candidate_concept
    )


def connected(node_count, relations, selected):
    adjacency = [set() for _ in range(node_count)]
    for edge_index in selected:
        _, source, target = relations[edge_index]
        adjacency[source].add(target)
        adjacency[target].add(source)
    seen = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adjacency[node])
    return len(seen) == node_count


def spanning_trees(graph):
    node_count = len(graph["concepts"])
    edge_count = len(graph["relations"])
    if edge_count < node_count - 1:
        raise ValueError("Template relation graph is disconnected")
    trees = [
        tuple(selected)
        for selected in itertools.combinations(
            range(edge_count),
            node_count - 1,
        )
        if connected(node_count, graph["relations"], selected)
    ]
    if not trees:
        raise ValueError("Template has no relation spanning tree")
    return trees


def rooted_children(graph, tree):
    adjacency = [[] for _ in graph["concepts"]]
    for edge_index in tree:
        _, source, target = graph["relations"][edge_index]
        adjacency[source].append((edge_index, target))
        adjacency[target].append((edge_index, source))
    children = [[] for _ in graph["concepts"]]
    seen = {graph["root"]}
    stack = [graph["root"]]
    while stack:
        node = stack.pop()
        for edge_index, neighbor in sorted(
            adjacency[node],
            reverse=True,
        ):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            children[node].append((edge_index, neighbor))
            stack.append(neighbor)
    if len(seen) != len(graph["concepts"]):
        raise ValueError("Spanning tree traversal missed a node")
    for rows in children:
        rows.sort()
    return children


def relation_signature(graph, edge_index):
    role, source, target = graph["relations"][edge_index]
    return [
        "relation",
        graph["concepts"][source],
        role,
        graph["concepts"][target],
    ]


def attribute_signature(graph, attribute_index):
    role, node, value = graph["attributes"][attribute_index]
    return [
        "attribute",
        graph["concepts"][node],
        role,
        value,
    ]


def wedge_arm(graph, edge_index, center):
    role, source, target = graph["relations"][edge_index]
    if source == center:
        return ["out", role, graph["concepts"][target]], target
    if target == center:
        return ["in", role, graph["concepts"][source]], source
    raise ValueError("Wedge edge is not incident on center")


def wedge_signature(graph, first, second, center):
    first_arm, first_neighbor = wedge_arm(graph, first, center)
    second_arm, second_neighbor = wedge_arm(graph, second, center)
    arms = sorted([first_arm, second_arm])
    return [
        "wedge",
        graph["concepts"][center],
        (
            "same_neighbor"
            if first_neighbor == second_neighbor
            else "distinct_neighbors"
        ),
        *arms,
    ]


def motif_units(graph):
    units = [
        {
            "kind": "relation",
            "edge": edge_index,
            "signature": relation_signature(graph, edge_index),
        }
        for edge_index in range(len(graph["relations"]))
    ]
    units.extend(
        {
            "kind": "attribute",
            "attribute": attribute_index,
            "signature": attribute_signature(graph, attribute_index),
        }
        for attribute_index in range(len(graph["attributes"]))
    )
    incident = [[] for _ in graph["concepts"]]
    for edge_index, (_, source, target) in enumerate(
        graph["relations"]
    ):
        incident[source].append(edge_index)
        incident[target].append(edge_index)
    for center, edges in enumerate(incident):
        for first, second in itertools.combinations(sorted(edges), 2):
            units.append(
                {
                    "kind": "wedge",
                    "edges": [first, second],
                    "center": center,
                    "signature": wedge_signature(
                        graph,
                        first,
                        second,
                        center,
                    ),
                }
            )
    return units


def prepare_template(amr_string, bank_index):
    graph = parse_graph(amr_string)
    trees = spanning_trees(graph)
    graph.update(
        {
            "bank_index": bank_index,
            "trees": trees,
            "tree_children": [
                rooted_children(graph, tree) for tree in trees
            ],
            "units": motif_units(graph),
        }
    )
    return graph


def signature_key(signature):
    return json.dumps(signature, separators=(",", ":"))


def arm_options(candidate, center, arm):
    direction, role, neighbor_type = arm
    options = []
    for edge_index, (candidate_role, source, target) in enumerate(
        candidate["relations"]
    ):
        if candidate_role != role:
            continue
        if direction == "out" and source == center:
            neighbor = target
        elif direction == "in" and target == center:
            neighbor = source
        else:
            continue
        if concept_compatible(
            neighbor_type,
            candidate["concepts"][neighbor],
        ):
            options.append((edge_index, neighbor))
    return options


def signature_present(candidate, signature):
    kind = signature[0]
    if kind == "relation":
        _, source_type, role, target_type = signature
        return any(
            candidate_role == role
            and concept_compatible(
                source_type,
                candidate["concepts"][source],
            )
            and concept_compatible(
                target_type,
                candidate["concepts"][target],
            )
            for candidate_role, source, target in candidate["relations"]
        )
    if kind == "attribute":
        _, node_type, role, value = signature
        return any(
            candidate_role == role
            and candidate_value == value
            and concept_compatible(
                node_type,
                candidate["concepts"][node],
            )
            for candidate_role, node, candidate_value
            in candidate["attributes"]
        )
    if kind != "wedge":
        raise ValueError(f"Unknown motif kind: {kind}")
    _, center_type, neighbor_rule, first_arm, second_arm = signature
    for center, concept in enumerate(candidate["concepts"]):
        if not concept_compatible(center_type, concept):
            continue
        first_options = arm_options(candidate, center, first_arm)
        second_options = arm_options(candidate, center, second_arm)
        for first_edge, first_neighbor in first_options:
            for second_edge, second_neighbor in second_options:
                if first_edge == second_edge:
                    continue
                same = first_neighbor == second_neighbor
                if same == (neighbor_rule == "same_neighbor"):
                    return True
    return False


def structural_inventory():
    templates = [
        prepare_template(amr, bank_index)
        for bank_index, amr in enumerate(bank_templates())
    ]
    return {
        "templates": len(templates),
        "node_range": [
            min(len(row["concepts"]) for row in templates),
            max(len(row["concepts"]) for row in templates),
        ],
        "relation_range": [
            min(len(row["relations"]) for row in templates),
            max(len(row["relations"]) for row in templates),
        ],
        "tree_templates": sum(len(row["trees"]) == 1 for row in templates),
        "unicyclic_templates": sum(
            len(row["relations"]) == len(row["concepts"])
            for row in templates
        ),
        "spanning_tree_counts": [
            len(row["trees"]) for row in templates
        ],
    }


def graph_inventory(graph):
    return {
        "graph_id": graph["graph_id"],
        "amr_sha256": text_sha(graph["amr"]),
        "known_target_template_indices": graph[
            "known_target_template_indices"
        ],
    }


def payload_inventory(payload):
    return {
        "document_id": payload["document_id"],
        "cohort": payload["cohort"],
        "representation": payload["representation"],
        "source_files": payload["source_files"],
        "graphs": [graph_inventory(graph) for graph in payload["graphs"]],
    }


def initial_payloads():
    return (
        local.human_payloads("human_calibration")
        + local.positive_payloads(local.DEVELOPMENT_IDS)
    )


def source_inventory(payloads):
    return [payload_inventory(payload) for payload in payloads]


def source_counts(payloads):
    counts = {}
    for payload in payloads:
        key = f"{payload['cohort']}:{payload['representation']}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def initial_source_guard(payloads):
    expected = {
        "human_calibration:sentence": 250,
        "positive_development:composition": 15,
        "positive_development:original": 15,
        "positive_development:reparse": 15,
    }
    if source_counts(payloads) != expected:
        raise ValueError("Initial source cohort inventory mismatch")


def prepare(folder=OUTPUT):
    path = folder / "manifest.json"
    if path.exists():
        return check(folder)
    payloads = initial_payloads()
    initial_source_guard(payloads)
    manifest = {
        "status": "frozen_before_motif_extraction",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "workflow": "swan_relation_motif_preflight",
            "features": list(FEATURES),
            "candidate_top_k": CANDIDATE_TOP_K,
            "candidate_min_original": CANDIDATE_MIN_ORIGINAL,
            "candidate_min_reparse": CANDIDATE_MIN_REPARSE,
            "document_top_k": list(DOCUMENT_TOP_KS),
            "calibration_documents": CALIBRATION_DOCUMENTS,
            "calibration_max_false_positives": (
                CALIBRATION_MAX_FALSE_POSITIVES
            ),
            "development_min_detections": (
                DEVELOPMENT_MIN_DETECTIONS
            ),
            "test_max_false_positives": TEST_MAX_FALSE_POSITIVES,
            "holdout_min_detections": HOLDOUT_MIN_DETECTIONS,
            "attack_min_detections": ATTACK_MIN_DETECTIONS,
            "new_api_calls": 0,
            "new_parser_calls": 0,
            "new_s2match_calls": 0,
        },
        "development_ids": local.DEVELOPMENT_IDS,
        "bank_template_sha256": [
            text_sha(template) for template in bank_templates()
        ],
        "structural_inventory": structural_inventory(),
        "source_counts": source_counts(payloads),
        "sources": source_inventory(payloads),
        "dependencies": dependencies(),
        "python": sys.version.split()[0],
    }
    freeze(path, manifest)
    return manifest


def check(folder=OUTPUT):
    manifest = json.loads((folder / "manifest.json").read_text())
    protocol = manifest["protocol"]
    if (
        protocol["workflow"] != "swan_relation_motif_preflight"
        or protocol["features"] != list(FEATURES)
        or protocol["candidate_top_k"] != CANDIDATE_TOP_K
        or protocol["document_top_k"] != list(DOCUMENT_TOP_KS)
        or manifest["development_ids"] != local.DEVELOPMENT_IDS
    ):
        raise ValueError("Unexpected frozen workflow")
    for relative, expected in manifest["dependencies"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Frozen dependency changed: {relative}")
    if manifest["python"] != sys.version.split()[0]:
        raise ValueError("Python version changed")
    if manifest["bank_template_sha256"] != [
        text_sha(template) for template in bank_templates()
    ]:
        raise ValueError("Bank inventory changed")
    if manifest["structural_inventory"] != structural_inventory():
        raise ValueError("Template structure changed")
    payloads = initial_payloads()
    initial_source_guard(payloads)
    if manifest["sources"] != source_inventory(payloads):
        raise ValueError("Initial source inventory changed")
    return manifest


def template_signatures():
    templates = [
        prepare_template(amr, bank_index)
        for bank_index, amr in enumerate(bank_templates())
    ]
    signatures = {}
    for template in templates:
        for unit in template["units"]:
            key = signature_key(unit["signature"])
            signatures.setdefault(key, unit["signature"])
    return [signatures[key] for key in sorted(signatures)]


def build_idf_payload(calibration):
    if (
        len(calibration) != CALIBRATION_DOCUMENTS
        or any(
            row["cohort"] != "human_calibration"
            or row["representation"] != "sentence"
            for row in calibration
        )
    ):
        raise ValueError("Calibration payload inventory mismatch")
    signatures = template_signatures()
    counts = [0] * len(signatures)
    for payload in sorted(calibration, key=lambda row: row["document_id"]):
        graphs = [parse_graph(graph["amr"]) for graph in payload["graphs"]]
        for index, signature in enumerate(signatures):
            if any(
                signature_present(graph, signature) for graph in graphs
            ):
                counts[index] += 1
    rows = [
        {
            "signature": signature,
            "document_frequency": count,
            "idf": 1
            + math.log(
                (CALIBRATION_DOCUMENTS + 1) / (count + 1)
            ),
        }
        for signature, count in zip(signatures, counts)
    ]
    return {
        "status": "frozen_before_positive_development_scoring",
        "calibration_documents": CALIBRATION_DOCUMENTS,
        "calibration_source_sha256": digest(
            [
                payload_inventory(payload)
                for payload in sorted(
                    calibration,
                    key=lambda row: row["document_id"],
                )
            ]
        ),
        "formula": "1 + log((250 + 1) / (document_frequency + 1))",
        "rows": rows,
    }


def idf_map(idf_payload):
    return {
        signature_key(row["signature"]): row["idf"]
        for row in idf_payload["rows"]
    }


def candidate_neighbors(template, candidate, edge_index, node, mapped):
    role, source, target = template["relations"][edge_index]
    if node == source:
        options = [
            candidate_target
            for candidate_role, candidate_source, candidate_target
            in candidate["relations"]
            if candidate_role == role and candidate_source == mapped
            and concept_compatible(
                template["concepts"][target],
                candidate["concepts"][candidate_target],
            )
        ]
    elif node == target:
        options = [
            candidate_source
            for candidate_role, candidate_source, candidate_target
            in candidate["relations"]
            if candidate_role == role and candidate_target == mapped
            and concept_compatible(
                template["concepts"][source],
                candidate["concepts"][candidate_source],
            )
        ]
    else:
        raise ValueError("Tree edge does not touch parent")
    return sorted(set(options))


def enumerate_tree_mappings(template, candidate, children, root_anchor):
    def expand_node(template_node, candidate_node, mapping, used):
        child_rows = children[template_node]

        def expand_child(index, current, occupied):
            if index == len(child_rows):
                yield current
                return
            edge_index, template_child = child_rows[index]
            yield from expand_child(index + 1, current, occupied)
            for candidate_child in candidate_neighbors(
                template,
                candidate,
                edge_index,
                template_node,
                candidate_node,
            ):
                if candidate_child in occupied:
                    continue
                extended = dict(current)
                extended[template_child] = candidate_child
                new_used = occupied | {candidate_child}
                for child_mapping in expand_node(
                    template_child,
                    candidate_child,
                    extended,
                    new_used,
                ):
                    yield from expand_child(
                        index + 1,
                        child_mapping,
                        set(child_mapping.values()),
                    )

        yield from expand_child(0, mapping, used)

    yield from expand_node(
        template["root"],
        root_anchor,
        {template["root"]: root_anchor},
        {root_anchor},
    )


def matched_evidence(template, candidate, mapping, weights):
    matched_relations = set()
    for edge_index, (role, source, target) in enumerate(
        template["relations"]
    ):
        if (
            source in mapping
            and target in mapping
            and (role, mapping[source], mapping[target])
            in candidate["relation_set"]
        ):
            matched_relations.add(edge_index)
    matched_attributes = set()
    for attribute_index, (role, node, value) in enumerate(
        template["attributes"]
    ):
        if (
            node in mapping
            and (role, mapping[node], value)
            in candidate["attribute_set"]
        ):
            matched_attributes.add(attribute_index)
    matched_units = []
    for unit_index, unit in enumerate(template["units"]):
        if unit["kind"] == "relation":
            matched = unit["edge"] in matched_relations
        elif unit["kind"] == "attribute":
            matched = unit["attribute"] in matched_attributes
        else:
            matched = all(
                edge in matched_relations for edge in unit["edges"]
            )
        if matched:
            matched_units.append(unit_index)
    relation_total = len(template["relations"])
    unit_total = len(template["units"])
    weighted_total = sum(
        weights[signature_key(unit["signature"])]
        for unit in template["units"]
    )
    weighted_match = sum(
        weights[
            signature_key(template["units"][unit_index]["signature"])
        ]
        for unit_index in matched_units
    )
    return {
        "scores": {
            "relation_coverage": (
                len(matched_relations) / relation_total
            ),
            "motif_coverage": len(matched_units) / unit_total,
            "idf_motif_coverage": weighted_match / weighted_total,
        },
        "matched_relations": len(matched_relations),
        "matched_attributes": len(matched_attributes),
        "matched_wedges": (
            len(matched_units)
            - len(matched_relations)
            - len(matched_attributes)
        ),
        "matched_units": len(matched_units),
        "mapped_nodes": len(mapping),
        "mapping": [
            mapping.get(node) for node in range(len(template["concepts"]))
        ],
    }


def evidence_better(candidate, current, feature, candidate_nodes):
    if current is None:
        return True
    candidate_key = (
        candidate["scores"][feature],
        candidate["matched_relations"],
        candidate["matched_units"],
        candidate["mapped_nodes"],
    )
    current_key = (
        current["scores"][feature],
        current["matched_relations"],
        current["matched_units"],
        current["mapped_nodes"],
    )
    if candidate_key != current_key:
        return candidate_key > current_key
    missing = candidate_nodes + 1
    candidate_mapping = tuple(
        missing if value is None else value
        for value in candidate["mapping"]
    )
    current_mapping = tuple(
        missing if value is None else value
        for value in current["mapping"]
    )
    return candidate_mapping < current_mapping


def score_template(template, candidate, weights):
    best = {feature: None for feature in FEATURES}
    seen = set()
    root_type = template["concepts"][template["root"]]
    anchors = [
        node
        for node, concept in enumerate(candidate["concepts"])
        if concept_compatible(root_type, concept)
    ]
    for children in template["tree_children"]:
        for root_anchor in anchors:
            for mapping in enumerate_tree_mappings(
                template,
                candidate,
                children,
                root_anchor,
            ):
                identity = tuple(
                    mapping.get(node)
                    for node in range(len(template["concepts"]))
                )
                if identity in seen:
                    continue
                seen.add(identity)
                evidence = matched_evidence(
                    template,
                    candidate,
                    mapping,
                    weights,
                )
                for feature in FEATURES:
                    if evidence_better(
                        evidence,
                        best[feature],
                        feature,
                        len(candidate["concepts"]),
                    ):
                        best[feature] = evidence
    empty = {
        "scores": {feature: 0.0 for feature in FEATURES},
        "matched_relations": 0,
        "matched_attributes": 0,
        "matched_wedges": 0,
        "matched_units": 0,
        "mapped_nodes": 0,
        "mapping": [None] * len(template["concepts"]),
    }
    return {
        "bank_index": template["bank_index"],
        "features": {
            feature: {
                "score": (best[feature] or empty)["scores"][feature],
                "matched_relations": (
                    best[feature] or empty
                )["matched_relations"],
                "matched_attributes": (
                    best[feature] or empty
                )["matched_attributes"],
                "matched_wedges": (
                    best[feature] or empty
                )["matched_wedges"],
                "matched_units": (
                    best[feature] or empty
                )["matched_units"],
                "mapped_nodes": (
                    best[feature] or empty
                )["mapped_nodes"],
                "mapping": (best[feature] or empty)["mapping"],
            }
            for feature in FEATURES
        },
        "enumerated_mappings": len(seen),
    }


def initialize_runtime(idf_payload):
    identity = digest(idf_payload)
    if _RUNTIME:
        if _RUNTIME["idf_identity"] != identity:
            raise ValueError("Runtime IDF identity changed")
        return
    _RUNTIME.update(
        {
            "idf_identity": identity,
            "weights": idf_map(idf_payload),
            "templates": [
                prepare_template(amr, bank_index)
                for bank_index, amr in enumerate(bank_templates())
            ],
        }
    )


def score_graph(graph):
    candidate = parse_graph(graph["amr"])
    scores = [
        score_template(template, candidate, _RUNTIME["weights"])
        for template in _RUNTIME["templates"]
    ]
    return {
        "graph_id": graph["graph_id"],
        "amr_sha256": text_sha(graph["amr"]),
        "known_target_template_indices": graph[
            "known_target_template_indices"
        ],
        "candidate_nodes": len(candidate["concepts"]),
        "candidate_relations": len(candidate["relations"]),
        "candidate_attributes": len(candidate["attributes"]),
        "template_scores": scores,
        "enumerated_mappings": sum(
            row["enumerated_mappings"] for row in scores
        ),
    }


def score_payload(payload):
    started = time.monotonic()
    graphs = [score_graph(graph) for graph in payload["graphs"]]
    return {
        "document_id": payload["document_id"],
        "cohort": payload["cohort"],
        "representation": payload["representation"],
        "source_identity": digest(payload_inventory(payload)),
        "graphs": graphs,
        "enumerated_mappings": sum(
            graph["enumerated_mappings"] for graph in graphs
        ),
        "elapsed_seconds": time.monotonic() - started,
    }


def checkpoint_path(folder, payload):
    return (
        folder
        / "checkpoints"
        / payload["cohort"]
        / payload["representation"]
        / f"{payload['document_id']}.json"
    )


def run_payloads(folder, payloads, idf_payload, workers, heartbeat):
    initialize_runtime(idf_payload)
    completed = []
    missing = []
    for payload in payloads:
        path = checkpoint_path(folder, payload)
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["source_identity"] != digest(
                payload_inventory(payload)
            ):
                raise ValueError("Checkpoint source identity changed")
            completed.append(saved)
        else:
            missing.append(payload)
    if not missing:
        return completed
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
                stage="relation_motif_matching",
                cohort=payload["cohort"],
                representation=payload["representation"],
                completed_documents=done,
                total_documents=len(payloads),
            )
    return completed


def feature_rank(row, feature):
    evidence = row["features"][feature]
    return (
        -evidence["score"],
        -evidence["matched_relations"],
        -evidence["matched_units"],
        -evidence["mapped_nodes"],
        row["bank_index"],
    )


def graph_top_indices(graph, feature, top_k=CANDIDATE_TOP_K):
    ordered = sorted(
        graph["template_scores"],
        key=lambda row: feature_rank(row, feature),
    )
    return [row["bank_index"] for row in ordered[:top_k]]


def candidate_condition(checkpoints, feature):
    document_rows = []
    for checkpoint in sorted(
        checkpoints,
        key=lambda row: row["document_id"],
    ):
        graph_rows = []
        for graph in checkpoint["graphs"]:
            top_indices = graph_top_indices(graph, feature)
            known = graph["known_target_template_indices"]
            graph_rows.append(
                {
                    "graph_id": graph["graph_id"],
                    "top_indices": top_indices,
                    "known_target_template_indices": known,
                    "localized_targets": sum(
                        target in top_indices for target in known
                    ),
                }
            )
        document_rows.append(
            {
                "document_id": checkpoint["document_id"],
                "known_targets": sum(
                    len(row["known_target_template_indices"])
                    for row in graph_rows
                ),
                "localized_targets": sum(
                    row["localized_targets"] for row in graph_rows
                ),
                "graphs": graph_rows,
            }
        )
    return {
        "documents": len(document_rows),
        "known_targets": sum(
            row["known_targets"] for row in document_rows
        ),
        "localized_targets": sum(
            row["localized_targets"] for row in document_rows
        ),
        "rows": document_rows,
    }


def candidate_feature_rank(row):
    result = row["positive_development"]
    original = result["original"]["localized_targets"]
    reparse = result["reparse"]["localized_targets"]
    return (
        min(original, reparse),
        reparse,
        original,
        result["composition"]["localized_targets"],
        FEATURE_PREFERENCE[row["feature"]],
    )


def candidate_development(checkpoints):
    by_representation = {
        representation: [
            row
            for row in checkpoints
            if row["cohort"] == "positive_development"
            and row["representation"] == representation
        ]
        for representation in REPRESENTATIONS
    }
    rows = []
    for feature in FEATURES:
        rows.append(
            {
                "feature": feature,
                "positive_development": {
                    representation: candidate_condition(
                        by_representation[representation],
                        feature,
                    )
                    for representation in REPRESENTATIONS
                },
            }
        )
    selected = max(rows, key=candidate_feature_rank)
    result = selected["positive_development"]
    gates = {
        "original_recall_at_5_pass": (
            result["original"]["localized_targets"]
            >= CANDIDATE_MIN_ORIGINAL
        ),
        "reparse_recall_at_5_pass": (
            result["reparse"]["localized_targets"]
            >= CANDIDATE_MIN_REPARSE
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "status": "complete_candidate_development",
        "top_k": CANDIDATE_TOP_K,
        "feature_rows": rows,
        "selected_feature": selected,
        "gates": gates,
    }


def document_template_scores(checkpoint, feature):
    scores = [0.0] * 50
    for graph in checkpoint["graphs"]:
        for row in graph["template_scores"]:
            bank_index = row["bank_index"]
            scores[bank_index] = max(
                scores[bank_index],
                row["features"][feature]["score"],
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
    threshold = sorted(statistics, reverse=True)[
        CALIBRATION_MAX_FALSE_POSITIVES
    ]
    false_positives = sum(value > threshold for value in statistics)
    if false_positives > CALIBRATION_MAX_FALSE_POSITIVES:
        raise AssertionError("Calibration false-positive budget exceeded")
    return threshold, false_positives


def document_condition(checkpoints, feature, top_k, matrix, threshold):
    rows = []
    for checkpoint in sorted(
        checkpoints,
        key=lambda row: row["document_id"],
    ):
        result = document_statistic(
            document_template_scores(checkpoint, feature),
            matrix,
            top_k,
        )
        top_indices = {
            row["bank_index"] for row in result["top_evidence"]
        }
        known = [
            target
            for graph in checkpoint["graphs"]
            for target in graph["known_target_template_indices"]
        ]
        rows.append(
            {
                "document_id": checkpoint["document_id"],
                "statistic": result["statistic"],
                "threshold_margin": result["statistic"] - threshold,
                "detected": result["statistic"] > threshold,
                "top_evidence": result["top_evidence"],
                "known_targets": len(known),
                "known_targets_in_top_k": sum(
                    target in top_indices for target in known
                ),
            }
        )
    return {
        "documents": len(rows),
        "detected": sum(row["detected"] for row in rows),
        "known_targets": sum(row["known_targets"] for row in rows),
        "known_targets_in_top_k": sum(
            row["known_targets_in_top_k"] for row in rows
        ),
        "rows": rows,
    }


def document_rule_rank(row):
    result = row["positive_development"]
    original = result["original"]["detected"]
    reparse = result["reparse"]["detected"]
    return (
        min(original, reparse),
        reparse,
        original,
        result["composition"]["detected"],
        -row["calibration_false_positives"],
        -row["top_k"],
    )


def document_development(checkpoints, feature):
    calibration = [
        row
        for row in checkpoints
        if row["cohort"] == "human_calibration"
    ]
    _, matrix = calibration_matrix(calibration, feature)
    by_representation = {
        representation: [
            row
            for row in checkpoints
            if row["cohort"] == "positive_development"
            and row["representation"] == representation
        ]
        for representation in REPRESENTATIONS
    }
    rows = []
    for top_k in DOCUMENT_TOP_KS:
        statistics = [
            document_statistic(
                scores,
                matrix,
                top_k,
                leave_out=index,
            )["statistic"]
            for index, scores in enumerate(matrix)
        ]
        threshold, false_positives = calibrated_threshold(statistics)
        rows.append(
            {
                "feature": feature,
                "top_k": top_k,
                "threshold": threshold,
                "calibration_false_positives": false_positives,
                "calibration_statistics": statistics,
                "positive_development": {
                    representation: document_condition(
                        by_representation[representation],
                        feature,
                        top_k,
                        matrix,
                        threshold,
                    )
                    for representation in REPRESENTATIONS
                },
            }
        )
    selected = max(rows, key=document_rule_rank)
    positive = selected["positive_development"]
    gates = {
        "original_development_pass": (
            positive["original"]["detected"]
            >= DEVELOPMENT_MIN_DETECTIONS
        ),
        "reparse_development_pass": (
            positive["reparse"]["detected"]
            >= DEVELOPMENT_MIN_DETECTIONS
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "status": "complete_document_development",
        "rules": rows,
        "selected_rule": selected,
        "gates": gates,
    }


def write_report(folder, summary):
    candidate = summary["candidate_development"]["selected_feature"]
    positive = candidate["positive_development"]
    lines = [
        "# SWAN relation-anchored motif preflight",
        "",
        f"Decision: `{summary['decision']}`.",
        "",
        "This is a feasibility preflight on previously observed synthetic "
        "positives, not a blind detector estimate.",
        "",
        "## Candidate localization",
        "",
        f"- Selected feature: `{candidate['feature']}`",
        (
            "- Recall@5 original/composition/reparse: "
            f"{positive['original']['localized_targets']}/56, "
            f"{positive['composition']['localized_targets']}/56, "
            f"{positive['reparse']['localized_targets']}/56"
        ),
    ]
    if "document_development" in summary:
        selected = summary["document_development"]["selected_rule"]
        result = selected["positive_development"]
        lines.extend(
            [
                "",
                "## Document development",
                "",
                f"- Selected top-k: `{selected['top_k']}`",
                (
                    "- Detections original/composition/reparse: "
                    f"{result['original']['detected']}/15, "
                    f"{result['composition']['detected']}/15, "
                    f"{result['reparse']['detected']}/15"
                ),
            ]
        )
    if "locked_evaluation" in summary:
        final = summary["locked_evaluation"]
        lines.extend(
            [
                "",
                "## Locked evaluation",
                "",
                (
                    "- Human test false positives: "
                    f"{final['human_test']['detected']}/250"
                ),
                (
                    "- Holdout original/reparse: "
                    f"{final['positive_holdout']['original']['detected']}"
                    "/15, "
                    f"{final['positive_holdout']['reparse']['detected']}"
                    "/15"
                ),
                (
                    "- Quality-valid attack sentence detections: "
                    f"{final['external_attack']['sentence']['quality_valid']['detected']}"
                    "/24"
                ),
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Held-out and external attack endpoints were not evaluated "
                "unless both development gates passed.",
            ]
        )
    (folder / "report.md").write_text("\n".join(lines) + "\n")


def stop_summary(
    folder,
    decision,
    candidate,
    document=None,
    checkpoints=None,
):
    stop = {
        "status": "stopped_at_development_gate",
        "decision": decision,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha(folder / "manifest.json"),
        "idf_sha256": sha(folder / "idf.json"),
        "candidate_development_sha256": sha(
            folder / "candidate_development.json"
        ),
        "heldout_accessed": False,
        "attack_accessed": False,
    }
    if document is not None:
        stop["document_development_sha256"] = sha(
            folder / "document_development.json"
        )
    freeze(folder / "stop_decision.json", stop)
    summary = {
        "status": "complete_development_stop",
        "decision": decision,
        "manifest_sha256": sha(folder / "manifest.json"),
        "idf_sha256": sha(folder / "idf.json"),
        "candidate_development": candidate,
        "integrity": {
            "initial_checkpoints": len(checkpoints or []),
            "enumerated_mappings": sum(
                row["enumerated_mappings"]
                for row in (checkpoints or [])
            ),
            "new_api_calls": 0,
            "new_parser_calls": 0,
            "new_s2match_calls": 0,
            "heldout_accessed": False,
            "attack_accessed": False,
        },
    }
    if document is not None:
        summary["document_development"] = document
    write_report(folder, summary)
    write_json(folder / "summary.json", summary)
    return summary


def develop(folder=OUTPUT, workers=8):
    check(folder)
    payloads = initial_payloads()
    calibration = [
        row for row in payloads if row["cohort"] == "human_calibration"
    ]
    idf_payload = build_idf_payload(calibration)
    freeze(folder / "idf.json", idf_payload)
    with claim(folder / "run.lock"):
        with Heartbeat(folder) as heartbeat:
            checkpoints = run_payloads(
                folder,
                payloads,
                idf_payload,
                workers,
                heartbeat,
            )
    candidate = candidate_development(checkpoints)
    freeze(folder / "candidate_development.json", candidate)
    if not candidate["gates"]["all_pass"]:
        return stop_summary(
            folder,
            "stop_relation_motif_candidate_localization",
            candidate,
            checkpoints=checkpoints,
        )
    freeze(
        folder / "candidate_feature.json",
        {
            "status": "candidate_gate_passed",
            "feature": candidate["selected_feature"]["feature"],
            "candidate_development_sha256": sha(
                folder / "candidate_development.json"
            ),
        },
    )
    document = document_development(
        checkpoints,
        candidate["selected_feature"]["feature"],
    )
    freeze(folder / "document_development.json", document)
    if not document["gates"]["all_pass"]:
        return stop_summary(
            folder,
            "stop_relation_motif_document_development",
            candidate,
            document,
            checkpoints,
        )
    selected = document["selected_rule"]
    locked = {
        "status": "locked_before_evaluation_source_access",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature": selected["feature"],
        "top_k": selected["top_k"],
        "threshold": selected["threshold"],
        "calibration_false_positives": selected[
            "calibration_false_positives"
        ],
        "manifest_sha256": sha(folder / "manifest.json"),
        "idf_sha256": sha(folder / "idf.json"),
        "candidate_development_sha256": sha(
            folder / "candidate_development.json"
        ),
        "document_development_sha256": sha(
            folder / "document_development.json"
        ),
    }
    freeze(folder / "locked_rule.json", locked)
    return evaluate(folder, workers)


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
    rows = []
    for document_id in sorted(records):
        path = ATTACK_SOURCE / "detected" / f"{document_id}.json"
        saved = json.loads(path.read_text())
        if saved["record"] != records[document_id]:
            raise ValueError("Attack source identity mismatch")
        rows.append(
            {
                "document_id": document_id,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha(path),
                "quality_valid": records[document_id]["quality_valid"],
                "historical_sentence_z": pairs[document_id][
                    "sentence"
                ]["z"],
                "historically_successful": (
                    pairs[document_id]["sentence"]["z"] < threshold
                ),
            }
        )
    if len(rows) != 30:
        raise ValueError("Attack inventory mismatch")
    return rows


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
                    "source_files": {
                        item["path"]: item["sha256"],
                    },
                    "graphs": graphs,
                }
            )
    return payloads


def evaluation_payloads():
    return (
        local.human_payloads("human_test")
        + local.positive_payloads(local.HOLDOUT_IDS)
        + attack_payloads()
    )


def freeze_evaluation_manifest(folder, payloads):
    counts = source_counts(payloads)
    expected = {
        "external_attack:clause": 30,
        "external_attack:sentence": 30,
        "human_test:sentence": 250,
        "positive_holdout:composition": 15,
        "positive_holdout:original": 15,
        "positive_holdout:reparse": 15,
    }
    if counts != expected:
        raise ValueError("Evaluation source cohort inventory mismatch")
    payload = {
        "status": "frozen_after_lock_before_evaluation_scoring",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_rule_sha256": sha(folder / "locked_rule.json"),
        "source_counts": counts,
        "sources": source_inventory(payloads),
        "attack_inventory": attack_inventory(),
    }
    freeze(folder / "evaluation_manifest.json", payload)
    return payload


def attack_condition(
    checkpoints,
    feature,
    top_k,
    matrix,
    threshold,
    inventory,
):
    result = document_condition(
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
        selected = [row for row in result["rows"] if predicate(row)]
        return {
            "documents": len(selected),
            "detected": sum(row["detected"] for row in selected),
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
    locked = json.loads((folder / "locked_rule.json").read_text())
    if locked["status"] != "locked_before_evaluation_source_access":
        raise ValueError("Invalid locked rule")
    idf_payload = json.loads((folder / "idf.json").read_text())
    initial = initial_payloads()
    with claim(folder / "run.lock"):
        with Heartbeat(folder) as heartbeat:
            initial_checkpoints = run_payloads(
                folder,
                initial,
                idf_payload,
                workers,
                heartbeat,
            )
            payloads = evaluation_payloads()
            freeze_evaluation_manifest(folder, payloads)
            evaluation_checkpoints = run_payloads(
                folder,
                payloads,
                idf_payload,
                workers,
                heartbeat,
            )
    calibration = [
        row
        for row in initial_checkpoints
        if row["cohort"] == "human_calibration"
    ]
    _, matrix = calibration_matrix(calibration, locked["feature"])
    human = [
        row
        for row in evaluation_checkpoints
        if row["cohort"] == "human_test"
    ]
    holdout = [
        row
        for row in evaluation_checkpoints
        if row["cohort"] == "positive_holdout"
    ]
    attacks = [
        row
        for row in evaluation_checkpoints
        if row["cohort"] == "external_attack"
    ]
    human_result = document_condition(
        human,
        locked["feature"],
        locked["top_k"],
        matrix,
        locked["threshold"],
    )
    positive_result = {
        representation: document_condition(
            [
                row
                for row in holdout
                if row["representation"] == representation
            ],
            locked["feature"],
            locked["top_k"],
            matrix,
            locked["threshold"],
        )
        for representation in REPRESENTATIONS
    }
    inventory = attack_inventory()
    attack_result = {
        representation: attack_condition(
            [
                row
                for row in attacks
                if row["representation"] == representation
            ],
            locked["feature"],
            locked["top_k"],
            matrix,
            locked["threshold"],
            inventory,
        )
        for representation in ATTACK_REPRESENTATIONS
    }
    gates = {
        "candidate_development_pass": True,
        "document_development_pass": True,
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
    candidate = json.loads(
        (folder / "candidate_development.json").read_text()
    )
    document = json.loads(
        (folder / "document_development.json").read_text()
    )
    summary = {
        "status": "complete_locked_evaluation",
        "decision": (
            "relation_motif_feasible"
            if gates["all_pass"]
            else "stop_relation_motif_locked_rule"
        ),
        "manifest_sha256": sha(folder / "manifest.json"),
        "idf_sha256": sha(folder / "idf.json"),
        "candidate_development": candidate,
        "document_development": document,
        "locked_rule": locked,
        "locked_evaluation": {
            "human_test": human_result,
            "positive_holdout": positive_result,
            "external_attack": attack_result,
        },
        "gates": gates,
        "integrity": {
            "initial_checkpoints": len(initial_checkpoints),
            "evaluation_checkpoints": len(evaluation_checkpoints),
            "new_api_calls": 0,
            "new_parser_calls": 0,
            "new_s2match_calls": 0,
        },
    }
    write_report(folder, summary)
    write_json(folder / "summary.json", summary)
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
        result = develop(args.output, args.workers)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
