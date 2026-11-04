"""Independent audit for the SWAN relation-anchored motif preflight."""

import argparse
import itertools
import json
import math
from pathlib import Path
from unittest.mock import patch

from run_confirmation_experiment import digest
from run_segmentation_intervention import sha, write_json
import run_swan_local_multi_instance as local
import run_swan_relation_motif as experiment


def stable_paths(folder, include_audit=False):
    paths = [folder / "manifest.json", folder / "idf.json"]
    for name in (
        "candidate_development.json",
        "candidate_feature.json",
        "document_development.json",
        "locked_rule.json",
        "evaluation_manifest.json",
        "stop_decision.json",
        "summary.json",
        "report.md",
    ):
        path = folder / name
        if path.exists():
            paths.append(path)
    paths.extend(sorted((folder / "checkpoints").glob("*/*/*.json")))
    if include_audit and (folder / "audit.json").exists():
        paths.append(folder / "audit.json")
    return paths


def aggregate(folder, include_audit=False):
    return digest(
        [
            {
                "path": str(path.relative_to(folder)),
                "sha256": sha(path),
            }
            for path in stable_paths(folder, include_audit)
        ]
    )


def compatible(template_type, candidate_type):
    return template_type == "X" or template_type == candidate_type


def independent_arm_options(candidate, center, arm):
    direction, role, neighbor_type = arm
    rows = []
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
        if compatible(neighbor_type, candidate["concepts"][neighbor]):
            rows.append((edge_index, neighbor))
    return rows


def independent_signature_present(candidate, signature):
    if signature[0] == "relation":
        _, source_type, role, target_type = signature
        for candidate_role, source, target in candidate["relations"]:
            if (
                candidate_role == role
                and compatible(
                    source_type,
                    candidate["concepts"][source],
                )
                and compatible(
                    target_type,
                    candidate["concepts"][target],
                )
            ):
                return True
        return False
    if signature[0] == "attribute":
        _, node_type, role, value = signature
        return any(
            candidate_role == role
            and candidate_value == value
            and compatible(
                node_type,
                candidate["concepts"][node],
            )
            for candidate_role, node, candidate_value
            in candidate["attributes"]
        )
    if signature[0] != "wedge":
        raise ValueError("Unknown saved motif signature")
    _, center_type, neighbor_rule, first_arm, second_arm = signature
    for center, concept in enumerate(candidate["concepts"]):
        if not compatible(center_type, concept):
            continue
        first = independent_arm_options(candidate, center, first_arm)
        second = independent_arm_options(candidate, center, second_arm)
        for first_edge, first_neighbor in first:
            for second_edge, second_neighbor in second:
                if first_edge == second_edge:
                    continue
                if (
                    (first_neighbor == second_neighbor)
                    == (neighbor_rule == "same_neighbor")
                ):
                    return True
    return False


def reconstruct_idf(calibration, saved):
    signatures = [row["signature"] for row in saved["rows"]]
    parsed_documents = [
        [
            experiment.parse_graph(graph["amr"])
            for graph in payload["graphs"]
        ]
        for payload in sorted(
            calibration,
            key=lambda row: row["document_id"],
        )
    ]
    rows = []
    for signature in signatures:
        count = sum(
            any(
                independent_signature_present(graph, signature)
                for graph in graphs
            )
            for graphs in parsed_documents
        )
        rows.append(
            {
                "signature": signature,
                "document_frequency": count,
                "idf": 1
                + math.log(
                    (experiment.CALIBRATION_DOCUMENTS + 1)
                    / (count + 1)
                ),
            }
        )
    expected = {
        "status": "frozen_before_positive_development_scoring",
        "calibration_documents": experiment.CALIBRATION_DOCUMENTS,
        "calibration_source_sha256": digest(
            [
                experiment.payload_inventory(payload)
                for payload in sorted(
                    calibration,
                    key=lambda row: row["document_id"],
                )
            ]
        ),
        "formula": "1 + log((250 + 1) / (document_frequency + 1))",
        "rows": rows,
    }
    if expected != saved:
        raise ValueError("IDF reconstruction mismatch")
    return {
        experiment.signature_key(row["signature"]): row["idf"]
        for row in rows
    }


def independent_neighbors(
    template,
    candidate,
    edge_index,
    template_parent,
    candidate_parent,
):
    role, source, target = template["relations"][edge_index]
    rows = []
    for candidate_role, candidate_source, candidate_target in (
        candidate["relations"]
    ):
        if candidate_role != role:
            continue
        if (
            template_parent == source
            and candidate_source == candidate_parent
            and compatible(
                template["concepts"][target],
                candidate["concepts"][candidate_target],
            )
        ):
            rows.append(candidate_target)
        if (
            template_parent == target
            and candidate_target == candidate_parent
            and compatible(
                template["concepts"][source],
                candidate["concepts"][candidate_source],
            )
        ):
            rows.append(candidate_source)
    return sorted(set(rows))


def independent_mappings(template, candidate, children, anchor):
    def descend(template_node, candidate_node, mapping):
        child_edges = children[template_node]

        def assign(position, current):
            if position == len(child_edges):
                yield current
                return
            edge_index, child = child_edges[position]
            yield from assign(position + 1, current)
            occupied = set(current.values())
            for neighbor in independent_neighbors(
                template,
                candidate,
                edge_index,
                template_node,
                candidate_node,
            ):
                if neighbor in occupied:
                    continue
                extended = {**current, child: neighbor}
                for subtree in descend(child, neighbor, extended):
                    yield from assign(position + 1, subtree)

        yield from assign(0, mapping)

    yield from descend(template["root"], anchor, {template["root"]: anchor})


def independent_evidence(template, candidate, mapping, weights):
    relation_matches = set()
    for edge_index, (role, source, target) in enumerate(
        template["relations"]
    ):
        if (
            source in mapping
            and target in mapping
            and (role, mapping[source], mapping[target])
            in candidate["relation_set"]
        ):
            relation_matches.add(edge_index)
    attribute_matches = set()
    for attribute_index, (role, node, value) in enumerate(
        template["attributes"]
    ):
        if (
            node in mapping
            and (role, mapping[node], value)
            in candidate["attribute_set"]
        ):
            attribute_matches.add(attribute_index)
    unit_matches = []
    for unit_index, unit in enumerate(template["units"]):
        if unit["kind"] == "relation":
            matched = unit["edge"] in relation_matches
        elif unit["kind"] == "attribute":
            matched = unit["attribute"] in attribute_matches
        else:
            matched = set(unit["edges"]).issubset(relation_matches)
        if matched:
            unit_matches.append(unit_index)
    weighted_total = sum(
        weights[experiment.signature_key(unit["signature"])]
        for unit in template["units"]
    )
    weighted_match = sum(
        weights[
            experiment.signature_key(
                template["units"][unit_index]["signature"]
            )
        ]
        for unit_index in unit_matches
    )
    return {
        "scores": {
            "relation_coverage": (
                len(relation_matches) / len(template["relations"])
            ),
            "motif_coverage": (
                len(unit_matches) / len(template["units"])
            ),
            "idf_motif_coverage": weighted_match / weighted_total,
        },
        "matched_relations": len(relation_matches),
        "matched_attributes": len(attribute_matches),
        "matched_wedges": (
            len(unit_matches)
            - len(relation_matches)
            - len(attribute_matches)
        ),
        "matched_units": len(unit_matches),
        "mapped_nodes": len(mapping),
        "mapping": [
            mapping.get(node) for node in range(len(template["concepts"]))
        ],
    }


def independent_better(row, prior, feature, candidate_nodes):
    if prior is None:
        return True
    row_key = (
        row["scores"][feature],
        row["matched_relations"],
        row["matched_units"],
        row["mapped_nodes"],
    )
    prior_key = (
        prior["scores"][feature],
        prior["matched_relations"],
        prior["matched_units"],
        prior["mapped_nodes"],
    )
    if row_key != prior_key:
        return row_key > prior_key
    sentinel = candidate_nodes + 1
    row_mapping = tuple(
        sentinel if value is None else value for value in row["mapping"]
    )
    prior_mapping = tuple(
        sentinel if value is None else value
        for value in prior["mapping"]
    )
    return row_mapping < prior_mapping


def independent_template_score(template, candidate, weights):
    best = {feature: None for feature in experiment.FEATURES}
    identities = set()
    root_type = template["concepts"][template["root"]]
    anchors = [
        index
        for index, concept in enumerate(candidate["concepts"])
        if compatible(root_type, concept)
    ]
    for children, anchor in itertools.product(
        template["tree_children"],
        anchors,
    ):
        for mapping in independent_mappings(
            template,
            candidate,
            children,
            anchor,
        ):
            identity = tuple(
                mapping.get(node)
                for node in range(len(template["concepts"]))
            )
            if identity in identities:
                continue
            identities.add(identity)
            evidence = independent_evidence(
                template,
                candidate,
                mapping,
                weights,
            )
            for feature in experiment.FEATURES:
                if independent_better(
                    evidence,
                    best[feature],
                    feature,
                    len(candidate["concepts"]),
                ):
                    best[feature] = evidence
    empty = {
        "scores": {
            feature: 0.0 for feature in experiment.FEATURES
        },
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
            for feature in experiment.FEATURES
        },
        "enumerated_mappings": len(identities),
    }


def assert_score_equal(saved, expected):
    if saved["bank_index"] != expected["bank_index"]:
        raise ValueError("Bank index mismatch")
    if saved["enumerated_mappings"] != expected["enumerated_mappings"]:
        raise ValueError("Enumerated mapping count mismatch")
    for feature in experiment.FEATURES:
        saved_row = saved["features"][feature]
        expected_row = expected["features"][feature]
        for key in (
            "matched_relations",
            "matched_attributes",
            "matched_wedges",
            "matched_units",
            "mapped_nodes",
            "mapping",
        ):
            if saved_row[key] != expected_row[key]:
                raise ValueError(
                    f"Saved mapping evidence mismatch: {feature}:{key}"
                )
        if not math.isclose(
            saved_row["score"],
            expected_row["score"],
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Saved feature score mismatch: {feature}")


def load_and_audit_checkpoints(folder, payloads, weights):
    templates = [
        experiment.prepare_template(amr, bank_index)
        for bank_index, amr in enumerate(experiment.bank_templates())
    ]
    checkpoints = []
    graph_template_pairs = 0
    mappings = 0
    for payload in payloads:
        path = experiment.checkpoint_path(folder, payload)
        if not path.exists():
            raise ValueError(f"Missing checkpoint: {path}")
        saved = json.loads(path.read_text())
        if saved["source_identity"] != digest(
            experiment.payload_inventory(payload)
        ):
            raise ValueError("Checkpoint source identity mismatch")
        source_graphs = {
            graph["graph_id"]: graph for graph in payload["graphs"]
        }
        if len(source_graphs) != len(saved["graphs"]):
            raise ValueError("Checkpoint graph count mismatch")
        checkpoint_mappings = 0
        for saved_graph in saved["graphs"]:
            source = source_graphs[saved_graph["graph_id"]]
            if saved_graph["amr_sha256"] != experiment.text_sha(
                source["amr"]
            ):
                raise ValueError("Checkpoint AMR identity mismatch")
            if saved_graph["known_target_template_indices"] != source[
                "known_target_template_indices"
            ]:
                raise ValueError("Known target identity mismatch")
            candidate = experiment.parse_graph(source["amr"])
            if (
                saved_graph["candidate_nodes"]
                != len(candidate["concepts"])
                or saved_graph["candidate_relations"]
                != len(candidate["relations"])
                or saved_graph["candidate_attributes"]
                != len(candidate["attributes"])
            ):
                raise ValueError("Candidate graph inventory mismatch")
            if len(saved_graph["template_scores"]) != 50:
                raise ValueError("Template score count mismatch")
            for template, saved_score in zip(
                templates,
                saved_graph["template_scores"],
            ):
                expected = independent_template_score(
                    template,
                    candidate,
                    weights,
                )
                assert_score_equal(saved_score, expected)
                checkpoint_mappings += expected["enumerated_mappings"]
                graph_template_pairs += 1
        if saved["enumerated_mappings"] != checkpoint_mappings:
            raise ValueError("Checkpoint mapping total mismatch")
        mappings += checkpoint_mappings
        checkpoints.append(saved)
    return checkpoints, graph_template_pairs, mappings


def independent_feature_rank(row, feature):
    evidence = row["features"][feature]
    return (
        -evidence["score"],
        -evidence["matched_relations"],
        -evidence["matched_units"],
        -evidence["mapped_nodes"],
        row["bank_index"],
    )


def independent_candidate_condition(checkpoints, feature):
    documents = []
    for checkpoint in sorted(
        checkpoints,
        key=lambda row: row["document_id"],
    ):
        graphs = []
        for graph in checkpoint["graphs"]:
            ordered = sorted(
                graph["template_scores"],
                key=lambda row: independent_feature_rank(row, feature),
            )
            top_indices = [
                row["bank_index"]
                for row in ordered[:experiment.CANDIDATE_TOP_K]
            ]
            known = graph["known_target_template_indices"]
            graphs.append(
                {
                    "graph_id": graph["graph_id"],
                    "top_indices": top_indices,
                    "known_target_template_indices": known,
                    "localized_targets": sum(
                        target in top_indices for target in known
                    ),
                }
            )
        documents.append(
            {
                "document_id": checkpoint["document_id"],
                "known_targets": sum(
                    len(row["known_target_template_indices"])
                    for row in graphs
                ),
                "localized_targets": sum(
                    row["localized_targets"] for row in graphs
                ),
                "graphs": graphs,
            }
        )
    return {
        "documents": len(documents),
        "known_targets": sum(row["known_targets"] for row in documents),
        "localized_targets": sum(
            row["localized_targets"] for row in documents
        ),
        "rows": documents,
    }


def reconstruct_candidate_development(checkpoints):
    by_representation = {
        representation: [
            row
            for row in checkpoints
            if row["cohort"] == "positive_development"
            and row["representation"] == representation
        ]
        for representation in experiment.REPRESENTATIONS
    }
    rows = [
        {
            "feature": feature,
            "positive_development": {
                representation: independent_candidate_condition(
                    by_representation[representation],
                    feature,
                )
                for representation in experiment.REPRESENTATIONS
            },
        }
        for feature in experiment.FEATURES
    ]

    def rank(row):
        positive = row["positive_development"]
        original = positive["original"]["localized_targets"]
        reparse = positive["reparse"]["localized_targets"]
        return (
            min(original, reparse),
            reparse,
            original,
            positive["composition"]["localized_targets"],
            experiment.FEATURE_PREFERENCE[row["feature"]],
        )

    selected = max(rows, key=rank)
    positive = selected["positive_development"]
    gates = {
        "original_recall_at_5_pass": (
            positive["original"]["localized_targets"]
            >= experiment.CANDIDATE_MIN_ORIGINAL
        ),
        "reparse_recall_at_5_pass": (
            positive["reparse"]["localized_targets"]
            >= experiment.CANDIDATE_MIN_REPARSE
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "status": "complete_candidate_development",
        "top_k": experiment.CANDIDATE_TOP_K,
        "feature_rows": rows,
        "selected_feature": selected,
        "gates": gates,
    }


def forbidden(name):
    def fail(*args, **kwargs):
        del args, kwargs
        raise AssertionError(
            f"Recovery attempted forbidden operation: {name}"
        )

    return fail


def assert_unaccessed(folder):
    forbidden_cohorts = (
        "human_test",
        "positive_holdout",
        "external_attack",
    )
    for cohort in forbidden_cohorts:
        path = folder / "checkpoints" / cohort
        if path.exists() and any(path.rglob("*.json")):
            raise ValueError(f"Forbidden held-out checkpoint exists: {cohort}")
    if (folder / "evaluation_manifest.json").exists():
        raise ValueError("Evaluation manifest exists after development stop")


def audit(folder=experiment.OUTPUT):
    experiment.check(folder)
    idf_payload = json.loads((folder / "idf.json").read_text())
    payloads = experiment.initial_payloads()
    calibration = [
        row for row in payloads if row["cohort"] == "human_calibration"
    ]
    weights = reconstruct_idf(calibration, idf_payload)
    checkpoints, graph_template_pairs, mappings = (
        load_and_audit_checkpoints(folder, payloads, weights)
    )
    candidate = reconstruct_candidate_development(checkpoints)
    saved_candidate = json.loads(
        (folder / "candidate_development.json").read_text()
    )
    if candidate != saved_candidate:
        raise ValueError("Candidate development reconstruction mismatch")
    if not candidate["gates"]["all_pass"]:
        assert_unaccessed(folder)
        summary = json.loads((folder / "summary.json").read_text())
        stop = json.loads((folder / "stop_decision.json").read_text())
        expected_decision = "stop_relation_motif_candidate_localization"
        if (
            summary["decision"] != expected_decision
            or stop["decision"] != expected_decision
            or summary["candidate_development"] != candidate
            or summary["integrity"]["heldout_accessed"]
            or summary["integrity"]["attack_accessed"]
        ):
            raise ValueError("Candidate stop decision mismatch")
        workflow = "candidate_development_stop"
    else:
        feature = candidate["selected_feature"]["feature"]
        expected_document = experiment.document_development(
            checkpoints,
            feature,
        )
        saved_document = json.loads(
            (folder / "document_development.json").read_text()
        )
        if expected_document != saved_document:
            raise ValueError("Document development reconstruction mismatch")
        if not expected_document["gates"]["all_pass"]:
            assert_unaccessed(folder)
            summary = json.loads((folder / "summary.json").read_text())
            expected_decision = (
                "stop_relation_motif_document_development"
            )
            if (
                summary["decision"] != expected_decision
                or summary["document_development"] != expected_document
            ):
                raise ValueError("Document stop decision mismatch")
            workflow = "document_development_stop"
        else:
            if not (folder / "locked_rule.json").exists():
                raise ValueError("Passing development lacks locked rule")
            summary = json.loads((folder / "summary.json").read_text())
            if summary["status"] != "complete_locked_evaluation":
                raise ValueError("Locked evaluation summary missing")
            workflow = "locked_evaluation"
    result = {
        "status": "passed",
        "mismatches": 0,
        "workflow": workflow,
        "idf_signatures_reconstructed": len(idf_payload["rows"]),
        "graph_template_pairs_reconstructed": graph_template_pairs,
        "mappings_reconstructed": mappings,
        "candidate_features_reconstructed": len(experiment.FEATURES),
        "new_api_calls": 0,
        "new_parser_calls": 0,
        "new_s2match_calls": 0,
        "summary_sha256": sha(folder / "summary.json"),
        "stable_aggregate_sha256": aggregate(folder),
    }
    write_json(folder / "audit.json", result)
    return result


def recovery(folder=experiment.OUTPUT):
    if not (folder / "audit.json").exists():
        audit(folder)
    before = aggregate(folder, include_audit=True)
    with (
        patch.object(
            experiment,
            "score_graph",
            forbidden("primary motif scorer"),
        ),
        patch.object(
            experiment,
            "score_template",
            forbidden("primary template matcher"),
        ),
        patch.object(
            local,
            "load_s2match",
            forbidden("S2Match"),
        ),
    ):
        result = audit(folder)
    after = aggregate(folder, include_audit=True)
    if before != after:
        raise ValueError("Recovery changed stable artifacts")
    verification = {
        "status": "passed",
        "stable_before_sha256": before,
        "stable_after_sha256": after,
        "primary_matcher_forbidden": True,
        "s2match_forbidden": True,
        "network_forbidden": True,
        "neural_parser_forbidden": True,
        "audit_sha256": sha(folder / "audit.json"),
        "audit_result": result,
    }
    write_json(folder / "recovery_verification.json", verification)
    return verification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "recovery", "all"))
    parser.add_argument(
        "--output",
        type=Path,
        default=experiment.OUTPUT,
    )
    args = parser.parse_args()
    if args.command == "audit":
        result = audit(args.output)
    elif args.command == "recovery":
        result = recovery(args.output)
    else:
        audit(args.output)
        result = recovery(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
