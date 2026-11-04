"""Deterministic, inference-free helpers for the semantic sequence pilot."""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

import penman
from penman.models.amr import model as AMR_MODEL

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "experiments/semantic_sequence_stage1_design1"
BANK = ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_100.json"
POOL = (9, 11, 13, 15, 17, 21, 29, 33, 37, 39, 43, 45, 47, 55, 61, 65, 67, 73, 83, 85, 87, 91, 97, 99)
SEED = 20261023


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def structural_features(amr):
    """Retain root/edge/path types, without lexical concepts or variable names.

    V and numbered predicates are PRED; other concept nodes, including N/NE/X,
    are ENTITY. All non-node terminals are CONST. AMR inverse roles are
    interpreted by penman's AMR model, avoiding manual inversion of 'consist-of'.
    """
    graph = penman.decode(amr, model=AMR_MODEL)
    concepts = {str(t.source): str(t.target) for t in graph.instances()}
    if graph.top not in concepts:
        raise ValueError("Template has no declared graph root")

    def kind(value):
        concept = concepts.get(value)
        if concept is None:
            return "CONST"
        return "PRED" if concept == "V" or re.fullmatch(r".+-\d\d", concept) else "ENTITY"

    features = Counter({("root", kind(graph.top)): 1})
    outgoing = {node: [] for node in concepts}
    for source, role, target in graph.triples:
        source, role, target = str(source), str(role), str(target)
        if role in (":instance", ":wiki"):
            continue
        features[("edge", kind(source), role, kind(target))] += 1
        outgoing[source].append((role, target))

    def walk(node, prefix, visited, depth):
        if depth == 2:
            return
        for role, target in sorted(outgoing.get(node, [])):
            if target in visited:
                continue
            path = prefix + (role, kind(target))
            features[("path",) + path] += 1
            if target in concepts:
                walk(target, path, visited | {target}, depth + 1)

    walk(graph.top, (kind(graph.top),), {graph.top}, 0)
    return features


def feature_signature(features):
    records = [[list(key), value] for key, value in sorted(features.items())]
    return hashlib.sha256(json_bytes(records)).hexdigest()


def jaccard_distance(left, right):
    keys = set(left) | set(right)
    union = sum(max(left[k], right[k]) for k in keys)
    overlap = sum(min(left[k], right[k]) for k in keys)
    return Fraction(union - overlap, union) if union else Fraction(0)


def balanced_assignment(costs, capacity):
    """Exact capacity-constrained assignment with rational arithmetic.

    Rows are ordered by template ID; columns by medoid ID. Equal-cost solutions
    choose the lexicographically smallest assignment of these ordered rows.
    """
    n = len(costs)
    k = len(costs[0])
    if n != k * capacity or any(len(row) != k for row in costs):
        raise ValueError("Unbalanced assignment dimensions")
    states = {(0,) * k: (Fraction(0), ())}
    for row in costs:
        following = {}
        for counts, (score, assignment) in states.items():
            for group in range(k):
                if counts[group] == capacity:
                    continue
                updated = list(counts)
                updated[group] += 1
                key = tuple(updated)
                candidate = (score + row[group], assignment + (group,))
                if key not in following or candidate < following[key]:
                    following[key] = candidate
        states = following
    return states[(capacity,) * k]


def build_symbol_map(bank):
    ids = list(POOL)
    features = {t: structural_features(bank[t]["normalized_template_amr"]) for t in ids}
    distances = {(a, b): jaccard_distance(features[a], features[b]) for a in ids for b in ids}
    medoids = [min(ids, key=lambda a: (sum(distances[a, b] for b in ids), a))]
    while len(medoids) < 4:
        available = [t for t in ids if t not in medoids]
        medoids.append(min(available, key=lambda a: (-min(distances[a, b] for b in medoids), a)))
    medoids.sort()
    history = []
    seen = set()
    for iteration in range(10):
        total, assignment = balanced_assignment(
            [[distances[t, center] for center in medoids] for t in ids], 6
        )
        groups = [[t for t, a in zip(ids, assignment) if a == g] for g in range(4)]
        updated = [min(group, key=lambda a: (sum(distances[a, b] for b in group), a)) for group in groups]
        history.append({
            "iteration": iteration,
            "assignment_medoids": list(medoids),
            "groups": groups,
            "updated_medoids": updated,
            "exact_assignment_cost": str(total),
        })
        state = (tuple(medoids), assignment)
        if state in seen or tuple(sorted(updated)) == tuple(medoids):
            break
        seen.add(state)
        medoids = sorted(updated)
    named = sorted(zip(updated, groups), key=lambda item: item[0])
    symbols = {
        chr(65 + i): {"medoid": center, "templates": group}
        for i, (center, group) in enumerate(named)
    }
    mapping = {t: symbol for symbol, spec in symbols.items() for t in spec["templates"]}
    signature_groups = {}
    for t in ids:
        signature_groups.setdefault(feature_signature(features[t]), []).append(t)
    collisions = [
        {"signature": signature, "templates": group, "symbols": sorted({mapping[t] for t in group})}
        for signature, group in signature_groups.items()
        if len({mapping[t] for t in group}) > 1
    ]
    intra = [float(distances[a, b]) for a in ids for b in ids if a < b and mapping[a] == mapping[b]]
    inter = [float(distances[a, b]) for a in ids for b in ids if a < b and mapping[a] != mapping[b]]
    return {
        "schema": "semantic-sequence-symbol-map/v1",
        "status": "DICTIONARY_NOT_READY" if collisions else "READY",
        "pool": ids,
        "symbols": symbols,
        "template_to_symbol": {str(t): mapping[t] for t in ids},
        "signature_groups": [
            {"signature": signature, "templates": group}
            for signature, group in sorted(signature_groups.items())
        ],
        "distinct_structural_signatures": len(signature_groups),
        "cross_symbol_collisions": collisions,
        "mean_intra_distance": sum(intra) / len(intra),
        "mean_inter_distance": sum(inter) / len(inter),
        "min_inter_distance": min(inter),
        "iterations": history,
        "feature_contract": "root, typed directed edge multiset, root paths of length 1/2; V or numbered predicate=PRED; other concept=ENTITY; literal=CONST; wiki excluded; AMR-model inverse normalization",
        "assignment_contract": "exact rational capacity DP; ascending template IDs and medoid IDs; lexicographic assignment tie break; capacity 6; <=10 rounds",
        "score_access_for_dictionary": False,
    }
