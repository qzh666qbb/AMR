"""Observation-capacity A0/A1/A2; no predicted identity grouping or Bank access."""
from __future__ import annotations

from dataclasses import asdict
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .multiview_alignment import (
    AlignmentConfig, event_conflicts, overlap_fraction, surface_conflicts,
)

VERSION = "atomic-correspondence-arms/v1"
CANDIDATE_FIELDS = {
    "candidate_index", "start_char", "end_char", "span_sha256",
    "graph_body_sha256", "usable",
}


def validate(candidates, events):
    for i, c in enumerate(candidates):
        if set(c) != CANDIDATE_FIELDS or c["candidate_index"] != i:
            raise ValueError("Only blind atomic candidate fields accepted")
        if not c["span_sha256"] or not c["graph_body_sha256"]:
            raise ValueError("Unbound candidate/graph")
        if c["end_char"] <= c["start_char"]:
            raise ValueError("Empty candidate span")
    if len({e.event_id for e in events}) != len(events):
        raise ValueError("Duplicate atomic observation ID")
    if any(not 0 <= e.candidate_index < len(candidates) for e in events):
        raise ValueError("Event origin outside frozen candidates")


def universe(candidates, events):
    validate(candidates, events)
    return [
        {"candidate_index": c["candidate_index"], "event_id": e.event_id,
         "event_candidate_index": e.candidate_index}
        for e in sorted(events, key=lambda e: e.event_id)
        if candidates[e.candidate_index]["usable"] and not e.missing_core_roles
        for c in candidates if c["usable"]
    ]


def no_alignment(candidates, events, config=AlignmentConfig()):
    """No matching/grouping/solver: original local relation, stable capacity."""
    validate(candidates, events)
    counts = [0] * len(candidates)
    selected = []
    for e in sorted(events, key=lambda e: e.event_id):
        ci = e.candidate_index
        if candidates[ci]["usable"] and not e.missing_core_roles and counts[ci] < config.max_events_per_candidate:
            selected.append({"candidate_index": ci, "event_id": e.event_id,
                             "event_candidate_index": ci})
            counts[ci] += 1
    return {"arm": "A0", "selected": selected, "identity_grouping": False,
            "solver": "none", "config": asdict(config)}


def predicted(candidates, events, text_vectors, event_vectors, surfaces,
              config=AlignmentConfig()):
    common = universe(candidates, events)
    tv, ev = np.asarray(text_vectors), np.asarray(event_vectors)
    if (tv.ndim != 2 or ev.ndim != 2 or tv.shape[0] != len(candidates)
            or ev.shape[0] != len(events) or tv.shape[1] != ev.shape[1]
            or len(surfaces) != len(candidates) or not np.isfinite(tv).all()
            or not np.isfinite(ev).all()):
        raise ValueError("Frozen shared observation vectors differ")
    lookup = {e.event_id: (i, e) for i, e in enumerate(events)}
    edges, rejected = [], []
    for pair in common:
        ci = pair["candidate_index"]
        ei, e = lookup[pair["event_id"]]
        c, source = candidates[ci], candidates[e.candidate_index]
        overlap = overlap_fraction(c, source)
        similarity = float(tv[ci] @ ev[ei])
        reasons = surface_conflicts(surfaces[ci], e)
        if overlap <= 0 and c["span_sha256"] != source["span_sha256"]:
            reasons.append("no_geometric_overlap")
        if similarity < config.min_similarity:
            reasons.append("below_similarity")
        if reasons:
            rejected.append({**pair, "reasons": reasons})
        else:
            edges.append({**pair, "similarity": similarity, "overlap": overlap,
                          "objective": config.semantic_weight * similarity + config.position_weight * overlap})
    selected = []
    if edges:
        incompatible = []
        for i, left in enumerate(edges):
            for j in range(i + 1, len(edges)):
                right = edges[j]
                if (left["candidate_index"] == right["candidate_index"]
                        and event_conflicts(lookup[left["event_id"]][1], lookup[right["event_id"]][1])):
                    incompatible.append((i, j))
        n = len(candidates) + len(events)
        matrix = lil_matrix((n + len(incompatible), len(edges)))
        for i, edge in enumerate(edges):
            matrix[edge["candidate_index"], i] = 1
            matrix[len(candidates) + lookup[edge["event_id"]][0], i] = 1
        for r, (i, j) in enumerate(incompatible, n):
            matrix[r, i] = matrix[r, j] = 1
        upper = [config.max_events_per_candidate] * len(candidates) + [1] * (len(events) + len(incompatible))
        result = milp(
            np.array([-edge["objective"] + i * 1e-10 for i, edge in enumerate(edges)]),
            integrality=np.ones(len(edges)), bounds=Bounds(0, 1),
            constraints=LinearConstraint(matrix.tocsr(), 0, np.asarray(upper)),
            options={"time_limit": config.solver_time_limit, "mip_rel_gap": 0.0})
        if not result.success:
            raise RuntimeError("Atomic A1 optimizer did not finish; no incumbent result released")
        selected = [e for e, x in zip(edges, result.x) if x > 0.5]
    return {"arm": "A1", "selected": selected, "eligible_edges": edges,
            "rejected": rejected, "identity_grouping": False,
            "capacity_unit": "event_observation", "config": asdict(config)}


def maximum_count(edges, capacities, used=frozenset(), deadline=float("inf")):
    """Exact bipartite cardinality with replicated candidate capacity."""
    options = {}
    for edge in edges:
        if edge["event_id"] not in used:
            options.setdefault(edge["event_id"], []).extend(
                (edge["candidate_index"], slot) for slot in range(capacities[edge["candidate_index"]]))
    occupied = {}

    def augment(event, seen):
        if time.monotonic() > deadline:
            raise TimeoutError("Oracle matching budget exhausted; no partial release")
        for slot in sorted(options[event]):
            if slot in seen:
                continue
            seen.add(slot)
            if slot not in occupied or augment(occupied[slot], seen):
                occupied[slot] = event
                return True
        return False

    return sum(augment(event, set()) for event in sorted(options))


def oracle(candidates, events, support, config=AlignmentConfig()):
    """All common pairs are annotated; unknown forbids Oracle completion."""
    common = universe(candidates, events)
    keys = {(p["candidate_index"], p["event_id"]) for p in common}
    if set(support) != keys or any(type(v) is not bool for v in support.values()):
        raise ValueError("Independent complete support for the entire common universe required")
    edges = [p for p in common if support[(p["candidate_index"], p["event_id"])]]
    edges.sort(key=lambda p: (p["event_id"], p["candidate_index"]))
    caps = [config.max_events_per_candidate] * len(candidates)
    deadline = time.monotonic() + config.solver_time_limit
    target = maximum_count(edges, caps, deadline=deadline)
    selected, used = [], set()
    for i, edge in enumerate(edges):
        ci, eid = edge["candidate_index"], edge["event_id"]
        if eid in used or caps[ci] == 0 or len(selected) == target:
            continue
        trial_caps = list(caps)
        trial_caps[ci] -= 1
        remaining = maximum_count(edges[i + 1:], trial_caps, used | {eid}, deadline)
        if len(selected) + 1 + remaining >= target:
            selected.append(edge)
            used.add(eid)
            caps = trial_caps
    if len(selected) != target:
        raise RuntimeError("Oracle maximum-cardinality reconstruction failed")
    return {"arm": "A2", "selected": selected, "maximum_correct_pairs": target,
            "identity_grouping": False, "tie_rule": "lexicographic_event_id_candidate_index",
            "config": asdict(config), "diagnostic_only": True}


def fixed_statistic(selected, observations, score_center=0.65):
    """Only after selection: each original graph contributes once, weight=1."""
    by_graph = {}
    for edge in selected:
        obs = observations[edge["event_candidate_index"]]
        if not obs["usable"] or not obs["bank_scores"]:
            raise ValueError("Selected graph has no usable frozen score")
        graph = obs["graph_body_sha256"]
        score = max(float(r["score"]) for r in obs["bank_scores"])
        if graph in by_graph and by_graph[graph] != score:
            raise ValueError("Same graph carries inconsistent Bank scores")
        by_graph[graph] = score
    values = list(by_graph.values())
    return {"statistic": sum(v - score_center for v in values) / len(values)**0.5 if values else 0.0,
            "unique_graphs": len(values), "weighting": "fixed",
            "score_center": score_center, "watermark_decision": None,
            "new_calibration_required": True}
