"""Embedding-assisted, target-free AMR correspondence and Bank verification.

Embeddings propose correspondences; only the selected candidate's own graph is
scored. No original watermarked text, source boundary, or target event is input.
This is a new statistic, not the official SWAN z detector or SemStamp.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re

import networkx as nx
import numpy as np
from penman.models.amr import model as amr_model

from .amr_extractor import decode_amr_with_audit
from .multiview_candidates import WindowConfig, candidate_windows, select_budget
from .multiview_views import extract_events

VERSION = "embedding-assisted-amr/v1"
ARMS = ("amr_only", "embedding_amr", "embedding_without_relation")


def digest(value):
    data = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()


@dataclass(frozen=True)
class FusionConfig:
    candidate_budget: int = 24
    top_k: int = 4
    min_similarity: float = 0.35
    score_center: float = 0.65
    max_graph_nodes: int = 256
    max_events: int = 256

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (
                self.candidate_budget, self.top_k, self.max_graph_nodes, self.max_events)):
            raise ValueError("Positive integer capacities required")
        if not -1 <= self.min_similarity <= 1 or not 0 <= self.score_center <= 1:
            raise ValueError("Invalid similarity or score center")


@dataclass(frozen=True)
class GraphObservation:
    """A raw graph bound to its exact parser input and upstream audit."""
    input_sha256: str
    amr: str | None
    parser_fingerprint: str
    usable: bool
    reasons: tuple = ()


def windows(text, config=FusionConfig(), window_config=WindowConfig()):
    pool = candidate_windows(text, window_config)
    candidates = select_budget(pool["candidates"], config.candidate_budget)
    return candidates, {
        "enumerated": pool["enumerated_count"],
        "pool_truncated": pool["truncated_count"],
        "budget_omitted": len(pool["candidates"]) - len(candidates),
    }


def _semantic_graph(amr, max_nodes):
    """Use AMR inversion rules (e.g. :consist-of is not an inverse ARG)."""
    decoded, warnings, valid, roundtrip = decode_amr_with_audit(amr)
    if warnings or not valid or not roundtrip:
        raise ValueError("invalid_or_recovered_penman")
    graph = nx.MultiDiGraph()
    instances = list(decoded.instances())
    concepts = {str(t.source): str(t.target) for t in instances}
    if len(concepts) != len(instances):
        raise ValueError("duplicate_amr_variable")
    if len(concepts) > max_nodes:
        raise ValueError("graph_node_capacity")
    for node, concept in concepts.items():
        graph.add_node(node, concept=concept, anchor=False, literal=False)
    for index, triple in enumerate(decoded.triples):
        if triple[1] == ":instance":
            continue
        s, role, t = map(str, amr_model.deinvert(triple))
        if s not in concepts:
            raise ValueError("undefined_amr_source")
        if t not in concepts:
            literal = ("literal", index)
            graph.add_node(literal, concept=t, anchor=False, literal=True)
            t = literal
        graph.add_edge(s, t, role=role)
    if not nx.is_weakly_connected(graph):
        raise ValueError("disconnected_amr")
    return graph


def _fact_graph(graph, anchor):
    # Drop only conjunctive/sentence scaffolding. Keep governing events and all of their
    # arguments: "wants to lend", "does not lend", and "lends" are distinct.
    ancestors = {anchor}
    queue = [anchor]
    while queue:
        node = queue.pop()
        for parent, _, edge in graph.in_edges(node, data=True):
            parent_concept = graph.nodes[parent]["concept"]
            if ((parent_concept == "and" and re.fullmatch(r":op\d+", edge["role"]))
                    or (parent_concept == "multi-sentence" and re.fullmatch(r":snt\d+", edge["role"]))):
                continue
            if parent not in ancestors:
                ancestors.add(parent)
                queue.append(parent)
    nodes = set(ancestors)
    for ancestor in ancestors:
        nodes.update(nx.descendants(graph, ancestor))
    fact = graph.subgraph(nodes).copy()
    fact.nodes[anchor]["anchor"] = True
    return fact


def same_fact(left, right):
    """Exact labeled multigraph isomorphism preserves reentrancy and roles."""
    return nx.is_isomorphic(
        left, right,
        node_match=nx.algorithms.isomorphism.categorical_node_match(
            ["concept", "anchor", "literal"], [None, False, False]),
        edge_match=lambda a, b: Counter(e["role"] for e in a.values()) == Counter(
            e["role"] for e in b.values()),
    )


def _normalized(vectors, count):
    vectors = np.asarray(vectors, dtype=np.float64)
    if (vectors.ndim != 2 or vectors.shape[0] != count or vectors.shape[1] == 0
            or not np.isfinite(vectors).all()):
        raise ValueError("Invalid encoder vectors")
    scales = np.max(np.abs(vectors), axis=1, keepdims=True)
    if np.any(scales <= 0):
        raise ValueError("Zero encoder vector")
    scaled = vectors / scales
    norms = np.sqrt(np.sum(scaled * scaled, axis=1, keepdims=True))
    return scaled / norms


def _select(eligible, candidates, identities, body_hashes):
    """Greedy, score-blind canonical representatives; never optimize Bank hits.

    A selected full graph occupies *all* of its fact identities. This prevents
    counting a merged graph again via one of its already counted constituents.
    """
    used_facts, used_bodies, used_spans, selected = set(), set(), set(), []
    for ci in sorted(eligible, key=lambda i: (
            candidates[i]["word_count"], candidates[i]["start_char"],
            candidates[i]["end_char"], i)):
        keys = set(identities[ci])
        span = digest(candidates[ci]["text"])
        if not keys or keys & used_facts or body_hashes[ci] in used_bodies or span in used_spans:
            continue
        selected.append(ci)
        used_facts.update(keys)
        used_bodies.add(body_hashes[ci])
        used_spans.add(span)
    return selected


class EmbeddingAMRDetector:
    def __init__(self, encoder, scorer, parser_fingerprint, config=FusionConfig(),
                 window_config=WindowConfig()):
        self.encoder, self.scorer = encoder, scorer
        self.config, self.window_config = config, window_config
        self.parser_fingerprint = parser_fingerprint
        self.fingerprint = digest({
            "version": VERSION, "config": asdict(config), "windows": asdict(window_config),
            "encoder": encoder.fingerprint, "scorer": scorer.fingerprint,
            "parser": parser_fingerprint,
        })

    def detect(self, text, observations):
        """Observations must cover exactly the blind windows, including failures.

        The provider parses these windows first. It never supplies original
        source text, human labels, preselected events, or precomputed Bank hits.
        """
        candidates, budget = windows(text, self.config, self.window_config)
        if len(observations) != len(candidates):
            raise ValueError("Complete candidate observation denominator required")
        events, excluded, identities, body_hashes = [], [], {}, {}
        representatives = []
        eligible = []
        for ci, (candidate, observation) in enumerate(zip(candidates, observations)):
            if (observation.input_sha256 != digest(candidate["text"])
                    or observation.parser_fingerprint != self.parser_fingerprint):
                raise ValueError("Candidate/parser binding mismatch")
            reasons = list(observation.reasons)
            if not observation.usable or not observation.amr:
                reasons.append("unusable_parse")
            if not self.encoder.fits(candidate["text"]):
                reasons.append("candidate_encoder_context")
            if reasons:
                excluded.append({"candidate_index": ci, "reasons": sorted(set(reasons))})
                continue
            try:
                graph = _semantic_graph(observation.amr, self.config.max_graph_nodes)
            except ValueError as exc:
                excluded.append({"candidate_index": ci, "reasons": [str(exc)]})
                continue
            # Description generation is for retrieval only. Verification uses
            # the original labeled graph, not this lossy event verbalization.
            descriptions = {e.node: e for e in extract_events(observation.amr, ci)}
            current, unencodable = [], False
            for node, data in graph.nodes(data=True):
                if data["literal"] or not re.fullmatch(r".+-\d\d", data["concept"]):
                    continue
                event = descriptions.get(node)
                if event is None or event.missing_core_roles:
                    continue
                if not self.encoder.fits(event.description):
                    unencodable = True
                    break
                fact = _fact_graph(graph, node)
                identity = next((i for i, other in enumerate(representatives)
                                 if same_fact(fact, other)), None)
                if identity is None:
                    identity = len(representatives)
                    representatives.append(fact)
                current.append({"candidate_index": ci, "node": node,
                                "identity": identity, "description": event.description,
                                "event_id": f"c{ci}:{node}"})
            if unencodable or not current:
                excluded.append({"candidate_index": ci, "reasons": [
                    "event_encoder_context" if unencodable else "no_encodable_core_event"]})
                continue
            if len(events) + len(current) > self.config.max_events:
                raise RuntimeError("Event capacity exceeded; no partial statistic released")
            identities[ci] = sorted({e["identity"] for e in current})
            body_hashes[ci] = digest("\n".join(
                line for line in observation.amr.splitlines() if not line.startswith("#")))
            events.extend(current)
            eligible.append(ci)

        verified, proposed, pairs = set(), set(), []
        if events:
            tv = _normalized(self.encoder.encode([candidates[i]["text"] for i in eligible]), len(eligible))
            ev = _normalized(self.encoder.encode([e["description"] for e in events]), len(events))
            if tv.shape[1] != ev.shape[1]:
                raise ValueError("Encoder dimensions differ between views")
            # Bounded candidate matrices do not need BLAS. Explicit contraction
            # also avoids platform BLAS floating-point status warning leakage.
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                similarities = np.einsum("ij,kj->ik", ev, tv, optimize=False)
            if not np.isfinite(similarities).all():
                raise RuntimeError("Nonfinite similarity; no clipping or partial release")
            similarities = np.clip(similarities, -1.0, 1.0)
            for ei, event in enumerate(events):
                ranked = sorted(range(len(eligible)), key=lambda j: (
                    -float(similarities[ei, j]), eligible[j]))[:self.config.top_k]
                for j in ranked:
                    ci, similarity = eligible[j], float(similarities[ei, j])
                    if similarity < self.config.min_similarity:
                        continue
                    proposed.add(ci)
                    supported = event["identity"] in identities[ci]
                    if supported:
                        verified.add(ci)
                    pairs.append({
                        "query_event_id": event["event_id"],
                        "query_candidate_index": event["candidate_index"],
                        "candidate_index": ci, "similarity": similarity,
                        "relation_supported": supported,
                        "reason": None if supported else "role_entity_scope_or_relation_mismatch",
                    })
        plans = {
            "amr_only": _select(eligible, candidates, identities, body_hashes),
            "embedding_amr": _select(verified, candidates, identities, body_hashes),
            # Diagnostic: drops the correspondence verification, but still uses
            # real Bank evidence. It is never called an embedding watermark.
            "embedding_without_relation": _select(proposed, candidates, identities, body_hashes),
        }
        # All plans are sealed in memory before the first Bank call.
        matches = {}
        for ci in sorted(set(i for plan in plans.values() for i in plan)):
            match = self.scorer.score_amr(observations[ci].amr)
            if match is None:
                # Official normalization could not produce a valid Bank graph for
                # this candidate. It contributes no evidence to any arm; the row
                # remains in the complete denominator.
                excluded.append({"candidate_index": ci,
                                 "reasons": ["official_normalization_unavailable"]})
                continue
            if not math.isfinite(match.score) or not 0 <= match.score <= 1:
                raise RuntimeError("Invalid Bank score; no partial statistic released")
            matches[ci] = match
        arms = {}
        for name, selected in plans.items():
            selected = [ci for ci in selected if ci in matches]
            evidence = [{
                "candidate_index": ci, "graph_body_sha256": body_hashes[ci],
                "fact_ids": identities[ci], "bank_score": float(matches[ci].score),
                "template_index": matches[ci].template_index,
            } for ci in selected]
            utility = sum(e["bank_score"] - self.config.score_center for e in evidence)
            arms[name] = {
                "statistic": utility / math.sqrt(len(evidence)) if evidence else None,
                "evidence_count": len(evidence), "evidence": evidence,
                "status": "uncalibrated" if evidence else "abstain",
                "is_watermarked": None, "p_value": None,
            }
        return {
            "version": VERSION, "detector_fingerprint": self.fingerprint,
            "text_sha256": digest(text), "candidate_count": len(candidates),
            "usable_candidates": len(eligible), "budget": budget,
            "exclusions": excluded, "event_count": len(events),
            "fact_identity_count": len(representatives), "pairs": pairs, "arms": arms,
            "semantic_truth_confirmed": False, "new_calibration_required": True,
        }
