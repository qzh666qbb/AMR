"""Target-free fact-core admissibility and exact constrained triple matching.

The v2 reader supplies a conservative *development rule*, not semantic truth.
All scores compare literal graph structure under injective node mappings. Public
role/object bindings and the observable anchor are fixed in every mapping.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re

from . import semantic_graph_fact_reader as base
from . import semantic_graph_fact_reader_v2 as reader

VERSION = "semantic-fact-core-matcher/v1"


class SearchBudgetExceeded(RuntimeError):
    """No incumbent/partial score is exposed after a search budget exhaustion."""


def graph_payload(triples, bindings):
    triples = sorted(set(tuple(t) for t in triples))
    concepts = {s: t for s, r, t in triples if r == ":instance"}
    if len(concepts) != sum(r == ":instance" for _, r, _ in triples):
        raise ValueError("Graph variable has incompatible concepts")
    if (any(v not in concepts for v in bindings.values())
            or len(set(bindings.values())) != len(bindings)):
        raise ValueError("Core bindings must identify distinct graph nodes")
    return {"triples": [list(t) for t in triples], "bindings": dict(bindings)}


def representation(original_penman, contract, anchor_lemma, context_lemmas):
    """Unique lexical anchoring, full graph and proof-core share one node space.

    No AMR word alignments are assumed. Repeated lexical anchors, ambiguous graph
    roots or incomplete facts are erasures, with the denominator unchanged.
    """
    if not isinstance(anchor_lemma, str) or not anchor_lemma:
        raise ValueError("Blind lexical anchor required")
    if Counter(context_lemmas)[anchor_lemma] != 1:
        return {"status": "invalid", "reason": "nonunique_surface_anchor"}
    result = reader.read_graph_facts(original_penman, contract)
    if result["prediction"] is None:
        return {"status": "invalid", "reason": "incomplete_rule_based_fact_core",
                "reader_reasons": result["reasons"]}
    triples = [tuple(t) for t in result["normalized_triples"]]
    decoded, _, _ = reader._decode(original_penman)
    index = reader.GraphIndex(triples, decoded.top, contract)
    roots = [node for node, concept in index.concepts.items()
             if re.sub(r"-\d\d$", "", concept).split("-")[0] == anchor_lemma]
    proof = set(tuple(t) for values in result["evidence_triples"].values() for t in values)
    if len(roots) != 1 or not any(s == roots[0] and r == ":instance" for s, r, _ in proof):
        return {"status": "invalid", "reason": "missing_ambiguous_or_noncore_graph_anchor"}
    bindings = {"anchor": roots[0]}
    normalization = {}
    for role in ("role_one", "role_two"):
        nodes = [node for node in index.concepts if index.name(node, contract)[0] == contract[role]]
        if len(nodes) != 1:
            return {"status": "invalid", "reason": "nonunique_public_role_binding"}
        bindings[role] = nodes[0]
        normalization[(nodes[0], ":instance", index.concepts[nodes[0]])] = (nodes[0], ":instance", "__" + role + "__")
        name_node = index.one(nodes[0], ":name")
        for s, relation, target in index.out[name_node]:
            if re.fullmatch(r":op\d+", relation):
                normalization[(s, relation, target)] = (s, relation, '"' + role + relation + '"')
    object_nodes = [node for node in index.concepts if index.object(node, contract)[0] == contract["object"]]
    if len(object_nodes) != 1:
        return {"status": "invalid", "reason": "nonunique_public_object_binding"}
    obj = object_nodes[0]
    bindings["object"] = obj
    normalization[(obj, ":instance", index.concepts[obj])] = (obj, ":instance", "__public_object__")
    # Keep every original edge. Public object modifiers retain their structure
    # but not their scene-specific lexical identities.
    _, object_proof = index.object(obj, contract)
    for s, r, t in object_proof:
        if r == ":instance" and s != obj:
            normalization[(s, r, t)] = (s, r, "__object_descriptor__")
    if len(set(bindings.values())) != len(bindings):
        return {"status": "invalid", "reason": "overlapping_core_bindings"}
    core_nodes = {s for s, _, _ in proof} | {t for _, _, t in proof if t in index.concepts}
    core = proof | {(s, ":instance", index.concepts[s]) for s in core_nodes}
    if not set(bindings.values()).issubset(core_nodes):
        return {"status": "invalid", "reason": "binding_outside_literal_fact_proof"}
    canonical = lambda values: [normalization.get(t, t) for t in values]
    return {
        "status": "complete", "reader_version": reader.VERSION,
        "full": graph_payload(canonical(triples), bindings),
        "core": graph_payload(canonical(core), bindings),
        "literal_proof_triples": [list(t) for t in sorted(core)],
        "original_graph_sha256": hashlib.sha256(original_penman.encode()).hexdigest(),
        "graph_only_sha256": result["graph_only_sha256"],
        "public_normalization": [{"original": list(k), "normalized": list(v)}
                                 for k, v in sorted(normalization.items())],
        "semantic_truth_confirmed": False,
    }


def exact_match(left, right, max_states=200000):
    """Exact maximum common triples subject to public fixed bindings.

    Non-core nodes may map to a same-concept node or remain unmatched. This
    defines the complete allowed mapping set; no random restarts or beam search.
    """
    if max_states < 1:
        raise ValueError("Positive search capacity required")
    left = graph_payload(left["triples"], left["bindings"])
    right = graph_payload(right["triples"], right["bindings"])
    lt, rt = [tuple(t) for t in left["triples"]], set(map(tuple, right["triples"]))
    lc = {s: t for s, r, t in lt if r == ":instance"}
    rc = {s: t for s, r, t in rt if r == ":instance"}
    if set(left["bindings"]) != set(right["bindings"]):
        raise ValueError("Both graphs require identical public core slots")
    fixed = {left["bindings"][slot]: right["bindings"][slot] for slot in sorted(left["bindings"])}
    used = set(fixed.values())
    domains = {node: sorted(n for n, c in rc.items() if c == concept and n not in used)
               for node, concept in lc.items() if node not in fixed}
    order = sorted(domains, key=lambda node: (len(domains[node]), -sum(node in (s, t) for s, _, t in lt), node))
    mapping = dict(fixed)
    best, best_map, states = -1, None, 0

    def triple_match(triple):
        s, relation, t = triple
        source = mapping.get(s)
        target = mapping.get(t) if relation != ":instance" and t in lc else t
        return source is not None and target is not None and (source, relation, target) in rt

    def visit(depth):
        nonlocal best, best_map, states
        states += 1
        if states > max_states:
            raise SearchBudgetExceeded("Exact mapping capacity exceeded")
        known, pending = [], 0
        for triple in lt:
            s, relation, t = triple
            required = {s} | ({t} if relation != ":instance" and t in lc else set())
            if required.issubset(mapping):
                known.append(triple_match(triple))
            else:
                pending += 1
        count = sum(known)
        if count + pending <= best:
            return
        if depth == len(order):
            best, best_map = count, dict(mapping)
            return
        node = order[depth]
        for target in domains[node] + [None]:
            if target is not None and target in used:
                continue
            mapping[node] = target
            if target is not None:
                used.add(target)
            visit(depth + 1)
            if target is not None:
                used.remove(target)
            del mapping[node]

    visit(0)
    mapping = best_map
    evidence = [list(t) for t in lt if triple_match(t)]
    return {
        "score": 2. * best / (len(lt) + len(rt)) if lt or rt else 1.,
        "matched_triples": best, "left_triples": len(lt), "right_triples": len(rt),
        "mapping": dict(sorted(best_map.items())), "matched_left_evidence": evidence,
        "search_states": states, "exact": True, "max_states": max_states,
    }


def score_view(observation, templates, view, max_states=200000):
    """All columns share the same core and capacity policy; failures are NaN."""
    if view not in ("full", "core"):
        raise ValueError("Unknown view")
    if observation["status"] != "complete":
        return {"status": "invalid", "scores": None, "reason": observation["reason"]}
    if len(templates) != 24 or any(t["status"] != "complete" for t in templates):
        raise ValueError("Exactly 24 complete frozen templates required")
    try:
        matches = [exact_match(observation[view], t[view], max_states) for t in templates]
    except SearchBudgetExceeded:
        return {"status": "budget_abstain", "scores": None, "reason": "complete_mapping_search_not_finished"}
    return {"status": "complete", "scores": [m["score"] for m in matches], "matches": matches,
            "shared_core_policy": True}
