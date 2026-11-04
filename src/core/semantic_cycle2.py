"""Deterministic, text-only cycle2 event views and post-seal diagnostics."""
from __future__ import annotations

from collections import deque
import hashlib
import math
import re

import numpy as np
import penman
from penman.models.amr import model as AMR_MODEL


def event_view(amr, lemma, depth=3, limit=12):
    """Select before scoring, preserving normalized AMR edge directions."""
    graph = penman.decode(amr, model=AMR_MODEL)
    concepts = {s: t for s, r, t in graph.triples if r == ":instance"}
    adjacency = {v: [] for v in concepts}
    incoming = {v: [] for v in concepts}
    for s, r, t in graph.triples:
        if r != ":instance" and t in concepts:
            adjacency[s].append((r, t))
            incoming[t].append((r, s))
    distances, queue = {graph.top: 0}, deque([graph.top])
    while queue:
        node = queue.popleft()
        for _, target in adjacency[node]:
            if target not in distances:
                distances[target] = distances[node] + 1
                queue.append(target)
    # Root-marked, variable-independent structural signature.
    def signature(root):
        labels = {v: concepts[v] + ("|ROOT" if v == root else "") for v in concepts}
        for _ in range(len(concepts)):
            labels = {v: hashlib.sha256(repr((
                labels[v], sorted((r, labels[t]) for r, t in adjacency[v]),
                sorted((r, labels[s]) for r, s in incoming[v]),
                sorted((r, t) for s, r, t in graph.triples
                       if s == v and r != ":instance" and t not in concepts),
            )).encode()).hexdigest() for v in concepts}
        return labels[root]
    predicates = [v for v, c in concepts.items() if re.fullmatch(r".+-\d+", c)]
    matches = [v for v in predicates if re.sub(r"-\d+$", "", concepts[v]) == lemma]
    if matches:
        root = min(matches, key=lambda v: (distances.get(v, math.inf), signature(v)))
    elif graph.top in predicates:
        root = graph.top
    else:
        return None
    selected, queue = {root}, deque([(root, 0)])
    while queue:
        node, d = queue.popleft()
        if d == depth:
            continue
        for _, target in sorted(adjacency[node], key=lambda x: (x[0], signature(x[1]))):
            if target not in selected and len(selected) < limit:
                selected.add(target)
                queue.append((target, d + 1))
    triples = [(s, r, t) for s, r, t in graph.triples
               if s in selected and (r == ":instance" or t not in concepts or t in selected)]
    output = penman.encode(penman.Graph(triples, top=root), model=AMR_MODEL)
    again = penman.decode(output, model=AMR_MODEL)
    if sorted(again.triples) != sorted(triples) or again.top != root:
        return None
    return output


def weights(raw, representation, groups, temperature):
    from semantic_sequence import symbol_weights, average_window_weights
    rows = []
    for row in raw:
        if row is None or any(v is None for v in row):
            rows.append(None)
        elif representation == "C":
            x = np.asarray(row, dtype=float) / temperature
            q = np.exp(x - x.max())
            rows.append(q / q.sum())
        else:
            rows.append(symbol_weights(row, groups, temperature))
    q = average_window_weights(rows)
    return q.tolist() if q is not None else None


def identity(q, threshold):
    if q is None or max(q) < threshold:
        return None
    winners = np.flatnonzero(np.isclose(q, max(q), rtol=0, atol=1e-12))
    return int(winners[0]) if len(winners) == 1 else None


def match_positions(observations, references):
    """Maximum cardinality, then minimum distance; never reads a symbol."""
    active = [(i, r["span"]) for i, r in enumerate(references) if r.get("span")]
    states = {0: (0, ())}
    for p, obs in enumerate(observations):
        a, b = obs["anchor_span"]
        updated = dict(states)
        for mask, (cost, pairs) in states.items():
            for j, (ref, (c, d)) in enumerate(active):
                if mask & (1 << j) or max(a, c) >= min(b, d):
                    continue
                item = (cost + abs(a + b - c - d), pairs + ((p, ref),))
                key = mask | (1 << j)
                if key not in updated or item < updated[key]:
                    updated[key] = item
        states = updated
    _, best = min(states.items(), key=lambda x: (-bin(x[0]).count("1"), x[1]))
    return list(best[1])


def diagnostic(observations, references, predictions):
    pairs = match_positions(observations, references)
    located = len(pairs)
    emitted = sum(p is not None for p in predictions)
    correct = sum(predictions[p] is not None and predictions[p] == references[r]["symbol"]
                  for p, r in pairs)
    per_class = [{"planned": sum(r["symbol"] == c for r in references),
                  "correct": sum(references[r]["symbol"] == c and predictions[p] == c
                                 for p, r in pairs)} for c in range(4)]
    def f1(tp, pred, planned):
        return 2 * tp / (pred + planned) if pred + planned else 0.0
    return {"planned": len(references), "reference_coverage": sum(bool(r.get("span")) for r in references),
            "observations": len(observations), "located": located, "emitted": emitted,
            "identity_correct": correct, "erasures": len(observations) - emitted,
            "localization_precision": located / len(observations) if observations else 0.0,
            "localization_recall": located / len(references) if references else 0.0,
            "localization_f1": f1(located, len(observations), len(references)),
            "identity_precision": correct / emitted if emitted else 0.0,
            "identity_recall": correct / len(references) if references else 0.0,
            "identity_f1": f1(correct, emitted, len(references)),
            "per_class": per_class}


def main_verb(text, nlp):
    verbs = [t for t in nlp(text) if t.pos_ == "VERB"]
    root = [t for t in verbs if t.dep_ == "ROOT"]
    if not verbs:
        return None
    token = (root or verbs)[0]
    return {"span": [token.idx, token.idx + len(token.text)], "lemma": token.lemma_}


def constructed_text(slots, condition):
    """Boundary-only edits; exact token occurrence coordinates are retained."""
    text, spans = "", []
    for i, slot in enumerate(slots):
        value = slot.strip()
        if condition == "split":
            words = list(re.finditer(r"\S+", value))
            if len(words) >= 2:
                at = words[len(words) // 2].start()
                value = value[:at].rstrip() + ". " + value[at:]
        if condition == "merge" and i % 2 == 0 and i + 1 < len(slots):
            value = re.sub(r"[.!?]+$", ",", value)
        start = len(text) + (1 if text else 0)
        text += (" " if text else "") + value
        spans.append([start, len(text)])
    return text, spans
