"""Text-only geometry for frozen SWAN score-fusion transfer diagnostics."""
from __future__ import annotations

from collections import deque
import re

VERSION = "multiscale-blind-text/v1"


def windows(text, length, stride):
    """Enumerate every fixed window, including the final tail, without truncation."""
    tokens = list(re.finditer(r"\S+", text))
    if not tokens:
        return []
    starts = list(range(0, max(1, len(tokens) - length + 1), stride))
    tail = max(0, len(tokens) - length)
    if tail not in starts:
        starts.append(tail)
    return [{"start": tokens[i].start(),
             "end": tokens[min(i + length, len(tokens)) - 1].end(),
             "text": text[tokens[i].start():tokens[min(i + length, len(tokens)) - 1].end()]}
            for i in sorted(starts)]


def overlaps(a, b):
    return max(a["start"], b["start"]) < min(a["end"], b["end"])


def blind_geometry(text):
    """No document IDs, arm labels, slot positions, or template IDs enter here."""
    return {"S6": windows(text, 6, 1), "S12": windows(text, 12, 6)}


def event_subgraphs(amr, max_depth=3, max_nodes=12):
    """Deterministic outgoing event scopes; never include dangling graph edges."""
    import penman
    graph = penman.decode(amr)
    concepts = {str(t.source): str(t.target) for t in graph.instances()}
    adjacency = {}
    for s, role, target in graph.triples:
        if role != ":instance":
            adjacency.setdefault(s, []).append((role, target))
    for edges in adjacency.values():
        edges.sort(key=lambda t: (t[0], concepts.get(t[1], str(t[1])), str(t[1])))
    results = []
    for root in sorted(concepts):
        if not re.fullmatch(r".+-\d\d", concepts[root]):
            continue
        kept, edges = {root}, []
        queue = deque([(root, 0)])
        while queue:
            source, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for role, target in adjacency.get(source, []):
                if target in concepts and target not in kept:
                    if len(kept) >= max_nodes:
                        continue
                    kept.add(target)
                    queue.append((target, depth + 1))
                edges.append((source, role, target))
        triples = [(v, ":instance", concepts[v]) for v in sorted(kept)] + edges
        encoded = penman.encode(penman.Graph(list(dict.fromkeys(triples)), top=root))
        decoded = penman.decode(encoded)
        if decoded.variables() != kept:
            raise ValueError("Event extractor produced invalid variable coverage")
        if encoded not in results:
            results.append(encoded)
    return results


def maximum(rows, n=100):
    """Per-bank-column maxima with missing values retained as None."""
    return [max(values) if values else None
            for values in ([r[i] for r in rows if r[i] is not None] for i in range(n))]


def assemble_features(geometry, short_scores, long_scores, event_scores, carriers=None):
    """Geometric anchor pooling. Carrier positions are optional oracle diagnostics."""
    if (len(short_scores) != len(geometry["S6"])
            or len(long_scores) != len(geometry["S12"])
            or len(event_scores) != len(geometry["S12"])):
        raise ValueError("Candidate scores do not cover the complete geometry")
    anchors = geometry["S12"] if carriers is None else carriers
    result = []
    for anchor in anchors:
        si = [i for i, w in enumerate(geometry["S6"]) if overlaps(anchor, w)]
        li = [i for i, w in enumerate(geometry["S12"]) if overlaps(anchor, w)]
        result.append([
            maximum([short_scores[i] for i in si]),
            maximum([long_scores[i] for i in li]),
            maximum([event_scores[i] for i in li]),
        ])
    return result
