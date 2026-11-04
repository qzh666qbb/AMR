"""Graph-only successor rules; frozen v1 and all original triples stay intact.

Concession interpretation and movement/receipt linking are development policies,
not independent AMR judgments. Rule traces expose their premises and literal
proofs. No text reference, intended symbol, or revised graph enters this module.
"""
from __future__ import annotations

from collections import defaultdict, deque
import hashlib
import logging
import re

from . import semantic_graph_fact_reader as v1
from .semantic_fact_carrier import classify_facts

VERSION = "semantic-graph-fact-reader:v2-development"
FIELDS = v1.FIELDS
compare_development = v1.compare_development
SPATIAL_HEADS = {
    "across", "along", "around", "away", "back", "beside", "by", "down",
    "forward", "into", "left", "near", "onto", "out", "over", "right",
    "through", "toward", "towards", "under", "up",
}
UNCERTAIN_MARKERS = {"future", "possible", "hypothetical", "even-if", "if"}
RULES = {
    "local_plan_concession": (
        "Interpret a directly attached, same-object, fully bound negative-yet "
        "transfer as a sibling assertion of a local definite plan. Preserve the "
        "original concession direction and every enclosing modal/conditional scope."
    ),
    "movement_receipt_join": (
        "Link an asserted slide-01 or pass-05 to one unique asserted receipt "
        "of the same described object. The recipient comes from receive-01, "
        "never from spatial ARG2, beneficiary, or noncanonical slide ARG4."
    ),
    "receipt_partial_facts": "Retain bound receipt roles and object on erasure.",
}


class GraphIndex(v1.GraphIndex):
    def __init__(self, triples, top, contract):
        super().__init__(triples, top)
        self.contract = contract
        self.rule_trace = []
        self.scope_removed = set()
        self.scope_added = defaultdict(list)
        self._prepare_concessions()

    def _bound_signature(self, node):
        if self.concepts.get(node) not in v1.TRANSFER_FRAMES:
            return None
        values = []
        for relation in (":ARG0", ":ARG2"):
            target = self.one(node, relation)
            value, _ = self.name(target, self.contract) if target else (None, [])
            values.append(value)
        item, _ = self.object(self.one(node, ":ARG1"), self.contract)
        values.append(item)
        return tuple(values) if all(value is not None for value in values) else None

    def _local_blockers(self, node):
        blockers, proof = [], []
        for triple in self.out[node]:
            _, relation, target = triple
            concept = self.concepts.get(target, target)
            if relation == ":condition":
                blockers.append("conditional_embedding")
            elif relation == ":mode":
                blockers.append("nonassertive_mode")
            elif relation in {":time", ":mod", ":concession"} and concept in UNCERTAIN_MARKERS:
                blockers.append("future_or_hypothetical_embedding")
            else:
                continue
            proof.append(triple)
            proof += self.instance(target)
        return blockers, proof

    def _prepare_concessions(self):
        # This is a narrow, explicit clause-scope policy, not an AMR edge flip.
        candidates = []
        for edge in self.triples:
            main, relation, subordinate = edge
            if relation != ":concession":
                continue
            for giving, denial in ((main, subordinate), (subordinate, main)):
                sig = self._bound_signature(giving)
                if sig is None or self._bound_signature(denial) != sig:
                    continue
                if self.one(giving, ":ARG1") != self.one(denial, ":ARG1"):
                    continue
                if self.targets(giving, ":polarity") or self.targets(denial, ":polarity") != ["-"]:
                    continue
                if not any(self.concepts.get(t) == "yet" for t in self.targets(denial, ":time")):
                    continue
                if self._local_blockers(giving)[0] or self._local_blockers(denial)[0]:
                    continue
                parents = [t for t in self.incoming[giving]
                           if t[1] == ":ARG1" and self.concepts.get(t[0]) in v1.PLAN_FRAMES]
                if len(parents) != 1:
                    continue
                plan_edge = parents[0]
                plan = plan_edge[0]
                agent = self.one(plan, ":ARG0")
                name, _ = self.name(agent, self.contract) if agent else (None, [])
                if self.one(plan, ":ARG1") != giving or self.targets(plan, ":polarity") or name != sig[0]:
                    continue
                candidates.append((edge, plan_edge, giving, denial))
        degree = defaultdict(int)
        for _, _, giving, denial in candidates:
            degree[giving] += 1
            degree[denial] += 1
        for edge, plan_edge, giving, denial in candidates:
            if degree[giving] != 1 or degree[denial] != 1:
                continue
            self.scope_removed.add(edge)
            # The effective traversal edge is internal only. All saved evidence
            # is the original concession edge plus the actual plan ARG1 edge.
            self.scope_added[denial].append((plan_edge[0], ":concession", [edge, plan_edge]))
            self.rule_trace.append({
                "rule": "local_plan_concession", "giving": giving, "denial": denial,
                "plan": plan_edge[0], "original_concession_direction": list(edge),
                "evidence_triples": v1.unique_triples([edge, plan_edge]),
                "interpretation_requires_independent_validation": True,
            })

    def contexts(self, node):
        queue = deque([(node, [])])
        seen, plans, blockers, evidence = set(), [], [], []
        while queue:
            current, path = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            if len(path) > 128:
                raise ValueError("Semantic embedding exceeds audit budget")
            local, proof = self._local_blockers(current)
            blockers += local
            evidence += path + proof if local else []
            incoming = [(s, r, [t]) for t in self.incoming[current]
                        for s, r, _ in [t] if t not in self.scope_removed]
            incoming += self.scope_added[current]
            for parent, relation, literal_proof in incoming:
                concept = self.concepts.get(parent, "")
                wrapper = concept in {"and", "or", "contrast-01", "multi-sentence"}
                argument = relation in {":ARG1", ":ARG2", ":ARG3"}
                attachment = relation in {":concession", ":manner", ":time", ":purpose", ":condition"}
                traversable = (
                    wrapper and (argument or re.fullmatch(r":(?:op|snt)\d+", relation))
                ) or (bool(re.search(r"-\d\d$", concept)) and (argument or attachment))
                if not traversable:
                    continue
                chain = path + literal_proof
                if concept == "or":
                    blockers.append("disjunctive_embedding")
                    evidence += chain
                if concept in v1.PLAN_FRAMES and relation == ":ARG1":
                    if self.targets(parent, ":polarity"):
                        blockers.append("negated_or_ambiguous_plan")
                    else:
                        plans.append(parent)
                        evidence += chain + self.instance(parent)
                if concept in v1.NONASSERTIVE_FRAMES and argument:
                    blockers.append("nonassertive_embedding:" + concept)
                    evidence += chain + self.instance(parent)
                if relation in {":purpose", ":condition"}:
                    blockers.append("goal_or_conditional_embedding")
                    evidence += chain
                if self.targets(parent, ":polarity") and not wrapper:
                    blockers.append("negated_embedding:" + concept)
                    evidence += chain + [t for t in self.out[parent] if t[1] == ":polarity"]
                queue.append((parent, chain))
        return sorted(set(plans)), sorted(set(blockers)), evidence

    def event(self, node, contract):
        event = super().event(node, contract)
        if self.targets(node, ":polarity") not in ([], ["-"]):
            event["blockers"].append("ambiguous_event_polarity")
            event["asserted"] = event["plan_agent_consistent"] = False
        if event["concept"] == "slide-01":
            # PropBank slide.01 has ARG0=causer, ARG1=moving thing.
            event["facts"]["recipient"] = None
            event["evidence"]["recipient"] = []
            event["binding_issues"] = sorted(set(
                event["binding_issues"] + ["recipient_requires_receipt"]))
        event["object_node"] = self.one(node, ":ARG1")
        event["movement_candidate"] = event["concept"] in {"slide-01", "pass-05"}
        event["recipient_hints"] = []
        event["join_blockers"] = []
        if event["movement_candidate"]:
            relations = {":ARG2"} if event["concept"] == "pass-05" else {
                ":ARG2", ":ARG4", ":beneficiary", ":direction"}
            for relation in sorted(relations):
                targets = self.targets(node, relation)
                if len(targets) > 1:
                    event["join_blockers"].append("multiple_motion_targets:" + relation)
                for target in targets:
                    concept = self.concepts.get(target, "")
                    name, proof = self.name(target, contract)
                    literal = [(node, relation, target)] + proof
                    if name:
                        event["recipient_hints"].append({
                            "name": name, "relation": relation,
                            "evidence_triples": v1.unique_triples(literal),
                            "role": "consistency_hint_only",
                        })
                    elif concept not in SPATIAL_HEADS:
                        event["join_blockers"].append("unbound_motion_target:" + relation)
            if event["facts"]["sender"] is None:
                event["join_blockers"].append("movement_sender_unbound")
        return event

    def object_fingerprint(self, node):
        """Rooted, literal full description for conservative distinct-node joins."""
        value, proof = self.object(node, self.contract)
        if value is None:
            return None
        edges = [tuple(t) for t in proof if t[1] != ":instance"]
        described = {s for s, r, _ in proof if r == ":instance"}
        # Unread descriptors cannot silently disappear in a cross-node join.
        for current in described:
            for triple in self.out[current]:
                _, relation, target = triple
                if relation == ":instance" or triple in edges:
                    continue
                if target not in self.concepts and relation in {":quant", ":value", ":wiki"}:
                    edges.append(triple)
                else:
                    return None
        adjacent = defaultdict(list)
        for s, r, t in edges:
            adjacent[s].append((r, t))
            if t in self.concepts:
                adjacent[t].append((r + "-of", s))

        def walk(current, previous, seen):
            if current not in self.concepts:
                return ("literal", current)
            if current in seen or len(seen) > 32:
                raise ValueError("Cyclic or oversized object description")
            branches = [(r, walk(target, current, seen | {current}))
                        for r, target in adjacent[current] if target != previous]
            return (self.concepts[current], tuple(sorted(branches)))

        try:
            return walk(node, None, set())
        except ValueError:
            return None

    def same_object(self, first, second):
        if first["facts"]["object"] is None or first["facts"]["object"] != second["facts"]["object"]:
            return None
        if first["object_node"] == second["object_node"]:
            return "shared_object_node"
        left = self.object_fingerprint(first["object_node"])
        right = self.object_fingerprint(second["object_node"])
        return "equal_full_object_description_unique_pair" if left is not None and left == right else None


def _decode(original_penman):
    import penman
    from penman.models.amr import model
    graph_only = "\n".join(line for line in original_penman.splitlines()
                           if not line.lstrip().startswith("#"))
    logs = []

    class Capture(logging.Handler):
        def emit(self, record):
            logs.append({"logger": record.name, "level": record.levelname,
                         "message": record.getMessage()})

    logger = logging.getLogger("penman")
    saved = logger.level, logger.propagate
    handler = Capture(logging.WARNING)
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    try:
        decoded = penman.decode(graph_only, model=model)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(saved[0])
        logger.propagate = saved[1]
    return decoded, graph_only, logs


def read_graph_facts(original_penman, contract):
    if set(contract) != {"role_one", "role_two", "object"} or any(
            not isinstance(v, str) or not v.strip() for v in contract.values()):
        raise ValueError("Only the public roles/object contract is accepted")
    if v1.normalize(contract["role_one"]) == v1.normalize(contract["role_two"]):
        raise ValueError("Public roles must be distinct")
    decoded, graph_only, logs = _decode(original_penman)
    graph = GraphIndex(decoded.triples, decoded.top, contract)
    events = [graph.event(node, contract) for node, concept in graph.concepts.items()
              if concept in v1.TRANSFER_FRAMES | {"receive-01", "slide-01"}]
    gives = [e for e in events if e["concept"] in v1.TRANSFER_FRAMES]
    receives = [e for e in events if e["concept"] == "receive-01"]
    proposals, conflicts = [], []
    join_diagnostics = []

    def signature(event):
        facts = event["facts"]
        values = tuple(facts[k] for k in ("sender", "recipient", "object"))
        return values if all(value is not None for value in values) else None

    for giving in gives:
        sig = signature(giving)
        if sig is None:
            continue
        if giving["concept"] != "pass-05" and giving["asserted"] and not giving["negative"]:
            for receipt in receives:
                facts = receipt["facts"]
                match = facts["recipient"] == sig[1] and facts["object"] == sig[2] and (
                    receipt["source_omitted"] or facts["sender"] == sig[0])
                if match and receipt["asserted"] and not receipt["negative"]:
                    proposals.append((sig, "completed", [giving, receipt]))
        if giving["plan_agent_consistent"] and not giving["negative"]:
            for denial in gives:
                if signature(denial) == sig and denial["asserted"] and denial["negative"] and denial["not_yet"]:
                    proposals.append((sig, "planned_not_occurred", [giving, denial]))
    for receipt in receives:
        sig = signature(receipt)
        if sig and receipt["asserted"] and not receipt["negative"]:
            proposals.append((sig, "completed", [receipt]))

    # Check uniqueness in both directions, including ordinary giving events that
    # could compete for the same receipt. Do not complete motion by itself.
    pairs = []
    for moving in [e for e in events if e["concept"] != "receive-01"]:
        if not moving["asserted"] or moving["negative"] or moving["join_blockers"]:
            continue
        if moving["facts"]["sender"] is None:
            continue
        for receipt in receives:
            recipient = receipt["facts"]["recipient"]
            if not receipt["asserted"] or receipt["negative"] or recipient is None:
                continue
            if not receipt["source_omitted"] and receipt["facts"]["sender"] != moving["facts"]["sender"]:
                continue
            if moving["facts"]["recipient"] not in (None, recipient):
                continue
            if any(hint["name"] != recipient for hint in moving["recipient_hints"]):
                continue
            match = graph.same_object(moving, receipt)
            if match:
                pairs.append((moving, receipt, match))
    movement_degree, receipt_degree = defaultdict(int), defaultdict(int)
    for moving, receipt, _ in pairs:
        movement_degree[moving["node"]] += 1
        receipt_degree[receipt["node"]] += 1
    for moving, receipt, match in pairs:
        if not moving["movement_candidate"]:
            continue
        unique = movement_degree[moving["node"]] == receipt_degree[receipt["node"]] == 1
        # An asserted competing movement with unresolved roles is evidence of
        # ambiguity, not evidence that the only fully bound candidate is unique.
        unresolved_competitor = any(
            other["node"] != moving["node"] and other["movement_candidate"]
            and other["asserted"] and not other["negative"]
            and (other["join_blockers"] or other["facts"]["sender"] is None)
            and graph.same_object(other, receipt)
            for other in events)
        unique = unique and not unresolved_competitor
        join_diagnostics.append({
            "movement": moving["node"], "receipt": receipt["node"], "object_link": match,
            "unique": unique, "unresolved_competing_movement": unresolved_competitor,
        })
        if not unique:
            continue
        sig = (moving["facts"]["sender"], receipt["facts"]["recipient"], moving["facts"]["object"])
        proposals.append((sig, "completed", [moving, receipt]))
        graph.rule_trace.append({
            "rule": "movement_receipt_join", "movement": moving["node"],
            "receipt": receipt["node"], "object_link": match,
            "recipient_from": "receive-01:ARG0", "source_from": moving["concept"] + ":ARG0",
            "evidence_triples": v1.unique_triples(
                [t for event in (moving, receipt) for field in FIELDS for t in event["evidence"][field]]
                + [t for hint in moving["recipient_hints"] for t in hint["evidence_triples"]]
                + [t for event in (moving, receipt) for t in graph.out[event["object_node"]]
                   if t[1] in {":quant", ":value", ":wiki"}]),
            "interpretation_requires_independent_validation": True,
        })
    for sig, state, supporting in proposals:
        if state != "completed":
            continue
        denial = any(signature(e) == sig and e["asserted"] and e["negative"] for e in gives)
        motion_denial = any(
            e["movement_candidate"] and e["asserted"] and e["negative"]
            and e["facts"]["sender"] == sig[0] and e["facts"]["object"] == sig[2]
            and e["facts"]["recipient"] in (None, sig[1])
            and any(graph.same_object(e, s) for s in supporting)
            for e in events)
        if denial or motion_denial:
            conflicts.append("receipt_and_denied_transfer_without_resolved_chronology")
    distinct = {(sig, state) for sig, state, _ in proposals}
    if len(distinct) > 1:
        conflicts.append("multiple_incompatible_transfer_states_or_directions")
    facts = dict.fromkeys(FIELDS, "unknown")
    facts.update(readability="unclear", explicit_status="unknown", contradiction=bool(conflicts))
    evidence = {field: [] for field in FIELDS}
    reasons = []
    if len(distinct) == 1 and not conflicts and not logs:
        sig, state = next(iter(distinct))
        facts.update(sender=sig[0], recipient=sig[1], object=sig[2],
                     transfer_status=state, readability="clear", explicit_status="yes")
        chosen = [event for s, st, supporting in proposals if (s, st) == (sig, state) for event in supporting]
        for field in FIELDS:
            evidence[field] = v1.unique_triples(t for event in chosen for t in event["evidence"][field])
    else:
        for field in ("sender", "recipient", "object"):
            values = {e["facts"][field] for e in events if e["facts"][field] is not None}
            if len(values) == 1:
                facts[field] = next(iter(values))
                evidence[field] = v1.unique_triples(
                    t for e in events if e["facts"][field] == facts[field] for t in e["evidence"][field])
        reasons = sorted(set(conflicts)) or ["no_complete_transfer_state_with_bound_roles_and_object"]
        if logs:
            reasons.append("decode_warning_fail_closed")
        if any(e["concept"] == "receive-01" for e in events):
            graph.rule_trace.append({
                "rule": "receipt_partial_facts",
                "evidence_triples": v1.unique_triples(t for field in FIELDS for t in evidence[field]),
            })
    unsupported = [{"node": n, "concept": c, "reason": "pass-04_is_allow_to_go_by_not_give",
                    "evidence_triples": v1.unique_triples(graph.instance(n))}
                   for n, c in graph.concepts.items() if c == "pass-04"]
    prediction = classify_facts(facts, contract)
    return {
        "version": VERSION, "facts": facts, "prediction": prediction.symbol,
        "prediction_reason": prediction.reason, "evidence_triples": evidence,
        "events": events, "reasons": reasons, "decode_diagnostics": logs,
        "rule_trace": graph.rule_trace, "join_diagnostics": join_diagnostics,
        "unsupported_events": unsupported, "development_rules": RULES,
        "original_penman_sha256": hashlib.sha256(original_penman.encode()).hexdigest(),
        "graph_only_sha256": hashlib.sha256(graph_only.encode()).hexdigest(),
        "normalized_triples": [list(t) for t in graph.triples],
        "human_confirmed": False, "formal_attribution_confirmed": False,
        "reference_or_target_accessed": False,
    }
