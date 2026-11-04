"""Graph-only development fact reader, with literal triple evidence and erasures.

Neither text, intended symbols nor reference facts enter this module. Facts are
rule-derived candidates, never independently verified annotations.
"""
from __future__ import annotations

from collections import defaultdict, deque
import hashlib
import logging
import re

from .semantic_fact_carrier import classify_facts
from .semantic_fact_layers import actual_signature

VERSION = "semantic-graph-fact-reader:v1-development"
FIELDS = ("sender", "recipient", "object", "transfer_status")
TRANSFER_FRAMES = {"hand-01", "hand-over-02", "give-01", "pass-05", "hand-off-02"}
PLAN_FRAMES = {"decide-01", "plan-01"}
NONASSERTIVE_FRAMES = {
    "want-01", "intend-01", "hope-01", "wish-01", "possible-01", "obligate-01",
    "recommend-01", "need-01", "say-01", "tell-01", "believe-01", "think-01",
    "expect-01", "promise-01", "confirm-01", "deny-01", "pretend-01",
}
OBJECT_RELATIONS = {":mod", ":topic", ":consist-of", ":purpose", ":name"}
NOMINAL_ARGUMENT_HEADS = {"bundle-01", "spool-01"}
OBJECT_INVERSE_QUALIFIERS = {
    "record-01": {":instrument"},
    "survey-01": {":instrument"},
    "sample-01": {":ARG1"},
    "label-01": {":ARG0"},
}
PRONOUNS = {"he", "she", "it", "they", "this", "that", "thing", "person"}


def normalize(value):
    return " ".join(value.split()).casefold()


def lemma(value):
    value = re.sub(r"-\d\d$", "", value.casefold())
    return {"recording": "record"}.get(value, value)


class GraphIndex:
    def __init__(self, triples, top):
        self.triples = [tuple(t) for t in triples]
        self.top = top
        self.out = defaultdict(list)
        self.incoming = defaultdict(list)
        self.concepts = {}
        for s, r, t in self.triples:
            self.out[s].append((s, r, t))
            self.incoming[t].append((s, r, t))
            if r == ":instance":
                if s in self.concepts and self.concepts[s] != t:
                    raise ValueError("A graph variable has incompatible concepts")
                self.concepts[s] = t
        if top not in self.concepts:
            raise ValueError("Graph has no valid top instance")

    def targets(self, node, relation):
        return list(dict.fromkeys(t for _, r, t in self.out[node] if r == relation))

    def one(self, node, relation):
        values = self.targets(node, relation)
        return values[0] if len(values) == 1 else None

    def instance(self, node):
        return [t for t in self.out[node] if t[1] == ":instance"]

    def name(self, node, contract):
        evidence = self.instance(node)
        name_node = self.one(node, ":name")
        if not name_node or self.concepts.get(node) != "person":
            return None, evidence
        operators = sorted((int(r[3:]), t, (s, r, t)) for s, r, t in self.out[name_node]
                           if re.fullmatch(r":op\d+", r))
        if not operators:
            return None, evidence
        # PENMAN quoted-string unescaping belongs to the decoded attribute.
        import penman.constant
        try:
            name = " ".join(str(penman.constant.evaluate(t)) for _, t, _ in operators)
        except (ValueError, TypeError):
            return None, evidence
        evidence += [(node, ":name", name_node)] + self.instance(name_node)
        evidence += [triple for _, _, triple in operators]
        matches = [contract[k] for k in ("role_one", "role_two")
                   if normalize(contract[k]) == normalize(name)]
        return (matches[0] if len(matches) == 1 else None), evidence

    def object(self, node, contract):
        """Require head and all public modifiers inside one object description."""
        if not node or node not in self.concepts:
            return None, []
        words = [lemma(w) for w in contract["object"].split()]
        root = self.concepts[node]
        if lemma(root) != words[-1]:
            return None, self.instance(node)
        evidence = []
        vocabulary = set()
        queue, seen = deque([(node, 0)]), set()
        while queue:
            current, depth = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            evidence += self.instance(current)
            vocabulary.add(lemma(self.concepts.get(current, "")))
            if depth >= 2:
                continue
            for triple in self.out[current]:
                _, relation, target = triple
                allowed = relation in OBJECT_RELATIONS or (
                    current == node and root in NOMINAL_ARGUMENT_HEADS and relation == ":ARG1")
                if allowed and target in self.concepts:
                    evidence.append(triple)
                    queue.append((target, depth + 1))
            if current == node:
                for triple in self.incoming[node]:
                    parent, relation, _ = triple
                    concept = self.concepts.get(parent)
                    if relation in OBJECT_INVERSE_QUALIFIERS.get(concept, set()):
                        evidence.append(triple)
                        evidence += self.instance(parent)
                        vocabulary.add(lemma(concept))
        if not set(words).issubset(vocabulary):
            return None, evidence
        return contract["object"], evidence

    def contexts(self, node):
        """Track semantic embedding without traversing entity coreference edges."""
        queue = deque([(node, [])])
        seen = set()
        plans, blockers, evidence = [], [], []
        while queue:
            current, path = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            if len(path) > 64:
                raise ValueError("Semantic embedding exceeds audit budget")
            for triple in self.incoming[current]:
                parent, relation, _ = triple
                concept = self.concepts.get(parent, "")
                wrapper = concept in {"and", "or", "contrast-01", "multi-sentence"}
                event_parent = bool(re.search(r"-\d\d$", concept))
                argument = relation in {":ARG1", ":ARG2", ":ARG3"}
                attachment = relation in {":concession", ":manner", ":time", ":purpose", ":condition"}
                traversable = (wrapper and (argument or re.fullmatch(r":(?:op|snt)\d+", relation))) or (
                    event_parent and (argument or attachment))
                if not traversable:
                    continue
                chain = path + [triple]
                if concept == "or":
                    blockers.append("disjunctive_embedding")
                    evidence += chain
                if concept in PLAN_FRAMES and relation == ":ARG1":
                    if self.targets(parent, ":polarity") == ["-"]:
                        blockers.append("negated_plan")
                    else:
                        plans.append(parent)
                        evidence += chain + self.instance(parent)
                if concept in NONASSERTIVE_FRAMES and argument:
                    blockers.append("nonassertive_embedding:" + concept)
                    evidence += chain + self.instance(parent)
                if relation in {":purpose", ":condition"}:
                    blockers.append("goal_or_conditional_embedding")
                    evidence += chain
                if self.targets(parent, ":polarity") == ["-"] and not wrapper:
                    blockers.append("negated_embedding:" + concept)
                    evidence += chain + [(parent, ":polarity", "-")]
                queue.append((parent, chain))
        if self.targets(node, ":condition"):
            blockers.append("conditional_transfer")
        return sorted(set(plans)), sorted(set(blockers)), evidence

    def event(self, node, contract):
        concept = self.concepts[node]
        receiver = concept == "receive-01"
        role_map = {"sender": ":ARG2", "recipient": ":ARG0"} if receiver else {
            "sender": ":ARG0", "recipient": ":ARG2"}
        facts, evidence, issues = {}, {}, []
        for field, relation in role_map.items():
            target = self.one(node, relation)
            value, proof = self.name(target, contract) if target else (None, [])
            facts[field] = value
            evidence[field] = ([(node, relation, target)] if target else []) + proof
            if value is None:
                issues.append(field + "_unbound_or_ambiguous")
        item_node = self.one(node, ":ARG1")
        item, proof = self.object(item_node, contract)
        facts["object"] = item
        evidence["object"] = ([(node, ":ARG1", item_node)] if item_node else []) + proof
        if item is None:
            issues.append("object_head_or_modifier_unresolved")
        plans, blockers, scope_evidence = self.contexts(node)
        negative = self.targets(node, ":polarity") == ["-"]
        yet = [t for _, r, t in self.out[node] if r == ":time" and self.concepts.get(t) == "yet"]
        state_evidence = self.instance(node) + scope_evidence
        if negative:
            state_evidence.append((node, ":polarity", "-"))
        for target in yet:
            state_evidence += [(node, ":time", target)] + self.instance(target)
        plan_agents = []
        for plan in plans:
            agent = self.one(plan, ":ARG0")
            name, proof = self.name(agent, contract) if agent else (None, [])
            plan_agents.append(name)
            state_evidence += ([(plan, ":ARG0", agent)] if agent else []) + proof
        plan_ok = bool(plans) and facts["sender"] is not None and all(
            value == facts["sender"] for value in plan_agents) and not blockers
        asserted = not plans and not blockers
        evidence["transfer_status"] = state_evidence
        return {
            "node": node, "concept": concept, "facts": facts, "evidence": evidence,
            "source_omitted": receiver and not self.targets(node, ":ARG2"),
            "negative": negative, "not_yet": bool(yet),
            "plans": plans, "plan_agent_consistent": plan_ok, "blockers": blockers,
            "asserted": asserted, "binding_issues": issues,
        }


def unique_triples(values):
    return [list(t) for t in dict.fromkeys(tuple(t) for t in values)]


def read_graph_facts(original_penman, contract):
    """Return fact candidates and a public-alphabet prediction, without labels."""
    import penman
    from penman.models.amr import model
    if set(contract) != {"role_one", "role_two", "object"} or any(
            not isinstance(v, str) or not v.strip() for v in contract.values()):
        raise ValueError("Only the public roles/object contract is accepted")
    if normalize(contract["role_one"]) == normalize(contract["role_two"]):
        raise ValueError("Public roles must be distinct")
    logs = []

    class Capture(logging.Handler):
        def emit(self, record):
            logs.append({"logger": record.name, "level": record.levelname, "message": record.getMessage()})

    handler = Capture(logging.WARNING)
    logger = logging.getLogger("penman")
    saved = (logger.level, logger.propagate)
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    try:
        # ::snt metadata is excluded from the reader, including literal text.
        graph_only = "\n".join(line for line in original_penman.splitlines()
                               if not line.lstrip().startswith("#"))
        decoded = penman.decode(graph_only, model=model)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(saved[0])
        logger.propagate = saved[1]
    graph = GraphIndex(decoded.triples, decoded.top)
    events = [graph.event(node, contract) for node, concept in graph.concepts.items()
              if concept in TRANSFER_FRAMES or concept == "receive-01"]
    gives = [e for e in events if e["concept"] in TRANSFER_FRAMES]
    receives = [e for e in events if e["concept"] == "receive-01"]
    proposals, conflicts = [], []

    def signature(event):
        f = event["facts"]
        if any(f[k] is None for k in ("sender", "recipient", "object")):
            return None
        return tuple(f[k] for k in ("sender", "recipient", "object"))

    for giving in gives:
        sig = signature(giving)
        if sig is None:
            continue
        if giving["asserted"] and not giving["negative"]:
            for receipt in receives:
                f = receipt["facts"]
                match = f["recipient"] == sig[1] and f["object"] == sig[2] and (
                    receipt["source_omitted"] or f["sender"] == sig[0])
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
    for sig, state, _ in proposals:
        if state == "completed" and any(signature(e) == sig and e["asserted"] and e["negative"] for e in gives):
            conflicts.append("receipt_and_denied_transfer_without_resolved_chronology")
    distinct = {(sig, state) for sig, state, _ in proposals}
    if len(distinct) > 1:
        conflicts.append("multiple_incompatible_transfer_states_or_directions")
    facts = dict.fromkeys(FIELDS, "unknown")
    facts.update(readability="unclear", explicit_status="unknown", contradiction=bool(conflicts))
    evidence = dict.fromkeys(FIELDS, None)
    evidence = {field: [] for field in FIELDS}
    reasons = []
    if len(distinct) == 1 and not conflicts:
        sig, state = next(iter(distinct))
        facts.update(sender=sig[0], recipient=sig[1], object=sig[2],
                     transfer_status=state, readability="clear", explicit_status="yes")
        chosen = [event for s, st, supporting in proposals if (s, st) == (sig, state) for event in supporting]
        for field in FIELDS:
            evidence[field] = unique_triples(t for event in chosen for t in event["evidence"][field])
    else:
        # Partial facts are retained for diagnosis without guessing missing roles.
        for field in ("sender", "recipient", "object"):
            values = {e["facts"][field] for e in gives if e["facts"][field] is not None}
            if len(values) == 1:
                facts[field] = next(iter(values))
                evidence[field] = unique_triples(t for e in gives if e["facts"][field] == facts[field]
                                                 for t in e["evidence"][field])
        reasons = conflicts or ["no_complete_transfer_state_with_bound_roles_and_object"]
    prediction = classify_facts(facts, contract)
    return {
        "version": VERSION, "facts": facts, "prediction": prediction.symbol,
        "prediction_reason": prediction.reason, "evidence_triples": evidence,
        "events": events, "reasons": reasons, "decode_diagnostics": logs,
        "original_penman_sha256": hashlib.sha256(original_penman.encode()).hexdigest(),
        "graph_only_sha256": hashlib.sha256(graph_only.encode()).hexdigest(),
        "normalized_triples": [list(t) for t in graph.triples],
        "human_confirmed": False, "reference_or_target_accessed": False,
    }


def compare_development(*, text_reference, graph_readback, public_contract,
                        structurally_eligible=True):
    """Conditional model-reference comparison; never calls it formal attribution."""
    text_sig = actual_signature(text_reference)
    graph_sig = actual_signature(graph_readback["facts"])
    text_symbol = classify_facts(text_reference, public_contract).symbol if text_sig else None
    raw_prediction = graph_readback["prediction"]
    prediction = raw_prediction if structurally_eligible else None
    matches = graph_sig == text_sig if text_sig is not None else None
    candidates = []
    if not structurally_eligible:
        candidates.append("parser_structure_audit_exclusion")
    elif text_sig is None:
        candidates.append("text_reference_unresolved")
    elif graph_sig is None:
        candidates.append("parser_or_graph_fact_reader_unresolved")
    elif not matches:
        candidates.append("parser_or_graph_fact_reader_disagrees")
    elif raw_prediction != text_symbol:
        candidates.append("graph_classifier_disagrees_on_matching_facts")
    return {
        "status": "DEVELOPMENT_COMPARISON_COMPLETE",
        "text_reference_symbol": text_symbol,
        "graph_only_prediction": raw_prediction, "primary_prediction": prediction,
        "primary_symbol_correct": prediction == text_symbol if text_symbol else False,
        "graph_facts_match_model_text_reference": matches,
        "candidate_failure_layers": candidates,
        "reference_source": "SEALED_MODEL_TEXT_FACTS_NOT_HUMAN_GOLD",
        "formal_attribution_confirmed": False,
    }
