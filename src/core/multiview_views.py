"""文本片段和真实AMR事件两个不同粒度的视图。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re

import numpy as np

from .amr_extractor import decode_amr_with_audit
from .sqlite_cache import normalize_text


ENCODER_ID = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
_EVENT = re.compile(r".+-\d\d$")
_NUMBER = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
_SMALL_NUMBERS = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(),
    map(str, range(21))))


def numeric_value(text):
    return text if _NUMBER.fullmatch(text) else _SMALL_NUMBERS.get(text.lower())


@dataclass(frozen=True)
class Event:
    event_id: str
    candidate_index: int
    node: str
    predicate: str
    roles: tuple
    negated: bool
    entities: tuple
    numbers: tuple
    description: str
    signature: str
    missing_core_roles: bool

    def to_dict(self):
        return asdict(self)


def extract_events(amr: str, candidate_index: int) -> list:
    graph, _, valid, roundtrip = decode_amr_with_audit(amr)
    if not valid or not roundtrip:
        return []
    concepts = {str(t.source): str(t.target) for t in graph.instances()}
    outgoing = {node: [] for node in concepts}
    for source, role, target in graph.triples:
        source, role, target = str(source), str(role), str(target)
        if role == ":instance":
            continue
        if role.endswith("-of") and target in concepts:
            source, role, target = target, role[:-3], source
        outgoing[source].append((role, target))

    def label(node):
        if node not in concepts:
            return node.strip('"')
        names = [target for role, target in outgoing[node] if role == ":name"]
        if names:
            operands = sorted(
                ((int(role[3:]), target.strip('"'))
                 for name in names for role, target in outgoing.get(name, [])
                 if re.fullmatch(r":op\d+", role)), key=lambda item: item[0])
            return concepts[node] + "[" + " ".join(value for _, value in operands) + "]"
        return concepts[node]

    def detail(node, depth=2, visited=frozenset()):
        if node not in concepts or depth == 0 or node in visited:
            return label(node)
        parts = [label(node)]
        for role, target in sorted(outgoing[node]):
            if role in (":name", ":wiki"):
                continue
            parts.append(role + "=" + detail(target, depth - 1, visited | {node}))
        return " ".join(parts)

    def literals(node, visited=frozenset()):
        if node not in concepts or node in visited:
            return set(), set()
        entities, numbers = set(), set()
        if any(role == ":name" for role, _ in outgoing[node]):
            entities.add(label(node).split("[", 1)[1].rstrip("]").lower())
        for role, target in outgoing[node]:
            if target in concepts:
                ent, num = literals(target, visited | {node})
                entities.update(ent)
                numbers.update(num)
            elif _NUMBER.fullmatch(target):
                numbers.add(target)
        return entities, numbers

    result = []
    for node, predicate in concepts.items():
        if not _EVENT.fullmatch(predicate):
            continue
        roles = tuple((role, detail(target)) for role, target in sorted(outgoing[node])
                      if role != ":wiki")
        negated = (":polarity", "-") in outgoing[node]
        entities, numbers = literals(node)
        description = predicate[:-3].replace("-", " ")
        if negated:
            description = "not " + description
        description += " " + " ".join(f"{role[1:]} {value}" for role, value in roles
                                      if role != ":polarity")
        full_roles = tuple((role, detail(target, len(concepts)))
                           for role, target in sorted(outgoing[node]) if role != ":wiki")
        identity = {"predicate": predicate, "roles": full_roles, "negated": negated,
                    "entities": sorted(entities), "numbers": sorted(numbers)}
        signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        result.append(Event(
            event_id=f"c{candidate_index}:e{len(result)}", candidate_index=candidate_index,
            node=node, predicate=predicate, roles=roles, negated=negated,
            entities=tuple(sorted(entities)), numbers=tuple(sorted(numbers)),
            description=description.strip(), signature=signature,
            missing_core_roles=not any(role.startswith(":ARG") for role, _ in roles),
        ))
    return result


class SemanticEncoder:
    def __init__(self, cache_folder: str, device="cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            ENCODER_ID, revision=ENCODER_REVISION, device=device,
            cache_folder=cache_folder, local_files_only=True, token=False)
        self.fingerprint = ENCODER_ID + "@" + ENCODER_REVISION
        self.memo = {}

    def encode(self, texts):
        normalized = [normalize_text(text) for text in texts]
        missing = list(dict.fromkeys(text for text in normalized if text not in self.memo))
        if missing:
            tokens = self.model.tokenizer(missing, truncation=False, add_special_tokens=True)
            lengths = [len(ids) for ids in tokens["input_ids"]]
            if any(length > self.model.max_seq_length for length in lengths):
                raise ValueError("候选/事件描述超过编码器上限；禁止静默截断")
            vectors = self.model.encode(missing, batch_size=32, normalize_embeddings=True,
                                        show_progress_bar=False)
            if not np.isfinite(vectors).all():
                raise ValueError("编码器输出非有限向量")
            self.memo.update(zip(missing, vectors))
        if not normalized:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        return np.stack([self.memo[text] for text in normalized])


class SurfaceAnalyzer:
    """spaCy规则是可出错的约束线索，输出不作为人工真值。"""
    def __init__(self):
        import spacy
        self.nlp = spacy.load("en_core_web_sm")
        self.fingerprint = "en_core_web_sm@" + self.nlp.meta["version"] + ":surface-v1"

    def analyze_many(self, texts):
        results = []
        for doc in self.nlp.pipe(texts):
            clauses = []
            for token in doc:
                if token.pos_ not in ("VERB", "AUX"):
                    continue
                args = {}
                passive = any(child.dep_ in ("nsubjpass", "auxpass") for child in token.children)
                for child in token.children:
                    role = None
                    if child.dep_ == "nsubj":
                        role = ":ARG0"
                    elif child.dep_ in ("nsubjpass", "dobj", "obj"):
                        role = ":ARG1"
                    if role:
                        args[role] = " ".join(t.text.lower() for t in child.subtree)
                    if child.dep_ == "agent":
                        objects = [t for t in child.children if t.dep_ == "pobj"]
                        if objects:
                            args[":ARG0"] = " ".join(t.text.lower() for t in objects[0].subtree)
                clauses.append({
                    "predicate": token.lemma_.lower(), "roles": args,
                    "numbers": sorted({numeric_value(t.text) for t in token.subtree
                                       if numeric_value(t.text) is not None}),
                    "negated": any(c.dep_ == "neg" or c.lower_ in ("never", "not")
                                   for c in token.children),
                    "passive": passive,
                })
            results.append({
                "entities": sorted({ent.text.lower() for ent in doc.ents}),
                "numbers": sorted({numeric_value(t.text) for t in doc
                                   if numeric_value(t.text) is not None}),
                "clauses": clauses,
            })
        return results
