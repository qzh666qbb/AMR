from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Counter as CounterType
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .amr_extractor import AMRExtractor, decode_amr_with_audit
from .models import AMRRecord, BankMatch
from .sqlite_cache import normalize_text


TEMPLATE_VERSION = "swan-template-features:v2"
_SENSE_RE = re.compile(r".+-\d+$")
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")
_NAME_OPERAND_RE = re.compile(r"^:op\d+$")
_TEMPLATE_CONCEPTS = {"ADJ", "LOC", "N", "NE", "NUM", "V", "X"}
_NUMERIC_CONCEPTS = {
    "date-entity",
    "monetary-quantity",
    "ordinal-entity",
    "percentage-entity",
    "temporal-quantity",
}
_LOCATION_CONCEPTS = {"city", "continent", "country", "province", "state"}

Feature = Tuple[str, ...]


@dataclass(frozen=True)
class TemplateFeatures:
    concepts: CounterType[Feature]
    relations: CounterType[Feature]
    paths: CounterType[Feature]


def _concept_class(concept: str, named_entity: bool) -> str:
    if concept in _TEMPLATE_CONCEPTS:
        return concept
    if named_entity:
        return "NE"
    if concept == "amr-unknown":
        return "X"
    if concept in _NUMERIC_CONCEPTS:
        return "NUM"
    if concept in _LOCATION_CONCEPTS:
        return "LOC"
    if concept.endswith("-91"):
        return "N"
    if _SENSE_RE.fullmatch(concept):
        return "V"
    return "N"


def _constant_class(value: object, role: str) -> str:
    text = str(value)
    if text == "NE" or _NAME_OPERAND_RE.fullmatch(role):
        return "CONST_NE"
    if text in {"+", "-"}:
        return "CONST_SIGN"
    if _NUMBER_RE.fullmatch(text):
        return "CONST_NUM"
    return "CONST"


def template_features(amr: str) -> TemplateFeatures:
    """Convert raw or templated AMR into variable-invariant structural features."""
    graph, _, graph_valid, roundtrip_valid = decode_amr_with_audit(amr)
    if not graph_valid or not roundtrip_valid:
        raise ValueError("SWAN Bank AMR failed PENMAN validity audit")
    instances = {
        str(instance.source): str(instance.target)
        for instance in graph.instances()
    }
    named_nodes = {
        str(source)
        for source, role, _ in graph.triples
        if str(role) == ":name"
    }
    classes = {
        variable: _concept_class(concept, variable in named_nodes)
        for variable, concept in instances.items()
    }
    concepts: CounterType[Feature] = Counter(
        (concept_class,) for concept_class in classes.values()
    )
    relations: CounterType[Feature] = Counter()
    outgoing: Mapping[str, List[Tuple[str, str]]] = defaultdict(list)

    for source, role, target in graph.triples:
        role_text = str(role)
        if role_text == ":instance":
            continue
        source_text = str(source)
        target_text = str(target)
        source_class = classes.get(source_text)
        if source_class is None:
            continue
        target_class = classes.get(target_text)
        if target_class is None:
            relations[
                (source_class, role_text, _constant_class(target, role_text))
            ] += 1
            continue
        relations[(source_class, role_text, target_class)] += 1
        outgoing[source_text].append((role_text, target_text))

    paths: CounterType[Feature] = Counter()
    for source, first_edges in outgoing.items():
        for first_role, middle in first_edges:
            for second_role, target in outgoing.get(middle, ()):
                paths[
                    (
                        classes[source],
                        first_role,
                        classes[middle],
                        second_role,
                        classes[target],
                    )
                ] += 1

    return TemplateFeatures(
        concepts=concepts,
        relations=relations,
        paths=paths,
    )


def _multiset_f1(
    left: CounterType[Feature],
    right: CounterType[Feature],
) -> float:
    left_total = sum(left.values())
    right_total = sum(right.values())
    if left_total == 0 and right_total == 0:
        return 1.0
    if left_total == 0 or right_total == 0:
        return 0.0
    overlap = sum((left & right).values())
    return 2.0 * overlap / (left_total + right_total)


class SwanAMRBankScorer:
    """Fast, deterministic structural scorer over an official SWAN AMR Bank."""

    def __init__(
        self,
        extractor: AMRExtractor,
        bank_path: str,
        concept_weight: float = 0.25,
        relation_weight: float = 0.50,
        path_weight: float = 0.25,
    ) -> None:
        weights = (concept_weight, relation_weight, path_weight)
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("Bank scorer weights must be non-negative and non-zero")
        total = sum(weights)
        self.concept_weight = concept_weight / total
        self.relation_weight = relation_weight / total
        self.path_weight = path_weight / total
        self.extractor = extractor
        self.bank_path = Path(bank_path).expanduser().resolve()
        bank_bytes = self.bank_path.read_bytes()
        self.bank_sha256 = hashlib.sha256(bank_bytes).hexdigest()
        rows = json.loads(bank_bytes.decode("utf-8"))
        if not isinstance(rows, list) or not rows:
            raise ValueError("SWAN AMR Bank must be a non-empty JSON list")

        templates: List[str] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"Invalid SWAN Bank row {index}")
            template = row.get("normalized_template_amr")
            if not isinstance(template, str) or not template.strip():
                raise ValueError(f"Missing normalized_template_amr at row {index}")
            templates.append(template)
        if len(set(templates)) != len(templates):
            raise ValueError("SWAN AMR Bank contains duplicate templates")

        self.templates = tuple(templates)
        self._bank_features = tuple(
            template_features(template) for template in self.templates
        )
        self._records: Dict[str, AMRRecord] = {}
        config = {
            "bank_sha256": self.bank_sha256,
            "concept_weight": self.concept_weight,
            "path_weight": self.path_weight,
            "relation_weight": self.relation_weight,
            "template_version": TEMPLATE_VERSION,
        }
        self.fingerprint = "swan-bank-scorer:" + hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest()

    @property
    def parser_fingerprint(self) -> str:
        return self.extractor.backend.fingerprint

    def prepare(self, texts: Iterable[str]) -> None:
        missing = []
        for text in texts:
            normalized = normalize_text(text)
            if normalized not in self._records:
                missing.append(normalized)
        self._records.update(self.extractor.extract_many(missing))

    def clear_prepared(self) -> None:
        self._records.clear()

    def _record(self, text: str) -> AMRRecord:
        normalized = normalize_text(text)
        record = self._records.get(normalized)
        if record is None:
            record = self.extractor.extract(normalized)
            self._records[normalized] = record
        return record

    def score_features(self, candidate: TemplateFeatures) -> BankMatch:
        best = BankMatch(
            template_index=-1,
            score=float("-inf"),
            concept_score=0.0,
            relation_score=0.0,
            path_score=0.0,
        )
        for index, bank_features in enumerate(self._bank_features):
            concept_score = _multiset_f1(
                candidate.concepts, bank_features.concepts
            )
            relation_score = _multiset_f1(
                candidate.relations, bank_features.relations
            )
            path_score = _multiset_f1(candidate.paths, bank_features.paths)
            score = (
                self.concept_weight * concept_score
                + self.relation_weight * relation_score
                + self.path_weight * path_score
            )
            if score > best.score:
                best = BankMatch(
                    template_index=index,
                    score=score,
                    concept_score=concept_score,
                    relation_score=relation_score,
                    path_score=path_score,
                )
        return best

    def score_amr(self, amr: str) -> BankMatch:
        return self.score_features(template_features(amr))

    def score(self, span_text: str) -> BankMatch:
        return self.score_amr(self._record(span_text).amr)

    def score_many(self, span_texts: Sequence[str]) -> Tuple[BankMatch, ...]:
        self.prepare(span_texts)
        return tuple(self.score(text) for text in span_texts)
