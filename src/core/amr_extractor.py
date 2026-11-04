from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import re
import threading
from typing import Any, Dict, Generator, Iterable, List, Protocol, Sequence, Tuple

from .models import AMR_AUDIT_VERSION, AMRRecord, Triple
from .sqlite_cache import AMRCache, normalize_text, text_sha256


_SENSE_SUFFIX_RE = re.compile(r"-\d+$")
_WORD_RE = re.compile(r"\b[a-z][a-z0-9_-]*\b", re.IGNORECASE)
_CORE_ROLE_RE = re.compile(r"^:ARG\d+(?:-of)?$")
_STOP_WORDS = {
    "a", "an", "and", "at", "in", "is", "it", "of", "on", "or", "the", "to",
}
_PENMAN_LOG_LOCK = threading.RLock()


class _PenmanWarningHandler(logging.Handler):
    def __init__(self, phase: str, records: List[str]) -> None:
        super().__init__(level=logging.WARNING)
        self.phase = phase
        self.records = records

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(
            f"{self.phase}|{record.name}|{record.levelname}|{record.getMessage()}"
        )


@contextmanager
def _capture_penman_warnings(
    phase: str,
    records: List[str],
) -> Generator[None, None, None]:
    logger = logging.getLogger("penman")
    handler = _PenmanWarningHandler(phase, records)
    with _PENMAN_LOG_LOCK:
        previous_level = logger.level
        previous_propagate = logger.propagate
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        logger.propagate = False
        try:
            yield
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
            logger.propagate = previous_propagate


class ParserBackend(Protocol):
    @property
    def fingerprint(self) -> str:
        ...

    def parse_batch(self, texts: Sequence[str]) -> Sequence[str]:
        ...


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AmrlibBackend:
    """Lazy amrlib sentence-to-graph backend."""

    def __init__(
        self,
        model_dir: str,
        tokenizer_dir: str,
        device: str = "cpu",
        batch_size: int = 8,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.tokenizer_dir = Path(tokenizer_dir).expanduser().resolve()
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._fingerprint = self._build_fingerprint()

    def _build_fingerprint(self) -> str:
        required = (
            self.model_dir / "amrlib_meta.json",
            self.model_dir / "model_parse_xfm_bart_large.json",
            self.tokenizer_dir / "tokenizer.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "AMR model assets are incomplete: " + ", ".join(missing)
            )
        payload = {
            "backend": "amrlib",
            "metadata": {path.name: _file_digest(path) for path in required},
        }
        return "amrlib:" + hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def _load(self):
        if self._model is None:
            try:
                import amrlib
            except ImportError as exc:
                raise RuntimeError(
                    "amrlib is required for the real AMR parser backend"
                ) from exc
            self._model = amrlib.load_stog_model(
                model_dir=str(self.model_dir),
                tok_name_or_path=str(self.tokenizer_dir),
                device=self.device,
                batch_size=self.batch_size,
            )
        return self._model

    def parse_batch(self, texts: Sequence[str]) -> Sequence[str]:
        if not texts:
            return ()
        outputs = self._load().parse_sents(list(texts))
        if len(outputs) != len(texts):
            raise RuntimeError(
                f"AMR parser returned {len(outputs)} graphs for {len(texts)} texts"
            )
        if any(not output or not output.strip() for output in outputs):
            raise RuntimeError("AMR parser returned an empty graph")
        return tuple(outputs)


class HeuristicBackend:
    """Deterministic PENMAN backend for local logic tests only."""

    @property
    def fingerprint(self) -> str:
        return "heuristic-penman:v1"

    def parse_batch(self, texts: Sequence[str]) -> Sequence[str]:
        return tuple(self._parse(text) for text in texts)

    @staticmethod
    def _parse(text: str) -> str:
        words = [
            word.lower()
            for word in _WORD_RE.findall(text)
            if word.lower() not in _STOP_WORDS
        ]
        if not words:
            return "(x0 / empty)"
        lines = [f"(x0 / {words[0]}"]
        for index, word in enumerate(words[1:], start=1):
            lines.append(f"    :ARG{min(index, 5)} (x{index} / {word})")
        lines[-1] += ")"
        return "\n".join(lines)


@dataclass
class ExtractorStats:
    cache_hits: int = 0
    cache_misses: int = 0
    parser_calls: int = 0
    parsed_texts: int = 0
    parsed_with_warnings: int = 0
    penman_warning_records: int = 0


def _canonical_graph(graph: Any) -> Tuple[str, Tuple[Tuple[str, str, str], ...]]:
    top = "" if graph.top is None else str(graph.top)
    triples = tuple(
        sorted(
            (
                "" if source is None else str(source),
                "" if role is None else str(role),
                "" if target is None else str(target),
            )
            for source, role, target in graph.triples
        )
    )
    return top, triples


def _graph_is_valid(graph: Any) -> bool:
    instances = tuple(graph.instances())
    if graph.top is None or not instances or not graph.triples:
        return False
    variables = {
        str(instance.source)
        for instance in instances
        if instance.source is not None
    }
    if str(graph.top) not in variables:
        return False
    if any(
        instance.source is None
        or instance.target is None
        or not str(instance.target).strip()
        for instance in instances
    ):
        return False
    return all(
        edge.source is not None and str(edge.source) in variables
        for edge in graph.edges()
    )


def decode_amr_with_audit(
    amr: str,
) -> Tuple[Any, Tuple[str, ...], bool, bool]:
    """Decode once, capture PENMAN recovery warnings, and verify round-trip."""
    try:
        import penman
    except ImportError as exc:
        raise RuntimeError("penman is required to decode AMR graphs") from exc

    warnings: List[str] = []
    with _capture_penman_warnings("decode", warnings):
        graph = penman.decode(amr)
    graph_valid = _graph_is_valid(graph)

    roundtrip_valid = False
    if graph_valid:
        try:
            with _capture_penman_warnings("roundtrip_encode", warnings):
                encoded = penman.encode(graph)
            with _capture_penman_warnings("roundtrip_decode", warnings):
                recovered = penman.decode(encoded)
            roundtrip_valid = (
                _graph_is_valid(recovered)
                and _canonical_graph(recovered) == _canonical_graph(graph)
            )
        except Exception as exc:
            warnings.append(
                "roundtrip|penman|ERROR|"
                f"{type(exc).__name__}: {str(exc).splitlines()[-1]}"
            )
    return graph, tuple(warnings), graph_valid, roundtrip_valid


def amr_features(amr: str) -> Tuple[Tuple[str, ...], Tuple[Triple, ...]]:
    graph, _, graph_valid, roundtrip_valid = decode_amr_with_audit(amr)
    if not graph_valid:
        raise ValueError("PENMAN graph failed structural validity audit")
    if not roundtrip_valid:
        raise ValueError("PENMAN graph failed encode/decode round-trip audit")
    variable_to_concept = {
        str(instance.source): _SENSE_SUFFIX_RE.sub("", str(instance.target))
        for instance in graph.instances()
    }
    concepts = tuple(sorted(set(variable_to_concept.values())))
    triples: List[Triple] = []
    for edge in graph.edges():
        role = str(edge.role)
        if not _CORE_ROLE_RE.match(role):
            continue
        source = variable_to_concept.get(str(edge.source))
        target = variable_to_concept.get(str(edge.target))
        if source and target:
            triples.append((source, role, target))
    return concepts, tuple(sorted(set(triples)))


class AMRExtractor:
    """Batch AMR extraction with a parser-aware persistent cache."""

    def __init__(
        self,
        backend: ParserBackend,
        cache: AMRCache,
        request_batch_size: int = 128,
    ) -> None:
        if request_batch_size < 1:
            raise ValueError("request_batch_size must be positive")
        self.backend = backend
        self.cache = cache
        self.request_batch_size = request_batch_size
        self.stats = ExtractorStats()

    def extract(self, text: str) -> AMRRecord:
        return self.extract_many((text,))[normalize_text(text)]

    def extract_many(self, texts: Iterable[str]) -> Dict[str, AMRRecord]:
        normalized_texts = []
        seen = set()
        for text in texts:
            normalized = normalize_text(text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            normalized_texts.append(normalized)

        records: Dict[str, AMRRecord] = {}
        missing: List[str] = []
        for text in normalized_texts:
            cached = self.cache.get(text, self.backend.fingerprint)
            if cached is None:
                self.stats.cache_misses += 1
                missing.append(text)
            else:
                self.stats.cache_hits += 1
                records[text] = cached

        for start in range(0, len(missing), self.request_batch_size):
            batch = missing[start:start + self.request_batch_size]
            self.stats.parser_calls += 1
            outputs = self.backend.parse_batch(batch)
            if len(outputs) != len(batch):
                raise RuntimeError("Parser backend violated the batch-size contract")
            parsed_records = []
            for text, amr in zip(batch, outputs):
                (
                    graph,
                    penman_warnings,
                    graph_valid,
                    roundtrip_valid,
                ) = decode_amr_with_audit(amr)
                if not graph_valid:
                    raise ValueError(
                        "PENMAN graph failed structural validity audit for "
                        f"text {text_sha256(text)}"
                    )
                if not roundtrip_valid:
                    raise ValueError(
                        "PENMAN graph failed encode/decode round-trip audit for "
                        f"text {text_sha256(text)}"
                    )
                variable_to_concept = {
                    str(instance.source): _SENSE_SUFFIX_RE.sub(
                        "", str(instance.target)
                    )
                    for instance in graph.instances()
                }
                concepts = tuple(sorted(set(variable_to_concept.values())))
                triples: List[Triple] = []
                for edge in graph.edges():
                    role = str(edge.role)
                    if not _CORE_ROLE_RE.match(role):
                        continue
                    source = variable_to_concept.get(str(edge.source))
                    target = variable_to_concept.get(str(edge.target))
                    if source and target:
                        triples.append((source, role, target))
                record = AMRRecord(
                    text=text,
                    text_sha256=text_sha256(text),
                    parser_fingerprint=self.backend.fingerprint,
                    amr=amr,
                    concepts=concepts,
                    triples=tuple(sorted(set(triples))),
                    audit_version=AMR_AUDIT_VERSION,
                    penman_warnings=penman_warnings,
                    graph_valid=graph_valid,
                    roundtrip_valid=roundtrip_valid,
                )
                parsed_records.append(record)
                records[text] = record
                if penman_warnings:
                    self.stats.parsed_with_warnings += 1
                    self.stats.penman_warning_records += len(penman_warnings)
            self.cache.put_many(parsed_records)
            self.stats.parsed_texts += len(parsed_records)

        return records
