"""SWAN text-only planning and strict aggregation; no parser or detector calls.

The official two-stage statistic is separate from the complete-coverage adapter.
Calibration and native parser limits must be frozen before actual decisions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import Sequence

from src.core.multiscale_blind import windows

VERSION = "swan-blind-protocol/v1"
METHODS = ("P0", "S6", "S8", "S10", "S12")
ROOT = Path(__file__).resolve().parents[2]


def digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


@dataclass(frozen=True)
class ProtocolConfig:
    method: str
    batch_size: int = 32
    candidate_budget: int | None = None
    bank_size: int = 100
    bank_sha256: str | None = None
    parser_sha256: str | None = None
    parser_tokenizer_sha256: str | None = None
    normalizer_sha256: str | None = None
    matcher_sha256: str | None = None
    parser_context_tokens: int | None = None

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError("Unknown blind method")
        for name in ("batch_size", "bank_size", "candidate_budget",
                     "parser_context_tokens"):
            value = getattr(self, name)
            if value is None and name in ("candidate_budget", "parser_context_tokens"):
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("bank_sha256", "parser_sha256", "parser_tokenizer_sha256",
                     "normalizer_sha256", "matcher_sha256"):
            value = getattr(self, name)
            if value is not None and not _is_sha256(value):
                raise ValueError(f"{name} must be SHA256")

    @property
    def execution_frozen(self) -> bool:
        return self.parser_context_tokens is not None and all(
            getattr(self, name) is not None for name in (
                "bank_sha256", "parser_sha256", "parser_tokenizer_sha256",
                "normalizer_sha256", "matcher_sha256"))


@lru_cache(maxsize=1)
def _punkt():
    import nltk
    from nltk.tokenize import PunktTokenizer
    nltk.data.path.insert(0, str(ROOT / ".nltk_data"))
    return PunktTokenizer("english")


def segmenter_metadata() -> dict:
    """Pin local English Punkt resources without downloading anything."""
    import nltk
    _punkt()
    folder = Path(str(nltk.data.find("tokenizers/punkt_tab/english/")))
    names = ("abbrev_types.txt", "collocations.tab", "ortho_context.tab",
             "sent_starters.txt")
    resources = {name: file_sha256(folder / name) for name in names}
    result = {"library": "nltk", "version": nltk.__version__,
              "language": "english", "resources_sha256": resources}
    return {**result, "fingerprint": digest(result)}


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Plan:
    config: ProtocolConfig
    configuration_sha256: str
    text_sha256: str
    candidates: tuple[Candidate, ...]
    uncovered_nonwhitespace: int
    status: str
    reasons: tuple[str, ...]

    def batches(self) -> tuple[tuple[Candidate, ...], ...]:
        if self.status != "geometry_ready":
            return ()
        return tuple(self.candidates[i:i + self.config.batch_size]
                     for i in range(0, len(self.candidates), self.config.batch_size))


def configuration_fingerprint(config: ProtocolConfig) -> str:
    return digest({
        "version": VERSION, "config": asdict(config),
        "source_sha256": file_sha256(Path(__file__)),
        "window_source_sha256": file_sha256(ROOT / "src/core/multiscale_blind.py"),
        "segmenter": segmenter_metadata() if config.method == "P0" else None,
        "geometry": "punkt-full-text" if config.method == "P0"
                    else {"words": int(config.method[1:]), "stride": 1,
                          "token_rule": r"\S+", "include_tail": True},
    })


def plan_document(text: str, config: ProtocolConfig) -> Plan:
    """Accept attacked text and config only; never use slots, labels or targets."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(config, ProtocolConfig):
        raise TypeError("config must be ProtocolConfig")
    fingerprint = configuration_fingerprint(config)
    if config.method == "P0":
        raw = [{"start": a, "end": b, "text": text[a:b]}
               for a, b in _punkt().span_tokenize(text)]
    else:
        raw = windows(text, int(config.method[1:]), 1)
    candidates = tuple(Candidate(i, w["start"], w["end"], w["text"])
                       for i, w in enumerate(raw))
    # Union coverage, not sum of overlapping candidate lengths.
    covered = bytearray(len(text))
    for c in candidates:
        if not 0 <= c.start < c.end <= len(text) or c.text != text[c.start:c.end]:
            raise ValueError("Invalid candidate geometry")
        covered[c.start:c.end] = b"\1" * (c.end - c.start)
    missing = sum(not ch.isspace() and not covered[i] for i, ch in enumerate(text))
    reasons = []
    if not candidates:
        reasons.append("empty_text")
    if missing:
        reasons.append("incomplete_text_coverage")
    if config.candidate_budget is not None and len(candidates) > config.candidate_budget:
        reasons.append("candidate_budget_exceeded")
    return Plan(config, fingerprint, hashlib.sha256(text.encode()).hexdigest(),
                candidates, missing, "abstain" if reasons else "geometry_ready",
                tuple(reasons))


@dataclass(frozen=True)
class CandidateScores:
    candidate_id: int
    bank_scores: tuple[float | None, ...]
    configuration_sha256: str
    text_sha256: str
    native_input_tokens: int | None = None
    parse_ok: bool = True
    error: str | None = None


@dataclass(frozen=True)
class Calibration:
    configuration_sha256: str
    negative_manifest_sha256: str
    unit_threshold: float
    document_threshold: float
    document_statistic: str

    def __post_init__(self):
        if not _is_sha256(self.configuration_sha256) \
                or not _is_sha256(self.negative_manifest_sha256):
            raise ValueError("Calibration must bind config and frozen negatives")
        if not _valid_score(self.unit_threshold):
            raise ValueError("Invalid calibration unit threshold")
        if isinstance(self.document_threshold, bool) \
                or not isinstance(self.document_threshold, Real) \
                or not math.isfinite(self.document_threshold):
            raise ValueError("Invalid calibration document threshold")
        if self.document_statistic not in ("z", "green_fraction"):
            raise ValueError("Unknown calibration document statistic")
        if self.document_statistic == "green_fraction" \
                and not 0 <= self.document_threshold <= 1:
            raise ValueError("Invalid fraction threshold")


def _valid_score(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, Real) \
        and math.isfinite(value) and 0 <= value <= 1


def official_two_stage_summary(score_slots: Sequence[float | None]) -> dict:
    """Official counting only: None = blank AMR, retained in N as non-green."""
    if any(v is not None and not _valid_score(v) for v in score_slots):
        raise ValueError("Invalid official bank-max score")
    n = len(score_slots)
    g = sum(v is not None and float(v) >= 0.65 for v in score_slots)
    return {"N": n, "G": g, "green_fraction": g / n if n else None,
            "z": (g - 0.25 * n) / math.sqrt(n * 0.25 * 0.75) if n else None,
            "decision": None, "z_interpretation": "raw_statistic_not_p_value"}


def aggregate_document(plan: Plan, records: Sequence[CandidateScores],
                       calibration: Calibration | None = None) -> dict:
    """Require every candidate and every bank score; never retain a scored prefix."""
    n = len(plan.candidates)
    result = {"method": plan.config.method,
              "configuration_sha256": plan.configuration_sha256,
              "text_sha256": plan.text_sha256, "N_expected": n,
              "N_scored": 0, "status": "abstain", "reasons": [],
              "G": None, "green_fraction": None, "z": None, "decision": None,
              "include_in_document_denominator": True}
    if plan.status == "abstain":
        result["reasons"] = list(plan.reasons)
        return result
    ids = [r.candidate_id for r in records]
    if any(type(i) is not int for i in ids) or sorted(ids) != list(range(n)):
        result["reasons"] = ["missing_duplicate_or_unexpected_candidate"]
        return result
    values = []
    for record in sorted(records, key=lambda r: r.candidate_id):
        if record.configuration_sha256 != plan.configuration_sha256 \
                or record.text_sha256 != plan.text_sha256:
            result["reasons"] = ["score_provenance_mismatch"]
            return result
        if not record.parse_ok or record.error is not None:
            result["reasons"] = ["parse_or_scoring_failure"]
            return result
        if len(record.bank_scores) != plan.config.bank_size \
                or not all(_valid_score(v) for v in record.bank_scores):
            result["reasons"] = ["incomplete_or_invalid_bank_scores"]
            return result
        if plan.config.parser_context_tokens is not None:
            if type(record.native_input_tokens) is not int or record.native_input_tokens < 1:
                result["reasons"] = ["missing_native_token_count"]
                return result
            if record.native_input_tokens > plan.config.parser_context_tokens:
                result["reasons"] = ["parser_context_exceeded"]
                return result
        values.append(float(max(record.bank_scores)))
        result["N_scored"] += 1
    if not plan.config.execution_frozen:
        result.update(status="execution_unfrozen", reasons=["execution_not_frozen"])
        return result
    if calibration is not None:
        expected_statistic = "z" if plan.config.method == "P0" else "green_fraction"
        if calibration.configuration_sha256 != plan.configuration_sha256 \
                or calibration.document_statistic != expected_statistic \
                or (plan.config.method == "P0" and calibration.unit_threshold != 0.65):
            result["reasons"] = ["calibration_configuration_mismatch"]
            return result
    unit_threshold = 0.65 if plan.config.method == "P0" \
        else float(calibration.unit_threshold) if calibration is not None else None
    if unit_threshold is not None:
        g = sum(v >= unit_threshold for v in values)
        result.update(G=g, green_fraction=g / n)
        if plan.config.method == "P0":
            result["z"] = official_two_stage_summary(values)["z"]
    if calibration is None:
        result.update(status="uncalibrated", reasons=["missing_calibration"])
        return result
    statistic = result["z"] if plan.config.method == "P0" else result["green_fraction"]
    result.update(status="decided", decision=bool(statistic > calibration.document_threshold),
                  reasons=[], calibration_manifest_sha256=calibration.negative_manifest_sha256)
    return result


def summarize_decisions(results: Sequence[dict]) -> dict:
    """Unconditional decision rates retain abstained/unconfigured documents."""
    n = len(results)
    counts = {status: sum(r["status"] == status for r in results)
              for status in sorted({r["status"] for r in results})}
    positive = sum(r["decision"] is True for r in results)
    return {"documents": n, "status_counts": counts, "positive_decisions": positive,
            "positive_decision_rate_all_documents": positive / n if n else None,
            "unresolved_documents": sum(r["decision"] is None for r in results)}
