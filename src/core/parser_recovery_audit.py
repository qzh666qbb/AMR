"""External audit of amrlib recovery, preserving parser outputs and beam choice."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import threading

from .sqlite_cache import normalize_text, text_sha256


AUDIT_VERSION = "amrlib-recovery-audit:v2-exact-metadata"
_LOCK = threading.RLock()


def bytes_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def extract_amr_graph(output: str, text: str) -> str:
    """Remove only amrlib's exact, unescaped input metadata prefix.

    Input text may contain newlines, comment markers or graph-looking text.
    Neither splitlines nor searching for the first '(' preserves this binding.
    Graph bytes (including trailing whitespace) must remain unchanged.
    """
    if output.startswith("# ::snt "):
        prefix = "# ::snt " + text + "\n"
        if not output.startswith(prefix):
            raise RuntimeError("Parser metadata differs from bound input")
        return output[len(prefix):]
    return output


class RecoveryHandler(logging.Handler):
    def __init__(self, state, events):
        super().__init__(logging.WARNING)
        self.state, self.events = state, events

    def emit(self, record):
        span_index = self.state.get("span_index")
        message = record.getMessage()
        if span_index is None:
            match = re.search(r"(?:Sentence number |snum=)(\d+)", message)
            if match:
                span_index = int(match.group(1))
        self.events.append({
            "phase": self.state.get("phase", "generation"),
            "span_index": span_index, "beam_index": self.state.get("beam_index"),
            "logger": record.name, "level": record.levelname, "message": message,
        })


@contextmanager
def capture_recovery_logs(state, events):
    handler = RecoveryHandler(state, events)
    saved = []
    with _LOCK:
        for name in ("amrlib", "penman"):
            logger = logging.getLogger(name)
            saved.append((logger, logger.level, logger.propagate))
            logger.addHandler(handler)
            logger.setLevel(logging.WARNING)
            logger.propagate = False
        try:
            yield
        finally:
            for logger, level, propagate in saved:
                logger.removeHandler(handler)
                logger.setLevel(level)
                logger.propagate = propagate


def parse_with_recovery_audit(model, texts, inference_module, allow_all_failed=False):
    """Instrument only logging around the original deserializer call."""
    if model.ret_raw_gen:
        raise ValueError("Raw-generation mode is incompatible with recovery audit")
    original = inference_module.PenmanDeSerializer
    events, attempts = [], []
    state = {"span_index": None, "beam_index": None, "phase": "generation"}
    next_span, next_beam = 0, 0

    def audited_deserializer(*args, **kwargs):
        nonlocal next_span, next_beam
        span_index, beam_index = next_span, next_beam
        if span_index >= len(texts):
            raise RuntimeError("Deserializer call cannot be bound to an input Span")
        state.update(span_index=span_index, beam_index=beam_index, phase="deserialize")
        before = len(events)
        try:
            recovered = original(*args, **kwargs)
            graph = recovered.get_graph_string()
        finally:
            state.update(span_index=None, beam_index=None, phase="generation")
        attempts.append({
            "span_index": span_index, "beam_index": beam_index,
            "raw_serialized_graph_sha256": bytes_sha(args[0]),
            "accepted": graph is not None,
            "graph_sha256": bytes_sha(graph) if graph is not None else None,
            "events": events[before:],
        })
        if graph is not None or beam_index + 1 == model.num_ret_seq:
            next_span, next_beam = span_index + 1, 0
        else:
            next_beam = beam_index + 1
        return recovered

    with _LOCK:
        inference_module.PenmanDeSerializer = audited_deserializer
        try:
            with capture_recovery_logs(state, events):
                outputs = model.parse_sents(list(texts))
        finally:
            inference_module.PenmanDeSerializer = original
    if len(outputs) != len(texts) or next_span != len(texts) or next_beam != 0:
        raise RuntimeError("Parser recovery audit did not cover all input Spans")
    unmapped = [event for event in events if event["span_index"] is None]
    if unmapped:
        raise RuntimeError(f"Unbound parser recovery diagnostics: {unmapped}")
    records = []
    for index, (text, output) in enumerate(zip(texts, outputs)):
        relevant = [attempt for attempt in attempts if attempt["span_index"] == index]
        accepted = [attempt for attempt in relevant if attempt["accepted"]]
        if (allow_all_failed and output is None and not accepted
                and len(relevant) == model.num_ret_seq
                and [a["beam_index"] for a in relevant] == list(range(model.num_ret_seq))):
            records.append({
                "version": AUDIT_VERSION, "text_sha256": text_sha256(text),
                "amr_output_sha256": None, "selected_beam": None, "attempts": relevant,
                "events": [event for event in events if event["span_index"] == index],
                "all_beams_failed": True,
            })
            continue
        if not output or len(accepted) != 1:
            raise RuntimeError("Parser recovery has no unique accepted beam")
        # The original parse_sents adds only ::snt metadata to the recovered graph.
        graph = extract_amr_graph(output, text)
        if bytes_sha(graph) != accepted[0]["graph_sha256"]:
            raise RuntimeError("Selected recovery beam differs from returned AMR")
        records.append({
            "version": AUDIT_VERSION, "text_sha256": text_sha256(text),
            "amr_output_sha256": bytes_sha(output),
            "selected_beam": accepted[0]["beam_index"], "attempts": relevant,
            "events": [event for event in events if event["span_index"] == index],
        })
    return tuple(outputs), records


class AuditedAmrlibBackend:
    """Delegate inference unchanged, persisting recovery provenance separately."""

    def __init__(self, backend, audit_path: Path):
        self.backend, self.audit_path = backend, audit_path
        self.records = {}
        if audit_path.exists():
            for line in audit_path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if (record["version"] != AUDIT_VERSION
                        or record["parser_fingerprint"] != self.fingerprint):
                    raise ValueError("Parser recovery journal version/fingerprint differs")
                self.records[record["text_sha256"]] = record

    @property
    def fingerprint(self):
        return self.backend.fingerprint

    def parse_batch(self, texts):
        import amrlib.models.parse_xfm.inference as inference_module
        model = self.backend._load()
        outputs, records = parse_with_recovery_audit(model, texts, inference_module)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as stream:
            for record in records:
                record.update(parser_fingerprint=self.fingerprint,
                              recorded_at=datetime.now(timezone.utc).isoformat())
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                self.records[record["text_sha256"]] = record
            stream.flush()
            os.fsync(stream.fileno())
        return outputs

    def assert_record(self, record):
        audit = self.records.get(record.text_sha256)
        if audit is None or audit["amr_output_sha256"] != bytes_sha(record.amr):
            raise RuntimeError("AMR cache lacks bound upstream recovery audit evidence")

    def summary(self):
        events = [event for record in self.records.values() for event in record["events"]]
        return {
            "version": AUDIT_VERSION, "unique_spans": len(self.records),
            "spans_with_events": sum(bool(r["events"]) for r in self.records.values()),
            "warning_events": sum(e["level"] == "WARNING" for e in events),
            "error_events": sum(e["level"] in ("ERROR", "CRITICAL") for e in events),
            "spans_with_error_events": sum(
                any(e["level"] in ("ERROR", "CRITICAL") for e in r["events"])
                for r in self.records.values()),
            "spans_with_selected_beam_errors": sum(
                any(e["level"] in ("ERROR", "CRITICAL")
                    and e["beam_index"] == r["selected_beam"]
                    for e in r["events"])
                for r in self.records.values()),
            "fallback_beam_spans": sum(r["selected_beam"] > 0 for r in self.records.values()),
        }


def attach_recovery_audit(detector, path: Path):
    original = detector.scorer.extractor.backend
    if not hasattr(original, "_load"):
        raise ValueError("Upstream recovery auditing requires the real amrlib backend")
    backend = AuditedAmrlibBackend(original, path)
    detector.scorer.extractor.backend = backend
    original_prepare = detector.scorer.prepare

    def audited_prepare(texts):
        normalized = tuple(normalize_text(text) for text in texts if normalize_text(text))
        original_prepare(normalized)
        for text in normalized:
            backend.assert_record(detector.scorer._records[text])

    detector.scorer.prepare = audited_prepare
    return backend
