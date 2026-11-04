"""Fresh, batched parser with source-bound diagnostics and exact batch replay."""
from dataclasses import asdict
import warnings

from .amr_extractor import decode_amr_with_audit
from .embedding_amr import GraphObservation, digest, windows
from .embedding_amr_runtime import FreshParser, atomic
from .parser_recovery_audit import extract_amr_graph, parse_with_recovery_audit


class BatchFreshParser(FreshParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fingerprint = digest({
            "parent": self.fingerprint, "execution": "within-document-batches/v1",
            "all_failed_beams": "bound_erasure",
        })
        self.batches = 0

    def _many(self, texts):
        import amrlib.models.parse_xfm.inference as module
        result, eligible = {}, []
        for text in texts:
            if len(self.model.tokenizer(text, truncation=False)["input_ids"]) >= self.model.max_sent_len:
                result[text] = (GraphObservation(digest(text), None, self.fingerprint, False,
                                                ("parser_native_input_limit",)), {})
            else:
                eligible.append(text)
        if eligible:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                outputs, records = parse_with_recovery_audit(
                    self.model, eligible, module, allow_all_failed=True)
            if caught:
                raise RuntimeError("Unbound Python warning in batched parser")
            for text, output, audit in zip(eligible, outputs, records):
                if output is None:
                    result[text] = (GraphObservation(digest(text), None, self.fingerprint, False,
                                                    ("all_beams_failed",)), audit)
                    continue
                raw = extract_amr_graph(output, text)
                _, diagnostics, valid, roundtrip = decode_amr_with_audit(raw)
                reasons = []
                if not valid or not roundtrip:
                    reasons.append("invalid_graph_or_roundtrip")
                if diagnostics or audit["events"]:
                    reasons.append("parser_or_penman_diagnostic")
                if audit["selected_beam"] != 0:
                    reasons.append("nonzero_selected_beam")
                result[text] = (GraphObservation(digest(text), raw, self.fingerprint,
                                                not reasons, tuple(reasons)), audit)
        return result

    def observe(self, text, config, window_config):
        candidates, _ = windows(text, config, window_config)
        missing = list(dict.fromkeys(c["text"] for c in candidates
                                     if digest(c["text"]) not in self.memo))
        if missing:
            result = self._many(missing)
            # First two document batches replay the complete batch, including
            # diagnostics, not just one easy sentence.
            if self.batches < 2:
                if result != self._many(missing):
                    raise RuntimeError("Batched parser exact replay failed")
                self.precheck_passed = True
            self.batches += 1
            for fragment, (observation, audit) in result.items():
                key = digest(fragment)
                atomic(self.output / (key + ".json"),
                       {"observation": asdict(observation), "audit": audit})
                self.memo[key] = observation
        return [self.memo[digest(c["text"])] for c in candidates]
