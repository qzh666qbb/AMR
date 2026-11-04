"""Frozen official SWAN normalization, bound parser audit and seeded S2Match."""
from __future__ import annotations

import ast
from collections import Counter, defaultdict, deque
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from functools import lru_cache
import hashlib
import io
import logging
from pathlib import Path
import random
import re
import warnings

from .swan_blind_protocol import ROOT, digest

OFFICIAL = ROOT / "third_party/SWAN_official_2ca0c207_clean"
RUNTIME = ROOT / "third_party/swan_s2match_runtime_711a231d"
SETTINGS = {
    "dtype": "float32", "tf32": False, "beams": 4, "batch_size": 8,
    "seed": 20261023, "input": "exact-candidate-text-no-normalization",
    "score": {"similarity_function": "cosine", "cutoff": .5, "diffsense": .5,
              "mwp": "split", "do_not_mark_quotes": False,
              "hillclimbing_iterations": 5, "pair_seed": "sha256-graph-bank-v1"},
    "audit": "selected-beam-strict-v1",
}


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


class BoundAuditError(RuntimeError):
    pass


class EventHandler(logging.Handler):
    def __init__(self, state, events):
        super().__init__(logging.WARNING)
        self.state, self.events = state, events

    def emit(self, record):
        state = dict(self.state)
        message = record.getMessage()
        if state.get("input") is None:
            match = re.search(r"(?:Sentence number |snum=)(\d+)", message)
            if match:
                state["input"] = int(match.group(1))
            match = re.search(r"beam=(\d+)", message)
            if match:
                state["beam"] = int(match.group(1))
        self.events.append({**state, "logger": record.name,
                            "level": record.levelname, "message": message})


@contextmanager
def logs(state, events, names=("amrlib", "penman", "swan_official_normalizer")):
    handler, saved = EventHandler(state, events), []
    for name in names:
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


def graph_valid(graph):
    import penman
    g = penman.decode(graph)
    instances = [t for t in g.triples if t[1] == ":instance"]
    variables = [t[0] for t in instances]
    if not variables or g.top not in variables or len(set(variables)) != len(variables):
        return False
    if any(t[0] not in variables for t in g.triples):
        return False
    replay = penman.decode(penman.encode(g))
    return g.top == replay.top and Counter(g.triples) == Counter(replay.triples)


class OfficialNormalizer:
    def __init__(self):
        import nltk
        import penman
        import spacy
        nltk.data.path.insert(0, str(ROOT / ".nltk_data"))
        from nltk.corpus import propbank
        source = OFFICIAL / "utils/amr_utils.py"
        names = {"classify_concept", "generate_template_amr",
                 "normalize_amr_variables", "remove_amr_comments"}
        functions = [n for n in ast.parse(source.read_text()).body
                     if isinstance(n, ast.FunctionDef) and n.name in names]
        if {n.name for n in functions} != names:
            raise RuntimeError("Official normalization source changed")
        namespace = {
            "re": re, "penman": penman, "defaultdict": defaultdict, "deque": deque,
            "nlp": spacy.load("en_core_web_sm"),
            "verb_lemmas": {i.roleset.split(".")[0] for i in propbank.instances()},
            "verb_pattern": re.compile(r"(.*)-\d+$"),
            "logger": logging.getLogger("swan_official_normalizer"),
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
             namespace)
        namespace["classify_concept"] = lru_cache(maxsize=None)(
            namespace["classify_concept"])
        self.namespace = namespace

    def __call__(self, output):
        state, events = {"input": 0, "phase": "normalization"}, []
        result = {"graph": None, "normalization_ok": False, "events": events}
        if not output:
            return result
        n = self.namespace
        try:
            with logs(state, events), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                graph = n["remove_amr_comments"](output)
                raw_valid = graph_valid(graph)
                normalized = n["normalize_amr_variables"](graph)
                template = n["generate_template_amr"](normalized)
                final = n["normalize_amr_variables"](template)
                final_valid = graph_valid(final)
                for warning in caught:
                    events.append({**state, "level": "WARNING",
                                   "message": str(warning.message)})
            result.update(graph=final, normalization_ok=raw_valid and final_valid
                          and not events)
        except Exception as exc:
            result["exception_type"] = type(exc).__name__
        return result


class AuditedParser:
    def __init__(self, normalizer=None):
        import torch
        from amrlib.models.parse_xfm.inference import Inference
        if not torch.cuda.is_available():
            raise RuntimeError("Linux CUDA is required")
        random.seed(SETTINGS["seed"])
        torch.manual_seed(SETTINGS["seed"])
        torch.cuda.manual_seed_all(SETTINGS["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.normalizer = normalizer or OfficialNormalizer()
        self.model = Inference(
            model_dir=str(ROOT / "models/model_parse_xfm_bart_large-v0_1_0"),
            tok_name_or_path=str(ROOT / "models/bart-large-tokenizer"),
            device="cuda", batch_size=SETTINGS["batch_size"], num_beams=4)
        if any(p.dtype != torch.float32 for p in self.model.model.parameters()):
            raise RuntimeError("Parser dtype differs from frozen FP32")

    def token_counts(self, texts):
        return [len(ids) for ids in self.model.tokenizer(
            list(texts), truncation=False, padding=False)["input_ids"]]

    def parse(self, texts):
        import amrlib.models.parse_xfm.inference as module
        texts = list(texts)
        counts = self.token_counts(texts)
        if any(n > self.model.max_sent_len for n in counts):
            raise ValueError("Over-context inputs must be rejected before inference")
        original = module.PenmanDeSerializer
        events, attempts = [], []
        state = {"input": None, "beam": None, "phase": "generation"}
        next_input, next_beam = 0, 0

        def deserialize(*args, **kwargs):
            nonlocal next_input, next_beam
            index, beam = next_input, next_beam
            if index >= len(texts):
                raise BoundAuditError("Deserializer input cannot be bound")
            state.update(input=index, beam=beam, phase="deserialize")
            before = len(events)
            try:
                recovered = original(*args, **kwargs)
                graph = recovered.get_graph_string()
            finally:
                state.update(input=None, beam=None, phase="generation")
            attempts.append({"input": index, "beam": beam, "accepted": graph is not None,
                             "graph_sha256": text_sha(graph) if graph else None,
                             "events": events[before:]})
            if graph is not None or beam + 1 == self.model.num_ret_seq:
                next_input, next_beam = index + 1, 0
            else:
                next_beam += 1
            return recovered

        module.PenmanDeSerializer = deserialize
        try:
            with logs(state, events), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                outputs = self.model.parse_sents(texts)
            # Python warnings without input/beam context must fail closed.
            if caught:
                raise BoundAuditError("Unbound Python warning during parser inference")
        finally:
            module.PenmanDeSerializer = original
        if len(outputs) != len(texts) or (next_input, next_beam) != (len(texts), 0):
            raise BoundAuditError("Deserializer coverage differs from inputs")
        if any(type(e.get("input")) is not int or not 0 <= e["input"] < len(texts)
               or (e["phase"] == "deserialize" and e.get("beam") is None)
               for e in events):
            raise BoundAuditError("Unbound parser diagnostics")
        records = []
        for index, (text, output, tokens) in enumerate(zip(texts, outputs, counts)):
            trial = [a for a in attempts if a["input"] == index]
            accepted = [a for a in trial if a["accepted"]]
            selected = accepted[0]["beam"] if accepted else None
            graph = None
            if output:
                prefix = "# ::snt " + text + "\n"
                if not output.startswith(prefix):
                    raise BoundAuditError("Returned metadata differs from exact input")
                graph = output[len(prefix):]
                if len(accepted) != 1 or text_sha(graph) != accepted[0]["graph_sha256"]:
                    raise BoundAuditError("Returned graph differs from selected beam")
            elif accepted:
                raise BoundAuditError("Accepted beam missing from parser output")
            relevant = [e for e in events if e["input"] == index]
            selected_events = [e for e in relevant if e.get("beam") == selected
                               or e.get("beam") is None]
            normalized = self.normalizer(graph)
            reasons = []
            if not output:
                reasons.append("no_valid_selected_graph")
            if selected_events:
                reasons.append("selected_beam_diagnostic")
            if output and not normalized["normalization_ok"]:
                reasons.append("normalization_or_roundtrip_failure")
            records.append({
                "input_sha256": text_sha(text), "native_input_tokens": tokens,
                "selected_beam": selected, "strict_ok": not reasons, "reasons": reasons,
                "graph": normalized["graph"], "attempts": trial, "events": relevant,
                "normalization_events": normalized["events"],
                "blank_output": not bool(output),
            })
        return records


@contextmanager
def fixed_seed(seed):
    import numpy as np
    state, np_state, original = random.getstate(), np.random.get_state(), random.seed
    original(seed)
    np.random.seed(seed)

    def preserve(value=None, version=2):
        if value is not None:
            original(value, version=version)

    random.seed = preserve
    try:
        yield
    finally:
        random.seed = original
        random.setstate(state)
        np.random.set_state(np_state)


_MATCHER = None
_BANK = None


def init_matcher(bank):
    import sys
    sys.path.insert(0, str(RUNTIME))
    import s2match
    global _MATCHER, _BANK
    vectors = s2match.load_vecs(str(ROOT / "vectors/glove.6B.100d.txt"))
    s2match.load_vecs = lambda _: vectors
    _MATCHER, _BANK = s2match, bank


def score_graph(graph):
    """All bank entries are evaluated, including entries after an exact match."""
    scores, diagnostics = [], io.StringIO()
    try:
        with redirect_stderr(diagnostics), redirect_stdout(diagnostics), \
                warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for template in _BANK:
                seed = int(digest(["sha256-graph-bank-v1", graph, template])[:8], 16)
                with fixed_seed(seed):
                    value = _MATCHER.compute_s2match_from_strings(
                        graph, template, vectors_file="frozen-vector-cache",
                        similarity_function="cosine", cutoff=.5, diffsense=.5,
                        mwp="split", do_not_mark_quotes=False, verbose=False)[2]
                scores.append(float(value))
        import math
        if caught or diagnostics.getvalue() or len(scores) != 100 \
                or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
            return {"scores": None, "reason": "bank_scoring_diagnostic_or_invalid"}
        return {"scores": scores, "reason": None}
    except Exception as exc:
        return {"scores": None, "reason": "bank_scoring_" + type(exc).__name__}
