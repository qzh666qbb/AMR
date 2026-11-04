"""Exact official injection with final1 identities and offline local resources."""
from __future__ import annotations

from contextlib import contextmanager
import logging
import os
import sys

import run_swan_positive_collection as swan
from swan_final1_budget import DocumentFailure, MODEL, URL
from swan_final1_common import OUT, digest, freeze, now, seed

BASE_INSTRUMENT = swan.instrument_official_injection


def validate_graph(text):
    import penman
    if not isinstance(text, str) or not text.strip():
        raise DocumentFailure("empty_parser_graph")
    try:
        graph = penman.decode(text)
        if graph.top is None or not graph.instances():
            raise ValueError("Empty graph")
    except Exception as exc:
        raise DocumentFailure("invalid_parser_graph") from exc
    return text


@contextmanager
def final1_instrument(module, folder, allocation_id, slot, bank, journal):
    """Add final1 R/T provenance without modifying the preserved old wrapper."""
    with BASE_INSTRUMENT(module, folder, allocation_id, slot, bank, journal) as trace:
        saved = {name: getattr(module, name)
                 for name in ("normalize_amr_variables", "generate_template_amr")}

        def normalize(amr):
            event = trace["current_event"]
            try:
                value = saved["normalize_amr_variables"](amr)
            except Exception as exc:
                event.update(status="normalize_failed", error_type=type(exc).__name__)
                swan.freeze_json(folder / "generation_attempts" / f"{event['call_id']}.json",
                                 event)
                raise
            event.setdefault("normalized_amrs", []).append(value)
            event.setdefault("normalized_amr_sha256", []).append(swan.text_sha256(value))
            return value

        def template(amr):
            event = trace["current_event"]
            try:
                value = saved["generate_template_amr"](amr)
            except Exception as exc:
                event.update(status="template_failed", error_type=type(exc).__name__)
                swan.freeze_json(folder / "generation_attempts" / f"{event['call_id']}.json",
                                 event)
                raise
            event["candidate_template_before_final_normalization"] = value
            event["candidate_template_before_final_normalization_sha256"] = swan.text_sha256(value)
            return value

        module.normalize_amr_variables = normalize
        module.generate_template_amr = template
        try:
            yield trace
        finally:
            for name, value in saved.items():
                setattr(module, name, value)


@contextmanager
def official_runtime(folder=OUT, error_folder=None):
    """No import/download/load occurs until explicitly entering this context."""
    import nltk
    modules_before, path_before = set(sys.modules), list(sys.path)
    required = {
        "propbank": swan.NLTK_DATA / "corpora/propbank.zip",
        "punkt_tab": swan.NLTK_DATA / "tokenizers/punkt_tab/english/ortho_context.tab",
    }
    if any(not p.is_file() for p in required.values()):
        raise FileNotFoundError("Frozen local NLTK assets missing")
    old_download = nltk.download
    bindings = {
        "SOURCE_MANIFEST": folder / "source_manifest.json", "CONTRACT": folder / "manifest.json",
        "MODEL": MODEL, "REGION": URL, "MAX_NEW_TOKENS": 50,
        "MAX_TRIALS": 50, "MAX_TEMPLATES": 10, "RETRIES_PER_TEMPLATE": 5,
        "MATCH_THRESHOLD": .75, "TEMPERATURE": .7, "DECLARED_TOP_P": .9,
        "slot_seed": lambda allocation, slot: seed("injection-template", allocation, slot),
        "score_seed": lambda allocation, slot, attempt: seed(
            "injection-score", allocation, slot, attempt),
        "freeze_json": freeze, "instrument_official_injection": final1_instrument,
    }
    saved = {k: getattr(swan, k) for k in bindings}
    environment = {
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "GLOVE_VECTORS_PATH": str(swan.GLOVE_VECTORS),
        "NLTK_DATA": str(swan.NLTK_DATA), "TOKENIZERS_PARALLELISM": "false",
    }
    old_env = {k: os.environ.get(k) for k in environment}

    def local_download(package, *args, **kwargs):
        if package not in required or not required[package].is_file():
            raise RuntimeError(f"Unfrozen NLTK download prohibited: {package}")
        return True

    def checked(function, graph_output=False):
        def run(*args, **kwargs):
            warnings = []
            class Capture(logging.Handler):
                def emit(self, record):
                    warnings.append(record.getMessage())
            handler = Capture(level=logging.WARNING)
            logger = logging.getLogger("utils.amr_utils")
            logger.addHandler(handler)
            try:
                result = function(*args, **kwargs)
                if warnings:
                    raise DocumentFailure(f"local_{function.__name__}_logged_warning")
                return validate_graph(result) if graph_output else result
            except Exception as exc:
                event = {"at": now(), "operation": function.__name__, "inputs": args,
                         "warnings": warnings, "error_type": type(exc).__name__}
                destination = error_folder if error_folder is not None else folder / "private/local_errors"
                freeze(destination / f"{digest(event)}.json", event)
                if isinstance(exc, DocumentFailure):
                    raise
                raise DocumentFailure(f"local_{function.__name__}_failed") from exc
            finally:
                logger.removeHandler(handler)
        return run

    try:
        nltk.download = local_download
        os.environ.update(environment)
        for k, v in bindings.items():
            setattr(swan, k, v)
        module = swan.load_official_injection()
        for name in ("parse_amr", "normalize_amr_variables", "generate_template_amr"):
            setattr(module, name, checked(getattr(module, name), graph_output=True))
        module.decide_s2match_match = checked(module.decide_s2match_match)
        yield module
    finally:
        nltk.download = old_download
        for k, v in saved.items():
            setattr(swan, k, v)
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        sys.path[:] = path_before
        for name in list(sys.modules):
            if name not in modules_before and (name == "utils" or name.startswith("utils.")):
                del sys.modules[name]


def score_triple(candidate, bank_graph, score_seed):
    """Exactly the official string entry and parameters, preserving P/R/F1."""
    from utils.amr_utils import compute_s2match_from_strings
    with swan.fixed_score_seed(score_seed):
        return compute_s2match_from_strings(
            candidate, bank_graph, vectors_file=str(swan.GLOVE_VECTORS),
            similarity_function="cosine", cutoff=.5, diffsense=.5, mwp="split",
            do_not_mark_quotes=False, verbose=False,
        )
