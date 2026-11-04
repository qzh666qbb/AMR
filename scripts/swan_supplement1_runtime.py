"""Frozen official SWAN transformations, CUDA FP32 and warning capture."""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout, redirect_stderr
import importlib
import io
import logging
import math
import os
import re
import sys
import warnings

from swan_final1_common import digest, freeze, now
import run_swan_positive_collection as swan
from swan_cached_s2match import CachedS2Matcher
from swan_supplement1_sources import ROOT, OUT, WORD, tokenizer


class IntegrityHalt(RuntimeError):
    pass


def content_seed(*parts):
    return int(digest(["swan_count_neutral_supplement1", *parts])[:8], 16)


def checked(function):
    """Capture every logger and Python warning, including parser stderr."""
    def call(*args, **kwargs):
        messages, out, err = [], io.StringIO(), io.StringIO()
        class Capture(logging.Handler):
            def emit(self, record):
                messages.append(record.getMessage())
        handler = Capture(level=logging.WARNING)
        logging.getLogger().addHandler(handler)
        exception, result = None, None
        try:
            with warnings.catch_warnings(record=True) as caught, redirect_stdout(out), redirect_stderr(err):
                warnings.simplefilter("always")
                try:
                    result = function(*args, **kwargs)
                except Exception as exc:
                    exception = type(exc).__name__
                messages.extend(str(w.message) for w in caught)
            if re.search(r"(?i)\b(warning|error|failed|invalid)\b", out.getvalue() + err.getvalue()):
                messages.append("warning_or_error_in_output")
            if messages or exception:
                event = {"at": now(), "operation": function.__name__, "inputs": args,
                         "stdout": out.getvalue(), "stderr": err.getvalue(),
                         "warnings": messages, "exception_type": exception}
                freeze(OUT / "private/local_errors" / f"{digest(event)}.json", event)
                raise IntegrityHalt(f"local_{function.__name__}_warning_or_error")
            return result
        finally:
            logging.getLogger().removeHandler(handler)
    return call


@contextmanager
def runtime():
    import torch
    import nltk
    import amrlib
    import penman
    if not torch.cuda.is_available():
        raise IntegrityHalt("CUDA_required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", NLTK_DATA=str(ROOT / ".nltk_data"),
                      GLOVE_VECTORS_PATH=str(ROOT / "vectors/glove.6B.100d.txt"))
    if "utils.amr_utils" in sys.modules:
        raise IntegrityHalt("ambiguous_utils_import")
    sys.path.insert(0, str(swan.S2MATCH_RUNTIME))
    sys.path.insert(0, str(swan.OFFICIAL_ROOT))
    nltk.data.path.insert(0, str(ROOT / ".nltk_data"))
    loader, downloader = amrlib.load_stog_model, nltk.download
    def load_cuda(*args, **kwargs):
        return loader(model_dir=str(swan.PARSER_MODEL),
                      tok_name_or_path=str(swan.PARSER_TOKENIZER),
                      device="cuda", batch_size=1)
    def offline_download(name, *args, **kwargs):
        paths = {"propbank": ROOT / ".nltk_data/corpora/propbank.zip",
                 "punkt_tab": ROOT / ".nltk_data/tokenizers/punkt_tab/english/ortho_context.tab"}
        if name not in paths or not paths[name].is_file():
            raise IntegrityHalt("unexpected_NLTK_download")
        return True
    try:
        amrlib.load_stog_model, nltk.download = load_cuda, offline_download
        utils = importlib.import_module("utils.amr_utils")
    finally:
        amrlib.load_stog_model, nltk.download = loader, downloader
    if any(p.dtype != torch.float32 for p in utils.stog.model.parameters()):
        raise IntegrityHalt("parser_not_FP32")
    cached = CachedS2Matcher(swan.GLOVE_VECTORS)
    score = checked(cached.score)
    parse = checked(utils.parse_amr)
    normalize = checked(utils.normalize_amr_variables)
    template = checked(utils.generate_template_amr)
    def valid_parse(text):
        free, total = torch.cuda.mem_get_info()
        if 1 - free / total >= .8:
            raise IntegrityHalt("GPU_total_memory_80pct")
        graph = parse(text)
        decoded = penman.decode(graph)
        if not decoded.instances() or decoded.top is None:
            raise IntegrityHalt("invalid_parser_graph")
        return graph
    active = {"candidate": None}
    def candidate_parse(text):
        active["candidate"] = text
        return valid_parse(text)
    tokenize = tokenizer()
    def match(candidate, target, threshold):
        text = active["candidate"].strip()
        gate = 7 <= len(WORD.findall(text)) <= 12 and tokenize(text) == [text] \
            and text.endswith(".") and not text.endswith("..")
        p, r, f = score(candidate, target, content_seed("score", candidate, target))
        if not math.isfinite(f) or not 0 <= f <= 1:
            raise IntegrityHalt("invalid_S2Match_score")
        return gate and f >= threshold, float(f) if gate else 0.0
    module = swan.official_function_module_for_test({
        "BedrockManager": object, "parse_amr": candidate_parse,
        "normalize_amr_variables": normalize, "generate_template_amr": template,
        "decide_s2match_match": match,
    })
    def transform(text):
        return normalize(template(normalize(valid_parse(text))))
    module.transform = transform
    module.score = score
    module.utils = utils
    module.content_seed = content_seed
    try:
        yield module
    finally:
        torch.cuda.empty_cache()


def readiness(module, bank):
    import torch
    from utils.amr_utils import compute_s2match_from_strings
    texts = ["The engineer repaired a small device.", "The scientist carefully examined the new sample."]
    replay = [module.transform(texts[0]) for _ in range(3)]
    if replay[1:] != replay[:1] * 2:
        raise IntegrityHalt("parser_replay_drift")
    graph = module.transform(texts[1])
    pairs = [(replay[0], replay[0]), (graph, bank[0]), (graph, bank[17])]
    comparisons = []
    for candidate, target in pairs:
        seed = content_seed("score", candidate, target)
        with swan.fixed_score_seed(seed):
            expected = checked(compute_s2match_from_strings)(
                candidate, target, vectors_file=str(swan.GLOVE_VECTORS),
                similarity_function="cosine", cutoff=.5, diffsense=.5, mwp="split",
                do_not_mark_quotes=False, verbose=False)
        actual = module.score(candidate, target, seed)
        delta = max(abs(float(a) - float(b)) for a, b in zip(actual, expected))
        comparisons.append(delta)
        if delta > 1e-6:
            raise IntegrityHalt("cached_S2Match_not_equivalent")
    return {"at": now(), "readiness": "Go", "device": torch.cuda.get_device_name(),
            "precision": "FP32", "TF32": False, "parser_replays": 3,
            "parser_replay_equal": True, "cached_match_max_errors": comparisons,
            "allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
            "generation_calls": 0, "detector_calls": 0}
