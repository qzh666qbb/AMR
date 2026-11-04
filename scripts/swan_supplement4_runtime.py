"""Candidate-scoped rejection and resumable CUDA execution for supplement4."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import importlib
import io
import logging
import math
import os
import sys
import warnings

from swan_final1_common import digest, freeze
from swan_supplement1_runtime import IntegrityHalt, content_seed
from swan_supplement1_sources import ROOT, WORD, tokenizer
from swan_cached_s2match import CachedS2Matcher
import run_swan_positive_collection as swan


class CandidateReject(RuntimeError):
    pass


class ResourcePause(RuntimeError):
    pass


def observed(function, out, candidate_id=None):
    """Only synchronous parser/normalizer diagnostics have candidate ownership."""
    def invoke(*args, **kwargs):
        messages, stdout, stderr = [], io.StringIO(), io.StringIO()
        class Capture(logging.Handler):
            def emit(self, record):
                messages.append(record.getMessage())
        handler = Capture(level=logging.WARNING)
        logging.getLogger().addHandler(handler)
        error, result = None, None
        try:
            with warnings.catch_warnings(record=True) as caught, \
                    redirect_stdout(stdout), redirect_stderr(stderr):
                warnings.simplefilter("always")
                try:
                    result = function(*args, **kwargs)
                except Exception as exc:
                    error = exc
                messages.extend(str(w.message) for w in caught)
        finally:
            logging.getLogger().removeHandler(handler)
        # Do not persist timestamps in immutable candidate events: replay is exact.
        if messages or stdout.getvalue() or stderr.getvalue() or error:
            event = {"operation": function.__name__, "candidate_id": candidate_id,
                     "input_sha256": digest(args), "warnings": messages,
                     "stdout": stdout.getvalue(), "stderr": stderr.getvalue(),
                     "exception_type": type(error).__name__ if error else None}
            freeze(out / "private/local_errors" / f"{digest(event)}.json", event)
            if error and ("OutOfMemoryError" in type(error).__name__ or
                          "CUDA out of memory" in str(error)):
                raise ResourcePause("CUDA_out_of_memory") from error
            if error and not isinstance(error, (ValueError, KeyError, IndexError)) \
                    and type(error).__module__.split(".")[0] != "penman":
                raise IntegrityHalt("unexpected_local_exception") from error
            if candidate_id is not None:
                raise CandidateReject(f"{function.__name__}_invalid_candidate") from error
            raise IntegrityHalt("unbound_local_diagnostic") from error
        return result
    return invoke


@contextmanager
def runtime(out):
    import torch
    import nltk
    import amrlib
    import penman
    if not torch.cuda.is_available():
        raise IntegrityHalt("CUDA_required")
    # Memory pressure is a scheduling condition, never a scientific result.
    free, _ = torch.cuda.mem_get_info()
    if free < 3072 * 1024**2:
        raise ResourcePause("parser_admission_requires_3072MiB_free")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", NLTK_DATA=str(ROOT / ".nltk_data"),
                      GLOVE_VECTORS_PATH=str(swan.GLOVE_VECTORS))
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
    score = observed(cached.score, out)
    module = swan.official_function_module_for_test({
        "BedrockManager": object, "parse_amr": utils.parse_amr,
        "normalize_amr_variables": utils.normalize_amr_variables,
        "generate_template_amr": utils.generate_template_amr,
        "decide_s2match_match": object,
    })
    def transform(text, candidate_id=None):
        free, _ = torch.cuda.mem_get_info()
        if free < 512 * 1024**2:
            raise ResourcePause("parse_headroom_requires_512MiB_free")
        graph = observed(utils.parse_amr, out, candidate_id)(text)
        try:
            decoded = penman.decode(graph)
            if not decoded.instances() or decoded.top is None:
                raise ValueError("missing_graph_root")
        except (ValueError, TypeError, penman.DecodeError) as exc:
            if candidate_id is None:
                raise IntegrityHalt("invalid_readiness_graph") from exc
            raise CandidateReject("invalid_parser_graph") from exc
        normalize = observed(utils.normalize_amr_variables, out, candidate_id)
        template = observed(utils.generate_template_amr, out, candidate_id)
        return normalize(template(normalize(graph)))
    tokenize = tokenizer()
    def evaluate(text, target, candidate_id):
        graph = transform(text, candidate_id)
        _, _, f = score(graph, target, content_seed("score", graph, target))
        if not math.isfinite(f) or not 0 <= f <= 1:
            raise IntegrityHalt("invalid_S2Match_score")
        gate = 7 <= len(WORD.findall(text)) <= 12 and tokenize(text) == [text] \
            and text.endswith(".") and not text.endswith("..")
        return {"accepted": bool(gate and f >= .75), "score": float(f) if gate else 0.0,
                "candidate_template": graph, "rejection": None if gate else "sentence_length_gate"}
    module.transform, module.score, module.evaluate = transform, score, evaluate
    try:
        yield module
    finally:
        torch.cuda.empty_cache()
