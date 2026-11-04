"""Pinned native AMRBART inference with explicit recovery and truncation evidence."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import importlib
import io
import json
import logging
from pathlib import Path
import re
import sys
import time
import warnings

MODEL = "models/amrbart-amr3-v2-12f7e78e3dfa"
SOURCE = "third_party/AMRBART-37e5742411e74d4269c33e8f2c5f5a8a0ed51241"
GENERATION = {
    "num_beams": 5, "num_return_sequences": 1, "do_sample": False,
    "max_length": 1024, "min_length": 0, "no_repeat_ngram_size": 0,
    "length_penalty": 1., "early_stopping": True, "use_cache": True,
    "decoder_start_token_id": 53226, "eos_token_id": 53227,
    "forced_eos_token_id": None, "forced_bos_token_id": None,
    "pad_token_id": 1,
}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


@contextmanager
def capture(events, phase):
    """Bind all diagnostics to this one synchronous sentence invocation."""
    class Handler(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                events.append({"phase": phase, "level": record.levelname,
                               "logger": record.name, "message": record.getMessage()})
    handler, output, errors, saved = Handler(), io.StringIO(), io.StringIO(), []
    for name in ("penman", "transformers", "common", "model_interface"):
        logger = logging.getLogger(name)
        saved.append((logger, logger.level, logger.propagate))
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        logger.propagate = False
    try:
        with warnings.catch_warnings(record=True) as caught, redirect_stdout(output), redirect_stderr(errors):
            warnings.simplefilter("always")
            yield
        for item in caught:
            events.append({"phase": phase, "level": "WARNING",
                           "logger": item.category.__name__, "message": str(item.message)})
    finally:
        for stream, name in ((output, "stdout"), (errors, "stderr")):
            if stream.getvalue().strip():
                events.append({"phase": phase, "level": "WARNING", "logger": name,
                               "message": stream.getvalue()})
        for logger, level, propagate in saved:
            logger.removeHandler(handler)
            logger.setLevel(level)
            logger.propagate = propagate


def native_tokenizer(code, model):
    path = (code / SOURCE / "fine-tune").resolve()
    # Fail on module-name collisions instead of silently importing another "common".
    for name in ("common", "model_interface"):
        module = sys.modules.get(name)
        if module is not None and not any(
                str(Path(p).resolve()).startswith(str(path)) for p in getattr(module, "__path__", [])):
            raise RuntimeError("Native AMRBART import namespace collision")
    sys.path.insert(0, str(path))
    mod = importlib.import_module("model_interface.tokenization_bart")
    with redirect_stdout(io.StringIO()):
        tok = mod.AMRBartTokenizer.from_pretrained(str(model), local_files_only=True)
    if (tok.modified != 0 or len(tok) != 53228 or tok.amr_bos_token_id != 53226
            or tok.amr_eos_token_id != 53227):
        raise RuntimeError("Native AMRBART vocabulary changed")
    return tok


def unified_input(tokenizer, text):
    ids = tokenizer(text, truncation=False, padding=False)["input_ids"]
    return ids + [tokenizer.amr_bos_token_id, tokenizer.mask_token_id, tokenizer.amr_eos_token_id]


def strict_graph(nodes):
    """Decode pointer syntax without inserting/deleting structure or fixing triples.

    This is a conservative recovery diagnostic, not a replacement decoder.
    The original native graph is always preserved.
    """
    import penman
    from penman.models.amr import model
    nodes = list(nodes or [])
    # Native BPE collapse can attach the terminal marker to the final ")".
    if nodes and isinstance(nodes[-1], str) and nodes[-1].endswith("</s>") and nodes[-1] != "</s>":
        nodes[-1] = nodes[-1][:-4]
    if nodes and nodes[0] == "<s>":
        nodes = nodes[1:]
    if nodes and nodes[-1] == "</s>":
        nodes = nodes[:-1]
    pieces, depth, declarations = [], 0, set()
    for node in nodes:
        value = str(node)
        if value == "(":
            if depth == 0 and pieces:
                raise ValueError("Multiple roots/trailing tokens")
            depth += 1
        elif value == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced close")
        match = re.fullmatch(r"<pointer:(\d+)>", value)
        if match:
            var = "z" + match.group(1)
            if pieces and pieces[-1] == "(":
                if var in declarations:
                    raise ValueError("Repeated variable declaration")
                declarations.add(var)
                pieces.extend([var, "/"])
            else:
                pieces.append(var)
        elif node is None or value.startswith("<"):
            raise ValueError("Unknown/damaged node token")
        else:
            pieces.append(value)
    if depth != 0 or not pieces or pieces[0] != "(" or pieces[-1] != ")":
        raise ValueError("Missing structural tokens")
    # penman.decode accepts some malformed missing values; reject those explicitly.
    graph = penman.decode(" ".join(pieces), model=model)
    instances = list(graph.instances())
    if (not instances or graph.top not in graph.variables()
            or any(x is None or y is None for x, _, y in graph.triples)
            or len(instances) != len(graph.variables())):
        raise ValueError("Incomplete graph")
    return graph


def decode_record(tokenizer, ids, text, freeze_sha, n_tokens):
    import penman
    from penman.models.amr import model
    ids = [int(i) for i in ids]
    while ids and ids[-1] == tokenizer.pad_token_id:
        ids.pop()
    # A forced terminal at the context boundary is never evidence of completeness.
    clipped = len(ids) >= 1024 or not ids or ids[-1] != tokenizer.amr_eos_token_id
    start_valid = bool(ids) and ids[0] == tokenizer.amr_bos_token_id
    decoded = list(ids)
    if decoded:
        decoded[0] = tokenizer.bos_token_id
    decoded = [tokenizer.eos_token_id if i == tokenizer.amr_eos_token_id else i for i in decoded]
    events, decode_events = [], []
    with capture(events, "native_decode"):
        graph, status, (nodes, backrefs) = tokenizer.decode_amr(decoded, restore_name_ops=False)
    status = status.name
    amr, valid, strict_ok, strict_reason = None, False, False, None
    with capture(decode_events, "roundtrip_and_recovery_audit"):
        try:
            amr = penman.encode(graph, model=model)
            rebuilt = penman.decode(amr, model=model)
            valid = (bool(graph.instances()) and graph.top in graph.variables()
                     and graph.top == rebuilt.top and sorted(graph.triples) == sorted(rebuilt.triples))
            before = strict_graph(nodes)
            strict_ok = before.top == graph.top and sorted(before.triples) == sorted(graph.triples)
            if not strict_ok:
                strict_reason = "native_structure_differs_from_strict_pointer_syntax"
        except Exception as exc:
            strict_reason = type(exc).__name__
    reasons = []
    if status != "OK":
        reasons.append("native_status_" + status)
    if not strict_ok:
        reasons.append("native_recovery_or_unverified_syntax")
    if tokenizer.unk_token_id in ids:
        reasons.append("unknown_token_replaced_by_native_decoder")
    if not start_valid:
        reasons.append("wrong_decoder_start")
    decode_events.extend({"phase": "recovery_audit", "level": "WARNING", "message": s} for s in reasons)
    return {
        "case_id": digest(text), "text_sha256": digest(text), "freeze_sha256": freeze_sha,
        "amr": amr, "amr_sha256": digest(amr) if amr else None,
        "valid": bool(valid and status != "BACKOFF" and not clipped),
        "attempts": [{"beam_index": 0, "accepted": bool(amr),
                      "graph_sha256": digest(amr) if amr else None}],
        "events": events, "decode_events": decode_events, "clipped": clipped,
        "beam_lengths": [{"length": len(ids), "eos": bool(ids) and ids[-1] == 53227}],
        "token_length": n_tokens, "native_status": status, "generated_token_ids": ids,
        "native_nodes": nodes, "native_backreferences": backrefs,
        "strict_syntax_matches": strict_ok, "strict_failure_type": strict_reason,
    }


def replay_signature(record):
    return {k: v for k, v in record.items() if k not in ("seconds", "cuda_peak_bytes")}


class Parser:
    def __init__(self, assets, out, contract):
        import torch
        from transformers import BartForConditionalGeneration, GenerationConfig
        import run_multiscale_transfer as rt
        self.torch, self.rt, self.out, self.contract = torch, rt, out, contract
        self.freeze_sha = rt.sha(out / "FREEZE.json")
        if not torch.cuda.is_available():
            raise RuntimeError("Linux CUDA required for experiment inference")
        rt.gpu_capacity(out, contract, "before_amrbart_load", 3072)
        for name, expected in contract["amrbart_assets"].items():
            if rt.sha(assets / name) != expected:
                raise RuntimeError("AMRBART model SHA changed")
        torch.set_num_threads(2)
        torch.manual_seed(20261102)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        self.tokenizer = native_tokenizer(out / "private/code", assets / MODEL)
        events = []
        with capture(events, "load"):
            self.model = BartForConditionalGeneration.from_pretrained(
                str(assets / MODEL), local_files_only=True, torch_dtype=torch.float32,
                attn_implementation="eager")
        if events:
            rt.atomic(out / "private/AMRBART_LOAD_EVENTS.json", events)
            raise RuntimeError("Unbound AMRBART load diagnostic")
        if (self.model.config.max_position_embeddings != 1024
                or self.model.config.vocab_size != len(self.tokenizer)):
            raise RuntimeError("Native model dimensions changed")
        self.model.to("cuda:0").eval()
        if not all(p.dtype == torch.float32 for p in self.model.parameters()):
            raise RuntimeError("FP32 required")
        self.generation = GenerationConfig(**GENERATION)

    def fresh(self, text):
        ids = unified_input(self.tokenizer, text)
        if len(ids) > 1024:
            import run_swan_independent_banks1 as bank
            return {**bank.excluded(text, self.freeze_sha, len(ids)),
                    "native_status": "INPUT_LIMIT", "strict_syntax_matches": False}
        self.rt.gpu_capacity(self.out, self.contract, "before_amrbart_parse", 512)
        start, events = time.monotonic(), []
        self.torch.cuda.reset_peak_memory_stats()
        tensor = self.torch.tensor([ids], device="cuda:0")
        with self.torch.inference_mode(), capture(events, "generation"):
            output = self.model.generate(
                input_ids=tensor, attention_mask=self.torch.ones_like(tensor),
                generation_config=self.generation)
        if len(output) != 1:
            raise RuntimeError("AMRBART output cardinality changed")
        result = decode_record(self.tokenizer, output[0].tolist(), text, self.freeze_sha, len(ids))
        result["events"] = events + result["events"]
        return {**result, "seconds": time.monotonic() - start,
                "cuda_peak_bytes": self.torch.cuda.max_memory_allocated()}
