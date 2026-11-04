"""Construct independent private Banks from disjoint, precommitted human sources."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import fcntl
import hmac
import importlib.util
import json
import logging
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_multiscale_transfer as rt
from semantic_fact_raw_amr import audit_raw, replay_signature

OUT = ROOT / "experiments/swan_independent_banks1"
BANK_ROLES = ("public_bank", "target_a_bank", "target_b_bank")
CODE = ("scripts/run_swan_independent_banks1.py", "scripts/test_swan_independent_banks1.py",
        "scripts/run_multiscale_transfer.py", "scripts/semantic_fact_raw_amr.py",
        "scripts/run_research_completion_queue.py", "src/core/parser_recovery_audit.py",
        "src/__init__.py", "src/core/__init__.py")
OFFICIAL = "third_party/SWAN_official_2ca0c207_clean/utils/amr_utils.py"


def immutable(path, value):
    if path.exists():
        if rt.read(path) != value:
            raise RuntimeError("Immutable artifact changed: " + path.name)
    else:
        rt.atomic(path, value)


def choose_templates(records, seed, size=50):
    """No watermark scores or cross-Bank membership enter key construction."""
    import penman
    counts = Counter(r["template"] for r in records if r["eligible"])
    accepted = [template for template, n in counts.items()
                if 1 <= n <= 20 and len(penman.decode(template).instances()) >= 3]
    ordered = sorted(accepted, key=lambda text: (
        hmac.new(bytes.fromhex(seed), text.encode(), "sha256").hexdigest(), text))
    return {"eligible_unique": len(accepted), "sufficient": len(accepted) >= size,
            "bank": [{"normalized_template_amr": template, "count": counts[template]}
                     for template in ordered[:size]],
            "unique_valid_templates": len(counts)}


def prepare(out):
    import nltk
    import spacy
    import swan_independent_banks1_sources as sources
    if (out / "FREEZE.json").exists():
        raise RuntimeError("Already frozen; create a new version for changes")
    if out != sources.OUT:
        raise ValueError("Use fixed source directory")
    sources.audit()
    previous = rt.read(ROOT / "experiments/swan_path0_completion1/FREEZE.json")
    private = out / "private"
    all_code = tuple(sorted(set(CODE + sources.CODE)))
    for name in all_code:
        path = private / "code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    assets = [OFFICIAL, ".nltk_data/corpora/propbank.zip"]
    assets += [name for name in previous["dependencies"] if name.startswith(".nltk_data/tokenizers/")]
    for name in assets:
        original = ROOT / name
        if rt.sha(original) != previous["dependencies"][name]:
            raise RuntimeError("Historical asset drift: " + name)
        target = private / "code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
    nltk.data.path.insert(0, str(private / "code/.nltk_data"))
    tokenizer = nltk.tokenize.PunktTokenizer("english")
    spec = importlib.util.find_spec("en_core_web_sm")
    model_root = Path(next(iter(spec.submodule_search_locations)))
    models = list(model_root.glob("en_core_web_sm-*"))
    if len(models) != 1:
        raise RuntimeError("Unique installed surface model required")
    target = private / "surface_model"
    if target.exists():
        existing = {str(p.relative_to(target)): rt.sha(p) for p in target.rglob("*") if p.is_file()}
        expected = {str(p.relative_to(models[0])): rt.sha(p) for p in models[0].rglob("*") if p.is_file()}
        if existing != expected:
            raise RuntimeError("Surface model snapshot changed")
    else:
        shutil.copytree(models[0], target)
    documents = []
    for role in BANK_ROLES:
        for row in rt.read(private / "cohorts" / (role + ".json")):
            documents.append({"source_id": row["allocation_id"], "role": role,
                              "text_sha256": row["window_sha256"],
                              "sentences": tokenizer.tokenize(row["text"])})
    immutable(private / "BANK_INPUTS.json", documents)
    transfer = ROOT / "第二篇_语义消息写入与读回/02_实验/semantic_sequence_stage1_run2/multiscale_transfer4_recovery1"
    parser = rt.read(transfer / "FREEZE.json")["parser"]
    frozen_files = [out / name for name in ("PROTOCOL.md", "SOURCE_CONTRACT.json",
                                            "INPUT_MANIFEST.json", "SOURCE_AUDIT.json")]
    frozen_files += [p for p in private.rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts]
    immutable(out / "FREEZE.json", {
        "version": "independent-source-bank-construction/v1", "at": rt.now(),
        "files": {str(p.relative_to(out)): rt.sha(p) for p in sorted(frozen_files)},
        "code_sha256": {p: rt.sha(private / "code" / p) for p in all_code},
        "parser": parser, "execution": {"resumable_gpu_pressure": True},
        "resource_gate": {"vram_fraction_max": .8},
        "normalizer": {"upstream_file_sha256": rt.sha(ROOT / OFFICIAL),
                       "spacy": spacy.__version__, "nltk": nltk.__version__,
                       "surface_model": "en_core_web_sm-3.8.0",
                       "pure_functions_from_verified_source": True},
        "counts": {"sources": 300, "cohorts": 3, "templates_requested_each": 50,
                   "sentence_occurrences": sum(len(r["sentences"]) for r in documents),
                   "unique_parse_texts": len({s for r in documents for s in r["sentences"]})},
        "detector_calls": 0, "provider_calls": 0, "attack_results": None,
    })
    return verify(out)


def verify(out):
    f = rt.read(out / "FREEZE.json")
    for name, expected in f["files"].items():
        if rt.sha(out / name) != expected:
            raise RuntimeError("Frozen file drift: " + name)
    return f


def excluded(text, freeze_sha, n):
    return {"case_id": rt.digest(text), "text_sha256": rt.digest(text),
            "amr": None, "amr_sha256": None, "valid": False, "token_length": n,
            "events": [], "attempts": [], "decode_events": [], "beam_lengths": [],
            "clipped": False, "seconds": 0., "freeze_sha256": freeze_sha,
            "reason": "native_input_limit"}


def parse(out, assets, f, documents):
    seal = out / "RAW_GRAPH_SEAL.json"
    if seal.exists():
        saved = rt.read(seal)
        if saved["freeze_sha256"] != rt.sha(out / "FREEZE.json"):
            raise RuntimeError("Raw seal contract differs")
        for name, expected in saved["files"].items():
            if rt.sha(out / name) != expected:
                raise RuntimeError("Sealed graph changed")
        return
    parser = rt.Parser(assets, out, f)
    texts = sorted({s for r in documents for s in r["sentences"]}, key=rt.digest)
    probes = []
    for text in texts:
        if len(parser.model.tokenizer(text, truncation=False)["input_ids"]) >= 1024:
            continue
        pair = [{**parser.fresh(text), "case_id": rt.digest(text)} for _ in range(2)]
        if replay_signature(pair[0]) != replay_signature(pair[1]):
            raise RuntimeError("Two raw replays differ")
        probes.append({"text_sha256": rt.digest(text), "two_replays_equal": True})
        if len(probes) == 2:
            break
    if len(probes) != 2:
        raise RuntimeError("Insufficient admissible precheck inputs")
    immutable(out / "GPU_PRECHECK.json", {"dtype": "float32", "probes": probes})
    for index, text in enumerate(texts):
        if shutil.disk_usage(out).free < 512 * 1024**2:
            raise rt.ResourcePause("Disk headroom insufficient")
        path = out / "private/raw_graphs" / (rt.digest(text) + ".json")
        if path.exists():
            record = rt.read(path)
        else:
            n = len(parser.model.tokenizer(text, truncation=False)["input_ids"])
            record = (excluded(text, parser.freeze_sha, n) if n >= 1024 else
                      {**parser.fresh(text), "case_id": rt.digest(text)})
            immutable(path, record)
        audit_raw(record, {"case_id": rt.digest(text), "input_text_sha256": rt.digest(text)},
                  parser.freeze_sha)
        rt.atomic(out / "HEARTBEAT.json", {"phase": "bank_source_parsing",
                                          "completed": index + 1, "total": len(texts), "at": rt.now()})
    immutable(seal, {"freeze_sha256": parser.freeze_sha,
                     "files": {str(p.relative_to(out)): rt.sha(p)
                               for p in sorted((out / "private/raw_graphs").glob("*.json"))}})
    del parser
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()


class Normalizer:
    def __init__(self, out):
        import nltk
        import penman
        import spacy
        f = rt.read(out / "FREEZE.json")["normalizer"]
        if nltk.__version__ != f["nltk"] or spacy.__version__ != f["spacy"]:
            raise RuntimeError("Normalizer package version changed")
        nltk.data.path.insert(0, str(out / "private/code/.nltk_data"))
        self.events = []
        class Handler(logging.Handler):
            def emit(_, record):
                if record.levelno >= logging.WARNING:
                    self.events.append({"level": record.levelname, "message": record.getMessage()})
        logger = logging.Logger("independent-bank-normalizer", level=logging.WARNING)
        logger.addHandler(Handler())
        self.ns = {"penman": penman, "nlp": spacy.load(out / "private/surface_model"),
                   "verb_lemmas": {i.roleset.split(".")[0] for i in nltk.corpus.propbank.instances()},
                   "verb_pattern": re.compile(r"(.*)-\d+$"), "re": re, "logger": logger,
                   "defaultdict": defaultdict, "deque": deque}
        rt.isolate(out / "private/code" / OFFICIAL, [
            "classify_concept", "generate_template_amr", "normalize_amr_variables",
            "remove_amr_comments"], self.ns)

    def __call__(self, raw):
        self.events.clear()
        clean = self.ns["remove_amr_comments"](raw)
        final = self.ns["normalize_amr_variables"](
            self.ns["generate_template_amr"](self.ns["normalize_amr_variables"](clean)))
        graph = self.ns["penman"].decode(final)
        rebuilt = self.ns["penman"].decode(self.ns["penman"].encode(graph))
        valid = (not self.events and bool(graph.instances()) and graph.top in graph.variables()
                 and graph.top == rebuilt.top and sorted(graph.triples) == sorted(rebuilt.triples))
        return {"template": final, "eligible": valid, "events": list(self.events)}


def construct(out, documents):
    normalizer, cache, rows = Normalizer(out), {}, {}
    contract = rt.sha(out / "FREEZE.json")
    for i, doc in enumerate(documents):
        records = []
        for text in doc["sentences"]:
            identity = rt.digest(text)
            if identity not in cache:
                raw = rt.read(out / "private/raw_graphs" / (identity + ".json"))
                checked = audit_raw(raw, {"case_id": identity, "input_text_sha256": identity}, contract)
                record = (normalizer(raw["amr"]) if checked["structurally_eligible"] else
                          {"template": None, "eligible": False, "events": checked["exclusion_reasons"]})
                if record["eligible"] and normalizer(raw["amr"]) != record:
                    raise RuntimeError("Template normalization replay differs")
                cache[identity] = {**record, "text_sha256": identity}
            records.append(cache[identity])
        rows[doc["source_id"]] = {"role": doc["role"], "records": records}
        rt.atomic(out / "HEARTBEAT.json", {"phase": "template_normalization",
                                          "completed": i + 1, "total": len(documents), "at": rt.now()})
    immutable(out / "private/TEMPLATE_RECORDS.json", rows)
    seeds = rt.read(out / "private/KEY_SELECTION.json")
    results = {}
    for role in BANK_ROLES:
        records = [record for doc in rows.values() if doc["role"] == role for record in doc["records"]]
        result = choose_templates(records, seeds[role])
        bank_path = out / "private/banks" / (role + ".json")
        immutable(bank_path, result["bank"])
        results[role] = {k: v for k, v in result.items() if k != "bank"}
        results[role].update(bank_sha256=rt.sha(bank_path), size=len(result["bank"]),
                             sentence_occurrences=len(records),
                             excluded_occurrences=sum(not r["eligible"] for r in records))
    overlaps = {}
    for i, left in enumerate(BANK_ROLES):
        for right in BANK_ROLES[i + 1:]:
            a, b = [{r["normalized_template_amr"] for r in
                     rt.read(out / "private/banks" / (role + ".json"))} for role in (left, right)]
            overlaps[left + "__" + right] = len(a & b)
    sufficient = all(r["sufficient"] for r in results.values())
    if sufficient:
        immutable(out / "PUBLIC_PROXY_BANK.json", rt.read(out / "private/banks/public_bank.json"))
    summary = {
        "status": "INDEPENDENT_BANKS_CONSTRUCTED" if sufficient else "INSUFFICIENT_BANK_POOL",
        "banks": results, "literal_template_overlaps": overlaps, "source_documents": len(rows),
        "detector_calls": 0, "provider_calls": 0, "calibration_complete": False,
        "ATR": None, "binary_attack_success": None, "multi_bank_defense_result": None,
        "independent_human_quality_reviews": 0,
    }
    immutable(out / "RESULT.json", summary)
    immutable(out / "AUDIT.json", {
        "freeze_sha256": contract, "source_manifest_sha256": rt.sha(out / "INPUT_MANIFEST.json"),
        "raw_graph_seal_sha256": rt.sha(out / "RAW_GRAPH_SEAL.json"),
        "template_records_sha256": rt.sha(out / "private/TEMPLATE_RECORDS.json"),
        "all_300_source_documents_processed": len(rows) == 300,
        "selection_before_detection": True, "bank_generation_same_rule": True,
        "bank_shortfall_relaxed": False, "banks_independent_source_cohorts": True,
        "secret_Banks_exposed_to_attacker": False, "attack_executed": False,
    })
    immutable(out / "TERMINAL.json", {"status": summary["status"],
                                     "result_sha256": rt.sha(out / "RESULT.json"),
                                     "audit_sha256": rt.sha(out / "AUDIT.json")})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare", "verify", "run"))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--assets", type=Path, default=ROOT)
    args = ap.parse_args()
    if args.action == "prepare":
        f = prepare(args.out)
        print(json.dumps({"freeze_sha256": rt.sha(args.out / "FREEZE.json"), "counts": f["counts"]}))
        return
    f = verify(args.out)
    if args.action == "verify":
        print(json.dumps({"verified_files": len(f["files"]), "counts": f["counts"]}))
        return
    if (args.out / "HALT.json").exists():
        raise RuntimeError("Integrity halt requires a distinct successor")
    if (args.out / "TERMINAL.json").exists():
        terminal = rt.read(args.out / "TERMINAL.json")
        if (terminal["result_sha256"] != rt.sha(args.out / "RESULT.json")
                or terminal["audit_sha256"] != rt.sha(args.out / "AUDIT.json")):
            raise RuntimeError("Terminal result changed")
        return
    with (args.out / "private/worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic()
        try:
            documents = rt.read(args.out / "private/BANK_INPUTS.json")
            parse(args.out, args.assets, f, documents)
            construct(args.out, documents)
        except Exception as exc:
            pause = isinstance(exc, rt.ResourcePause) or type(exc).__name__ == "OutOfMemoryError"
            rt.atomic(args.out / ("PAUSED.json" if pause else "HALT.json"),
                      {"type": type(exc).__name__, "at": rt.now(), "resume_allowed": pause,
                       "scientific_result": None})
            if pause:
                sys.exit(75)
            raise
        finally:
            rt.atomic(args.out / "private/sessions" / (str(time.time_ns()) + ".json"),
                      {"seconds": time.monotonic() - start, "at": rt.now()})


if __name__ == "__main__":
    main()
