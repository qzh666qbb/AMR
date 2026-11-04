"""Fresh three-bank calibration and independent human-source FPR, resumable."""
import argparse
import fcntl
import gc
import importlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_multiscale_transfer as rt
import run_swan_independent_banks1 as construction
from semantic_fact_raw_amr import audit_raw, replay_signature
from swan_independent_calibration1_core import (
    BANK_KEYS, CONDITIONS, FLOWS, variants, document_statistics, fit_threshold, evaluate_null)

OUT = ROOT / "experiments/swan_independent_calibration1"
BANK = ROOT / "experiments/swan_independent_banks1"
OFFICIAL = "third_party/SWAN_official_2ca0c207_clean/detection/detection_utils.py"
CODE = ("scripts/run_swan_independent_calibration1.py",
        "scripts/swan_independent_calibration1_core.py",
        "scripts/test_swan_independent_calibration1.py",
        "scripts/run_multiscale_transfer.py", "scripts/run_swan_independent_banks1.py",
        "scripts/semantic_fact_raw_amr.py", "scripts/run_research_completion_queue.py",
        "src/core/parser_recovery_audit.py")

immutable = construction.immutable


def prepare(out, bank):
    import nltk
    if (out / "FREEZE.json").exists():
        raise RuntimeError("Already frozen")
    parent = construction.verify(bank)
    nltk.data.path.insert(0, str(bank / "private/code/.nltk_data"))
    if nltk.__version__ != parent["normalizer"]["nltk"]:
        raise RuntimeError("Source tokenizer version differs")
    tokenizer = nltk.tokenize.PunktTokenizer("english")
    documents = []
    for role in ("calibration", "null"):
        sources = rt.read(bank / "private/cohorts" / (role + ".json"))
        if len(sources) != 400:
            raise RuntimeError("Expected 400 per role")
        for row in sources:
            if rt.digest(row["text"]) != row["window_sha256"]:
                raise RuntimeError("Source hash differs")
            for condition, text in variants(row["text"], tokenizer).items():
                documents.append({"source_id": row["allocation_id"], "role": role,
                                  "condition": condition, "text_sha256": rt.digest(text),
                                  "sentences": tokenizer.tokenize(text)})
    if len({r["source_id"] for r in documents}) != 800:
        raise RuntimeError("Calibration and Null sources overlap")
    immutable(out / "private/INPUTS.json", documents)
    paths = list(CODE) + [OFFICIAL]
    paths += [str(p.relative_to(ROOT)) for p in
              sorted((ROOT / "third_party/swan_s2match_runtime_711a231d").glob("*.py"))]
    for relative in paths:
        dst = out / "private/code" / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, dst)
    immutable(out / "FREEZE.json", {
        "version": "independent-bank-calibration/v1",
        "parent_bank_freeze_sha256": rt.sha(bank / "FREEZE.json"),
        "parser": parent["parser"], "execution": {"resumable_gpu_pressure": True},
        "resource_gate": {"vram_fraction_max": .8},
        "files": {str(p.relative_to(out)): rt.sha(p) for p in sorted(out.rglob("*"))
                  if p.is_file() and "__pycache__" not in p.parts},
        "glove_sha256": rt.sha(ROOT / "vectors/glove.6B.100d.txt"),
        "counts": {"calibration_sources": 400, "independent_null_sources": 400,
                   "conditions": 5, "flows": 5, "document_condition_rows": len(documents),
                   "unique_parse_texts": len({s for r in documents for s in r["sentences"]})},
        "official_match_threshold": .65, "official_lambda": .25,
        "joint_statistic": "(z_a+z_b)/sqrt(2); independently calibrated, no independence assumption",
        "pooled_statistic": "official z against concatenated A/B capacity slots",
        "alpha": .01, "generation_or_attack_calls": 0,
    })
    return verify(out, bank)


def verify(out, bank):
    f = rt.read(out / "FREEZE.json")
    if rt.sha(bank / "FREEZE.json") != f["parent_bank_freeze_sha256"]:
        raise RuntimeError("Bank source contract changed")
    construction.verify(bank)
    for name, expected in f["files"].items():
        if rt.sha(out / name) != expected:
            raise RuntimeError("Calibration frozen file drift: " + name)
    return f


def bind_banks(out, bank):
    terminal = rt.read(bank / "TERMINAL.json")
    if terminal["status"] != "INDEPENDENT_BANKS_CONSTRUCTED":
        raise RuntimeError("Three complete independent banks required")
    for field, filename in (("result_sha256", "RESULT.json"), ("audit_sha256", "AUDIT.json")):
        if terminal[field] != rt.sha(bank / filename):
            raise RuntimeError("Bank terminal hash differs")
    result, banks = rt.read(bank / "RESULT.json"), {}
    for key, role in zip(BANK_KEYS, construction.BANK_ROLES):
        path = bank / "private/banks" / (role + ".json")
        if rt.sha(path) != result["banks"][role]["bank_sha256"]:
            raise RuntimeError("Bank key changed")
        values = [r["normalized_template_amr"] for r in rt.read(path)]
        if len(values) != 50:
            raise RuntimeError("Bank size differs")
        banks[key] = values
    immutable(out / "private/BANK_BINDING.json", {
        "bank_terminal_sha256": rt.sha(bank / "TERMINAL.json"),
        "bank_result_sha256": rt.sha(bank / "RESULT.json"), "banks": banks,
        "calibration_freeze_sha256": rt.sha(out / "FREEZE.json")})
    return banks


class Matcher:
    def __init__(self, out, assets):
        sys.path.insert(0, str(out / "private/code/third_party/swan_s2match_runtime_711a231d"))
        self.mod = importlib.import_module("s2match")
        amr = importlib.import_module("amr_py3")
        self.logger = rt.BoundLogger()
        self.logger.warning = self.logger.error
        self.mod.log_helper = amr.log_helper = self.logger
        self.glove = assets / "vectors/glove.6B.100d.txt"
        if rt.sha(self.glove) != rt.read(out / "FREEZE.json")["glove_sha256"]:
            raise RuntimeError("GloVe identity changed")
        vectors = self.mod.load_vecs(str(self.glove))
        self.mod.load_vecs = lambda _: vectors
        self.db = sqlite3.connect(out / "private/pair_scores.sqlite")
        self.db.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, value REAL)")

    def fresh(self, candidate, target):
        self.logger.events.clear()
        self.mod.random = rt.PairRandom(int(rt.digest(candidate + "\0" + target), 16))
        self.mod.match_triple_dict.clear()
        try:
            _, _, score = self.mod.compute_s2match_from_strings(
                candidate, target, vectors_file=str(self.glove), similarity_function="cosine",
                cutoff=.5, diffsense=.5, mwp="split", do_not_mark_quotes=False)
            if self.logger.events or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Bound S2Match diagnostic")
            return float(score)
        finally:
            self.mod.match_triple_dict.clear()

    def pair(self, candidate, target):
        identity = rt.digest(candidate + "\0" + target)
        old = self.db.execute("SELECT value FROM scores WHERE key=?", (identity,)).fetchone()
        if old:
            return old[0]
        value = self.fresh(candidate, target)
        self.db.execute("INSERT INTO scores VALUES (?,?)", (identity, value))
        self.db.commit()
        return value


def run(out, bank, assets):
    f = verify(out, bank)
    if (out / "TERMINAL.json").exists():
        t = rt.read(out / "TERMINAL.json")
        if any(t[k] != rt.sha(out / p) for k, p in
               (("result_sha256", "RESULT.json"), ("audit_sha256", "AUDIT.json"))):
            raise RuntimeError("Final calibration artifacts changed")
        return
    if (out / "HALT.json").exists():
        raise RuntimeError("Integrity halt requires successor")
    banks = bind_banks(out, bank)
    docs = rt.read(out / "private/INPUTS.json")
    # Complete every raw graph before normalizing or computing detector scores.
    construction.parse(out, assets, f, docs)
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    normalizer = construction.Normalizer(bank)
    matcher = Matcher(out, assets)
    probe = banks["a"][0]
    pair = [matcher.fresh(probe, probe) for _ in range(2)]
    if pair != [1., 1.]:
        raise RuntimeError("S2Match self/replay precheck failed")
    immutable(out / "MATCHER_PRECHECK.json", {"self_scores": pair,
                                             "bank_binding_sha256": rt.sha(out / "private/BANK_BINDING.json")})
    template_cache = {}
    rows = []
    cal_ids = sorted({r["source_id"] for r in docs if r["role"] == "calibration"})
    null_ids = sorted({r["source_id"] for r in docs if r["role"] == "null"})
    if set(cal_ids) & set(null_ids):
        raise RuntimeError("Source overlap")
    thresholds = None
    for index, doc in enumerate(docs):
        if doc["role"] == "null" and thresholds is None:
            thresholds = {c: {flow: fit_threshold(
                [r for r in rows if r["role"] == "calibration" and r["condition"] == c],
                cal_ids, flow, c) for flow in FLOWS} for c in CONDITIONS}
            immutable(out / "THRESHOLDS.json", thresholds)
        identity = rt.digest(doc["role"] + "\0" + doc["source_id"] + "\0" + doc["condition"])
        path = out / "private/rows" / (identity + ".json")
        binding = {k: doc[k] for k in ("source_id", "role", "condition", "text_sha256")}
        binding["freeze_sha256"] = rt.sha(out / "FREEZE.json")
        if path.exists():
            envelope = rt.read(path)
            row = envelope["row"]
            if envelope["sha256"] != rt.digest(json.dumps(row, sort_keys=True)):
                raise RuntimeError("Row checkpoint checksum differs")
            if any(row[k] != v for k, v in binding.items()):
                raise RuntimeError("Row resume identity differs")
        else:
            sentence_scores, erased = [], []
            for text in doc["sentences"]:
                textid = rt.digest(text)
                if textid not in template_cache:
                    graph = rt.read(out / "private/raw_graphs" / (textid + ".json"))
                    check = audit_raw(graph, {"case_id": textid, "input_text_sha256": textid},
                                      binding["freeze_sha256"])
                    if check["structurally_eligible"]:
                        normalized = normalizer(graph["amr"])
                        template_cache[textid] = normalized["template"] if normalized["eligible"] else None
                    else:
                        template_cache[textid] = None
                template = template_cache[textid]
                scores = {}
                for key, values in banks.items():
                    scores[key] = (max(matcher.pair(template, t) for t in values)
                                   if template is not None else None)
                sentence_scores.append(scores)
                if template is None:
                    erased.append(textid)
            row = {**binding, **document_statistics(sentence_scores),
                   "sentence_scores": sentence_scores, "erased_text_sha256": erased}
            immutable(path, {"row": row, "sha256": rt.digest(json.dumps(row, sort_keys=True))})
        rows.append(row)
        rt.atomic(out / "HEARTBEAT.json", {"phase": "null_scoring", "completed": index + 1,
                                          "total": len(docs), "at": rt.now()})
        if shutil.disk_usage(out).free < 512 * 1024**2:
            raise rt.ResourcePause("Disk headroom insufficient")
    replay_thresholds = {c: {flow: fit_threshold(
        [r for r in rows if r["role"] == "calibration" and r["condition"] == c],
        cal_ids, flow, c) for flow in FLOWS} for c in CONDITIONS}
    if thresholds != replay_thresholds:
        raise RuntimeError("Sealed thresholds differ on replay")
    immutable(out / "THRESHOLDS.json", thresholds)
    summary = {"status": "CALIBRATION_AND_INDEPENDENT_NULL_COMPLETE",
               "rows": len(rows), "calibration_sources": len(cal_ids),
               "independent_human_null_sources": len(null_ids),
               "condition_FPR": evaluate_null([r for r in rows if r["role"] == "null"],
                                             null_ids, thresholds),
               "TPR": None, "ASR": None, "ATR": None, "joint_writer_evaluated": False,
               "independent_human_generated_quality_reviews": 0}
    immutable(out / "RESULT.json", summary)
    immutable(out / "AUDIT.json", {
        "freeze_sha256": rt.sha(out / "FREEZE.json"),
        "bank_binding_sha256": rt.sha(out / "private/BANK_BINDING.json"),
        "raw_graph_seal_sha256": rt.sha(out / "RAW_GRAPH_SEAL.json"),
        "thresholds_sha256": rt.sha(out / "THRESHOLDS.json"),
        "row_files": {str(p.relative_to(out)): rt.sha(p)
                      for p in sorted((out / "private/rows").glob("*.json"))},
        "same_positive_and_negative_pipeline_required": True,
        "source_overlap": 0, "scored_all_4000_condition_rows": len(rows) == 4000,
        "new_provider_calls": 0})
    immutable(out / "TERMINAL.json", {"status": summary["status"],
                                     "result_sha256": rt.sha(out / "RESULT.json"),
                                     "audit_sha256": rt.sha(out / "AUDIT.json")})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare", "verify", "run"))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--bank", type=Path, default=BANK)
    ap.add_argument("--assets", type=Path, default=ROOT)
    args = ap.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(args.out, args.bank)["counts"]))
    elif args.action == "verify":
        print(json.dumps(verify(args.out, args.bank)["counts"]))
    else:
        with (args.out / "private/worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            start = time.monotonic()
            try:
                run(args.out, args.bank, args.assets)
            except Exception as exc:
                pause = isinstance(exc, rt.ResourcePause) or type(exc).__name__ == "OutOfMemoryError"
                rt.atomic(args.out / ("PAUSED.json" if pause else "HALT.json"),
                          {"at": rt.now(), "reason_type": type(exc).__name__,
                           "scientific_result": None, "resume_allowed": pause})
                if pause:
                    sys.exit(75)
                raise
            finally:
                rt.atomic(args.out / "private/sessions" / (str(time.time_ns()) + ".json"),
                          {"seconds": time.monotonic() - start})


if __name__ == "__main__":
    main()
