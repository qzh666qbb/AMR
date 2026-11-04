"""X03: fresh same-candidate, fixed-Bank parser comparison and full recalibration."""
import argparse
from collections import Counter
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_multiscale_transfer as rt
import run_swan_independent_banks1 as construction
import run_swan_independent_calibration1 as calibration
import run_swan_independent_endpoint1 as endpoint
import swan_amrbart_audited1 as native
from semantic_fact_raw_amr import audit_raw, replay_signature as amrlib_signature
from swan_final1_common import freeze
from swan_multiparser1_core import FLOWS, statistics, thresholds, null_summary
from swan_independent_calibration1_core import CONDITIONS, detected

OUT = ROOT / "experiments/swan_multiparser1"
BANK = ROOT / "experiments/swan_independent_banks1"
CAL = ROOT / "experiments/swan_independent_calibration1"
ENDPOINT = ROOT / "experiments/swan_independent_endpoint1"
CODE = (
    "scripts/run_swan_multiparser1.py", "scripts/swan_amrbart_audited1.py",
    "scripts/swan_multiparser1_core.py", "scripts/test_swan_amrbart_audited1.py",
    "scripts/test_swan_multiparser1.py",
)


def prepare(out, bank, cal, ep):
    if (out / "FREEZE.json").exists():
        raise RuntimeError("Frozen; choose a successor")
    parent = endpoint.verify(ep, bank, cal)
    out.mkdir(parents=True, exist_ok=True)
    (out / "private").mkdir(mode=0o700, exist_ok=True)
    (out / ".gitignore").write_text("private/\n*.log\n")
    shutil.copytree(ep / "private/code", out / "private/code",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in CODE:
        shutil.copyfile(ROOT / name, out / "private/code" / name)
    source_manifest = rt.read(ROOT / native.SOURCE / "LOCAL_SOURCE_MANIFEST.json")
    for name, expected in source_manifest["files"].items():
        if rt.sha(ROOT / native.SOURCE / name) != expected:
            raise RuntimeError("Native author source drift")
    shutil.copytree(ROOT / native.SOURCE, out / "private/code" / native.SOURCE,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    model = rt.read(ROOT / native.MODEL / "ASSET_MANIFEST.json")
    assets = {native.MODEL + "/" + r["file"]: r["sha256"] for r in model["files"]}
    for name, expected in assets.items():
        if rt.sha(ROOT / name) != expected:
            raise RuntimeError("AMRBART checkpoint drift")
    freeze(out / "FREEZE.json", {
        "version": "same-candidate-multiparser/v1",
        "parents": {"bank": rt.sha(bank / "FREEZE.json"), "calibration": rt.sha(cal / "FREEZE.json"),
                    "endpoint": rt.sha(ep / "FREEZE.json")},
        "files": {str(p.relative_to(out)): rt.sha(p) for p in sorted(out.rglob("*"))
                  if p.is_file() and "__pycache__" not in p.parts},
        "parser": parent["parser"], "amrbart_assets": assets,
        "amrbart_source_manifest_sha256": rt.sha(ROOT / native.SOURCE / "LOCAL_SOURCE_MANIFEST.json"),
        "amrbart_source_commit": "37e5742411e74d4269c33e8f2c5f5a8a0ed51241",
        "amrbart_generation": native.GENERATION,
        "amrbart_dtype": "float32", "amrbart_attention": "eager",
        "amrbart_input": "native tokenized text plus <AMR><mask></AMR>, no truncation, <=1024",
        "amrbart_validity": "OK, no diagnostics, strict pointer graph unchanged, full roundtrip",
        "amrbart_native_limit_comparison": "length>=1024 or missing natural AMR EOS is erased",
        "glove_sha256": parent["glove_sha256"],
        "execution": {"resumable_gpu_pressure": True}, "resource_gate": {"vram_fraction_max": .8},
        "bank": "target A fixed from original amrlib construction; system audit, not parser-neutral training",
        "flows": list(FLOWS), "ensemble": "(z_amrlib+z_amrbart)/sqrt(2), separately calibrated",
        "calibration_sources": 400, "independent_null_sources": 400,
        "positive_sources": 72, "positive_arm": "a", "candidate_paraphrases": 16,
        "fresh_graphs_both_parsers": True, "all_graphs_sealed_before_scoring": True,
        "new_provider_calls": 0, "alpha": .01,
        "attack": "same parent-selected texts, not adaptive ensemble feedback",
        "human_quality": "reuse parent independent reviews; absent means no TPR/ASR/ATR",
    })
    return verify(out, bank, cal, ep)


def verify(out, bank, cal, ep):
    f = rt.read(out / "FREEZE.json")
    endpoint.verify(ep, bank, cal)
    for key, directory in (("bank", bank), ("calibration", cal), ("endpoint", ep)):
        if rt.sha(directory / "FREEZE.json") != f["parents"][key]:
            raise RuntimeError("Parent contract drift")
    for name, expected in f["files"].items():
        if rt.sha(out / name) != expected:
            raise RuntimeError("Multiparser frozen file drift: " + name)
    return f


def validate_documents(docs):
    cal = [r for r in docs if r["role"] == "calibration"]
    null = [r for r in docs if r["role"] == "null"]
    pos = [r for r in docs if r["role"] in ("pilot", "main")]
    roles = {}
    for group, count, role in ((cal, 400, "calibration"), (null, 400, "null")):
        ids = {r["source_id"] for r in group}
        if len(ids) != count or len(group) != count * 5 or {
                (r["source_id"], r["condition"]) for r in group} != {
                    (i, c) for i in ids for c in CONDITIONS}:
            raise RuntimeError("Null/calibration denominator changed")
        roles[role] = ids
    expected = {(r["source_id"], c, None) for r in pos for c in CONDITIONS}
    expected |= {(r["source_id"], "paraphrase", str(i)) for r in pos for i in range(16)}
    actual = {(r["source_id"], r["condition"], r["candidate_id"]) for r in pos}
    ids = {r["source_id"] for r in pos}
    if (len(ids) != 72 or len(pos) != 72 * 21 or expected != actual
            or any(r["arm"] != "a" for r in pos)):
        raise RuntimeError("Complete same-source A arm required")
    if len({r["source_id"] for r in pos if r["role"] == "pilot"}) != 12 or len({
            r["source_id"] for r in pos if r["role"] == "main"}) != 60:
        raise RuntimeError("Pilot/main split changed")
    if roles["calibration"] & roles["null"] or (roles["calibration"] | roles["null"]) & ids:
        raise RuntimeError("Source-role overlap")
    if len(docs) != 5512 or len({r["row_id"] for r in docs}) != len(docs):
        raise RuntimeError("Unexpected document roles or duplicate identity")


def bind_inputs(out, bank, cal, ep):
    endpoint.bind(out, bank, cal)
    for name in ("WRITING_COMPLETE.json", "ATTACK_GENERATION_COMPLETE.json",
                 "SCORING_COMPLETE.json", "SELECTIONS_COMPLETE.json"):
        endpoint.check_seal(ep, name)
    # Parent evaluation input hash is sealed by SCORING_COMPLETE, not by its FREEZE.
    parent_seal = rt.read(ep / "SCORING_COMPLETE.json")
    if rt.sha(ep / "private/EVALUATION_INPUTS.json") != parent_seal["evaluation_inputs_sha256"]:
        raise RuntimeError("Parent candidate input changed")
    docs = []
    for doc in rt.read(cal / "private/INPUTS.json"):
        docs.append({**doc, "row_id": rt.digest(doc["role"] + "/" + doc["source_id"] + "/" + doc["condition"])})
    for doc in rt.read(ep / "private/EVALUATION_INPUTS.json"):
        if doc["arm"] == "a":
            if rt.digest(doc["text"]) != doc["text_sha256"]:
                raise RuntimeError("Endpoint candidate text changed")
            docs.append({**doc, "row_id": rt.digest(doc["doc_id"] + "/" + doc["condition"]
                                                   + "/" + str(doc["candidate_id"]))})
    validate_documents(docs)
    freeze(out / "private/INPUTS.json", docs)
    endpoint.seal(out, "INPUT_BINDING.json", [out / "private/INPUTS.json"],
                  parent_input_sha256=rt.sha(ep / "private/EVALUATION_INPUTS.json"),
                  selection_seal_sha256=rt.sha(ep / "SELECTIONS_COMPLETE.json"))
    return docs, rt.read(out / "private/BANK_BINDING.json")["banks"]["a"]


def read_graph(out, name, text):
    path = out / "private/raw" / name / (rt.digest(text) + ".json")
    row = endpoint.read_row(path, {"text_sha256": rt.digest(text),
                                  "freeze_sha256": rt.sha(out / "FREEZE.json")})
    audit_raw(row, {"case_id": rt.digest(text), "input_text_sha256": rt.digest(text)},
              rt.sha(out / "FREEZE.json"))
    return row


def parse_all(out, assets, f, docs, name):
    seal_name = "RAW_" + name.upper() + "_SEAL.json"
    if (out / seal_name).exists():
        endpoint.check_seal(out, seal_name)
        return
    parser = native.Parser(assets, out, f) if name == "amrbart" else rt.Parser(assets, out, f)
    signature = native.replay_signature if name == "amrbart" else amrlib_signature
    samples = ["The researcher described the result.", "The dog barks."]
    probes = []
    for text in samples:
        pair = [{**parser.fresh(text), "case_id": rt.digest(text)} for _ in range(2)]
        if signature(pair[0]) != signature(pair[1]):
            raise RuntimeError("Two parser replays differ")
        for row in pair:
            if not audit_raw(row, {"case_id": rt.digest(text), "input_text_sha256": rt.digest(text)},
                             rt.sha(out / "FREEZE.json"))["structurally_eligible"]:
                freeze(out / ("private/" + name + "_FAILED_PROBE.json"), row)
                raise RuntimeError("Native parser integration probe not eligible")
        probes.append({"text_sha256": rt.digest(text), "signature": signature(pair[0])})
    freeze(out / (name.upper() + "_GPU_PRECHECK.json"), {
        "dtype": "float32", "replays": probes, "tf32": False, "probes": 2})
    texts = sorted({s for d in docs for s in d["sentences"]}, key=rt.digest)
    for index, text in enumerate(texts):
        endpoint.disk(out)
        path = out / "private/raw" / name / (rt.digest(text) + ".json")
        if path.exists():
            read_graph(out, name, text)
        else:
            if name == "amrlib":
                n = len(parser.model.tokenizer(text, truncation=False)["input_ids"])
                row = (construction.excluded(text, parser.freeze_sha, n) if n >= 1024
                       else {**parser.fresh(text), "case_id": rt.digest(text)})
            else:
                row = parser.fresh(text)
            endpoint.save_row(path, row)
            read_graph(out, name, text)
        rt.atomic(out / "HEARTBEAT.json", {"phase": "raw_" + name, "completed": index + 1,
                                          "total": len(texts), "at": rt.now()})
    files = list((out / "private/raw" / name).glob("*.json"))
    if {p.stem for p in files} != {rt.digest(t) for t in texts}:
        raise RuntimeError("Parser candidate denominator differs")
    endpoint.seal(out, seal_name, files, unique_texts=len(texts))
    del parser
    gc.collect()
    import torch
    torch.cuda.empty_cache()


def score_all(out, bank, assets, docs, targets):
    for name in ("AMRLIB", "AMRBART"):
        endpoint.check_seal(out, "RAW_" + name + "_SEAL.json")
    matcher, normalizer = calibration.Matcher(out, assets), construction.Normalizer(bank)
    if [matcher.fresh(targets[0], targets[0]) for _ in range(2)] != [1., 1.]:
        raise RuntimeError("Matcher self/replay failed")
    cache, rows, fitted = {}, [], None
    ids = sorted({r["source_id"] for r in docs if r["role"] == "calibration"})
    for index, doc in enumerate(docs):
        if doc["role"] != "calibration" and fitted is None:
            fitted = thresholds(rows, ids)
            freeze(out / "THRESHOLDS.json", fitted)
        binding = {k: v for k, v in doc.items() if k not in ("text", "sentences")}
        binding["freeze_sha256"] = rt.sha(out / "FREEZE.json")
        path = out / "private/rows" / (doc["row_id"] + ".json")
        if path.exists():
            row = endpoint.read_row(path, binding)
        else:
            scores = {}
            for name in ("amrlib", "amrbart"):
                values = []
                for text in doc["sentences"]:
                    key = (name, rt.digest(text))
                    if key not in cache:
                        raw = read_graph(out, name, text)
                        valid = audit_raw(raw, {"case_id": rt.digest(text),
                                                "input_text_sha256": rt.digest(text)}, binding["freeze_sha256"])
                        value = normalizer(raw["amr"]) if valid["structurally_eligible"] else None
                        cache[key] = (max(matcher.pair(value["template"], t) for t in targets)
                                      if value and value["eligible"] else None)
                    values.append(cache[key])
                scores[name] = values
            row = {**binding, **statistics(scores["amrlib"], scores["amrbart"]),
                   "sentence_scores": scores}
            endpoint.save_row(path, row)
        rows.append(row)
        rt.atomic(out / "HEARTBEAT.json", {"phase": "paired_scoring",
                                          "completed": index + 1, "total": len(docs), "at": rt.now()})
        endpoint.disk(out)
    matcher.db.close()
    replayed = thresholds([r for r in rows if r["role"] == "calibration"], ids)
    if fitted != replayed:
        raise RuntimeError("Threshold replay changed")
    freeze(out / "THRESHOLDS.json", fitted)
    files = list((out / "private/rows").glob("*.json"))
    if {p.stem for p in files} != {d["row_id"] for d in docs}:
        raise RuntimeError("Scored document denominator differs")
    endpoint.seal(out, "SCORING_COMPLETE.json", files, rows=len(rows))
    return rows, fitted


def summarize(out, ep, rows, fitted):
    neg = [r for r in rows if r["role"] == "null"]
    null = null_summary(neg, sorted({r["source_id"] for r in neg}), fitted)
    positives = [r for r in rows if r["role"] in ("pilot", "main")]
    boundary = []
    for role in ("pilot", "main"):
        for condition in CONDITIONS:
            subset = [r for r in positives if (r["role"], r["condition"]) == (role, condition)]
            for flow in FLOWS:
                for reference in ("clean", condition) if condition != "clean" else ("clean",):
                    boundary.append({"role": role, "condition": condition, "flow": flow,
                        "threshold_condition": reference, "sources": len(subset),
                        "unfiltered_detections": sum(detected(r["scores"][flow], fitted[reference][flow])
                                                     for r in subset),
                        "abstentions": sum(r["abstain"] for r in subset),
                        "complementarity": {k: sum(r["complementarity"][k] for r in subset)
                                            for k in subset[0]["complementarity"]}})
    clean = {r["doc_id"]: r for r in positives if r["condition"] == "clean"}
    candidates = {(r["doc_id"], r["candidate_id"]): r for r in positives if r["condition"] == "paraphrase"}
    attacks = []
    for path in sorted((ep / "private/selections").glob("*.json")):
        selection = rt.read(path)
        if selection["arm"] != "a":
            continue
        candidate = candidates[(selection["doc_id"], selection["selection"]["selected"])]
        attacks.append({
            "source_id": selection["source_id"], "role": selection["role"],
            "oracle": selection["oracle"], "mode": selection["mode"], "budget": selection["budget"],
            "candidate_id": candidate["candidate_id"], "text_sha256": candidate["text_sha256"],
            "clean_detected": {f: detected(clean[selection["doc_id"]]["scores"][f], fitted["clean"][f])
                               for f in FLOWS},
            "attack_detected": {f: detected(candidate["scores"][f], fitted["clean"][f]) for f in FLOWS},
        })
    if len(attacks) != 72 * 2 * 3 * 3:
        raise RuntimeError("Parent selected attack denominator differs")
    freeze(out / "private/SELECTED_ATTACKS.json", attacks)
    comparison = []
    for role in ("pilot", "main"):
        for oracle in ("public", "a"):
            for mode in ("none", "label", "score"):
                for budget in (1, 4, 16):
                    selected = [r for r in attacks if (r["role"], r["oracle"], r["mode"], r["budget"])
                                == (role, oracle, mode, budget)]
                    for flow in FLOWS:
                        eligible = [r for r in selected if r["clean_detected"][flow]]
                        comparison.append({"role": role, "oracle": oracle, "mode": mode, "budget": budget,
                            "flow": flow, "sources": len(selected), "target_clean_detected": len(eligible),
                            "unfiltered_escape_given_clean": sum(not r["attack_detected"][flow] for r in eligible),
                            "ASR": None, "ATR": None})
    costs = {}
    for name in ("amrlib", "amrbart"):
        raw = [endpoint.read_row(p, {}) for p in (out / "private/raw" / name).glob("*.json")]
        costs[name] = {"unique_texts": len(raw), "successful_record_parse_seconds": sum(
            r.get("seconds", 0) for r in raw),
            "native_status_counts": dict(Counter(r.get("native_status", "amrlib") for r in raw)),
            "structurally_eligible": sum(audit_raw(r, {"case_id": r["case_id"],
                "input_text_sha256": r["text_sha256"]}, r["freeze_sha256"])["structurally_eligible"] for r in raw),
            "logical_template_comparisons": 50 * sum(
                v is not None for r in rows for v in r["sentence_scores"][name])}
    return {"status": "MULTIPARSER_MACHINE_COMPARISON_COMPLETE_HUMAN_QUALITY_PENDING",
            "condition_FPR": null, "positive_boundary": boundary, "fixed_attack_comparison": comparison,
            "parser_costs": costs, "source_counts": {"calibration": 400, "null": 400, "pilot": 12, "main": 60},
            "document_rows": len(rows), "new_provider_calls": 0,
            "independent_architecture": False, "independent_checkpoint": True,
            "TPR": None, "ASR": None, "ATR": None, "scientific_closure": False,
            "scope": "same Bank A and amrlib-written sources; parent attacks fixed, no ensemble-aware search"}


def run(out, bank, cal, ep, assets):
    f = verify(out, bank, cal, ep)
    if (out / "TERMINAL.json").exists():
        t = rt.read(out / "TERMINAL.json")
        if any(t[k] != rt.sha(out / name) for k, name in (
                ("result_sha256", "RESULT.json"), ("audit_sha256", "AUDIT.json"))):
            raise RuntimeError("Final artifact drift")
        return
    if (out / "HALT.json").exists():
        raise RuntimeError("Integrity halt needs successor")
    docs, targets = bind_inputs(out, bank, cal, ep)
    for name in ("amrbart", "amrlib"):
        parse_all(out, assets, f, docs, name)
    rows, fitted = score_all(out, bank, assets, docs, targets)
    result = summarize(out, ep, rows, fitted)
    freeze(out / "RESULT.json", result)
    freeze(out / "AUDIT.json", {
        "freeze_sha256": rt.sha(out / "FREEZE.json"),
        "seals": {name: rt.sha(out / name) for name in (
            "INPUT_BINDING.json", "RAW_AMRLIB_SEAL.json", "RAW_AMRBART_SEAL.json",
            "SCORING_COMPLETE.json", "THRESHOLDS.json")},
        "same_candidate_texts": True, "source_overlap": 0, "rows": len(rows),
        "all_failed_candidates_retained": True,
        "calibrated_before_independent_null_scoring": True,
        "same_positive_and_negative_flow": True, "human_quality_pending": True,
        "new_provider_calls": 0})
    freeze(out / "TERMINAL.json", {"status": result["status"],
        "result_sha256": rt.sha(out / "RESULT.json"), "audit_sha256": rt.sha(out / "AUDIT.json")})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare", "verify", "run"))
    for name, value in (("out", OUT), ("bank", BANK), ("calibration", CAL), ("endpoint", ENDPOINT),
                        ("assets", ROOT)):
        ap.add_argument("--" + name, type=Path, default=value)
    args = ap.parse_args()
    if args.action in ("prepare", "verify"):
        f = (prepare if args.action == "prepare" else verify)(
            args.out, args.bank, args.calibration, args.endpoint)
        print(json.dumps({"version": f["version"], "files": len(f["files"])}))
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with (args.out / "private/worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic()
        try:
            run(args.out, args.bank, args.calibration, args.endpoint, args.assets)
        except Exception as exc:
            pause = isinstance(exc, rt.ResourcePause) or type(exc).__name__ == "OutOfMemoryError"
            rt.atomic(args.out / ("PAUSED.json" if pause else "HALT.json"),
                      {"reason_type": type(exc).__name__, "at": rt.now(),
                       "resume_allowed": pause, "scientific_result": None})
            if pause:
                sys.exit(75)
            raise
        finally:
            rt.atomic(args.out / "private/sessions" / (str(time.time_ns()) + ".json"),
                      {"seconds": time.monotonic() - start})


if __name__ == "__main__":
    main()
