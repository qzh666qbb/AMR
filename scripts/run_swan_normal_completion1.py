"""Unfiltered normal-length collection and source-isolated W02/W05 successor."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_multiscale_transfer as rt
import run_swan_supplement4 as generation
import run_swan_supplement1 as pilot
import swan_final1_budget as budget
from swan_final1_common import freeze, load, sha256, single_writer
from swan_supplement1_sources import WORD, tokenizer
from swan_supplement4_runtime import CandidateReject, ResourcePause, runtime
from swan_supplement1_runtime import readiness
from swan_atomic_calibration1 import threshold, reject, binomial, paired_difference

OUT = ROOT / "experiments/swan_normal_completion1"
PARENT = ROOT / "experiments/swan_count_neutral_supplement4"
BANK_ENV = ROOT / "experiments/swan_independent_banks1"
ALIGNMENT = ROOT / "experiments/swan_atomic_alignment_completion2"
CONDITIONS = ("clean", "A", "B")
CAPS = {"main": (160000, 40_000_000_000), "llm_null": (640000, 15_000_000_000)}
NULL_PROMPT = (
    "Write one next sentence continuing the general topic of the context, using "
    "plausible fictional details when needed. Write 7 to 12 lexical words in one "
    "complete sentence ending with a period. Do not output headings or explanations.\nContext: "
)
FAMILIES = ("whole_text", "surface_boundary", "word_window_12", "word_window_24", "word_window_48")


def verify(out):
    f = load(out / "FREEZE.json")
    for path, expected in f["files"].items():
        if sha256(out / path) != expected:
            raise RuntimeError("Frozen input/code changed: " + path)
    return f


def file_seal(out, path, files):
    return freeze(path, {"freeze_sha256": sha256(out / "FREEZE.json"),
                         "files": {str(p.relative_to(out)): sha256(p) for p in sorted(files)}})


def verify_seal(out, path):
    seal = load(path)
    if seal["freeze_sha256"] != sha256(out / "FREEZE.json"):
        raise RuntimeError("Seal belongs to another contract")
    for name, expected in seal["files"].items():
        if sha256(out / name) != expected:
            raise RuntimeError("Sealed artifact changed: " + name)
    return seal


def prepare(out):
    if (out / "FREEZE.json").exists():
        raise RuntimeError("Use a new version for changed contracts")
    parent = load(PARENT / "FREEZE.json")
    selected = {}
    for phase in ("main", "calibration", "test"):
        name = f"private/{phase}_sources.json"
        if sha256(PARENT / name) != parent["source_file_hashes"][name]:
            raise RuntimeError("Original source changed")
        selected[phase] = load(PARENT / name)
    if {k: len(v) for k, v in selected.items()} != {"main": 200, "calibration": 1000, "test": 1000}:
        raise RuntimeError("Original fixed cohort changed")
    identities, urls = set(), set()
    for rows in selected.values():
        for row in rows:
            identity = row["normalized_source_sha256"]
            if identity in identities or (row["url"] and row["url_sha256"] in urls):
                raise RuntimeError("Source overlap")
            identities.add(identity)
            if row["url"]:
                urls.add(row["url_sha256"])
    for phase, rows in selected.items():
        freeze(out / "private/sources" / (phase + ".json"), rows)
    # All allocations are made from source identity, before any new model scores.
    design = {"positive_development": [r["allocation_id"] for r in selected["main"][:40]],
              "positive_test": [r["allocation_id"] for r in selected["main"][40:]],
              "human_calibration": [r["allocation_id"] for r in selected["calibration"][:400]],
              "human_test": [r["allocation_id"] for r in selected["test"][:400]],
              "llm_calibration": [r["allocation_id"] for r in selected["calibration"][400:800]],
              "llm_test": [r["allocation_id"] for r in selected["test"][400:800]]}
    if sum(map(len, design.values())) != len(set(sum(design.values(), []))):
        raise RuntimeError("W05 source reuse")
    freeze(out / "private/ALIGNMENT_DESIGN.json", design)
    # Copy the concrete dependency closures used by the two already verified runners.
    names = {p for p in parent["dependencies"] if not p.startswith(
        ("experiments/", "models/", "vectors/"))}
    for folder in (BANK_ENV, ALIGNMENT,
                   ROOT / "experiments/swan_independent_calibration1"):
        for p in (folder / "private/code").rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                names.add(str(p.relative_to(folder / "private/code")))
    names.update((
        "scripts/run_swan_normal_completion1.py", "scripts/swan_normal_alignment1.py",
        "scripts/test_swan_normal_completion1.py",
        "scripts/run_swan_supplement2.py", "scripts/swan_atomic_calibration1.py",
    ))
    for name in sorted(names):
        target = out / "private/code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    for filename in ("amr_bank_50.json", "amr_bank_100.json"):
        src = ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks" / filename
        shutil.copyfile(src, out / "private" / filename)
    parser = load(BANK_ENV / "FREEZE.json")["parser"]
    af = load(ALIGNMENT / "FREEZE.json")
    environment = load(BANK_ENV / "FREEZE.json")
    freeze(out / "FREEZE.json", {
        "version": "normal-length-unfiltered-completion/v1",
        "parent_freeze_sha256": sha256(PARENT / "FREEZE.json"),
        "bank_environment_freeze_sha256": sha256(BANK_ENV / "FREEZE.json"),
        "alignment_parent_freeze_sha256": sha256(ALIGNMENT / "FREEZE.json"),
        "parser": parser, "normalizer": environment["normalizer"],
        "execution": {"resumable_gpu_pressure": True},
        "resource_gate": {"vram_fraction_max": .8},
        "glove_sha256": sha256(ROOT / "vectors/glove.6B.100d.txt"),
        "encoder_files": {p: h for p, h in af["files"].items() if p.startswith("private/encoder/")},
        "surface_files": {p: h for p, h in af["files"].items() if p.startswith("private/surface_model/")},
        "counts": {"main": 200, "calibration": 1000, "independent_human_test": 1000,
                   "llm_null_calibration": 400, "llm_null_test": 400},
        "caps": CAPS, "conditions": CONDITIONS, "alpha": .01,
        "W05_primary_condition": "B", "W05_families": FAMILIES,
        "power_plan": {"confirm_sources": 160, "development_sources": 40,
                      "target_absolute_gain": .15, "two_sided_alpha": .05,
                      "normal_approx_power_at_discordance_0_4": .85,
                      "worst_case_discordance_power_not_guaranteed": True,
                      "sample_size_changes_after_scores_forbidden": True},
        "pilot_gate_policy": "new_unfiltered_feasibility_collection; old pilot gate unchanged",
        "positive_rule": "official 10x5 attempts; 16 slots; stop document at first exhausted slot",
        "llm_null_rule": "same model/settings/length/context; 50 length-only attempts per slot",
        "null_matching_limit": "domain/prompt/length matched; no AMR acceptance conditioning",
        "selection": "all frozen allocations retained; no score filtering or replacement",
        "files": {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*"))
                  if p.is_file() and p.name != ".gitignore" and "__pycache__" not in p.parts},
    })
    return verify(out)


def journal(out, phase, key):
    budget.CAPS = dict(CAPS)
    j = generation.ReplayJournal(out, phase, CAPS[phase][0],
                                lambda body: budget.transport(body, key),
                                sha256(out / "FREEZE.json"), parents=[])
    if j.uncertain:
        raise RuntimeError("Unknown provider consumption; cannot redispatch")
    return j


def null_slot(out, source, slot, context, j, tokenize):
    identity = source["allocation_id"] + f"-null-slot-{slot:02d}"
    path = out / "private/null_slots" / (identity + ".json")
    binding = {"source_id": source["allocation_id"], "slot": slot,
               "context_sha256": rt.digest(context), "freeze_sha256": sha256(out / "FREEZE.json")}
    if path.exists():
        saved = load(path)
        if any(saved[k] != v for k, v in binding.items()):
            raise RuntimeError("Null slot identity changed")
        for call in saved["call_ids"]:
            if call not in j.calls or j.calls[call]["terminal"]["kind"] != "complete":
                raise RuntimeError("Null slot accounting missing")
        return saved
    calls = []
    for attempt in range(50):
        cid = identity + f"-attempt-{attempt:02d}"
        response = j.request(cid, NULL_PROMPT + context, "", 50, .7, .9)
        calls.append(cid)
        text = response["text"]
        valid = (response["stop_reason"] == "stop" and 7 <= len(WORD.findall(text)) <= 12
                 and tokenize(text) == [text] and text.endswith(".") and not text.endswith(".."))
        if valid:
            break
    return freeze(path, {**binding, "text": text, "accepted": valid, "call_ids": calls})


def generate(out, key_file, phase):
    verify(out)
    seal = out / (phase.upper() + "_GENERATION_SEAL.json")
    if seal.exists():
        verify_seal(out, seal)
        return
    generation.OUT = pilot.OUT = out
    key = pilot.credential.read_credential(key_file)
    j = journal(out, phase, key)
    main = phase == "main"
    if main:
        sources = load(out / "private/sources/main.json")
    else:
        design = load(out / "private/ALIGNMENT_DESIGN.json")
        ids = set(design["llm_calibration"] + design["llm_test"])
        sources = [r for p in ("calibration", "test")
                   for r in load(out / "private/sources" / (p + ".json")) if r["allocation_id"] in ids]
    folder = out / "private" / ("documents" if main else "null_documents")
    tokenize = tokenizer()

    def collect(module=None):
        bank = generation.load_bank(out / "private/amr_bank_50.json") if main else None
        if main:
            ready = out / "GENERATION_READINESS.json"
            # Re-run readiness after each process restart; only stable result is sealed.
            result = readiness(module, bank)
            if not ready.exists():
                freeze(ready, result)
        for index, source in enumerate(sources):
            path = folder / (source["allocation_id"] + ".json")
            if path.exists():
                record = load(path)
                expected_hash = rt.digest(json.dumps(
                    {k: v for k, v in record.items() if k != "record_sha256"}, sort_keys=True))
                if (record.get("record_sha256") != expected_hash or
                        record.get("source_id") != source["allocation_id"] or
                        record.get("freeze_sha256") != sha256(out / "FREEZE.json")):
                    raise RuntimeError("Generated document checkpoint changed")
                continue
            sentences, failure = [], None
            for slot in range(16):
                context = source["generation_prompt"] + " " + " ".join(sentences)
                record = (generation.run_slot(module, source, slot, context, bank, j) if main else
                          null_slot(out, source, slot, context, j, tokenize))
                sentences.append(record["text"])
                if not record["accepted"]:
                    failure = "slot_exhausted"
                    break
                if main:
                    # The old runner permitted length-terminated accepted candidates.
                    # This successor records that distinction and rejects the document.
                    hashes = record["attempt_file_hashes"]
                    final = load(out / "private/attempts" / (list(hashes)[-1] + ".json"))
                    if final["stop_reason"] != "stop":
                        failure = "accepted_but_truncated"
                        break
            text = " ".join(sentences)
            eligible = (failure is None and len(sentences) == 16
                        and tokenize(text) == sentences and 100 <= len(WORD.findall(text)) <= 220)
            record = {"source_id": source["allocation_id"], "sentences": sentences,
                      "text": text, "eligible": eligible, "failure": failure,
                      "words": len(WORD.findall(text)), "freeze_sha256": sha256(out / "FREEZE.json")}
            freeze(path, {**record, "record_sha256": rt.digest(json.dumps(record, sort_keys=True))})
            rt.atomic(out / "HEARTBEAT.json", {"phase": phase + "_generation",
                                             "complete": index + 1, "total": len(sources),
                                             "budget": j.totals(), "at": rt.now()})
    if main:
        with runtime(out) as module:
            collect(module)
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    else:
        collect()
    files = list(folder.glob("*.json"))
    if len(files) != len(sources):
        raise RuntimeError("Generation denominator incomplete")
    freeze(out / (phase.upper() + "_BUDGET.json"), {
        k: v for k, v in j.totals().items() if k != "new_calls"})
    file_seal(out, seal, files + [out / (phase.upper() + "_BUDGET.json")])


def pair_rows(source_id, sentences, syntax, tokenize):
    original = " ".join(sentences)
    try:
        pair = pilot.make_pair(source_id, sentences, syntax, tokenize)
        return {c: pair["variants"][c]["text"] for c in ("A", "B")}, pair
    except ValueError as exc:
        return {"A": None, "B": None}, {"reason": str(exc), "source_text_sha256": rt.digest(original)}


def assemble(out):
    verify_seal(out, out / "MAIN_GENERATION_SEAL.json")
    sealed = out / "INPUT_SEAL.json"
    if sealed.exists():
        verify_seal(out, sealed)
        return load(out / "private/INPUTS.json")
    syntax, tokenize = pilot.FrozenSyntax(), tokenizer()
    docs, reviews, key = [], [], []
    for role, phase in (("calibration", "calibration"), ("null_test", "test"), ("positive", "main")):
        for source in load(out / "private/sources" / (phase + ".json")):
            sid = source["allocation_id"]
            if phase == "main":
                generated = load(out / "private/documents" / (sid + ".json"))
                sentences = generated["sentences"] if generated["eligible"] else []
            else:
                sentences = source["control_window"]["sentences"]
            clean = " ".join(sentences) if sentences else None
            pairs, evidence = pair_rows(sid, sentences, syntax, tokenize) if clean else (
                {"A": None, "B": None}, {"reason": "generation_failure"})
            freeze(out / "private/pairs" / (sid + ".json"), evidence)
            for condition, text in {"clean": clean, **pairs}.items():
                docs.append({"source_id": sid, "role": role, "condition": condition,
                             "text": text, "text_sha256": rt.digest(text) if text else None,
                             "sentences": tokenize(text) if text else [],
                             "unavailable_reason": None if text else evidence.get("reason", "transform_failure")})
            if role == "positive" and pairs["A"] and pairs["B"]:
                item = rt.digest("quality\0" + sid)[:20]
                order = ["A", "B"]
                random.Random(int(item, 16)).shuffle(order)
                reviews.append({"item_id": item, "versions": {
                    str(i + 1): pairs[c] for i, c in enumerate(order)}})
                key.append({"item_id": item, "source_id": sid})
    freeze(out / "private/INPUTS.json", docs)
    freeze(out / "private/human_review/PAIR_PACKET.json", {
        "items": reviews, "independent_human_reviewers_required": 2})
    freeze(out / "private/QUALITY_KEY.json", key)
    for reviewer in (1, 2):
        path = out / "private/human_review" / f"reviewer{reviewer}_UNFILLED.csv"
        if not path.exists():
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "item_id", "version1_grammar", "version1_coherence", "version1_no_AMR_leak",
                    "version2_grammar", "version2_coherence", "version2_no_AMR_leak",
                    "meaning_preserved", "reviewer_id", "is_human", "notes"])
                writer.writeheader()
                writer.writerows({"item_id": r["item_id"]} for r in reviews)
    file_seal(out, sealed, [out / "private/INPUTS.json", out / "private/QUALITY_KEY.json",
                           out / "private/human_review/PAIR_PACKET.json",
                           *sorted((out / "private/pairs").glob("*.json"))])
    return docs


def score_rows(out, assets, environment, docs):
    from run_swan_independent_banks1 import parse, Normalizer
    from run_swan_independent_calibration1 import Matcher
    from semantic_fact_raw_amr import audit_raw
    f = verify(out)
    if sha256(environment / "FREEZE.json") != f["bank_environment_freeze_sha256"]:
        raise RuntimeError("Normalizer environment changed")
    parse(out, assets, f, docs)
    normalizer, matcher = Normalizer(environment), Matcher(out, assets)
    bank = generation.load_bank(out / "private/amr_bank_50.json")
    if [matcher.fresh(bank[0], bank[0]) for _ in range(2)] != [1., 1.]:
        raise RuntimeError("S2Match replay failed")
    cache, rows, cuts = {}, [], None
    for index, doc in enumerate(docs):
        if doc["role"] != "calibration" and cuts is None:
            cuts = calibration(rows)
            freeze(out / "THRESHOLDS.json", cuts)
        path = out / "private/scores" / (rt.digest(doc["source_id"] + "\0" + doc["condition"]) + ".json")
        binding = {k: doc[k] for k in ("source_id", "role", "condition", "text_sha256")}
        binding["freeze_sha256"] = sha256(out / "FREEZE.json")
        if path.exists():
            envelope = load(path)
            row = envelope["row"]
            if envelope["sha256"] != rt.digest(json.dumps(row, sort_keys=True)) or any(
                    row[k] != v for k, v in binding.items()):
                raise RuntimeError("Score checkpoint changed")
        else:
            maxima, erased = [], 0
            for text in doc["sentences"]:
                h = rt.digest(text)
                if h not in cache:
                    raw = load(out / "private/raw_graphs" / (h + ".json"))
                    check = audit_raw(raw, {"case_id": h, "input_text_sha256": h}, binding["freeze_sha256"])
                    result = normalizer(raw["amr"]) if check["structurally_eligible"] else None
                    template = result["template"] if result and result["eligible"] else None
                    cache[h] = (max(matcher.pair(template, t) for t in bank) if template else None)
                maxima.append(cache[h])
                erased += cache[h] is None
            n = len(maxima)
            green = sum(v is not None and v >= .65 for v in maxima)
            z = (green - .25 * n) / math.sqrt(.25 * .75 * n) if n >= 2 else None
            row = {**binding, "score": z, "N": n, "green": green, "erased": erased,
                   "sentence_scores": maxima, "unavailable_reason": doc["unavailable_reason"]}
            freeze(path, {"row": row, "sha256": rt.digest(json.dumps(row, sort_keys=True))})
        rows.append(row)
        rt.atomic(out / "HEARTBEAT.json", {"phase": "W02_scoring", "complete": index + 1,
                                         "total": len(docs), "at": rt.now()})
        if shutil.disk_usage(out).free < 512 * 1024**2:
            raise rt.ResourcePause("Disk headroom")
    matcher.db.close()
    if cuts != calibration(rows):
        raise RuntimeError("Threshold replay differs")
    freeze(out / "private/SCORES.json", rows)
    result = {"status": "W02_MACHINE_COMPLETE_QUALITY_PENDING", **evaluate(rows, cuts),
              "independent_human_reviews": 0, "formal_ASR": None, "quality_adjusted_TPR": None}
    freeze(out / "RESULT.json", result)
    file_seal(out, out / "AUDIT.json", [
        out / "private/SCORES.json", out / "THRESHOLDS.json", out / "INPUT_SEAL.json",
        out / "RAW_GRAPH_SEAL.json", out / "MAIN_GENERATION_SEAL.json"])
    freeze(out / "TERMINAL.json", {"status": result["status"],
                                   "result_sha256": sha256(out / "RESULT.json"),
                                   "audit_sha256": sha256(out / "AUDIT.json")})


def calibration(rows):
    selected = [r for r in rows if r["role"] == "calibration"]
    mapped = {(r["source_id"], r["condition"]): r for r in selected}
    ids = sorted({r["source_id"] for r in selected})
    if not ids or len(mapped) != len(selected) or set(mapped) != {(s, c) for s in ids for c in CONDITIONS}:
        raise ValueError("Calibration source-condition product missing")
    return {c: threshold([mapped[s, c]["score"] for s in ids], .01) for c in CONDITIONS}


def evaluate(rows, cuts):
    result = {}
    for role in ("null_test", "positive"):
        selected = [r for r in rows if r["role"] == role]
        ids = sorted({r["source_id"] for r in selected})
        mapped = {(r["source_id"], r["condition"]): r for r in selected}
        if len(mapped) != len(selected) or set(mapped) != {(s, c) for s in ids for c in CONDITIONS}:
            raise ValueError("Full evaluation denominator required")
        result[role] = {c: {
            "own_condition_threshold": binomial(sum(reject(mapped[s, c]["score"], cuts[c]) for s in ids), len(ids), .05/6),
            "clean_threshold": binomial(sum(reject(mapped[s, c]["score"], cuts["clean"]) for s in ids), len(ids), .05/6),
            "unavailable": sum(mapped[s, c]["score"] is None for s in ids),
        } for c in CONDITIONS}
        if role == "positive":
            complete = [s for s in ids if all(mapped[s, c]["score"] is not None for c in ("A", "B"))]
            differences = [mapped[s, "B"]["score"] - mapped[s, "A"]["score"] for s in complete]
            # Preselected A/B, no feedback. Do not label this a quality-gated attack ASR.
            result["count_neutral"] = {
                "full_sources": len(ids), "complete_pairs": len(complete),
                "mean_B_minus_A_complete_pairs": sum(differences)/len(differences) if differences else None,
                "paired_clean_threshold_detection_difference": paired_difference(
                    [reject(mapped[s, "B"]["score"], cuts["clean"]) for s in ids],
                    [reject(mapped[s, "A"]["score"], cuts["clean"]) for s in ids]),
                "interpretation": "detector-blind paired boundary dependence; human quality pending"}
    return result


def quality(out, first, second, destination):
    from swan_supplement1_review import read_review
    verify(out)
    verify_seal(out, out / "INPUT_SEAL.json")
    terminal = load(out / "TERMINAL.json")
    if terminal["result_sha256"] != sha256(out / "RESULT.json") or terminal["audit_sha256"] != sha256(out / "AUDIT.json"):
        raise RuntimeError("Machine terminal changed")
    verify_seal(out, out / "AUDIT.json")
    items = load(out / "private/human_review/PAIR_PACKET.json")["items"]
    ids = {r["item_id"] for r in items}
    a, labels_a = read_review(first, ids)
    b, labels_b = read_review(second, ids)
    if a == b or sha256(first) == sha256(second):
        raise ValueError("Two independent humans required")
    mapping = {r["item_id"]: r["source_id"] for r in load(out / "private/QUALITY_KEY.json")}
    passing = [mapping[k] for k in ids if labels_a[k] and labels_b[k]]
    rows = load(out / "private/SCORES.json")
    cuts = load(out / "THRESHOLDS.json")
    bykey = {(r["source_id"], r["condition"]): r for r in rows if r["role"] == "positive"}
    return freeze(destination, {
        "status": "W02_INDEPENDENT_PAIR_QUALITY_COLLECTED", "full_sources": 200,
        "eligible_pair_items": len(ids), "two_reviewer_quality_pass": len(passing),
        "quality_and_detected_over_full_200": {c: sum(
            reject(bykey[s, c]["score"], cuts[c]) for s in passing) / 200 for c in ("A", "B")},
        "passing_source_ids": sorted(passing), "review_file_sha256": [sha256(first), sha256(second)],
        "machine_terminal_sha256": sha256(out / "TERMINAL.json"),
        "independence_basis": "distinct self-attested humans; not identity verification",
        "clean_text_quality_reviewed": False})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("prepare", "verify", "generate", "null-generate", "detect", "alignment", "quality"))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--assets", type=Path, default=ROOT)
    ap.add_argument("--environment", type=Path, default=BANK_ENV)
    ap.add_argument("--alignment-assets", type=Path, default=ALIGNMENT)
    ap.add_argument("--key-file", type=Path)
    ap.add_argument("--reviewer1", type=Path)
    ap.add_argument("--reviewer2", type=Path)
    ap.add_argument("--result", type=Path)
    args = ap.parse_args()
    if args.stage == "prepare":
        print(json.dumps(prepare(args.out)["counts"]))
        return
    f = verify(args.out)
    if args.stage == "verify":
        print(json.dumps({"verified_files": len(f["files"]), "counts": f["counts"]}))
        return
    halt = args.out / (args.stage.upper() + "_HALT.json")
    if halt.exists():
        raise RuntimeError("Frozen stage halt requires successor")
    with single_writer(args.out):
        start = time.monotonic()
        try:
            if args.stage in ("generate", "null-generate"):
                generate(args.out, args.key_file, "main" if args.stage == "generate" else "llm_null")
            elif args.stage == "detect":
                if (args.out / "TERMINAL.json").exists():
                    terminal = load(args.out / "TERMINAL.json")
                    if any(terminal[k] != sha256(args.out / v) for k, v in (
                            ("result_sha256", "RESULT.json"), ("audit_sha256", "AUDIT.json"))):
                        raise RuntimeError("Terminal changed")
                    verify_seal(args.out, args.out / "AUDIT.json")
                else:
                    score_rows(args.out, args.assets, args.environment, assemble(args.out))
            elif args.stage == "alignment":
                from swan_normal_alignment1 import run
                run(args.out, args.assets, args.alignment_assets)
            else:
                quality(args.out, args.reviewer1, args.reviewer2, args.result)
        except Exception as exc:
            resource = isinstance(exc, (ResourcePause, rt.ResourcePause)) or type(exc).__name__ == "OutOfMemoryError"
            limit = isinstance(exc, budget.Pause)
            dependent = isinstance(exc, FileNotFoundError)
            rt.atomic(args.out / (args.stage.upper() + ("_PAUSED.json" if resource or limit or dependent else "_HALT.json")),
                      {"reason_type": type(exc).__name__, "at": rt.now(), "scientific_result": None,
                       "resource_pause": resource, "budget_pause": limit, "missing_dependency": dependent})
            if resource or limit or dependent:
                raise SystemExit(75 if resource else 76)
            raise
        finally:
            rt.atomic(args.out / "private/sessions" / (str(time.time_ns()) + ".json"),
                      {"stage": args.stage, "seconds": time.monotonic() - start})


if __name__ == "__main__":
    main()
