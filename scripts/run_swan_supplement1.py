"""Pilot generation and blind review preparation for SWAN supplement1."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import unicodedata

from swan_final1_common import digest, freeze, load, now, sha256, single_writer
import swan_final1_budget as budget
from swan_final1_transform import FrozenSyntax
import run_swan_positive_deepseek_length_confirmation3 as credential
import run_swan_positive_collection as swan
from swan_supplement1_sources import ROOT, OUT, WORD, tokenizer, audit
from swan_supplement1_runtime import content_seed, IntegrityHalt, runtime, readiness

PILOT_CALL_CAP = 6400
LENGTH = (
    "\n\nFixed additional constraints: Write 7 to 12 lexical words in one complete "
    "sentence ending with a period. Continue the general topic of the context, "
    "using plausible fictional details when needed. Respect the AMR structure. "
    "Do not output graph notation, placeholders, headings or explanations."
)
DEPENDENCIES = [
    "scripts/run_swan_supplement1.py", "scripts/swan_supplement1_runtime.py",
    "scripts/swan_supplement1_sources.py", "scripts/swan_supplement1_review.py",
    "scripts/swan_final1_common.py",
    "scripts/swan_final1_budget.py", "scripts/swan_final1_transform.py",
    "scripts/swan_final1_sources.py", "scripts/swan_cached_s2match.py",
    "scripts/run_swan_positive_collection.py",
    "scripts/run_swan_positive_deepseek_length_confirmation3.py",
    "scripts/prepare_swan_positive_collection.py",
    "third_party/SWAN_official_2ca0c207_clean/injection/injection_amr_utils.py",
    "third_party/SWAN_official_2ca0c207_clean/utils/amr_utils.py",
    "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_50.json",
    "third_party/SWAN_official_2ca0c207_clean/amr_bank/artifacts/normalized_amr_examples.json",
    "third_party/swan_s2match_runtime_711a231d/s2match.py",
    "third_party/swan_s2match_runtime_711a231d/amr_py3.py",
    "third_party/swan_s2match_runtime_711a231d/helpers.py",
    "models/model_parse_xfm_bart_large-v0_1_0/pytorch_model.bin",
    "models/model_parse_xfm_bart_large-v0_1_0/config.json",
    "models/model_parse_xfm_bart_large-v0_1_0/amrlib_meta.json",
    "models/bart-large-tokenizer/tokenizer.json",
    "models/bart-large-tokenizer/tokenizer_config.json",
    "models/bart-large-tokenizer/config.json",
    "models/bart-large-tokenizer/merges.txt", "models/bart-large-tokenizer/vocab.json",
    "vectors/glove.6B.100d.txt", ".nltk_data/corpora/propbank.zip",
    ".nltk_data/tokenizers/punkt_tab/english/ortho_context.tab",
    ".nltk_data/tokenizers/punkt_tab/english/abbrev_types.txt",
    ".nltk_data/tokenizers/punkt_tab/english/collocations.tab",
    ".nltk_data/tokenizers/punkt_tab/english/sent_starters.txt",
    "experiments/swan_count_neutral_supplement1/PROTOCOL.md",
    "experiments/swan_count_neutral_supplement1/HUMAN_REVIEW.md",
    "experiments/swan_count_neutral_supplement1/source_manifest.json",
    "experiments/swan_count_neutral_supplement1/SOURCE_AUDIT.json",
]


def heartbeat(stage, **details):
    path = OUT / "heartbeat.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w") as stream:
        json.dump({"at": now(), "stage": stage, **details}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def prepare():
    audit()
    dependencies = {p: sha256(ROOT / p) for p in DEPENDENCIES}
    return freeze(OUT / "FREEZE.json", {
        "at": now(), "experiment": "swan_count_neutral_supplement1",
        "stage": "pilot", "dependencies": dependencies,
        "source_file_hashes": load(OUT / "source_manifest.json")["sealed_file_hashes"],
        "generator": "deepseek-flash", "prompt_addition": LENGTH,
        "pilot": {"documents": 8, "slots": 16, "max_calls": 6400,
                  "agent_selected_cost_guard_usd": 20, "worst_case_usd": 18.07872},
        "user_instruction": "行，那就补全一下实验吧",
        "authorization_basis": "User authorized supplemental experiment; guard selected by agent.",
        "main_dispatched": False,
        "http_retries": 0,
    })


def verify():
    contract = load(OUT / "FREEZE.json")
    for path, expected in contract["dependencies"].items():
        if sha256(ROOT / path) != expected:
            raise IntegrityHalt(f"dependency_SHA_changed:{path}")
    for path, expected in contract["source_file_hashes"].items():
        if sha256(OUT / path) != expected:
            raise IntegrityHalt(f"source_SHA_changed:{path}")
    return contract


def make_pair(allocation_id, sentences, syntax, tokenize=None):
    tokenize = tokenize or tokenizer()
    original = " ".join(sentences)
    if len(sentences) != 16 or tokenize(original) != sentences:
        raise ValueError("source_sentence_recovery_failed")
    def merge(index):
        return sentences[:index] + [
            sentences[index][:-1] + "; " + sentences[index + 1]] + sentences[index + 2:]
    boundaries = [i for i in range(15)
                  if sentences[i].endswith(".") and not sentences[i].endswith("..")
                  and syntax(sentences[i])["complete"] and syntax(sentences[i + 1])["complete"]
                  and tokenize(" ".join(merge(i))) == merge(i)]
    pairs = [(a, b) for j, a in enumerate(boundaries) for b in boundaries[j + 1:] if b - a > 1]
    if not pairs:
        raise ValueError("no_nonadjacent_merge_pair")
    choice = content_seed("boundary", allocation_id, original)
    a, b = pairs[choice % len(pairs)]
    if (choice // len(pairs)) % 2:
        a, b = b, a
    variants = {arm: {"boundary": i, "units": merge(i), "text": " ".join(merge(i))}
                for arm, i in (("A", a), ("B", b))}
    punctuation = lambda s: Counter(c for c in s if unicodedata.category(c).startswith("P"))
    checks = {
        "n_A": len(variants["A"]["units"]), "n_B": len(variants["B"]["units"]),
        "lexical_sequence_equal": WORD.findall(variants["A"]["text"]) ==
            WORD.findall(variants["B"]["text"]) == WORD.findall(original),
        "punctuation_multiset_equal": punctuation(variants["A"]["text"]) == punctuation(variants["B"]["text"]),
        "nonadjacent_boundaries": abs(a - b) > 1,
    }
    if checks["n_A"] != 15 or checks["n_B"] != 15 or not all(
            v for k, v in checks.items() if not k.startswith("n_")):
        raise IntegrityHalt("pair_invariant_failed")
    return {"allocation_id": allocation_id, "source_text_sha256": digest(original),
            "selection_seed": choice, "candidate_pairs": len(pairs),
            "variants": variants, "checks": checks, "detector_feedback": False}


def run_slot(module, source, slot, context, bank, journal):
    allocation = source["allocation_id"]
    slot_id = f"{allocation}-slot-{slot:02d}"
    path = OUT / "private/slots" / f"{slot_id}.json"
    identity = {"slot": slot, "allocation_id": allocation, "context_sha256": digest(context),
                "contract_sha256": sha256(OUT / "FREEZE.json")}
    if path.exists():
        saved = load(path)
        if any(saved[k] != v for k, v in identity.items()):
            raise IntegrityHalt("slot_identity_changed")
        return saved
    if any(c.startswith(slot_id + "-") for c in journal.calls):
        raise IntegrityHalt("incomplete_slot_requires_audit_no_retry")
    events = []
    saved = {k: getattr(module, k) for k in
             ("BedrockManager", "build_user_prompt", "parse_amr", "decide_s2match_match", "random")}
    class Manager:
        def __init__(self, region, model_id):
            if model_id != budget.MODEL:
                raise IntegrityHalt("model_identity_changed")
        def generate(self, user_text, system_text, max_tokens, temperature, top_p):
            call_id = f"{slot_id}-attempt-{len(events):02d}"
            event = {"call_id": call_id, "template": None, "candidate": None}
            events.append(event)
            heartbeat("pilot_generating", allocation=allocation, slot=slot,
                      attempt=len(events), calls=journal.totals()["calls"])
            response = journal.request(call_id, user_text, system_text, max_tokens, temperature, top_p)
            event.update(candidate=response["text"], response_sha256=response["call_sha256"],
                         stop_reason=response["stop_reason"])
            return response["text"]
    def parse(text):
        result = saved["parse_amr"](text)
        events[-1]["raw_amr"] = result
        return result
    def match(candidate, target, threshold):
        accepted, f = saved["decide_s2match_match"](candidate, target, threshold)
        events[-1].update(template=target, candidate_template=candidate,
                          score=f, accepted=accepted)
        freeze(OUT / "private/attempts" / f"{events[-1]['call_id']}.json", events[-1])
        return accepted, f
    module.BedrockManager = Manager
    module.build_user_prompt = lambda *a, **k: saved["build_user_prompt"](*a, **k) + LENGTH
    module.parse_amr, module.decide_s2match_match = parse, match
    module.random = random.Random(content_seed("template", allocation, slot))
    try:
        text, count, template = module.amr_instruction_completion(
            context=context, amr_bank=bank, model_id=budget.MODEL, region=budget.URL,
            max_new_tokens=50, temperature=.7, top_p=.9, max_trials=50,
            amr_examples_path=str(swan.OFFICIAL_ROOT / "amr_bank/artifacts/normalized_amr_examples.json"),
            s2match_threshold=.75, max_templates_to_try=10, max_retries_per_template=5)
    finally:
        for k, v in saved.items():
            setattr(module, k, v)
    if count != len(events) or text.strip() != events[-1]["candidate"].strip():
        raise IntegrityHalt("official_return_trace_mismatch")
    return freeze(path, {**identity, "text": text.strip(), "attempts": count,
                         "accepted": events[-1]["accepted"], "fallback": not events[-1]["accepted"],
                         "template": template, "score": events[-1]["score"],
                         "attempt_file_hashes": {e["call_id"]: sha256(
                             OUT / "private/attempts" / f"{e['call_id']}.json") for e in events}})


def review_packet():
    packet_path = OUT / "private/human_review/pilot_packet.json"
    if packet_path.exists():
        return load(packet_path)
    documents = []
    for path in sorted((OUT / "private/documents").glob("*.json")):
        document = load(path)
        if not document["engineering_pass"]:
            continue
        pair = load(OUT / "private/pairs" / path.name)
        item_id = digest(["blind", document["allocation_id"]])[:16]
        order = ["A", "B"]
        random.Random(content_seed("review_order", item_id)).shuffle(order)
        documents.append({"item_id": item_id,
                          "versions": {str(i + 1): pair["variants"][arm]["text"]
                                       for i, arm in enumerate(order)}})
    packet = freeze(packet_path, {"at": now(), "protocol_sha256": sha256(OUT / "PROTOCOL.md"),
                                 "items": documents, "independent_human_reviewers_required": 2})
    for reviewer in ("reviewer1", "reviewer2"):
        path = packet_path.parent / f"{reviewer}.csv"
        if path.exists():
            continue
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "item_id", "version1_grammar", "version1_coherence", "version1_no_AMR_leak",
                "version2_grammar", "version2_coherence", "version2_no_AMR_leak",
                "meaning_preserved", "reviewer_id", "is_human", "notes"])
            writer.writeheader()
            writer.writerows({"item_id": d["item_id"]} for d in documents)
    return packet


def summarize(journal=None):
    documents = [load(p) for p in sorted((OUT / "private/documents").glob("*.json"))]
    slots = [load(p) for p in sorted((OUT / "private/slots").glob("*.json"))]
    successes = sum(d["engineering_pass"] for d in documents)
    halted = (OUT / "HALT.json").exists()
    terminal = len(documents) == 8
    stage = "Halt" if halted else "Stop" if terminal and successes < 6 else \
        "Waiting-human-review" if terminal else "Running"
    result = {"at": now(), "stage": stage, "pilot_documents_complete": len(documents),
              "pilot_denominator": 8, "engineering_pass": successes,
              "slots_complete": len(slots), "slots_accepted": sum(s["accepted"] for s in slots),
              "calls": sum(s["attempts"] for s in slots),
              "human_reviews_complete": 0, "main_started": False,
              "calibration_started": False, "test_started": False, "detector_calls": 0}
    if journal is not None:
        result["budget"] = journal.totals()
    heartbeat(stage, **{k: v for k, v in result.items() if k not in ("at", "stage")})
    path = OUT / "STATUS.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temp, path)
    return result


def run(key_file):
    verify()
    if (OUT / "HALT.json").exists():
        raise IntegrityHalt("permanent_halt_exists")
    key = credential.read_credential(key_file)
    budget.CAPS = {"pilot": (PILOT_CALL_CAP, 20_000_000_000)}
    journal = budget.Journal(OUT, "pilot", PILOT_CALL_CAP,
                             lambda body: budget.transport(body, key), sha256(OUT / "FREEZE.json"))
    if journal.uncertain:
        raise IntegrityHalt("uncertain_call_blocks_resume")
    bank = load(swan.OFFICIAL_ROOT / "amr_bank/banks/amr_bank_50.json")
    if isinstance(bank, dict):
        raise IntegrityHalt("bank_schema_changed")
    syntax = FrozenSyntax()
    heartbeat("CUDA_readiness")
    with runtime() as module:
        if not (OUT / "READINESS.json").exists():
            freeze(OUT / "READINESS.json", readiness(module, bank))
        for source in load(OUT / "private/pilot_sources.json"):
            path = OUT / "private/documents" / f"{source['allocation_id']}.json"
            if path.exists():
                continue
            sentences, failure = [], None
            for slot in range(16):
                context = source["generation_prompt"] + " " + " ".join(sentences)
                record = run_slot(module, source, slot, context, bank, journal)
                sentences.append(record["text"])
                if not record["accepted"]:
                    failure = "slot_exhausted_fallback_is_failure"
                    break
            pair = None
            if failure is None:
                try:
                    pair = make_pair(source["allocation_id"], sentences, syntax)
                except ValueError as exc:
                    failure = str(exc)
            if pair is not None:
                freeze(OUT / "private/pairs" / path.name, pair)
            freeze(path, {"allocation_id": source["allocation_id"], "at": now(),
                          "engineering_pass": failure is None, "failure": failure,
                          "slots": len(sentences), "words": len(WORD.findall(" ".join(sentences))),
                          "text": " ".join(sentences)})
            summarize(journal)
    review_packet()
    result = summarize(journal)
    freeze(OUT / "PILOT_RESULT.json", result)
    return result


def environment():
    names = ["torch", "transformers", "amrlib", "spacy", "nltk", "penman", "scipy",
             "requests", "numpy", "en-core-web-sm"]
    return {name: importlib.metadata.version(name) for name in names}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "status", "review-packet"))
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    with single_writer(OUT):
        try:
            if args.stage == "prepare":
                result = prepare()
                print(json.dumps({"freeze": True, "dependencies": len(result["dependencies"])}))
            elif args.stage == "run":
                freeze(OUT / "ENVIRONMENT.json", environment())
                result = run(args.key_file)
                print(json.dumps(result))
            elif args.stage == "status":
                print(json.dumps(summarize()))
            else:
                print(json.dumps({"review_items": len(review_packet()["items"])}))
        except Exception as exc:
            if args.stage == "run":
                halt = {"at": now(), "error_type": type(exc).__name__,
                        "reason": str(exc) if isinstance(exc, (IntegrityHalt, budget.GlobalStop, budget.DocumentFailure))
                        else "execution_error_details_private", "automatic_restart_allowed": False}
                freeze(OUT / "HALT.json", halt)
                review_packet()
                summarize()
            print(json.dumps({"stage": "Halt", "error_type": type(exc).__name__}), file=sys.stderr)
            sys.exit(1)
