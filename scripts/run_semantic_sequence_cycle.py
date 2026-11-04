"""New, gated write/read repair pilot; never resumes a historical manifest."""
from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import fcntl
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import random
import re
import sys
import threading
import time

import penman
from penman.models.amr import model as AMR_MODEL
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/semantic_sequence_cycle1"
PRIVATE = Path.home() / ".cache/amr/semantic_sequence_cycle1"
BANK = ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_100.json"
OLD_MAP = ROOT / "experiments/semantic_sequence_stage1_run2/SYMBOL_MAP.json"
SEED = 20261024
PARAMETERS = {"model": "deepseek-flash", "temperature": 0.9, "top_p": 1,
              "max_tokens": 1600, "n": 1, "stream": False,
              "presence_penalty": 0, "frequency_penalty": 0,
              "thinking": {"type": "disabled"}}
API_URL = "https://api.deepseek.com/chat/completions"
SYSTEM = ("Write natural fictional English for a controlled research study. "
          "Output only the requested prose, without title, list or explanation.")
SCENARIOS = [
    "planning a removable awning for an outdoor pottery stall",
    "organizing a neighborhood seedling delivery by bicycle",
    "comparing echo patterns using cardboard tubes in a fictional workshop",
    "designing a folding rack for wet theatre costumes",
    "retrieving a borrowed telescope before an evening gathering",
    "testing how colored paper changes under different lamps",
    "preparing reusable signs for a weekend repair fair",
    "moving a music stand between two community rooms",
    "observing how toy wheels move across soft fabric",
    "arranging a hand-operated exhibit of paper mechanisms",
    "restoring a damaged album for a family celebration",
    "comparing the buoyancy of carved wooden shapes",
    "planning a portable shade screen for an art class",
    "transporting handmade masks to a rehearsal",
    "observing how folded cards support small weights",
    "designing a modular tray for a craft demonstration",
    "organizing borrowed tools after a courtyard workshop",
    "comparing the cooling of sealed cups in a fictional class",
    "preparing adjustable display stands for a textile fair",
    "finding a misplaced stencil before decorating a room",
    "exploring light paths through water-filled glass shapes",
    "planning a wheeled cabinet for a traveling puppet group",
    "sorting ceramic pieces for a collaborative wall panel",
    "comparing vibration patterns in stretched paper strips",
]
FIRST = ("Alda", "Borin", "Celia", "Daven", "Elsa", "Farlen", "Gilda", "Hovan",
         "Ilva", "Jorin", "Kaela", "Loden", "Mira", "Norvin", "Olia", "Peren",
         "Quilla", "Ronan", "Sera", "Tobin", "Ulla", "Valen", "Willa", "Xeran")
LAST = ("Ashford", "Bellwood", "Clifford", "Daleson", "Eastwell", "Firthall",
        "Greenvale", "Hillford", "Ironwood", "Jadewell", "Kingsley", "Lakeson",
        "Moorfield", "Northall", "Oakwell", "Pineford", "Quarryton", "Ridgewell",
        "Stonevale", "Thornfield", "Underhill", "Valecrest", "Westford", "Yewdale")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        stream.write(encode(value))
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def required_markers(template):
    graph = penman.decode(template, model=AMR_MODEL)
    concepts = {s: t for s, r, t in graph.triples if r == ":instance"}
    name_nodes = {t for s, r, t in graph.triples if r == ":name"}
    sizes = sorted(sum(s == node and bool(re.fullmatch(r":op\d+", r))
                       for s, r, t in graph.triples) for node in name_nodes)
    return {"unknown_nodes": sum(t == "X" for t in concepts.values()),
            "name_sizes": sizes,
            "imperative": any(r == ":mode" and t == "imperative" for s, r, t in graph.triples),
            "polite": any(r == ":polite" and t == "+" for s, r, t in graph.triples),
            "unknown_roles": sorted(r for s, r, t in graph.triples
                                    if concepts.get(t) == "X" and r != ":instance")}


def surface_contract(markers):
    if markers["imperative"]:
        speech = "Use a direct imperative/request, preserving who is asked to do what."
    elif ":polarity" in markers["unknown_roles"]:
        speech = "Use a yes/no question expressing unknown polarity, not an assertion or ordinary negation."
    elif markers["unknown_nodes"]:
        speech = "Use a genuine information-seeking question; do not answer its unknown information."
    else:
        speech = "Use a natural statement."
    return (speech + " Aim for 12 to 22 whitespace-separated words in one sentence. "
            "This is a soft length target; preserve the required meaning if more words are needed. "
            "Use complete one-, two-, or three-part names as needed from the supplied participants. "
            "Do not replace a two-part name with two separate people. "
            + ("Use an explicit polite request." if markers["polite"] else ""))


def carrier_messages(prompt, prior, slot, markers, template=None):
    text = (f"SCENARIO\n{prompt}\n\nPRECEDING TEXT\n{prior or '(none)'}\n\n"
            f"Write sentence {slot + 1} of eight. Continue the account naturally. "
            f"{surface_contract(markers)}")
    if template is not None:
        text += (
            "\nRealize the following normalized AMR structure. V is a predicate, "
            "N a concept, NE a named entity, and ADJ an adjective. X specifically "
            "means amr-unknown: an unknown answer in a question, never just an "
            "unspecified ordinary object. Its role determines what is unknown. "
            "A :polarity X node asks whether the proposition is true; it is not "
            ":polarity -. Preserve role direction, embedded events, topic/domain/"
            "purpose distinctions and repeated-variable coreference. A :name node "
            "with op1/op2/op3 belongs to ONE entity with that many name parts. "
            "Keep :mode imperative and :polite + when present. Do not convert "
            "questions or requests into past-tense assertions. Do not wrap the "
            "target in 'someone asked/said' or print graph variables.\n"
            f"Required name-part counts: {markers['name_sizes']}.\nTARGET\n{template}")
    else:
        text += "\nWrite a natural continuation; no particular semantic graph is required."
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]


def paraphrase_messages(text):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content":
        "Paraphrase the entire fictional account once, with approximately the same "
        "length. Preserve all propositions, participants, role directions and full "
        "names. Preserve questions as questions and requests as requests; never "
        "turn requested or unknown events into established facts. Split or merge "
        "sentences and reorder narration if natural. Output only the paragraph.\n\n" + text}]


def prepare(out, private):
    if (out / "FREEZE.json").exists():
        raise RuntimeError("Existing freeze is immutable")
    symbol_map = json.loads(OLD_MAP.read_text())
    bank = json.loads(BANK.read_text())
    prompts, assignments = [], []
    for i, scenario in enumerate(SCENARIOS):
        # Each split has every template four times; first three docs cover all 24.
        j = i % 12
        names = [f"{FIRST[i]} {LAST[i]}", f"{LAST[i]} {FIRST[i]} Marlow"]
        prompt = (f"A fictional account of {scenario}. Participants: {names[0]} and "
                  f"{names[1]}. Names may be used in full or shortened when the "
                  "required structure calls for fewer name parts. Invent concrete "
                  "details. Natural statements, direct questions and requests are "
                  "all allowed; no uniform tense or narrator framing is imposed.")
        templates = []
        for symbol in "ABCD":
            members = sorted(symbol_map["symbols"][symbol]["templates"])
            templates.extend([members[2 * j % 6], members[(2 * j + 1) % 6]])
        random.Random(SEED + i).shuffle(templates)
        split = "development" if i < 12 else "readout"
        prompts.append({"doc_id": i, "split": split, "domain": i % 3,
                        "prompt": prompt, "prompt_sha256": digest(prompt)})
        assignments.append({"doc_id": i, "split": split, "slots": [
            {"slot": s, "template_id": t, "symbol": symbol_map["template_to_symbol"][str(t)],
             "markers": required_markers(bank[t]["normalized_template_amr"])}
            for s, t in enumerate(templates)]})
    for split in ("development", "readout"):
        counts = Counter(s["template_id"] for a in assignments if a["split"] == split for s in a["slots"])
        assert len(counts) == 24 and set(counts.values()) == {4}
    assert len({s["template_id"] for a in assignments[:3] for s in a["slots"]}) == 24
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic(private / "PROMPTS.json", prompts)
    atomic(private / "ASSIGNMENTS.json", assignments)
    atomic(out / "SYMBOL_MAP.json", symbol_map)
    # Exact historical bytes, rather than semantic reserialization, preserve SHA.
    (out / "SYMBOL_MAP.json").write_bytes(OLD_MAP.read_bytes())
    sys.path.insert(0, str(ROOT / "src/core"))
    from semantic_sequence import equal_composition_sequences
    atomic(out / "PERMUTATIONS.json", ["".join("ABCD"[x] for x in row)
                                     for row in equal_composition_sequences()])
    sources = [Path(__file__), ROOT / "src/core/semantic_sequence.py",
               ROOT / "scripts/test_semantic_sequence_cycle.py",
               out / "PROTOCOL.md", BANK, OLD_MAP]
    freeze = {"schema": "semantic-sequence-cycle/v1", "seed": SEED,
              "authorization": "2026-10-24 user requested all four ordered stages with common candidates and all equal-frequency permutations",
              "current_release": "first_three_development_pairs_only",
              "pilot_doc_ids": [0, 1, 2], "pilot_request_cap": 54, "total_request_cap": 432,
              "generation_parameters": PARAMETERS, "api_url": API_URL,
              "pilot_gate_g1e": {"denominator": 24, "minimum_successes": 20},
              "formal_human_gate": "NOT_EVALUATED; automated checks cannot replace it",
              "python": sys.version, "packages": {p: md.version(p) for p in
                  ("numpy", "requests", "penman", "spacy", "en-core-web-sm")},
              "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
              "private_sha256": {p.name: sha(p) for p in [private / "PROMPTS.json", private / "ASSIGNMENTS.json"]},
              "symbol_map_sha256": sha(out / "SYMBOL_MAP.json"),
              "permutations_sha256": sha(out / "PERMUTATIONS.json")}
    atomic(out / "FREEZE.json", freeze)
    atomic(out / "STATUS.json", {"stage": 0, "status": "PREPARED",
                               "stages": {"write_read": "PENDING", "symbol_recognition": "PENDING",
                                          "blind_localization": "PENDING", "order": "PENDING"}})


def verify(out, private):
    freeze = json.loads((out / "FREEZE.json").read_text())
    for rel, expected in freeze["source_sha256"].items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError("Frozen source changed: " + rel)
    for rel, expected in freeze["private_sha256"].items():
        if sha(private / rel) != expected:
            raise RuntimeError("Frozen private input changed")
    if sha(out / "PERMUTATIONS.json") != freeze["permutations_sha256"]:
        raise RuntimeError("Permutation set changed")
    if sha(out / "SYMBOL_MAP.json") != freeze["symbol_map_sha256"]:
        raise RuntimeError("Dictionary changed")
    return freeze


class Generator:
    def __init__(self, out, private, key):
        self.out, self.private, self.key = out, private, key
        self.freeze = verify(out, private)
        self.calls = private / "calls"
        self.calls.mkdir(exist_ok=True)
        self.mutex, self.stopped = threading.RLock(), threading.Event()
        self.prompts = json.loads((private / "PROMPTS.json").read_text())
        self.assignments = json.loads((private / "ASSIGNMENTS.json").read_text())
        self.bank = json.loads(BANK.read_text())
        if (out / "HALT.json").exists():
            raise RuntimeError("Permanent halt")
        for path in self.calls.glob("*.intent.json"):
            if not path.with_name(path.name.replace(".intent.", ".response.")).exists():
                self.halt("unresolved_dispatch")
                raise RuntimeError("Unresolved dispatch; no retry")
        start = private / "STARTED.json"
        if not start.exists():
            atomic(start, {"epoch": time.time(), "freeze_sha256": sha(out / "FREEZE.json")})
        self.start = json.loads(start.read_text())
        if self.start["freeze_sha256"] != sha(out / "FREEZE.json"):
            raise RuntimeError("Start/freeze mismatch")

    def halt(self, reason):
        self.stopped.set()
        with self.mutex:
            if not (self.out / "HALT.json").exists():
                atomic(self.out / "HALT.json", {"reason": reason, "permanent": True})

    def call(self, call_id, messages):
        body = {**PARAMETERS, "messages": messages}
        body_sha = hashlib.sha256(encode(body)).hexdigest()
        receipt = self.calls / f"{call_id}.response.json"
        with self.mutex:
            if self.stopped.is_set():
                raise RuntimeError("Dispatch halted")
            if receipt.exists():
                record = json.loads(receipt.read_text())
                if not record["complete"] or record["request_sha256"] != body_sha:
                    self.halt("invalid_receipt")
                    raise RuntimeError("Receipt is not resumable")
                return record["text"]
            intent = self.calls / f"{call_id}.intent.json"
            if intent.exists():
                self.halt("unresolved_dispatch")
                raise RuntimeError("At-most-once violation")
            if len(list(self.calls.glob("*.intent.json"))) >= 54:
                raise RuntimeError("Pilot request cap")
            if time.time() - self.start["epoch"] > 86400:
                self.halt("wall_clock_cap")
                raise RuntimeError("Wall clock cap")
            atomic(intent, {"body": body, "request_sha256": body_sha})
        record = {"complete": False, "request_sha256": body_sha, "text": ""}
        try:
            response = requests.post(API_URL, json=body, allow_redirects=False, timeout=(15, 180),
                                     headers={"Authorization": "Bearer " + self.key})
            payload = response.json()
            atomic(self.calls / f"{call_id}.provider.json", payload)
            choices = payload.get("choices", [])
            if response.status_code != 200 or len(choices) != 1:
                raise ValueError("No single response")
            choice = choices[0]
            text = choice.get("message", {}).get("content")
            record.update(complete=isinstance(text, str), text=text if isinstance(text, str) else "",
                          finish_reason=choice.get("finish_reason"), model=payload.get("model"),
                          usage=payload.get("usage", {}))
        except Exception as exc:
            record["failure_class"] = type(exc).__name__
        atomic(receipt, record)
        if not record["complete"]:
            self.halt("provider_or_transport_failure")
            raise RuntimeError("Provider/transport failure; no retry")
        with self.mutex:
            atomic(self.out / "heartbeat.json", {
                "stage": "write_read_generation", "completed_requests": len(list(self.calls.glob("*.response.json"))),
                "released_request_cap": 54, "elapsed_seconds": round(time.time() - self.start["epoch"], 1)})
        return record["text"]

    def pair(self, doc_id):
        prompt, assignment = self.prompts[doc_id], self.assignments[doc_id]
        arms = {}
        for arm in ("positive", "unwatermarked"):
            slots = []
            for slot in assignment["slots"]:
                template = self.bank[slot["template_id"]]["normalized_template_amr"] if arm == "positive" else None
                text = self.call(f"d{doc_id:02d}-{arm}-s{slot['slot']}", carrier_messages(
                    prompt["prompt"], " ".join(slots), slot["slot"], slot["markers"], template))
                slots.append(text.strip())
            text = " ".join(slots)
            paraphrase = self.call(f"d{doc_id:02d}-{arm}-paraphrase", paraphrase_messages(text))
            arms[arm] = {"slots": slots, "text": text, "text_sha256": digest(text),
                         "paraphrase": paraphrase, "paraphrase_sha256": digest(paraphrase)}
        atomic(self.private / "documents" / f"d{doc_id:02d}.json", {
            "doc_id": doc_id, "split": prompt["split"], "domain": prompt["domain"], "arms": arms})

    def run(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(self.pair, i) for i in self.freeze["pilot_doc_ids"]]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        atomic(self.out / "GENERATION_COMPLETE.json", {
            "pilot_pairs": 3, "completed_requests": 54, "positive_carriers": 24,
            "negative_carriers": 24, "paraphrases": 6,
            "semantic_success": "NOT_EVALUATED", "freeze_sha256": sha(self.out / "FREEZE.json")})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare", "generate", "verify"])
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--private", type=Path, default=PRIVATE)
    p.add_argument("--credential-file", type=Path)
    args = p.parse_args()
    os.umask(0o077)
    if args.action == "prepare":
        prepare(args.out, args.private)
    elif args.action == "verify":
        verify(args.out, args.private)
    else:
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key and args.credential_file:
            keys = set(re.findall(r"\bsk-[A-Za-z0-9_-]+", args.credential_file.read_text()))
            if len(keys) != 1:
                raise RuntimeError("Expected one configured credential")
            key = keys.pop()
        if not key:
            raise RuntimeError("Explicit credential configuration required")
        with (args.private / "generation.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            Generator(args.out, args.private, key).run()
    print(json.dumps({"action": args.action, "completed": True}))


if __name__ == "__main__":
    main()
