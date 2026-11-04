"""Frozen, one-candidate generation for the stage1B semantic sequence pilot.

No parser, detector, score-based selection, or automatic API retry is used.
Private request/response records form an at-most-once dispatch ledger.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/semantic_sequence_stage1_run2"
DESIGN = ROOT / "experiments/semantic_sequence_stage1_design1"
BANK = ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_100.json"
SEED = 20261023
API_URL = "https://api.deepseek.com/chat/completions"
PARAMETERS = {
    "model": "deepseek-flash", "temperature": 0.9, "top_p": 1,
    "max_tokens": 1600, "n": 1, "stream": False,
    "presence_penalty": 0, "frequency_penalty": 0,
    "thinking": {"type": "disabled"},
}
SYSTEM = (
    "You write coherent fictional English prose for a controlled research study. "
    "Follow the requested output format. Output only the requested prose, without "
    "a title, a list, quotation marks around the answer, or commentary."
)
LENGTH = (
    "Write exactly one complete, natural English sentence, aiming for 12 to 22 "
    "whitespace-separated words. This is a writing target, not permission to omit "
    "required meaning. End with sentence-final punctuation."
)
# These scenarios and entity lists are fixed before any candidate generation.
SCENARIOS = {
    "fictional_project_minutes": [
        ("Sablehaven", "Oren", "Miska", "a team planning a foldable rain shelter for a small ferry stop"),
        ("Brindleport", "Tavi", "Leora", "a team developing a quiet book-return trolley for a lending room"),
        ("Coppermead", "Neris", "Paxon", "a team testing reusable labels for a community seed exchange"),
        ("Violetquay", "Iven", "Sula", "a team arranging a portable stage for a courtyard puppet show"),
        ("Alderreach", "Zerin", "Mavie", "a team designing a hand-cranked display for a local craft exhibit"),
        ("Cindervale", "Elvar", "Rinelle", "a team preparing a floating planter for a sheltered ornamental pond"),
        ("Morrowbay", "Jessa", "Calven", "a team prototyping a padded carrier for fragile ceramic tiles"),
        ("Larchwell", "Veda", "Torin", "a team reorganizing a mobile repair cart for broken umbrellas"),
    ],
    "fictional_daily_events": [
        ("Amberwick", "Pella", "Nolan", "neighbors preparing an evening lantern walk after a windy afternoon"),
        ("Birchmere", "Arlen", "Fenna", "friends sorting forgotten costumes in an attic before a neighborhood play"),
        ("Dunebrook", "Kira", "Oswin", "housemates rescuing a picnic when their garden table becomes unstable"),
        ("Elmharbor", "Darin", "Ysola", "siblings searching for a missing recipe card while cooking a small supper"),
        ("Foxhollow", "Lumi", "Brevan", "neighbors moving a collection of potted herbs before painting a balcony"),
        ("Glimmerfen", "Cora", "Ravel", "friends preparing handmade kites for an afternoon at an empty meadow"),
        ("Hazelstrand", "Aven", "Selki", "housemates repairing a squeaking shelf before guests arrive for a game"),
        ("Juniperfold", "Merrit", "Dova", "siblings returning borrowed blankets after an overnight backyard camp"),
    ],
    "fictional_science_scenarios": [
        ("Kelpshire", "Enna", "Talric", "a fictional classroom comparing paper boats in a shallow tray of moving water"),
        ("Lunaford", "Zola", "Herin", "a fictional workshop exploring shadows with cardboard shapes and a lamp"),
        ("Mossridge", "Petra", "Finlo", "a fictional club comparing condensation on covered and uncovered clear jars"),
        ("Nacrepoint", "Yara", "Solven", "a fictional class observing how fabric strips respond to a small electric fan"),
        ("Opalmarsh", "Bexley", "Ardis", "a fictional workshop sorting pebbles with simple sieves and recording the groups"),
        ("Pinecrestle", "Lenna", "Corvin", "a fictional club using ribbons to illustrate paths of warm air above a radiator"),
        ("Quartzgrove", "Safra", "Emlin", "a fictional classroom comparing sound vibrations using stretched rubber bands"),
        ("Reedwharf", "Nessa", "Ulric", "a fictional group exploring color mixing with transparent sheets and a white screen"),
    ],
}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return digest_bytes(Path(path).read_bytes())


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_json(path, value):
    atomic_bytes(path, encoded(value))


def jsonl(path, records):
    atomic_bytes(path, b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
        for row in records))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def build_manifests(symbol_map):
    prompts = []
    assignments = []
    half_indices = Counter()
    for domain_index, (domain, scenarios) in enumerate(SCENARIOS.items()):
        order = list(range(8))
        random.Random(SEED + domain_index).shuffle(order)
        development = set(order[:4])
        for local_index, (place, first, second, scenario) in enumerate(scenarios):
            doc_id = len(prompts)
            split = "development" if local_index in development else "readout"
            half_j = half_indices[split]
            half_indices[split] += 1
            prompt = (
                f"Write a short fictional account about {scenario}. "
                f"The setting is {place}; the participants are {first} and {second}. "
                "Use those names when named entities are needed; do not introduce "
                "other proper names. Use clear, matter-of-fact prose in the past tense, "
                "with concrete actions and a coherent progression. Invent details as "
                "needed; this is fiction, not a report of real events."
            )
            prompts.append({
                "doc_id": doc_id, "pair_id": f"pair-{doc_id:02d}", "split": split,
                "domain": domain, "half_index": half_j, "entities": [place, first, second],
                "prompt": prompt, "prompt_sha256": digest_bytes(prompt.encode()),
            })
            templates = []
            for symbol in "ABCD":
                members = sorted(symbol_map["symbols"][symbol]["templates"])
                templates.extend([members[(2 * half_j) % 6], members[(2 * half_j + 1) % 6]])
            random.Random(SEED + doc_id).shuffle(templates)
            assignments.append({
                "doc_id": doc_id, "split": split, "half_index": half_j,
                "shuffle_seed": SEED + doc_id,
                "slots": [{"slot": i, "intended_template": template,
                           "intended_state": symbol_map["template_to_symbol"][str(template)]}
                          for i, template in enumerate(templates)],
            })
    return prompts, assignments


def carrier_messages(prompt, prior, slot, template=None):
    user = (
        f"SCENARIO\n{prompt}\n\n"
        f"ALREADY WRITTEN IN THIS ACCOUNT\n{prior or '(No preceding text.)'}\n\n"
        f"Write sentence {slot + 1} of an eight-sentence account. Continue from the "
        f"already written text without repeating it. {LENGTH}"
    )
    if template is not None:
        user += (
            "\n\nThis sentence must realize the following abstract AMR graph. "
            "V is a predicate placeholder, N a noun/concept, NE a named entity, "
            "and X another unspecified concept. Replace placeholders with natural "
            "words appropriate to the scenario. Preserve the graph's role relations, "
            "nesting, modifiers, and repeated-node references. Variable names are "
            "not words to emit. Express the structure naturally; output no AMR.\n"
            f"TARGET AMR\n{template}"
        )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def paraphrase_messages(text):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": (
            "Paraphrase the following entire fictional account once, as one natural "
            "English paragraph of approximately the same length. Preserve all "
            "propositions, participants, and their roles; do not add or remove events. "
            "Use different wording and sentence constructions. You may split or merge "
            "sentences and reorder narration where natural without changing event "
            "meaning. Output only the rewritten paragraph.\n\nSOURCE\n" + text)},
    ]


def surface_audit(text):
    chunks = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    return {
        "whitespace_words": len(text.split()), "nonempty": bool(text.strip()),
        "target_length_met": 12 <= len(text.split()) <= 22,
        "single_sentence_heuristic": len(chunks) == 1 and "\n" not in text.strip(),
        "terminal_punctuation": bool(re.search(r"[.!?][\"')\]]*\s*$", text)),
        "placeholder_literal": bool(re.search(r"\b(?:NE|N|X|V)\b", text)),
        "note": "Surface diagnostics only; retained regardless; not a semantic judgment.",
    }


def prepare(run=RUN):
    run = Path(run)
    if (run / "FREEZE_1B.json").exists():
        raise RuntimeError("1B is already frozen; do not overwrite")
    a_result = json.loads((run / "STAGE1A_RESULTS.json").read_text())
    if a_result["status"] != "GO_TO_STAGE1B":
        raise RuntimeError("Stage1A did not pass")
    map_path = run / "SYMBOL_MAP.json"
    symbol_map = json.loads(map_path.read_text())
    a_freeze = json.loads((run / "FREEZE_1A.json").read_text())
    if sha(map_path) != a_freeze["symbol_map_sha256"]:
        raise RuntimeError("Symbol map SHA changed")
    if sha(DESIGN / "PROTOCOL.md") != a_freeze["protocol_sha256"]:
        raise RuntimeError("Protocol SHA changed")
    prompts, assignments = build_manifests(symbol_map)
    entities = [e for p in prompts for e in p["entities"]]
    assert len(set(entities)) == len(entities)
    for split in ("development", "readout"):
        rows = [a for a in assignments if a["split"] == split]
        counts = Counter(s["intended_template"] for a in rows for s in a["slots"])
        assert len(rows) == 12 and set(counts.values()) == {4} and len(counts) == 24
    private = run / "private"
    private.mkdir(mode=0o700, exist_ok=True)
    jsonl(private / "PROMPTS.jsonl", prompts)
    jsonl(private / "ASSIGNMENTS.jsonl", assignments)
    paths = [map_path, run / "STAGE1A_RESULTS.json", run / "FREEZE_1A.json",
             DESIGN / "PROTOCOL.md", BANK, Path(__file__),
             private / "PROMPTS.jsonl", private / "ASSIGNMENTS.jsonl"]
    for part in ("dataset5", "dataset5_paired_neg"):
        paths.append(ROOT / f"experiments/swan_multiview_development1/{part}/documents.json")
    freeze = {
        "schema": "semantic-sequence-stage1b-freeze/v1", "frozen_at": now(),
        "authorization": "User explicitly requested 24 positive plus 24 paired unwatermarked new texts.",
        "generation_parameters": PARAMETERS, "api_url": API_URL,
        "provider_revision": "Alias only; immutable snapshot not exposed. Actual response model/fingerprint recorded per call.",
        "python": sys.version, "requests_version": requests.__version__,
        "seed": SEED, "pairs": 24, "carrier_requests": 384, "paraphrase_requests": 48,
        "completed_request_cap": 432, "dispatch_cap": 432, "automatic_retries": 0,
        "pair_workers": 4, "timeout_connect_seconds": 15, "timeout_read_seconds": 180,
        "wall_clock_cap_seconds": 86400, "gpu_cap_hours": 12,
        "system_prompt": SYSTEM, "carrier_length_instruction": LENGTH,
        "template_indexing": "zero based; bank[index].normalized_template_amr",
        "chain_contract": "Each arm conditions only on its own retained prior completions.",
        "freeze_scope": "Generation only. W/E1/C model revisions and runtime must pass separate pre-inference readiness.",
        "implementation_refinements": [
            "12-22 whitespace words is a prompt target; failures are retained, not screened.",
            "No few-shot source sentences; only abstract target graph is supplied to positives.",
            "Conservative at-most-once dispatch; uncertain dispatch completion permanently halts generation.",
            "Provider requests use one identical parameter configuration, including paraphrases.",
            "Merge changes terminal .!? to a comma at the four paired boundaries; split adds a period at floor(n/2).",
            "Insert picks two distinct paired-negative slots and two distinct original-document gaps.",
        ],
        "frozen_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths},
    }
    atomic_json(run / "FREEZE_1B.json", freeze)
    atomic_json(run / "heartbeat.json", {"status": "FROZEN_NOT_STARTED", "updated_at": now(),
                                       "completed_requests": 0, "request_cap": 432})
    print("Frozen 24 paired prompts, 192 positive assignments, cap 432.")


class PermanentHalt(RuntimeError):
    pass


class Generator:
    def __init__(self, run, key):
        self.run = Path(run)
        self.private = self.run / "private"
        self.calls = self.private / "calls"
        self.calls.mkdir(exist_ok=True, mode=0o700)
        self.key = key
        self.mutex = threading.RLock()
        self.stop = threading.Event()
        self.freeze = json.loads((self.run / "FREEZE_1B.json").read_text())
        for rel, expected in self.freeze["frozen_sha256"].items():
            if sha(ROOT / rel) != expected:
                raise PermanentHalt("Frozen input mismatch: " + rel)
        if (self.run / "GENERATION_HALT.json").exists():
            raise PermanentHalt("Permanent halt already exists; no dispatch permitted")
        self.prompts = read_jsonl(self.private / "PROMPTS.jsonl")
        self.assignments = read_jsonl(self.private / "ASSIGNMENTS.jsonl")
        self.bank = json.loads(BANK.read_text())
        started = self.run / "STARTED_1B.json"
        if not started.exists():
            atomic_json(started, {"started_at": now(), "started_epoch": time.time(),
                                 "freeze_sha256": sha(self.run / "FREEZE_1B.json")})
        self.started = json.loads(started.read_text())
        if self.started["freeze_sha256"] != sha(self.run / "FREEZE_1B.json"):
            raise PermanentHalt("Freeze changed after start")
        # A crash after dispatch but before receipt cannot be safely retried.
        for intent in self.calls.glob("*.intent.json"):
            prefix = intent.name.removesuffix(".intent.json")
            if not (self.calls / f"{prefix}.response.json").exists():
                self.halt("unresolved_dispatch", prefix)
                raise PermanentHalt("Unresolved dispatch; no automatic retry")

    def halt(self, reason, call_id):
        with self.mutex:
            self.stop.set()
            path = self.run / "GENERATION_HALT.json"
            if not path.exists():
                atomic_json(path, {"halted_at": now(), "reason": reason, "call_id": call_id,
                                   "permanent_for_manifest": True, "retry_allowed": False})
            self.heartbeat("HALTED")

    def heartbeat(self, status="RUNNING"):
        with self.mutex:
            intents = list(self.calls.glob("*.intent.json"))
            responses = list(self.calls.glob("*.response.json"))
            values = [json.loads(p.read_text()) for p in responses]
            complete = [v for v in values if v["completion_received"]]
            atomic_json(self.run / "heartbeat.json", {
                "status": "HALTED" if self.stop.is_set() else status, "updated_at": now(),
                "started_at": self.started["started_at"], "dispatched_requests": len(intents),
                "completed_requests": len(complete), "response_records": len(responses),
                "completed_carriers": sum(v["kind"] == "carrier" for v in complete),
                "completed_paraphrases": sum(v["kind"] == "paraphrase" for v in complete),
                "request_cap": 432, "pair_workers": 4,
                "wall_elapsed_seconds": round(time.time() - self.started["started_epoch"], 2),
                "provider_response_models": sorted({str(v.get("response_model")) for v in complete}),
                "total_tokens_reported": sum(v.get("usage", {}).get("total_tokens", 0) for v in complete),
                "semantic_labels": "PENDING_INDEPENDENT_DUAL_HUMAN_REVIEW",
            })

    def call(self, call_id, messages, kind):
        intent_path = self.calls / f"{call_id}.intent.json"
        response_path = self.calls / f"{call_id}.response.json"
        body = dict(PARAMETERS, messages=messages)
        request_sha = digest_bytes(encoded(body))
        with self.mutex:
            if self.stop.is_set():
                raise PermanentHalt("Dispatch stopped")
            if response_path.exists():
                result = json.loads(response_path.read_text())
                if result["request_sha256"] != request_sha or not result["completion_received"]:
                    self.halt("response_request_mismatch_or_failure", call_id)
                    raise PermanentHalt("Stored response is not resumable")
                return result
            if intent_path.exists():
                self.halt("unresolved_dispatch", call_id)
                raise PermanentHalt("At-most-once guard")
            if len(list(self.calls.glob("*.intent.json"))) >= 432:
                self.halt("dispatch_cap", call_id)
                raise PermanentHalt("Dispatch cap")
            if time.time() - self.started["started_epoch"] > 86400:
                self.halt("wall_clock_cap", call_id)
                raise PermanentHalt("Wall clock cap")
            atomic_json(intent_path, {
                "call_id": call_id, "kind": kind, "dispatched_at": now(),
                "request_sha256": request_sha, "body": body,
                "authorization_header_stored": False,
            })
        t0 = time.monotonic()
        result = {"call_id": call_id, "kind": kind, "request_sha256": request_sha,
                  "completion_received": False, "text": "", "received_at": None}
        try:
            response = requests.post(
                API_URL, json=body, timeout=(15, 180), allow_redirects=False,
                headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
            atomic_bytes(self.calls / f"{call_id}.provider.json", response.content)
            result["http_status"] = response.status_code
            result["raw_sha256"] = digest_bytes(response.content)
            payload = response.json()
            choices = payload.get("choices") if isinstance(payload, dict) else None
            if response.status_code != 200 or not choices or len(choices) != 1:
                raise PermanentHalt("Provider returned no single completion")
            choice = choices[0]
            content = choice.get("message", {}).get("content")
            result.update({
                "completion_received": True, "text": content if isinstance(content, str) else "",
                "content_is_string": isinstance(content, str),
                "finish_reason": choice.get("finish_reason"),
                "response_id": payload.get("id"), "response_model": payload.get("model"),
                "system_fingerprint": payload.get("system_fingerprint"),
                "usage": payload.get("usage", {}),
            })
            if kind == "carrier":
                result["surface_audit"] = surface_audit(result["text"])
        except Exception as exc:
            # Do not emit exception strings or headers: they may contain private data.
            result["failure_class"] = type(exc).__name__
        result["elapsed_seconds"] = round(time.monotonic() - t0, 4)
        result["received_at"] = now()
        atomic_json(response_path, result)
        if not result["completion_received"]:
            self.halt("provider_or_transport_failure_no_retry", call_id)
            raise PermanentHalt("Provider/transport failure; see private ledger")
        self.heartbeat()
        return result

    def run_pair(self, doc_id):
        prompt = self.prompts[doc_id]
        assignment = self.assignments[doc_id]
        assert prompt["doc_id"] == assignment["doc_id"] == doc_id
        arms = {}
        # Each paired chain receives only its own prior text.
        for arm in ("positive", "unwatermarked"):
            slots = []
            for slot in range(8):
                prior = " ".join(row["text"].strip() for row in slots)
                target = assignment["slots"][slot]["intended_template"] if arm == "positive" else None
                template = self.bank[target]["normalized_template_amr"] if target is not None else None
                result = self.call(f"d{doc_id:02d}-{arm}-s{slot}", carrier_messages(
                    prompt["prompt"], prior, slot, template), "carrier")
                slots.append({"slot": slot, "call_id": result["call_id"], "text": result["text"],
                              "finish_reason": result["finish_reason"],
                              "surface_audit": result["surface_audit"]})
            text = " ".join(row["text"].strip() for row in slots)
            para = self.call(f"d{doc_id:02d}-{arm}-paraphrase",
                             paraphrase_messages(text), "paraphrase")
            arms[arm] = {"slots": slots, "text": text, "text_sha256": digest_bytes(text.encode()),
                         "paraphrase": para["text"], "paraphrase_call_id": para["call_id"],
                         "paraphrase_sha256": digest_bytes(para["text"].encode())}
        atomic_json(self.private / "documents" / f"d{doc_id:02d}.json",
                    {"doc_id": doc_id, "split": prompt["split"], "domain": prompt["domain"],
                     "arms": arms, "planned_slots_per_arm": 8})
        return doc_id

    def run_all(self):
        def keep_alive():
            while not finished.wait(10):
                self.heartbeat()
        finished = threading.Event()
        thread = threading.Thread(target=keep_alive, daemon=True)
        thread.start()
        self.heartbeat()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                pending = [pool.submit(self.run_pair, p["doc_id"]) for p in self.prompts]
                for future in concurrent.futures.as_completed(pending):
                    try:
                        future.result()
                    except Exception as exc:
                        self.halt(type(exc).__name__, "pair-worker")
                        for other in pending:
                            other.cancel()
                        raise
            self.heartbeat("GENERATION_COMPLETE")
            atomic_json(self.run / "GENERATION_COMPLETE.json", {
                "completed_at": now(), "document_pairs": 24, "original_documents": 48,
                "carrier_completions": 384, "paraphrase_completions": 48,
                "freeze_sha256": sha(self.run / "FREEZE_1B.json"),
                "validation_status": "NOT_EVALUATED",
            })
        finally:
            finished.set()
            thread.join(timeout=2)


def load_key():
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        # Existing local credential, never copied into artifacts or log messages.
        text = (ROOT / ".secrets/legacy_api.txt").read_text()
        matches = set(re.findall(r"\bsk-[A-Za-z0-9_-]+", text))
        if len(matches) != 1:
            raise RuntimeError("Expected exactly one configured generation credential")
        key = matches.pop()
    return key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "probe", "run"])
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args()
    os.umask(0o077)
    args.run.mkdir(parents=True, exist_ok=True)
    if args.action == "prepare":
        prepare(args.run)
        return
    with (args.run / "private" / "generation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        generator = Generator(args.run, load_key())
        if args.action == "probe":
            prompt = generator.prompts[0]
            target = generator.assignments[0]["slots"][0]["intended_template"]
            template = generator.bank[target]["normalized_template_amr"]
            result = generator.call("d00-positive-s0", carrier_messages(
                prompt["prompt"], "", 0, template), "carrier")
            print(json.dumps({k: result[k] for k in (
                "call_id", "completion_received", "finish_reason", "surface_audit")}))
            generator.heartbeat("FIRST_COMPLETION_VERIFIED_PENDING_BACKGROUND_START")
        else:
            generator.run_all()
            print("Generation complete; 24 paired documents and 48 paraphrases.")


if __name__ == "__main__":
    main()
