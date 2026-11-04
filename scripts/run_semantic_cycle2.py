"""Cycle2 generation/coordinator. Private receipts never enter the workspace."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import shlex
import shutil
import sys
import time

import penman
from penman.models.amr import model as AMR_MODEL
import requests

from run_semantic_sequence_cycle import (
    ROOT, BANK, OLD_MAP, PARAMETERS, API_URL, SYSTEM, atomic, digest, sha,
    required_markers, surface_contract, paraphrase_messages,
)
sys.path.insert(0, str(ROOT / "src/core"))
from remote_execution import Remote, RemoteUnavailable

OUT = ROOT / "experiments/semantic_sequence_cycle2/execution5"
PRIVATE = Path.home() / ".cache/amr/semantic_sequence_cycle2/execution5"
SEED = 20261024
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SCENARIOS = [
    "packing a miniature shadow theatre", "assembling a rain gauge for a garden",
    "arranging wooden chimes in a reading room", "planning a cloth lantern display",
    "testing clay stamps for a community mural", "transporting bamboo frames to a park",
    "organizing a hand-bound poetry exchange", "repairing a wheeled embroidery stand",
    "comparing glass marbles on cork tracks", "cataloguing pressed leaves for an exhibit",
    "designing an adjustable book cradle", "preparing painted tiles for a picnic shelter",
    "restoring a folding chess table", "testing paper sails on a classroom fan",
    "moving wicker baskets to a craft stall", "organizing a reusable ribbon workshop",
    "planning a tabletop sundial demonstration", "examining dyed wool under colored glass",
    "building a removable backdrop for puppets", "sorting illustrated postcards for a display",
    "testing felt pads under pottery bowls", "packing woven screens for an outdoor reading",
    "preparing a rotating holder for paint brushes", "comparing copper bells in a rehearsal room",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def status(phase, **extra):
    value = {"phase": phase, "updated_at": now(), "formal_G1_G2_G3": "NOT_EVALUATED",
             "G4": "NOT_RUN", "reference_type": "constructed_or_model_assisted", **extra}
    atomic(OUT / "STATUS.json", value)


def entity_contract(template, doc_id, slot_id):
    graph = penman.decode(template, model=AMR_MODEL)
    name_nodes = sorted(t for s, r, t in graph.triples if r == ":name")
    words = ["Iris", "Nolan", "Priya", "Felix", "Hazel", "Owen"]
    lines = []
    for j, node in enumerate(name_nodes):
        size = sum(s == node and bool(re.fullmatch(r":op\d+", r)) for s, r, t in graph.triples)
        parts = [words[(doc_id + slot_id + j) % len(words)],
                 f"Fairbrook{chr(65 + doc_id)}", "Linden"]
        name = " ".join(parts[:size])
        owner = next(s for s, r, t in graph.triples if r == ":name" and t == node)
        lines.append(f"Entity {owner}, name node {node}: exactly {size} name parts, '{name}'. "
                     "This is one person; use the full name when naming this entity.")
    return "\n".join(lines) or "No named entity is required."


def messages(prompt, prior, slot, positive):
    body = (f"SCENARIO\n{prompt}\nPRECEDING TEXT\n{prior or '(none)'}\n"
            f"Write sentence {slot['slot'] + 1} of eight. {surface_contract(slot['markers'])}\n"
            f"Available participants and full-name constraints:\n{slot['entities']}\n")
    if positive:
        body += (
            "Realize TARGET exactly as meaning in natural English. V is a predicate, "
            "N a concept, NE a named entity, ADJ an adjective. Each X is amr-unknown, "
            "an unknown answer, not an unnamed ordinary object. Use its exact incoming "
            "role to decide what the question asks. :polarity X is a yes/no question. "
            "A question about an embedded action must remain about that action; do not "
            "replace it with a question about the main action. Preserve ARG role direction, "
            "topic/domain/purpose and nested predicates. Repeated variables refer to the "
            "same entity. A name with multiple ops belongs to one person. Use all name "
            "parts assigned above. Keep imperative and explicit politeness if specified. "
            "Do not insert 'someone asked/said' wrappers or name graph variables.\n"
            f"Unknown-role contract: {slot['markers']['unknown_roles']}.\nTARGET\n{slot['target']}")
    else:
        body += "Write a natural continuation. No semantic graph is required."
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]


def prepare():
    if (OUT / "FREEZE.json").exists():
        raise RuntimeError("Immutable freeze already exists")
    PRIVATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    bank, mapping = json.loads(BANK.read_text()), json.loads(OLD_MAP.read_text())
    prompts, assignments = [], []
    for i, scenario in enumerate(SCENARIOS):
        ids = []
        for symbol in "ABCD":
            members = sorted(mapping["symbols"][symbol]["templates"])
            ids += [members[2 * (i % 12) % 6], members[(2 * (i % 12) + 1) % 6]]
        random.Random(SEED + 1000 + i).shuffle(ids)
        split = "development" if i < 12 else "readout"
        prompt = f"A natural fictional account of {scenario}. Use varied concrete details, questions and requests."
        prompts.append({"doc_id": i, "split": split, "prompt": prompt})
        slots = []
        for s, t in enumerate(ids):
            target = bank[t]["normalized_template_amr"]
            slots.append({"slot": s, "template_id": t,
                          "symbol": mapping["template_to_symbol"][str(t)], "target": target,
                          "markers": required_markers(target), "entities": entity_contract(target, i, s)})
        assignments.append({"doc_id": i, "split": split, "slots": slots})
    for split in ("development", "readout"):
        c = Counter(s["template_id"] for d in assignments if d["split"] == split for s in d["slots"])
        assert len(c) == 24 and set(c.values()) == {4}
    assert len({s["template_id"] for d in assignments[:3] for s in d["slots"]}) == 24
    atomic(PRIVATE / "PROMPTS.json", prompts)
    atomic(PRIVATE / "ASSIGNMENTS.json", assignments)
    OUT.mkdir(exist_ok=True)
    shutil.copyfile(OLD_MAP, OUT / "SYMBOL_MAP.json")
    sys.path.insert(0, str(ROOT / "src/core"))
    from semantic_sequence import equal_composition_sequences
    atomic(OUT / "PERMUTATIONS.json", equal_composition_sequences().tolist())
    sources = [Path(__file__), ROOT / "scripts/semantic_cycle2_gpu.py", ROOT / "src/core/remote_execution.py",
               ROOT / "scripts/run_multiscale_transfer.py", ROOT / "scripts/run_semantic_sequence_cycle.py",
               ROOT / "src/core/semantic_sequence.py", ROOT / "src/core/semantic_cycle2.py",
               ROOT / "src/core/parser_recovery_audit.py", ROOT / "src/core/bank_scorer.py",
               ROOT / "third_party/SWAN_official_2ca0c207_clean/utils/amr_utils.py"]
    runtime = ROOT / "third_party/swan_s2match_runtime_711a231d"
    sources += sorted(runtime.glob("*.py"))
    for source in sources:
        target = PRIVATE / "code" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    ids = sorted({s["template_id"] for d in assignments for s in d["slots"]})
    atomic(PRIVATE / "BANK.json", [bank[t] for t in ids])
    old_verbs = ROOT / "experiments/semantic_sequence_stage1_run2/alignment_review1/private/propbank_verbs.json"
    shutil.copyfile(old_verbs, PRIVATE / "propbank_verbs.json")
    parser = json.loads((ROOT / "experiments/semantic_sequence_cycle1/PARSER_FREEZE.json").read_text())
    # The remote asset inventory is prepared and verified before generation.
    assets = json.loads((OUT / "ASSET_MANIFEST.json").read_text())
    frozen = {"schema": "semantic-cycle2/v1", "date": now(), "seed": SEED,
              "protocol_sha256": sha(OUT / "PROTOCOL.md"),
              "sources": {str(p.relative_to(ROOT)): sha(p) for p in sources},
              "private_sha256": {p: sha(PRIVATE / p) for p in
                                  ["PROMPTS.json", "ASSIGNMENTS.json", "BANK.json", "propbank_verbs.json"]},
              "symbol_map_sha256": sha(OUT / "SYMBOL_MAP.json"),
              "permutations_sha256": sha(OUT / "PERMUTATIONS.json"),
              "generation_parameters": PARAMETERS, "parser": {"assets": parser["parser_assets"]},
              "assets": assets, "pilot_doc_ids": [0, 1, 2], "G1e_minimum": 20,
              "request_caps": {"pilot": 54, "generation": 432, "annotation": 24, "all": 456},
              "temperatures": [.05, .1, .2, .5, 1.0], "retain_thresholds": [.25, .4, .55, .7],
              "gpu_budget_seconds": 43200, "wall_budget_seconds": 86400,
              "safety_factor": 1.5, "human_gates": "NOT_EVALUATED",
              "failed_G1e_diagnostic": "fixed_first_three_pairs_only"}
    prior_out = OUT.parent / "execution4"
    prior = prior_out / "FREEZE.json"
    previous_calls = PRIVATE.parent / "execution4/calls"
    if prior.exists():
        frozen["execution_revision"] = {
            "prior_freeze_sha256": sha(prior), "prior_permanent_halt_sha256": sha(prior_out / "HALT.json"),
            "change": "retry read-only SSH transport failures; reconcile launches; complete sealed post-evaluation",
            "resolved_receipts_inherited_sha256": {p.name: sha(p) for p in sorted(previous_calls.glob("*.response.json"))},
            "inherited_G1e_sha256": sha(prior_out / "G1E.json"),
            "inherited_cuda_precheck_sha256": sha(prior_out / "CUDA_PRECHECK.json"),
            "inherited_prediction_seal_sha256": sha(prior_out / "PREDICTION_SEAL.json"),
            "new_generation_for_repair": False, "new_identity_not_resume": True}
        inheritance = json.loads((OUT / "INHERITANCE.json").read_text())
        frozen["prediction_origin"] = inheritance
        frozen["inherited_seal_sha256"] = sha(OUT / "PREDICTION_SEAL.json")
        frozen["inheritance_sha256"] = sha(OUT / "INHERITANCE.json")
        frozen["completion_only"] = True
        frozen["released_ids"] = [0, 1, 2]
        frozen["released_request_cap"] = 57
        frozen["remote_retry"] = {"attempts": 6, "delays_seconds": [1, 2, 4, 8, 15],
                                  "retry_only": ["read", "immutable_upload", "idempotent_write"],
                                  "exhaustion": "pause_exit75_then_supervisor_restart",
                                  "provider_requests": "at_most_once_no_retry"}
    atomic(OUT / "FREEZE.json", frozen)
    status("FROZEN_NOT_STARTED", freeze_sha256=sha(OUT / "FREEZE.json"))


def verify():
    frozen = json.loads((OUT / "FREEZE.json").read_text())
    if sha(OUT / "PROTOCOL.md") != frozen["protocol_sha256"]:
        raise RuntimeError("Protocol changed")
    for name, expected in frozen["sources"].items():
        if sha(ROOT / name) != expected or sha(PRIVATE / "code" / name) != expected:
            raise RuntimeError("Frozen code changed")
    for name, expected in frozen["private_sha256"].items():
        if sha(PRIVATE / name) != expected:
            raise RuntimeError("Private input changed")
    for name, field in [("SYMBOL_MAP.json", "symbol_map_sha256"), ("PERMUTATIONS.json", "permutations_sha256")]:
        if sha(OUT / name) != frozen[field]:
            raise RuntimeError("Public input changed")
    if frozen.get("completion_only"):
        if sha(OUT / "INHERITANCE.json") != frozen["inheritance_sha256"]:
            raise RuntimeError("Inheritance manifest changed")
        if sha(OUT / "PREDICTION_SEAL.json") != frozen["inherited_seal_sha256"]:
            raise RuntimeError("Inherited prediction seal changed")
        for item in frozen["prediction_origin"]["documents"]:
            if sha(PRIVATE / "documents" / item["name"]) != item["sha256"]:
                raise RuntimeError("Inherited document changed")
        for name, expected in frozen["execution_revision"]["resolved_receipts_inherited_sha256"].items():
            if sha(PRIVATE / "calls" / name) != expected:
                raise RuntimeError("Inherited receipt changed")
    return frozen


class Provider:
    def __init__(self, credential):
        self.freeze = verify()
        keys = set(re.findall(r"\bsk-[A-Za-z0-9_-]+", credential.read_text()))
        if len(keys) != 1:
            raise RuntimeError("Expected one configured key")
        self.key = keys.pop()
        self.calls = PRIVATE / "calls"
        self.calls.mkdir(exist_ok=True)
        for intent in self.calls.glob("*.intent.json"):
            if not intent.with_name(intent.name.replace(".intent.", ".response.")).exists():
                raise RuntimeError("Uncertain dispatch; permanent halt")
        started = PRIVATE / "STARTED.json"
        if not started.exists():
            atomic(started, {"epoch": time.time(), "freeze_sha256": sha(OUT / "FREEZE.json")})
        self.started = json.loads(started.read_text())
        if self.started["freeze_sha256"] != sha(OUT / "FREEZE.json"):
            raise RuntimeError("Start/freeze mismatch")

    def call(self, cid, messages, cap, json_response=False):
        body = {**PARAMETERS, "messages": messages}
        if json_response:
            body["response_format"] = {"type": "json_object"}
        body_sha = digest(json.dumps(body, sort_keys=True, ensure_ascii=False))
        receipt, intent = self.calls / f"{cid}.response.json", self.calls / f"{cid}.intent.json"
        if receipt.exists():
            value = json.loads(receipt.read_text())
            if not value["complete"] or value["request_sha256"] != body_sha:
                raise RuntimeError("Unusable stored receipt")
            return value["text"].strip() if value["finish_reason"] == "stop" else ""
        if intent.exists():
            raise RuntimeError("Uncertain dispatch; no retry")
        if len(list(self.calls.glob("*.intent.json"))) >= cap:
            raise RuntimeError("Released provider cap")
        if time.time() - self.started["epoch"] > self.freeze["wall_budget_seconds"]:
            raise RuntimeError("Wall budget exceeded")
        atomic(intent, {"request_sha256": body_sha, "body": body, "at": now()})
        record = {"request_sha256": body_sha, "complete": False, "text": ""}
        try:
            response = requests.post(API_URL, json=body, timeout=(15, 180), allow_redirects=False,
                                     headers={"Authorization": "Bearer " + self.key})
            payload = response.json()
            atomic(self.calls / f"{cid}.provider.json", payload)
            choices = payload.get("choices", [])
            if response.status_code != 200 or len(choices) != 1:
                raise ValueError("No unique response")
            choice = choices[0]
            text = choice.get("message", {}).get("content")
            record.update(text=text if isinstance(text, str) else "",
                          complete=isinstance(text, str), finish_reason=choice.get("finish_reason"),
                          provider_model=payload.get("model"), usage=payload.get("usage"))
        except Exception as exc:
            record["failure_class"] = type(exc).__name__
        atomic(receipt, record)
        if not record["complete"]:
            raise RuntimeError("Provider failure; no retry")
        status("GENERATION" if not json_response else "REFERENCE_ANNOTATION",
               completed_requests=len(list(self.calls.glob("*.response.json"))), released_request_cap=cap)
        return record["text"].strip() if record["finish_reason"] == "stop" else ""

    def generate(self, ids, cap):
        prompts = json.loads((PRIVATE / "PROMPTS.json").read_text())
        assignments = json.loads((PRIVATE / "ASSIGNMENTS.json").read_text())
        for i in ids:
            path = PRIVATE / "documents" / f"d{i:02d}.json"
            if path.exists():
                continue
            arms = {}
            for arm in ("positive", "unwatermarked"):
                slots = []
                for slot in assignments[i]["slots"]:
                    slots.append(self.call(f"d{i:02d}-{arm}-s{slot['slot']}",
                                           messages(prompts[i]["prompt"], " ".join(slots), slot,
                                                    arm == "positive"), cap))
                text = " ".join(slots)
                para = self.call(f"d{i:02d}-{arm}-paraphrase", paraphrase_messages(text), cap)
                arms[arm] = {"slots": slots, "text": text, "paraphrase": para}
            atomic(path, {"doc_id": i, "split": prompts[i]["split"], "arms": arms})

    def annotate(self, ids, cap):
        for i in ids:
            doc = json.loads((PRIVATE / "documents" / f"d{i:02d}.json").read_text())
            source = [{"slot": j, "text": t} for j, t in enumerate(doc["arms"]["positive"]["slots"])]
            msg = [{"role": "system", "content": "Return only a JSON object."},
                   {"role": "user", "content":
                    "For each source slot, locate the corresponding MAIN EVENT predicate in the "
                    "paraphrased account. Use only text meaning. Return {\"events\": [{\"slot\": 0, "
                    "\"phrase\": \"exact unique substring from paraphrase containing that predicate\", "
                    "\"predicate\": \"exact verb token within phrase\", \"status\": \"located\"}]}. "
                    "List every slot 0..7 once. If absent, ambiguous, or changed use status "
                    "\"unknown\", phrase and predicate empty. A phrase must occur exactly once. "
                    "Do not invent a predicate for a missing event.\nSOURCE\n" + json.dumps(source) +
                    "\nPARAPHRASE\n" + doc["arms"]["positive"]["paraphrase"]}]
            value = self.call(f"d{i:02d}-reference", msg, cap, True)
            try:
                parsed = json.loads(value)
            except (ValueError, TypeError):
                parsed = {"events": [], "invalid_json": True}
            atomic(PRIVATE / "references" / f"d{i:02d}.json", parsed)


def remote_module(helper):
    spec = importlib.util.spec_from_file_location("cycle2_access", helper)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def coordinate(args):
    retry_path = OUT / "TRANSPORT_RETRIES.json"
    def on_retry(event):
        events = json.loads(retry_path.read_text()) if retry_path.exists() else []
        events.append({**event, "at": now()})
        atomic(retry_path, events)
        status("REMOTE_RETRY", overall="NO_GO_DIAGNOSTIC_ONLY", **event)
    access = Remote(remote_module(args.remote_helper), on_retry=on_retry)
    remote_private = "/workspace/semantic-sequence-cycle2-execution5-private"
    remote_out = remote_private + "/run"
    frozen = verify()
    provider = Provider(args.credential_file)
    workflow = PRIVATE / "WORKFLOW.json"
    state = json.loads(workflow.read_text()) if workflow.exists() else {
        "phase": "blind_wait" if frozen.get("completion_only") else "pilot_generate",
        "ids": [0, 1, 2], "overall": "NO_GO_DIAGNOSTIC_ONLY"}
    # Persist phase boundaries before a potentially ambiguous external launch.
    def save(phase, **kwargs):
        state.update(phase=phase, **kwargs)
        atomic(workflow, state)
    def upload():
        documents = []
        seen = {}
        inert = []
        historic = set()
        history = Path.home() / ".cache/amr/semantic_sequence_cycle1/documents"
        for path in history.glob("d*.json"):
            d = json.loads(path.read_text())
            for arm in d["arms"].values():
                historic.update(digest(re.sub(r"\s+", " ", t).strip())
                                for t in [arm["text"], arm["paraphrase"]] if t)
        for path in sorted((PRIVATE / "documents").glob("d*.json")):
            d = json.loads(path.read_text())
            if d["split"] != ("development" if d["doc_id"] < 12 else "readout"):
                raise RuntimeError("Document split changed")
            for arm_name, arm in d["arms"].items():
                for s, text in enumerate(arm["slots"]):
                    receipt = json.loads((PRIVATE / "calls" /
                                          f"d{d['doc_id']:02d}-{arm_name}-s{s}.response.json").read_text())
                    expected = receipt["text"].strip() if receipt["finish_reason"] == "stop" else ""
                    if text != expected:
                        raise RuntimeError("Document/receipt text mismatch")
                if arm["text"] != " ".join(arm["slots"]):
                    raise RuntimeError("Carrier composition mismatch")
                for text in [arm["text"], arm["paraphrase"]]:
                    if text:
                        signature = digest(re.sub(r"\s+", " ", text).strip())
                        identity = (d["doc_id"], arm_name)
                        if signature in historic or (signature in seen and seen[signature] != identity):
                            raise RuntimeError("Duplicate or cycle1 text contamination")
                        if signature in seen:
                            inert.append({"doc_id": d["doc_id"], "arm": arm_name, "condition": "paraphrase"})
                        seen[signature] = identity
            documents.append({"doc_id": d["doc_id"], "sha256": sha(path)})
        atomic(OUT / "INPUT_INTEGRITY.json", {"documents": documents,
                                              "cycle1_document_overlap": 0,
                                              "cross_source_duplicate_documents": 0,
                                              "inert_paraphrases": inert,
                                              "receipts_bound": True})
        access.upload([OUT / "FREEZE.json", OUT / "SYMBOL_MAP.json", PRIVATE / "code",
                             PRIVATE / "BANK.json", PRIVATE / "propbank_verbs.json", PRIVATE / "documents"],
                            remote_private, ["run/FREEZE.json", "run/SYMBOL_MAP.json", "run/private/code",
                                             "run/private/BANK.json", "run/private/propbank_verbs.json",
                                             "run/private/documents"])
    def launch(action):
        script = remote_out + "/private/code/scripts/semantic_cycle2_gpu.py"
        command = (f"/workspace/swan-elastic-reconstruct/.venv/bin/python {shlex.quote(script)} "
                   f"{action} --out {shlex.quote(remote_out)}")
        # supervisor metadata/logs are private. No API keys are sent remotely.
        config = ("[unix_http_server]\nfile=" + remote_private + "/supervisor.sock\n"
                  "[supervisord]\nnodaemon=false\nlogfile=" + remote_private +
                  "/supervisord.log\npidfile=" + remote_private + "/supervisord.pid\n"
                  "[rpcinterface:supervisor]\nsupervisor.rpcinterface_factory=supervisor.rpcinterface:make_main_rpcinterface\n"
                  "[supervisorctl]\nserverurl=unix://" + remote_private + "/supervisor.sock\n"
                  "[program:cycle2]\ncommand=/usr/bin/flock -n " + remote_private +
                  "/gpu.lock " + command + "\nautostart=true\nautorestart=false\nstartsecs=1\n"
                  "stdout_logfile=" + remote_private + "/gpu.stdout.log\nstderr_logfile=" +
                  remote_private + "/gpu.stderr.log\n")
        identity = {"action": action, "freeze_sha256": sha(OUT / "FREEZE.json")}
        previous_launch = PRIVATE / "REMOTE_LAUNCH.json"
        if previous_launch.exists():
            prior_launch = json.loads(previous_launch.read_text())
            if prior_launch["action"] == action and prior_launch["freeze_sha256"] != identity["freeze_sha256"]:
                raise RuntimeError("Remote launch identity changed")
        atomic(previous_launch, {**identity, "at": now()})
        local_config = PRIVATE / "remote_supervisor.conf"
        local_config.write_text(config)
        existing = access.read("if test -S " + remote_private + "/supervisor.sock; then "
                              "/usr/bin/supervisorctl -s unix://" + remote_private +
                              "/supervisor.sock status cycle2 || test $? -eq 3; fi", timeout=35)
        if "RUNNING" in existing or "STARTING" in existing:
            return
        access.upload([local_config], remote_private, ["supervisor.conf"])
        # This instance is task-specific; stop its prior exited supervisor only.
        try:
            access.launch("if test -S " + remote_private + "/supervisor.sock; then "
                   "/usr/bin/supervisorctl -c " + remote_private + "/supervisor.conf shutdown; fi\n"
                   "/usr/bin/supervisord -c " + remote_private + "/supervisor.conf", timeout=40)
        except RemoteUnavailable:
            # The command may have started: keep *_launch phase and reconcile on restart.
            status("PAUSED_LAUNCH_RECONCILIATION", overall="NO_GO_DIAGNOSTIC_ONLY", action=action)
            raise
    def pull(name):
        value = json.loads(access.read("cat " + shlex.quote(remote_out + "/" + name), timeout=35))
        if name == "PREDICTION_SEAL.json" and frozen.get("completion_only"):
            # Preserve exact inherited bytes rather than serializing a new seal.
            if value != json.loads((OUT / name).read_text()):
                raise RuntimeError("Remote inherited prediction seal mismatch")
            return value
        atomic(OUT / name, value)
        return value
    def wait(name):
        last = None
        while True:
            code = ("import json,pathlib; p=pathlib.Path(" + repr(remote_out) +
                    "); print(json.dumps({k:json.loads((p/k).read_text()) if (p/k).exists() else None for k in " +
                    repr([name, "HALT.json", "heartbeat.json"]) + "}))")
            values = json.loads(access.read("/workspace/swan-elastic-reconstruct/.venv/bin/python -c " +
                                           shlex.quote(code), timeout=35))
            if values["HALT.json"] is not None:
                atomic(OUT / "REMOTE_HALT.json", values["HALT.json"])
                raise RuntimeError("Remote permanent halt")
            if values[name] is not None:
                return pull(name)
            if values["heartbeat.json"] is not None:
                hb = values["heartbeat.json"]
                atomic(OUT / "heartbeat.json", hb)
                if hb != last:
                    status("CUDA_" + hb["phase"], **{k: v for k, v in hb.items() if k != "phase"})
                    last = hb
            worker = access.read("if test -S " + remote_private + "/supervisor.sock; then "
                                 "/usr/bin/supervisorctl -s unix://" + remote_private +
                                 "/supervisor.sock status cycle2 || test $? -eq 3; fi", timeout=35)
            if any(x in worker for x in ["EXITED", "FATAL", "STOPPED", "BACKOFF"]):
                # The worker may have completed between the first read and status.
                value = json.loads(access.read("cat " + shlex.quote(remote_out + "/" + name), timeout=35))
                atomic(OUT / name, value)
                return value
            if time.time() - provider.started["epoch"] > frozen["wall_budget_seconds"]:
                raise RuntimeError("Coordinator wall budget")
            time.sleep(15)
    if state["phase"] == "pilot_generate":
        provider.generate([0, 1, 2], 54)
        atomic(OUT / "GENERATION_PILOT.json", {"pairs": 3, "requests": 54, "carriers": 24})
        upload()
        access.upload([PRIVATE / "ASSIGNMENTS.json"], remote_private, ["pilot/ASSIGNMENTS.json"])
        save("pilot_launch")
    if state["phase"] == "pilot_launch":
        launch("pilot")
        save("pilot_wait")
    if state["phase"] == "pilot_wait":
        gate = wait("G1E.json")
        pull("CUDA_PRECHECK.json")
        save("expand" if gate["Go"] else "blind_launch", ids=list(range(24)) if gate["Go"] else [0, 1, 2],
             overall="G1e_GO_FORMAL_PENDING" if gate["Go"] else "NO_GO_DIAGNOSTIC_ONLY")
    if state["phase"] == "expand":
        provider.generate(list(range(3, 24)), 432)
        upload()
        save("blind_launch")
    if state["phase"] == "blind_launch":
        # Remove targets before blind worker invocation; scorer cannot read them.
        access.idempotent("rm -f " + remote_private + "/pilot/ASSIGNMENTS.json", timeout=35)
        launch("blind")
        save("blind_wait")
    if state["phase"] == "blind_wait":
        wait("PREDICTION_SEAL.json")
        pull("RESOURCE_PRECHECK.json")
        save("annotate")
    if state["phase"] == "annotate":
        provider.annotate(state["ids"], 456 if len(state["ids"]) == 24 else 57)
        access.upload([PRIVATE / "references", PRIVATE / "ASSIGNMENTS.json"], remote_private,
                            ["run/private/references", "run/private/ASSIGNMENTS.json"])
        save("evaluate_launch")
    if state["phase"] == "evaluate_launch":
        launch("evaluate")
        save("evaluate_wait")
    if state["phase"] == "evaluate_wait":
        result = wait("RESULT.json")
        pull("SELECTION.json")
        status("COMPLETE", overall=state["overall"], pairs=len(state["ids"]), result_available=True)
        save("complete")
    if state["phase"] == "complete":
        status("COMPLETE", overall=state["overall"], pairs=len(state["ids"]), result_available=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "verify", "coordinate"])
    parser.add_argument("--credential-file", type=Path)
    parser.add_argument("--remote-helper", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    if args.action == "prepare":
        prepare()
    elif args.action == "verify":
        verify()
    else:
        with (PRIVATE / "coordinator.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (OUT / "HALT.json").exists():
                raise RuntimeError("Permanent halt")
            try:
                coordinate(args)
            except RemoteUnavailable as exc:
                atomic(OUT / "TRANSPORT_PAUSE.json", {"at": now(), "reason": str(exc),
                                                      "permanent": False, "exit_code": 75})
                status("PAUSED_REMOTE_UNAVAILABLE", overall="NO_GO_DIAGNOSTIC_ONLY")
                raise SystemExit(75)
            except Exception as exc:
                if not (OUT / "HALT.json").exists():
                    atomic(OUT / "HALT.json", {"permanent": True, "reason_class": type(exc).__name__,
                                             "reason": str(exc) if isinstance(exc, RuntimeError) else "execution_failure",
                                             "at": now()})
                status("HALTED", overall="STOP")
                raise SystemExit(1)
    print(json.dumps({"action": args.action, "complete": True}))


if __name__ == "__main__":
    main()
