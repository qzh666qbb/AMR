"""Independent diagnostic coordinator, with private at-most-once receipts."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import threading
import time

import run_semantic_cycle2 as old
from run_semantic_sequence_cycle import ROOT, atomic, sha, digest, PARAMETERS, API_URL
from remote_execution import Remote, RemoteUnavailable

OUT = ROOT / "experiments/semantic_sequence_readout1"
PRIVATE = Path.home() / ".cache/amr/semantic_sequence_readout1"
ORIGIN = ROOT / "experiments/semantic_sequence_cycle2/execution5"
PRIOR = PRIVATE.parent / "semantic_sequence_cycle2/execution5"
REMOTE = "/workspace/semantic-sequence-readout1-private"
GPU = "/workspace/swan-elastic-reconstruct/.venv/bin/python"
IDS = list(range(12, 24))


def status(phase, **values):
    atomic(OUT / "STATUS.json", {"phase": phase, "at": old.now(),
                                "scope": "post-No-Go independent diagnostic",
                                "cycle2": "NO_GO_19_OF_24", "human": "NOT_EVALUATED", **values})


def verify():
    frozen = json.loads((OUT / "FREEZE.json").read_text())
    if sha(OUT / "PROTOCOL.md") != frozen["protocol_sha256"]:
        raise RuntimeError("Frozen protocol changed")
    for name, expected in frozen["sources"].items():
        if sha(PRIVATE / "code" / name) != expected:
            raise RuntimeError("Frozen source changed")
    for name, expected in frozen["private_sha256"].items():
        if sha(PRIVATE / name) != expected:
            raise RuntimeError("Frozen private artifact changed")
    for name, field in [("LOCKED_SELECTION.json", "selection_sha256"),
                        ("PERMUTATIONS.json", "permutations_sha256"),
                        ("SYMBOL_MAP.json", "symbol_map_sha256")]:
        if sha(OUT / name) != frozen[field]:
            raise RuntimeError("Locked public artifact changed")
    return frozen


def prepare(helper):
    if (OUT / "FREEZE.json").exists():
        raise RuntimeError("Immutable freeze already exists")
    PRIVATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name in ("PROMPTS.json", "ASSIGNMENTS.json", "BANK.json", "propbank_verbs.json", "PROTOTYPES.json"):
        shutil.copyfile(PRIOR / name, PRIVATE / name)
    for name in ("SYMBOL_MAP.json", "PERMUTATIONS.json"):
        shutil.copyfile(ORIGIN / name, OUT / name)
    shutil.copyfile(ORIGIN / "SELECTION.json", OUT / "LOCKED_SELECTION.json")
    shutil.copytree(PRIOR / "code", PRIVATE / "code")
    for name in ("run_semantic_readout.py", "semantic_readout_gpu.py"):
        destination = PRIVATE / "code/scripts" / name
        shutil.copyfile(ROOT / "scripts" / name, destination)
    old_freeze = json.loads((ORIGIN / "FREEZE.json").read_text())
    assignments = json.loads((PRIVATE / "ASSIGNMENTS.json").read_text())
    prompts = json.loads((PRIVATE / "PROMPTS.json").read_text())
    if any(prompts[i]["split"] != "readout" or assignments[i]["split"] != "readout" for i in IDS):
        raise RuntimeError("Readout identity mismatch")
    for i in IDS:
        if (PRIOR / "documents" / f"d{i:02d}.json").exists():
            raise RuntimeError("Readout data already generated")
    from collections import Counter
    counts = Counter(s["template_id"] for i in IDS for s in assignments[i]["slots"])
    if len(counts) != 24 or set(counts.values()) != {4}:
        raise RuntimeError("Template composition mismatch")
    sources = {str(p.relative_to(PRIVATE / "code")): sha(p)
               for p in sorted((PRIVATE / "code").rglob("*.py"))}
    frozen = {"schema": "semantic-independent-readout/v1", "seed": 20261024, "at": old.now(),
              "authorization": "User requested remaining feasible human preparation, independent readout, order diagnostics",
              "protocol_sha256": sha(OUT / "PROTOCOL.md"), "sources": sources,
              "private_sha256": {name: sha(PRIVATE / name) for name in
                                ("PROMPTS.json", "ASSIGNMENTS.json", "BANK.json", "propbank_verbs.json", "PROTOTYPES.json")},
              "prototypes_sha256": sha(PRIVATE / "PROTOTYPES.json"),
              "selection_sha256": sha(OUT / "LOCKED_SELECTION.json"),
              "permutations_sha256": sha(OUT / "PERMUTATIONS.json"),
              "symbol_map_sha256": sha(OUT / "SYMBOL_MAP.json"),
              "origin_freeze_sha256": sha(ORIGIN / "FREEZE.json"), "origin_result_sha256": sha(ORIGIN / "RESULT.json"),
              "readout_doc_ids": IDS, "request_caps": {"generation": 216, "all": 228},
              "generation_parameters": PARAMETERS, "parser": old_freeze["parser"], "assets": old_freeze["assets"],
              "gpu_budget_seconds": 43200, "wall_budget_seconds": 86400, "safety_factor": 1.5,
              "formal_G1_G2_G3_G4": "NOT_EVALUATED", "old_cycle2": "NO_GO_19_OF_24"}
    atomic(OUT / "FREEZE.json", frozen)
    access = Remote(old.remote_module(helper))
    command = ("mkdir -p " + REMOTE + "/run/private\ncp -a " +
               "/workspace/semantic-sequence-cycle2-execution5-private/run/private/minilm " +
               REMOTE + "/run/private/\ncp -a " +
               "/workspace/semantic-sequence-cycle2-execution5-private/run/private/spacy_model " +
               REMOTE + "/run/private/")
    access.idempotent(command, timeout=55)
    atomic(OUT / "IDENTITY_PRECHECK.json", {"Go": True, "new_readout_pairs": 12, "previously_generated": 0,
                                           "template_counts": dict(counts), "params_retuned": False,
                                           "old_NoGo_preserved": True})
    status("READY", pairs=12, versions=72, request_cap=228)


class Provider(old.Provider):
    def __init__(self, credential):
        old.OUT, old.PRIVATE, old.verify = OUT, PRIVATE, verify
        super().__init__(credential)
        self.mutex = threading.RLock()

    def call(self, cid, messages, cap, json_response=False):
        import requests
        body = {**PARAMETERS, "messages": messages}
        if json_response:
            body["response_format"] = {"type": "json_object"}
        body_sha = digest(json.dumps(body, sort_keys=True, ensure_ascii=False))
        receipt = self.calls / f"{cid}.response.json"
        intent = self.calls / f"{cid}.intent.json"
        with self.mutex:
            if receipt.exists():
                record = json.loads(receipt.read_text())
                if not record["complete"] or record["request_sha256"] != body_sha:
                    raise RuntimeError("Stored receipt identity/failure")
                return record["text"].strip() if record["finish_reason"] == "stop" else ""
            if intent.exists():
                raise RuntimeError("Uncertain dispatch; no retry")
            if len(list(self.calls.glob("*.intent.json"))) >= cap:
                raise RuntimeError("Provider cap")
            if time.time() - self.started["epoch"] > self.freeze["wall_budget_seconds"]:
                raise RuntimeError("Wall budget")
            atomic(intent, {"request_sha256": body_sha, "body": body, "at": old.now()})
        record = {"request_sha256": body_sha, "complete": False, "text": ""}
        try:
            response = requests.post(API_URL, json=body, timeout=(15, 180), allow_redirects=False,
                                     headers={"Authorization": "Bearer " + self.key})
            payload = response.json()
            atomic(self.calls / f"{cid}.provider.json", payload)
            choices = payload.get("choices", [])
            if response.status_code != 200 or len(choices) != 1:
                raise ValueError("No unique provider response")
            choice = choices[0]
            text = choice.get("message", {}).get("content")
            record.update(text=text if isinstance(text, str) else "", complete=isinstance(text, str),
                          finish_reason=choice.get("finish_reason"), provider_model=payload.get("model"),
                          usage=payload.get("usage"))
        except Exception as exc:
            record["failure_class"] = type(exc).__name__
        atomic(receipt, record)
        if not record["complete"]:
            raise RuntimeError("Provider failure; no retry")
        with self.mutex:
            status("GENERATION" if not json_response else "MODEL_ASSISTED_REFERENCE",
                   completed_requests=len(list(self.calls.glob("*.response.json"))), request_cap=cap)
        return record["text"].strip() if record["finish_reason"] == "stop" else ""


def integrity():
    historic, seen, inert, documents = set(), {}, [], []
    for directory in (PRIVATE.parent / "semantic_sequence_cycle1/documents", PRIOR / "documents"):
        for path in directory.glob("d*.json"):
            for arm in json.loads(path.read_text())["arms"].values():
                historic.update(digest(re.sub(r"\s+", " ", t).strip())
                                for t in (arm["text"], arm["paraphrase"]) if t)
    for i in IDS:
        path = PRIVATE / "documents" / f"d{i:02d}.json"
        doc = json.loads(path.read_text())
        for arm_name, arm in doc["arms"].items():
            for s, text in enumerate(arm["slots"]):
                record = json.loads((PRIVATE / "calls" / f"d{i:02d}-{arm_name}-s{s}.response.json").read_text())
                expected = record["text"].strip() if record["finish_reason"] == "stop" else ""
                if text != expected:
                    raise RuntimeError("Carrier/receipt mismatch")
            for label, text in (("text", arm["text"]), ("paraphrase", arm["paraphrase"])):
                if not text:
                    continue
                signature, identity = digest(re.sub(r"\s+", " ", text).strip()), (i, arm_name)
                if signature in historic or (signature in seen and seen[signature] != identity):
                    raise RuntimeError("Historic or cross-document duplicate")
                if signature in seen:
                    inert.append({"doc_id": i, "arm": arm_name, "condition": label})
                seen[signature] = identity
        documents.append({"doc_id": i, "sha256": sha(path)})
    atomic(OUT / "INPUT_INTEGRITY.json", {"Go": True, "documents": documents,
                                         "historical_overlap": 0, "cross_document_duplicate": 0,
                                         "inert_paraphrases": inert, "receipts_bound": True})


def coordinate(args):
    frozen = verify()
    provider = Provider(args.credential_file)
    access = Remote(old.remote_module(args.remote_helper))
    workflow = PRIVATE / "WORKFLOW.json"
    state = json.loads(workflow.read_text()) if workflow.exists() else {"phase": "generate"}

    def save(phase):
        state["phase"] = phase
        atomic(workflow, state)

    def pull(name):
        value = json.loads(access.read("cat " + shlex.quote(REMOTE + "/run/" + name)))
        atomic(OUT / name, value)
        return value

    def worker_status():
        return access.read("if test -S " + REMOTE + "/supervisor.sock; then "
                           "/usr/bin/supervisorctl -s unix://" + REMOTE +
                           "/supervisor.sock status readout || test $? -eq 3; fi", timeout=35)

    def launch(action):
        existing = worker_status()
        if "RUNNING" in existing or "STARTING" in existing:
            return
        command = GPU + " " + REMOTE + "/run/private/code/scripts/semantic_readout_gpu.py " + action + " --out " + REMOTE + "/run"
        config = ("[unix_http_server]\nfile=" + REMOTE + "/supervisor.sock\n"
                  "[supervisord]\nlogfile=" + REMOTE + "/supervisord.log\npidfile=" + REMOTE + "/supervisord.pid\n"
                  "[rpcinterface:supervisor]\nsupervisor.rpcinterface_factory=supervisor.rpcinterface:make_main_rpcinterface\n"
                  "[supervisorctl]\nserverurl=unix://" + REMOTE + "/supervisor.sock\n"
                  "[program:readout]\ncommand=/usr/bin/flock -n " + REMOTE + "/gpu.lock " + command +
                  "\nautostart=true\nautorestart=false\nstartsecs=1\nstdout_logfile=" + REMOTE +
                  "/gpu.stdout.log\nstderr_logfile=" + REMOTE + "/gpu.stderr.log\n")
        path = PRIVATE / "remote_supervisor.conf"
        path.write_text(config)
        access.upload([path], REMOTE, ["supervisor.conf"])
        atomic(PRIVATE / "REMOTE_LAUNCH.json", {"action": action, "freeze_sha256": sha(OUT / "FREEZE.json"),
                                               "at": old.now(), "reconciled_previous_status": existing.strip()})
        access.launch("if test -S " + REMOTE + "/supervisor.sock; then /usr/bin/supervisorctl -s unix://" +
                      REMOTE + "/supervisor.sock shutdown; fi\n/usr/bin/supervisord -c " + REMOTE + "/supervisor.conf")

    def wait(name):
        while True:
            code = ("import json,pathlib; p=pathlib.Path(" + repr(REMOTE + "/run") +
                    "); print(json.dumps({k:json.loads((p/k).read_text()) if (p/k).exists() else None for k in " +
                    repr([name, "HALT.json", "heartbeat.json"]) + "}))")
            values = json.loads(access.read(GPU + " -c " + shlex.quote(code), timeout=35))
            if values["HALT.json"]:
                atomic(OUT / "REMOTE_HALT.json", values["HALT.json"])
                raise RuntimeError("Remote permanent halt")
            if values[name] is not None:
                return pull(name)
            if values["heartbeat.json"]:
                hb = values["heartbeat.json"]
                atomic(OUT / "heartbeat.json", hb)
                status("CUDA_" + hb["phase"], **{k: v for k, v in hb.items() if k != "phase"})
            worker = worker_status()
            if any(x in worker for x in ("EXITED", "FATAL", "STOPPED", "BACKOFF")):
                return pull(name)
            if time.time() - provider.started["epoch"] > frozen["wall_budget_seconds"]:
                raise RuntimeError("Wall budget")
            time.sleep(15)

    if state["phase"] == "generate":
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda i: provider.generate([i], 216), IDS))
        integrity()
        paths = [OUT / "FREEZE.json", OUT / "SYMBOL_MAP.json", OUT / "PERMUTATIONS.json",
                 OUT / "LOCKED_SELECTION.json", PRIVATE / "code", PRIVATE / "BANK.json",
                 PRIVATE / "propbank_verbs.json", PRIVATE / "PROTOTYPES.json", PRIVATE / "documents"]
        names = ["run/FREEZE.json", "run/SYMBOL_MAP.json", "run/PERMUTATIONS.json",
                 "run/LOCKED_SELECTION.json", "run/private/code", "run/private/BANK.json",
                 "run/private/propbank_verbs.json", "run/private/PROTOTYPES.json", "run/private/documents"]
        access.upload(paths, REMOTE, names)
        save("blind_launch")
    if state["phase"] == "blind_launch":
        launch("blind")
        save("blind_wait")
    if state["phase"] == "blind_wait":
        wait("PREDICTION_SEAL.json")
        for name in ("ORDER_SEAL.json", "CUDA_PRECHECK.json", "RESOURCE_PRECHECK.json"):
            pull(name)
        save("annotate")
    if state["phase"] == "annotate":
        provider.annotate(IDS, 228)
        access.upload([PRIVATE / "references", PRIVATE / "ASSIGNMENTS.json"], REMOTE,
                      ["run/private/references", "run/private/ASSIGNMENTS.json"])
        save("evaluate_launch")
    if state["phase"] == "evaluate_launch":
        launch("evaluate")
        save("evaluate_wait")
    if state["phase"] == "evaluate_wait":
        wait("RESULT.json")
        pull("ORDER_RESULT.json")
        save("complete")
    if state["phase"] == "complete":
        worker = worker_status()
        # RESULT can become visible immediately before the worker exits.
        if "RUNNING" in worker:
            time.sleep(2)
            worker = worker_status()
        access.launch("if test -S " + REMOTE + "/supervisor.sock; then /usr/bin/supervisorctl -s unix://" +
                      REMOTE + "/supervisor.sock shutdown; fi")
        atomic(OUT / "EXECUTION_CLOSEOUT.json", {"completed_requests": len(list(provider.calls.glob("*.response.json"))),
                                               "unresolved_dispatches": 0, "remote_supervisor_shutdown": True,
                                               "worker_final_status": worker.strip(), "at": old.now()})
        status("COMPUTATION_COMPLETE", pairs=12, versions=72, human="NOT_EVALUATED",
               independent_audit="PENDING")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "verify", "coordinate"))
    p.add_argument("--credential-file", type=Path)
    p.add_argument("--remote-helper", type=Path)
    args = p.parse_args()
    os.umask(0o077)
    if args.action == "prepare":
        prepare(args.remote_helper)
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
                atomic(OUT / "TRANSPORT_PAUSE.json", {"permanent": False, "reason": str(exc), "at": old.now()})
                status("PAUSED_REMOTE")
                raise SystemExit(75)
            except Exception as exc:
                atomic(OUT / "HALT.json", {"permanent": True, "reason_class": type(exc).__name__,
                                          "reason": str(exc) if isinstance(exc, RuntimeError) else "execution_failure",
                                          "at": old.now()})
                status("HALTED")
                raise SystemExit(1)


if __name__ == "__main__":
    main()
