"""Frozen pilot successor: replay paid requests, reject invalid candidates, resume."""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import sys

import run_swan_supplement1 as pilot
from run_swan_supplement2 import load_bank
import swan_supplement1_sources as sources
import swan_supplement1_review as review
import swan_final1_budget as budget
from swan_final1_common import digest, freeze, load, now, sha256, single_writer
from swan_supplement1_runtime import IntegrityHalt, content_seed, readiness
from swan_supplement4_runtime import CandidateReject, ResourcePause, runtime

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/swan_count_neutral_supplement4"
PARENTS = [ROOT / f"experiments/swan_count_neutral_supplement{n}" for n in (2, 3)]


class ReplayJournal(budget.Journal):
    """Verify every old request/response and charge it exactly once."""
    def __init__(self, *args, parents=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.inherited = {}
        for parent in PARENTS if parents is None else parents:
            previous = budget.Journal(parent, "pilot", 6400, None, sha256(parent / "FREEZE.json"))
            if previous.uncertain or any(v["terminal"] is None for v in previous.calls.values()):
                raise IntegrityHalt("incomplete_parent_provider_accounting")
            for call_id, call in previous.calls.items():
                if call_id in self.inherited or call_id in self.calls:
                    raise IntegrityHalt("duplicate_lineage_request")
                terminal = call["terminal"]
                if terminal["kind"] != "complete":
                    raise IntegrityHalt("noncomplete_parent_response")
                response = previous._parse(
                    load(previous.folder / "responses" / f"{call_id}.json"),
                    call["reserve"]["reserve"])
                self.inherited[call_id] = (call, response)
    def totals(self):
        result = super().totals()
        inherited = getattr(self, "inherited", {})
        cost = sum(response["actual_nanodollars"] for _, response in inherited.values())
        result["calls"] += len(inherited)
        result["actual_peak_nanodollars"] += cost
        result["exposure_nanodollars"] += cost
        result["inherited_complete_calls"] = len(inherited)
        return result
    def request(self, call_id, user_text, system_text, max_tokens, temperature, top_p):
        body = budget.body_for(user_text, system_text, max_tokens, temperature, top_p)
        if call_id in self.inherited:
            call, response = self.inherited[call_id]
        elif call_id in self.calls:
            call = self.calls[call_id]
            if call["terminal"] is None or call["terminal"]["kind"] != "complete":
                raise IntegrityHalt("uncertain_call_blocks_resume")
            response = self._parse(load(self.folder / "responses" / f"{call_id}.json"),
                                   call["reserve"]["reserve"])
        else:
            return super().request(call_id, user_text, system_text, max_tokens, temperature, top_p)
        if call["reserve"]["request"] != body:
            raise IntegrityHalt("replay_request_differs_from_paid_request")
        return {**response, "call_sha256": call["terminal"]["response_sha256"]}


def bind():
    for module in (pilot, sources, review):
        module.OUT = OUT
    budget.CAPS = {"pilot": (6400, 20_000_000_000)}


def prepare():
    parent = PARENTS[-1]
    # Source copies retain their exact serialized bytes, including history.
    for name in ("source_manifest.json", "prior_identity_snapshot.json", "SOURCE_AUDIT.json",
                 "HUMAN_REVIEW.md", "private/source_screening.json",
                 *(f"private/{phase}_sources.json" for phase, _ in sources.COHORTS)):
        target = OUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256(target) != sha256(parent / name):
                raise IntegrityHalt("source_copy_changed")
        else:
            shutil.copy2(parent / name, target)
    sources.audit()
    dependencies = {
        p: sha256(ROOT / p) for p in pilot.DEPENDENCIES
        if not p.startswith("experiments/")
    }
    dependencies.update({p: sha256(ROOT / p) for p in (
        "scripts/run_swan_supplement2.py", "scripts/run_swan_supplement4.py",
        "scripts/swan_supplement4_runtime.py")})
    for previous in PARENTS:
        for path in (previous / "FREEZE.json", previous / "HALT.json",
                     previous / "private/pilot/journal.jsonl",
                     *sorted((previous / "private/pilot/responses").glob("*.json"))):
            dependencies[str(path.relative_to(ROOT))] = sha256(path)
    dependencies[str((OUT / "PROTOCOL.md").relative_to(ROOT))] = sha256(OUT / "PROTOCOL.md")
    return freeze(OUT / "FREEZE.json", {
        "at": now(), "experiment": OUT.name, "stage": "pilot",
        "dependencies": dependencies,
        "source_file_hashes": load(OUT / "source_manifest.json")["sealed_file_hashes"],
        "user_instruction": "都补一补吧", "generator": budget.MODEL,
        "prompt_addition": pilot.LENGTH,
        "pilot": {"documents": 8, "slots": 16, "max_cumulative_calls": 6400,
                  "max_cumulative_nanodollars": 20_000_000_000},
        "candidate_rule": "Official 10 templates x 5 attempts; same RNG, prompt and threshold",
        "bounded_parser_diagnostic": "candidate rejection; retained in denominator",
        "resource_policy": "checkpoint_and_pause_exit_75",
        "completed_requests": "byte_identical_replay_only_including_all_prior_cost",
        "unknown_consumption": "integrity_halt_no_redispatch",
        "no_replacement": True, "main_dispatched": False, "http_retries": 0})


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
        for call_id, expected in saved["attempt_file_hashes"].items():
            if sha256(OUT / "private/attempts" / f"{call_id}.json") != expected:
                raise IntegrityHalt("attempt_SHA_changed")
        return saved
    rng = random.Random(content_seed("template", allocation, slot))
    examples = module.load_amr_examples(str(
        pilot.swan.OFFICIAL_ROOT / "amr_bank/artifacts/normalized_amr_examples.json"), 3)
    hashes, event = {}, None
    for template_index in range(10):
        target = rng.choice(list(bank))
        prompt = module.build_user_prompt(target, examples, context=context) + pilot.LENGTH
        for within in range(5):
            attempt = template_index * 5 + within
            call_id = f"{slot_id}-attempt-{attempt:02d}"
            response = journal.request(call_id, prompt, module.build_system_prompt(), 50, .7, .9)
            event_path = OUT / "private/attempts" / f"{call_id}.json"
            binding = {"call_id": call_id, "template": target, "candidate": response["text"],
                       "response_sha256": response["call_sha256"],
                       "stop_reason": response["stop_reason"], **identity}
            if event_path.exists():
                event = load(event_path)
                if any(event[k] != v for k, v in binding.items()):
                    raise IntegrityHalt("attempt_binding_changed")
            else:
                pilot.heartbeat("pilot_generating", allocation=allocation, slot=slot,
                                attempt=attempt, budget=journal.totals())
                try:
                    result = module.evaluate(response["text"], target, call_id)
                except CandidateReject as exc:
                    result = {"accepted": False, "score": None, "rejection": str(exc)}
                event = freeze(event_path, {**binding, **result})
            hashes[call_id] = sha256(event_path)
            if event["accepted"]:
                break
        if event["accepted"]:
            break
    return freeze(path, {**identity, "text": event["candidate"], "attempts": len(hashes),
                         "accepted": event["accepted"], "fallback": not event["accepted"],
                         "template": event["template"], "score": event["score"],
                         "attempt_file_hashes": hashes})


def run(key_file):
    pilot.verify()
    if (OUT / "HALT.json").exists():
        raise IntegrityHalt("permanent_halt_exists")
    journal = ReplayJournal(OUT, "pilot", 6400, None, sha256(OUT / "FREEZE.json"))
    if journal.uncertain:
        raise IntegrityHalt("uncertain_call_blocks_resume")
    key = pilot.credential.read_credential(key_file)
    journal.send = lambda body: budget.transport(body, key)
    bank = load_bank(pilot.swan.OFFICIAL_ROOT / "amr_bank/banks/amr_bank_50.json")
    syntax = pilot.FrozenSyntax()
    with runtime(OUT) as module:
        if not (OUT / "READINESS.json").exists():
            result = readiness(module, bank)
            result["inherited_requests_verified"] = len(journal.inherited)
            freeze(OUT / "READINESS.json", result)
        for source in load(OUT / "private/pilot_sources.json"):
            path = OUT / "private/documents" / f"{source['allocation_id']}.json"
            if path.exists():
                continue
            sentences, failure = [], None
            for slot in range(16):
                record = run_slot(module, source, slot,
                                  source["generation_prompt"] + " " + " ".join(sentences), bank, journal)
                sentences.append(record["text"])
                if not record["accepted"]:
                    failure = "slot_exhausted_fallback_is_failure"
                    break
            if failure is None:
                try:
                    pair = pilot.make_pair(source["allocation_id"], sentences, syntax)
                    freeze(OUT / "private/pairs" / path.name, pair)
                except ValueError as exc:
                    failure = str(exc)
            freeze(path, {"allocation_id": source["allocation_id"], "at": now(),
                          "engineering_pass": failure is None, "failure": failure,
                          "slots": len(sentences), "words": len(sources.WORD.findall(" ".join(sentences))),
                          "text": " ".join(sentences)})
            pilot.summarize(journal)
    pilot.review_packet()
    result = pilot.summarize(journal)
    freeze(OUT / "PILOT_RESULT.json", result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("prepare", "verify", "run", "status"))
    ap.add_argument("--key-file", type=Path)
    args = ap.parse_args()
    bind()
    with single_writer(OUT):
        try:
            if args.stage == "prepare":
                prepare()
                result = {"prepared": True}
            elif args.stage == "verify":
                pilot.verify()
                j = ReplayJournal(OUT, "pilot", 6400, None, sha256(OUT / "FREEZE.json"))
                result = {"verified": True, "budget": j.totals(), "detector_calls": 0}
            elif args.stage == "status":
                result = pilot.summarize()
            else:
                if not (OUT / "ENVIRONMENT.json").exists():
                    freeze(OUT / "ENVIRONMENT.json", pilot.environment())
                result = run(args.key_file)
            print(json.dumps(result))
        except (ResourcePause, budget.Pause) as exc:
            pilot.heartbeat("resource_pause", reason=str(exc))
            freeze(OUT / "private/pauses" / f"{digest(now())}.json",
                   {"at": now(), "reason": str(exc), "scientific_result": None})
            raise SystemExit(75)
        except Exception as exc:
            if args.stage == "run":
                freeze(OUT / "HALT.json", {"at": now(), "error_type": type(exc).__name__,
                       "reason": str(exc) if isinstance(exc, (
                           IntegrityHalt, budget.GlobalStop, budget.DocumentFailure)) else "details_private",
                       "automatic_restart_allowed": False})
            print(json.dumps({"stage": "Halt", "error_type": type(exc).__name__}), file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
