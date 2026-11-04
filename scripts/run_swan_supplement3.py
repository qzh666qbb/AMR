"""Explicit serialization fix, with byte-identical replay of one completed call."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import run_swan_supplement2 as parent
from swan_final1_common import digest, freeze, load, now, sha256, single_writer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/swan_count_neutral_supplement3"
PARENT = ROOT / "experiments/swan_count_neutral_supplement2"
RAW_RUNTIME = parent.runtime.runtime
RAW_JOURNAL = parent.runner.budget.Journal


@contextmanager
def serializable_runtime():
    with RAW_RUNTIME() as module:
        match = module.decide_s2match_match
        def native_match(*args, **kwargs):
            accepted, score = match(*args, **kwargs)
            return bool(accepted), float(score)
        module.decide_s2match_match = native_match
        yield module


def inherited_call():
    journal_path = PARENT / "private/pilot/journal.jsonl"
    events = [json.loads(line) for line in journal_path.read_text().splitlines()]
    if len(events) != 2 or [e["kind"] for e in events] != ["reserve", "complete"]:
        raise parent.runtime.IntegrityHalt("parent_not_exactly_one_completed_call")
    reserve, complete = events
    if reserve["call_id"] != complete["call_id"]:
        raise parent.runtime.IntegrityHalt("parent_call_identity_mismatch")
    response_path = PARENT / "private/pilot/responses" / f"{reserve['call_id']}.json"
    if sha256(response_path) != complete["response_sha256"]:
        raise parent.runtime.IntegrityHalt("parent_response_SHA_changed")
    for index, event in enumerate(events):
        body = {k: v for k, v in event.items() if k != "event_sha256"}
        previous = events[index - 1]["event_sha256"] if index else None
        if event["event_sha256"] != digest(body) or event["previous"] != previous:
            raise parent.runtime.IntegrityHalt("parent_journal_chain_failed")
        if event["contract_sha256"] != sha256(PARENT / "FREEZE.json"):
            raise parent.runtime.IntegrityHalt("parent_journal_contract_failed")
    reserve_expected = parent.runner.budget.reserve_for(reserve["request"])
    if reserve["reserve"] != reserve_expected:
        raise parent.runtime.IntegrityHalt("parent_reservation_changed")
    parser = object.__new__(RAW_JOURNAL)
    response = parser._parse(load(response_path), reserve["reserve"])
    if complete["actual_nanodollars"] != response["actual_nanodollars"]:
        raise parent.runtime.IntegrityHalt("parent_usage_accounting_changed")
    return reserve, complete, response


class ReplayJournal(RAW_JOURNAL):
    """Replay the identical already-paid request once; new requests get a new chain."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inherited_reserve, self.inherited_complete, self.inherited_response = inherited_call()
        self.replayed = False
    def request(self, call_id, user_text, system_text, max_tokens, temperature, top_p):
        if call_id == self.inherited_reserve["call_id"]:
            if self.replayed:
                raise parent.runtime.IntegrityHalt("parent_call_replayed_twice")
            request = parent.runner.budget.body_for(
                user_text, system_text, max_tokens, temperature, top_p)
            if request != self.inherited_reserve["request"]:
                raise parent.runtime.IntegrityHalt("replay_request_differs_from_paid_parent")
            self.replayed = True
            freeze(OUT / "REPLAY_AUDIT.json", {
                "call_id": call_id, "parent_response_sha256": self.inherited_complete["response_sha256"],
                "request_identical": True, "new_transport_dispatches": 0,
                "parent_actual_nanodollars": self.inherited_response["actual_nanodollars"],
            })
            return {**self.inherited_response,
                    "call_sha256": self.inherited_complete["response_sha256"]}
        return super().request(call_id, user_text, system_text, max_tokens, temperature, top_p)
    def totals(self):
        totals = super().totals()
        if hasattr(self, "inherited_response"):
            totals["calls"] += 1
            totals["actual_peak_nanodollars"] += self.inherited_response["actual_nanodollars"]
            totals["exposure_nanodollars"] += self.inherited_response["actual_nanodollars"]
            totals["inherited_complete_calls"] = 1
        return totals


def bind():
    parent.bind()
    for module in (parent.runner, parent.sources, parent.runtime, parent.review):
        module.OUT = OUT
    parent.runner.DEPENDENCIES = [
        p.replace("experiments/swan_count_neutral_supplement2/",
                  "experiments/swan_count_neutral_supplement3/")
        if p.startswith("experiments/swan_count_neutral_supplement2/") else p
        for p in parent.runner.DEPENDENCIES]
    parent.runner.DEPENDENCIES += [
        "scripts/run_swan_supplement3.py",
        "experiments/swan_count_neutral_supplement2/FREEZE.json",
        "experiments/swan_count_neutral_supplement2/PROTOCOL.md",
        "experiments/swan_count_neutral_supplement2/HALT.json",
        "experiments/swan_count_neutral_supplement2/private/pilot/journal.jsonl",
    ]
    reserve, complete, response = inherited_call()
    parent.runner.DEPENDENCIES.append(
        f"experiments/swan_count_neutral_supplement2/private/pilot/responses/{reserve['call_id']}.json")
    parent.runner.runtime = serializable_runtime
    parent.runner.budget.Journal = ReplayJournal


def prepare():
    parent.sources.audit()
    reserve, complete, response = inherited_call()
    return freeze(OUT / "FREEZE.json", {
        "at": now(), "experiment": "swan_count_neutral_supplement3", "stage": "pilot",
        "dependencies": {p: sha256(ROOT / p) for p in parent.runner.DEPENDENCIES},
        "source_file_hashes": load(OUT / "source_manifest.json")["sealed_file_hashes"],
        "generator": "deepseek-flash", "prompt_addition": parent.runner.LENGTH,
        "pilot": {"documents": 8, "slots": 16, "max_cumulative_calls": 6400,
                  "agent_selected_cost_guard_usd": 20, "worst_case_usd": 18.07872},
        "user_instruction": "行，那就补全一下实验吧",
        "authorization_basis": "Authorized supplement; engineering serialization successor.",
        "correction": "Cast matcher acceptance/score to Python bool/float before immutable JSON",
        "parent_observation": "One complete provider response, no complete slot or detector outcome",
        "replay": {"call_id": reserve["call_id"], "request_must_be_identical": True,
                   "response_sha256": complete["response_sha256"],
                   "actual_nanodollars": response["actual_nanodollars"],
                   "new_transport_dispatches": 0},
        "source_inheritance": "Same fixed allocation; no replacement or tuning",
        "main_dispatched": False, "http_retries": 0,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "preflight", "run", "status",
                                        "review-packet", "finalize-review"))
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--reviewer1", type=Path)
    parser.add_argument("--reviewer2", type=Path)
    args = parser.parse_args()
    bind()
    with single_writer(OUT):
        try:
            if args.stage == "prepare":
                prepare()
                result = {"new_freeze": True, "parent_complete_calls": 1}
            elif args.stage == "preflight":
                parent.runner.verify()
                parent.runner.heartbeat("CUDA_readiness")
                with serializable_runtime() as module:
                    bank = parent.load_bank(ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_50.json")
                    result = parent.runtime.readiness(module, bank)
                    # Exercise the full slot logging path with the exact saved response.
                    reserve, complete, response = inherited_call()
                    original_out = parent.runner.OUT
                    parent.runner.OUT = OUT / "private/offline_slot_validation"
                    parent.runner.OUT.mkdir(parents=True, exist_ok=True)
                    freeze(parent.runner.OUT / "FREEZE.json", load(OUT / "FREEZE.json"))
                    source = load(OUT / "private/pilot_sources.json")[0]
                    class EndReplay(RuntimeError):
                        pass
                    class OneResponse:
                        calls = {}
                        count = 0
                        def totals(self):
                            return {"calls": 1}
                        def request(self, call_id, user_text, system_text, max_tokens, temperature, top_p):
                            self.count += 1
                            if self.count > 1:
                                raise EndReplay()
                            request = parent.runner.budget.body_for(
                                user_text, system_text, max_tokens, temperature, top_p)
                            if request != reserve["request"]:
                                raise parent.runtime.IntegrityHalt("offline_request_mismatch")
                            return {**response, "call_sha256": complete["response_sha256"]}
                    try:
                        try:
                            parent.runner.run_slot(module, source, 0,
                                source["generation_prompt"] + " ", bank, OneResponse())
                        except EndReplay:
                            pass
                        attempt = next((parent.runner.OUT / "private/attempts").glob("*.json"))
                        item = load(attempt)
                        assert type(item["accepted"]) is bool and type(item["score"]) is float
                        result["full_attempt_JSON_replay"] = "Go"
                    finally:
                        parent.runner.OUT = original_out
                freeze(OUT / "READINESS.json", result)
            elif args.stage == "run":
                freeze(OUT / "ENVIRONMENT.json", parent.runner.environment())
                result = parent.runner.run(args.key_file)
            elif args.stage == "status":
                result = parent.runner.summarize()
            elif args.stage == "finalize-review":
                result = parent.review.finalize(args.reviewer1, args.reviewer2)
            else:
                result = {"review_items": len(parent.runner.review_packet()["items"])}
            print(json.dumps(result))
        except Exception as exc:
            if args.stage in ("run", "preflight"):
                freeze(OUT / "HALT.json", {
                    "error_type": type(exc).__name__,
                    "reason": str(exc) if isinstance(exc, (parent.runtime.IntegrityHalt,
                        parent.runner.budget.GlobalStop, parent.runner.budget.DocumentFailure)) else "details_private",
                    "automatic_restart_allowed": False,
                })
                parent.runner.summarize()
            print(json.dumps({"stage": "Halt", "error_type": type(exc).__name__}), file=sys.stderr)
            sys.exit(1)
