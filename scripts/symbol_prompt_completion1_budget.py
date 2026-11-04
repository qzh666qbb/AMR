"""At-most-once paired-prompt journal; retained outputs never trigger retries."""
import json
import re
from pathlib import Path

from swan_final1_common import append_event, digest, freeze, load, now, sha256

PARAMETERS = {
    "model": "deepseek-flash", "temperature": .9, "top_p": 1,
    "max_tokens": 1600, "n": 1, "stream": False,
    "presence_penalty": 0, "frequency_penalty": 0,
    "thinking": {"type": "disabled"},
}
CALL_CAP = 432
MONEY_CAP = 10_000_000_000
MAX_BYTES = 32768


class IntegrityError(RuntimeError):
    pass


class BudgetPause(RuntimeError):
    pass


def reserve(body):
    if {k: v for k, v in body.items() if k != "messages"} != PARAMETERS:
        raise IntegrityError("generation_parameters_changed")
    size = len(json.dumps(body, ensure_ascii=True).encode())
    if size > MAX_BYTES:
        raise BudgetPause("request_exceeds_frozen_byte_bound")
    return {"input_tokens": size + 1024, "output_tokens": 1600,
            "nanodollars": (size + 1024) * 300 + 1600 * 1200}


def parse(payload, allowance):
    usage = payload.get("usage", {})
    ni, no = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if any(type(x) is not int or x < 0 for x in (ni, no)):
        raise IntegrityError("unknown_provider_usage")
    if ni > allowance["input_tokens"] or no > allowance["output_tokens"]:
        raise IntegrityError("usage_exceeds_reservation")
    if payload.get("model") != PARAMETERS["model"]:
        raise IntegrityError("provider_model_changed")
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise IntegrityError("expected_single_completion")
    choice = choices[0]
    text = choice.get("message", {}).get("content")
    if not isinstance(text, str) or choice.get("finish_reason") not in ("stop", "length"):
        raise IntegrityError("invalid_completion")
    return {"text": text.strip(), "finish_reason": choice["finish_reason"],
            "usage": usage, "nanodollars": ni * 300 + no * 1200,
            "model": payload["model"], "fingerprint": payload.get("system_fingerprint")}


class Journal:
    def __init__(self, out, contract, send=None):
        self.out, self.contract, self.send = Path(out), contract, send
        self.path = self.out / "private/calls/journal.jsonl"
        self.events, self.calls = [], {}
        if self.path.exists():
            raw = self.path.read_bytes()
            if raw and not raw.endswith(b"\n"):
                raise IntegrityError("truncated_journal")
            for line in raw.splitlines():
                self.accept(json.loads(line))
        if any(c["complete"] is None for c in self.calls.values()):
            raise IntegrityError("unresolved_dispatch_no_automatic_retry")

    def accept(self, event):
        body = {k: v for k, v in event.items() if k != "sha256"}
        if (event["sha256"] != digest(body) or event["contract"] != self.contract
                or event["sequence"] != len(self.events)
                or event["previous"] != (self.events[-1]["sha256"] if self.events else None)):
            raise IntegrityError("journal_chain_changed")
        call_id = event["call_id"]
        if event["kind"] == "intent":
            if call_id in self.calls or event["allowance"] != reserve(event["body"]):
                raise IntegrityError("invalid_intent")
            self.calls[call_id] = {"intent": event, "complete": None}
        elif event["kind"] == "complete":
            call = self.calls.get(call_id)
            if call is None or call["complete"] is not None:
                raise IntegrityError("invalid_completion_transition")
            path = self.out / "private/calls" / (call_id + ".json")
            if sha256(path) != event["response_sha256"]:
                raise IntegrityError("response_changed")
            result = parse(load(path), call["intent"]["allowance"])
            if result["nanodollars"] != event["nanodollars"]:
                raise IntegrityError("cost_changed")
            call["complete"] = event
        else:
            raise IntegrityError("unknown_journal_event")
        self.events.append(event)

    def event(self, kind, call_id, **fields):
        body = {"kind": kind, "call_id": call_id, "contract": self.contract,
                "sequence": len(self.events), "at": now(),
                "previous": self.events[-1]["sha256"] if self.events else None, **fields}
        event = {**body, "sha256": digest(body)}
        append_event(self.path, event)
        self.accept(event)

    def totals(self):
        return {
            "calls": len(self.calls),
            "exposure_nanodollars": sum(c["complete"]["nanodollars"] if c["complete"]
                                       else c["intent"]["allowance"]["nanodollars"]
                                       for c in self.calls.values()),
            "uncertain_calls": sum(c["complete"] is None for c in self.calls.values()),
        }

    def request(self, call_id, messages):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", call_id):
            raise IntegrityError("invalid_call_id")
        body = dict(PARAMETERS, messages=messages)
        allowance = reserve(body)
        if call_id in self.calls:
            call = self.calls[call_id]
            if call["intent"]["body"] != body or call["complete"] is None:
                raise IntegrityError("request_binding_changed_or_uncertain")
            return parse(load(self.out / "private/calls" / (call_id + ".json")), allowance)
        total = self.totals()
        if (total["calls"] >= CALL_CAP
                or total["exposure_nanodollars"] + allowance["nanodollars"] > MONEY_CAP):
            raise BudgetPause("frozen_call_or_cost_cap")
        self.event("intent", call_id, body=body, allowance=allowance)
        # A transport exception or unknown response leaves an unresolved intent.
        # It retains its reserve and cannot be dispatched again on recovery.
        payload = self.send(body)
        path = self.out / "private/calls" / (call_id + ".json")
        freeze(path, payload)
        result = parse(payload, allowance)
        self.event("complete", call_id, response_sha256=sha256(path),
                   nanodollars=result["nanodollars"])
        return result
