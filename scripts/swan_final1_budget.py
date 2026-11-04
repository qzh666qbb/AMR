"""Final1 durable reservation journal. Integer nanodollars; zero HTTP retries."""
from __future__ import annotations

import json
import re
from pathlib import Path

from swan_final1_common import append_event, digest, freeze, load, now, sha256
from swan_final1_transform import text_hash

MODEL = "deepseek-flash"
URL = "https://api.deepseek.com/chat/completions"
MAX_BYTES = 8192
MAX_RESERVE = (MAX_BYTES + 1024) * 300 + 50 * 1200
CAPS = {"pilot": (9601, 27_130_000_000), "main": (600000, 1_694_880_000_000)}


class Pause(RuntimeError):
    """No new slot may start; a later authorized invocation can continue."""


class GlobalStop(RuntimeError):
    """Integrity, provider identity or accounting drift; do not continue."""


class DocumentFailure(RuntimeError):
    """Terminal failure of the current allocation, with no retry."""


def budget_estimate():
    return {
        "currency": "USD", "accounting_unit": "nanodollar",
        "price_url": "https://api-docs.deepseek.com/quick_start/pricing",
        "model": MODEL, "input_nanodollars_per_token": 300,
        "output_nanodollars_per_token": 1200, "max_request_bytes": MAX_BYTES,
        "input_overhead_tokens": 1024, "max_output_tokens": 50,
        "max_call_reserve_nanodollars": MAX_RESERVE,
        "phases": {p: {"max_calls": n, "max_cost_usd": cost / 1e9,
                       "max_cost_nanodollars": cost,
                       "worst_case_nanodollars": n * MAX_RESERVE}
                   for p, (n, cost) in CAPS.items()},
        "transport_retries": 0, "uncertain_calls_retain_full_reserve": True,
    }


def body_for(user_text, system_text, max_tokens=50, temperature=.7, top_p=.9):
    if max_tokens not in (1, 50) or temperature != .7 or top_p != .9:
        raise GlobalStop("Official request settings changed")
    messages = ([{"role": "system", "content": system_text}] if system_text else [])
    messages.append({"role": "user", "content": user_text})
    return {"model": MODEL, "messages": messages, "temperature": temperature,
            "top_p": top_p, "max_tokens": max_tokens, "thinking": {"type": "disabled"}}


def reserve_for(body):
    size = len(json.dumps(body, ensure_ascii=True).encode("utf-8"))
    if size > MAX_BYTES:
        raise DocumentFailure("request_body_exceeds_8192_bytes")
    return {"request_body_bytes": size, "input_token_bound": size + 1024,
            "output_token_bound": 50, "nanodollars": (size + 1024) * 300 + 50 * 1200}


def transport(body, key):
    import requests
    # Send exactly the bytes whose size was reserved; redirects are not followed.
    response = requests.post(
        URL, data=json.dumps(body, ensure_ascii=True).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        timeout=(20, 240), allow_redirects=False,
    )
    if response.status_code != 200:
        raise DocumentFailure(f"HTTP_{response.status_code}")
    return response.json()


class Journal:
    """Caller must hold the final1 single_writer lock for the whole invocation."""

    def __init__(self, folder, phase, max_new_calls, send, contract_hash, allow_halted=False):
        if phase not in CAPS or type(max_new_calls) is not int or max_new_calls <= 0:
            raise ValueError("Valid phase and positive invocation call cap required")
        self.folder = Path(folder) / "private" / phase
        self.path = self.folder / "journal.jsonl"
        self.phase, self.limit, self.send = phase, max_new_calls, send
        self.contract_hash = contract_hash
        self.new_calls = 0
        self.events, self.calls = [], {}
        self.exposure, self.actual, self.uncertain = 0, 0, 0
        if self.path.exists():
            raw = self.path.read_bytes()
            if raw and not raw.endswith(b"\n"):
                raise GlobalStop("Truncated journal; automatic repair prohibited")
            try:
                for line in raw.splitlines():
                    self._accept(json.loads(line))
            except (ValueError, KeyError, TypeError) as exc:
                raise GlobalStop("Journal corrupt") from exc
        if not allow_halted and any(e["kind"] == "halt" for e in self.events):
            raise GlobalStop("Journal contains a permanent integrity halt")

    def _accept(self, event):
        body = {k: v for k, v in event.items() if k != "event_sha256"}
        previous = self.events[-1]["event_sha256"] if self.events else None
        if (event["event_sha256"] != digest(body) or event["previous"] != previous
                or event["sequence"] != len(self.events)
                or event["contract_sha256"] != self.contract_hash):
            raise GlobalStop("Journal chain or contract mismatch")
        kind, call_id = event["kind"], event.get("call_id")
        if kind == "reserve":
            if call_id in self.calls or event["reserve"] != reserve_for(event["request"]):
                raise GlobalStop("Repeated call or changed reservation")
            self.calls[call_id] = {"reserve": event, "terminal": None}
            self.exposure += event["reserve"]["nanodollars"]
            self.uncertain += 1
        elif kind in ("complete", "uncertain", "invalid_response"):
            call = self.calls.get(call_id)
            if call is None or call["terminal"] is not None:
                raise GlobalStop("Invalid terminal call transition")
            if kind in ("complete", "invalid_response"):
                response_path = self.folder / "responses" / f"{call_id}.json"
                if sha256(response_path) != event["response_sha256"]:
                    raise GlobalStop("Saved provider response changed")
            if kind == "complete":
                parsed = self._parse(load(response_path), call["reserve"]["reserve"])
                if event["actual_nanodollars"] != parsed["actual_nanodollars"]:
                    raise GlobalStop("Usage accounting changed")
                self.actual += event["actual_nanodollars"]
                self.exposure += event["actual_nanodollars"] - call["reserve"]["reserve"]["nanodollars"]
                self.uncertain -= 1
            call["terminal"] = event
        elif kind != "halt":
            raise GlobalStop("Unknown journal event")
        self.events.append(event)

    def event(self, kind, **fields):
        body = {"sequence": len(self.events),
                "previous": self.events[-1]["event_sha256"] if self.events else None,
                "contract_sha256": self.contract_hash, "at": now(), "kind": kind, **fields}
        event = {**body, "event_sha256": digest(body)}
        # Persist before mutating in-memory state, and before any transport.
        append_event(self.path, event)
        self._accept(event)
        return event

    def totals(self):
        return {"calls": len(self.calls), "actual_peak_nanodollars": self.actual,
                "exposure_nanodollars": self.exposure, "uncertain_or_invalid_calls": self.uncertain,
                "new_calls": self.new_calls}

    def capacity(self, calls=50):
        totals = self.totals()
        count_cap, money_cap = CAPS[self.phase]
        if self.new_calls + calls > self.limit:
            return "invocation_cap"
        if totals["calls"] + calls > count_cap:
            return "phase_call_cap"
        if totals["exposure_nanodollars"] + calls * MAX_RESERVE > money_cap:
            return "phase_money_cap"
        return None

    def _parse(self, payload, reserve):
        if payload.get("model") != MODEL:
            raise GlobalStop("Provider model identity changed")
        usage = payload.get("usage", {})
        inputs, outputs = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if any(type(n) is not int or n < 0 for n in (inputs, outputs)):
            raise GlobalStop("Missing valid provider token usage")
        if inputs > reserve["input_token_bound"] or outputs > reserve["output_token_bound"]:
            raise GlobalStop("Provider token usage exceeds reserved bound")
        cost = inputs * 300 + outputs * 1200
        if cost > reserve["nanodollars"]:
            raise GlobalStop("Provider usage exceeds reserved cost")
        try:
            choice = payload["choices"][0]
            text, reason = choice["message"]["content"], choice["finish_reason"]
            if not isinstance(payload["id"], str) or not isinstance(text, str) \
                    or reason not in ("stop", "length"):
                raise ValueError("Invalid completion schema")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise GlobalStop("Unexpected provider response schema") from exc
        return {
            "text": text.strip(), "text_sha256": text_hash(text.strip()),
            "stop_reason": reason, "usage": usage, "actual_nanodollars": cost,
            "response_metadata": {"provider": "deepseek", "response_id": payload["id"],
                                  "model": payload["model"],
                                  "system_fingerprint": payload.get("system_fingerprint")},
        }

    def request(self, call_id, user_text, system_text, max_tokens, temperature, top_p):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", call_id) or call_id in self.calls:
            raise GlobalStop("Call identity invalid or already reserved; no retry")
        body = body_for(user_text, system_text, max_tokens, temperature, top_p)
        reserve = reserve_for(body)
        reason = self.capacity(1)
        if reason:
            raise Pause(reason)
        self.event("reserve", call_id=call_id, request=body, reserve=reserve)
        self.new_calls += 1
        try:
            payload = self.send(body)
        except Exception as exc:
            self.event("uncertain", call_id=call_id, error_type=type(exc).__name__)
            raise DocumentFailure("request_uncertain_no_retry") from exc
        response_path = self.folder / "responses" / f"{call_id}.json"
        freeze(response_path, payload)
        response_hash = sha256(response_path)
        try:
            result = self._parse(payload, reserve)
        except GlobalStop as exc:
            self.event("invalid_response", call_id=call_id, response_sha256=response_hash)
            self.event("halt", reason=str(exc))
            raise
        self.event("complete", call_id=call_id, response_sha256=response_hash,
                   actual_nanodollars=result["actual_nanodollars"])
        if not result["text"]:
            raise DocumentFailure("empty_model_output")
        return {**result, "call_sha256": response_hash}
