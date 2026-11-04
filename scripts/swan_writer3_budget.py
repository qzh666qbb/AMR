"""Explicit variable-output extension of the frozen, no-retry writer ledger."""
import json
import re

import swan_writer_repair2 as previous
from swan_final1_common import digest, freeze, load, sha256

base = previous.budget
MAX_BYTES = 24000
MAX_OUTPUT = 2400
MAX_RESERVE = (MAX_BYTES + 1024) * 300 + MAX_OUTPUT * 1200


def body_for(user, system, max_tokens):
    if max_tokens not in (50, 600, 2400):
        raise base.GlobalStop("Unfrozen output limit")
    return {"model": base.MODEL,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": .7, "top_p": .9, "max_tokens": max_tokens,
            "thinking": {"type": "disabled"}}


def reserve_for(body):
    size = len(json.dumps(body, ensure_ascii=True).encode("utf-8"))
    output = body["max_tokens"]
    if size > MAX_BYTES:
        raise base.DocumentFailure("request_body_exceeds_24000_bytes")
    if output not in (50, 600, 2400):
        raise base.GlobalStop("Output reserve changed")
    return {"request_body_bytes": size, "input_token_bound": size + 1024,
            "output_token_bound": output,
            "nanodollars": (size + 1024) * 300 + output * 1200}


class Journal(previous.Journal):
    """Keep the event chain and schema checks; change reserves explicitly."""
    def _accept(self, event):
        if event["kind"] != "reserve":
            return super()._accept(event)
        body = {k: v for k, v in event.items() if k != "event_sha256"}
        if (event["event_sha256"] != digest(body)
                or event["previous"] != (self.events[-1]["event_sha256"] if self.events else None)
                or event["sequence"] != len(self.events)
                or event["contract_sha256"] != self.contract_hash
                or event["call_id"] in self.calls
                or event["reserve"] != reserve_for(event["request"])):
            raise base.GlobalStop("Variable-output reserve or chain mismatch")
        self.calls[event["call_id"]] = {"reserve": event, "terminal": None}
        self.exposure += event["reserve"]["nanodollars"]
        self.uncertain += 1
        self.events.append(event)

    def capacity(self, calls=1):
        totals = self.totals()
        count_cap, money_cap = base.CAPS[self.phase]
        if self.new_calls + calls > self.limit or totals["calls"] + calls > count_cap:
            return "new_call_cap"
        if totals["exposure_nanodollars"] + calls * MAX_RESERVE > money_cap:
            return "new_money_cap"
        if totals["calls"] + self.inherited["calls"] + calls > previous.PARENT_CALL_CAP:
            return "parent_call_cap"
        if (totals["exposure_nanodollars"] + self.inherited["exposure_nanodollars"]
                + calls * MAX_RESERVE > previous.PARENT_MONEY_CAP):
            return "parent_money_cap"
        return None

    def request(self, call_id, user_text, system_text, max_tokens=50):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", call_id):
            raise base.GlobalStop("Invalid request ID")
        body = body_for(user_text, system_text, max_tokens)
        if call_id in self.calls:
            call = self.calls[call_id]
            if (call["terminal"] is None or call["terminal"]["kind"] != "complete"
                    or call["reserve"]["request"] != body):
                raise base.GlobalStop("Unresolved or changed request; never redispatch")
            response_path = self.folder / "responses" / (call_id + ".json")
            if sha256(response_path) != call["terminal"]["response_sha256"]:
                raise base.GlobalStop("Saved response changed")
            return {**self._parse(load(response_path), call["reserve"]["reserve"]),
                    "call_sha256": sha256(response_path)}
        reserve = reserve_for(body)
        reason = self.capacity()
        if reason:
            raise base.Pause(reason)
        self.event("reserve", call_id=call_id, request=body, reserve=reserve)
        self.new_calls += 1
        try:
            payload = self.send(body)
        except Exception as exc:
            self.event("uncertain", call_id=call_id, error_type=type(exc).__name__)
            raise base.DocumentFailure("request_uncertain_no_retry") from exc
        path = self.folder / "responses" / (call_id + ".json")
        freeze(path, payload)
        response_hash = sha256(path)
        try:
            result = self._parse(payload, reserve)
        except base.GlobalStop as exc:
            self.event("invalid_response", call_id=call_id, response_sha256=response_hash)
            self.event("halt", reason=str(exc))
            raise
        self.event("complete", call_id=call_id, response_sha256=response_hash,
                   actual_nanodollars=result["actual_nanodollars"])
        return {**result, "call_sha256": response_hash}
