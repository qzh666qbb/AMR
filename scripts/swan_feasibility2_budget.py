"""Frozen paid-call ceiling for the 20-document feasibility pilot."""
from __future__ import annotations

from swan_final1_budget import (
    DocumentFailure,
    GlobalStop,
    Journal as _BaseJournal,
    MAX_BYTES,
    MAX_RESERVE,
    MODEL,
    Pause,
    URL,
    body_for,
    transport,
)

PILOT_CALLS = 1 + 20 * 10 * 50
PILOT_CAP = (PILOT_CALLS, PILOT_CALLS * MAX_RESERVE)


def budget_estimate():
    calls, cost = PILOT_CAP
    return {
        "currency": "USD",
        "accounting_unit": "nanodollar",
        "price_url": "https://api-docs.deepseek.com/quick_start/pricing",
        "price_checked_date_at_design": "2026-09-20",
        "model": MODEL,
        "pricing_basis": "peak cache-miss upper bound",
        "input_nanodollars_per_token": 300,
        "output_nanodollars_per_token": 1200,
        "max_request_bytes": MAX_BYTES,
        "input_overhead_tokens": 1024,
        "max_output_tokens": 50,
        "max_call_reserve_nanodollars": MAX_RESERVE,
        "pilot": {
            "documents": 20,
            "slots_per_document": 10,
            "max_attempts_per_slot": 50,
            "model_identity_probe_calls": 1,
            "max_calls": calls,
            "max_cost_nanodollars": cost,
            "max_cost_usd": cost / 1e9,
        },
        "transport_retries": 0,
        "uncertain_calls_retain_full_reserve": True,
    }


class Journal(_BaseJournal):
    def __init__(self, folder, max_new_calls, send, contract_hash, allow_halted=False):
        super().__init__(
            folder, "pilot", max_new_calls, send, contract_hash,
            allow_halted=allow_halted,
        )

    def capacity(self, calls=50):
        totals = self.totals()
        count_cap, money_cap = PILOT_CAP
        if self.new_calls + calls > self.limit:
            return "invocation_cap"
        if totals["calls"] + calls > count_cap:
            return "phase_call_cap"
        if totals["exposure_nanodollars"] + calls * MAX_RESERVE > money_cap:
            return "phase_money_cap"
        return None
