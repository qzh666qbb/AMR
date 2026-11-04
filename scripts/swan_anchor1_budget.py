"""Budget and authorization constants for source-anchored generation."""
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
    reserve_for,
    transport,
)

PILOT_CALLS = 1 + 8 * 10 * 30
MAIN_CALLS = 500 * 10 * 30
CAPS = {
    "pilot": (PILOT_CALLS, PILOT_CALLS * MAX_RESERVE),
    "main": (MAIN_CALLS, MAIN_CALLS * MAX_RESERVE),
}


def budget_estimate():
    return {
        "currency": "USD",
        "accounting_unit": "nanodollar",
        "price_url": "https://api-docs.deepseek.com/quick_start/pricing",
        "price_checked_date": "2026-09-19",
        "model": MODEL,
        "resolved_model": "DeepSeek-V4.1-Flash",
        "pricing_basis": "peak cache-miss upper bound",
        "input_nanodollars_per_token": 300,
        "output_nanodollars_per_token": 1200,
        "max_request_bytes": MAX_BYTES,
        "input_overhead_tokens": 1024,
        "max_output_tokens": 50,
        "max_call_reserve_nanodollars": MAX_RESERVE,
        "slots_per_document": 10,
        "templates_per_slot": 10,
        "attempts_per_template": 3,
        "phases": {
            phase: {
                "max_calls": calls,
                "max_cost_nanodollars": cost,
                "max_cost_usd": cost / 1e9,
                "worst_case_nanodollars": calls * MAX_RESERVE,
            }
            for phase, (calls, cost) in CAPS.items()
        },
        "transport_retries": 0,
        "uncertain_calls_retain_full_reserve": True,
    }


class Journal(_BaseJournal):
    """Reuse the durable final1 journal with anchor1 phase caps."""

    def __init__(self, folder, phase, max_new_calls, send, contract_hash, allow_halted=False):
        super().__init__(
            folder, phase, max_new_calls, send, contract_hash, allow_halted=allow_halted
        )

    def capacity(self, calls=30):
        totals = self.totals()
        count_cap, money_cap = CAPS[self.phase]
        if self.new_calls + calls > self.limit:
            return "invocation_cap"
        if totals["calls"] + calls > count_cap:
            return "phase_call_cap"
        if totals["exposure_nanodollars"] + calls * MAX_RESERVE > money_cap:
            return "phase_money_cap"
        return None
