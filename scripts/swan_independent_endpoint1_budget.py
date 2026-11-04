"""Isolated instances of the frozen at-most-once journal, with bounded profiles."""
import importlib.util
import json
from pathlib import Path

from swan_independent_endpoint1_core import WRITER_PARAMETERS, ATTACK_PARAMETERS

PROFILES = {
    "writer": (WRITER_PARAMETERS, 72 * 4 * 16 * 50, 40_000_000_000),
    "attack": (ATTACK_PARAMETERS, 72 * 4 * 16, 10_000_000_000),
}


def journal(out, contract, phase, send=None):
    parameters, calls, money = PROFILES[phase]
    spec = importlib.util.spec_from_file_location(
        "_independent_endpoint1_" + phase,
        Path(__file__).with_name("symbol_prompt_completion1_budget.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PARAMETERS = parameters
    module.CALL_CAP, module.MONEY_CAP = calls, money
    original_parse = module.parse

    def reserve(body):
        if {k: v for k, v in body.items() if k != "messages"} != parameters:
            raise module.IntegrityError("profile_parameters_changed")
        size = len(json.dumps(body, ensure_ascii=True).encode())
        if size > 32768:
            raise module.BudgetPause("request_exceeds_frozen_byte_bound")
        return {"input_tokens": size + 1024, "output_tokens": parameters["max_tokens"],
                "nanodollars": (size + 1024) * 300 + parameters["max_tokens"] * 1200}

    module.reserve = reserve
    module.parse = original_parse
    return module.Journal(Path(out) / "private/provider" / phase, contract, send)
