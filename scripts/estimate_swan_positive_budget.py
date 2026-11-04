"""Estimate and freeze the paid budget for the five-document SWAN pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import run_swan_positive_collection as runner


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/swan_positive_collection1"
ESTIMATE = OUTPUT / "pilot_budget_estimate_v3.json"
REPORT = OUTPUT / "pilot_budget_report_v3.md"
AUTHORIZATION_TEMPLATE = OUTPUT / "budget_authorization_template_v3.json"
PRICE_RETRIEVED_AT = "2026-09-16"
BASE_INPUT_PRICE = 3.00
BASE_OUTPUT_PRICE = 15.00
GEOGRAPHIC_PREMIUM = 1.10
SCENARIOS = {
    "planning_4_bytes_per_prior_output_token": 4,
    "conservative_8_bytes_per_prior_output_token": 8,
}


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_text(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"Frozen budget artifact differs: {path}")
    path.write_text(text, encoding="utf-8")


def prompt_module():
    def no_match(*arguments):
        if len(arguments) != 3:
            raise ValueError("Unexpected official matcher signature")
        return False, 0.0

    return runner.official_function_module_for_test(
        {
            "BedrockManager": object,
            "parse_amr": lambda value: value,
            "normalize_amr_variables": lambda value: value,
            "generate_template_amr": lambda value: value,
            "decide_s2match_match": no_match,
        }
    )


def scenario_cost(
    module,
    pilot: list,
    bank: list,
    examples: list,
    prior_output_bytes_per_token: int,
) -> dict:
    input_token_bound = 0
    output_token_bound = 0
    request_count = 0
    maximum_request = None
    for source in pilot:
        for slot in range(runner.SLOTS):
            prior_bytes = (
                slot
                * runner.MAX_NEW_TOKENS
                * prior_output_bytes_per_token
            )
            context = source["generation_prompt"]
            if prior_bytes:
                context += " " + ("x" * prior_bytes)
            slot_maximum = None
            for template_index, template in enumerate(bank):
                user_text = module.build_user_prompt(
                    template,
                    examples,
                    context=context,
                )
                body = runner.anthropic_request_body(
                    user_text,
                    module.build_system_prompt(),
                    runner.MAX_NEW_TOKENS,
                    runner.TEMPERATURE,
                    runner.DECLARED_TOP_P,
                )
                request = {"body": body}
                reserve = runner.request_cost_reserve(
                    request,
                    runner.MAX_NEW_TOKENS,
                )
                candidate = {
                    **reserve,
                    "allocation_id": source["allocation_id"],
                    "slot": slot,
                    "template_index": template_index,
                }
                if (
                    slot_maximum is None
                    or candidate["reserved_cost_usd"]
                    > slot_maximum["reserved_cost_usd"]
                ):
                    slot_maximum = candidate
            input_token_bound += (
                slot_maximum["input_token_bound"] * runner.MAX_TRIALS
            )
            output_token_bound += (
                slot_maximum["output_token_bound"] * runner.MAX_TRIALS
            )
            request_count += runner.MAX_TRIALS
            if (
                maximum_request is None
                or slot_maximum["reserved_cost_usd"]
                > maximum_request["reserved_cost_usd"]
            ):
                maximum_request = slot_maximum

    generation_cost = (
        input_token_bound
        * runner.INPUT_PRICE_PER_MILLION_USD
        / 1_000_000
        + output_token_bound
        * runner.OUTPUT_PRICE_PER_MILLION_USD
        / 1_000_000
    )
    return {
        "prior_output_bytes_per_token": prior_output_bytes_per_token,
        "generation_requests": request_count,
        "input_token_bound": input_token_bound,
        "output_token_bound": output_token_bound,
        "generation_cost_bound_usd": round(generation_cost, 6),
        "maximum_single_request": maximum_request,
    }


def build(created_at=None) -> dict:
    contract = load_json(runner.CONTRACT)
    if (
        contract["runner_sha256"] != sha256(Path(runner.__file__))
        or contract["model_request"]["input_price_per_million_usd"]
        != runner.INPUT_PRICE_PER_MILLION_USD
        or contract["model_request"]["output_price_per_million_usd"]
        != runner.OUTPUT_PRICE_PER_MILLION_USD
    ):
        raise ValueError("Execution contract does not freeze budget logic")
    source = load_json(runner.SOURCE_MANIFEST)
    pilot = [
        row for row in source["sources"] if row["cohort"] == "pilot"
    ]
    if len(pilot) != 5:
        raise ValueError("Frozen pilot membership changed")
    bank_records = load_json(
        runner.OFFICIAL_ROOT / "amr_bank/banks/amr_bank_50.json"
    )
    bank = [row["normalized_template_amr"] for row in bank_records]
    examples = load_json(
        runner.OFFICIAL_ROOT
        / "amr_bank/artifacts/normalized_amr_examples.json"
    )[:3]
    module = prompt_module()
    scenarios = {
        name: scenario_cost(
            module,
            pilot,
            bank,
            examples,
            bytes_per_token,
        )
        for name, bytes_per_token in SCENARIOS.items()
    }
    probe_request = {"body": runner.anthropic_request_body(
        "Reply exactly OK.",
        None,
        16,
        0.0,
        runner.DECLARED_TOP_P,
    )}
    probe_reserve = runner.request_cost_reserve(probe_request, 16)
    conservative = scenarios[
        "conservative_8_bytes_per_prior_output_token"
    ]["generation_cost_bound_usd"] + probe_reserve["reserved_cost_usd"]
    recommended = math.ceil(conservative * 1.25 / 5) * 5
    return {
        "schema_version": 3,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "status": "frozen_budget_estimate_remote_unauthorized",
        "execution_contract_sha256": sha256(runner.CONTRACT),
        "source_manifest_sha256": sha256(runner.SOURCE_MANIFEST),
        "model_id": runner.MODEL,
        "region": runner.REGION,
        "pricing": {
            "retrieved_at": PRICE_RETRIEVED_AT,
            "base_input_per_million_usd": BASE_INPUT_PRICE,
            "base_output_per_million_usd": BASE_OUTPUT_PRICE,
            "geographic_profile_premium": GEOGRAPHIC_PREMIUM,
            "budget_input_per_million_usd": (
                runner.INPUT_PRICE_PER_MILLION_USD
            ),
            "budget_output_per_million_usd": (
                runner.OUTPUT_PRICE_PER_MILLION_USD
            ),
            "sources": [
                (
                    "https://aws.amazon.com/bedrock/pricing/"
                ),
                (
                    "https://platform.claude.com/docs/en/"
                    "about-claude/pricing"
                ),
                (
                    "https://docs.aws.amazon.com/bedrock/latest/"
                    "userguide/model-card-anthropic-"
                    "claude-sonnet-4-5.html"
                ),
            ],
            "interpretation": (
                "Use the published 3/15 USD base rates plus a conservative "
                "10 percent geographic-profile premium because the frozen "
                "model ID starts with us. Runtime reserves this bound even "
                "if the invoiced US-profile rate is lower."
            ),
        },
        "method": {
            "pilot_documents": 5,
            "slots_per_document": 16,
            "max_trials_per_slot": 50,
            "maximum_generation_requests": 4000,
            "model_probe_requests": 1,
            "maximum_total_model_calls": 4001,
            "input_token_bound": (
                "UTF-8 bytes of the exact request body plus 1,024 tokens"
            ),
            "output_token_bound_per_generation": 50,
            "template_handling": (
                "Use the largest reserve among all 50 official templates "
                "for every slot."
            ),
        },
        "scenarios": scenarios,
        "probe_cost_reserve": probe_reserve,
        "conservative_total_bound_usd": round(conservative, 6),
        "recommended_authorization": {
            "max_cost_usd": recommended,
            "max_model_calls": 4001,
            "contingency": "25 percent, rounded up to the next 5 USD",
        },
        "runtime_enforcement": {
            "pre_transport": (
                "Reject a call when completed estimated cost plus the "
                "current request reserve exceeds max_cost_usd."
            ),
            "post_response": (
                "Record provider input/output usage and reject evidence if "
                "actual estimated cost exceeds its pre-call reserve."
            ),
            "uncertain_call": (
                "Freeze and block all continuation; never retry "
                "automatically."
            ),
        },
        "resource_counts": {
            "aws_control_plane_calls": 0,
            "model_probe_calls": 0,
            "generation_calls": 0,
        },
    }


def report_text(payload: dict) -> str:
    planning = payload["scenarios"][
        "planning_4_bytes_per_prior_output_token"
    ]
    conservative = payload["scenarios"][
        "conservative_8_bytes_per_prior_output_token"
    ]
    recommended = payload["recommended_authorization"]
    return f"""# SWAN pilot budget estimate

No AWS or model call was made for this estimate.

- Frozen pilot: 5 documents, 16 slots each, at most 50 requests per slot
- Maximum generation requests: 4,000
- Model probe requests: 1
- Maximum total model calls: 4,001
- Conservative pricing bound: $3.30/M input and $16.50/M output tokens
- Planning scenario: ${planning["generation_cost_bound_usd"]:.2f}
- Conservative generation scenario: ${conservative["generation_cost_bound_usd"]:.2f}
- Conservative scenario plus probe: ${payload["conservative_total_bound_usd"]:.2f}
- Recommended authorization ceiling: ${recommended["max_cost_usd"]:.2f}

The recommendation adds 25 percent contingency and rounds upward to the next
$5. It is a spending ceiling, not a forecast that the pilot will consume all
4,001 calls. The runner enforces both call count and USD bounds before every
transport using the actual serialized request body, then records provider
token usage after each response. An uncertain request blocks continuation.

Authorization remains absent. To continue, the user must explicitly approve
both `${recommended["max_cost_usd"]:.2f}` and `{recommended["max_model_calls"]}`
model calls, or provide lower limits and accept that the pilot may pause.
"""


def authorization_template(payload: dict) -> dict:
    recommended = payload["recommended_authorization"]
    return {
        "schema_version": 3,
        "status": "awaiting_explicit_user_approval",
        "phase": "pilot_collection",
        "model_id": runner.MODEL,
        "region": runner.REGION,
        "source_manifest_sha256": sha256(runner.SOURCE_MANIFEST),
        "execution_contract_sha256": sha256(runner.CONTRACT),
        "budget_estimate_sha256": sha256(ESTIMATE),
        "max_model_calls": recommended["max_model_calls"],
        "max_cost_usd": recommended["max_cost_usd"],
        "approval": {
            "explicit_user_statement": None,
            "approved_at": None,
        },
        "activation": (
            "This template is not authorization. Only an explicit user "
            "approval may create budget_authorization.json with status "
            "explicitly_authorized."
        ),
    }


def create() -> dict:
    paths = (ESTIMATE, REPORT, AUTHORIZATION_TEMPLATE)
    if any(path.exists() for path in paths):
        if not all(path.is_file() for path in paths):
            raise RuntimeError("Partial budget freeze requires review")
        return check()
    payload = build()
    freeze_text(
        ESTIMATE,
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
    )
    freeze_text(REPORT, report_text(payload))
    freeze_text(
        AUTHORIZATION_TEMPLATE,
        json.dumps(
            authorization_template(payload),
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
    )
    return payload


def check() -> dict:
    saved = load_json(ESTIMATE)
    live = build(saved["created_at"])
    if saved != live:
        raise ValueError("Frozen budget estimate differs from live inputs")
    if REPORT.read_text(encoding="utf-8") != report_text(saved):
        raise ValueError("Frozen budget report differs")
    expected_template = (
        json.dumps(
            authorization_template(saved),
            indent=2,
            ensure_ascii=True,
        )
        + "\n"
    )
    if AUTHORIZATION_TEMPLATE.read_text(encoding="utf-8") != (
        expected_template
    ):
        raise ValueError("Frozen authorization template differs")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("create", "check"))
    args = parser.parse_args()
    payload = create() if args.stage == "create" else check()
    print(
        json.dumps(
            {
                "status": payload["status"],
                "conservative_total_bound_usd": payload[
                    "conservative_total_bound_usd"
                ],
                "recommended_authorization": payload[
                    "recommended_authorization"
                ],
                "resource_counts": payload["resource_counts"],
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
