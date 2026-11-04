"""Prepare and run a bounded DeepSeek SWAN backend preflight."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import stat
from typing import Callable, Optional

import run_swan_positive_collection as swan
import run_swan_positive_local_preflight as local


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/swan_positive_deepseek_preflight1"
MANIFEST = OUTPUT / "manifest.json"
BUDGET = OUTPUT / "budget_estimate.json"
AUTH_TEMPLATE = OUTPUT / "budget_authorization_template.json"
AUTHORIZATION = OUTPUT / "budget_authorization.json"
MODEL = "deepseek-flash"
RETURNED_MODEL = "deepseek-flash"
URL = "https://api.deepseek.com/chat/completions"
PRICE_SOURCE = "https://api-docs.deepseek.com/quick_start/pricing"
PRICE_RETRIEVED = "2026-09-17"
PEAK_CACHE_MISS_INPUT_USD_PER_MILLION = 0.30
PEAK_OUTPUT_USD_PER_MILLION = 1.20
INPUT_OVERHEAD_RESERVE = 1024
SOURCES = 5
SLOTS = 1
MAX_TRIALS = 10
MAX_TEMPLATES = 2
RETRIES_PER_TEMPLATE = 5
GENERATION_CALL_CAP = SOURCES * SLOTS * MAX_TRIALS
PROBE_CALLS = 1
TOTAL_CALL_CAP = GENERATION_CALL_CAP + PROBE_CALLS
CONTINUE_ACCEPTED = 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest(value) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def freeze_json(path: Path, payload) -> dict:
    if path.exists():
        saved = load_json(path)
        if saved != payload:
            raise ValueError(f"Frozen DeepSeek artifact differs: {path}")
        return saved
    write_json(path, payload)
    return payload


def request_body(
    user_text: str,
    system_text: Optional[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> dict:
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})
    return {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
    }


def request_reserve(body: dict) -> dict:
    request_bytes = len(
        json.dumps(body, ensure_ascii=True).encode("utf-8")
    )
    input_bound = request_bytes + INPUT_OVERHEAD_RESERVE
    output_bound = body["max_tokens"]
    cost = (
        input_bound
        * PEAK_CACHE_MISS_INPUT_USD_PER_MILLION
        / 1_000_000
        + output_bound
        * PEAK_OUTPUT_USD_PER_MILLION
        / 1_000_000
    )
    return {
        "request_body_bytes": request_bytes,
        "input_token_bound": input_bound,
        "output_token_bound": output_bound,
        "peak_all_cache_miss_cost_usd": round(cost, 12),
    }


def actual_cost(usage: dict) -> dict:
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if (
        type(prompt) is not int
        or prompt < 0
        or type(completion) is not int
        or completion < 0
    ):
        raise ValueError("DeepSeek response lacks valid token usage")
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get(
        "cached_tokens",
        usage.get("prompt_cache_hit_tokens", 0),
    )
    if type(cached) is not int or cached < 0 or cached > prompt:
        raise ValueError("DeepSeek response has invalid cached-token usage")
    uncached = prompt - cached
    conservative = (
        prompt
        * PEAK_CACHE_MISS_INPUT_USD_PER_MILLION
        / 1_000_000
        + completion
        * PEAK_OUTPUT_USD_PER_MILLION
        / 1_000_000
    )
    return {
        "prompt_tokens": prompt,
        "cached_prompt_tokens": cached,
        "uncached_prompt_tokens": uncached,
        "completion_tokens": completion,
        "conservative_peak_cost_usd": round(conservative, 12),
    }


def pure_prompt_module():
    def no_match(*arguments):
        if len(arguments) != 3:
            raise ValueError("Unexpected official matcher signature")
        return False, 0.0

    return swan.official_function_module_for_test(
        {
            "BedrockManager": object,
            "parse_amr": lambda value: value,
            "normalize_amr_variables": lambda value: value,
            "generate_template_amr": lambda value: value,
            "decide_s2match_match": no_match,
        }
    )


def dependencies() -> dict:
    paths = [
        Path(__file__),
        ROOT / "scripts/test_swan_positive_deepseek_preflight.py",
        ROOT / "scripts/run_swan_positive_collection.py",
        local.SOURCE_MANIFEST,
        local.OFFICIAL_CONTRACT,
        swan.OFFICIAL_INJECTION,
        swan.S2MATCH_RUNTIME / "provenance.json",
    ]
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for path in paths
    }


def build_manifest(created_at=None) -> dict:
    return {
        "schema_version": 1,
        "created_at": created_at or utc_now(),
        "status": "frozen_remote_unauthorized",
        "workflow": "swan_positive_deepseek_preflight1",
        "source_manifest_sha256": sha256(local.SOURCE_MANIFEST),
        "official_execution_contract_sha256": sha256(
            local.OFFICIAL_CONTRACT
        ),
        "sources": [
            {
                "allocation_id": row["allocation_id"],
                "generation_prompt": row["generation_prompt"],
                "normalized_source_sha256": row[
                    "normalized_source_sha256"
                ],
            }
            for row in local.pilot_sources()
        ],
        "backend": {
            "endpoint": URL,
            "model_requested": MODEL,
            "required_model_returned": RETURNED_MODEL,
            "thinking": "disabled",
            "not_official_equivalent": True,
            "reason": (
                "DeepSeek-V4.1-Flash replaces Claude Sonnet 4.5 only for "
                "backend feasibility."
            ),
        },
        "protocol": {
            "documents": SOURCES,
            "slots_per_document": SLOTS,
            "max_trials_per_slot": MAX_TRIALS,
            "max_templates_per_slot": MAX_TEMPLATES,
            "retries_per_template": RETRIES_PER_TEMPLATE,
            "s2match_threshold": swan.MATCH_THRESHOLD,
            "max_new_tokens": swan.MAX_NEW_TOKENS,
            "temperature": swan.TEMPERATURE,
            "top_p": swan.DECLARED_TOP_P,
            "probe_calls": PROBE_CALLS,
            "maximum_generation_calls": GENERATION_CALL_CAP,
            "maximum_total_calls": TOTAL_CALL_CAP,
            "continue_gate": (
                "At least 3 of 5 first slots accepted within 10 trials. "
                "Otherwise stop the DeepSeek route."
            ),
            "no_parameter_changes_after_outputs": True,
            "no_detection": True,
            "no_attack": True,
            "no_tpr_or_asr_claim": True,
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("requests", "torch", "transformers")
        },
        "dependencies": dependencies(),
        "authorization": {
            "required_status": "explicitly_authorized",
            "required_fields": [
                "max_cost_usd",
                "max_model_calls",
                "explicit_user_statement",
                "approved_at",
                "credential_rotated",
            ],
            "active_file_exists": False,
        },
        "resource_counts": {
            "deepseek_calls": 0,
            "bedrock_calls": 0,
            "parser_calls": 0,
            "s2match_calls": 0,
            "detector_calls": 0,
            "attack_calls": 0,
        },
    }


def build_budget() -> dict:
    module = pure_prompt_module()
    bank_records = load_json(
        swan.OFFICIAL_ROOT / "amr_bank/banks/amr_bank_50.json"
    )
    bank = [row["normalized_template_amr"] for row in bank_records]
    examples = load_json(
        swan.OFFICIAL_ROOT
        / "amr_bank/artifacts/normalized_amr_examples.json"
    )[:3]
    system = module.build_system_prompt()
    generation_reserves = []
    for source in local.pilot_sources():
        maximum = max(
            (
                request_reserve(
                    request_body(
                        module.build_user_prompt(
                            template,
                            examples,
                            context=source["generation_prompt"],
                        ),
                        system,
                        swan.MAX_NEW_TOKENS,
                        swan.TEMPERATURE,
                        swan.DECLARED_TOP_P,
                    )
                )
                for template in bank
            ),
            key=lambda row: row["peak_all_cache_miss_cost_usd"],
        )
        generation_reserves.append(
            {
                "allocation_id": source["allocation_id"],
                **maximum,
                "calls_reserved": MAX_TRIALS,
            }
        )
    probe = request_reserve(
        request_body("Reply exactly OK.", None, 16, 0.0, 1.0)
    )
    bound = probe["peak_all_cache_miss_cost_usd"] + sum(
        row["peak_all_cache_miss_cost_usd"] * MAX_TRIALS
        for row in generation_reserves
    )
    recommended = max(0.01, math.ceil(bound * 1.25 * 100) / 100)
    return {
        "schema_version": 1,
        "status": "estimated_remote_unauthorized",
        "manifest_sha256": sha256(MANIFEST),
        "pricing": {
            "source": PRICE_SOURCE,
            "retrieved_at": PRICE_RETRIEVED,
            "model": MODEL,
            "model_version": "DeepSeek-V4.1-Flash",
            "peak_cache_miss_input_usd_per_million": (
                PEAK_CACHE_MISS_INPUT_USD_PER_MILLION
            ),
            "peak_output_usd_per_million": (
                PEAK_OUTPUT_USD_PER_MILLION
            ),
            "assumption": (
                "Peak pricing and all prompt tokens billed as cache misses."
            ),
        },
        "method": {
            "input_token_bound": (
                "Exact serialized request UTF-8 bytes plus 1,024 tokens"
            ),
            "output_token_bound": "Full requested max_tokens",
            "probe_calls": PROBE_CALLS,
            "generation_calls": GENERATION_CALL_CAP,
            "total_calls": TOTAL_CALL_CAP,
            "largest_official_template_used_for_each_source": True,
        },
        "probe_reserve": probe,
        "generation_reserves": generation_reserves,
        "conservative_total_bound_usd": round(bound, 6),
        "recommended_authorization": {
            "max_cost_usd": recommended,
            "max_model_calls": TOTAL_CALL_CAP,
            "contingency": "25 percent, rounded up to one cent",
        },
        "resource_counts": {
            "deepseek_calls": 0,
            "bedrock_calls": 0,
        },
    }


def authorization_template(budget: dict) -> dict:
    recommended = budget["recommended_authorization"]
    return {
        "schema_version": 1,
        "status": "awaiting_explicit_user_approval",
        "phase": "deepseek_backend_preflight",
        "manifest_sha256": sha256(MANIFEST),
        "budget_estimate_sha256": sha256(BUDGET),
        "model": MODEL,
        "endpoint": URL,
        "max_cost_usd": recommended["max_cost_usd"],
        "max_model_calls": recommended["max_model_calls"],
        "credential_rotated": None,
        "explicit_user_statement": None,
        "approved_at": None,
        "activation": (
            "This template is not authorization. A rotated credential and "
            "explicit user approval are both required."
        ),
    }


def prepare() -> dict:
    if MANIFEST.exists() or BUDGET.exists() or AUTH_TEMPLATE.exists():
        if not all(
            path.is_file() for path in (MANIFEST, BUDGET, AUTH_TEMPLATE)
        ):
            raise RuntimeError("Partial DeepSeek preflight freeze")
        return check()
    manifest = freeze_json(MANIFEST, build_manifest())
    budget = freeze_json(BUDGET, build_budget())
    freeze_json(AUTH_TEMPLATE, authorization_template(budget))
    return {"manifest": manifest, "budget": budget}


def check() -> dict:
    manifest = load_json(MANIFEST)
    if manifest != build_manifest(manifest["created_at"]):
        raise ValueError("DeepSeek preflight manifest differs")
    budget = load_json(BUDGET)
    if budget != build_budget():
        raise ValueError("DeepSeek budget estimate differs")
    if load_json(AUTH_TEMPLATE) != authorization_template(budget):
        raise ValueError("DeepSeek authorization template differs")
    return {"manifest": manifest, "budget": budget}


def require_authorization() -> dict:
    if not AUTHORIZATION.is_file():
        raise PermissionError("DeepSeek preflight is not authorized")
    authorization = load_json(AUTHORIZATION)
    checks = {
        "status": authorization.get("status") == "explicitly_authorized",
        "phase": (
            authorization.get("phase") == "deepseek_backend_preflight"
        ),
        "manifest": (
            authorization.get("manifest_sha256") == sha256(MANIFEST)
        ),
        "budget": (
            authorization.get("budget_estimate_sha256") == sha256(BUDGET)
        ),
        "model": authorization.get("model") == MODEL,
        "endpoint": authorization.get("endpoint") == URL,
        "calls": (
            type(authorization.get("max_model_calls")) is int
            and 0 < authorization["max_model_calls"] <= TOTAL_CALL_CAP
        ),
        "cost": (
            isinstance(authorization.get("max_cost_usd"), (int, float))
            and type(authorization.get("max_cost_usd")) is not bool
            and 0 < authorization["max_cost_usd"]
        ),
        "statement": bool(
            str(authorization.get("explicit_user_statement", "")).strip()
        ),
        "approved_at": bool(
            str(authorization.get("approved_at", "")).strip()
        ),
        "credential_rotated": (
            authorization.get("credential_rotated") is True
        ),
    }
    if not all(checks.values()):
        raise PermissionError("DeepSeek authorization contract failed")
    return authorization


def read_credential(path: Path) -> str:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise PermissionError("Credential file must not be group/world readable")
    keys = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("sk-")
    ]
    if len(keys) != 1:
        raise ValueError("Expected exactly one DeepSeek credential")
    return keys[0]


def default_transport(body: dict, key: str) -> dict:
    import requests

    response = requests.post(
        URL,
        headers={"Authorization": "Bearer " + key},
        json=body,
        timeout=(20, 240),
    )
    if response.status_code != 200:
        raise RuntimeError(f"DeepSeek HTTP {response.status_code}")
    return response.json()


def parse_response(payload: dict) -> dict:
    choice = payload["choices"][0]
    response = {
        "id": payload["id"],
        "model": payload["model"],
        "usage": payload["usage"],
        "text": choice["message"]["content"].strip(),
        "stop_reason": choice["finish_reason"],
        "response_metadata": {
            "provider": "deepseek",
            "response_id": payload["id"],
            "model": payload["model"],
        },
    }
    if (
        response["model"] != RETURNED_MODEL
        or not response["text"]
        or response["stop_reason"] != "stop"
    ):
        raise ValueError("Unexpected DeepSeek response identity or content")
    return response


class DeepSeekJournal:
    def __init__(
        self,
        folder: Path,
        key: str,
        authorization: dict,
        transport: Optional[Callable[[dict, str], dict]] = None,
    ):
        self.folder = folder
        self.key = key
        self.authorization = authorization
        self.transport = transport or default_transport

    def inventory(self) -> list:
        return sorted((self.folder / "deepseek_calls").glob("*.json"))

    def completed_cost(self) -> float:
        total = 0.0
        for path in self.inventory():
            saved = load_json(path)
            if saved["status"] != "complete":
                raise RuntimeError(
                    f"Uncertain call blocks continuation: {path.name}"
                )
            total += saved["cost"]["actual"][
                "conservative_peak_cost_usd"
            ]
        return total

    def request(
        self,
        call_id: str,
        user_text: str,
        system_text: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict:
        body = request_body(
            user_text,
            system_text,
            max_tokens,
            temperature,
            top_p,
        )
        request_hash = digest(body)
        path = self.folder / "deepseek_calls" / f"{call_id}.json"
        if path.exists():
            saved = load_json(path)
            if (
                saved["request_sha256"] != request_hash
                or saved["request"] != body
                or saved["status"] != "complete"
            ):
                raise RuntimeError("DeepSeek call is not safely replayable")
            return {
                **saved["response"],
                "call_sha256": sha256(path),
                "replayed": True,
            }
        if len(self.inventory()) >= self.authorization["max_model_calls"]:
            raise RuntimeError("Approved DeepSeek call cap reached")
        reserve = request_reserve(body)
        completed = self.completed_cost()
        if (
            completed + reserve["peak_all_cache_miss_cost_usd"]
            > self.authorization["max_cost_usd"] + 1e-12
        ):
            raise RuntimeError("Approved DeepSeek USD cap would be exceeded")
        saved = {
            "schema_version": 1,
            "call_id": call_id,
            "request_sha256": request_hash,
            "request": body,
            "status": "in_flight",
            "transport_attempts": 1,
            "cost": {
                "completed_before_call_usd": round(completed, 12),
                "reserve": reserve,
                "actual": None,
            },
            "started_at": utc_now(),
        }
        write_json(path, saved)
        try:
            response = parse_response(self.transport(body, self.key))
            actual = actual_cost(response["usage"])
            if (
                actual["conservative_peak_cost_usd"]
                > reserve["peak_all_cache_miss_cost_usd"] + 1e-12
            ):
                raise ValueError("DeepSeek actual cost exceeds reserve")
            saved["cost"]["actual"] = actual
            saved.update(status="complete", response=response)
        except Exception as exc:
            saved.update(
                status="failed_or_uncertain",
                error_type=type(exc).__name__,
            )
            raise
        finally:
            write_json(path, saved)
        return {
            **response,
            "call_sha256": sha256(path),
            "replayed": False,
        }


def summarize() -> dict:
    state = check()
    slots = sorted((OUTPUT / "generation_slots").glob("*.json"))
    rows = [load_json(path) for path in slots]
    accepted = sum(row["accepted"] for row in rows)
    complete = len(rows) == SOURCES * SLOTS
    calls = sorted((OUTPUT / "deepseek_calls").glob("*.json"))
    result = {
        "schema_version": 1,
        "status": "complete" if complete else "remote_unauthorized",
        "model": MODEL,
        "allocated_slots": SOURCES * SLOTS,
        "completed_slots": len(rows),
        "accepted_slots": accepted,
        "attempts": sum(row["attempts"] for row in rows),
        "deepseek_calls": len(calls),
        "decision": (
            "prepare_separate_deepseek_pilot"
            if complete and accepted >= CONTINUE_ACCEPTED
            else "stop_deepseek_route"
            if complete
            else "pending_authorization"
        ),
        "continue_gate": {
            "required": CONTINUE_ACCEPTED,
            "observed": accepted,
            "passed": complete and accepted >= CONTINUE_ACCEPTED,
        },
        "manifest_sha256": sha256(MANIFEST),
        "budget_estimate_sha256": sha256(BUDGET),
        "claim_boundary": (
            "Backend feasibility only; not official Claude equivalence, "
            "not a formal positive set, and no TPR or ASR."
        ),
        "resource_counts": {
            **state["manifest"]["resource_counts"],
            "deepseek_calls": len(calls),
            "parser_calls": sum(row["attempts"] for row in rows),
            "s2match_calls": sum(row["attempts"] for row in rows),
        },
    }
    write_json(OUTPUT / "summary.json", result)
    return result


def run(key_file: Path) -> dict:
    check()
    authorization = require_authorization()
    key = read_credential(key_file)
    journal = DeepSeekJournal(OUTPUT, key, authorization)
    probe = journal.request(
        "model-probe",
        "Reply exactly OK.",
        None,
        16,
        0.0,
        1.0,
    )
    if probe["text"] != "OK":
        raise ValueError("DeepSeek model probe failed")

    module = swan.load_official_injection()
    bank_records = load_json(
        swan.OFFICIAL_ROOT / "amr_bank/banks/amr_bank_50.json"
    )
    bank = [row["normalized_template_amr"] for row in bank_records]
    examples = (
        swan.OFFICIAL_ROOT
        / "amr_bank/artifacts/normalized_amr_examples.json"
    )
    original = {
        "CONTRACT": swan.CONTRACT,
        "MAX_TRIALS": swan.MAX_TRIALS,
        "MAX_TEMPLATES": swan.MAX_TEMPLATES,
        "RETRIES": swan.RETRIES_PER_TEMPLATE,
    }
    swan.CONTRACT = MANIFEST
    swan.MAX_TRIALS = MAX_TRIALS
    swan.MAX_TEMPLATES = MAX_TEMPLATES
    swan.RETRIES_PER_TEMPLATE = RETRIES_PER_TEMPLATE
    try:
        for source in local.pilot_sources():
            swan.run_official_slot(
                module,
                OUTPUT,
                source,
                0,
                source["generation_prompt"],
                bank,
                examples,
                journal,
            )
    finally:
        swan.CONTRACT = original["CONTRACT"]
        swan.MAX_TRIALS = original["MAX_TRIALS"]
        swan.MAX_TEMPLATES = original["MAX_TEMPLATES"]
        swan.RETRIES_PER_TEMPLATE = original["RETRIES"]
    return summarize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("prepare", "check", "run", "summarize"),
    )
    parser.add_argument("--key-file", type=Path, default=ROOT / "api.txt")
    args = parser.parse_args()
    if args.stage == "prepare":
        result = prepare()
    elif args.stage == "check":
        result = check()
    elif args.stage == "run":
        result = run(args.key_file)
    else:
        result = summarize()
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
