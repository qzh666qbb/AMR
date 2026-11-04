"""Checkpointed execution wrapper for the official SWAN injection function.

The default ``check`` command is offline. Paid Bedrock calls require a frozen
authorization artifact and an explicit positive per-invocation call limit.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import ModuleType
from typing import Callable, Optional


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/swan_positive_collection1"
OFFICIAL_ROOT = ROOT / "third_party/SWAN_official_2ca0c207_clean"
OFFICIAL_INJECTION = OFFICIAL_ROOT / "injection/injection_amr_utils.py"
S2MATCH_RUNTIME = ROOT / "third_party/swan_s2match_runtime_711a231d"
PARSER_MODEL = ROOT / "models/model_parse_xfm_bart_large-v0_1_0"
PARSER_TOKENIZER = ROOT / "models/bart-large-tokenizer"
GLOVE_VECTORS = ROOT / "vectors/glove.6B.100d.txt"
NLTK_DATA = ROOT / ".nltk_data"
SOURCE_MANIFEST = OUTPUT / "source_manifest.json"
RUNTIME_V2 = OUTPUT / "runtime_v2.json"
CONTRACT = OUTPUT / "execution_contract_v3.json"
CONTRACT_AUDIT = OUTPUT / "execution_contract_audit_v3.json"
AUTHORIZATION = OUTPUT / "budget_authorization.json"
PROFILE_EVIDENCE = OUTPUT / "model_profile_v3.json"
MODEL_PROBE = OUTPUT / "model_probe_v3.json"

MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
REGION = "us-east-1"
EXPECTED_COMMIT = "2ca0c2071c8d0c6bb3739b2b2bcfb34cfe63eb2a"
SLOTS = 16
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.7
DECLARED_TOP_P = 0.9
MAX_TRIALS = 50
MAX_TEMPLATES = 10
RETRIES_PER_TEMPLATE = 5
MATCH_THRESHOLD = 0.75
GLOBAL_GENERATION_CAP = 35 * SLOTS * MAX_TRIALS
PILOT_GENERATION_CAP = 5 * SLOTS * MAX_TRIALS
INPUT_PRICE_PER_MILLION_USD = 3.30
OUTPUT_PRICE_PER_MILLION_USD = 16.50
INPUT_TOKEN_OVERHEAD_RESERVE = 1024


class InvocationBudgetExhausted(RuntimeError):
    """The approved or per-invocation paid-call budget was exhausted."""


class UncertainCallExists(RuntimeError):
    """A transmitted call lacks a definitely complete response."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def digest(value) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
            raise ValueError(f"Frozen artifact differs: {path}")
        return saved
    write_json(path, payload)
    return payload


def slot_seed(allocation_id: str, slot: int) -> int:
    return int(
        hashlib.sha256(
            (
                "swan-positive-collection1:template:"
                f"{allocation_id}:{slot}"
            ).encode("utf-8")
        ).hexdigest()[:16],
        16,
    )


def score_seed(allocation_id: str, slot: int, attempt: int) -> int:
    return int(
        hashlib.sha256(
            (
                "swan-positive-collection1:s2match:"
                f"{allocation_id}:{slot}:{attempt}"
            ).encode("utf-8")
        ).hexdigest()[:8],
        16,
    )


def anthropic_request_body(
    user_text: str,
    system_text: Optional[str],
    max_tokens: int,
    temperature: float,
    declared_top_p: float,
    model_id: str = MODEL,
) -> dict:
    body = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": declared_top_p,
        "messages": [{"role": "user", "content": user_text}],
        "anthropic_version": "bedrock-2023-05-31",
    }
    if (
        "claude-sonnet-4-5" in model_id
        or "claude-haiku-4-5" in model_id
    ):
        body.pop("top_p")
    if system_text:
        body["system"] = system_text
    return body


def _response_metadata(response: dict) -> dict:
    metadata = response.get("ResponseMetadata", {})
    return {
        "request_id": metadata.get("RequestId"),
        "http_status_code": metadata.get("HTTPStatusCode"),
        "content_type": response.get("contentType"),
        "performance_config_latency": response.get(
            "performanceConfigLatency"
        ),
    }


def _read_response_body(response: dict) -> dict:
    stream = response["body"]
    raw = stream.read() if hasattr(stream, "read") else stream
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if not isinstance(raw, bytes):
        raise TypeError("Bedrock response body is not bytes")
    return json.loads(raw.decode("utf-8"))


def parse_anthropic_response(response: dict) -> dict:
    body = _read_response_body(response)
    content = body.get("content")
    if not isinstance(content, list):
        raise ValueError("Anthropic response content is not a list")
    pieces = []
    for item in content:
        if not isinstance(item, dict):
            raise ValueError("Anthropic response content item is invalid")
        text = item.get("text", "")
        if not isinstance(text, str):
            raise ValueError("Anthropic response text is invalid")
        pieces.append(text)
    text = " ".join(pieces).strip()
    if not text:
        raise ValueError("Anthropic response text is empty")
    return {
        "raw_body": body,
        "text": text,
        "text_sha256": text_sha256(text),
        "provider_reported_model": body.get("model"),
        "stop_reason": body.get("stop_reason"),
        "usage": body.get("usage", {}),
        "response_metadata": _response_metadata(response),
    }


def default_transport(region: str, request: dict) -> dict:
    import boto3

    client = boto3.client("bedrock-runtime", region_name=region)
    return client.invoke_model(
        body=json.dumps(request["body"]),
        modelId=request["model_id"],
        accept="application/json",
        contentType="application/json",
    )


def request_cost_reserve(request: dict, max_tokens: int) -> dict:
    body_bytes = len(json.dumps(request["body"]).encode("utf-8"))
    input_token_bound = body_bytes + INPUT_TOKEN_OVERHEAD_RESERVE
    input_cost = (
        input_token_bound * INPUT_PRICE_PER_MILLION_USD / 1_000_000
    )
    output_cost = (
        max_tokens * OUTPUT_PRICE_PER_MILLION_USD / 1_000_000
    )
    return {
        "pricing_basis": "US geographic-profile conservative bound",
        "input_price_per_million_usd": INPUT_PRICE_PER_MILLION_USD,
        "output_price_per_million_usd": OUTPUT_PRICE_PER_MILLION_USD,
        "request_body_bytes": body_bytes,
        "input_token_bound": input_token_bound,
        "output_token_bound": max_tokens,
        "reserved_cost_usd": round(input_cost + output_cost, 12),
    }


def actual_response_cost(response: dict) -> dict:
    usage = response["usage"]
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if (
        type(input_tokens) is not int
        or input_tokens < 0
        or type(output_tokens) is not int
        or output_tokens < 0
    ):
        raise ValueError("Provider response lacks valid token usage")
    input_cost = input_tokens * INPUT_PRICE_PER_MILLION_USD / 1_000_000
    output_cost = (
        output_tokens * OUTPUT_PRICE_PER_MILLION_USD / 1_000_000
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_billed_cost_usd": round(
            input_cost + output_cost,
            12,
        ),
    }


class BedrockCallJournal:
    """Single-transport-attempt, replayable Bedrock call journal."""

    def __init__(
        self,
        folder: Path,
        max_new_calls: int,
        approved_call_cap: int,
        approved_cost_usd: float,
        transport: Optional[Callable[[str, dict], dict]] = None,
    ):
        if max_new_calls < 0:
            raise ValueError("max_new_calls must be nonnegative")
        if approved_call_cap < 0:
            raise ValueError("approved_call_cap must be nonnegative")
        if approved_cost_usd <= 0 or not math.isfinite(approved_cost_usd):
            raise ValueError("approved_cost_usd must be positive and finite")
        self.folder = folder
        self.max_new_calls = max_new_calls
        self.approved_call_cap = approved_call_cap
        self.approved_cost_usd = approved_cost_usd
        self.transport = transport or default_transport
        self.new_calls = 0

    def inventory(self) -> list:
        return sorted((self.folder / "calls").glob("*.json"))

    def completed_cost_usd(self) -> float:
        total = 0.0
        for path in self.inventory():
            saved = load_json(path)
            if saved["status"] == "complete":
                total += saved["cost"]["actual"][
                    "estimated_billed_cost_usd"
                ]
            else:
                raise UncertainCallExists(
                    f"Unfinished call blocks budgeting: {path.name}"
                )
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
        body = anthropic_request_body(
            user_text,
            system_text,
            max_tokens,
            temperature,
            top_p,
        )
        request = {
            "model_id": MODEL,
            "region": REGION,
            "accept": "application/json",
            "content_type": "application/json",
            "declared_top_p": top_p,
            "effective_top_p": body.get("top_p"),
            "body": body,
        }
        request_hash = digest(request)
        reserve = request_cost_reserve(request, max_tokens)
        path = self.folder / "calls" / f"{call_id}.json"
        if path.exists():
            saved = load_json(path)
            if (
                saved["call_id"] != call_id
                or saved["request_sha256"] != request_hash
                or saved["request"] != request
            ):
                raise ValueError(f"Replay request changed: {call_id}")
            if saved["status"] != "complete":
                raise UncertainCallExists(
                    f"Call is not safely replayable: {path.name}"
                )
            return {
                **saved["response"],
                "cost": saved["cost"],
                "call_sha256": sha256(path),
                "replayed": True,
            }

        existing = len(self.inventory())
        if existing >= GLOBAL_GENERATION_CAP + 1:
            raise InvocationBudgetExhausted(
                "Frozen global model-call cap reached"
            )
        if existing >= self.approved_call_cap:
            raise InvocationBudgetExhausted(
                "Frozen approved model-call cap reached"
            )
        if self.new_calls >= self.max_new_calls:
            raise InvocationBudgetExhausted(
                "Per-invocation model-call cap reached"
            )
        completed_cost = self.completed_cost_usd()
        if (
            completed_cost + reserve["reserved_cost_usd"]
            > self.approved_cost_usd + 1e-12
        ):
            raise InvocationBudgetExhausted(
                "Frozen approved US-dollar budget would be exceeded"
            )

        saved = {
            "schema_version": 1,
            "call_id": call_id,
            "request_sha256": request_hash,
            "request": request,
            "status": "in_flight",
            "transport_attempts": 1,
            "cost": {
                "approved_max_cost_usd": self.approved_cost_usd,
                "completed_cost_before_call_usd": round(
                    completed_cost,
                    12,
                ),
                "reserve": reserve,
                "actual": None,
            },
            "started_at": utc_now(),
        }
        write_json(path, saved)
        self.new_calls += 1
        started = time.monotonic()
        try:
            raw_response = self.transport(REGION, request)
            response = parse_anthropic_response(raw_response)
            actual = actual_response_cost(response)
            if (
                actual["estimated_billed_cost_usd"]
                > reserve["reserved_cost_usd"] + 1e-12
            ):
                raise ValueError("Actual token cost exceeds frozen reserve")
            saved["cost"]["actual"] = actual
            saved.update(status="complete", response=response)
        except Exception as exc:
            saved.update(
                status="failed_or_uncertain",
                error_type=type(exc).__name__,
            )
            raise UncertainCallExists(
                f"Bedrock call stopped; inspect {path.name}"
            ) from None
        finally:
            saved["elapsed_seconds"] = time.monotonic() - started
            write_json(path, saved)
        return {
            **response,
            "cost": saved["cost"],
            "call_sha256": sha256(path),
            "replayed": False,
        }


@contextmanager
def fixed_score_seed(seed: int):
    import numpy as np

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    original_seed = random.seed
    original_seed(seed)
    np.random.seed(seed)

    def preserve_explicit_seed(value=None, version=2):
        if value is not None:
            original_seed(value, version=version)

    random.seed = preserve_explicit_seed
    try:
        yield
    finally:
        random.seed = original_seed
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class ChoiceRecorder:
    def __init__(self, seed: int, state: dict, bank: list):
        self.rng = random.Random(seed)
        self.state = state
        self.bank = bank

    def choice(self, values):
        chosen = self.rng.choice(values)
        try:
            index = self.bank.index(chosen)
        except ValueError as exc:
            raise ValueError("Official injection chose a non-bank template") from exc
        self.state["current_template"] = chosen
        self.state["current_template_index"] = index
        self.state["templates_chosen"].append(
            {
                "template_index": index,
                "template_sha256": text_sha256(chosen),
            }
        )
        return chosen


@contextmanager
def instrument_official_injection(
    module,
    folder: Path,
    allocation_id: str,
    slot: int,
    bank: list,
    journal: BedrockCallJournal,
):
    names = (
        "BedrockManager",
        "random",
        "parse_amr",
        "decide_s2match_match",
    )
    saved = {name: getattr(module, name) for name in names}
    state = {
        "attempts": [],
        "templates_chosen": [],
        "current_template": None,
        "current_template_index": None,
        "current_event": None,
    }

    class JournalManager:
        def __init__(self, region: str, model_id: str):
            if region != REGION or model_id != MODEL:
                raise ValueError("Official injection changed model or region")

        def generate(
            self,
            user_text: str,
            system_text: Optional[str],
            max_tokens: int,
            temperature: float,
            top_p: float,
        ) -> str:
            attempt = len(state["attempts"])
            call_id = (
                f"{allocation_id}-slot-{slot:02d}-attempt-{attempt:02d}"
            )
            event = {
                "allocation_id": allocation_id,
                "slot": slot,
                "attempt": attempt,
                "call_id": call_id,
                "template_index": state["current_template_index"],
                "template": state["current_template"],
                "template_sha256": text_sha256(state["current_template"]),
                "user_prompt_sha256": text_sha256(user_text),
                "system_prompt_sha256": text_sha256(system_text or ""),
                "status": "requesting",
            }
            state["attempts"].append(event)
            state["current_event"] = event
            trace_path = (
                folder
                / "generation_attempts"
                / f"{call_id}.json"
            )
            try:
                response = journal.request(
                    call_id,
                    user_text,
                    system_text,
                    max_tokens,
                    temperature,
                    top_p,
                )
            except Exception as exc:
                event.update(
                    status="transport_failed_or_uncertain",
                    error_type=type(exc).__name__,
                )
                freeze_json(trace_path, event)
                raise
            event.update(
                status="response_complete_pending_parse",
                call_sha256=response["call_sha256"],
                candidate=response["text"],
                candidate_sha256=response["text_sha256"],
                stop_reason=response["stop_reason"],
                usage=response["usage"],
                response_metadata=response["response_metadata"],
            )
            return response["text"]

    def parse_candidate(candidate: str):
        event = state["current_event"]
        try:
            raw_amr = saved["parse_amr"](candidate)
        except Exception as exc:
            event.update(
                status="parse_failed",
                error_type=type(exc).__name__,
            )
            freeze_json(
                folder
                / "generation_attempts"
                / f"{event['call_id']}.json",
                event,
            )
            raise
        event["raw_amr"] = raw_amr
        event["raw_amr_sha256"] = text_sha256(raw_amr)
        return raw_amr

    def match_candidate(amr: str, target: str, threshold: float):
        event = state["current_event"]
        if target != event["template"]:
            raise ValueError("Match target differs from sampled template")
        seed = score_seed(allocation_id, slot, event["attempt"])
        try:
            with fixed_score_seed(seed):
                accepted, score = saved["decide_s2match_match"](
                    amr,
                    target,
                    threshold,
                )
        except Exception as exc:
            event.update(
                status="s2match_failed",
                score_seed=seed,
                error_type=type(exc).__name__,
            )
            freeze_json(
                folder
                / "generation_attempts"
                / f"{event['call_id']}.json",
                event,
            )
            raise
        score = float(score)
        if (
            not math.isfinite(score)
            or score < 0
            or score > 1
            or bool(accepted) != (score >= threshold)
        ):
            raise ValueError("Invalid official S2Match decision")
        event.update(
            status="complete",
            candidate_template_amr=amr,
            candidate_template_amr_sha256=text_sha256(amr),
            score_seed=seed,
            threshold=threshold,
            score=score,
            accepted=bool(accepted),
        )
        freeze_json(
            folder
            / "generation_attempts"
            / f"{event['call_id']}.json",
            event,
        )
        return bool(accepted), score

    module.BedrockManager = JournalManager
    module.random = ChoiceRecorder(
        slot_seed(allocation_id, slot),
        state,
        bank,
    )
    module.parse_amr = parse_candidate
    module.decide_s2match_match = match_candidate
    try:
        yield state
    finally:
        for name, value in saved.items():
            setattr(module, name, value)


def run_official_slot(
    module,
    folder: Path,
    source: dict,
    slot: int,
    context: str,
    bank: list,
    examples_path: Path,
    journal: BedrockCallJournal,
) -> dict:
    allocation_id = source["allocation_id"]
    slot_path = (
        folder / "generation_slots" / f"{allocation_id}-slot-{slot:02d}.json"
    )
    identity = {
        "allocation_id": allocation_id,
        "slot": slot,
        "context_sha256": text_sha256(context),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "execution_contract_sha256": sha256(CONTRACT),
    }
    if slot_path.exists():
        saved = load_json(slot_path)
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError(f"Slot identity changed: {slot_path.name}")
        return saved

    failure_path = (
        folder
        / "generation_failures"
        / f"{allocation_id}-slot-{slot:02d}.json"
    )
    if failure_path.exists():
        raise RuntimeError(
            f"Frozen slot failure requires manual audit: {failure_path.name}"
        )

    with instrument_official_injection(
        module,
        folder,
        allocation_id,
        slot,
        bank,
        journal,
    ) as trace:
        try:
            text, attempts, template = module.amr_instruction_completion(
                context=context,
                amr_bank=bank,
                model_id=MODEL,
                region=REGION,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=DECLARED_TOP_P,
                max_trials=MAX_TRIALS,
                amr_examples_path=str(examples_path),
                s2match_threshold=MATCH_THRESHOLD,
                max_templates_to_try=MAX_TEMPLATES,
                max_retries_per_template=RETRIES_PER_TEMPLATE,
            )
        except Exception as exc:
            freeze_json(
                failure_path,
                {
                    **identity,
                    "status": "failed_or_uncertain",
                    "error_type": type(exc).__name__,
                    "attempts_recorded": len(trace["attempts"]),
                    "attempt_trace_sha256": [
                        sha256(
                            folder
                            / "generation_attempts"
                            / f"{event['call_id']}.json"
                        )
                        for event in trace["attempts"]
                        if (
                            folder
                            / "generation_attempts"
                            / f"{event['call_id']}.json"
                        ).is_file()
                    ],
                },
            )
            raise

    events = trace["attempts"]
    if attempts != len(events):
        raise ValueError("Official attempt count differs from trace")
    if any(event["status"] != "complete" for event in events):
        raise ValueError("A completed slot contains an incomplete attempt")
    accepted = bool(events and events[-1]["accepted"])
    selected = events[-1] if events else None
    if text and (
        selected is None
        or selected["candidate"].strip() != text.strip()
        or selected["template"] != template
    ):
        raise ValueError("Official fallback/acceptance output differs from trace")
    payload = {
        **identity,
        "status": "accepted" if accepted else "exhausted",
        "template_seed": slot_seed(allocation_id, slot),
        "attempts": attempts,
        "templates_chosen": trace["templates_chosen"],
        "attempt_trace_sha256": [
            sha256(
                folder
                / "generation_attempts"
                / f"{event['call_id']}.json"
            )
            for event in events
        ],
        "text": text.strip(),
        "text_sha256": text_sha256(text.strip()),
        "chosen_template": template,
        "chosen_template_sha256": text_sha256(template),
        "score": selected["score"] if selected else None,
        "accepted": accepted,
        "fallback": bool(text and not accepted),
    }
    return freeze_json(slot_path, payload)


def load_official_injection() -> ModuleType:
    head = subprocess.run(
        ["git", "-C", str(OFFICIAL_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "-C",
            str(OFFICIAL_ROOT),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != EXPECTED_COMMIT or status:
        raise ValueError("Official SWAN checkout is not exact and clean")
    if "utils.amr_utils" in sys.modules:
        raise RuntimeError("Refusing ambiguous preloaded SWAN utils module")
    required = (
        S2MATCH_RUNTIME / "s2match.py",
        S2MATCH_RUNTIME / "amr_py3.py",
        S2MATCH_RUNTIME / "helpers.py",
        PARSER_MODEL / "pytorch_model.bin",
        PARSER_TOKENIZER / "tokenizer.json",
        GLOVE_VECTORS,
        NLTK_DATA / "corpora/propbank.zip",
        NLTK_DATA / "tokenizers/punkt_tab/english/ortho_context.tab",
    )
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("Frozen SWAN runtime resource is missing")
    os.environ["GLOVE_VECTORS_PATH"] = str(GLOVE_VECTORS)
    os.environ["NLTK_DATA"] = str(NLTK_DATA)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import amrlib
    import nltk

    nltk_path = str(NLTK_DATA)
    if nltk_path not in nltk.data.path:
        nltk.data.path.insert(0, nltk_path)
    sys.path.insert(0, str(S2MATCH_RUNTIME))
    sys.path.insert(0, str(OFFICIAL_ROOT))
    original_loader = amrlib.load_stog_model

    def load_frozen_parser(*args, **kwargs):
        if args or kwargs:
            return original_loader(*args, **kwargs)
        return original_loader(
            model_dir=str(PARSER_MODEL),
            tok_name_or_path=str(PARSER_TOKENIZER),
            device="cpu",
            batch_size=1,
        )

    amrlib.load_stog_model = load_frozen_parser
    spec = importlib.util.spec_from_file_location(
        "swan_official_injection_amr_utils",
        OFFICIAL_INJECTION,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load official SWAN injection module")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        amrlib.load_stog_model = original_loader
    return module


def official_function_module_for_test(bindings: dict) -> ModuleType:
    import ast
    import logging

    source = OFFICIAL_INJECTION.read_text(encoding="utf-8")
    names = {
        "load_amr_examples",
        "build_system_prompt",
        "build_user_prompt",
        "amr_instruction_completion",
    }
    nodes = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in nodes} != names:
        raise ValueError("Official injection function inventory changed")
    module = ModuleType("official_swan_injection_test")
    module.__dict__.update(
        json=json,
        random=random,
        logger=logging.getLogger("official_swan_injection_test"),
        **bindings,
    )
    code = compile(
        ast.Module(body=nodes, type_ignores=[]),
        str(OFFICIAL_INJECTION),
        "exec",
    )
    exec(code, module.__dict__)
    return module


def require_authorization(folder: Path, phase: str) -> dict:
    path = folder / AUTHORIZATION.name
    if not path.is_file():
        raise PermissionError(
            "Paid execution is not authorized; budget_authorization.json "
            "is absent"
        )
    authorization = load_json(path)
    required = {
        "status": "explicitly_authorized",
        "phase": phase,
        "model_id": MODEL,
        "region": REGION,
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "execution_contract_sha256": sha256(CONTRACT),
    }
    if any(authorization.get(key) != value for key, value in required.items()):
        raise PermissionError("Budget authorization does not match execution")
    if (
        type(authorization.get("max_model_calls")) is not int
        or authorization["max_model_calls"] < 1
        or not isinstance(authorization.get("max_cost_usd"), (int, float))
        or authorization["max_cost_usd"] <= 0
    ):
        raise PermissionError("Budget authorization limits are invalid")
    return authorization


def require_ready_contract() -> dict:
    if not CONTRACT.is_file():
        raise RuntimeError("Execution contract is not frozen")
    if not CONTRACT_AUDIT.is_file():
        raise RuntimeError("Independent execution-contract audit is missing")
    contract = load_json(CONTRACT)
    audit = load_json(CONTRACT_AUDIT)
    if (
        contract.get("status") != "frozen_execution_contract"
        or contract.get("execution_ready") is not True
        or contract.get("runner_sha256") != sha256(Path(__file__))
        or contract.get("source_manifest_sha256")
        != sha256(SOURCE_MANIFEST)
        or str(Path(sys.prefix).resolve())
        != str(
            Path(
                contract.get("python_environment", {}).get(
                    "prefix",
                    "",
                )
            ).resolve()
        )
        or audit.get("status")
        != "verified_local_execution_ready_remote_unauthorized"
        or audit.get("all_checks_pass") is not True
        or audit.get("execution_contract_sha256") != sha256(CONTRACT)
    ):
        raise RuntimeError("Execution contract is not ready")
    return contract


def _json_safe(value):
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def fetch_profile_evidence(
    folder: Path,
    transport: Optional[Callable[[str, str], dict]] = None,
) -> dict:
    path = folder / PROFILE_EVIDENCE.name
    request = {
        "service": "bedrock",
        "operation": "GetInferenceProfile",
        "region": REGION,
        "inference_profile_identifier": MODEL,
    }
    request_hash = digest(request)
    if path.exists():
        saved = load_json(path)
        if saved["request_sha256"] != request_hash:
            raise ValueError("Inference-profile request changed")
        if saved["status"] != "complete":
            raise UncertainCallExists(
                "Inference-profile lookup is not safely replayable"
            )
        return saved

    if transport is None:
        import boto3

        def transport(region: str, identifier: str) -> dict:
            client = boto3.client("bedrock", region_name=region)
            return client.get_inference_profile(
                inferenceProfileIdentifier=identifier
            )

    saved = {
        "schema_version": 1,
        "request_sha256": request_hash,
        "request": request,
        "status": "in_flight",
        "transport_attempts": 1,
        "started_at": utc_now(),
    }
    write_json(path, saved)
    started = time.monotonic()
    try:
        response = _json_safe(transport(REGION, MODEL))
        if response.get("inferenceProfileId") != MODEL:
            raise ValueError("Returned inference-profile ID differs")
        saved.update(status="complete", response=response)
    except Exception as exc:
        saved.update(
            status="failed_or_uncertain",
            error_type=type(exc).__name__,
        )
        raise UncertainCallExists(
            "Inference-profile lookup stopped; inspect its journal"
        ) from None
    finally:
        saved["elapsed_seconds"] = time.monotonic() - started
        write_json(path, saved)
    return saved


def run_model_probe(
    folder: Path,
    max_new_calls: int,
    profile_transport: Optional[Callable[[str, str], dict]] = None,
    model_transport: Optional[Callable[[str, dict], dict]] = None,
) -> dict:
    require_ready_contract()
    authorization = require_authorization(folder, "pilot_collection")
    if max_new_calls < 1:
        raise InvocationBudgetExhausted(
            "Probe requires an explicit positive --max-new-calls"
        )
    probe_path = folder / MODEL_PROBE.name
    if probe_path.is_file():
        saved = load_json(probe_path)
        if (
            saved["execution_contract_sha256"] != sha256(CONTRACT)
            or saved["model_id_requested"] != MODEL
        ):
            raise ValueError("Frozen model probe provenance changed")
        return saved

    profile = fetch_profile_evidence(folder, profile_transport)
    journal = BedrockCallJournal(
        folder,
        max_new_calls=1,
        approved_call_cap=authorization["max_model_calls"],
        approved_cost_usd=float(authorization["max_cost_usd"]),
        transport=model_transport,
    )
    response = journal.request(
        "model-probe",
        "Reply exactly OK.",
        None,
        16,
        0.0,
        DECLARED_TOP_P,
    )
    payload = {
        "schema_version": 3,
        "status": "complete",
        "execution_contract_sha256": sha256(CONTRACT),
        "authorization_sha256": sha256(folder / AUTHORIZATION.name),
        "profile_evidence_sha256": sha256(
            folder / PROFILE_EVIDENCE.name
        ),
        "call_sha256": response["call_sha256"],
        "model_id_requested": MODEL,
        "profile_id_returned_by_control_plane": profile["response"][
            "inferenceProfileId"
        ],
        "provider_reported_model": response[
            "provider_reported_model"
        ],
        "aws_request_id": response["response_metadata"]["request_id"],
        "response_text": response["text"],
        "response_text_sha256": response["text_sha256"],
        "identity_interpretation": (
            "InvokeModel has no top-level returned modelId. Evidence proves "
            "that the request named the frozen inference profile and records "
            "the control-plane profile plus provider response metadata; it "
            "does not claim an independently returned execution identity."
        ),
    }
    return freeze_json(probe_path, payload)


def check_contract(folder: Path = OUTPUT) -> dict:
    runtime = load_json(RUNTIME_V2)
    source = load_json(SOURCE_MANIFEST)
    if (
        runtime["official_swan"]["commit"] != EXPECTED_COMMIT
        or runtime["source_lineage"]["source_manifest_sha256"]
        != sha256(SOURCE_MANIFEST)
        or len(source["sources"]) != 35
        or source["cohort_counts"] != {"pilot": 5, "main": 30}
    ):
        raise ValueError("Frozen source or official provenance changed")
    contract_status = "not_frozen"
    if CONTRACT.is_file():
        contract = load_json(CONTRACT)
        if (
            contract["source_manifest_sha256"] != sha256(SOURCE_MANIFEST)
            or contract["official_injection_sha256"]
            != sha256(OFFICIAL_INJECTION)
            or contract["runner_sha256"] != sha256(Path(__file__))
        ):
            raise ValueError("Execution contract provenance changed")
        contract_status = contract["status"]
    return {
        "status": "offline_check_complete",
        "contract_status": contract_status,
        "sources": len(source["sources"]),
        "pilot": source["cohort_counts"]["pilot"],
        "main": source["cohort_counts"]["main"],
        "authorization_present": (folder / AUTHORIZATION.name).is_file(),
        "profile_evidence_present": (
            folder / PROFILE_EVIDENCE.name
        ).is_file(),
        "model_probe_present": (folder / MODEL_PROBE.name).is_file(),
        "model_calls": len(list((folder / "calls").glob("*.json"))),
        "generation_slots": len(
            list((folder / "generation_slots").glob("*.json"))
        ),
    }


def run_pilot(folder: Path, max_new_calls: int) -> dict:
    require_ready_contract()
    authorization = require_authorization(folder, "pilot_collection")
    if not (folder / MODEL_PROBE.name).is_file():
        raise RuntimeError("Frozen model probe is required before generation")
    probe = load_json(folder / MODEL_PROBE.name)
    if (
        probe["execution_contract_sha256"] != sha256(CONTRACT)
        or probe["authorization_sha256"]
        != sha256(folder / AUTHORIZATION.name)
    ):
        raise ValueError("Model probe does not match pilot authorization")
    if max_new_calls < 1:
        raise InvocationBudgetExhausted(
            "Pilot requires an explicit positive --max-new-calls"
        )
    journal = BedrockCallJournal(
        folder,
        min(max_new_calls, authorization["max_model_calls"]),
        authorization["max_model_calls"],
        float(authorization["max_cost_usd"]),
    )
    module = load_official_injection()
    source_manifest = load_json(SOURCE_MANIFEST)
    bank_path = OFFICIAL_ROOT / "amr_bank/banks/amr_bank_50.json"
    examples_path = (
        OFFICIAL_ROOT
        / "amr_bank/artifacts/normalized_amr_examples.json"
    )
    bank_records = load_json(bank_path)
    bank = [
        row["normalized_template_amr"]
        for row in bank_records
        if "normalized_template_amr" in row
    ]
    if len(bank) != 50 or len(set(bank)) != 50:
        raise ValueError("Official bank50 size changed")
    pilot = [
        source
        for source in source_manifest["sources"]
        if source["cohort"] == "pilot"
    ]
    if len(pilot) != 5:
        raise ValueError("Frozen pilot membership changed")

    for source in pilot:
        document_path = (
            folder
            / "generated_documents"
            / f"{source['allocation_id']}.json"
        )
        if document_path.exists():
            continue
        slots = []
        prompt = source["generation_prompt"]
        for slot in range(SLOTS):
            context = prompt + (
                (" " + " ".join(row["text"] for row in slots))
                if slots
                else ""
            )
            slots.append(
                run_official_slot(
                    module,
                    folder,
                    source,
                    slot,
                    context,
                    bank,
                    examples_path,
                    journal,
                )
            )
        text = " ".join(row["text"] for row in slots)
        freeze_json(
            document_path,
            {
                "allocation_id": source["allocation_id"],
                "source_manifest_sha256": sha256(SOURCE_MANIFEST),
                "execution_contract_sha256": sha256(CONTRACT),
                "slot_sha256": [
                    sha256(
                        folder
                        / "generation_slots"
                        / (
                            f"{source['allocation_id']}-slot-"
                            f"{slot:02d}.json"
                        )
                    )
                    for slot in range(SLOTS)
                ],
                "text": text,
                "text_sha256": text_sha256(text),
                "slots": SLOTS,
                "all_slots_accepted": all(
                    row["accepted"] for row in slots
                ),
            },
        )
    return check_contract(folder)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "probe", "pilot"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--max-new-calls", type=int, default=0)
    args = parser.parse_args()
    if args.stage == "pilot":
        result = run_pilot(args.output, args.max_new_calls)
    elif args.stage == "probe":
        result = run_model_probe(args.output, args.max_new_calls)
    else:
        result = check_contract(args.output)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
