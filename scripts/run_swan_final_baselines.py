"""Checkpoint final SWAN baselines and export a detector-blind review packet."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time

from allocate_swan_final_sources import (
    OUTPUT,
    check_allocations,
    check_runtime,
    canonical,
)
from feedback2_runtime import claim
from run_confirmation_experiment import digest, freeze, request_body
from run_segmentation_intervention import ROOT, sha, text_sha, write_json
from supervise_confirmation_experiment import read_credential


MODEL = "deepseek-v4-flash"
URL = "https://api.deepseek.com/chat/completions"
PRICING_URL = "https://api-docs.deepseek.com/quick_start/pricing"
PRICING_RETRIEVED = "2026-09-16"
REMOTE_CALL_CAP = 20417
SLOTS = 8
TEMPLATES_PER_SLOT = 2
ATTEMPTS_PER_TEMPLATE = 5
MAX_ATTEMPTS_PER_SLOT = 10
TARGET_THRESHOLD = 0.75
MIN_ACCEPTED_SLOTS = 5
MIN_WORDS = 100
MAX_WORDS = 220
MIN_UNITS = 8
QUALITY_RANDOM_SEED = 20260916
PLACEHOLDER = re.compile(
    r"(?:\b(?:NE|ADJ|NUM|LOC|AMR)\b|"
    r":(?:ARG\d|op\d|name|mode|time|location)\b)"
)

FROZEN_ADDITION = """

Additional frozen constraints for this experiment:
- Realize every role and replace every placeholder with ordinary English.
- Emit no AMR notation, labels, explanations, quotation wrapper, or list.
- Add one new fact that is consistent with the running context.
- Do not restate a fact already present in the running context.
- Return exactly one complete sentence.
""".strip()


class InvocationBudgetExhausted(RuntimeError):
    """The user-selected per-invocation paid-call budget has been consumed."""


class PaidCallJournal:
    """Globally capped, non-retrying API journal."""

    def __init__(
        self,
        folder: Path,
        key: str,
        max_new_calls: int,
        expected_model: str | None = None,
    ):
        self.folder = folder
        self.key = key
        self.max_new_calls = max_new_calls
        self.expected_model = expected_model
        self.new_calls = 0

    def _inventory(self) -> list[Path]:
        return sorted((self.folder / "calls").glob("*.json"))

    def request(self, call_id: str, body: dict) -> dict:
        import requests

        path = self.folder / "calls" / f"{call_id}.json"
        request_hash = digest(body)
        if path.exists():
            saved = json.loads(path.read_text())
            if (
                saved["call_id"] != call_id
                or saved["request_sha256"] != request_hash
            ):
                raise ValueError(f"Replay request changed: {call_id}")
            if saved["status"] != "complete":
                return {
                    "status": "unavailable",
                    "journal_status": saved["status"],
                    "call_sha256": sha(path),
                }
            response = saved["response"]
            self._check_model(response)
            return {
                "status": "complete",
                "response": response,
                "call_sha256": sha(path),
            }
        if len(self._inventory()) >= REMOTE_CALL_CAP:
            raise RuntimeError("Frozen global remote-call cap reached")
        if self.new_calls >= self.max_new_calls:
            raise InvocationBudgetExhausted(
                "Per-invocation paid-call limit reached"
            )
        saved = {
            "call_id": call_id,
            "request_sha256": request_hash,
            "request": body,
            "status": "in_flight",
            "transport_attempts": 1,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(path, saved)
        self.new_calls += 1
        start = time.monotonic()
        try:
            result = requests.post(
                URL,
                headers={"Authorization": "Bearer " + self.key},
                json=body,
                timeout=(20, 240),
            )
            saved["http_status"] = result.status_code
            if result.status_code != 200:
                raise RuntimeError(f"HTTP {result.status_code}")
            payload = result.json()
            choice = payload["choices"][0]
            response = {
                "id": payload["id"],
                "model": payload["model"],
                "usage": payload.get("usage", {}),
                "content": choice["message"]["content"],
                "finish_reason": choice["finish_reason"],
            }
            if (
                not isinstance(response["content"], str)
                or not response["content"].strip()
            ):
                raise ValueError("Empty API content")
            self._check_model(response)
            saved.update(status="complete", response=response)
        except Exception as exc:
            saved.update(
                status="failed_or_uncertain",
                error_type=type(exc).__name__,
            )
            raise RuntimeError(
                f"API call stopped ({type(exc).__name__}); "
                f"inspect {path.name}"
            ) from None
        finally:
            saved["elapsed_seconds"] = time.monotonic() - start
            write_json(path, saved)
        return {
            "status": "complete",
            "response": response,
            "call_sha256": sha(path),
        }

    def _check_model(self, response: dict) -> None:
        if (
            self.expected_model is not None
            and response["model"] != self.expected_model
        ):
            raise ValueError("Returned model differs from frozen probe")


def configure_environment() -> None:
    os.environ.setdefault("NLTK_DATA", str(ROOT / ".nltk_data"))
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
    os.environ.setdefault(
        "GLOVE_VECTORS_PATH",
        str(ROOT / "vectors/glove.6B.100d.txt"),
    )
    import nltk

    nltk_path = str(ROOT / ".nltk_data")
    if nltk_path not in nltk.data.path:
        nltk.data.path.insert(0, nltk_path)


def slot_seed(prompt_id: str, slot: int) -> int:
    return int(
        text_sha(f"swan-final-confirmation1:{prompt_id}:{slot}")[:16],
        16,
    )


def score_seed(prompt_id: str, slot: int, attempt: int) -> int:
    return int(
        text_sha(
            f"swan-final-confirmation1:s2match:{prompt_id}:{slot}:{attempt}"
        )[:8],
        16,
    )


@contextmanager
def fixed_random_seed(seed: int):
    import numpy as np

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def generation_context(source: dict, prior_slots: list[dict]) -> str:
    text = source["generation_prompt"]
    previous = [
        row["selected_text"]
        for row in prior_slots
        if row.get("selected_text")
    ]
    return canonical(" ".join([text, *previous]))


def generation_prompt(
    target: str,
    examples: list[dict],
    context: str,
    correction: str | None = None,
) -> tuple[str, str]:
    configure_environment()
    sys.path.insert(0, str(ROOT / "third_party/SWAN"))
    from injection import injection_amr_utils

    system = injection_amr_utils.build_system_prompt()
    user = (
        injection_amr_utils.build_user_prompt(
            target,
            examples,
            context=context,
        )
        + "\n\n"
        + FROZEN_ADDITION
    )
    if correction:
        user += "\n\nCorrection feedback:\n" + correction
    return system, user


def load_generation_assets(detector) -> tuple[list[str], list[dict]]:
    examples_path = (
        ROOT
        / "third_party/SWAN/amr_bank/artifacts/normalized_amr_examples.json"
    )
    examples = json.loads(examples_path.read_text())[:3]
    bank = list(detector.bank)
    if len(bank) != 50 or len(examples) != 3:
        raise ValueError("Frozen SWAN bank or examples changed")
    return bank, examples


def slot_plan(
    folder: Path,
    source: dict,
    slot: int,
    bank: list[str],
    context: str,
) -> dict:
    seed = slot_seed(source["source_id"], slot)
    indices = random.Random(seed).sample(
        range(len(bank)),
        TEMPLATES_PER_SLOT,
    )
    payload = {
        "source_id": source["source_id"],
        "slot": slot,
        "seed": seed,
        "context_sha256": text_sha(context),
        "template_indices": indices,
        "template_sha256": [text_sha(bank[index]) for index in indices],
        "attempts_per_template": ATTEMPTS_PER_TEMPLATE,
    }
    return freeze(
        folder / "baseline_plans" / f'{source["source_id"]}-s{slot}.json',
        payload,
    )


def _surface_result(text: str, finish_reason: str) -> dict:
    configure_environment()
    import nltk

    stripped = text.strip()
    units = nltk.sent_tokenize(stripped) if stripped else []
    checks = {
        "nonempty": bool(stripped),
        "finish_reason_stop": finish_reason == "stop",
        "exactly_one_sentence": len(units) == 1,
        "no_amr_marker": not bool(PLACEHOLDER.search(stripped)),
    }
    return {
        "text": stripped,
        "text_sha256": text_sha(stripped),
        "sentence_units": len(units),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _correction(previous: dict) -> str:
    if previous["transport_status"] != "complete":
        return (
            "The preceding request did not yield a usable response. Produce "
            "one sentence under the original constraints."
        )
    surface = previous["surface"]
    if not surface["pass"]:
        failures = [
            key for key, value in surface["checks"].items() if not value
        ]
        return (
            f"The previous output was: {surface['text']}\n"
            f"It failed: {', '.join(failures)}. "
            "Return one complete natural sentence with no AMR notation."
        )
    if previous["parser_status"] != "complete":
        return (
            "The official AMR parser could not parse the previous sentence. "
            "Use a simpler grammatical realization of every target role."
        )
    return (
        f"The previous sentence was: {surface['text']}\n"
        "Its normalized template was:\n"
        f"{previous['normalized_amr']}\n"
        "The normalized template matched the target at "
        f"{previous['target_score']:.3f}, below {TARGET_THRESHOLD:.3f}. "
        "Correct role, node-type, name-token-count, and extra-edge "
        "mismatches while returning only one sentence."
    )


def evaluate_attempt(
    source: dict,
    slot: int,
    attempt: int,
    template_position: int,
    target: str,
    response_state: dict,
    detector,
) -> dict:
    if response_state["status"] != "complete":
        return {
            "source_id": source["source_id"],
            "slot": slot,
            "attempt": attempt,
            "template_position": template_position,
            "target_sha256": text_sha(target),
            "call_sha256": response_state["call_sha256"],
            "transport_status": response_state["journal_status"],
            "surface": None,
            "parser_status": "not_run",
            "normalized_amr": None,
            "target_score": None,
            "accepted": False,
        }
    response = response_state["response"]
    surface = _surface_result(
        response["content"],
        response["finish_reason"],
    )
    parser_status = "not_run"
    normalized_amr = None
    target_score = None
    parser_error = None
    if surface["pass"]:
        try:
            details = detector.parse(surface["text"])
            if len(details) != 1:
                raise ValueError("Candidate parser changed unit count")
            normalized_amr = details[0]["amr"]
            with fixed_random_seed(
                score_seed(source["source_id"], slot, attempt)
            ):
                target_score = float(
                    detector.utils.compute_s2match_score(
                        normalized_amr,
                        target,
                    )
                )
            if not math.isfinite(target_score) or not 0 <= target_score <= 1:
                raise ValueError("Invalid target S2Match score")
            parser_status = "complete"
        except Exception as exc:
            parser_status = "failed"
            parser_error = type(exc).__name__
    return {
        "source_id": source["source_id"],
        "slot": slot,
        "attempt": attempt,
        "template_position": template_position,
        "target_sha256": text_sha(target),
        "call_sha256": response_state["call_sha256"],
        "transport_status": "complete",
        "response_id": response["id"],
        "response_model": response["model"],
        "usage": response["usage"],
        "surface": surface,
        "parser_status": parser_status,
        "parser_error_type": parser_error,
        "normalized_amr": normalized_amr,
        "target_score": target_score,
        "accepted": bool(
            parser_status == "complete"
            and target_score >= TARGET_THRESHOLD
        ),
    }


def _load_attempt(path: Path, expected: dict) -> dict:
    saved = json.loads(path.read_text())
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError(f"Attempt checkpoint changed: {path.name}")
    return saved


def run_slot(
    folder: Path,
    source: dict,
    slot: int,
    prior_slots: list[dict],
    detector,
    api: PaidCallJournal,
    bank: list[str],
    examples: list[dict],
) -> dict:
    result_path = (
        folder / "baseline_slots" / f'{source["source_id"]}-s{slot}.json'
    )
    context = generation_context(source, prior_slots)
    plan = slot_plan(folder, source, slot, bank, context)
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if (
            saved["source_id"] != source["source_id"]
            or saved["slot"] != slot
            or saved["plan_sha256"]
            != sha(
                folder
                / "baseline_plans"
                / f'{source["source_id"]}-s{slot}.json'
            )
        ):
            raise ValueError(f"Slot checkpoint changed: {result_path.name}")
        return saved

    attempts = []
    correction = None
    last_complete = None
    for attempt in range(MAX_ATTEMPTS_PER_SLOT):
        template_position = attempt // ATTEMPTS_PER_TEMPLATE
        if attempt % ATTEMPTS_PER_TEMPLATE == 0:
            correction = None
        target = bank[plan["template_indices"][template_position]]
        system, user = generation_prompt(
            target,
            examples,
            context,
            correction,
        )
        body = request_body(
            user,
            temperature=0.7,
            max_tokens=120,
            system=system,
            top_p=0.9,
        )
        call_id = (
            f'baseline-{source["source_id"]}-s{slot}-a{attempt:02d}'
        )
        attempt_path = (
            folder
            / "baseline_attempts"
            / f'{source["source_id"]}-s{slot}-a{attempt:02d}.json'
        )
        expected = {
            "source_id": source["source_id"],
            "slot": slot,
            "attempt": attempt,
            "template_position": template_position,
            "target_sha256": text_sha(target),
            "request_sha256": digest(body),
        }
        if attempt_path.exists():
            result = _load_attempt(attempt_path, expected)
        else:
            response_state = api.request(call_id, body)
            result = {
                **expected,
                **evaluate_attempt(
                    source,
                    slot,
                    attempt,
                    template_position,
                    target,
                    response_state,
                    detector,
                ),
            }
            freeze(attempt_path, result)
        attempts.append(result)
        if (
            result["transport_status"] == "complete"
            and result["surface"]["checks"]["nonempty"]
            and result["surface"]["checks"]["finish_reason_stop"]
        ):
            last_complete = result
        if result["accepted"]:
            break
        correction = _correction(result)

    selected = next(
        (row for row in attempts if row["accepted"]),
        last_complete,
    )
    payload = {
        "source_id": source["source_id"],
        "slot": slot,
        "plan_sha256": sha(
            folder
            / "baseline_plans"
            / f'{source["source_id"]}-s{slot}.json'
        ),
        "status": (
            "accepted"
            if selected and selected["accepted"]
            else "exhausted"
        ),
        "attempts": len(attempts),
        "attempt_checkpoint_sha256": [
            sha(
                folder
                / "baseline_attempts"
                / f'{source["source_id"]}-s{slot}-a{index:02d}.json'
            )
            for index in range(len(attempts))
        ],
        "accepted": bool(selected and selected["accepted"]),
        "selected_attempt": selected["attempt"] if selected else None,
        "selected_text": (
            selected["surface"]["text"]
            if selected and selected["surface"]
            else ""
        ),
        "selected_text_sha256": (
            selected["surface"]["text_sha256"]
            if selected and selected["surface"]
            else text_sha("")
        ),
        "selected_surface_pass": bool(
            selected and selected["surface"] and selected["surface"]["pass"]
        ),
        "selected_target_sha256": (
            selected["target_sha256"] if selected else None
        ),
        "selected_target_score": (
            selected["target_score"] if selected else None
        ),
    }
    return freeze(result_path, payload)


def baseline_gates(slots: list[dict]) -> tuple[str, dict]:
    configure_environment()
    import nltk

    text = canonical(
        " ".join(row["selected_text"] for row in slots if row["selected_text"])
    )
    units = nltk.sent_tokenize(text) if text else []
    words = len(text.split())
    checks = {
        "eight_complete_single_sentence_slots": (
            len(slots) == SLOTS
            and all(row["selected_surface_pass"] for row in slots)
        ),
        "minimum_accepted_target_slots": (
            sum(row["accepted"] for row in slots) >= MIN_ACCEPTED_SLOTS
        ),
        "word_range": MIN_WORDS <= words <= MAX_WORDS,
        "minimum_detector_units": len(units) >= MIN_UNITS,
        "no_amr_marker": not bool(PLACEHOLDER.search(text)),
    }
    return text, {
        "checks": checks,
        "mechanically_eligible": all(checks.values()),
        "words": words,
        "detector_units": len(units),
        "accepted_slots": sum(row["accepted"] for row in slots),
    }


def run_baseline(
    folder: Path,
    source: dict,
    detector,
    api: PaidCallJournal,
    bank: list[str],
    examples: list[dict],
) -> dict:
    path = folder / "baselines" / f'{source["source_id"]}.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if (
            saved["source_id"] != source["source_id"]
            or saved["source_manifest_sha256"]
            != sha(folder / "source_manifest.json")
        ):
            raise ValueError(f"Baseline checkpoint changed: {path.name}")
        return saved
    slots = []
    for slot in range(SLOTS):
        slots.append(
            run_slot(
                folder,
                source,
                slot,
                slots,
                detector,
                api,
                bank,
                examples,
            )
        )
    text, mechanical = baseline_gates(slots)
    payload = {
        "source_id": source["source_id"],
        "source_manifest_sha256": sha(folder / "source_manifest.json"),
        "source_text_sha256": source["text_sha256"],
        "generation_prompt_sha256": source["generation_prompt_sha256"],
        "slot_checkpoint_sha256": [
            sha(
                folder
                / "baseline_slots"
                / f'{source["source_id"]}-s{slot}.json'
            )
            for slot in range(SLOTS)
        ],
        "text": text,
        "text_sha256": text_sha(text),
        "mechanical": mechanical,
    }
    return freeze(path, payload)


def load_model_probe(folder: Path) -> dict:
    path = folder / "model_probe.json"
    probe = json.loads(path.read_text())
    if (
        probe["status"] != "complete"
        or probe["model_requested"] != MODEL
        or probe["call_sha256"] != sha(folder / "calls/model-probe.json")
        or probe["cost_estimate_sha256"]
        != sha(folder / "cost_estimate.json")
    ):
        raise ValueError("Model probe provenance mismatch")
    return probe


def probe_model(
    folder: Path,
    key: str,
    max_new_calls: int,
) -> dict:
    check_allocations(folder)
    if not (folder / "cost_estimate.json").exists():
        raise RuntimeError("Freeze a monetary cost estimate before API probe")
    path = folder / "model_probe.json"
    if path.exists():
        return load_model_probe(folder)
    api = PaidCallJournal(folder, key, max_new_calls)
    body = request_body("Reply exactly OK.", 0, 16)
    state = api.request("model-probe", body)
    if state["status"] != "complete":
        raise RuntimeError("Model probe is unavailable")
    response = state["response"]
    if response["finish_reason"] != "stop" or response["content"].strip() != "OK":
        raise ValueError("Unexpected model probe response")
    return freeze(
        path,
        {
            "status": "complete",
            "model_requested": MODEL,
            "model_returned": response["model"],
            "response_id": response["id"],
            "usage": response["usage"],
            "call_sha256": state["call_sha256"],
            "cost_estimate_sha256": sha(folder / "cost_estimate.json"),
        },
    )


def run_all_baselines(
    folder: Path,
    key: str,
    max_new_calls: int,
    limit: int | None = None,
) -> dict:
    configure_environment()
    source_manifest, _controls = check_allocations(folder)
    probe = load_model_probe(folder)
    from feedback2_runtime import CachedDetector

    runtime_signature = digest(
        {
            "runtime_sha256": sha(folder / "runtime.json"),
            "model_probe_sha256": sha(folder / "model_probe.json"),
        }
    )
    detector = CachedDetector(folder, runtime_signature)
    bank, examples = load_generation_assets(detector)
    api = PaidCallJournal(
        folder,
        key,
        max_new_calls,
        expected_model=probe["model_returned"],
    )
    sources = source_manifest["sources"]
    if limit is not None:
        if limit < 0 or limit > len(sources):
            raise ValueError("Invalid source limit")
        sources = sources[:limit]
    complete = 0
    with claim(folder / "locks/baseline_generation.lock"):
        for source in sources:
            run_baseline(
                folder,
                source,
                detector,
                api,
                bank,
                examples,
            )
            complete += 1
            print(
                json.dumps(
                    {
                        "stage": "baseline",
                        "source_id": source["source_id"],
                        "completed_this_invocation": complete,
                        "new_paid_calls": api.new_calls,
                    }
                ),
                flush=True,
            )
    return {
        "completed": len(list((folder / "baselines").glob("wm-*.json"))),
        "new_paid_calls": api.new_calls,
        "cache_counts": detector.counts,
    }


def historical_generation_usage() -> list[dict]:
    paths = sorted(
        (
            ROOT
            / "experiments/swan_generator_comparison1/calls"
        ).glob("generate-local_adapted-*.json")
    )
    rows = []
    for path in paths:
        saved = json.loads(path.read_text())
        if saved.get("status") != "complete":
            continue
        response = saved["response"]
        if response.get("model") != "deepseek-flash":
            continue
        usage = response.get("usage", {})
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if not isinstance(prompt, int) or not isinstance(completion, int):
            continue
        details = usage.get("prompt_tokens_details", {})
        cached = details.get(
            "cached_tokens",
            usage.get("prompt_cache_hit_tokens", 0),
        )
        rows.append(
            {
                "prompt_tokens": prompt,
                "cached_prompt_tokens": int(cached or 0),
                "uncached_prompt_tokens": prompt - int(cached or 0),
                "completion_tokens": completion,
            }
        )
    if not rows:
        raise RuntimeError("No exact-alias historical usage sample")
    return rows


def _percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    return (
        ordered[low] * (high - position)
        + ordered[high] * (position - low)
    )


def make_cost_estimate(
    cache_hit_usd_per_million: float,
    cache_miss_usd_per_million: float,
    output_usd_per_million: float,
) -> dict:
    rates = {
        "cache_hit_input_usd_per_million": cache_hit_usd_per_million,
        "cache_miss_input_usd_per_million": cache_miss_usd_per_million,
        "output_usd_per_million": output_usd_per_million,
    }
    if not all(math.isfinite(value) and value >= 0 for value in rates.values()):
        raise ValueError("Prices must be finite and nonnegative")
    rows = historical_generation_usage()
    per_call = {}
    for key in (
        "prompt_tokens",
        "cached_prompt_tokens",
        "uncached_prompt_tokens",
        "completion_tokens",
    ):
        values = [row[key] for row in rows]
        per_call[key] = {
            "mean": statistics.fmean(values),
            "p50": statistics.median(values),
            "p95": _percentile(values, 0.95),
            "maximum": max(values),
        }

    def monetary(count: int, statistic: str) -> float:
        return count / 1_000_000 * (
            per_call["cached_prompt_tokens"][statistic]
            * cache_hit_usd_per_million
            + per_call["uncached_prompt_tokens"][statistic]
            * cache_miss_usd_per_million
            + per_call["completion_tokens"][statistic]
            * output_usd_per_million
        )

    baseline_cap = 250 * 8 * 10
    paraphrase_cap = 104 * 4
    return {
        "schema_version": 1,
        "status": "estimated_before_paid_final_requests",
        "model_requested": MODEL,
        "historical_model_returned": "deepseek-flash",
        "pricing_source": PRICING_URL,
        "pricing_retrieved": PRICING_RETRIEVED,
        "currency": "USD",
        "pricing_applicability": (
            "The official pricing page states that legacy "
            "deepseek-v4-flash requests are served by DeepSeek-V4.1-Flash "
            "and billed at deepseek-flash rates. The numeric rates remain "
            "explicit inputs so peak and off-peak estimates cannot be mixed."
        ),
        "rates": rates,
        "historical_sample": {
            "path_glob": (
                "experiments/swan_generator_comparison1/calls/"
                "generate-local_adapted-*.json"
            ),
            "calls": len(rows),
            "per_call_tokens": per_call,
        },
        "request_budget": {
            "probe": 1,
            "baseline_injection": baseline_cap,
            "ordinary_paraphrase": paraphrase_cap,
            "total": 1 + baseline_cap + paraphrase_cap,
        },
        "estimated_usd": {
            "baseline_injection_mean_usage": monetary(
                baseline_cap,
                "mean",
            ),
            "baseline_injection_p95_per_call_usage": monetary(
                baseline_cap,
                "p95",
            ),
            "baseline_injection_observed_max_per_field": monetary(
                baseline_cap,
                "maximum",
            ),
            "full_cap_mean_usage": monetary(
                baseline_cap + paraphrase_cap + 1,
                "mean",
            ),
            "full_cap_p95_per_call_usage": monetary(
                baseline_cap + paraphrase_cap + 1,
                "p95",
            ),
            "full_cap_observed_max_per_field": monetary(
                baseline_cap + paraphrase_cap + 1,
                "maximum",
            ),
        },
        "limitations": [
            "Historical sample contains final-style eight-slot local-adapted injection calls but not final sources.",
            "P95-per-call multiplication is a conservative scenario, not a confidence bound.",
            "Paraphrase calls are costed using injection-call token usage because no final paraphrase has run.",
            "No paid final request is authorized by this estimate artifact.",
        ],
    }


def freeze_cost_estimate(
    folder: Path,
    cache_hit: float,
    cache_miss: float,
    output: float,
) -> dict:
    check_allocations(folder)
    estimate = make_cost_estimate(cache_hit, cache_miss, output)
    return freeze(folder / "cost_estimate.json", estimate)


def create_quality_packet(folder: Path) -> dict:
    source_manifest, _controls = check_allocations(folder)
    paths = [
        folder / "baselines" / f'{row["source_id"]}.json'
        for row in source_manifest["sources"]
    ]
    if not all(path.exists() for path in paths):
        raise RuntimeError("All 250 baseline checkpoints are required")
    source_by_id = {
        row["source_id"]: row for row in source_manifest["sources"]
    }
    eligible = []
    flow = {
        "allocated": len(paths),
        "baseline_complete": 0,
        "mechanically_eligible_for_human_review": 0,
    }
    for path in paths:
        baseline = json.loads(path.read_text())
        flow["baseline_complete"] += 1
        if baseline["mechanical"]["mechanically_eligible"]:
            source = source_by_id[baseline["source_id"]]
            eligible.append(
                {
                    "source_id": baseline["source_id"],
                    "source_prompt": source["generation_prompt"],
                    "text": baseline["text"],
                    "text_sha256": baseline["text_sha256"],
                    "baseline_sha256": sha(path),
                }
            )
    flow["mechanically_eligible_for_human_review"] = len(eligible)
    random.Random(QUALITY_RANDOM_SEED).shuffle(eligible)
    public_rows = []
    private_rows = []
    for index, row in enumerate(eligible):
        review_id = f"Q{index:04d}"
        public_rows.append(
            {
                "review_id": review_id,
                "source_prompt": row["source_prompt"],
                "candidate_text": row["text"],
            }
        )
        private_rows.append(
            {
                "review_id": review_id,
                "source_id": row["source_id"],
                "candidate_text_sha256": row["text_sha256"],
                "baseline_sha256": row["baseline_sha256"],
            }
        )
    packet = {
        "schema_version": 1,
        "status": "blind_quality_packet_before_detection",
        "random_seed": QUALITY_RANDOM_SEED,
        "flow": flow,
        "instructions": {
            "raters": 2,
            "dimensions": [
                "fluency",
                "coherence",
                "nonredundancy",
                "prompt_consistency",
            ],
            "scale": [1, 5],
            "pass": (
                "Both raters score all four dimensions >=4 and neither "
                "marks severe_error; disagreements require a third blinded "
                "adjudicator."
            ),
            "forbidden_information": [
                "source_id",
                "model",
                "template",
                "target_score",
                "acceptance",
                "parser_output",
                "S2Match",
                "green",
                "z",
                "attack_condition",
            ],
        },
        "rows": public_rows,
    }
    key = {
        "schema_version": 1,
        "quality_packet_digest": digest(packet),
        "rows": private_rows,
    }
    freeze(folder / "quality_packet.json", packet)
    freeze(folder / "quality_packet_key.json", key)
    return packet


def summarize_calls(folder: Path) -> dict:
    rows = [
        json.loads(path.read_text())
        for path in sorted((folder / "calls").glob("*.json"))
    ]
    complete = [row for row in rows if row["status"] == "complete"]
    return {
        "attempted_calls": len(rows),
        "complete_calls": len(complete),
        "failed_or_uncertain_calls": len(rows) - len(complete),
        "usage": {
            key: sum(
                row["response"].get("usage", {}).get(key, 0)
                for row in complete
            )
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "prompt_cache_hit_tokens",
                "prompt_cache_miss_tokens",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "cost-estimate",
            "probe",
            "generate",
            "quality-packet",
            "status",
        ),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--max-new-calls", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cache-hit-price", type=float)
    parser.add_argument("--cache-miss-price", type=float)
    parser.add_argument("--output-price", type=float)
    args = parser.parse_args()

    if args.stage == "cost-estimate":
        prices = (
            args.cache_hit_price,
            args.cache_miss_price,
            args.output_price,
        )
        if any(value is None for value in prices):
            parser.error("cost-estimate requires all three price arguments")
        result = freeze_cost_estimate(args.output, *prices)
    elif args.stage in {"probe", "generate"}:
        if args.key_file is None or args.max_new_calls < 1:
            parser.error(
                f"{args.stage} requires --key-file and --max-new-calls >= 1"
            )
        key = read_credential(args.key_file)
        if args.stage == "probe":
            result = probe_model(
                args.output,
                key,
                args.max_new_calls,
            )
        else:
            result = run_all_baselines(
                args.output,
                key,
                args.max_new_calls,
                args.limit,
            )
    elif args.stage == "quality-packet":
        result = create_quality_packet(args.output)
    else:
        check_runtime(args.output)
        result = {
            "calls": summarize_calls(args.output),
            "baselines": len(
                list((args.output / "baselines").glob("wm-*.json"))
            ),
            "model_probe": (args.output / "model_probe.json").exists(),
            "quality_packet": (
                args.output / "quality_packet.json"
            ).exists(),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
