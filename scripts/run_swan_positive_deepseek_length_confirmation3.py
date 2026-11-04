"""Run the resilient, length-controlled DeepSeek SWAN confirmation.

Source preparation downloads public C4 data but makes no model calls. Paid
DeepSeek execution requires a separate phase authorization and a positive
per-invocation call allowance.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Callable, Optional
import unicodedata

import prepare_swan_positive_collection as allocator
import run_swan_positive_collection as swan


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/swan_positive_deepseek_length_confirmation3"
PROTOCOL = OUTPUT / "protocol.md"
SOURCE_SNAPSHOT = OUTPUT / "prior_identity_snapshot.json"
SOURCE_MANIFEST = OUTPUT / "source_manifest.json"
MANIFEST = OUTPUT / "manifest.json"
BUDGET = OUTPUT / "budget_estimate.json"
PILOT_AUTH_TEMPLATE = OUTPUT / "pilot_authorization_template.json"
MAIN_AUTH_TEMPLATE = OUTPUT / "main_authorization_template.json"
PILOT_AUTHORIZATION = OUTPUT / "pilot_authorization.json"
MAIN_AUTHORIZATION = OUTPUT / "main_authorization.json"

DATASET_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
SOURCE_SHARD = 12
SHARD_COUNT = 512
PILOT_DOCUMENTS = 5
MAIN_DOCUMENTS = 30
SLOTS = 16
MAX_TRIALS = 128
MAX_TEMPLATES = 26
RETRIES_PER_TEMPLATE = 5
MATCH_THRESHOLD = 0.75
MIN_WORDS = 100
MAX_WORDS = 220
MIN_SLOT_WORDS = 7
MAX_SLOT_WORDS = 12
LENGTH_INSTRUCTION = (
    "\n\nAdditional fixed constraints:\n"
    f"- Write exactly {MIN_SLOT_WORDS} to {MAX_SLOT_WORDS} lexical words; "
    "count a hyphenated term as one word.\n"
    "- Write one complete, fluent sentence ending with '.', '!', or '?'.\n"
    "- Preserve the AMR meaning and context, and output no explanation."
)

MODEL = "deepseek-flash"
RETURNED_MODEL = "deepseek-flash"
URL = "https://api.deepseek.com/chat/completions"
PRICE_SOURCE = "https://api-docs.deepseek.com/quick_start/pricing"
PRICE_RETRIEVED = "2026-09-18"
INPUT_USD_PER_MILLION = 0.30
OUTPUT_USD_PER_MILLION = 1.20
INPUT_OVERHEAD_RESERVE = 1024
CONTEXT_TOKEN_RESERVE = (SLOTS - 1) * swan.MAX_NEW_TOKENS
PROBE_CALLS = 1
PILOT_CALL_CAP = PROBE_CALLS + PILOT_DOCUMENTS * SLOTS * MAX_TRIALS
MAIN_CALL_CAP = MAIN_DOCUMENTS * SLOTS * MAX_TRIALS

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)*")
TERMINAL_PUNCTUATION_RE = re.compile(r"[.!?][\"')\]]*$")
AMR_LEAK_RE = re.compile(
    r"(?i)(?:template\s+amr|abstract meaning representation|"
    r":ARG\d|:[a-z][a-z0-9_-]*\s+\(|\(\s*[a-z]\d*\s*/)"
)
TEXT_IDENTITY_KEYS = allocator.TEXT_IDENTITY_KEYS
HASH_IDENTITY_KEYS = allocator.HASH_IDENTITY_KEYS
URL_KEYS = allocator.URL_KEYS


class InvocationBudgetExhausted(RuntimeError):
    """The explicit per-command allowance cannot start another full slot."""


class UncertainCallExists(RuntimeError):
    """A transmitted call has no definitely complete response."""


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


def canonical(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def normalized_hash(text: str) -> str:
    return text_sha256(canonical(text).casefold())


def normalized_url_hash(url: str) -> str:
    return text_sha256(canonical(url))


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


def package_versions() -> dict:
    result = {}
    for name in (
        "amrlib",
        "nltk",
        "numpy",
        "penman",
        "requests",
        "scipy",
        "spacy",
        "torch",
        "transformers",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not_installed"
    return result


def dependency_hashes() -> dict:
    paths = [
        Path(__file__),
        PROTOCOL,
        Path(swan.__file__),
        swan.OFFICIAL_INJECTION,
        swan.OFFICIAL_ROOT / "amr_bank/banks/amr_bank_50.json",
        swan.OFFICIAL_ROOT
        / "amr_bank/artifacts/normalized_amr_examples.json",
        swan.S2MATCH_RUNTIME / "provenance.json",
        swan.PARSER_MODEL / "pytorch_model.bin",
        swan.PARSER_TOKENIZER / "tokenizer.json",
        swan.GLOVE_VECTORS,
        swan.NLTK_DATA / "corpora/propbank.zip",
        swan.NLTK_DATA
        / "tokenizers/punkt_tab/english/ortho_context.tab",
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing frozen dependency: " + ", ".join(map(str, missing))
        )
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for path in paths
    }


def shard_name(index: int = SOURCE_SHARD) -> str:
    if index not in range(SHARD_COUNT):
        raise ValueError(f"Invalid C4 shard: {index}")
    return f"realnewslike/c4-train.{index:05d}-of-{SHARD_COUNT:05d}.json.gz"


def shard_url(index: int = SOURCE_SHARD) -> str:
    return (
        "https://huggingface.co/datasets/allenai/c4/resolve/"
        f"{DATASET_REVISION}/{shard_name(index)}"
    )


def extract_prompt(text: str, length: int = 32) -> str:
    """Reproduce the official SWAN whitespace/punctuation prompt extractor."""
    excerpt = " ".join(text.split(" ")[:length])
    prompts = []
    for punctuation in "!.?":
        index = excerpt.find(punctuation)
        if index != -1 and len(excerpt[: index + 1].split(" ")) > 3:
            prompts.append(excerpt[: index + 1])
    return min(prompts, key=len) if prompts else excerpt + "."


def stream_source_rows(
    transport: Optional[Callable[[str], object]] = None,
):
    if transport is None:
        import requests

        def transport(url: str):
            return requests.get(
                url,
                headers={
                    "User-Agent": (
                        "SWAN-DeepSeek-length-confirmation3-source/1.0"
                    )
                },
                stream=True,
                timeout=(20, 90),
            )

    response = transport(shard_url())
    response.raise_for_status()
    response.raw.decode_content = False
    with response:
        with gzip.GzipFile(fileobj=response.raw) as handle:
            for row_index, line in enumerate(handle):
                yield row_index, json.loads(line)


def build_history_snapshot() -> dict:
    paths, scanned = allocator.discover_history_artifacts(
        root=ROOT,
        excluded_tree=OUTPUT,
    )
    snapshot = allocator.build_prior_identity_snapshot(
        paths,
        root=ROOT,
        all_files_scanned=scanned,
    )
    snapshot["selection"]["excluded_tree"] = str(OUTPUT.relative_to(ROOT))
    snapshot["selection"]["purpose"] = (
        "Exclude every historical source/text identity before the independent "
        "DeepSeek length-controlled follow-up allocation."
    )
    return snapshot


def screen_source(
    row: dict,
    row_index: int,
    seen_text: set[str],
    seen_url: set[str],
) -> dict:
    text = row.get("text", "")
    url = row.get("url", "")
    if not isinstance(text, str):
        text = ""
    if not isinstance(url, str):
        url = ""
    prompt = extract_prompt(text) if canonical(text) else ""
    text_identity = normalized_hash(text)
    url_identity = normalized_url_hash(url)
    checks = {
        "nonempty_source": bool(canonical(text)),
        "nonempty_prompt": bool(canonical(prompt)),
        "historically_unseen_text": text_identity not in seen_text,
        "historically_unseen_url": not url or url_identity not in seen_url,
    }
    return {
        "shard_index": SOURCE_SHARD,
        "shard_row": row_index,
        "dataset_locator": f"{shard_name()}:{row_index}",
        "text": text,
        "text_sha256": text_sha256(text),
        "normalized_source_sha256": text_identity,
        "url": url,
        "url_sha256": url_identity,
        "generation_prompt": prompt,
        "generation_prompt_sha256": text_sha256(prompt),
        "checks": checks,
        "decision": "selected" if all(checks.values()) else "excluded",
    }


def source_record(screen: dict, ordinal: int) -> dict:
    pilot = ordinal < PILOT_DOCUMENTS
    cohort = "pilot" if pilot else "main"
    cohort_ordinal = ordinal if pilot else ordinal - PILOT_DOCUMENTS
    return {
        "allocation_id": f"dslength3-{cohort}-{cohort_ordinal:03d}",
        "allocation_ordinal": ordinal,
        "cohort": cohort,
        "cohort_ordinal": cohort_ordinal,
        **{
            key: screen[key]
            for key in (
                "shard_index",
                "shard_row",
                "dataset_locator",
                "text",
                "text_sha256",
                "normalized_source_sha256",
                "url",
                "url_sha256",
                "generation_prompt",
                "generation_prompt_sha256",
            )
        },
        "role": "watermarked_generation_source",
    }


def allocate_sources(
    transport: Optional[Callable[[str], object]] = None,
) -> dict:
    if SOURCE_MANIFEST.exists():
        return check_sources()
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    if SOURCE_SNAPSHOT.exists():
        raise RuntimeError("Partial source freeze requires manual inspection")

    snapshot = freeze_json(SOURCE_SNAPSHOT, build_history_snapshot())
    seen_text = set(snapshot["normalized_text_sha256"])
    seen_url = set(snapshot["url_sha256"])
    sources = []
    screened = []
    for row_index, row in stream_source_rows(transport):
        screen = screen_source(row, row_index, seen_text, seen_url)
        screened.append(screen)
        if screen["decision"] != "selected":
            continue
        record = source_record(screen, len(sources))
        sources.append(record)
        seen_text.add(record["normalized_source_sha256"])
        if record["url"]:
            seen_url.add(record["url_sha256"])
        if len(sources) == PILOT_DOCUMENTS + MAIN_DOCUMENTS:
            break
    if len(sources) != PILOT_DOCUMENTS + MAIN_DOCUMENTS:
        raise RuntimeError("C4 shard ended before all sources were allocated")

    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "sources_frozen_before_model_output",
        "dataset": {
            "name": "allenai/c4",
            "configuration": "realnewslike",
            "revision": DATASET_REVISION,
            "shard": shard_name(),
            "shard_url": shard_url(),
        },
        "selection": (
            "First 35 nonempty prompt-bearing records in row order whose "
            "normalized text and URL identities are absent from the complete "
            "pre-allocation workspace-history snapshot."
        ),
        "prior_identity_snapshot_sha256": sha256(SOURCE_SNAPSHOT),
        "cohort_counts": {
            "pilot": PILOT_DOCUMENTS,
            "main": MAIN_DOCUMENTS,
        },
        "replacement_allowed": False,
        "rows_scanned": len(screened),
        "screened_rows": screened,
        "sources": sources,
        "resource_counts": {
            "public_dataset_http_requests": 1,
            "model_calls": 0,
            "parser_calls": 0,
            "s2match_calls": 0,
        },
    }
    return freeze_json(SOURCE_MANIFEST, payload)


def check_sources() -> dict:
    snapshot = load_json(SOURCE_SNAPSHOT)
    manifest = load_json(SOURCE_MANIFEST)
    if (
        manifest["status"] != "sources_frozen_before_model_output"
        or manifest["prior_identity_snapshot_sha256"]
        != sha256(SOURCE_SNAPSHOT)
        or manifest["cohort_counts"]
        != {"pilot": PILOT_DOCUMENTS, "main": MAIN_DOCUMENTS}
        or len(manifest["sources"]) != PILOT_DOCUMENTS + MAIN_DOCUMENTS
        or manifest["replacement_allowed"] is not False
    ):
        raise ValueError("Source manifest contract differs")
    prior_text = set(snapshot["normalized_text_sha256"])
    prior_url = set(snapshot["url_sha256"])
    seen_text = set()
    seen_url = set()
    for ordinal, source in enumerate(manifest["sources"]):
        if source != source_record(
            next(
                row
                for row in manifest["screened_rows"]
                if (
                    row["shard_row"] == source["shard_row"]
                    and row["decision"] == "selected"
                )
            ),
            ordinal,
        ):
            raise ValueError("Source allocation does not replay")
        if (
            source["normalized_source_sha256"] in prior_text | seen_text
            or (
                source["url"]
                and source["url_sha256"] in prior_url | seen_url
            )
        ):
            raise ValueError("Source identity is not independent")
        seen_text.add(source["normalized_source_sha256"])
        if source["url"]:
            seen_url.add(source["url_sha256"])
    return manifest


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


def request_reserve(body: dict, extra_input_tokens: int = 0) -> dict:
    request_bytes = len(
        json.dumps(body, ensure_ascii=True).encode("utf-8")
    )
    input_bound = (
        request_bytes + INPUT_OVERHEAD_RESERVE + extra_input_tokens
    )
    output_bound = body["max_tokens"]
    cost = (
        input_bound * INPUT_USD_PER_MILLION / 1_000_000
        + output_bound * OUTPUT_USD_PER_MILLION / 1_000_000
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
    conservative = (
        prompt * INPUT_USD_PER_MILLION / 1_000_000
        + completion * OUTPUT_USD_PER_MILLION / 1_000_000
    )
    return {
        "prompt_tokens": prompt,
        "cached_prompt_tokens": cached,
        "uncached_prompt_tokens": prompt - cached,
        "completion_tokens": completion,
        "conservative_peak_cost_usd": round(conservative, 12),
    }


def parse_response(payload: dict) -> dict:
    choice = payload["choices"][0]
    text = choice["message"]["content"].strip()
    result = {
        "id": payload["id"],
        "model": payload["model"],
        "usage": payload["usage"],
        "text": text,
        "text_sha256": text_sha256(text),
        "stop_reason": choice["finish_reason"],
        "response_metadata": {
            "provider": "deepseek",
            "response_id": payload["id"],
            "model": payload["model"],
        },
    }
    if (
        result["model"] != RETURNED_MODEL
        or not text
        or result["stop_reason"] not in {"stop", "length"}
    ):
        raise ValueError("Unexpected DeepSeek response")
    return result


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


def read_credential(path: Path) -> str:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise PermissionError(
            "Credential file must not be group/world readable"
        )
    keys = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("sk-")
    ]
    if len(keys) != 1:
        raise ValueError("Expected exactly one DeepSeek credential")
    return keys[0]


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


def length_controlled_user_prompt(base_prompt: str) -> str:
    if LENGTH_INSTRUCTION in base_prompt:
        raise ValueError("Length instruction was appended more than once")
    return base_prompt.rstrip() + LENGTH_INSTRUCTION


def candidate_gate(candidate: str) -> dict:
    import nltk

    local_nltk = str(swan.NLTK_DATA)
    if local_nltk not in nltk.data.path:
        nltk.data.path.insert(0, local_nltk)
    text = candidate.strip()
    sentences = nltk.sent_tokenize(text) if text else []
    words = len(WORD_RE.findall(text))
    checks = {
        "word_count": words,
        "word_range_pass": MIN_SLOT_WORDS <= words <= MAX_SLOT_WORDS,
        "sentence_units": len(sentences),
        "single_sentence_pass": sentences == [text],
        "terminal_punctuation_pass": bool(
            TERMINAL_PUNCTUATION_RE.search(text)
        ),
    }
    checks["candidate_gate_pass"] = all(
        checks[key]
        for key in (
            "word_range_pass",
            "single_sentence_pass",
            "terminal_punctuation_pass",
        )
    )
    return checks


BASE_SWAN_INSTRUMENT = swan.instrument_official_injection


@contextmanager
def instrument_length_controlled_injection(
    module,
    folder: Path,
    allocation_id: str,
    slot: int,
    bank: list,
    journal,
):
    """Add the preregistered prompt and candidate gate to SWAN's trace."""
    with BASE_SWAN_INSTRUMENT(
        module,
        folder,
        allocation_id,
        slot,
        bank,
        journal,
    ) as trace:
        base_prompt = module.build_user_prompt
        base_match = module.decide_s2match_match

        def build_user_prompt(
            chosen_template: str,
            examples: list,
            context: str = "",
        ) -> str:
            return length_controlled_user_prompt(
                base_prompt(chosen_template, examples, context=context)
            )

        def decide_s2match_match(
            amr: str,
            target: str,
            threshold: float,
        ):
            event = trace["current_event"]
            if event is None:
                raise RuntimeError("Candidate gate has no active attempt")
            gate = candidate_gate(event["candidate"])
            event["candidate_gate"] = gate
            if gate["candidate_gate_pass"]:
                event["s2match_evaluated"] = True
                event["candidate_gate_rejection"] = None
                return base_match(amr, target, threshold)
            if target != event["template"]:
                raise ValueError(
                    "Match target differs from sampled template"
                )
            event.update(
                status="complete",
                candidate_template_amr=amr,
                candidate_template_amr_sha256=text_sha256(amr),
                score_seed=None,
                threshold=threshold,
                score=0.0,
                accepted=False,
                s2match_evaluated=False,
                candidate_gate_rejection="length_or_sentence_structure",
            )
            freeze_json(
                folder
                / "generation_attempts"
                / f"{event['call_id']}.json",
                event,
            )
            return False, 0.0

        module.build_user_prompt = build_user_prompt
        module.decide_s2match_match = decide_s2match_match
        try:
            yield trace
        finally:
            module.build_user_prompt = base_prompt
            module.decide_s2match_match = base_match


def build_manifest(created_at: Optional[str] = None) -> dict:
    source = check_sources()
    return {
        "schema_version": 1,
        "created_at": created_at or utc_now(),
        "status": "frozen_remote_unauthorized",
        "workflow": "swan_positive_deepseek_length_confirmation3",
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "prior_identity_snapshot_sha256": sha256(SOURCE_SNAPSHOT),
        "protocol_sha256": sha256(PROTOCOL),
        "backend": {
            "endpoint": URL,
            "model_requested": MODEL,
            "required_model_returned": RETURNED_MODEL,
            "thinking": "disabled",
            "not_official_claude_equivalent": True,
        },
        "generation": {
            "official_swan_commit": swan.EXPECTED_COMMIT,
            "official_base_prompt_and_parser": True,
            "prompt_modification": (
                "Append the frozen 7-12-word, one-sentence instruction."
            ),
            "documents": {
                "pilot": PILOT_DOCUMENTS,
                "main": MAIN_DOCUMENTS,
            },
            "slots_per_document": SLOTS,
            "max_trials_per_slot": MAX_TRIALS,
            "max_templates_per_slot": MAX_TEMPLATES,
            "retries_per_template": RETRIES_PER_TEMPLATE,
            "schedule_capacity": MAX_TEMPLATES * RETRIES_PER_TEMPLATE,
            "s2match_threshold": MATCH_THRESHOLD,
            "max_new_tokens": swan.MAX_NEW_TOKENS,
            "temperature": swan.TEMPERATURE,
            "top_p": swan.DECLARED_TOP_P,
            "early_stop_failed_document": True,
            "fallback_is_never_accepted": True,
            "uncertain_call_policy": {
                "retry_same_call": False,
                "current_document": "operational_failure",
                "later_preallocated_documents_continue": True,
                "authorization_accounting": "full_request_reserve",
            },
            "candidate_acceptance_gate": {
                "word_range": [MIN_SLOT_WORDS, MAX_SLOT_WORDS],
                "single_sentence": True,
                "terminal_punctuation": True,
                "s2match_only_after_gate": True,
            },
        },
        "quality": {
            "word_range": [MIN_WORDS, MAX_WORDS],
            "implied_word_range_from_accepted_slots": [
                SLOTS * MIN_SLOT_WORDS,
                SLOTS * MAX_SLOT_WORDS,
            ],
            "exact_sentence_units": SLOTS,
            "all_slots_accepted": True,
            "amr_leakage_forbidden": True,
            "independent_blind_reviewers": 2,
            "minimum_rating_each_dimension": 4,
        },
        "gates": {
            "pilot_to_main": (
                "At least one of five pilot documents passes injection, "
                "structure, leakage, and both blind human reviews."
            ),
            "no_source_replacement": True,
            "no_parameter_change_after_any_output": True,
            "main_requires_separate_authorization": True,
            "no_detection_before_quality_freeze": True,
            "prior_pilot_outputs_excluded_from_all_denominators": True,
        },
        "source_counts": source["cohort_counts"],
        "dependencies": dependency_hashes(),
        "packages": package_versions(),
        "resource_counts": {
            "deepseek_calls": 0,
            "bedrock_calls": 0,
            "parser_calls": 0,
            "s2match_calls": 0,
            "detector_calls": 0,
            "attack_calls": 0,
        },
    }


def budget_for_phase(phase: str, sources: list[dict]) -> dict:
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
    per_source = []
    for source in sources:
        maximum = max(
            (
                request_reserve(
                    request_body(
                        length_controlled_user_prompt(
                            module.build_user_prompt(
                                template,
                                examples,
                                context=source["generation_prompt"],
                            )
                        ),
                        system,
                        swan.MAX_NEW_TOKENS,
                        swan.TEMPERATURE,
                        swan.DECLARED_TOP_P,
                    ),
                    extra_input_tokens=CONTEXT_TOKEN_RESERVE,
                )
                for template in bank
            ),
            key=lambda row: row["peak_all_cache_miss_cost_usd"],
        )
        per_source.append(
            {
                "allocation_id": source["allocation_id"],
                **maximum,
                "calls_reserved": SLOTS * MAX_TRIALS,
            }
        )
    probe = (
        request_reserve(
            request_body("Reply exactly OK.", None, 16, 0.0, 1.0)
        )
        if phase == "pilot"
        else None
    )
    bound = sum(
        row["peak_all_cache_miss_cost_usd"] * row["calls_reserved"]
        for row in per_source
    ) + (
        probe["peak_all_cache_miss_cost_usd"] if probe is not None else 0
    )
    recommendation = math.ceil(bound * 1.25 * 100) / 100
    call_cap = PILOT_CALL_CAP if phase == "pilot" else MAIN_CALL_CAP
    return {
        "phase": phase,
        "sources": per_source,
        "probe_reserve": probe,
        "maximum_model_calls": call_cap,
        "conservative_total_bound_usd": round(bound, 6),
        "recommended_authorization": {
            "max_cost_usd": recommendation,
            "max_model_calls": call_cap,
            "contingency": "25 percent, rounded up to one cent",
        },
    }


def build_budget() -> dict:
    source = check_sources()["sources"]
    phases = {}
    for phase in ("pilot", "main"):
        rows = [row for row in source if row["cohort"] == phase]
        phases[phase] = budget_for_phase(phase, rows)
    return {
        "schema_version": 1,
        "status": "estimated_remote_unauthorized",
        "manifest_sha256": sha256(MANIFEST),
        "pricing": {
            "source": PRICE_SOURCE,
            "retrieved_at": PRICE_RETRIEVED,
            "model": MODEL,
            "peak_cache_miss_input_usd_per_million": INPUT_USD_PER_MILLION,
            "peak_output_usd_per_million": OUTPUT_USD_PER_MILLION,
            "assumption": (
                "Peak pricing; all prompt tokens billed as cache misses."
            ),
        },
        "method": {
            "input_token_bound": (
                "Serialized request UTF-8 bytes plus 1,024 tokens and a "
                "750-token maximum prior-slot context reserve."
            ),
            "output_token_bound": "Full requested max_tokens",
            "early_document_failure_can_only_reduce_use": True,
        },
        "phases": phases,
        "combined_maximum_model_calls": PILOT_CALL_CAP + MAIN_CALL_CAP,
        "combined_conservative_bound_usd": round(
            sum(row["conservative_total_bound_usd"] for row in phases.values()),
            6,
        ),
    }


def authorization_template(phase: str, budget: dict) -> dict:
    recommendation = budget["phases"][phase]["recommended_authorization"]
    return {
        "schema_version": 1,
        "status": "awaiting_explicit_terminal_authorization",
        "phase": phase,
        "manifest_sha256": sha256(MANIFEST),
        "budget_estimate_sha256": sha256(BUDGET),
        "model": MODEL,
        "endpoint": URL,
        **recommendation,
        "credential_rotated": None,
        "credential_risk_acknowledged": None,
        "explicit_user_statement": None,
        "approved_at": None,
        "active": False,
    }


def prepare(
    transport: Optional[Callable[[str], object]] = None,
) -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    allocate_sources(transport)
    if MANIFEST.exists() or BUDGET.exists():
        return check()
    manifest = freeze_json(MANIFEST, build_manifest())
    budget = freeze_json(BUDGET, build_budget())
    freeze_json(
        PILOT_AUTH_TEMPLATE,
        authorization_template("pilot", budget),
    )
    freeze_json(
        MAIN_AUTH_TEMPLATE,
        authorization_template("main", budget),
    )
    return {"manifest": manifest, "budget": budget}


def check() -> dict:
    manifest = load_json(MANIFEST)
    if manifest != build_manifest(manifest["created_at"]):
        raise ValueError("Frozen length-confirmation manifest differs")
    budget = load_json(BUDGET)
    if budget != build_budget():
        raise ValueError("Frozen length-confirmation budget differs")
    for phase, path in (
        ("pilot", PILOT_AUTH_TEMPLATE),
        ("main", MAIN_AUTH_TEMPLATE),
    ):
        if load_json(path) != authorization_template(phase, budget):
            raise ValueError(f"{phase} authorization template differs")
    return {"manifest": manifest, "budget": budget}


def authorization_path(phase: str) -> Path:
    return PILOT_AUTHORIZATION if phase == "pilot" else MAIN_AUTHORIZATION


def authorize(
    phase: str,
    statement: str,
    credential_rotated: bool,
    risk_acknowledged: bool,
) -> dict:
    state = check()
    if phase == "main":
        decision = quality_decision("pilot")
        if decision["quality_pass_documents"] < 1:
            raise PermissionError("Pilot quality gate did not pass")
    if not statement.strip():
        raise ValueError("Explicit authorization statement is required")
    if not (credential_rotated or risk_acknowledged):
        raise ValueError(
            "Credential rotation or explicit credential-risk "
            "acknowledgement is required"
        )
    recommendation = state["budget"]["phases"][phase][
        "recommended_authorization"
    ]
    payload = {
        "schema_version": 1,
        "status": "explicitly_authorized",
        "phase": phase,
        "manifest_sha256": sha256(MANIFEST),
        "budget_estimate_sha256": sha256(BUDGET),
        "model": MODEL,
        "endpoint": URL,
        **recommendation,
        "credential_rotated": credential_rotated,
        "credential_risk_acknowledged": risk_acknowledged,
        "explicit_user_statement": statement.strip(),
        "approved_at": utc_now(),
        "active": True,
    }
    return freeze_json(authorization_path(phase), payload)


def require_authorization(phase: str) -> dict:
    path = authorization_path(phase)
    if not path.is_file():
        raise PermissionError(f"{phase} execution is not authorized")
    authorization = load_json(path)
    expected = load_json(BUDGET)["phases"][phase][
        "recommended_authorization"
    ]
    checks = {
        "status": authorization.get("status") == "explicitly_authorized",
        "phase": authorization.get("phase") == phase,
        "manifest": authorization.get("manifest_sha256") == sha256(MANIFEST),
        "budget": authorization.get("budget_estimate_sha256") == sha256(BUDGET),
        "model": authorization.get("model") == MODEL,
        "endpoint": authorization.get("endpoint") == URL,
        "limits": all(
            authorization.get(key) == expected[key]
            for key in ("max_cost_usd", "max_model_calls")
        ),
        "statement": bool(
            str(authorization.get("explicit_user_statement", "")).strip()
        ),
        "credential": bool(
            authorization.get("credential_rotated")
            or authorization.get("credential_risk_acknowledged")
        ),
        "active": authorization.get("active") is True,
    }
    if not all(checks.values()):
        raise PermissionError(f"{phase} authorization contract failed")
    return authorization


class DeepSeekJournal:
    """Single-attempt, replayable journal with phase and invocation caps."""

    def __init__(
        self,
        folder: Path,
        key: str,
        phase: str,
        authorization: dict,
        max_new_calls: int,
        transport: Optional[Callable[[dict, str], dict]] = None,
    ):
        self.folder = folder
        self.key = key
        self.phase = phase
        self.authorization = authorization
        self.max_new_calls = max_new_calls
        self.new_calls = 0
        self.transport = transport or default_transport

    def inventory(self) -> list[Path]:
        return sorted((self.folder / "deepseek_calls").glob("*.json"))

    def records(self) -> list[dict]:
        rows = []
        for path in self.inventory():
            row = load_json(path)
            if row.get("status") not in {
                "complete",
                "in_flight",
                "failed_or_uncertain",
            }:
                raise ValueError(f"Invalid call status: {path.name}")
            rows.append(row)
        return rows

    def phase_records(self) -> list[dict]:
        return [
            row for row in self.records()
            if row["phase"] == self.phase
        ]

    def authorization_accounting_cost(self) -> float:
        return sum(
            (
                row["cost"]["actual"]["conservative_peak_cost_usd"]
                if (
                    row["status"] == "complete"
                    and row["cost"]["actual"] is not None
                )
                else row["cost"]["reserve"][
                    "peak_all_cache_miss_cost_usd"
                ]
            )
            for row in self.phase_records()
        )

    def remaining_new_calls(self) -> int:
        return self.max_new_calls - self.new_calls

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
                or saved["phase"] != self.phase
            ):
                raise UncertainCallExists(
                    f"Call identity is not safely replayable: {path.name}"
                )
            if saved["status"] != "complete":
                raise UncertainCallExists(
                    f"Call outcome is uncertain and will not be retried: "
                    f"{path.name}"
                )
            return {
                **saved["response"],
                "call_sha256": sha256(path),
                "replayed": True,
            }

        records = self.phase_records()
        if self.new_calls >= self.max_new_calls:
            raise InvocationBudgetExhausted(
                "Per-command DeepSeek call allowance reached"
            )
        if len(records) >= self.authorization["max_model_calls"]:
            raise InvocationBudgetExhausted(
                f"{self.phase} authorized call cap reached"
            )
        reserve = request_reserve(body)
        accounted = self.authorization_accounting_cost()
        if (
            accounted + reserve["peak_all_cache_miss_cost_usd"]
            > self.authorization["max_cost_usd"] + 1e-12
        ):
            raise InvocationBudgetExhausted(
                f"{self.phase} authorized USD cap would be exceeded"
            )

        saved = {
            "schema_version": 1,
            "call_id": call_id,
            "phase": self.phase,
            "request_sha256": request_hash,
            "request": body,
            "status": "in_flight",
            "transport_attempts": 1,
            "cost": {
                "authorization_accounted_before_call_usd": round(
                    accounted,
                    12,
                ),
                "reserve": reserve,
                "actual": None,
            },
            "started_at": utc_now(),
        }
        write_json(path, saved)
        self.new_calls += 1
        try:
            response = parse_response(self.transport(body, self.key))
            actual = actual_cost(response["usage"])
            if (
                actual["conservative_peak_cost_usd"]
                > reserve["peak_all_cache_miss_cost_usd"] + 1e-12
            ):
                raise ValueError("Actual cost exceeds request reserve")
            saved["cost"]["actual"] = actual
            saved.update(
                status="complete",
                completed_at=utc_now(),
                response=response,
            )
        except Exception as exc:
            saved.update(
                status="failed_or_uncertain",
                failed_at=utc_now(),
                error_type=type(exc).__name__,
            )
            raise UncertainCallExists(
                f"Call {call_id} is uncertain; inspect its journal"
            ) from None
        finally:
            write_json(path, saved)
        return {
            **response,
            "call_sha256": sha256(path),
            "replayed": False,
        }


@contextmanager
def patched_swan_runtime():
    """Bind local NLTK assets and this experiment's frozen identities."""
    import nltk

    modules_before = set(sys.modules)
    sys_path_before = list(sys.path)
    required_nltk = {
        "propbank": swan.NLTK_DATA / "corpora/propbank.zip",
        "punkt_tab": (
            swan.NLTK_DATA
            / "tokenizers/punkt_tab/english/ortho_context.tab"
        ),
    }
    if any(not path.is_file() for path in required_nltk.values()):
        raise FileNotFoundError("Frozen local NLTK resource is missing")
    original_download = nltk.download
    original = {
        "SOURCE_MANIFEST": swan.SOURCE_MANIFEST,
        "CONTRACT": swan.CONTRACT,
        "MAX_TRIALS": swan.MAX_TRIALS,
        "MAX_TEMPLATES": swan.MAX_TEMPLATES,
        "RETRIES_PER_TEMPLATE": swan.RETRIES_PER_TEMPLATE,
        "MATCH_THRESHOLD": swan.MATCH_THRESHOLD,
        "instrument_official_injection": (
            swan.instrument_official_injection
        ),
    }

    def local_download(package: str, *args, **kwargs):
        del args, kwargs
        path = required_nltk.get(package)
        if path is None or not path.is_file():
            raise RuntimeError(
                f"Unfrozen NLTK download request rejected: {package}"
            )
        return True

    nltk.download = local_download
    swan.SOURCE_MANIFEST = SOURCE_MANIFEST
    swan.CONTRACT = MANIFEST
    swan.MAX_TRIALS = MAX_TRIALS
    swan.MAX_TEMPLATES = MAX_TEMPLATES
    swan.RETRIES_PER_TEMPLATE = RETRIES_PER_TEMPLATE
    swan.MATCH_THRESHOLD = MATCH_THRESHOLD
    swan.instrument_official_injection = (
        instrument_length_controlled_injection
    )
    try:
        yield swan.load_official_injection()
    finally:
        nltk.download = original_download
        for name, value in original.items():
            setattr(swan, name, value)
        sys.path[:] = sys_path_before
        for name in list(sys.modules):
            if (
                name not in modules_before
                and (name == "utils" or name.startswith("utils."))
            ):
                del sys.modules[name]


def phase_sources(phase: str) -> list[dict]:
    rows = [
        row for row in check_sources()["sources"]
        if row["cohort"] == phase
    ]
    expected = PILOT_DOCUMENTS if phase == "pilot" else MAIN_DOCUMENTS
    if len(rows) != expected:
        raise ValueError(f"{phase} membership changed")
    return rows


def probe(journal: DeepSeekJournal) -> dict:
    path = OUTPUT / "model_probe.json"
    if path.exists():
        saved = load_json(path)
        if (
            saved["manifest_sha256"] != sha256(MANIFEST)
            or saved["model"] != MODEL
            or saved["response_text"] != "OK"
        ):
            raise ValueError("Model probe differs")
        return saved
    response = journal.request(
        "dslength3-model-probe",
        "Reply exactly OK.",
        None,
        16,
        0.0,
        1.0,
    )
    if response["text"] != "OK":
        raise ValueError("DeepSeek model probe failed")
    return freeze_json(
        path,
        {
            "schema_version": 1,
            "status": "complete",
            "manifest_sha256": sha256(MANIFEST),
            "authorization_sha256": sha256(PILOT_AUTHORIZATION),
            "call_sha256": response["call_sha256"],
            "model": response["model"],
            "response_text": response["text"],
            "response_text_sha256": response["text_sha256"],
        },
    )


def document_path(source: dict) -> Path:
    return OUTPUT / "generated_documents" / f"{source['allocation_id']}.json"


def build_document(source: dict, slots: list[dict]) -> dict:
    accepted = len(slots) == SLOTS and all(row["accepted"] for row in slots)
    text = " ".join(row["text"] for row in slots) if accepted else ""
    return {
        "schema_version": 1,
        "allocation_id": source["allocation_id"],
        "cohort": source["cohort"],
        "status": "injection_complete" if accepted else "injection_failed",
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "manifest_sha256": sha256(MANIFEST),
        "completed_slots": len(slots),
        "required_slots": SLOTS,
        "all_slots_accepted": accepted,
        "failed_slot": (
            next(
                (
                    row["slot"]
                    for row in slots
                    if not row["accepted"]
                ),
                None,
            )
            if not accepted
            else None
        ),
        "slot_sha256": [
            sha256(
                OUTPUT
                / "generation_slots"
                / (
                    f"{source['allocation_id']}-slot-"
                    f"{row['slot']:02d}.json"
                )
            )
            for row in slots
        ],
        "slot_texts": [row["text"] for row in slots] if accepted else [],
        "slot_templates": (
            [row["chosen_template"] for row in slots] if accepted else []
        ),
        "text": text,
        "text_sha256": text_sha256(text),
    }


def build_operational_failure_document(
    source: dict,
    slots: list[dict],
    failed_slot: int,
) -> dict:
    failure_path = (
        OUTPUT
        / "generation_failures"
        / f"{source['allocation_id']}-slot-{failed_slot:02d}.json"
    )
    failure = load_json(failure_path)
    attempt = failure["attempts_recorded"] - 1
    call_id = (
        f"{source['allocation_id']}-slot-{failed_slot:02d}-"
        f"attempt-{attempt:02d}"
    )
    call_path = OUTPUT / "deepseek_calls" / f"{call_id}.json"
    call = load_json(call_path)
    if (
        call["status"] not in {"in_flight", "failed_or_uncertain"}
        or call["phase"] != source["cohort"]
    ):
        raise ValueError("Operational failure lacks an uncertain call")
    return {
        "schema_version": 1,
        "allocation_id": source["allocation_id"],
        "cohort": source["cohort"],
        "status": "operational_failure",
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "manifest_sha256": sha256(MANIFEST),
        "completed_slots": len(slots),
        "required_slots": SLOTS,
        "all_slots_accepted": False,
        "failed_slot": failed_slot,
        "slot_sha256": [
            sha256(
                OUTPUT
                / "generation_slots"
                / (
                    f"{source['allocation_id']}-slot-"
                    f"{row['slot']:02d}.json"
                )
            )
            for row in slots
        ],
        "slot_texts": [],
        "slot_templates": [],
        "text": "",
        "text_sha256": text_sha256(""),
        "operational_failure": {
            "rule": (
                "Uncertain call is not retried; current document fails and "
                "later preallocated documents continue."
            ),
            "generation_failure_sha256": sha256(failure_path),
            "uncertain_call_id": call_id,
            "uncertain_call_sha256": sha256(call_path),
            "uncertain_call_status": call["status"],
            "error_type": call.get("error_type"),
            "authorization_reserve_usd": call["cost"]["reserve"][
                "peak_all_cache_miss_cost_usd"
            ],
        },
    }


def run_phase(
    phase: str,
    key_file: Path,
    max_new_calls: int,
    transport: Optional[Callable[[dict, str], dict]] = None,
) -> dict:
    check()
    if max_new_calls < 1:
        raise InvocationBudgetExhausted(
            "Run requires a positive --max-new-calls"
        )
    if phase == "main":
        decision = quality_decision("pilot")
        if decision["quality_pass_documents"] < 1:
            raise PermissionError("Pilot quality gate did not pass")
    authorization = require_authorization(phase)
    journal = DeepSeekJournal(
        OUTPUT,
        read_credential(key_file),
        phase,
        authorization,
        max_new_calls,
        transport,
    )
    if phase == "pilot":
        probe(journal)
    if journal.remaining_new_calls() < MAX_TRIALS:
        return status()

    bank_records = load_json(
        swan.OFFICIAL_ROOT / "amr_bank/banks/amr_bank_50.json"
    )
    bank = [row["normalized_template_amr"] for row in bank_records]
    if len(bank) != 50 or len(set(bank)) != 50:
        raise ValueError("Official bank50 changed")
    examples_path = (
        swan.OFFICIAL_ROOT
        / "amr_bank/artifacts/normalized_amr_examples.json"
    )

    with patched_swan_runtime() as module:
        for source in phase_sources(phase):
            destination = document_path(source)
            if destination.exists():
                continue
            slots = []
            for slot in range(SLOTS):
                slot_path = (
                    OUTPUT
                    / "generation_slots"
                    / f"{source['allocation_id']}-slot-{slot:02d}.json"
                )
                if not slot_path.exists() and (
                    journal.remaining_new_calls() < MAX_TRIALS
                ):
                    return status()
                context = source["generation_prompt"]
                if slots:
                    context += " " + " ".join(row["text"] for row in slots)
                try:
                    result = swan.run_official_slot(
                        module,
                        OUTPUT,
                        source,
                        slot,
                        context,
                        bank,
                        examples_path,
                        journal,
                    )
                except UncertainCallExists:
                    freeze_json(
                        destination,
                        build_operational_failure_document(
                            source,
                            slots,
                            slot,
                        ),
                    )
                    break
                slots.append(result)
                if not result["accepted"]:
                    break
            if not destination.exists():
                freeze_json(destination, build_document(source, slots))
    return summarize()


def structural_evaluation(document: dict) -> dict:
    if not document["all_slots_accepted"]:
        return {
            "all_slots_accepted": False,
            "word_count": 0,
            "word_range_pass": False,
            "sentence_units": 0,
            "sentence_units_pass": False,
            "each_slot_one_sentence": False,
            "amr_leakage": False,
            "structural_pass": False,
        }
    import nltk

    os.environ.setdefault("NLTK_DATA", str(swan.NLTK_DATA))
    text = document["text"]
    sentences = nltk.sent_tokenize(text)
    each_slot = all(
        nltk.sent_tokenize(slot) == [slot]
        for slot in document["slot_texts"]
    )
    words = len(WORD_RE.findall(text))
    leakage = bool(AMR_LEAK_RE.search(text))
    checks = {
        "all_slots_accepted": True,
        "word_count": words,
        "word_range_pass": MIN_WORDS <= words <= MAX_WORDS,
        "sentence_units": len(sentences),
        "sentence_units_pass": len(sentences) == SLOTS,
        "each_slot_one_sentence": each_slot,
        "amr_leakage": leakage,
    }
    checks["structural_pass"] = (
        checks["word_range_pass"]
        and checks["sentence_units_pass"]
        and each_slot
        and not leakage
    )
    return checks


def phase_terminal(phase: str) -> bool:
    return all(document_path(row).is_file() for row in phase_sources(phase))


def quality_paths(phase: str) -> dict[str, Path]:
    folder = OUTPUT / "quality" / phase
    return {
        "packet": folder / "blind_packet.json",
        "index": folder / "private_index.json",
        "reviewer_a": folder / "reviewer-a.json",
        "reviewer_b": folder / "reviewer-b.json",
        "decision": folder / "decision.json",
    }


def export_quality(phase: str) -> dict:
    check()
    if not phase_terminal(phase):
        raise RuntimeError(f"{phase} generation is not terminal")
    packet_rows = []
    index_rows = []
    for source in phase_sources(phase):
        document = load_json(document_path(source))
        structure = structural_evaluation(document)
        if not structure["structural_pass"]:
            continue
        blind_id = text_sha256(
            "swan-dslength3-quality:"
            + phase
            + ":"
            + source["allocation_id"]
        )[:16]
        packet_rows.append(
            {
                "blind_id": blind_id,
                "source_excerpt": source["generation_prompt"],
                "candidate": document["text"],
                "instructions": (
                    "Judge only fluency, coherence, nonredundancy, source "
                    "faithfulness, major semantic/factual error, and visible "
                    "AMR/template leakage. Do not use detector information."
                ),
            }
        )
        index_rows.append(
            {
                "blind_id": blind_id,
                "allocation_id": source["allocation_id"],
                "document_sha256": sha256(document_path(source)),
                "structural": structure,
            }
        )
    paths = quality_paths(phase)
    packet = freeze_json(
        paths["packet"],
        {
            "schema_version": 1,
            "phase": phase,
            "blind": True,
            "model_identity_hidden": True,
            "detector_results_available": False,
            "documents": packet_rows,
        },
    )
    freeze_json(
        paths["index"],
        {
            "schema_version": 1,
            "phase": phase,
            "packet_sha256": sha256(paths["packet"]),
            "documents": index_rows,
        },
    )
    template = {
        "schema_version": 1,
        "phase": phase,
        "reviewer_id": None,
        "independent_review": True,
        "packet_sha256": sha256(paths["packet"]),
        "reviews": [
            {
                "blind_id": row["blind_id"],
                "fluency": None,
                "coherence": None,
                "nonredundancy": None,
                "source_faithfulness": None,
                "major_error": None,
                "amr_or_template_leakage": None,
                "overall_pass": None,
                "notes": "",
            }
            for row in packet_rows
        ],
    }
    for key in ("reviewer_a", "reviewer_b"):
        if not paths[key].exists():
            write_json(paths[key], template)
    return {
        "status": "quality_packet_ready",
        "phase": phase,
        "packet": str(paths["packet"]),
        "eligible_documents": len(packet_rows),
        "review_files": [
            str(paths["reviewer_a"]),
            str(paths["reviewer_b"]),
        ],
        "packet_payload": packet,
    }


def validate_review(path: Path, packet: dict) -> dict:
    review = load_json(path)
    expected_ids = [row["blind_id"] for row in packet["documents"]]
    rows = review.get("reviews")
    if (
        not str(review.get("reviewer_id", "")).strip()
        or review.get("independent_review") is not True
        or review.get("packet_sha256")
        != sha256(quality_paths(packet["phase"])["packet"])
        or not isinstance(rows, list)
        or [row.get("blind_id") for row in rows] != expected_ids
    ):
        raise ValueError(f"Invalid blind review envelope: {path}")
    for row in rows:
        ratings = [
            row.get(key)
            for key in (
                "fluency",
                "coherence",
                "nonredundancy",
                "source_faithfulness",
            )
        ]
        if not all(type(value) is int and 1 <= value <= 5 for value in ratings):
            raise ValueError(f"Invalid rating in {path}")
        for key in (
            "major_error",
            "amr_or_template_leakage",
            "overall_pass",
        ):
            if type(row.get(key)) is not bool:
                raise ValueError(f"Invalid {key} in {path}")
    return review


def review_row_pass(row: dict) -> bool:
    return (
        all(
            row[key] >= 4
            for key in (
                "fluency",
                "coherence",
                "nonredundancy",
                "source_faithfulness",
            )
        )
        and not row["major_error"]
        and not row["amr_or_template_leakage"]
        and row["overall_pass"]
    )


def finalize_quality(phase: str) -> dict:
    paths = quality_paths(phase)
    packet = load_json(paths["packet"])
    index = load_json(paths["index"])
    review_a = validate_review(paths["reviewer_a"], packet)
    review_b = validate_review(paths["reviewer_b"], packet)
    if review_a["reviewer_id"] == review_b["reviewer_id"]:
        raise ValueError("Two distinct reviewer IDs are required")
    private = {
        row["blind_id"]: row for row in index["documents"]
    }
    rows = []
    for left, right in zip(review_a["reviews"], review_b["reviews"]):
        blind_id = left["blind_id"]
        mapping = private[blind_id]
        rows.append(
            {
                "allocation_id": mapping["allocation_id"],
                "blind_id": blind_id,
                "structural_pass": mapping["structural"]["structural_pass"],
                "reviewer_a_pass": review_row_pass(left),
                "reviewer_b_pass": review_row_pass(right),
                "quality_pass": (
                    mapping["structural"]["structural_pass"]
                    and review_row_pass(left)
                    and review_row_pass(right)
                ),
            }
        )
    payload = {
        "schema_version": 1,
        "status": "quality_frozen_before_detection",
        "phase": phase,
        "manifest_sha256": sha256(MANIFEST),
        "packet_sha256": sha256(paths["packet"]),
        "review_sha256": [
            sha256(paths["reviewer_a"]),
            sha256(paths["reviewer_b"]),
        ],
        "reviewer_ids": [
            review_a["reviewer_id"],
            review_b["reviewer_id"],
        ],
        "structurally_eligible_documents": len(rows),
        "quality_pass_documents": sum(row["quality_pass"] for row in rows),
        "documents": rows,
        "main_gate_passed": (
            phase == "pilot" and any(row["quality_pass"] for row in rows)
        ),
    }
    return freeze_json(paths["decision"], payload)


def quality_decision(phase: str) -> dict:
    path = quality_paths(phase)["decision"]
    if not path.is_file():
        raise RuntimeError(f"{phase} quality decision is not frozen")
    decision = load_json(path)
    if (
        decision["status"] != "quality_frozen_before_detection"
        or decision["phase"] != phase
        or decision["manifest_sha256"] != sha256(MANIFEST)
    ):
        raise ValueError(f"{phase} quality decision differs")
    return decision


def summarize() -> dict:
    check()
    documents = []
    for phase in ("pilot", "main"):
        for source in phase_sources(phase):
            path = document_path(source)
            if path.exists():
                row = load_json(path)
                documents.append(
                    {
                        "allocation_id": source["allocation_id"],
                        "phase": phase,
                        "status": row["status"],
                        "completed_slots": row["completed_slots"],
                        "structural": structural_evaluation(row),
                    }
                )
    calls = sorted((OUTPUT / "deepseek_calls").glob("*.json"))
    attempts = sorted((OUTPUT / "generation_attempts").glob("*.json"))
    phase_rows = {}
    for phase, expected in (
        ("pilot", PILOT_DOCUMENTS),
        ("main", MAIN_DOCUMENTS),
    ):
        rows = [row for row in documents if row["phase"] == phase]
        quality_path = quality_paths(phase)["decision"]
        phase_rows[phase] = {
            "allocated_documents": expected,
            "terminal_documents": len(rows),
            "injection_complete_documents": sum(
                row["status"] == "injection_complete" for row in rows
            ),
            "operational_failure_documents": sum(
                row["status"] == "operational_failure" for row in rows
            ),
            "structural_pass_documents": sum(
                row["structural"]["structural_pass"] for row in rows
            ),
            "quality_frozen": quality_path.is_file(),
            "quality_pass_documents": (
                load_json(quality_path)["quality_pass_documents"]
                if quality_path.is_file()
                else None
            ),
        }
    call_rows = [load_json(path) for path in calls]
    result = {
        "schema_version": 1,
        "status": (
            "main_quality_complete"
            if phase_rows["main"]["quality_frozen"]
            else "main_generation_terminal"
            if phase_rows["main"]["terminal_documents"] == MAIN_DOCUMENTS
            else "pilot_quality_complete"
            if phase_rows["pilot"]["quality_frozen"]
            else "pilot_generation_terminal"
            if phase_rows["pilot"]["terminal_documents"] == PILOT_DOCUMENTS
            else "execution_pending_or_paused"
        ),
        "phases": phase_rows,
        "deepseek_calls": len(calls),
        "generation_attempts": len(attempts),
        "calls_by_phase": {
            phase: sum(row.get("phase") == phase for row in call_rows)
            for phase in ("pilot", "main")
        },
        "uncertain_calls": sum(
            row.get("status") != "complete" for row in call_rows
        ),
        "conservative_peak_cost_usd": round(
            sum(
                row["cost"]["actual"]["conservative_peak_cost_usd"]
                for row in call_rows
                if (
                    row.get("status") == "complete"
                    and row["cost"]["actual"] is not None
                )
            ),
            9,
        ),
        "authorization_accounting_usd": round(
            sum(
                (
                    row["cost"]["actual"][
                        "conservative_peak_cost_usd"
                    ]
                    if (
                        row.get("status") == "complete"
                        and row["cost"]["actual"] is not None
                    )
                    else row["cost"]["reserve"][
                        "peak_all_cache_miss_cost_usd"
                    ]
                )
                for row in call_rows
            ),
            9,
        ),
        "claim_boundary": (
            "DeepSeek is a non-equivalent generation backend. No detector, "
            "attack, TPR, or ASR claim is authorized by this summary."
        ),
    }
    write_json(OUTPUT / "summary.json", result)
    return result


def status() -> dict:
    if not SOURCE_MANIFEST.exists():
        return {
            "status": "awaiting_source_preparation",
            "next": "prepare",
            "model_calls": 0,
        }
    if not MANIFEST.exists():
        return {
            "status": "source_frozen_manifest_pending",
            "next": "prepare",
            "model_calls": 0,
        }
    return summarize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "check",
            "authorize",
            "run",
            "status",
            "summarize",
            "export-quality",
            "finalize-quality",
        ),
    )
    parser.add_argument("--phase", choices=("pilot", "main"), default="pilot")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--max-new-calls", type=int, default=0)
    parser.add_argument("--statement", default="")
    parser.add_argument("--credential-rotated", action="store_true")
    parser.add_argument(
        "--acknowledge-credential-risk",
        action="store_true",
    )
    args = parser.parse_args()
    os.environ.setdefault("NLTK_DATA", str(swan.NLTK_DATA))
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))

    if args.stage == "prepare":
        result = prepare()
    elif args.stage == "check":
        result = check()
    elif args.stage == "authorize":
        result = authorize(
            args.phase,
            args.statement,
            args.credential_rotated,
            args.acknowledge_credential_risk,
        )
    elif args.stage == "run":
        if args.key_file is None:
            parser.error("run requires an explicit --key-file")
        result = run_phase(
            args.phase,
            args.key_file,
            args.max_new_calls,
        )
    elif args.stage == "status":
        result = status()
    elif args.stage == "summarize":
        result = summarize()
    elif args.stage == "export-quality":
        result = export_quality(args.phase)
    else:
        result = finalize_quality(args.phase)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
