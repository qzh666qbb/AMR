"""Offline audit and no-compute recovery for constrained microfacts."""

import argparse
import json
import math
from pathlib import Path

from run_confirmation_experiment import CallJournal, digest
from run_segmentation_intervention import sha, text_sha, write_json
from run_swan_atomic_fact_preflight import analyze_parser_output, s2_counts
from run_swan_constrained_microfact import (
    API_CAP,
    CANDIDATES,
    FACTS,
    OUTPUT,
    QWEN_PATH,
    SYSTEM,
    candidate_metrics,
    check,
    choose_candidate,
    decode_baseline_review,
    decode_grounding_review,
    document_structure,
    generated_path,
    generation_prompt,
    generation_seed,
    grounding_body,
    local_runtime_path,
    quality_gates,
    review_runtime_path,
    run,
    safe_token_ids,
    slot_path,
    style_body,
    summarize,
)
from run_swan_bank100_microfact_null import BANK_PATH
from run_swan_known_positive_audit import load_utils


def validate_token_path(tokenizer, token_ids):
    safe = set(safe_token_ids(tokenizer))
    period = tokenizer.encode(".", add_special_tokens=False)[0]
    eos = tokenizer.eos_token_id
    prefix = []
    for position, token_id in enumerate(token_ids):
        text = tokenizer.decode(prefix, skip_special_tokens=True).strip()
        words = len(text.split())
        if text.endswith("."):
            allowed = token_id == eos
        elif words >= 10:
            allowed = token_id == period
        else:
            allowed = token_id in safe or (
                words >= 7 and token_id == period
            )
        if not allowed:
            raise ValueError(
                f"Illegal constrained token at {position}: {token_id}"
            )
        prefix.append(token_id)
        if token_id == eos and position != len(token_ids) - 1:
            raise ValueError("Tokens occur after constrained EOS")
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def rendered_prompt(tokenizer, prompt):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    tokens = tokenizer(rendered, return_tensors="pt")["input_ids"].shape[1]
    return rendered, tokens


def audit_generation(folder, manifest, tokenizer, local_runtime):
    import nltk

    expected_safe = safe_token_ids(tokenizer)
    if (
        local_runtime["safe_token_count"] != len(expected_safe)
        or local_runtime["safe_token_sha256"] != digest(expected_safe)
        or local_runtime["period_token_id"]
        != tokenizer.encode(".", add_special_tokens=False)[0]
        or local_runtime["eos_token_id"] != tokenizer.eos_token_id
    ):
        raise ValueError("Local tokenizer runtime mismatch")
    checked = 0
    candidates_checked = 0
    for source in manifest["sources"]:
        previous = []
        slot_hashes = []
        selection_passes = 0
        for slot in range(FACTS):
            path = slot_path(folder, source["prompt_id"], slot)
            row = json.loads(path.read_text())
            prompt = generation_prompt(source["source"], previous, slot)
            rendered, prompt_tokens = rendered_prompt(tokenizer, prompt)
            identity = {
                "prompt_id": source["prompt_id"],
                "slot": slot,
                "source_sha256": source["source_sha256"],
                "previous_sha256": digest(previous),
                "prompt_sha256": text_sha(prompt),
                "seed": generation_seed(source["prompt_id"], slot),
                "local_runtime_sha256": digest(local_runtime),
            }
            if any(row[key] != value for key, value in identity.items()):
                raise ValueError("Generation identity mismatch")
            if (
                row["rendered_prompt_sha256"] != text_sha(rendered)
                or row["prompt_tokens"] != prompt_tokens
                or not math.isfinite(row["elapsed_seconds"])
                or row["elapsed_seconds"] < 0
                or len(row["candidates"]) != CANDIDATES
            ):
                raise ValueError("Generation runtime metadata mismatch")
            texts = []
            expected_candidates = []
            for index, candidate in enumerate(row["candidates"]):
                decoded = validate_token_path(
                    tokenizer,
                    candidate["token_ids"],
                )
                if decoded != candidate["text"]:
                    raise ValueError("Saved token IDs do not decode to text")
                texts.append(decoded)
                expected_candidates.append(
                    {
                        "candidate": index,
                        "text": decoded,
                        "text_sha256": text_sha(decoded),
                        "metrics": candidate_metrics(
                            decoded,
                            source["source"],
                            previous,
                            nltk.sent_tokenize,
                        ),
                        "token_ids": candidate["token_ids"],
                    }
                )
                candidates_checked += 1
            selection = choose_candidate(
                texts,
                source["source"],
                previous,
                nltk.sent_tokenize,
            )
            expected = {
                **identity,
                "rendered_prompt_sha256": text_sha(rendered),
                "prompt_tokens": prompt_tokens,
                "candidates": expected_candidates,
                "selected_candidate": selection["selected_candidate"],
                "selected_text": selection["selected_text"],
                "selection_pass": selection["selection_pass"],
                "fallback_reason": selection["fallback_reason"],
                "elapsed_seconds": row["elapsed_seconds"],
            }
            if row != expected:
                raise ValueError("Candidate metrics or selection mismatch")
            previous.append(row["selected_text"])
            selection_passes += row["selection_pass"]
            slot_hashes.append(sha(path))
            checked += 1
        facts = previous
        structure = document_structure(facts, nltk.sent_tokenize)
        expected_document = {
            "prompt_id": source["prompt_id"],
            "source_sha256": source["source_sha256"],
            "local_runtime_sha256": digest(local_runtime),
            "slot_sha256": slot_hashes,
            "facts": facts,
            "passage": " ".join(facts),
            "passage_sha256": text_sha(" ".join(facts)),
            "structure": structure,
            "selection": {
                "eligible_slots": selection_passes,
                "fallback_slots": FACTS - selection_passes,
                "minimum_eligible_slots": 12,
                "pass": bool(structure["pass"] and selection_passes >= 12),
            },
        }
        saved_document = json.loads(
            generated_path(folder, source["prompt_id"]).read_text()
        )
        if saved_document != expected_document:
            raise ValueError("Generated document mismatch")
    return checked, candidates_checked


class Replay(CallJournal):
    def __init__(self, folder):
        super().__init__(folder, "", API_CAP)

    def request(self, scope, body):
        ordinal = self.counters.get(scope, 0)
        path = self.folder / "calls" / f"{scope}-{ordinal:03d}.json"
        if not path.exists():
            raise ValueError("Audit replay would require an API call")
        return super().request(scope, body)


def audit_reviews(folder, manifest, runtime):
    api = Replay(folder)
    review_runtime = runtime["review"]
    probe_body = __import__(
        "run_confirmation_experiment",
        fromlist=["request_body"],
    ).request_body("Reply exactly OK.", 0, 16)
    probe = api.request("probe", probe_body)
    if (
        probe["model"] != review_runtime["model_returned"]
        or probe["id"] != review_runtime["probe_id"]
        or probe["content"].strip() != "OK"
    ):
        raise ValueError("Review probe mismatch")
    api.model_returned = review_runtime["model_returned"]
    for source in manifest["sources"]:
        generated = json.loads(
            generated_path(folder, source["prompt_id"]).read_text()
        )
        style_request = style_body(generated["passage"])
        grounding_request = grounding_body(
            source["source"],
            generated["facts"],
        )
        style_response = api.request(
            f"style-{source['prompt_id']}",
            style_request,
        )
        grounding_response = api.request(
            f"facts-{source['prompt_id']}",
            grounding_request,
        )
        errors = {}
        try:
            style = decode_baseline_review(style_response["content"])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            style = None
            errors["style"] = type(exc).__name__
        try:
            grounding = decode_grounding_review(
                grounding_response["content"]
            )
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            grounding = None
            errors["grounding"] = type(exc).__name__
        gates = quality_gates(generated, style, grounding)
        expected = {
            "prompt_id": source["prompt_id"],
            "source_sha256": source["source_sha256"],
            "passage_sha256": generated["passage_sha256"],
            "runtime_sha256": digest(runtime),
            "style_request_sha256": digest(style_request),
            "grounding_request_sha256": digest(grounding_request),
            "style_response": style_response,
            "grounding_response": grounding_response,
            "style": style,
            "grounding": grounding,
            "decode_errors": errors,
            "gates": gates,
            "quality_pass": all(gates.values()),
        }
        saved = json.loads(
            (folder / "reviews" / f"{source['prompt_id']}.json").read_text()
        )
        if saved != expected:
            raise ValueError("Quality review reconstruction mismatch")
    return sum(api.counters.values()), api.sent


def audit_parser(folder, manifest, runtime):
    utils = load_utils()
    bank = json.loads(BANK_PATH.read_text())
    bank_counts = [
        s2_counts(row["normalized_template_amr"])
        for row in bank
    ]
    checked = 0
    for source in manifest["sources"]:
        review = json.loads(
            (folder / "reviews" / f"{source['prompt_id']}.json").read_text()
        )
        path = folder / "parser" / f"{source['prompt_id']}.json"
        if not review["quality_pass"]:
            if path.exists():
                raise ValueError("Parser exists for quality-failed document")
            continue
        parsed = json.loads(path.read_text())
        generated = json.loads(
            generated_path(folder, source["prompt_id"]).read_text()
        )
        rebuilt = []
        for index, (sentence, saved) in enumerate(
            zip(generated["facts"], parsed["sentences"])
        ):
            identity = {
                "sentence_index": index,
                "text": sentence,
                "text_sha256": text_sha(sentence),
                "words": len(sentence.split()),
            }
            row = analyze_parser_output(
                identity,
                saved["raw_parser_output"],
                utils,
                bank_counts,
            )
            row["input_tokens"] = saved["input_tokens"]
            rebuilt.append(row)
        expected = {
            **{
                key: parsed[key]
                for key in ("prompt_id", "source_sha256", "passage_sha256")
            },
            "runtime_sha256": digest(runtime),
            "sentences": rebuilt,
            "elapsed_seconds": parsed["elapsed_seconds"],
        }
        if (
            parsed != expected
            or len(rebuilt) != FACTS
            or parsed["prompt_id"] != source["prompt_id"]
            or parsed["source_sha256"] != source["source_sha256"]
            or parsed["passage_sha256"] != generated["passage_sha256"]
            or not math.isfinite(parsed["elapsed_seconds"])
        ):
            raise ValueError("Parser reconstruction mismatch")
        checked += len(rebuilt)
    return checked


def audit(folder=OUTPUT):
    manifest = check(folder)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(QWEN_PATH),
        local_files_only=True,
    )
    local_runtime = json.loads(local_runtime_path(folder).read_text())
    review_runtime = json.loads(review_runtime_path(folder).read_text())
    runtime = {"local": local_runtime, "review": review_runtime}
    slots, candidates = audit_generation(
        folder,
        manifest,
        tokenizer,
        local_runtime,
    )
    api_calls, new_api = audit_reviews(folder, manifest, runtime)
    parser_sentences = audit_parser(folder, manifest, runtime)
    summary = summarize(folder)
    if (
        summary["status"] != "complete"
        or summary["generated_slots"] != slots
        or summary["generated_candidates"] != candidates
        or summary["api_attempts"] != api_calls
        or summary["neural_parser_sentences"] != parser_sentences
        or new_api != 0
        or summary["new_s2match_calls"] != 0
    ):
        raise ValueError("Summary accounting mismatch")
    result = {
        "status": "passed",
        "mismatches": 0,
        "local_slots_checked": slots,
        "candidate_token_paths_checked": candidates,
        "api_calls_replayed": api_calls,
        "parser_sentences_reconstructed": parser_sentences,
        "new_api_calls": new_api,
        "new_local_model_calls": 0,
        "new_neural_parser_calls": 0,
        "new_s2match_calls": 0,
        "summary_sha256": sha(folder / "summary.json"),
    }
    write_json(folder / "audit.json", result)
    return result


def stable_files(folder):
    paths = [
        folder / "manifest.json",
        local_runtime_path(folder),
        review_runtime_path(folder),
        folder / "summary.json",
        folder / "report.md",
        folder / "audit.json",
    ]
    for subfolder in ("calls", "slots", "generated", "reviews", "parser"):
        paths.extend(sorted((folder / subfolder).glob("*.json")))
    return [path for path in paths if path.exists()]


def recovery(folder=OUTPUT):
    before_paths = stable_files(folder)
    before = {
        str(path.relative_to(folder)): sha(path)
        for path in before_paths
    }
    aggregate_before = digest(before)

    import amrlib
    import requests
    import transformers

    original_post = requests.post
    original_model = transformers.AutoModelForCausalLM.from_pretrained
    original_parser = amrlib.load_stog_model

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Recovery attempted forbidden computation")

    requests.post = forbidden
    transformers.AutoModelForCausalLM.from_pretrained = forbidden
    amrlib.load_stog_model = forbidden
    try:
        result = run(folder, "")
    finally:
        requests.post = original_post
        transformers.AutoModelForCausalLM.from_pretrained = original_model
        amrlib.load_stog_model = original_parser

    after_paths = stable_files(folder)
    after = {
        str(path.relative_to(folder)): sha(path)
        for path in after_paths
    }
    aggregate_after = digest(after)
    if before != after or aggregate_before != aggregate_after:
        raise ValueError("Recovery changed stable artifacts")
    verification = {
        "status": "complete_no_compute_recovery",
        "decision": result["decision"],
        "files_checked": len(before),
        "stable_aggregate_sha256": aggregate_before,
        "network_forbidden": True,
        "local_generation_model_forbidden": True,
        "neural_parser_forbidden": True,
        "new_api_calls": 0,
        "new_local_model_calls": 0,
        "new_neural_parser_calls": 0,
        "new_s2match_calls": 0,
    }
    write_json(folder / "recovery_verification.json", verification)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "recovery"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = audit(args.output) if args.stage == "audit" else recovery(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
