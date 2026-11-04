"""Run WP2-v14 termination-safe, score-blind SemStamp generation."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import run_certified_resegmentation_wp2_v12_semstamp_smoke as smoke
import run_certified_resegmentation_wp2_v13_generation as parent
from semstamp_bounded_generation import (
    PATCH_ID,
    bounded_lsh_reject_completion,
)


STAGE = "WP2-v14-G1-SemStamp-bounded-generation"
PROTOCOL_SHA256 = "5a27c3c541b885510fc4b1f5e3647a77540367fd980e5cc824509d35df6cb4be"
CONTRACT_SHA256 = "784e43ec624f169edf1491bcda3aed1065f04f64d228f7769748167c6b1b786b"
PARENT_RUNNER_SHA256 = "9a2785b302a4a66b2a432eb148b08d2856fa4308693d9117854e5ef8ceedb87b"
BOUNDED_SAMPLER_SHA256 = "f00c6ddbad960b64aa386e06e519ed9c3011a70b8726fabeb9b1643828ef93cb"
V13_RUNTIME_STOP_SHA256 = "387eb8c2c1f3226c85f7bd6fb8bbf851d5b36f6d9bf99e0bedd849607abbdfac"
V13_PREFIX_SHA256 = "607c7bc63c7271419c063d4d31cfc2b1fbf8b0e91a533c130cf20ec4851cce2e"
PREFIX_ATTEMPTS = 372
TARGET_ELIGIBLE = parent.TARGET_ELIGIBLE
MAX_ATTEMPTS = parent.MAX_ATTEMPTS
MIN_FREE_BYTES = parent.MIN_FREE_BYTES
MAX_VRAM_FRACTION = parent.MAX_VRAM_FRACTION


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def verify_inputs(args):
    parent.verify_file(
        Path(parent.__file__).resolve(),
        PARENT_RUNNER_SHA256,
        "frozen v13 parent runner",
    )
    parent.verify_file(
        Path(__file__).with_name("semstamp_bounded_generation.py"),
        BOUNDED_SAMPLER_SHA256,
        "bounded sampler",
    )
    parent.verify_file(
        args.v13_runtime_stop,
        V13_RUNTIME_STOP_SHA256,
        "v13 runtime Stop evidence",
    )
    parent.verify_file(
        args.v13_prefix,
        V13_PREFIX_SHA256,
        "v13 committed prefix",
    )
    original_protocol = parent.PROTOCOL_SHA256
    original_contract = parent.CONTRACT_SHA256
    try:
        parent.PROTOCOL_SHA256 = PROTOCOL_SHA256
        parent.CONTRACT_SHA256 = CONTRACT_SHA256
        inherited = parent.verify_inputs(args)
    finally:
        parent.PROTOCOL_SHA256 = original_protocol
        parent.CONTRACT_SHA256 = original_contract

    stop = json.loads(args.v13_runtime_stop.read_text(encoding="utf-8"))
    heartbeat = stop.get("heartbeat", {})
    if (
        stop.get("stage") != "WP2-v13-G1-runtime-stop-evidence"
        or stop.get("reason") != "nonterminating_attempt_372"
        or stop.get("sealed_checkpoint_content_parsed") is not False
        or heartbeat.get("attempt_count") != PREFIX_ATTEMPTS
        or heartbeat.get("eligible_count") != 326
        or heartbeat.get("detector_scores_computed") != 0
        or heartbeat.get("certificate_calls") != 0
    ):
        raise ValueError("v13 runtime Stop evidence mismatch")
    prefix = parent.read_jsonl(args.v13_prefix)
    if len(prefix) != PREFIX_ATTEMPTS:
        raise ValueError("v13 committed prefix count mismatch")
    eligible_count = 0
    for index, row in enumerate(prefix):
        codes = row.get("exclusion_codes")
        eligible = row.get("eligible")
        if (
            row.get("attempt_index") != index
            or type(eligible) is not bool
            or not isinstance(codes, list)
            or codes != sorted(set(codes))
            or eligible is not (not codes)
            or not is_sha256(row.get("output_raw_text_sha256"))
        ):
            raise ValueError(f"invalid v13 committed prefix row {index}")
        eligible_count += int(eligible)
        if row.get("eligible_count") != eligible_count:
            raise ValueError("v13 committed prefix cumulative count mismatch")
    if eligible_count != 326:
        raise ValueError("v13 committed prefix eligible count mismatch")
    return (*inherited, prefix)


def validate_sampling_metrics(metrics):
    expected_keys = {
        "patch_id",
        "max_trials",
        "total_trials",
        "accepted_segments",
        "max_trials_reached_segments",
        "fallback_acceptances",
        "max_current_trials_observed",
        "terminated_with_empty_generation",
    }
    if not isinstance(metrics, dict) or set(metrics) != expected_keys:
        raise ValueError("sampling metrics schema mismatch")
    if (
        metrics["patch_id"] != PATCH_ID
        or metrics["max_trials"]
        != smoke.GENERATION_PARAMETERS["max_trials"]
        or type(metrics["terminated_with_empty_generation"]) is not bool
    ):
        raise ValueError("sampling metrics contract mismatch")
    for key in (
        "total_trials",
        "accepted_segments",
        "max_trials_reached_segments",
        "fallback_acceptances",
        "max_current_trials_observed",
    ):
        if type(metrics[key]) is not int or metrics[key] < 0:
            raise ValueError(f"invalid sampling metric {key}")
    if (
        metrics["max_current_trials_observed"] > metrics["max_trials"]
        or metrics["max_trials_reached_segments"]
        > metrics["accepted_segments"]
        or metrics["fallback_acceptances"]
        > metrics["max_trials_reached_segments"]
        or metrics["total_trials"]
        > (metrics["accepted_segments"] + 1) * metrics["max_trials"]
    ):
        raise ValueError("bounded sampling invariant failed")
    return metrics


def prefix_comparison(row, expected, eligible_count):
    if expected is None:
        return None
    actual = {
        "attempt_index": row["attempt_index"],
        "output_raw_text_sha256": row["output"]["raw_text_sha256"],
        "eligible": row["eligible"],
        "exclusion_codes": row["exclusion_codes"],
        "eligible_count": eligible_count,
    }
    return {
        "expected_sha256": sha256_bytes(
            json.dumps(
                expected, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ),
        "actual_sha256": sha256_bytes(
            json.dumps(
                actual, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ),
        "match": actual == expected,
    }


def load_or_create_checkpoint(path, runner_sha256):
    if path.is_file():
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if (
            checkpoint.get("schema_version") != 1
            or checkpoint.get("stage") != STAGE
            or checkpoint.get("contract_sha256") != CONTRACT_SHA256
            or checkpoint.get("g0_sha256") != parent.G0_SHA256
            or checkpoint.get("runner_sha256") != runner_sha256
            or checkpoint.get("bounded_sampler_sha256")
            != BOUNDED_SAMPLER_SHA256
            or checkpoint.get("v13_prefix_sha256") != V13_PREFIX_SHA256
        ):
            raise ValueError("v14 checkpoint input/code identity mismatch")
        attempts = checkpoint.get("attempts", [])
        if [row.get("attempt_index") for row in attempts] != list(
            range(len(attempts))
        ):
            raise ValueError("v14 checkpoint attempt order mismatch")
        for row in attempts:
            validate_sampling_metrics(row.get("sampling"))
            comparison = row.get("v13_prefix_replay")
            if row["attempt_index"] < PREFIX_ATTEMPTS:
                if not isinstance(comparison, dict):
                    raise ValueError("missing v13 prefix replay evidence")
            elif comparison is not None:
                raise ValueError("unexpected v13 prefix replay evidence")
        return checkpoint
    checkpoint = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "running",
        "contract_sha256": CONTRACT_SHA256,
        "g0_sha256": parent.G0_SHA256,
        "runner_sha256": runner_sha256,
        "bounded_sampler_sha256": BOUNDED_SAMPLER_SHA256,
        "v13_prefix_sha256": V13_PREFIX_SHA256,
        "contains_raw_text": True,
        "process_loads": [],
        "attempts": [],
        "terminal_reason": None,
    }
    smoke.atomic_write_json(path, checkpoint)
    os.chmod(path, 0o600)
    return checkpoint


def write_heartbeat(path, checkpoint, current_attempt=None):
    attempt_count, eligible_count = parent.checkpoint_counts(checkpoint)
    prefix_rows = checkpoint["attempts"][:PREFIX_ATTEMPTS]
    prefix_matches = sum(
        row.get("v13_prefix_replay", {}).get("match") is True
        for row in prefix_rows
    )
    value = {
        "schema_version": 1,
        "stage": STAGE,
        "status": checkpoint["status"],
        "contract_sha256": CONTRACT_SHA256,
        "runner_sha256": checkpoint["runner_sha256"],
        "attempt_count": attempt_count,
        "eligible_count": eligible_count,
        "failure_count": attempt_count - eligible_count,
        "target_eligible": TARGET_ELIGIBLE,
        "max_attempts": MAX_ATTEMPTS,
        "current_attempt": current_attempt,
        "v13_prefix_expected": PREFIX_ATTEMPTS,
        "v13_prefix_checked": min(attempt_count, PREFIX_ATTEMPTS),
        "v13_prefix_matches": prefix_matches,
        "bounded_fallback_acceptances": sum(
            row["sampling"]["fallback_acceptances"]
            for row in checkpoint["attempts"]
        ),
        "updated_at_utc": parent.utc_now(),
        "contains_raw_text": False,
        "detector_scores_computed": 0,
        "certificate_calls": 0,
    }
    smoke.atomic_write_json(path, value)


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--root", type=Path, required=True)
    value.add_argument("--protocol", type=Path, required=True)
    value.add_argument("--contract", type=Path, required=True)
    value.add_argument("--g0", type=Path, required=True)
    value.add_argument("--identity-manifest", type=Path, required=True)
    value.add_argument("--sealed-source", type=Path, required=True)
    value.add_argument("--v12-source-identities", type=Path, required=True)
    value.add_argument("--v12-a1-result", type=Path, required=True)
    value.add_argument("--v12-smoke-runner", type=Path, required=True)
    value.add_argument("--v13-runtime-stop", type=Path, required=True)
    value.add_argument("--v13-prefix", type=Path, required=True)
    value.add_argument("--generator-dir", type=Path, required=True)
    value.add_argument("--embedder-dir", type=Path, required=True)
    value.add_argument("--nltk-data", type=Path, required=True)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--heartbeat", type=Path, required=True)
    value.add_argument("--positive-output", type=Path, required=True)
    value.add_argument("--result-output", type=Path, required=True)
    value.add_argument("--preflight-only", action="store_true")
    value.add_argument("--preflight-output", type=Path)
    return value


def main():
    args = parser().parse_args()
    args.root = args.root.resolve()
    os.umask(0o077)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["NLTK_DATA"] = str(args.nltk_data.resolve())

    runner_sha256 = smoke.sha256_file(Path(__file__).resolve())
    (
        public_sources,
        sealed_sources,
        forbidden_identities,
        generator_manifest,
        prefix,
    ) = verify_inputs(args)
    versions = parent.verify_versions()
    if args.preflight_only:
        if args.preflight_output is None:
            raise ValueError("--preflight-output is required for preflight")
        fresh_outputs = (
            args.checkpoint,
            args.heartbeat,
            args.positive_output,
            args.result_output,
        )
        existing = [str(path) for path in fresh_outputs if path.exists()]
        if existing:
            raise ValueError(
                f"v14 fresh-output gate failed; existing paths: {existing!r}"
            )
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("preflight requires exactly one visible CUDA GPU")
        preflight = {
            "schema_version": 1,
            "stage": "WP2-v14-G1-preflight",
            "status": "go",
            "contains_raw_text": False,
            "protocol_sha256": PROTOCOL_SHA256,
            "contract_sha256": CONTRACT_SHA256,
            "g0_sha256": parent.G0_SHA256,
            "identity_manifest_sha256": parent.IDENTITY_MANIFEST_SHA256,
            "sealed_source_sha256": parent.SEALED_SOURCE_SHA256,
            "runner_sha256": runner_sha256,
            "parent_runner_sha256": PARENT_RUNNER_SHA256,
            "bounded_sampler_sha256": BOUNDED_SAMPLER_SHA256,
            "v13_runtime_stop_sha256": V13_RUNTIME_STOP_SHA256,
            "v13_prefix_sha256": V13_PREFIX_SHA256,
            "counts": {
                "attempt_sources": len(public_sources),
                "sealed_sources": len(sealed_sources),
                "v13_prefix": len(prefix),
            },
            "generator_model": generator_manifest,
            "runtime": {
                **versions,
                "torch_cuda": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(0),
                "total_cuda_memory_bytes": (
                    torch.cuda.get_device_properties(0).total_memory
                ),
            },
            "fresh_output_paths_absent": True,
            "execution_attestation": {
                "generator_model_loads": 0,
                "embedder_model_loads": 0,
                "generation_calls": 0,
                "detector_imports": 0,
                "detector_scores_computed": 0,
                "resegmentation_calls": 0,
                "certificate_calls": 0,
                "v13_sealed_checkpoint_content_parsed": False,
            },
            "next_action": "freeze_readiness_then_start_v14_G1",
        }
        smoke.atomic_write_json(args.preflight_output, preflight)
        print(json.dumps({
            "status": "go",
            "stage": preflight["stage"],
            "output": str(args.preflight_output),
        }, sort_keys=True))
        return
    checkpoint = load_or_create_checkpoint(args.checkpoint, runner_sha256)
    if checkpoint["status"] == "stop":
        raise ValueError("terminal Stop checkpoint cannot be resumed")
    attempt_count, eligible_count = parent.checkpoint_counts(checkpoint)
    if attempt_count > MAX_ATTEMPTS or eligible_count > TARGET_ELIGIBLE:
        raise ValueError("checkpoint counts exceed contract")
    for index, (attempt, source) in enumerate(
        zip(checkpoint["attempts"], sealed_sources)
    ):
        if (
            attempt["source_raw_text_sha256"]
            != source["raw_text_sha256"]
            or smoke.sha256_text(attempt["source_text"])
            != source["raw_text_sha256"]
            or smoke.sha256_text(attempt["prompt"])
            != attempt["prompt_sha256"]
            or (
                attempt["output"]["raw_text_sha256"] is not None
                and smoke.sha256_text(attempt["generated_text"])
                != attempt["output"]["raw_text_sha256"]
            )
        ):
            raise ValueError("checkpoint raw identity mismatch")
        expected = prefix[index] if index < PREFIX_ATTEMPTS else None
        comparison = prefix_comparison(
            attempt,
            expected,
            sum(not row["exclusion_codes"]
                for row in checkpoint["attempts"][:index + 1]),
        )
        if comparison != attempt.get("v13_prefix_replay"):
            raise ValueError("checkpoint prefix replay evidence mismatch")
        if comparison is not None and comparison["match"] is not True:
            raise ValueError("cannot resume mismatched v13 prefix")

    write_heartbeat(args.heartbeat, checkpoint, attempt_count)
    if eligible_count < TARGET_ELIGIBLE and attempt_count < MAX_ATTEMPTS:
        if shutil.disk_usage(args.root).free < MIN_FREE_BYTES:
            checkpoint["status"] = "stop"
            checkpoint["terminal_reason"] = "disk_below_5_gib"
            smoke.atomic_write_json(args.checkpoint, checkpoint)
            write_heartbeat(args.heartbeat, checkpoint)
        else:
            import nltk
            from nltk.tokenize import sent_tokenize
            import numpy
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                GenerationConfig,
            )

            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise ValueError("G1 requires exactly one visible CUDA GPU")
            nltk.data.path.insert(0, str(args.nltk_data.resolve()))
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)

            repository = args.root / "repos/SemStamp"
            sys.path.insert(0, str(repository))
            import sampling_lsh_utils
            import sampling_utils
            from sampling_utils import extract_prompt_from_text
            from sbert_lsh_model import SBERTLSHModel

            if (
                sampling_utils.MAX_TRIALS
                != smoke.GENERATION_PARAMETERS["max_trials"]
                or sampling_lsh_utils.MAX_TRIALS
                != smoke.GENERATION_PARAMETERS["max_trials"]
            ):
                raise ValueError("official max_trials mismatch")

            tokenizer = AutoTokenizer.from_pretrained(
                str(args.generator_dir.resolve()),
                local_files_only=True,
                use_fast=True,
            )
            generation_config = GenerationConfig(
                max_new_tokens=smoke.GENERATION_PARAMETERS["max_new_tokens"],
                min_new_tokens=smoke.GENERATION_PARAMETERS["min_new_tokens"],
                do_sample=smoke.GENERATION_PARAMETERS["do_sample"],
                temperature=smoke.GENERATION_PARAMETERS["temperature"],
                top_k=smoke.GENERATION_PARAMETERS["top_k"],
                repetition_penalty=smoke.GENERATION_PARAMETERS[
                    "repetition_penalty"
                ],
                pad_token_id=tokenizer.eos_token_id,
            )
            model_load_started = time.monotonic()
            model = AutoModelForCausalLM.from_pretrained(
                str(args.generator_dir.resolve()),
                local_files_only=True,
                use_safetensors=False,
            ).to("cuda:0")
            model.eval()
            lsh_model = SBERTLSHModel(
                lsh_model_path=str(args.embedder_dir.resolve()),
                device="cuda:0",
                batch_size=1,
                lsh_dim=smoke.GENERATION_PARAMETERS["sp_dim"],
                sbert_type="base",
            )
            total_vram = torch.cuda.get_device_properties(0).total_memory
            checkpoint["runtime_context"] = {
                "torch_cuda": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(0),
                "total_cuda_memory_bytes": total_vram,
            }
            checkpoint["process_loads"].append({
                "loaded_at_utc": parent.utc_now(),
                "model_load_seconds": time.monotonic() - model_load_started,
            })
            smoke.atomic_write_json(args.checkpoint, checkpoint)

            prior_output_identities = {
                row["output"]["raw_text_sha256"]
                for row in checkpoint["attempts"]
                if row["output"]["raw_text_sha256"] is not None
            }
            for attempt_index in range(attempt_count, MAX_ATTEMPTS):
                attempt_count, eligible_count = parent.checkpoint_counts(
                    checkpoint
                )
                if eligible_count >= TARGET_ELIGIBLE:
                    break
                if shutil.disk_usage(args.root).free < MIN_FREE_BYTES:
                    checkpoint["terminal_reason"] = "disk_below_5_gib"
                    break
                source_row = sealed_sources[attempt_index]
                public_row = public_sources[attempt_index]
                if (
                    source_row["raw_text_sha256"]
                    != public_row["raw_text_sha256"]
                ):
                    raise ValueError("attempt source identity mismatch")
                prompt = extract_prompt_from_text(
                    source_row["text"],
                    smoke.GENERATION_PARAMETERS["len_prompt"],
                )
                seed = smoke.generation_seed(source_row["raw_text_sha256"])
                smoke.reset_random_state(torch, numpy, seed)
                write_heartbeat(args.heartbeat, checkpoint, attempt_index)
                started = time.monotonic()
                hidden_log = io.StringIO()
                with torch.inference_mode(), redirect_stdout(hidden_log):
                    generated, sampling = bounded_lsh_reject_completion(
                        prompt=prompt,
                        model=model,
                        tokenizer=tokenizer,
                        gen_config=generation_config,
                        lsh_model=lsh_model,
                        lsh_dim=smoke.GENERATION_PARAMETERS["sp_dim"],
                        lmbd=smoke.GENERATION_PARAMETERS["lmbd"],
                        device="cuda:0",
                        margin=smoke.GENERATION_PARAMETERS["margin"],
                        official_module=sampling_lsh_utils,
                    )
                    generated = generated.strip()
                validate_sampling_metrics(sampling)
                elapsed_seconds = time.monotonic() - started
                output_summary, exclusion_codes = parent.validate_output(
                    generated,
                    prompt,
                    sent_tokenize,
                    lsh_model.embedder.tokenizer,
                    forbidden_identities,
                    prior_output_identities,
                )
                if not math.isfinite(elapsed_seconds):
                    exclusion_codes = sorted(set(
                        exclusion_codes + ["non_finite_runtime"]
                    ))
                output_identity = output_summary["raw_text_sha256"]
                if output_identity is not None:
                    prior_output_identities.add(output_identity)
                row = {
                    "attempt_index": attempt_index,
                    "source_raw_text_sha256": source_row["raw_text_sha256"],
                    "seed": seed,
                    "prompt_sha256": smoke.sha256_text(prompt),
                    "output": output_summary,
                    "exclusion_codes": exclusion_codes,
                    "eligible": not exclusion_codes,
                    "elapsed_seconds": elapsed_seconds,
                    "sampling": sampling,
                    "hidden_log_sha256": smoke.sha256_text(
                        hidden_log.getvalue()
                    ),
                    "source_text": source_row["text"],
                    "prompt": prompt,
                    "generated_text": generated,
                }
                next_eligible = eligible_count + int(not exclusion_codes)
                expected = (
                    prefix[attempt_index]
                    if attempt_index < PREFIX_ATTEMPTS
                    else None
                )
                row["v13_prefix_replay"] = prefix_comparison(
                    row, expected, next_eligible
                )
                checkpoint["attempts"].append(row)
                checkpoint["max_cuda_memory_allocated_bytes"] = max(
                    checkpoint.get("max_cuda_memory_allocated_bytes", 0),
                    torch.cuda.max_memory_allocated(0),
                )
                if (
                    row["v13_prefix_replay"] is not None
                    and row["v13_prefix_replay"]["match"] is not True
                ):
                    checkpoint["terminal_reason"] = (
                        "v13_committed_prefix_mismatch"
                    )
                if (
                    checkpoint["max_cuda_memory_allocated_bytes"]
                    / total_vram
                    >= MAX_VRAM_FRACTION
                ):
                    checkpoint["terminal_reason"] = (
                        "vram_at_or_above_80_percent"
                    )
                smoke.atomic_write_json(args.checkpoint, checkpoint)
                os.chmod(args.checkpoint, 0o600)
                write_heartbeat(
                    args.heartbeat, checkpoint, attempt_index
                )
                print(json.dumps({
                    "attempt_index": attempt_index,
                    "eligible": not exclusion_codes,
                    "eligible_count": parent.checkpoint_counts(checkpoint)[1],
                    "elapsed_seconds": round(elapsed_seconds, 3),
                    "output_sha256": output_identity,
                    "exclusion_codes": exclusion_codes,
                    "v13_prefix_match": (
                        None
                        if row["v13_prefix_replay"] is None
                        else row["v13_prefix_replay"]["match"]
                    ),
                    "total_trials": sampling["total_trials"],
                    "fallback_acceptances": sampling[
                        "fallback_acceptances"
                    ],
                }, sort_keys=True), flush=True)
                if checkpoint["terminal_reason"] is not None:
                    break

    attempt_count, eligible_count = parent.checkpoint_counts(checkpoint)
    if (
        checkpoint["status"] == "go"
        and checkpoint["terminal_reason"] == "target_eligible_reached"
        and eligible_count == TARGET_ELIGIBLE
    ):
        pass
    elif checkpoint["terminal_reason"] is not None:
        checkpoint["status"] = "stop"
    elif eligible_count == TARGET_ELIGIBLE:
        checkpoint["status"] = "go"
        checkpoint["terminal_reason"] = "target_eligible_reached"
    elif attempt_count == MAX_ATTEMPTS:
        checkpoint["status"] = "stop"
        checkpoint["terminal_reason"] = "attempt_queue_exhausted"
    else:
        raise RuntimeError("v14 G1 exited without a terminal condition")
    smoke.atomic_write_json(args.checkpoint, checkpoint)
    os.chmod(args.checkpoint, 0o600)

    public_attempts = [
        parent.public_attempt(row) for row in checkpoint["attempts"]
    ]
    positive_rows = []
    for row in public_attempts:
        if row["exclusion_codes"]:
            continue
        positive_rows.append({
            "positive_index": len(positive_rows),
            "attempt_index": row["attempt_index"],
            "source_raw_text_sha256": row["source_raw_text_sha256"],
            "output_raw_text_sha256": row["output"]["raw_text_sha256"],
            "seed": row["seed"],
            "unit_count": row["output"]["unit_count"],
            "max_native_tokens": row["output"]["max_native_tokens"],
            "contains_raw_text": False,
        })
    positive_manifest = parent.atomic_write_jsonl(
        args.positive_output, positive_rows
    )
    failure_counts = Counter(
        code
        for row in public_attempts
        for code in row["exclusion_codes"]
    )
    prefix_rows = public_attempts[:PREFIX_ATTEMPTS]
    output = {
        "schema_version": 1,
        "stage": STAGE,
        "status": checkpoint["status"],
        "terminal_reason": checkpoint["terminal_reason"],
        "contains_raw_text": False,
        "scope": "formal_bounded_generation_only_no_detector_scores",
        "protocol_sha256": PROTOCOL_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "g0_sha256": parent.G0_SHA256,
        "identity_manifest_sha256": parent.IDENTITY_MANIFEST_SHA256,
        "sealed_source_sha256": parent.SEALED_SOURCE_SHA256,
        "runner_sha256": runner_sha256,
        "parent_runner_sha256": PARENT_RUNNER_SHA256,
        "bounded_sampler_sha256": BOUNDED_SAMPLER_SHA256,
        "v13_runtime_stop_sha256": V13_RUNTIME_STOP_SHA256,
        "v13_prefix_sha256": V13_PREFIX_SHA256,
        "v12_smoke_runner_sha256": parent.V12_SMOKE_RUNNER_SHA256,
        "generator_model": generator_manifest,
        "generation_parameters": smoke.GENERATION_PARAMETERS,
        "bounded_sampling": {
            "patch_id": PATCH_ID,
            "maximum_trials_per_segment": (
                smoke.GENERATION_PARAMETERS["max_trials"]
            ),
            "total_trials": sum(
                row["sampling"]["total_trials"] for row in public_attempts
            ),
            "fallback_acceptances": sum(
                row["sampling"]["fallback_acceptances"]
                for row in public_attempts
            ),
        },
        "prefix_replay": {
            "expected": PREFIX_ATTEMPTS,
            "checked": min(attempt_count, PREFIX_ATTEMPTS),
            "matches": sum(
                row.get("v13_prefix_replay", {}).get("match") is True
                for row in prefix_rows
            ),
        },
        "determinism": {
            "seed_salt": smoke.SEED_SALT,
            "cublas_workspace_config": ":4096:8",
            "deterministic_algorithms": True,
            "tf32": False,
            "fresh_v14_checkpoint_from_attempt_zero": True,
            "one_completed_attempt_per_source": True,
        },
        "counts": {
            "attempt_count": attempt_count,
            "eligible_count": eligible_count,
            "failure_count": attempt_count - eligible_count,
            "target_eligible": TARGET_ELIGIBLE,
            "max_attempts": MAX_ATTEMPTS,
            "failure_codes": dict(sorted(failure_counts.items())),
        },
        "positive_manifest": {
            **positive_manifest,
            "relative_path": str(
                args.positive_output.resolve().relative_to(args.root)
            ),
            "contains_raw_text": False,
        },
        "attempts": public_attempts,
        "sealed_checkpoint": {
            "relative_path": str(
                args.checkpoint.resolve().relative_to(args.root)
            ),
            "sha256": smoke.sha256_file(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
            "contains_raw_text": True,
            "public_copy_allowed": False,
        },
        "runtime": {
            **versions,
            **checkpoint.get("runtime_context", {}),
            "process_loads": checkpoint["process_loads"],
            "attempt_seconds": sum(
                row["elapsed_seconds"] for row in public_attempts
            ),
            "max_cuda_memory_allocated_bytes": checkpoint.get(
                "max_cuda_memory_allocated_bytes", 0
            ),
        },
        "execution_attestation": {
            "detector_imports": 0,
            "detector_model_loads": 0,
            "detector_score_reads": 0,
            "detector_scores_computed": 0,
            "resegmentation_calls": 0,
            "certificate_calls": 0,
            "v13_sealed_checkpoint_content_parsed": False,
            "raw_text_emitted_to_public_artifacts": False,
        },
        "next_gate": (
            "freeze_and_run_v14_H0_identity_audit"
            if checkpoint["status"] == "go"
            else "stop_without_detector"
        ),
    }
    smoke.atomic_write_json(args.result_output, output)
    write_heartbeat(args.heartbeat, checkpoint)
    print(json.dumps({
        "status": output["status"],
        "terminal_reason": output["terminal_reason"],
        "attempt_count": attempt_count,
        "eligible_count": eligible_count,
        "result": str(args.result_output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
