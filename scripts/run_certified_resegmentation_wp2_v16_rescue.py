"""Run the WP2-v16 fixed-queue prompt-only rescue supplement."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
import time

import run_certified_resegmentation_wp2_v12_semstamp_smoke as smoke
import run_certified_resegmentation_wp2_v13_generation as parent
import run_certified_resegmentation_wp2_v15_generation as v15
from semstamp_bounded_generation import (
    PATCH_ID,
    bounded_lsh_reject_completion,
)


STAGE = "WP2-v16-G1R-prompt-only-rescue"
PROTOCOL_SHA256 = (
    "0449b363abff408007de8e6024c2d147b5aa586c00ad8b19a8b519e080c41cfe"
)
CONTRACT_SHA256 = (
    "c8344700cf6c420bfb4cbe0fe6fb4870febf70aa772a5c328d5ecfba5dfe3fe9"
)
V15_PROTOCOL_SHA256 = v15.PROTOCOL_SHA256
V15_CONTRACT_SHA256 = v15.CONTRACT_SHA256
V15_RUNNER_SHA256 = (
    "1d91a61f24d82f483652244a0c526dbda1371ad940d99fec20ce1998d2a2230f"
)
V15_G1_RESULT_SHA256 = (
    "c9d8bc4d1c56d39efbc01993e563403a299a66d55ea9de27a522b0ecdf1a2a3f"
)
V15_POSITIVE_SHA256 = (
    "fde40f2ecb8fcfe5653111739ea42bd2a7a8b5ae05da0b66c2b654fc6d5d6d38"
)
V15_H0_STOP_SHA256 = (
    "c832cea4ac029111a3a4239d441cd3618f578690dc1fe8155bf4cc1262c1b681"
)
V15_SEALED_SHA256 = (
    "b60bfdb8d84e358dca049f73a681609e23991c05e1f2327f1038c5d5e11d9404"
)
V15_SEALED_BYTES = 5_329_796
BOUNDED_SAMPLER_SHA256 = v15.BOUNDED_SAMPLER_SHA256
START_ATTEMPT = 1111
TARGET_SUPPLEMENT = 5
RETAINED_PARENT = 995
MAX_ATTEMPTS = parent.MAX_ATTEMPTS
MIN_FREE_BYTES = parent.MIN_FREE_BYTES
MAX_VRAM_FRACTION = parent.MAX_VRAM_FRACTION
PROMPT_ONLY_ATTEMPTS = (8, 561, 823, 834, 912)


def verify_file(path, expected, label):
    parent.verify_file(path, expected, label)
    return {
        "sha256": expected,
        "bytes": path.stat().st_size,
    }


def validate_parent_evidence(result, positive_rows, h0_stop):
    counts = result.get("counts", {})
    prefix = result.get("prefix_replay", {})
    attestation = result.get("execution_attestation", {})
    if (
        result.get("stage") != v15.STAGE
        or result.get("status") != "go"
        or result.get("terminal_reason") != "target_eligible_reached"
        or result.get("runner_sha256") != V15_RUNNER_SHA256
        or result.get("protocol_sha256") != V15_PROTOCOL_SHA256
        or result.get("contract_sha256") != V15_CONTRACT_SHA256
        or result.get("bounded_sampler_sha256")
        != BOUNDED_SAMPLER_SHA256
        or counts.get("attempt_count") != START_ATTEMPT
        or counts.get("eligible_count") != 1000
        or counts.get("failure_count") != 111
        or prefix.get("checked") != v15.PREFIX_ATTEMPTS
        or prefix.get("acceptable") != v15.PREFIX_ATTEMPTS
        or prefix.get("classifications", {}).get(
            "unexplained_divergence"
        ) != 0
    ):
        raise ValueError("v15 terminal G1 evidence mismatch")
    for key in (
        "detector_imports",
        "detector_model_loads",
        "detector_score_reads",
        "detector_scores_computed",
        "resegmentation_calls",
        "certificate_calls",
    ):
        if attestation.get(key) != 0:
            raise ValueError("v15 score-blind attestation mismatch")
    attempts = result.get("attempts")
    if (
        not isinstance(attempts, list)
        or len(attempts) != START_ATTEMPT
        or [row.get("attempt_index") for row in attempts]
        != list(range(START_ATTEMPT))
    ):
        raise ValueError("v15 public attempt sequence mismatch")
    claimed = [row for row in attempts if row.get("eligible") is True]
    if len(claimed) != 1000 or len(positive_rows) != 1000:
        raise ValueError("v15 claimed positive count mismatch")
    for positive_index, (row, positive) in enumerate(
        zip(claimed, positive_rows)
    ):
        expected = {
            "positive_index": positive_index,
            "attempt_index": row["attempt_index"],
            "source_raw_text_sha256": row["source_raw_text_sha256"],
            "output_raw_text_sha256": row["output"]["raw_text_sha256"],
            "seed": row["seed"],
            "unit_count": row["output"]["unit_count"],
            "max_native_tokens": row["output"]["max_native_tokens"],
            "contains_raw_text": False,
        }
        if positive != expected:
            raise ValueError("v15 positive manifest does not match G1")

    prompt_only = [
        row for row in claimed
        if row.get("output", {}).get("output_equals_prompt") is True
    ]
    if tuple(row["attempt_index"] for row in prompt_only) != (
        PROMPT_ONLY_ATTEMPTS
    ):
        raise ValueError("v15 prompt-only attempt set mismatch")
    for row in prompt_only:
        sampling = row.get("sampling", {})
        if (
            row.get("exclusion_codes") != []
            or sampling.get("accepted_segments") != 0
            or sampling.get("terminated_with_empty_generation") is not True
        ):
            raise ValueError("v15 prompt-only evidence mismatch")

    retained_pairs = [
        (row, positive)
        for row, positive in zip(claimed, positive_rows)
        if row["output"]["output_equals_prompt"] is False
    ]
    if len(retained_pairs) != RETAINED_PARENT:
        raise ValueError("v15 retained parent count mismatch")
    retained = []
    for row, positive in retained_pairs:
        output = row.get("output", {})
        if (
            row.get("exclusion_codes") != []
            or output.get("unit_count", 0) < 2
            or output.get("max_native_tokens") is None
            or output["max_native_tokens"] > smoke.NATIVE_TOKEN_LIMIT
        ):
            raise ValueError("retained v15 positive fails P0 metadata")
        retained.append(positive)
    retained_output_ids = {
        row["output_raw_text_sha256"] for row in retained
    }
    retained_source_ids = {
        row["source_raw_text_sha256"] for row in retained
    }
    if (
        len(retained_output_ids) != RETAINED_PARENT
        or len(retained_source_ids) != RETAINED_PARENT
    ):
        raise ValueError("retained v15 positives are not identity-unique")

    violation = h0_stop.get("protocol_violation", {})
    sealed = h0_stop.get("inputs", {}).get(
        "sealed_checkpoint_metadata_only", {}
    )
    if (
        h0_stop.get("stage") != "WP2-v15-H0-identity-freeze-stop"
        or h0_stop.get("status") != "stop"
        or h0_stop.get("stop_reason")
        != "prompt_only_positive_contract_violation"
        or violation.get("affected_count") != len(PROMPT_ONLY_ATTEMPTS)
        or tuple(violation.get("attempt_indices", []))
        != PROMPT_ONLY_ATTEMPTS
        or violation.get("all_output_equals_prompt") is not True
        or violation.get("all_accepted_segments_zero") is not True
        or violation.get("all_terminated_with_empty_generation") is not True
        or sealed.get("sha256") != V15_SEALED_SHA256
        or sealed.get("bytes") != V15_SEALED_BYTES
        or sealed.get("content_parsed") is not False
    ):
        raise ValueError("v15 H0 Stop evidence mismatch")

    all_output_ids = {
        row["output"]["raw_text_sha256"]
        for row in attempts
        if row.get("output", {}).get("raw_text_sha256") is not None
    }
    if not retained_output_ids <= all_output_ids:
        raise ValueError("retained output identity set mismatch")
    return retained, all_output_ids


def verify_inputs(args):
    verify_file(
        args.protocol, PROTOCOL_SHA256, "v16 rescue protocol"
    )
    verify_file(
        args.contract, CONTRACT_SHA256, "v16 G1R contract"
    )
    verify_file(
        Path(v15.__file__).resolve(),
        V15_RUNNER_SHA256,
        "v15 generation runner",
    )
    verify_file(
        Path(__file__).with_name("semstamp_bounded_generation.py"),
        BOUNDED_SAMPLER_SHA256,
        "bounded sampler",
    )
    verify_file(
        args.v15_protocol, V15_PROTOCOL_SHA256, "v15 protocol"
    )
    verify_file(
        args.v15_contract, V15_CONTRACT_SHA256, "v15 G1 contract"
    )
    verify_file(
        args.v15_g1_result, V15_G1_RESULT_SHA256, "v15 G1 result"
    )
    verify_file(
        args.v15_positive_identities,
        V15_POSITIVE_SHA256,
        "v15 positive identities",
    )
    verify_file(
        args.v15_h0_stop, V15_H0_STOP_SHA256, "v15 H0 Stop"
    )
    sealed_metadata = verify_file(
        args.v15_sealed_checkpoint,
        V15_SEALED_SHA256,
        "v15 sealed checkpoint",
    )
    if (
        sealed_metadata["bytes"] != V15_SEALED_BYTES
        or stat.S_IMODE(args.v15_sealed_checkpoint.stat().st_mode) != 0o600
    ):
        raise ValueError("v15 sealed checkpoint metadata mismatch")

    original_protocol = v15.PROTOCOL_SHA256
    original_contract = v15.CONTRACT_SHA256
    try:
        v15.PROTOCOL_SHA256 = PROTOCOL_SHA256
        v15.CONTRACT_SHA256 = CONTRACT_SHA256
        inherited = v15.verify_inputs(args)
    finally:
        v15.PROTOCOL_SHA256 = original_protocol
        v15.CONTRACT_SHA256 = original_contract

    result = json.loads(args.v15_g1_result.read_text(encoding="utf-8"))
    positives = parent.read_jsonl(args.v15_positive_identities)
    h0_stop = json.loads(args.v15_h0_stop.read_text(encoding="utf-8"))
    retained, parent_output_ids = validate_parent_evidence(
        result, positives, h0_stop
    )
    return (*inherited, retained, parent_output_ids)


def validate_rescue_output(
    text,
    prompt,
    sent_tokenize,
    embedder_tokenizer,
    forbidden_identities,
    v15_output_identities,
    supplement_output_identities,
):
    output, codes = parent.validate_output(
        text,
        prompt,
        sent_tokenize,
        embedder_tokenizer,
        forbidden_identities,
        set(),
    )
    identity = output["raw_text_sha256"]
    if output["output_equals_prompt"]:
        codes.append("prompt_only_output")
    if identity in v15_output_identities:
        codes.append("duplicate_v15_output")
    if identity in supplement_output_identities:
        codes.append("duplicate_supplement_output")
    return output, sorted(set(codes))


def checkpoint_counts(checkpoint):
    attempts = checkpoint["attempts"]
    return len(attempts), sum(row["eligible"] for row in attempts)


def load_or_create_checkpoint(path, runner_sha256):
    if path.is_file():
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if (
            checkpoint.get("schema_version") != 1
            or checkpoint.get("stage") != STAGE
            or checkpoint.get("protocol_sha256") != PROTOCOL_SHA256
            or checkpoint.get("contract_sha256") != CONTRACT_SHA256
            or checkpoint.get("runner_sha256") != runner_sha256
            or checkpoint.get("v15_g1_result_sha256")
            != V15_G1_RESULT_SHA256
            or checkpoint.get("v15_h0_stop_sha256")
            != V15_H0_STOP_SHA256
            or checkpoint.get("bounded_sampler_sha256")
            != BOUNDED_SAMPLER_SHA256
        ):
            raise ValueError("v16 checkpoint input/code identity mismatch")
        attempts = checkpoint.get("attempts", [])
        expected = list(range(START_ATTEMPT, START_ATTEMPT + len(attempts)))
        if [row.get("attempt_index") for row in attempts] != expected:
            raise ValueError("v16 checkpoint attempt order mismatch")
        for row in attempts:
            v15.validate_sampling_metrics(row.get("sampling"))
            if row.get("eligible") is not (not row.get("exclusion_codes")):
                raise ValueError("v16 checkpoint eligibility mismatch")
        return checkpoint
    checkpoint = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "running",
        "protocol_sha256": PROTOCOL_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "runner_sha256": runner_sha256,
        "v15_g1_result_sha256": V15_G1_RESULT_SHA256,
        "v15_h0_stop_sha256": V15_H0_STOP_SHA256,
        "bounded_sampler_sha256": BOUNDED_SAMPLER_SHA256,
        "start_attempt": START_ATTEMPT,
        "target_supplement": TARGET_SUPPLEMENT,
        "retained_parent_count": RETAINED_PARENT,
        "contains_raw_text": True,
        "process_loads": [],
        "attempts": [],
        "terminal_reason": None,
    }
    smoke.atomic_write_json(path, checkpoint)
    os.chmod(path, 0o600)
    return checkpoint


def write_heartbeat(path, checkpoint, current_attempt=None):
    attempt_count, eligible_count = checkpoint_counts(checkpoint)
    value = {
        "schema_version": 1,
        "stage": STAGE,
        "status": checkpoint["status"],
        "protocol_sha256": PROTOCOL_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "runner_sha256": checkpoint["runner_sha256"],
        "start_attempt": START_ATTEMPT,
        "current_attempt": current_attempt,
        "supplement_attempt_count": attempt_count,
        "supplement_eligible_count": eligible_count,
        "supplement_failure_count": attempt_count - eligible_count,
        "target_supplement": TARGET_SUPPLEMENT,
        "combined_eligible_if_frozen": RETAINED_PARENT + eligible_count,
        "next_attempt": START_ATTEMPT + attempt_count,
        "max_attempt_index": MAX_ATTEMPTS - 1,
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
    value.add_argument("--v14-runtime-stop", type=Path, required=True)
    value.add_argument("--v14-g1-result", type=Path, required=True)
    value.add_argument(
        "--v14-positive-identities", type=Path, required=True
    )
    value.add_argument("--v15-protocol", type=Path, required=True)
    value.add_argument("--v15-contract", type=Path, required=True)
    value.add_argument("--v15-g1-result", type=Path, required=True)
    value.add_argument(
        "--v15-positive-identities", type=Path, required=True
    )
    value.add_argument("--v15-h0-stop", type=Path, required=True)
    value.add_argument(
        "--v15-sealed-checkpoint", type=Path, required=True
    )
    value.add_argument("--generator-dir", type=Path, required=True)
    value.add_argument("--embedder-dir", type=Path, required=True)
    value.add_argument("--nltk-data", type=Path, required=True)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--heartbeat", type=Path, required=True)
    value.add_argument("--supplement-output", type=Path, required=True)
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
        _prefix,
        retained_parent,
        v15_output_identities,
    ) = verify_inputs(args)
    versions = parent.verify_versions()

    if args.preflight_only:
        if args.preflight_output is None:
            raise ValueError("--preflight-output is required for preflight")
        fresh_outputs = (
            args.checkpoint,
            args.heartbeat,
            args.supplement_output,
            args.result_output,
        )
        existing = [str(path) for path in fresh_outputs if path.exists()]
        if existing:
            raise ValueError(
                f"v16 fresh-output gate failed; existing paths: {existing!r}"
            )
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("preflight requires exactly one visible CUDA GPU")
        preflight = {
            "schema_version": 1,
            "stage": "WP2-v16-G1R-preflight",
            "status": "go",
            "contains_raw_text": False,
            "protocol_sha256": PROTOCOL_SHA256,
            "contract_sha256": CONTRACT_SHA256,
            "runner_sha256": runner_sha256,
            "v15_g1_result_sha256": V15_G1_RESULT_SHA256,
            "v15_positive_sha256": V15_POSITIVE_SHA256,
            "v15_h0_stop_sha256": V15_H0_STOP_SHA256,
            "v15_sealed_checkpoint": {
                "sha256": V15_SEALED_SHA256,
                "bytes": V15_SEALED_BYTES,
                "content_parsed": False,
            },
            "counts": {
                "source_queue": len(public_sources),
                "retained_parent": len(retained_parent),
                "v15_output_exclusion_identities": len(
                    v15_output_identities
                ),
                "start_attempt": START_ATTEMPT,
                "target_supplement": TARGET_SUPPLEMENT,
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
                "v15_sealed_checkpoint_content_parsed": False,
            },
            "next_action": "freeze_readiness_then_start_v16_G1R",
        }
        smoke.atomic_write_json(args.preflight_output, preflight)
        print(json.dumps({
            "status": "go",
            "stage": preflight["stage"],
            "output": str(args.preflight_output),
        }, sort_keys=True))
        return

    checkpoint = load_or_create_checkpoint(args.checkpoint, runner_sha256)
    supplement_attempts, supplement_eligible = checkpoint_counts(checkpoint)
    if (
        supplement_attempts > MAX_ATTEMPTS - START_ATTEMPT
        or supplement_eligible > TARGET_SUPPLEMENT
    ):
        raise ValueError("v16 checkpoint counts exceed contract")
    for offset, attempt in enumerate(checkpoint["attempts"]):
        attempt_index = START_ATTEMPT + offset
        source = sealed_sources[attempt_index]
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
            raise ValueError("v16 checkpoint raw identity mismatch")

    write_heartbeat(
        args.heartbeat,
        checkpoint,
        START_ATTEMPT + supplement_attempts,
    )
    if (
        checkpoint["status"] == "running"
        and supplement_eligible < TARGET_SUPPLEMENT
        and START_ATTEMPT + supplement_attempts < MAX_ATTEMPTS
    ):
        if shutil.disk_usage(args.root).free < MIN_FREE_BYTES:
            checkpoint["terminal_reason"] = "disk_below_5_gib"
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
                raise ValueError("G1R requires exactly one visible CUDA GPU")
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

            supplement_output_ids = {
                row["output"]["raw_text_sha256"]
                for row in checkpoint["attempts"]
                if row["output"]["raw_text_sha256"] is not None
            }
            first_attempt = START_ATTEMPT + supplement_attempts
            for attempt_index in range(first_attempt, MAX_ATTEMPTS):
                supplement_attempts, supplement_eligible = (
                    checkpoint_counts(checkpoint)
                )
                if supplement_eligible >= TARGET_SUPPLEMENT:
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
                v15.validate_sampling_metrics(sampling)
                elapsed_seconds = time.monotonic() - started
                output_summary, exclusion_codes = validate_rescue_output(
                    generated,
                    prompt,
                    sent_tokenize,
                    lsh_model.embedder.tokenizer,
                    forbidden_identities,
                    v15_output_identities,
                    supplement_output_ids,
                )
                if not math.isfinite(elapsed_seconds):
                    exclusion_codes = sorted(set(
                        exclusion_codes + ["non_finite_runtime"]
                    ))
                output_identity = output_summary["raw_text_sha256"]
                if output_identity is not None:
                    supplement_output_ids.add(output_identity)
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
                checkpoint["attempts"].append(row)
                checkpoint["max_cuda_memory_allocated_bytes"] = max(
                    checkpoint.get("max_cuda_memory_allocated_bytes", 0),
                    torch.cuda.max_memory_allocated(0),
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
                write_heartbeat(args.heartbeat, checkpoint, attempt_index)
                print(json.dumps({
                    "attempt_index": attempt_index,
                    "eligible": not exclusion_codes,
                    "supplement_eligible_count": checkpoint_counts(
                        checkpoint
                    )[1],
                    "elapsed_seconds": round(elapsed_seconds, 3),
                    "output_sha256": output_identity,
                    "exclusion_codes": exclusion_codes,
                    "total_trials": sampling["total_trials"],
                    "fallback_acceptances": sampling[
                        "fallback_acceptances"
                    ],
                }, sort_keys=True), flush=True)
                if checkpoint["terminal_reason"] is not None:
                    break

    supplement_attempts, supplement_eligible = checkpoint_counts(checkpoint)
    if (
        checkpoint["status"] == "go"
        and checkpoint["terminal_reason"]
        == "target_supplement_reached"
        and supplement_eligible == TARGET_SUPPLEMENT
    ):
        pass
    elif checkpoint["terminal_reason"] is not None:
        checkpoint["status"] = "stop"
    elif supplement_eligible == TARGET_SUPPLEMENT:
        checkpoint["status"] = "go"
        checkpoint["terminal_reason"] = "target_supplement_reached"
    elif START_ATTEMPT + supplement_attempts == MAX_ATTEMPTS:
        checkpoint["status"] = "stop"
        checkpoint["terminal_reason"] = "attempt_queue_exhausted"
    else:
        raise RuntimeError("v16 G1R exited without a terminal condition")
    smoke.atomic_write_json(args.checkpoint, checkpoint)
    os.chmod(args.checkpoint, 0o600)

    public_attempts = [
        parent.public_attempt(row) for row in checkpoint["attempts"]
    ]
    supplement_rows = []
    for row in public_attempts:
        if row["exclusion_codes"]:
            continue
        supplement_rows.append({
            "supplement_index": len(supplement_rows),
            "attempt_index": row["attempt_index"],
            "source_raw_text_sha256": row["source_raw_text_sha256"],
            "output_raw_text_sha256": row["output"]["raw_text_sha256"],
            "seed": row["seed"],
            "unit_count": row["output"]["unit_count"],
            "max_native_tokens": row["output"]["max_native_tokens"],
            "contains_raw_text": False,
        })
    supplement_manifest = parent.atomic_write_jsonl(
        args.supplement_output, supplement_rows
    )
    failure_counts = Counter(
        code
        for row in public_attempts
        for code in row["exclusion_codes"]
    )
    output = {
        "schema_version": 1,
        "stage": STAGE,
        "status": checkpoint["status"],
        "terminal_reason": checkpoint["terminal_reason"],
        "contains_raw_text": False,
        "scope": (
            "fixed_queue_prompt_only_rescue_no_detector_scores"
        ),
        "protocol_sha256": PROTOCOL_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "runner_sha256": runner_sha256,
        "bounded_sampler_sha256": BOUNDED_SAMPLER_SHA256,
        "v15_parent": {
            "protocol_sha256": V15_PROTOCOL_SHA256,
            "contract_sha256": V15_CONTRACT_SHA256,
            "runner_sha256": V15_RUNNER_SHA256,
            "g1_result_sha256": V15_G1_RESULT_SHA256,
            "positive_identities_sha256": V15_POSITIVE_SHA256,
            "h0_stop_sha256": V15_H0_STOP_SHA256,
            "sealed_checkpoint_sha256": V15_SEALED_SHA256,
            "sealed_checkpoint_bytes": V15_SEALED_BYTES,
            "sealed_checkpoint_content_parsed": False,
        },
        "parent_reclassification": {
            "claimed_positive": 1000,
            "excluded_prompt_only": len(PROMPT_ONLY_ATTEMPTS),
            "prompt_only_attempts": list(PROMPT_ONLY_ATTEMPTS),
            "retained_positive": len(retained_parent),
            "rule": "output_equals_prompt_true_to_prompt_only_output",
            "score_conditioned": False,
        },
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
            "fallback_attempts": sum(
                row["sampling"]["fallback_acceptances"] > 0
                for row in public_attempts
            ),
            "fallback_acceptances": sum(
                row["sampling"]["fallback_acceptances"]
                for row in public_attempts
            ),
        },
        "continuation": {
            "start_attempt": START_ATTEMPT,
            "end_attempt_inclusive": (
                None
                if not public_attempts
                else public_attempts[-1]["attempt_index"]
            ),
            "strictly_sequential": True,
            "v15_output_identity_exclusion_count": len(
                v15_output_identities
            ),
        },
        "counts": {
            "retained_parent_count": len(retained_parent),
            "supplement_attempt_count": supplement_attempts,
            "supplement_eligible_count": supplement_eligible,
            "supplement_failure_count": (
                supplement_attempts - supplement_eligible
            ),
            "combined_eligible_count": (
                len(retained_parent) + supplement_eligible
            ),
            "target_supplement": TARGET_SUPPLEMENT,
            "target_combined": 1000,
            "failure_codes": dict(sorted(failure_counts.items())),
        },
        "supplement_manifest": {
            **supplement_manifest,
            "relative_path": str(
                args.supplement_output.resolve().relative_to(args.root)
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
            "v15_sealed_checkpoint_content_parsed": False,
            "raw_text_emitted_to_public_artifacts": False,
        },
        "next_gate": (
            "run_independent_v16_H0_identity_audit"
            if checkpoint["status"] == "go"
            else "stop_without_detector"
        ),
    }
    smoke.atomic_write_json(args.result_output, output)
    write_heartbeat(args.heartbeat, checkpoint)
    print(json.dumps({
        "status": output["status"],
        "terminal_reason": output["terminal_reason"],
        "supplement_attempt_count": supplement_attempts,
        "supplement_eligible_count": supplement_eligible,
        "combined_eligible_count": len(retained_parent)
        + supplement_eligible,
        "result": str(args.result_output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
