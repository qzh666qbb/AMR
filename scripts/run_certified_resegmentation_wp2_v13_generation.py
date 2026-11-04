"""Run WP2-v13 score-blind replacement-aware SemStamp generation."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime, timezone
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import run_certified_resegmentation_wp2_v12_semstamp_smoke as smoke


PROTOCOL_SHA256 = (
    "e3fa05e532986077840651ae37a1ff1b9a43a9a52706d2337a2435420b274768"
)
CONTRACT_SHA256 = (
    "1114df7d706d6cca2354b4d1cdddbfc6850bbefd1427238004bd251ab34a4b26"
)
G0_SHA256 = (
    "937e0388da8aba10bde503f80e13df831a007f96369a76cf4e2c5ceed4ca7953"
)
IDENTITY_MANIFEST_SHA256 = (
    "27e9967972acff5118a5bb9c112e0724463644139a8aed321ac0c2e8d965bddf"
)
SEALED_SOURCE_SHA256 = (
    "5b172d040257f53ee843ee4ca50d1a05c38ba7f75b26d9f27f7576192c7e35b7"
)
V12_SOURCE_IDENTITIES_SHA256 = (
    "ebc1b60759008bc97e0701eef7ba35baa5df35ba8fb1d7c4d8141fcf7ff9cceb"
)
V12_A1_RESULT_SHA256 = (
    "86cae250106c7de5fc7d7f1025a696737e572d109c350f7e0dab18c270bf8faf"
)
V12_SMOKE_RUNNER_SHA256 = (
    "c266f6d8e14acccdc6360454c384a405c285dfbc52717c771339aef4823864df"
)
TARGET_ELIGIBLE = 1000
MAX_ATTEMPTS = 1600
MIN_FREE_BYTES = 5 * 1024 ** 3
MAX_VRAM_FRACTION = 0.80
EXPECTED_VERSIONS = {
    "python": "3.10.12",
    "torch": "2.3.0+cu121",
    "transformers": "4.35.2",
    "sentence-transformers": "3.0.1",
    "numpy": "1.26.4",
    "nltk": "3.8.1",
    "openai": "1.30.3",
    "tenacity": "8.3.0",
    "NearPy": "1.0.0",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def atomic_write_jsonl(path, rows):
    value = "\n".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in rows
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)
    return {
        "sha256": smoke.sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
    }


def package_versions():
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        **{
            name: importlib.metadata.version(name)
            for name in EXPECTED_VERSIONS
            if name not in {"python", "torch"}
        },
    }


def verify_versions():
    observed = package_versions()
    if observed != EXPECTED_VERSIONS:
        raise ValueError(
            f"runtime version mismatch: {observed!r}"
        )
    return observed


def git_text(arguments, cwd):
    return subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_file(path, expected, label):
    if not path.is_file() or smoke.sha256_file(path) != expected:
        raise ValueError(f"{label} identity mismatch")


def verify_inputs(args):
    expected_files = (
        (args.protocol, PROTOCOL_SHA256, "protocol"),
        (args.contract, CONTRACT_SHA256, "G1 contract"),
        (args.g0, G0_SHA256, "G0 readiness"),
        (
            args.identity_manifest,
            IDENTITY_MANIFEST_SHA256,
            "G0 identity manifest",
        ),
        (
            args.sealed_source,
            SEALED_SOURCE_SHA256,
            "G0 sealed source manifest",
        ),
        (
            args.v12_source_identities,
            V12_SOURCE_IDENTITIES_SHA256,
            "v12 source identities",
        ),
        (
            args.v12_a1_result,
            V12_A1_RESULT_SHA256,
            "v12 A1 result",
        ),
        (
            args.v12_smoke_runner,
            V12_SMOKE_RUNNER_SHA256,
            "v12 smoke runner dependency",
        ),
        (
            args.embedder_dir / "pytorch_model.bin",
            smoke.EMBEDDER_WEIGHT_SHA256,
            "embedder weight",
        ),
    )
    for path, expected, label in expected_files:
        verify_file(path, expected, label)

    repository = args.root / "repos/SemStamp"
    if git_text(["rev-parse", "HEAD"], repository) != smoke.SEMSTAMP_REVISION:
        raise ValueError("SemStamp revision mismatch")
    if git_text(["status", "--porcelain"], repository):
        raise ValueError("SemStamp worktree is not clean")
    for relative_path, expected in smoke.OFFICIAL_FILES.items():
        verify_file(
            repository / relative_path,
            expected,
            f"official source {relative_path}",
        )

    g0 = json.loads(args.g0.read_text(encoding="utf-8"))
    if (
        g0["decision"]["g0_source_readiness"] != "go"
        or g0["decision"]["g1_formal_generation"] != "go"
        or g0["allocation"]["selected_unique"] != MAX_ATTEMPTS
    ):
        raise ValueError("G0 does not authorize G1")

    public_rows = read_jsonl(args.identity_manifest)
    sealed_rows = read_jsonl(args.sealed_source)
    if len(public_rows) != MAX_ATTEMPTS or len(sealed_rows) != MAX_ATTEMPTS:
        raise ValueError("G0 attempt queue length mismatch")
    for attempt_index, (public_row, sealed_row) in enumerate(
        zip(public_rows, sealed_rows)
    ):
        if (
            public_row["attempt_index"] != attempt_index
            or sealed_row["attempt_index"] != attempt_index
            or public_row["raw_text_sha256"]
            != sealed_row["raw_text_sha256"]
            or smoke.sha256_text(sealed_row["text"])
            != public_row["raw_text_sha256"]
        ):
            raise ValueError("public/sealed attempt queue mismatch")

    historical_identities = set()
    for specification in g0["historical_manifests"].values():
        path = args.root / specification["relative_path"]
        verify_file(
            path,
            specification["actual_sha256"],
            "historical manifest",
        )
        historical_identities.update(
            row["raw_text_sha256"] for row in read_jsonl(path)
        )
    v12_source_identities = {
        row["raw_text_sha256"]
        for row in read_jsonl(args.v12_source_identities)
    }
    v12_result = json.loads(
        args.v12_a1_result.read_text(encoding="utf-8")
    )
    v12_smoke_outputs = {
        row["output"]["raw_text_sha256"]
        for row in v12_result["replays"]["0"]
    }
    if (
        v12_result["status"] != "stop"
        or v12_result["determinism"]["row_matches"] != 20
        or len(v12_smoke_outputs) != 20
    ):
        raise ValueError("v12 A1 evidence mismatch")
    candidate_sources = {
        row["raw_text_sha256"] for row in public_rows
    }
    if (
        len(candidate_sources) != MAX_ATTEMPTS
        or candidate_sources & historical_identities
        or candidate_sources & v12_source_identities
        or candidate_sources & v12_smoke_outputs
    ):
        raise ValueError("G1 source exclusion mismatch")

    forbidden = (
        candidate_sources
        | historical_identities
        | v12_source_identities
        | v12_smoke_outputs
    )
    return (
        public_rows,
        sealed_rows,
        forbidden,
        smoke.verify_model_manifest(args.generator_dir),
    )


def validate_output(
    text,
    prompt,
    sent_tokenize,
    embedder_tokenizer,
    forbidden_identities,
    prior_output_identities,
):
    if not isinstance(text, str) or not text:
        return {
            "raw_text_sha256": None,
            "unit_count": None,
            "max_native_tokens": None,
            "output_equals_prompt": False,
        }, ["empty_output"]
    identity = smoke.sha256_text(text)
    codes = []
    if identity in forbidden_identities:
        codes.append("identity_overlaps_forbidden_set")
    if identity in prior_output_identities:
        codes.append("duplicate_generation_output")
    try:
        units = sent_tokenize(text)
    except Exception:
        return {
            "raw_text_sha256": identity,
            "unit_count": None,
            "max_native_tokens": None,
            "output_equals_prompt": text == prompt,
        }, sorted(set(codes + ["sentence_split_error"]))
    if (
        len(units) < 2
        or any(
            not isinstance(unit, str)
            or not unit
            or unit != unit.strip()
            for unit in units
        )
    ):
        codes.append("invalid_or_short_p0")
    token_counts = [
        len(embedder_tokenizer(
            unit.strip(),
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"])
        for unit in units
    ]
    max_native_tokens = max(token_counts) if token_counts else None
    if (
        max_native_tokens is not None
        and max_native_tokens > smoke.NATIVE_TOKEN_LIMIT
    ):
        codes.append("unit_above_native_limit")
    return {
        "raw_text_sha256": identity,
        "unit_count": len(units),
        "max_native_tokens": max_native_tokens,
        "output_equals_prompt": text == prompt,
    }, sorted(set(codes))


def load_or_create_checkpoint(path, runner_sha256):
    if path.is_file():
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if (
            checkpoint.get("contract_sha256") != CONTRACT_SHA256
            or checkpoint.get("g0_sha256") != G0_SHA256
            or checkpoint.get("runner_sha256") != runner_sha256
        ):
            raise ValueError("checkpoint input/code identity mismatch")
        attempts = checkpoint.get("attempts", [])
        if [
            row.get("attempt_index") for row in attempts
        ] != list(range(len(attempts))):
            raise ValueError("checkpoint attempt order mismatch")
        return checkpoint
    checkpoint = {
        "schema_version": 1,
        "stage": "WP2-v13-G1-SemStamp-generation",
        "status": "running",
        "contract_sha256": CONTRACT_SHA256,
        "g0_sha256": G0_SHA256,
        "runner_sha256": runner_sha256,
        "contains_raw_text": True,
        "process_loads": [],
        "attempts": [],
        "terminal_reason": None,
    }
    smoke.atomic_write_json(path, checkpoint)
    os.chmod(path, 0o600)
    return checkpoint


def checkpoint_counts(checkpoint):
    attempts = checkpoint["attempts"]
    eligible = sum(not row["exclusion_codes"] for row in attempts)
    return len(attempts), eligible


def write_heartbeat(path, checkpoint, current_attempt=None):
    attempt_count, eligible_count = checkpoint_counts(checkpoint)
    value = {
        "schema_version": 1,
        "stage": checkpoint["stage"],
        "status": checkpoint["status"],
        "contract_sha256": CONTRACT_SHA256,
        "runner_sha256": checkpoint["runner_sha256"],
        "attempt_count": attempt_count,
        "eligible_count": eligible_count,
        "failure_count": attempt_count - eligible_count,
        "target_eligible": TARGET_ELIGIBLE,
        "max_attempts": MAX_ATTEMPTS,
        "current_attempt": current_attempt,
        "updated_at_utc": utc_now(),
        "contains_raw_text": False,
        "detector_scores_computed": 0,
        "certificate_calls": 0,
    }
    smoke.atomic_write_json(path, value)


def public_attempt(row):
    return {
        key: value
        for key, value in row.items()
        if key not in {"source_text", "prompt", "generated_text"}
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--g0", type=Path, required=True)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--sealed-source", type=Path, required=True)
    parser.add_argument("--v12-source-identities", type=Path, required=True)
    parser.add_argument("--v12-a1-result", type=Path, required=True)
    parser.add_argument("--v12-smoke-runner", type=Path, required=True)
    parser.add_argument("--generator-dir", type=Path, required=True)
    parser.add_argument("--embedder-dir", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--positive-output", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    args = parser.parse_args()
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
    ) = verify_inputs(args)
    versions = verify_versions()
    checkpoint = load_or_create_checkpoint(
        args.checkpoint, runner_sha256
    )
    if checkpoint["status"] == "stop":
        raise ValueError("terminal Stop checkpoint cannot be resumed")
    attempt_count, eligible_count = checkpoint_counts(checkpoint)
    if attempt_count > MAX_ATTEMPTS or eligible_count > TARGET_ELIGIBLE:
        raise ValueError("checkpoint counts exceed contract")
    for attempt, source in zip(
        checkpoint["attempts"],
        sealed_sources,
    ):
        if (
            attempt["source_raw_text_sha256"]
            != source["raw_text_sha256"]
        ):
            raise ValueError("checkpoint source order mismatch")
        if (
            smoke.sha256_text(attempt["source_text"])
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
            from sampling_lsh_utils import lsh_reject_completion
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
                max_new_tokens=smoke.GENERATION_PARAMETERS[
                    "max_new_tokens"
                ],
                min_new_tokens=smoke.GENERATION_PARAMETERS[
                    "min_new_tokens"
                ],
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
                "loaded_at_utc": utc_now(),
                "model_load_seconds": (
                    time.monotonic() - model_load_started
                ),
            })
            smoke.atomic_write_json(args.checkpoint, checkpoint)

            prior_output_identities = {
                row["output"]["raw_text_sha256"]
                for row in checkpoint["attempts"]
                if row["output"]["raw_text_sha256"] is not None
            }
            for attempt_index in range(attempt_count, MAX_ATTEMPTS):
                attempt_count, eligible_count = checkpoint_counts(
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
                seed = smoke.generation_seed(
                    source_row["raw_text_sha256"]
                )
                smoke.reset_random_state(torch, numpy, seed)
                write_heartbeat(
                    args.heartbeat, checkpoint, attempt_index
                )
                started = time.monotonic()
                hidden_log = io.StringIO()
                with torch.inference_mode(), redirect_stdout(hidden_log):
                    generated = lsh_reject_completion(
                        prompt=prompt,
                        model=model,
                        tokenizer=tokenizer,
                        gen_config=generation_config,
                        lsh_model=lsh_model,
                        lsh_dim=smoke.GENERATION_PARAMETERS["sp_dim"],
                        lmbd=smoke.GENERATION_PARAMETERS["lmbd"],
                        device="cuda:0",
                        margin=smoke.GENERATION_PARAMETERS["margin"],
                    ).strip()
                elapsed_seconds = time.monotonic() - started
                output_summary, exclusion_codes = validate_output(
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
                    "source_raw_text_sha256": (
                        source_row["raw_text_sha256"]
                    ),
                    "seed": seed,
                    "prompt_sha256": smoke.sha256_text(prompt),
                    "output": output_summary,
                    "exclusion_codes": exclusion_codes,
                    "eligible": not exclusion_codes,
                    "elapsed_seconds": elapsed_seconds,
                    "hidden_log_sha256": smoke.sha256_text(
                        hidden_log.getvalue()
                    ),
                    "source_text": source_row["text"],
                    "prompt": prompt,
                    "generated_text": generated,
                }
                checkpoint["attempts"].append(row)
                checkpoint["max_cuda_memory_allocated_bytes"] = max(
                    checkpoint.get(
                        "max_cuda_memory_allocated_bytes", 0
                    ),
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
                write_heartbeat(
                    args.heartbeat, checkpoint, attempt_index
                )
                print(json.dumps({
                    "attempt_index": attempt_index,
                    "eligible": not exclusion_codes,
                    "eligible_count": checkpoint_counts(
                        checkpoint
                    )[1],
                    "elapsed_seconds": round(elapsed_seconds, 3),
                    "output_sha256": output_identity,
                    "exclusion_codes": exclusion_codes,
                }, sort_keys=True), flush=True)
                if checkpoint["terminal_reason"] is not None:
                    break

    attempt_count, eligible_count = checkpoint_counts(checkpoint)
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
        raise RuntimeError("G1 exited without a terminal condition")
    smoke.atomic_write_json(args.checkpoint, checkpoint)
    os.chmod(args.checkpoint, 0o600)

    public_attempts = [
        public_attempt(row) for row in checkpoint["attempts"]
    ]
    positive_rows = []
    for row in public_attempts:
        if row["exclusion_codes"]:
            continue
        positive_rows.append({
            "positive_index": len(positive_rows),
            "attempt_index": row["attempt_index"],
            "source_raw_text_sha256": (
                row["source_raw_text_sha256"]
            ),
            "output_raw_text_sha256": (
                row["output"]["raw_text_sha256"]
            ),
            "seed": row["seed"],
            "unit_count": row["output"]["unit_count"],
            "max_native_tokens": row["output"][
                "max_native_tokens"
            ],
            "contains_raw_text": False,
        })
    positive_manifest = atomic_write_jsonl(
        args.positive_output, positive_rows
    )
    failure_counts = Counter(
        code
        for row in public_attempts
        for code in row["exclusion_codes"]
    )
    output = {
        "schema_version": 1,
        "stage": "WP2-v13-G1-SemStamp-generation",
        "status": checkpoint["status"],
        "terminal_reason": checkpoint["terminal_reason"],
        "contains_raw_text": False,
        "scope": "formal_generation_only_no_detector_scores",
        "protocol_sha256": PROTOCOL_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "g0_sha256": G0_SHA256,
        "identity_manifest_sha256": IDENTITY_MANIFEST_SHA256,
        "sealed_source_sha256": SEALED_SOURCE_SHA256,
        "runner_sha256": runner_sha256,
        "v12_smoke_runner_sha256": V12_SMOKE_RUNNER_SHA256,
        "generator_model": generator_manifest,
        "generation_parameters": smoke.GENERATION_PARAMETERS,
        "determinism": {
            "seed_salt": smoke.SEED_SALT,
            "cublas_workspace_config": ":4096:8",
            "deterministic_algorithms": True,
            "tf32": False,
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
            "raw_text_emitted_to_public_artifacts": False,
        },
        "next_gate": (
            "run_H0_identity_freeze"
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
