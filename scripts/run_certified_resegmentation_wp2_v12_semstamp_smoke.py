"""Run the score-blind WP2-v12 SemStamp deterministic generation smoke."""

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import time


CONTRACT_SHA256 = (
    "67e66412bd0a2930d0606719d4492a2b599162195080d1263e4063aa955e666d"
)
A0_SHA256 = (
    "1496e754c912e105169bf6db63ac6c03062cbc081093887614740840073a3924"
)
IDENTITY_MANIFEST_SHA256 = (
    "ebc1b60759008bc97e0701eef7ba35baa5df35ba8fb1d7c4d8141fcf7ff9cceb"
)
SEALED_SOURCE_SHA256 = (
    "8a65987196c14fab5eb2d0bf5ff7ec23b4748607b32e076e8fbcc2595efa18b3"
)
SEMSTAMP_REVISION = "97db73d11fd80f376a02b0a604d500627622f7e6"
GENERATOR_REVISION = "32662c77794b4f5cbe3d00bcf76fa8653b59bf73"
GENERATOR_WEIGHT_SHA256 = (
    "9f90897038cfabef26c1d797864e22d00720e3255666e8f9076c54fd43fecbbf"
)
EMBEDDER_WEIGHT_SHA256 = (
    "f23f44b44fcad45f2f6178c3936ac7c7d8b16b09b3b70a0043b61c4a6a2e9dee"
)
SEED_SALT = "WP2-v12-semstamp-generation-seed-v1"
SMOKE_SIZE = 20
REPLAY_COUNT = 2
NATIVE_TOKEN_LIMIT = 512
OFFICIAL_FILES = {
    "sampling_utils.py": (
        "35164c4e99057617b93dcb98f4a507f34f8007b61536f7f7ab3333987f3e721e"
    ),
    "sampling_lsh_utils.py": (
        "5fe536616f8d4845cd7d19eddc622b055380dd9364499ce03d4e0b6a5dde2a65"
    ),
    "sbert_lsh_model.py": (
        "5e8735e0bd2b8530035b1a9b7951c5f1b002ab875bb706066be5b5925896963a"
    ),
}
GENERATION_PARAMETERS = {
    "max_new_tokens": 205,
    "min_new_tokens": 195,
    "do_sample": True,
    "temperature": 0.7,
    "top_k": 0,
    "repetition_penalty": 1.05,
    "sp_dim": 3,
    "lmbd": 0.25,
    "margin": 0.02,
    "max_trials": 100,
    "len_prompt": 32,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def generation_seed(raw_text_sha256):
    payload = f"{SEED_SALT}\0{raw_text_sha256}".encode("ascii")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 31)


def atomic_write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(encoded)
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def git_text(arguments, cwd):
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_model_manifest(model_dir):
    manifest_path = model_dir / "MODEL_MANIFEST.sha256"
    if not manifest_path.is_file():
        raise ValueError("generator MODEL_MANIFEST.sha256 is missing")
    rows = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        digest, relative_path = line.split(None, 1)
        relative_path = relative_path.strip()
        path = model_dir / relative_path
        actual = sha256_file(path) if path.is_file() else None
        rows.append({
            "relative_path": relative_path,
            "expected_sha256": digest,
            "actual_sha256": actual,
            "sha256_matches": actual == digest,
            "bytes": path.stat().st_size if path.is_file() else None,
        })
    if not rows or not all(row["sha256_matches"] for row in rows):
        raise ValueError("generator model manifest verification failed")
    by_path = {row["relative_path"]: row for row in rows}
    weight = by_path.get("pytorch_model.bin")
    if (
        weight is None
        or weight["actual_sha256"] != GENERATOR_WEIGHT_SHA256
        or weight["bytes"] != 5_263_157_438
    ):
        raise ValueError("generator weight identity mismatch")
    return {
        "revision": GENERATOR_REVISION,
        "manifest_sha256": sha256_file(manifest_path),
        "files": rows,
    }


def verify_inputs(args):
    expected_files = {
        args.contract: CONTRACT_SHA256,
        args.a0: A0_SHA256,
        args.identity_manifest: IDENTITY_MANIFEST_SHA256,
        args.sealed_source: SEALED_SOURCE_SHA256,
        args.embedder_dir / "pytorch_model.bin": EMBEDDER_WEIGHT_SHA256,
    }
    for path, expected in expected_files.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"input identity mismatch: {path}")

    repository = args.root / "repos/SemStamp"
    if git_text(["rev-parse", "HEAD"], repository) != SEMSTAMP_REVISION:
        raise ValueError("SemStamp revision mismatch")
    if git_text(["status", "--porcelain"], repository):
        raise ValueError("SemStamp worktree is not clean")
    for relative_path, expected in OFFICIAL_FILES.items():
        if sha256_file(repository / relative_path) != expected:
            raise ValueError(f"official source mismatch: {relative_path}")

    a0 = json.loads(args.a0.read_text(encoding="utf-8"))
    if a0["decision"]["a0_source_readiness"] != "go":
        raise ValueError("A0 source readiness is not Go")
    historical_hashes = set()
    for specification in a0["historical_manifests"].values():
        path = args.root / specification["relative_path"]
        if (
            not path.is_file()
            or sha256_file(path) != specification["actual_sha256"]
        ):
            raise ValueError("historical manifest changed after A0")
        historical_hashes.update(
            row["raw_text_sha256"] for row in read_jsonl(path)
        )
    public_rows = read_jsonl(args.identity_manifest)
    sealed_rows = read_jsonl(args.sealed_source)
    public_by_index = {
        row["allocation_index"]: row for row in public_rows
    }
    sealed_by_index = {
        row["allocation_index"]: row for row in sealed_rows
    }
    if set(public_by_index) != set(sealed_by_index):
        raise ValueError("public/sealed allocation index mismatch")
    for allocation_index, public_row in public_by_index.items():
        sealed_row = sealed_by_index[allocation_index]
        if (
            public_row["cohort"] != sealed_row["cohort"]
            or public_row["cohort_index"] != sealed_row["cohort_index"]
            or public_row["raw_text_sha256"]
            != sealed_row["raw_text_sha256"]
            or sha256_text(sealed_row["text"])
            != public_row["raw_text_sha256"]
        ):
            raise ValueError("public/sealed source identity mismatch")
    selected = sorted(
        (
            sealed_row
            for sealed_row in sealed_rows
            if sealed_row["cohort"] == "generation_primary"
            and sealed_row["cohort_index"] < SMOKE_SIZE
        ),
        key=lambda row: row["cohort_index"],
    )
    if (
        len(selected) != SMOKE_SIZE
        or [row["cohort_index"] for row in selected]
        != list(range(SMOKE_SIZE))
    ):
        raise ValueError("frozen smoke selection mismatch")
    return (
        public_rows,
        selected,
        historical_hashes,
        verify_model_manifest(args.generator_dir),
    )


def reset_random_state(torch, numpy, seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_output(
    text,
    sent_tokenize,
    embedder_tokenizer,
    forbidden_identities,
    replay_identities,
):
    if not isinstance(text, str) or not text:
        return None, ["empty_output"]
    identity = sha256_text(text)
    codes = []
    if identity in forbidden_identities:
        codes.append("identity_overlaps_source_or_history")
    if identity in replay_identities:
        codes.append("duplicate_smoke_output")
    try:
        units = sent_tokenize(text)
    except Exception:
        return {
            "raw_text_sha256": identity,
            "unit_count": None,
            "max_native_tokens": None,
        }, codes + ["sentence_split_error"]
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
        and max_native_tokens > NATIVE_TOKEN_LIMIT
    ):
        codes.append("unit_above_native_limit")
    return {
        "raw_text_sha256": identity,
        "unit_count": len(units),
        "max_native_tokens": max_native_tokens,
    }, sorted(set(codes))


def load_or_create_checkpoint(path, runner_sha256):
    if path.is_file():
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if (
            checkpoint.get("contract_sha256") != CONTRACT_SHA256
            or checkpoint.get("runner_sha256") != runner_sha256
        ):
            raise ValueError("checkpoint code/contract identity mismatch")
        return checkpoint
    checkpoint = {
        "schema_version": 1,
        "stage": "WP2-v12-A1-SemStamp-smoke",
        "contract_sha256": CONTRACT_SHA256,
        "runner_sha256": runner_sha256,
        "contains_raw_text": True,
        "replays": {str(index): {} for index in range(REPLAY_COUNT)},
    }
    atomic_write_json(path, checkpoint)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--a0", type=Path, required=True)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--sealed-source", type=Path, required=True)
    parser.add_argument("--generator-dir", type=Path, required=True)
    parser.add_argument("--embedder-dir", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    args.root = args.root.resolve()

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["NLTK_DATA"] = str(args.nltk_data.resolve())

    runner_sha256 = sha256_file(Path(__file__).resolve())
    (
        public_source_rows,
        smoke_rows,
        historical_hashes,
        generator_manifest,
    ) = verify_inputs(args)
    forbidden_identities = {
        row["raw_text_sha256"] for row in public_source_rows
    } | historical_hashes

    import nltk
    from nltk.tokenize import sent_tokenize
    import numpy
    import sentence_transformers
    import torch
    import transformers
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        GenerationConfig,
    )

    if nltk.__version__ != "3.8.1":
        raise ValueError("SemStamp NLTK version mismatch")
    nltk.data.path.insert(0, str(args.nltk_data.resolve()))
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required")
    if torch.cuda.device_count() != 1:
        raise ValueError("A1 smoke requires exactly one visible GPU")
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
        != GENERATION_PARAMETERS["max_trials"]
        or sampling_lsh_utils.MAX_TRIALS
        != GENERATION_PARAMETERS["max_trials"]
    ):
        raise ValueError("official max_trials mismatch")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.generator_dir.resolve()),
        local_files_only=True,
        use_fast=True,
    )
    generation_config = GenerationConfig(
        max_new_tokens=GENERATION_PARAMETERS["max_new_tokens"],
        min_new_tokens=GENERATION_PARAMETERS["min_new_tokens"],
        do_sample=GENERATION_PARAMETERS["do_sample"],
        temperature=GENERATION_PARAMETERS["temperature"],
        top_k=GENERATION_PARAMETERS["top_k"],
        repetition_penalty=GENERATION_PARAMETERS[
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
        lsh_dim=GENERATION_PARAMETERS["sp_dim"],
        sbert_type="base",
    )
    model_load_seconds = time.monotonic() - model_load_started
    checkpoint = load_or_create_checkpoint(
        args.checkpoint, runner_sha256
    )

    for replay_index in range(REPLAY_COUNT):
        replay_key = str(replay_index)
        replay_rows = checkpoint["replays"][replay_key]
        replay_identities = {
            value["output"]["raw_text_sha256"]
            for value in replay_rows.values()
        }
        for source_row in smoke_rows:
            row_key = str(source_row["cohort_index"])
            if row_key in replay_rows:
                continue
            source_identity = source_row["raw_text_sha256"]
            seed = generation_seed(source_identity)
            prompt = extract_prompt_from_text(
                source_row["text"],
                GENERATION_PARAMETERS["len_prompt"],
            )
            reset_random_state(torch, numpy, seed)
            started = time.monotonic()
            hidden_log = io.StringIO()
            with torch.inference_mode(), redirect_stdout(hidden_log):
                generated = lsh_reject_completion(
                    prompt=prompt,
                    model=model,
                    tokenizer=tokenizer,
                    gen_config=generation_config,
                    lsh_model=lsh_model,
                    lsh_dim=GENERATION_PARAMETERS["sp_dim"],
                    lmbd=GENERATION_PARAMETERS["lmbd"],
                    device="cuda:0",
                    margin=GENERATION_PARAMETERS["margin"],
                ).strip()
            elapsed_seconds = time.monotonic() - started
            output_summary, exclusion_codes = validate_output(
                generated,
                sent_tokenize,
                lsh_model.embedder.tokenizer,
                forbidden_identities,
                replay_identities,
            )
            if output_summary is None:
                output_summary = {
                    "raw_text_sha256": None,
                    "unit_count": None,
                    "max_native_tokens": None,
                }
            if output_summary["raw_text_sha256"] is not None:
                replay_identities.add(output_summary["raw_text_sha256"])
            replay_rows[row_key] = {
                "cohort_index": source_row["cohort_index"],
                "source_raw_text_sha256": source_identity,
                "seed": seed,
                "prompt_sha256": sha256_text(prompt),
                "output": output_summary,
                "exclusion_codes": exclusion_codes,
                "elapsed_seconds": elapsed_seconds,
                "generated_text": generated,
                "hidden_log_sha256": sha256_text(hidden_log.getvalue()),
            }
            atomic_write_json(args.checkpoint, checkpoint)
            print(json.dumps({
                "replay": replay_index,
                "cohort_index": source_row["cohort_index"],
                "output_sha256": output_summary["raw_text_sha256"],
                "eligible": not exclusion_codes,
                "elapsed_seconds": round(elapsed_seconds, 3),
            }, sort_keys=True), flush=True)

    public_replays = {}
    for replay_index in range(REPLAY_COUNT):
        replay_key = str(replay_index)
        public_replays[replay_key] = []
        for cohort_index in range(SMOKE_SIZE):
            row = checkpoint["replays"][replay_key][str(cohort_index)]
            public_replays[replay_key].append({
                key: value
                for key, value in row.items()
                if key not in {"generated_text"}
            })

    replay_matches = []
    for cohort_index in range(SMOKE_SIZE):
        first = public_replays["0"][cohort_index]
        second = public_replays["1"][cohort_index]
        replay_matches.append(
            first["source_raw_text_sha256"]
            == second["source_raw_text_sha256"]
            and first["seed"] == second["seed"]
            and first["prompt_sha256"] == second["prompt_sha256"]
            and first["output"]["raw_text_sha256"]
            == second["output"]["raw_text_sha256"]
        )
    all_eligible = all(
        not row["exclusion_codes"]
        for rows in public_replays.values()
        for row in rows
    )
    status = (
        "go"
        if all(replay_matches) and all_eligible
        else "stop"
    )
    output = {
        "schema_version": 1,
        "stage": "WP2-v12-A1-SemStamp-smoke",
        "status": status,
        "contains_raw_text": False,
        "scope": "generation_reproducibility_only_no_detector_scores",
        "contract_sha256": CONTRACT_SHA256,
        "a0_sha256": A0_SHA256,
        "identity_manifest_sha256": IDENTITY_MANIFEST_SHA256,
        "sealed_source_sha256": SEALED_SOURCE_SHA256,
        "runner_sha256": runner_sha256,
        "official_revision": SEMSTAMP_REVISION,
        "official_source_sha256": OFFICIAL_FILES,
        "generator_model": generator_manifest,
        "embedder": {
            "revision": (
                "a77dce8530e100ac1101981e4be390546c1cb282"
            ),
            "weight_sha256": EMBEDDER_WEIGHT_SHA256,
        },
        "generation_parameters": GENERATION_PARAMETERS,
        "determinism": {
            "seed_salt": SEED_SALT,
            "cublas_workspace_config": ":4096:8",
            "deterministic_algorithms": True,
            "tf32": False,
            "replay_count": REPLAY_COUNT,
            "row_matches": sum(replay_matches),
            "all_rows_match": all(replay_matches),
        },
        "eligibility": {
            "expected_rows_per_replay": SMOKE_SIZE,
            "eligible_rows": sum(
                not row["exclusion_codes"]
                for rows in public_replays.values()
                for row in rows
            ),
            "expected_total_rows": SMOKE_SIZE * REPLAY_COUNT,
            "all_eligible": all_eligible,
        },
        "replays": public_replays,
        "sealed_checkpoint": {
            "relative_path": str(
                args.checkpoint.resolve().relative_to(args.root)
            ),
            "sha256": sha256_file(args.checkpoint),
            "contains_raw_text": True,
            "public_copy_allowed": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "numpy": numpy.__version__,
            "nltk": nltk.__version__,
            "device_name": torch.cuda.get_device_name(0),
            "model_load_seconds": model_load_seconds,
            "cuda_max_memory_allocated_bytes": (
                torch.cuda.max_memory_allocated(0)
            ),
        },
        "execution_attestation": {
            "generator_model_loads": 1,
            "embedder_model_loads": 1,
            "detector_imports": 0,
            "detector_score_reads": 0,
            "detector_scores_computed": 0,
            "resegmentation_calls": 0,
            "certificate_calls": 0,
            "smoke_outputs_reusable_as_formal_positive": False,
        },
        "next_gate": (
            "freeze_formal_semstamp_generation_runner_and_resource_budget"
            if status == "go"
            else "stop_and_diagnose_generation_replay_or_eligibility"
        ),
    }
    if any(
        not math.isfinite(row["elapsed_seconds"])
        for rows in public_replays.values()
        for row in rows
    ):
        output["status"] = "stop"
        output["next_gate"] = "stop_non_finite_runtime"
    atomic_write_json(args.output, output)
    print(json.dumps({
        "status": output["status"],
        "replay_matches": sum(replay_matches),
        "eligible_rows": output["eligibility"]["eligible_rows"],
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
