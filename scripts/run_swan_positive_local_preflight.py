"""Local Qwen feasibility preflight for the frozen SWAN positive sources."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import run_swan_positive_collection as swan


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/swan_positive_local_qwen_preflight1"
MODEL_ROOT = ROOT / "models/qwen2.5-0.5b-instruct"
SOURCE_MANIFEST = (
    ROOT / "experiments/swan_positive_collection1/source_manifest.json"
)
OFFICIAL_CONTRACT = (
    ROOT
    / "experiments/swan_positive_collection1/execution_contract_v3.json"
)
MANIFEST = OUTPUT / "manifest.json"
SOURCES = 5
SLOTS = 1
MAX_TRIALS = 10
MAX_TEMPLATES = 2
RETRIES_PER_TEMPLATE = 5
CALL_CAP = SOURCES * SLOTS * MAX_TRIALS
CONTINUE_ACCEPTED = 3
MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


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
            raise ValueError(f"Frozen local artifact differs: {path}")
        return saved
    write_json(path, payload)
    return payload


def pilot_sources() -> list:
    source = load_json(SOURCE_MANIFEST)
    rows = [row for row in source["sources"] if row["cohort"] == "pilot"]
    if len(rows) != SOURCES:
        raise ValueError("Frozen pilot membership changed")
    return rows


def dependency_hashes() -> dict:
    paths = [
        Path(__file__),
        ROOT / "scripts/test_swan_positive_local_preflight.py",
        ROOT / "scripts/run_swan_positive_collection.py",
        SOURCE_MANIFEST,
        OFFICIAL_CONTRACT,
        swan.OFFICIAL_INJECTION,
        swan.S2MATCH_RUNTIME / "provenance.json",
    ]
    paths.extend(MODEL_ROOT / name for name in MODEL_FILES)
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for path in paths
    }


def prepare() -> dict:
    if MANIFEST.exists():
        return check()
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "frozen_before_local_generation",
        "workflow": "swan_positive_local_qwen_preflight1",
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "official_execution_contract_sha256": sha256(
            OFFICIAL_CONTRACT
        ),
        "sources": [
            {
                "allocation_id": row["allocation_id"],
                "generation_prompt": row["generation_prompt"],
                "normalized_source_sha256": row[
                    "normalized_source_sha256"
                ],
            }
            for row in pilot_sources()
        ],
        "backend": {
            "type": "local_huggingface",
            "model_path": str(MODEL_ROOT.relative_to(ROOT)),
            "model_config_sha256": sha256(MODEL_ROOT / "config.json"),
            "model_weights_sha256": sha256(
                MODEL_ROOT / "model.safetensors"
            ),
            "not_official_equivalent": True,
            "reason": (
                "Local Qwen2.5-0.5B replaces Claude Sonnet 4.5 only for "
                "backend feasibility; no official-path equivalence claim."
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
            "maximum_local_generations": CALL_CAP,
            "continue_gate": (
                "At least 3 of 5 first slots accepted within 10 trials. "
                "Otherwise stop the local-Qwen route."
            ),
            "no_parameter_changes_after_outputs": True,
            "no_detection": True,
            "no_attack": True,
            "no_tpr_or_asr_claim": True,
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers")
        },
        "dependencies": dependency_hashes(),
        "resource_counts": {
            "deepseek_calls": 0,
            "bedrock_calls": 0,
            "local_generation_calls": 0,
            "parser_calls": 0,
            "s2match_calls": 0,
        },
    }
    return freeze_json(MANIFEST, manifest)


def check() -> dict:
    manifest = load_json(MANIFEST)
    if (
        manifest["workflow"]
        != "swan_positive_local_qwen_preflight1"
        or manifest["source_manifest_sha256"]
        != sha256(SOURCE_MANIFEST)
        or manifest["sources"]
        != [
            {
                "allocation_id": row["allocation_id"],
                "generation_prompt": row["generation_prompt"],
                "normalized_source_sha256": row[
                    "normalized_source_sha256"
                ],
            }
            for row in pilot_sources()
        ]
        or manifest["dependencies"] != dependency_hashes()
    ):
        raise ValueError("Local preflight manifest differs")
    return manifest


class LocalQwen:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        dtype = torch.float16 if self.device.type == "mps" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_ROOT),
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_ROOT),
            local_files_only=True,
            torch_dtype=dtype,
            attn_implementation="eager",
        ).to(self.device).eval()
        self.runtime = {
            "device": str(self.device),
            "dtype": str(dtype),
            "model_config_sha256": sha256(MODEL_ROOT / "config.json"),
            "model_weights_sha256": sha256(
                MODEL_ROOT / "model.safetensors"
            ),
        }

    def generate(
        self,
        system_text: str,
        user_text: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> dict:
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
        ).to(self.device)
        prompt_tokens = int(encoded["input_ids"].shape[1])
        self.torch.manual_seed(seed)
        started = time.monotonic()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )[0]
        token_ids = generated[prompt_tokens:].detach().cpu().tolist()
        text = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        ).strip()
        if not text:
            raise ValueError("Local generator returned empty text")
        stopped = bool(
            token_ids
            and token_ids[-1] == self.tokenizer.eos_token_id
        )
        return {
            "text": text,
            "text_sha256": text_sha256(text),
            "stop_reason": "eos" if stopped else "length",
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": len(token_ids),
            },
            "response_metadata": {
                "backend": "local_qwen2.5-0.5b-instruct",
                "device": str(self.device),
                "seed": seed,
                "elapsed_seconds": time.monotonic() - started,
            },
            "rendered_prompt_sha256": text_sha256(rendered),
        }


class LocalJournal:
    def __init__(self, folder: Path, generator: LocalQwen):
        self.folder = folder
        self.generator = generator

    def request(
        self,
        call_id: str,
        user_text: str,
        system_text: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict:
        request = {
            "backend": "local_qwen2.5-0.5b-instruct",
            "user_text": user_text,
            "system_text": system_text,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        request_hash = digest(request)
        path = self.folder / "local_calls" / f"{call_id}.json"
        if path.exists():
            saved = load_json(path)
            if (
                saved["request_sha256"] != request_hash
                or saved["request"] != request
                or saved["status"] != "complete"
            ):
                raise ValueError(f"Local call is not replayable: {call_id}")
            return {
                **saved["response"],
                "call_sha256": sha256(path),
                "replayed": True,
            }
        if len(list((self.folder / "local_calls").glob("*.json"))) >= CALL_CAP:
            raise RuntimeError("Frozen local generation cap reached")
        seed = int(
            hashlib.sha256(
                f"local-qwen-preflight:{call_id}".encode()
            ).hexdigest()[:8],
            16,
        )
        saved = {
            "schema_version": 1,
            "call_id": call_id,
            "request_sha256": request_hash,
            "request": request,
            "seed": seed,
            "status": "in_flight",
            "started_at": utc_now(),
        }
        write_json(path, saved)
        try:
            response = self.generator.generate(
                system_text,
                user_text,
                max_tokens,
                temperature,
                top_p,
                seed,
            )
            saved.update(status="complete", response=response)
        except Exception as exc:
            saved.update(
                status="failed",
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
    manifest = check()
    slots = sorted((OUTPUT / "generation_slots").glob("*.json"))
    rows = [load_json(path) for path in slots]
    accepted = sum(row["accepted"] for row in rows)
    complete = len(rows) == SOURCES * SLOTS
    calls = sorted((OUTPUT / "local_calls").glob("*.json"))
    result = {
        "schema_version": 1,
        "status": "complete" if complete else "incomplete",
        "backend": manifest["backend"],
        "allocated_slots": SOURCES * SLOTS,
        "completed_slots": len(rows),
        "accepted_slots": accepted,
        "attempts": sum(row["attempts"] for row in rows),
        "local_generation_calls": len(calls),
        "all_local_calls_complete": all(
            load_json(path)["status"] == "complete" for path in calls
        ),
        "decision": (
            "prepare_separate_local_qwen_pilot"
            if complete and accepted >= CONTINUE_ACCEPTED
            else "stop_local_qwen_route"
            if complete
            else "pending"
        ),
        "continue_gate": {
            "required": CONTINUE_ACCEPTED,
            "observed": accepted,
            "passed": complete and accepted >= CONTINUE_ACCEPTED,
        },
        "slot_sha256": {
            path.name: sha256(path) for path in slots
        },
        "manifest_sha256": sha256(MANIFEST),
        "claim_boundary": (
            "Backend feasibility only; not official Claude equivalence, "
            "not a formal positive set, and no TPR or ASR."
        ),
        "resource_counts": {
            "deepseek_calls": 0,
            "bedrock_calls": 0,
            "local_generation_calls": len(calls),
            "parser_calls": len(calls),
            "s2match_calls": len(calls),
            "detector_calls": 0,
            "attack_calls": 0,
        },
    }
    write_json(OUTPUT / "summary.json", result)
    return result


def run() -> dict:
    manifest = check()
    generator = LocalQwen()
    freeze_json(
        OUTPUT / "runtime.json",
        {
            "schema_version": 1,
            "manifest_sha256": sha256(MANIFEST),
            "backend": manifest["backend"],
            "runtime": generator.runtime,
            "packages": manifest["packages"],
        },
    )
    module = swan.load_official_injection()
    bank_records = load_json(
        swan.OFFICIAL_ROOT / "amr_bank/banks/amr_bank_50.json"
    )
    bank = [row["normalized_template_amr"] for row in bank_records]
    examples = (
        swan.OFFICIAL_ROOT
        / "amr_bank/artifacts/normalized_amr_examples.json"
    )
    journal = LocalJournal(OUTPUT, generator)

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
        for source in pilot_sources():
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
            current = summarize()
            print(
                json.dumps(
                    {
                        "completed_slots": current["completed_slots"],
                        "accepted_slots": current["accepted_slots"],
                        "local_generation_calls": current[
                            "local_generation_calls"
                        ],
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
    finally:
        swan.CONTRACT = original["CONTRACT"]
        swan.MAX_TRIALS = original["MAX_TRIALS"]
        swan.MAX_TEMPLATES = original["MAX_TEMPLATES"]
        swan.RETRIES_PER_TEMPLATE = original["RETRIES"]
    return summarize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "summarize"))
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.stage == "prepare":
        result = prepare()
    elif args.stage == "run":
        result = run()
    else:
        result = summarize()
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
