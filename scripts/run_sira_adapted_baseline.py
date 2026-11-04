"""Run a checkpointed SIRA-method adaptation on the 30 confirmation baselines.

This is not a model-equivalent reproduction of official SIRA. It retains the
official self-information blanking and two-stage rewrite structure, uses a
local Qwen2.5-0.5B model for surprisal, and uses the same DeepSeek generation
model, K=4 candidate budget, quality gate, and SWAN detector as our comparison.
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from evaluate_adaptive_retry import gates
from run_confirmation_experiment import (
    CallJournal, Detector, MODEL, REVIEW, decode_reviews, digest, freeze,
    request_body,
)
from run_segmentation_intervention import ROOT, sha, text_sha, write_json


K = 4
MASK_PERCENTILE = 30
QWEN_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
QWEN_PATH = ROOT / "models/qwen2.5-0.5b-instruct"
SOURCE = ROOT / "experiments/confirmation_three_arm30"
OUTPUT = ROOT / "experiments/sira_adapted30"

REFERENCE_PROMPT = (
    "You are a paraphraser. You are given an input passage 'INPUT'. "
    "You should paraphrase 'INPUT' to print 'OUTPUT'. 'OUTPUT' should be diverse and different "
    "as much as possible from 'INPUT' and should not copy any part verbatim from 'INPUT'. "
    "However, 'OUTPUT' should preserve the information in the INPUT. "
    "You should print 'OUTPUT' and nothing else so that it is easy for me to parse.\nINPUT: "
)

ATTACK_PROMPT = (
    "You will be shown one reference paragraph and one incomplete paragraph.\n"
    "Your task is to write a complete paragraph using incomplete paragraph.\n"
    "The complete paragraph should have similar length with reference paragraph.\n"
    "You need to include all the information in the reference. \n"
    "But do not take the expression and words in the reference paragraph.\n"
    "You should only answer the complete paragraph.\n"
)


def git_commit(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def mask_tokens(tokens, self_information, percentile=MASK_PERCENTILE):
    """Match official attack_onestep.py: retain low-SI chunks, blank the rest."""
    if len(tokens) != len(self_information) or not tokens:
        raise ValueError("Token and self-information vectors must be nonempty and aligned")
    if not 0 <= percentile <= 100:
        raise ValueError("Percentile must be in [0, 100]")
    if not all(math.isfinite(value) for value in self_information):
        raise ValueError("Self-information values must be finite")
    cutoff = float(np.percentile(self_information, percentile))
    transformed, retained = [], []
    mask = []
    for token, value in zip(tokens, self_information):
        is_masked = value > cutoff
        mask.append(is_masked)
        if is_masked:
            if retained:
                transformed.append(f"({' '.join(retained)})")
                retained = []
            transformed.append("_")
        else:
            retained.append(token)
    if retained:
        transformed.append(f"({' '.join(retained)})")
    return {
        "blank_text": "".join(transformed),
        "percentile": percentile,
        "cutoff": cutoff,
        "mask": mask,
        "masked_tokens": sum(mask),
        "token_count": len(tokens),
    }


class SelfInformationModel:
    def __init__(self, model_path=QWEN_PATH):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if self.device == "mps" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=dtype,
            attn_implementation="eager").to(self.device).eval()

    def calculate(self, text):
        torch = self.torch
        encoded = self.tokenizer(
            text, add_special_tokens=False, return_tensors="pt").to(self.device)
        if encoded["input_ids"].shape[1] <= 1:
            raise ValueError("SIRA source must contain at least two model tokens")
        with torch.inference_mode():
            logits = self.model(**encoded).logits[:, :-1]
            values = -torch.log_softmax(logits, dim=-1).gather(
                -1, encoded["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1).squeeze(0)
        token_ids = encoded["input_ids"].squeeze(0).tolist()[1:]
        tokens = [self.tokenizer.decode(token) for token in token_ids]
        self_information = values.float().cpu().tolist()
        masked = mask_tokens(tokens, self_information)
        return {
            **masked, "tokens": tokens, "token_ids": token_ids,
            "self_information": self_information, "device": self.device,
            "note": "As in official code, the first source token has no preceding-token score and is omitted.",
        }


def prepare(folder):
    baseline_paths = sorted((SOURCE / "baselines").glob("*.json"))
    records = []
    dependencies = {}
    for path in baseline_paths:
        row = json.loads(path.read_text())
        records.append({
            "prompt_id": row["prompt_id"], "source": row["text"],
            "source_sha256": row["text_sha256"],
            "baseline_detection": {k: row["detection"][k] for k in ("n", "green", "z")},
            "baseline_path": str(path.relative_to(ROOT)),
            "baseline_sha256": sha(path),
        })
        dependencies[str(path.relative_to(ROOT))] = sha(path)
    if len(records) != 30 or len({r["prompt_id"] for r in records}) != 30:
        raise ValueError("Expected 30 unique confirmation baselines")
    manifest = {
        "protocol": {
            "name": "SIRA-adapted", "official_sira_commit": git_commit(ROOT / "third_party/SIRA"),
            "official_dipper_commit": git_commit(ROOT / "third_party/DIPPER"),
            "not_official_reproduction": True,
            "adaptation": (
                "Qwen2.5-0.5B-Instruct computes raw-text self-information; "
                "DeepSeek generates one reference and four final rewrites"
            ),
            "reference_prompt_sha256": text_sha(REFERENCE_PROMPT),
            "attack_prompt_sha256": text_sha(ATTACK_PROMPT),
            "mask_percentile": MASK_PERCENTILE,
            "candidate_budget": K,
            "reference_generation": {"temperature": 0, "max_tokens": 256, "calls_per_source": 1},
            "attack_generation": {"temperature": .7, "max_tokens": 256, "calls_per_source": K},
            "selection": (
                "No-feedback selects first quality-valid candidate before detection; "
                "feedback selects minimum z among quality-valid candidates"
            ),
            "quality_and_detector": "Identical implementation and thresholds to confirmation_three_arm30",
        },
        "records": records, "dependencies": dependencies,
    }
    freeze(folder / "manifest.json", manifest)
    print(json.dumps({"stage": "prepared", "records": len(records),
                      "output": str(folder / "manifest.json")}), flush=True)


def runtime(folder, api):
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run prepare first")
    probe = api.request("probe", request_body("Reply exactly OK.", 0, 16))
    if probe["finish_reason"] != "stop" or probe["content"].strip() != "OK":
        raise ValueError("API smoke response unexpected")
    api.model_returned = probe["model"]
    model_files = sorted(p for p in QWEN_PATH.iterdir() if p.is_file())
    paths = [
        Path(__file__), manifest_path, ROOT / "scripts/run_confirmation_experiment.py",
        ROOT / "scripts/evaluate_adaptive_retry.py",
        ROOT / "third_party/SIRA/scripts/attack_onestep.py",
        ROOT / "third_party/SWAN/utils/amr_utils.py",
        ROOT / "third_party/SWAN/amr_bank/banks/amr_bank_50.json",
        ROOT / "vectors/glove.6B.100d.txt",
        *model_files,
    ]
    paths += [p for name in ("model_parse_xfm_bart_large-v0_1_0", "bart-large-tokenizer")
              for p in sorted((ROOT / "models" / name).glob("*")) if p.is_file()]
    config = {
        "model_requested": MODEL, "model_returned": probe["model"],
        "probe_id": probe["id"], "prompt_count": 30,
        "method": "SIRA-adapted, not official model-equivalent reproduction",
        "self_information": {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": QWEN_REVISION, "device_preference": "mps_then_cpu",
            "dtype": "float16_on_mps_float32_on_cpu",
            "add_special_tokens": False, "mask_percentile": MASK_PERCENTILE,
            "official_logic": "attack_onestep.py SelfInformationCalculator.transform_tokens",
        },
        "reference": {"temperature": 0, "max_tokens": 256, "prompt": REFERENCE_PROMPT},
        "attack": {"k": K, "temperature": .7, "max_tokens": 256,
                   "prompt": ATTACK_PROMPT,
                   "adaptation": "Candidate variant suffix and sampling provide K=4"},
        "review": {"template": REVIEW, "temperature": 0, "max_tokens": 700,
                   "expose_method_or_z": False},
        "detector": {"lambda": .25, "match_threshold": .65, "escape_threshold": 2.33},
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "amrlib", "penman", "nltk", "numpy", "requests")},
        "dependencies": {str(path.relative_to(ROOT)): sha(path) for path in paths},
        "failure_policy": "Journal before transmission; never silently retry uncertain calls",
    }
    freeze(folder / "runtime.json", config)
    return config


def generate_reference(folder, record, api, config):
    path = folder / "references" / f'{record["prompt_id"]}.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["source_sha256"] != record["source_sha256"] or saved["runtime_sha256"] != digest(config):
            raise ValueError("Reference checkpoint mismatch")
        return saved
    response = api.request(
        f'reference-{record["prompt_id"]}',
        request_body(REFERENCE_PROMPT + record["source"], 0, 256,
                     system="You are a helpful rewriter."))
    if response["finish_reason"] != "stop":
        raise ValueError("Truncated SIRA reference")
    saved = {
        "prompt_id": record["prompt_id"], "source_sha256": record["source_sha256"],
        "runtime_sha256": digest(config), "reference": response["content"].strip(),
        "reference_sha256": text_sha(response["content"].strip()),
        "finish_reason": response["finish_reason"],
    }
    freeze(path, saved)
    return saved


def calculate_mask(folder, record, calculator, config):
    path = folder / "masks" / f'{record["prompt_id"]}.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["source_sha256"] != record["source_sha256"] or saved["runtime_sha256"] != digest(config):
            raise ValueError("Mask checkpoint mismatch")
        return saved
    started = time.monotonic()
    result = calculator.calculate(record["source"])
    saved = {
        "prompt_id": record["prompt_id"], "source_sha256": record["source_sha256"],
        "runtime_sha256": digest(config), **result,
        "elapsed_seconds": time.monotonic() - started,
    }
    freeze(path, saved)
    return saved


def generate_attacks(folder, record, reference, mask, api, config):
    rows = []
    for candidate in range(K):
        path = folder / "attacks" / f'{record["prompt_id"]}-{candidate}.json'
        if path.exists():
            row = json.loads(path.read_text())
            if (row["source_sha256"] != record["source_sha256"]
                    or row["runtime_sha256"] != digest(config)):
                raise ValueError("Attack checkpoint mismatch")
        else:
            prompt = (
                ATTACK_PROMPT + f"reference: {reference['reference']}\n"
                f"incomplete pragraph: {mask['blank_text']}\n"
                f"Candidate variant: {candidate}"
            )
            response = api.request(
                f'attack-{record["prompt_id"]}-{candidate}',
                request_body(prompt, .7, 256))
            row = {
                "prompt_id": record["prompt_id"], "candidate": candidate,
                "source": record["source"], "source_sha256": record["source_sha256"],
                "reference_sha256": reference["reference_sha256"],
                "mask_sha256": digest(mask), "runtime_sha256": digest(config),
                "attack": response["content"].strip(),
                "attack_sha256": text_sha(response["content"].strip()),
                "finish_reason": response["finish_reason"],
            }
            freeze(path, row)
        rows.append(row)
    return rows


def review_and_detect(folder, record, rows, detector, api, config):
    pid = record["prompt_id"]
    quality_path = folder / "quality" / f"{pid}.json"
    group_hash = digest(rows)
    if quality_path.exists():
        quality = json.loads(quality_path.read_text())
        if quality["group_sha256"] != group_hash or quality["runtime_sha256"] != digest(config):
            raise ValueError("Quality checkpoint mismatch")
    else:
        prompt = REVIEW + "SOURCE:\n" + record["source"] + "\n\n" + "\n\n".join(
            f'CANDIDATE {row["candidate"]}:\n{row["attack"]}' for row in rows)
        response = api.request(f"review-{pid}", request_body(prompt, 0, 700))
        if response["finish_reason"] != "stop":
            raise ValueError("Truncated quality review")
        reviews = decode_reviews(response["content"])
        labeled = []
        for row in rows:
            details = detector.parse(row["attack"]) if row["attack"] else []
            surface = gates(row, [detail["amr"] for detail in details])
            review = reviews[row["candidate"]]
            valid = (all(surface.values()) and review["score"] >= 4
                     and not review["major_error"] and row["finish_reason"] == "stop")
            labeled.append({
                **row, "details": details, "surface": surface,
                "review": review, "quality_valid": valid,
            })
        quality = {
            "prompt_id": pid, "group_sha256": group_hash,
            "runtime_sha256": digest(config), "rows": labeled,
            "no_feedback_candidate": next(
                (row["candidate"] for row in labeled if row["quality_valid"]), None),
        }
        freeze(quality_path, quality)
    evaluated_path = folder / "evaluated" / f"{pid}.json"
    if evaluated_path.exists():
        saved = json.loads(evaluated_path.read_text())
        if saved["quality_sha256"] != digest(quality):
            raise ValueError("Evaluation checkpoint mismatch")
        return saved
    scored = [
        {**row, "detection": detector.score(row["details"]) if row["details"] else None}
        for row in quality["rows"]
    ]
    eligible = [row for row in scored if row["quality_valid"] and row["detection"]]
    best = min(eligible, key=lambda row: (row["detection"]["z"], row["candidate"])) if eligible else None
    first = next(
        (row for row in eligible if row["candidate"] == quality["no_feedback_candidate"]), None)
    saved = {
        "prompt_id": pid, "quality_sha256": digest(quality),
        "runtime_sha256": digest(config), "rows": scored,
        "feedback_candidate": best["candidate"] if best else None,
        "feedback_success": bool(best and best["detection"]["z"] <= 2.33),
        "no_feedback_candidate": quality["no_feedback_candidate"],
        "no_feedback_success": bool(first and first["detection"]["z"] <= 2.33),
    }
    freeze(evaluated_path, saved)
    print(json.dumps({
        "stage": "evaluated", "id": pid, "quality_candidates": len(eligible),
        "best_z": best["detection"]["z"] if best else None,
    }), flush=True)
    return saved


def summarize(folder):
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        return {"status": "not_prepared"}
    planned = len(json.loads(manifest_path.read_text())["records"])
    counts = {
        name: len(list((folder / name).glob("*.json")))
        for name in ("references", "masks", "attacks", "quality", "evaluated")
    }
    evaluated = [json.loads(path.read_text()) for path in (folder / "evaluated").glob("*.json")]
    calls = [json.loads(path.read_text()) for path in (folder / "calls").glob("*.json")]
    costs = {}
    for row in calls:
        stage = row["scope"].split("-")[0]
        item = costs.setdefault(stage, {
            "attempted_calls": 0, "complete_calls": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "unknown_usage": 0,
        })
        item["attempted_calls"] += 1
        if row["status"] == "complete":
            item["complete_calls"] += 1
            usage = row["response"].get("usage", {})
            if not all(key in usage for key in ("prompt_tokens", "completion_tokens")):
                item["unknown_usage"] += 1
            for key in ("prompt_tokens", "completion_tokens"):
                item[key] += usage.get(key, 0)
    complete = (
        counts["references"] == planned and counts["masks"] == planned
        and counts["attacks"] == planned * K and counts["quality"] == planned
        and counts["evaluated"] == planned
    )
    result = {
        "status": "complete" if complete else "incomplete",
        "planned_paragraphs": planned, "artifact_counts": counts,
        "quality_candidates": sum(
            row["quality_valid"] for group in evaluated for row in group["rows"]),
        "feedback_successes": sum(row["feedback_success"] for row in evaluated),
        "no_feedback_successes": sum(row["no_feedback_success"] for row in evaluated),
        "feedback_valid_asr": (
            sum(row["feedback_success"] for row in evaluated) / planned if complete else None),
        "no_feedback_valid_asr": (
            sum(row["no_feedback_success"] for row in evaluated) / planned if complete else None),
        "api_costs": costs,
    }
    write_json(folder / "execution_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "summarize"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-calls", type=int, default=300)
    parser.add_argument("--key-stdin", action="store_true")
    args = parser.parse_args()
    for key, value in {
        "NLTK_DATA": ".nltk_data", "HF_HOME": ".hf-cache",
        "GLOVE_VECTORS_PATH": "vectors/glove.6B.100d.txt",
    }.items():
        os.environ.setdefault(key, str(ROOT / value))
    if args.stage == "prepare":
        prepare(args.output)
        return
    if args.stage == "summarize":
        print(json.dumps(summarize(args.output), indent=2))
        return
    if not 1 <= args.limit <= 30 or args.max_calls < 1:
        parser.error("limit must be 1..30 and max-calls positive")
    if args.key_stdin:
        if sys.stdin.isatty():
            raise RuntimeError("Use piped stdin, not an echoing terminal")
        key = sys.stdin.readline().strip()
    else:
        key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError("API credential missing")
    api = CallJournal(args.output, key, args.max_calls)
    try:
        config = runtime(args.output, api)
        calculator = SelfInformationModel()
        detector = Detector()
        records = json.loads((args.output / "manifest.json").read_text())["records"]
        for record in records[:args.limit]:
            reference = generate_reference(args.output, record, api, config)
            mask = calculate_mask(args.output, record, calculator, config)
            rows = generate_attacks(args.output, record, reference, mask, api, config)
            review_and_detect(args.output, record, rows, detector, api, config)
    finally:
        print(json.dumps(summarize(args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
