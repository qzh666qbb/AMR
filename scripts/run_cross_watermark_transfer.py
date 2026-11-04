"""Run checkpointed no-feedback boundary transfer on KGW and SynthID-Text.

The two watermark schemes generate their own text from the same frozen prompts
with the same Qwen2.5-0.5B model and sampling budget. Human calibration and test
sets are independent from the 30 generation prompts. Quality-only candidate
selection is persisted before target detector scores are computed.
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

from evaluate_adaptive_retry import gates
from run_confirmation_experiment import (
    CallJournal,
    Detector as QualityParser,
    REVIEW,
    decode_reviews,
    digest,
    freeze,
    request_body,
)
from run_segmentation_intervention import ROOT, sha, text_sha, write_json


OUTPUT = ROOT / "experiments/cross_watermark_transfer30"
SOURCE = ROOT / "experiments/confirmation_three_arm30/preparation.json"
HUMAN_SOURCE = ROOT / "experiments/segmentation_intervention/manifest.json"
MODEL_PATH = ROOT / "models/qwen2.5-0.5b-instruct"
QWEN_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
WATERMARKS = ("kgw", "synthid")
K = 4
GENERATION = {
    "temperature": 0.5,
    "top_k": 40,
    "top_p": 0.99,
    "min_new_tokens": 160,
    "max_new_tokens": 200,
    "do_sample": True,
}
KGW = {
    "gamma": 0.25,
    "delta": 2.0,
    "seeding_scheme": "simple_1",
    "hash_key": 15485863,
    "select_green_tokens": True,
    "ignore_repeated_bigrams": True,
}
SYNTHID_KEYS = (
    654, 400, 836, 123, 340, 443, 597, 160, 57, 29,
    590, 639, 13, 715, 468, 990, 966, 226, 324, 585,
    118, 504, 421, 521, 129, 669, 732, 225, 90, 960,
)
SYNTHID = {
    "ngram_len": 5,
    "keys": list(SYNTHID_KEYS),
    "context_history_size": 1024,
    "temperature": GENERATION["temperature"],
    "top_k": GENERATION["top_k"],
}


def git_commit(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def calibrated_threshold(scores, alpha=0.01):
    """Largest strict-positive threshold with empirical FPR at most alpha."""
    if not scores or not 0 < alpha < 1:
        raise ValueError("Calibration requires scores and 0 < alpha < 1")
    if not all(math.isfinite(value) for value in scores):
        raise ValueError("Calibration scores must be finite")
    return sorted(scores, reverse=True)[math.floor(alpha * len(scores))]


def generation_seed(prompt_id):
    return int(text_sha(f"cross-watermark-v1:{prompt_id}")[:8], 16)


def synthid_module():
    source = ROOT / "third_party/synthid-text/src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from synthid_text import logits_processing
    return logits_processing


def make_synthid_processor():
    import torch

    logits_processing = synthid_module()
    return logits_processing.SynthIDLogitsProcessor(
        ngram_len=SYNTHID["ngram_len"],
        keys=SYNTHID["keys"],
        context_history_size=SYNTHID["context_history_size"],
        temperature=float(SYNTHID["temperature"]),
        top_k=SYNTHID["top_k"],
        device=torch.device("cpu"),
    )


class DenseCpuSynthIDProcessor:
    """Adapt official sparse SynthID processing to dense HF generation logits.

    The full logits and token IDs move to CPU before official hashing. Only the
    returned dense top-k distribution moves back to the model device. This
    avoids the incorrect MPS int64 bit-shift fallback.
    """

    def __init__(self):
        self.processor = make_synthid_processor()

    def __call__(self, input_ids, scores):
        import torch

        cpu_ids = input_ids.detach().to(device="cpu", dtype=torch.long)
        cpu_scores = scores.detach().to(device="cpu", dtype=torch.float32)
        updated, indices, _ = self.processor.watermarked_call(
            cpu_ids, cpu_scores
        )
        dense = torch.full_like(cpu_scores, -math.inf)
        dense.scatter_(1, indices, updated)
        return dense.to(device=scores.device, dtype=scores.dtype)


def synthid_score_token_ids(token_ids):
    import torch

    processor = make_synthid_processor()
    ids = torch.as_tensor(token_ids, dtype=torch.long, device="cpu").unsqueeze(0)
    if ids.shape[1] < SYNTHID["ngram_len"]:
        raise ValueError("SynthID detection needs at least one complete n-gram")
    g_values = processor.compute_g_values(ids)
    repetition = processor.compute_context_repetition_mask(ids)
    eos_mask = processor.compute_eos_token_mask(ids, eos_token_id=-1)[
        :, SYNTHID["ngram_len"] - 1:
    ]
    mask = repetition * eos_mask
    count = int(mask.sum().item())
    if count < 1:
        raise ValueError("SynthID detection has no usable n-grams")
    depth = g_values.shape[-1]
    score = float(
        (g_values * mask.unsqueeze(2)).sum().item() / (count * depth)
    )
    return {
        "score": score,
        "usable_ngrams": count,
        "depth": depth,
        "green_values": int((g_values * mask.unsqueeze(2)).sum().item()),
        "device": "cpu",
    }


class WatermarkRuntime:
    def __init__(self, load_model=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_PATH), local_files_only=True
        )
        self.model = None
        if load_model:
            dtype = torch.float16 if self.device.type == "mps" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                str(MODEL_PATH),
                local_files_only=True,
                torch_dtype=dtype,
                attn_implementation="eager",
            ).to(self.device).eval()
        lm_path = ROOT / "third_party/lm-watermarking"
        if str(lm_path) not in sys.path:
            sys.path.insert(0, str(lm_path))
        from watermark_processor import WatermarkDetector, WatermarkLogitsProcessor

        vocab = list(self.tokenizer.get_vocab().values())
        self.kgw_processor_type = WatermarkLogitsProcessor
        self.kgw_detector = WatermarkDetector(
            vocab=vocab,
            gamma=KGW["gamma"],
            seeding_scheme=KGW["seeding_scheme"],
            hash_key=KGW["hash_key"],
            select_green_tokens=KGW["select_green_tokens"],
            device=self.device,
            tokenizer=self.tokenizer,
            z_threshold=4.0,
            normalizers=[],
            ignore_repeated_bigrams=KGW["ignore_repeated_bigrams"],
        )

    def score(self, watermark, text):
        if watermark == "kgw":
            result = self.kgw_detector.detect(
                text=text, return_prediction=False, return_scores=True
            )
            return {
                "score": float(result["z_score"]),
                "num_tokens_scored": int(result["num_tokens_scored"]),
                "num_green_tokens": int(result["num_green_tokens"]),
                "green_fraction": float(result["green_fraction"]),
                "device": str(self.device),
            }
        if watermark == "synthid":
            token_ids = self.tokenizer(
                text, add_special_tokens=False
            )["input_ids"]
            return synthid_score_token_ids(token_ids)
        raise ValueError(f"Unknown watermark: {watermark}")

    def generate(self, watermark, prompt, seed):
        from transformers import LogitsProcessorList

        if self.model is None:
            raise RuntimeError("Model was not loaded")
        torch = self.torch
        torch.manual_seed(seed)
        encoded = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self.device)
        if watermark == "kgw":
            processor = self.kgw_processor_type(
                vocab=list(self.tokenizer.get_vocab().values()),
                gamma=KGW["gamma"],
                delta=KGW["delta"],
                seeding_scheme=KGW["seeding_scheme"],
                hash_key=KGW["hash_key"],
                select_green_tokens=KGW["select_green_tokens"],
            )
            sampling = {
                "temperature": GENERATION["temperature"],
                "top_k": GENERATION["top_k"],
                "top_p": GENERATION["top_p"],
            }
        elif watermark == "synthid":
            processor = DenseCpuSynthIDProcessor()
            # The official SynthID processor applies temperature and top-k.
            sampling = {"temperature": 1.0, "top_k": 0, "top_p": 1.0}
        else:
            raise ValueError(f"Unknown watermark: {watermark}")
        started = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                logits_processor=LogitsProcessorList([processor]),
                do_sample=True,
                min_new_tokens=GENERATION["min_new_tokens"],
                max_new_tokens=GENERATION["max_new_tokens"],
                pad_token_id=self.tokenizer.eos_token_id,
                **sampling,
            )
        generated_ids = output[0, encoded["input_ids"].shape[1]:].detach().cpu()
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if not text:
            raise ValueError("Watermarked generation returned empty text")
        return {
            "text": text,
            "text_sha256": text_sha(text),
            "generated_token_ids": generated_ids.tolist(),
            "generated_tokens": len(generated_ids),
            "elapsed_seconds": time.monotonic() - started,
            "seed": seed,
            "model_device": str(self.device),
            "hash_device": str(self.device) if watermark == "kgw" else "cpu",
        }


def prepare(folder):
    source = json.loads(SOURCE.read_text())
    human = json.loads(HUMAN_SOURCE.read_text())
    prompts = [
        {
            "prompt_id": row["prompt_id"],
            "generation_prompt": row["generation_prompt"],
            "generation_prompt_sha256": row["generation_prompt_sha256"],
            "source_shard_row": row["shard_row"],
        }
        for row in source["prompts"]
    ]
    calibration = [
        {
            "id": row["id"],
            "cohort": row["cohort"],
            "shard_row": row["shard_row"],
            "text": row["text"],
            "text_sha256": row["text_sha256"],
        }
        for row in human["records"]
        if row["cohort"] in ("human_calibration", "human_test")
    ]
    if len(prompts) != 30 or len(calibration) != 500:
        raise ValueError("Expected 30 prompts and two independent 250-text cohorts")
    attack_template = source["attack"]["templates"]["boundary_only"]
    dependencies = [
        Path(__file__),
        SOURCE,
        HUMAN_SOURCE,
        ROOT / "scripts/run_confirmation_experiment.py",
        ROOT / "scripts/evaluate_adaptive_retry.py",
        ROOT / "third_party/lm-watermarking/watermark_processor.py",
        ROOT / "third_party/synthid-text/src/synthid_text/logits_processing.py",
        ROOT / "third_party/synthid-text/src/synthid_text/hashing_function.py",
        *sorted(path for path in MODEL_PATH.iterdir() if path.is_file()),
    ]
    payload = {
        "status": "frozen_before_formal_generation",
        "protocol": {
            "name": "cross-watermark no-feedback boundary transfer",
            "watermarks": list(WATERMARKS),
            "generator": {
                "name": "Qwen/Qwen2.5-0.5B-Instruct",
                "revision": QWEN_REVISION,
                "prompt_mode": "raw continuation",
                **GENERATION,
            },
            "kgw": {
                **KGW,
                "implementation": "official minimal watermark_processor.py",
                "repository_commit": git_commit(
                    ROOT / "third_party/lm-watermarking"
                ),
                "generation_and_detection_rng_device": "mps_if_available_else_cpu",
            },
            "synthid": {
                **SYNTHID,
                "implementation": "official logits_processing.py",
                "repository_commit": git_commit(
                    ROOT / "third_party/synthid-text"
                ),
                "generation_adapter": (
                    "Dense HF wrapper around official sparse watermarked_call"
                ),
                "hash_and_g_value_device": "cpu_only",
                "detector": "official mean score formula",
            },
            "calibration": {
                "target_fpr": 0.01,
                "rule": (
                    "strict score > threshold; threshold fixed from 250 human "
                    "calibration texts, then evaluated on 250 human test texts"
                ),
            },
            "attack": {
                "name": "boundary_only",
                "k": K,
                "template": attack_template,
                "template_sha256": text_sha(attack_template),
                "model": "deepseek-v4-flash",
                "temperature": 0.7,
                "max_tokens": 1400,
            },
            "quality": {
                "review": REVIEW,
                "review_temperature": 0,
                "review_max_tokens": 700,
                "selection": (
                    "first quality-valid candidate by index, persisted before "
                    "any target attack score is read"
                ),
                "validity": (
                    "all surface gates, semantic score >=4, no major error, "
                    "and complete generation"
                ),
            },
            "reporting_scope": (
                "Results are specific to Qwen2.5-0.5B-Instruct generation"
            ),
        },
        "prompts": prompts,
        "human_records": calibration,
        "dependencies": {
            str(path.relative_to(ROOT)): sha(path) for path in dependencies
        },
    }
    freeze(folder / "manifest.json", payload)
    print(json.dumps({
        "stage": "prepared",
        "prompts": len(prompts),
        "human_calibration": sum(
            row["cohort"] == "human_calibration" for row in calibration
        ),
        "human_test": sum(row["cohort"] == "human_test" for row in calibration),
        "manifest": str(folder / "manifest.json"),
    }), flush=True)


def check_manifest(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    for relative, expected in manifest["dependencies"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Frozen dependency changed: {relative}")
    return manifest


def calibrate(folder, watermark, cohort, offset, limit):
    manifest = check_manifest(folder)
    runtime = WatermarkRuntime(load_model=False)
    rows = [
        row for row in manifest["human_records"] if row["cohort"] == cohort
    ][offset:offset + limit]
    for ordinal, row in enumerate(rows, start=offset):
        path = folder / "calibration" / watermark / f'{row["id"]}.json'
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["text_sha256"] != row["text_sha256"]:
                raise ValueError("Calibration checkpoint mismatch")
        else:
            started = time.monotonic()
            detection = runtime.score(watermark, row["text"])
            saved = {
                "id": row["id"],
                "cohort": cohort,
                "text_sha256": row["text_sha256"],
                "watermark": watermark,
                "detection": detection,
                "elapsed_seconds": time.monotonic() - started,
            }
            freeze(path, saved)
        print(json.dumps({
            "stage": "calibration",
            "watermark": watermark,
            "cohort": cohort,
            "completed_in_chunk": ordinal - offset + 1,
            "chunk_size": len(rows),
            "score": saved["detection"]["score"],
        }), flush=True)


def finalize_thresholds(folder):
    manifest = check_manifest(folder)
    expected = {
        cohort: {
            row["id"] for row in manifest["human_records"]
            if row["cohort"] == cohort
        }
        for cohort in ("human_calibration", "human_test")
    }
    result = {}
    for watermark in WATERMARKS:
        rows = [
            json.loads(path.read_text())
            for path in sorted((folder / "calibration" / watermark).glob("*.json"))
        ]
        by_cohort = {
            cohort: [row for row in rows if row["cohort"] == cohort]
            for cohort in expected
        }
        for cohort, values in by_cohort.items():
            if {row["id"] for row in values} != expected[cohort]:
                raise ValueError(
                    f"Incomplete {watermark} {cohort}: "
                    f"{len(values)}/{len(expected[cohort])}"
                )
        scores = [
            row["detection"]["score"]
            for row in by_cohort["human_calibration"]
        ]
        threshold = calibrated_threshold(scores)
        calibration_fp = sum(value > threshold for value in scores)
        test_scores = [
            row["detection"]["score"] for row in by_cohort["human_test"]
        ]
        test_fp = sum(value > threshold for value in test_scores)
        result[watermark] = {
            "threshold": threshold,
            "strict_comparison": "score > threshold",
            "calibration_n": len(scores),
            "calibration_false_positives": calibration_fp,
            "calibration_fpr": calibration_fp / len(scores),
            "test_n": len(test_scores),
            "test_false_positives": test_fp,
            "test_fpr": test_fp / len(test_scores),
        }
    freeze(folder / "thresholds.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def generate_baselines(folder, watermarks, offset, limit):
    manifest = check_manifest(folder)
    thresholds = json.loads((folder / "thresholds.json").read_text())
    runtime = WatermarkRuntime(load_model=True)
    prompts = manifest["prompts"][offset:offset + limit]
    for prompt_index, prompt in enumerate(prompts, start=offset):
        for watermark in watermarks:
            path = folder / "baselines" / watermark / f'{prompt["prompt_id"]}.json'
            if path.exists():
                saved = json.loads(path.read_text())
                if (
                    saved["generation_prompt_sha256"]
                    != prompt["generation_prompt_sha256"]
                ):
                    raise ValueError("Baseline checkpoint mismatch")
            else:
                generated = runtime.generate(
                    watermark,
                    prompt["generation_prompt"],
                    generation_seed(prompt["prompt_id"]),
                )
                detection = runtime.score(watermark, generated["text"])
                saved = {
                    "prompt_id": prompt["prompt_id"],
                    "watermark": watermark,
                    "generation_prompt_sha256":
                        prompt["generation_prompt_sha256"],
                    **generated,
                    "detection": detection,
                    "threshold": thresholds[watermark]["threshold"],
                    "detected": (
                        detection["score"] > thresholds[watermark]["threshold"]
                    ),
                }
                freeze(path, saved)
            print(json.dumps({
                "stage": "baseline",
                "prompt": prompt_index + 1,
                "planned_prompts": len(manifest["prompts"]),
                "watermark": watermark,
                "tokens": saved["generated_tokens"],
                "score": saved["detection"]["score"],
                "threshold": saved["threshold"],
                "detected": saved["detected"],
            }), flush=True)


def attack_rows(folder, baseline, template, api):
    rows = []
    watermark, prompt_id = baseline["watermark"], baseline["prompt_id"]
    for candidate in range(K):
        path = folder / "attacks" / watermark / f"{prompt_id}-{candidate}.json"
        if path.exists():
            row = json.loads(path.read_text())
            if row["source_sha256"] != baseline["text_sha256"]:
                raise ValueError("Attack checkpoint mismatch")
        else:
            response = api.request(
                f"cross-attack-{watermark}-{prompt_id}-{candidate}",
                request_body(
                    template.format(baseline["text"], candidate=candidate),
                    0.7,
                    1400,
                ),
            )
            row = {
                "prompt_id": prompt_id,
                "watermark": watermark,
                "candidate": candidate,
                "source": baseline["text"],
                "source_sha256": baseline["text_sha256"],
                "attack": response["content"].strip(),
                "attack_sha256": text_sha(response["content"].strip()),
                "finish_reason": response["finish_reason"],
            }
            freeze(path, row)
        rows.append(row)
    return rows


def quality_select(folder, baseline, rows, parser, api):
    watermark, prompt_id = baseline["watermark"], baseline["prompt_id"]
    path = folder / "quality" / watermark / f"{prompt_id}.json"
    group_hash = digest(rows)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["group_sha256"] != group_hash:
            raise ValueError("Quality checkpoint mismatch")
        return saved
    prompt = REVIEW + "SOURCE:\n" + baseline["text"] + "\n\n" + "\n\n".join(
        f'CANDIDATE {row["candidate"]}:\n{row["attack"]}' for row in rows
    )
    response = api.request(
        f"cross-review-{watermark}-{prompt_id}",
        request_body(prompt, 0, 700),
    )
    if response["finish_reason"] != "stop":
        raise ValueError("Truncated quality review")
    reviews = decode_reviews(response["content"])
    labeled = []
    for row in rows:
        details = parser.parse(row["attack"]) if row["attack"] else []
        surface = gates(row, [detail["amr"] for detail in details])
        review = reviews[row["candidate"]]
        valid = (
            all(surface.values())
            and review["score"] >= 4
            and not review["major_error"]
            and row["finish_reason"] == "stop"
        )
        labeled.append({
            **row,
            "details": details,
            "surface": surface,
            "review": review,
            "quality_valid": valid,
        })
    selected = next(
        (row["candidate"] for row in labeled if row["quality_valid"]), None
    )
    saved = {
        "prompt_id": prompt_id,
        "watermark": watermark,
        "group_sha256": group_hash,
        "rows": labeled,
        "no_feedback_candidate": selected,
        "selection_timestamp": datetime.now(timezone.utc).isoformat(),
        "target_scores_read": False,
    }
    freeze(path, saved)
    return saved


def run_attacks_and_quality(folder, watermarks, offset, limit, key, max_calls):
    manifest = check_manifest(folder)
    api = CallJournal(folder, key, max_calls)
    parser = QualityParser()
    template = manifest["protocol"]["attack"]["template"]
    for prompt_index, prompt in enumerate(
        manifest["prompts"][offset:offset + limit], start=offset
    ):
        for watermark in watermarks:
            baseline_path = (
                folder / "baselines" / watermark / f'{prompt["prompt_id"]}.json'
            )
            if not baseline_path.exists():
                raise FileNotFoundError(f"Missing baseline: {baseline_path}")
            baseline = json.loads(baseline_path.read_text())
            rows = attack_rows(folder, baseline, template, api)
            quality = quality_select(folder, baseline, rows, parser, api)
            print(json.dumps({
                "stage": "quality",
                "prompt": prompt_index + 1,
                "planned_prompts": len(manifest["prompts"]),
                "watermark": watermark,
                "quality_valid": sum(
                    row["quality_valid"] for row in quality["rows"]
                ),
                "no_feedback_candidate": quality["no_feedback_candidate"],
                "new_api_calls_this_process": api.sent,
            }), flush=True)


def detect_attacks(folder, watermarks, offset, limit):
    manifest = check_manifest(folder)
    thresholds = json.loads((folder / "thresholds.json").read_text())
    runtime = WatermarkRuntime(load_model=False)
    for prompt_index, prompt in enumerate(
        manifest["prompts"][offset:offset + limit], start=offset
    ):
        for watermark in watermarks:
            quality_path = (
                folder / "quality" / watermark / f'{prompt["prompt_id"]}.json'
            )
            quality = json.loads(quality_path.read_text())
            if quality.get("target_scores_read") is not False:
                raise ValueError("Quality selection provenance is not score-blind")
            path = (
                folder / "evaluated" / watermark / f'{prompt["prompt_id"]}.json'
            )
            if path.exists():
                saved = json.loads(path.read_text())
                if saved["quality_sha256"] != digest(quality):
                    raise ValueError("Evaluation checkpoint mismatch")
            else:
                rows = []
                for row in quality["rows"]:
                    detection = runtime.score(watermark, row["attack"])
                    rows.append({**row, "target_detection": detection})
                selected = next(
                    (
                        row for row in rows
                        if row["candidate"] == quality["no_feedback_candidate"]
                    ),
                    None,
                )
                threshold = thresholds[watermark]["threshold"]
                saved = {
                    "prompt_id": prompt["prompt_id"],
                    "watermark": watermark,
                    "quality_sha256": digest(quality),
                    "threshold": threshold,
                    "rows": rows,
                    "no_feedback_candidate": quality["no_feedback_candidate"],
                    "no_feedback_success": bool(
                        selected
                        and selected["quality_valid"]
                        and selected["target_detection"]["score"] <= threshold
                    ),
                }
                freeze(path, saved)
            print(json.dumps({
                "stage": "attack_detection",
                "prompt": prompt_index + 1,
                "planned_prompts": len(manifest["prompts"]),
                "watermark": watermark,
                "selected": saved["no_feedback_candidate"],
                "success": saved["no_feedback_success"],
            }), flush=True)


def summarize(folder):
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        return {"status": "not_prepared"}
    manifest = json.loads(manifest_path.read_text())
    planned = len(manifest["prompts"])
    result = {
        "status": "incomplete",
        "planned_prompts_per_watermark": planned,
        "watermarks": {},
    }
    complete = (folder / "thresholds.json").exists()
    for watermark in WATERMARKS:
        baselines = [
            json.loads(path.read_text())
            for path in sorted((folder / "baselines" / watermark).glob("*.json"))
        ]
        quality = [
            json.loads(path.read_text())
            for path in sorted((folder / "quality" / watermark).glob("*.json"))
        ]
        evaluated = [
            json.loads(path.read_text())
            for path in sorted((folder / "evaluated" / watermark).glob("*.json"))
        ]
        watermark_complete = (
            len(baselines) == len(quality) == len(evaluated) == planned
        )
        complete = complete and watermark_complete
        baseline_detected = sum(row["detected"] for row in baselines)
        quality_candidates = sum(
            row["quality_valid"] for group in quality for row in group["rows"]
        )
        quality_coverage = sum(
            group["no_feedback_candidate"] is not None for group in quality
        )
        successes = sum(row["no_feedback_success"] for row in evaluated)
        candidate_low = 0
        candidate_total = 0
        for group in evaluated:
            for row in group["rows"]:
                if row["quality_valid"]:
                    candidate_total += 1
                    candidate_low += (
                        row["target_detection"]["score"] <= group["threshold"]
                    )
        conditional_ids = {
            row["prompt_id"] for row in baselines if row["detected"]
        }
        conditional_success = sum(
            row["no_feedback_success"]
            and row["prompt_id"] in conditional_ids
            for row in evaluated
        )
        result["watermarks"][watermark] = {
            "artifact_counts": {
                "baselines": len(baselines),
                "quality_groups": len(quality),
                "evaluated_groups": len(evaluated),
            },
            "baseline_detected": baseline_detected,
            "baseline_detection_rate": (
                baseline_detected / planned if watermark_complete else None
            ),
            "quality_candidates": quality_candidates,
            "quality_candidate_rate": (
                quality_candidates / (planned * K)
                if watermark_complete else None
            ),
            "quality_coverage": quality_coverage,
            "quality_coverage_rate": (
                quality_coverage / planned if watermark_complete else None
            ),
            "no_feedback_successes": successes,
            "no_feedback_valid_asr": (
                successes / planned if watermark_complete else None
            ),
            "conditional_initially_detected_n": len(conditional_ids),
            "conditional_successes": conditional_success,
            "conditional_valid_asr": (
                conditional_success / len(conditional_ids)
                if watermark_complete and conditional_ids else None
            ),
            "quality_valid_candidate_low_score": candidate_low,
            "quality_valid_candidate_total": candidate_total,
            "quality_valid_candidate_low_score_rate": (
                candidate_low / candidate_total
                if watermark_complete and candidate_total else None
            ),
        }
    calls = [
        json.loads(path.read_text())
        for path in sorted((folder / "calls").glob("*.json"))
    ]
    result["api"] = {
        "attempted_calls": len(calls),
        "complete_calls": sum(row["status"] == "complete" for row in calls),
        "prompt_tokens": sum(
            row.get("response", {}).get("usage", {}).get("prompt_tokens", 0)
            for row in calls
        ),
        "completion_tokens": sum(
            row.get("response", {}).get("usage", {}).get(
                "completion_tokens", 0
            )
            for row in calls
        ),
    }
    result["status"] = "complete" if complete else "incomplete"
    write_json(folder / "execution_summary.json", result)
    return result


def read_key(args):
    if args.key_stdin:
        if sys.stdin.isatty():
            raise RuntimeError("Use piped stdin, not an echoing terminal")
        key = sys.stdin.readline().strip()
    else:
        key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not key and args.key_file:
            lines = [
                line.strip() for line in args.key_file.read_text().splitlines()
                if line.strip()
            ]
            key = lines[-1] if lines else ""
    if not key:
        raise RuntimeError("API credential missing")
    return key


def smoke():
    runtime = WatermarkRuntime(load_model=True)
    source = json.loads(SOURCE.read_text())["prompts"][0]
    rows = {}
    for watermark in WATERMARKS:
        generated = runtime.generate(
            watermark,
            source["generation_prompt"],
            generation_seed(source["prompt_id"]),
        )
        detection = runtime.score(watermark, generated["text"])
        rows[watermark] = {
            "generated_tokens": generated["generated_tokens"],
            "elapsed_seconds": generated["elapsed_seconds"],
            "score": detection["score"],
            "text_sha256": generated["text_sha256"],
            "preview": generated["text"][:180],
        }
    print(json.dumps({"stage": "smoke_complete", "results": rows}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "smoke", "prepare", "calibrate", "finalize-thresholds",
            "generate", "attack-quality", "detect", "summarize",
        ),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--watermark", choices=(*WATERMARKS, "both"), default="both")
    parser.add_argument(
        "--cohort",
        choices=("human_calibration", "human_test"),
        default="human_calibration",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--max-calls", type=int, default=300)
    parser.add_argument("--key-stdin", action="store_true")
    parser.add_argument("--key-file", type=Path, default=ROOT / "api.txt")
    args = parser.parse_args()
    if args.offset < 0 or args.limit < 1 or args.max_calls < 1:
        parser.error("offset must be nonnegative; limit and max-calls positive")
    os.environ.setdefault("NLTK_DATA", str(ROOT / ".nltk_data"))
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
    os.environ.setdefault(
        "GLOVE_VECTORS_PATH", str(ROOT / "vectors/glove.6B.100d.txt")
    )
    watermarks = WATERMARKS if args.watermark == "both" else (args.watermark,)
    if args.stage == "smoke":
        smoke()
    elif args.stage == "prepare":
        prepare(args.output)
    elif args.stage == "calibrate":
        for watermark in watermarks:
            calibrate(
                args.output, watermark, args.cohort, args.offset, args.limit
            )
    elif args.stage == "finalize-thresholds":
        finalize_thresholds(args.output)
    elif args.stage == "generate":
        generate_baselines(
            args.output, watermarks, args.offset, min(args.limit, 30)
        )
    elif args.stage == "attack-quality":
        run_attacks_and_quality(
            args.output,
            watermarks,
            args.offset,
            min(args.limit, 30),
            read_key(args),
            args.max_calls,
        )
    elif args.stage == "detect":
        detect_attacks(
            args.output, watermarks, args.offset, min(args.limit, 30)
        )
    else:
        print(json.dumps(summarize(args.output), indent=2))


if __name__ == "__main__":
    main()
