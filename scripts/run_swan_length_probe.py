"""Bounded preflight of SWAN sentence length control, not an attack-ASR study."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time

from feedback2_runtime import CachedDetector, Heartbeat, claim
from prepare_confirmation_experiment import choose_prompts, stream_validation
from run_confirmation_experiment import CallJournal, digest, freeze, request_body
from run_segmentation_intervention import ROOT, sha, text_sha, write_json
from supervise_confirmation_experiment import read_credential

OUTPUT = ROOT / "experiments/swan_length_probe_v2"
LENGTHS = {"short": [5, 10], "medium": [24, 36], "long": [48, 64]}
SHORTEST = "- Prefer the shortest natural sentence that realizes the complete AMR structure."


def length_prompt(original, low, high):
    if original.count(SHORTEST) != 1:
        raise ValueError("Upstream prompt changed; review length-control amendment")
    return original.replace(SHORTEST, (
        f"- Produce between {low} and {high} whitespace-separated words in the single sentence. "
        "Keep all the AMR structure constraints above; do not add unsupported facts to reach the length."
    ))


def prepare(folder, reuse_preparation=None):
    sys.path.insert(0, str(ROOT / "third_party/SWAN"))
    from injection.injection_amr_utils import build_user_prompt, build_system_prompt, load_amr_examples

    if (folder / "manifest.json").exists():
        check(folder)
        return
    previous_path = ROOT / "experiments/confirmation_three_arm30/preparation.json"
    history_path = ROOT / "experiments/planE_e2_joint250/quality_aware_reviews.json"
    humans_path = ROOT / "experiments/segmentation_intervention/manifest.json"
    previous = json.loads(previous_path.read_text())
    excluded = [r["text"] for r in previous["prompts"]]
    excluded += [r["source"] for r in json.loads(history_path.read_text())]
    excluded += [r["text"] for r in json.loads(humans_path.read_text())["records"]]
    if reuse_preparation:
        old = json.loads((reuse_preparation / "manifest.json").read_text())
        calls = [json.loads(p.read_text()) for p in (reuse_preparation / "calls").glob("*.json")]
        if any(c.get("error_type") != "InvalidHeader" or "http_status" in c for c in calls):
            raise ValueError("Can only reuse a preparation with local pre-transmission failures")
        if any((reuse_preparation / "evaluated").glob("*.json")):
            raise ValueError("Cannot reuse inspected generated probe texts")
        prompts, rejected, scanned = old["prompts"], old["rejected"], old["scanned_rows"]
    else:
        stream = stream_validation()
        try:
            prompts, rejected, scanned = choose_prompts(stream, 3, excluded)
        finally:
            stream.close()
    bank_path = ROOT / "third_party/SWAN/amr_bank/banks/amr_bank_50.json"
    examples_path = ROOT / "third_party/SWAN/amr_bank/artifacts/normalized_amr_examples.json"
    bank = [r["normalized_template_amr"] for r in json.loads(bank_path.read_text())]
    if len(bank) != 50 or len(set(bank)) != 50:
        raise ValueError("Expected 50 distinct templates in official order")
    examples = load_amr_examples(str(examples_path), 3)
    jobs = []
    for prompt in prompts:
        for replicate in range(2):
            template_index = int(text_sha(f"length-probe-v1:{prompt['prompt_id']}:{replicate}")[:16], 16) % len(bank)
            original = build_user_prompt(bank[template_index], examples, prompt["generation_prompt"])
            for stratum, (low, high) in LENGTHS.items():
                jobs.append({
                    "id": f"{prompt['prompt_id']}-{replicate}-{stratum}",
                    "prompt_id": prompt["prompt_id"], "replicate": replicate, "stratum": stratum,
                    "word_range": [low, high], "template_index": template_index,
                    "template": bank[template_index],
                    "request": request_body(length_prompt(original, low, high), .7, 180,
                                            build_system_prompt(), .9),
                })
    if reuse_preparation and jobs != old["jobs"]:
        raise ValueError("Prepared requests changed during credential-only recovery")
    paths = [Path(__file__), ROOT / "scripts/feedback2_runtime.py",
             ROOT / "scripts/supervise_confirmation_experiment.py",
             ROOT / "scripts/run_confirmation_experiment.py",
             ROOT / "scripts/run_segmentation_intervention.py",
             ROOT / "scripts/prepare_confirmation_experiment.py",
             ROOT / "third_party/SWAN/injection/injection_amr_utils.py",
             ROOT / "third_party/SWAN/utils/amr_utils.py",
             ROOT / "vectors/glove.6B.100d.txt", bank_path, examples_path,
             previous_path, history_path, humans_path]
    for name in ("model_parse_xfm_bart_large-v0_1_0", "bart-large-tokenizer"):
        paths.extend(p for p in (ROOT / "models" / name).glob("*") if p.is_file())
    paths.extend((ROOT / ".nltk_data/tokenizers/punkt_tab/english").glob("*.tab"))
    paths.extend((ROOT / ".nltk_data/tokenizers/punkt_tab/english").glob("*.txt"))
    paths.extend((ROOT / "third_party/SWAN/amr-metric-suite/py3-Smatch-and-S2match/smatch").glob("*.py"))
    paths.extend((ROOT / ".nltk_data/corpora").glob("propbank.zip"))
    manifest = {
        "status": "frozen_before_generation", "created_at": datetime.now(timezone.utc).isoformat(),
        "preparation_reused_from": str(reuse_preparation) if reuse_preparation else None,
        "parent_manifest_sha256": sha(reuse_preparation / "manifest.json") if reuse_preparation else None,
        "purpose": "Engineering feasibility of longer SWAN sentences at fixed structural templates",
        "prompts": prompts, "jobs": jobs, "scanned_rows": scanned, "rejected": rejected,
        "protocol": {
            "planned_calls_including_smoke": len(jobs) + 1,
            "lengths_words_per_sentence": LENGTHS,
            "corresponding_five_sentence_paragraph_ranges": {
                k: [5 * low, 5 * high] for k, (low, high) in LENGTHS.items()},
            "paired_units": "Same new C4 prompt and template in each length stratum; two template draws per prompt",
            "retry_policy": "One draw per cell, no rejection sampling, correction or automatic retry",
            "injection_success": "finish=stop AND no AMR markers AND one parsed sentence AND length in range AND target S2Match>=0.75",
            "detector": "Official BART CPU, bank50, match>=0.65, null=.25; first score observation cached",
            "cache_randomness": "S2Match remains stochastic; cache shares first observation, not a repeatability study",
            "expansion_gate": "At least 4/6 joint successes in EACH stratum permits planning a paragraph pilot, not inference",
            "no_asr": "Single-sentence preflight is not paragraph injection, attack evaluation, domain generalization or FPR calibration",
            "failures": "Keep every allocated cell. Technical failures halt for diagnosis; completed bad generations remain failures",
            "formal_attack_contract": {
                "k": 4, "arms": ["plain_paraphrase", "boundary_only", "official_strong_paraphraser"],
                "primary": "All allocated paragraphs, first quality-valid candidate chosen before target scores",
                "quality": "score>=4, no major error; same number, negation, 0.7..1.6 length and AMR gates",
                "threshold": "Freeze per full pipeline using independent calibration; report independent test FPR",
                "data": "Use fresh prompts, not this inspected probe, for final tests",
            },
        },
        "dependencies": {str(p.relative_to(ROOT)): sha(p) for p in paths},
        "packages": {p: importlib.metadata.version(p) for p in
                     ("torch", "amrlib", "transformers", "nltk", "penman", "scipy", "spacy", "en_core_web_sm")},
    }
    freeze(folder / "manifest.json", manifest)


def check(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    for path, expected in manifest["dependencies"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Frozen dependency changed: {path}")
    for package, version in manifest["packages"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Frozen package changed: {package}")
    return manifest


def assess(text, finish_reason, details, target_score, word_range):
    words = len(text.split())
    marker_free = not re.search(r"(?:\b(?:NE|ADJ|NUM|LOC|AMR)\b|:(?:ARG\d|op\d|name|mode|time|location)\b)", text)
    return {
        "words": words, "sentences": len(details),
        "length_ok": word_range[0] <= words <= word_range[1],
        "single_sentence": len(details) == 1, "finish_ok": finish_reason == "stop",
        "marker_free": marker_free,
        "target_match": target_score is not None and target_score >= .75,
        "joint_success": (
            finish_reason == "stop" and marker_free and len(details) == 1
            and word_range[0] <= words <= word_range[1]
            and target_score is not None and target_score >= .75),
    }


def run(folder, key):
    manifest = check(folder)
    with Heartbeat(folder) as heartbeat:
        api = CallJournal(folder, key, manifest["protocol"]["planned_calls_including_smoke"])
        heartbeat.update(stage="api_smoke")
        probe = api.request("probe", request_body("Reply exactly OK.", 0, 16))
        if probe["finish_reason"] != "stop" or probe["content"].strip() != "OK":
            raise ValueError("Unexpected API smoke result")
        api.model_returned = probe["model"]
        runtime = freeze(folder / "runtime.json", {
            "manifest_sha256": sha(folder / "manifest.json"),
            "model_returned": probe["model"], "probe_id": probe["id"],
        })
        heartbeat.update(stage="loading_detector")
        detector = CachedDetector(folder, digest(runtime))
        for job in manifest["jobs"]:
            output = folder / "evaluated" / f"{job['id']}.json"
            heartbeat.update(stage="job", job=job["id"], new_api_calls=api.sent,
                             cache_counts=detector.counts.copy())
            if output.exists():
                saved = json.loads(output.read_text())
                if saved["job_sha256"] != digest(job) or saved["runtime_sha256"] != digest(runtime):
                    raise ValueError("Completed job checkpoint mismatch")
                continue
            started = time.monotonic()
            heartbeat.update(stage="generation")
            response = api.request(job["id"], job["request"])
            text = response["content"].strip()
            heartbeat.update(stage="parse")
            details = detector.parse(text)
            heartbeat.update(stage="score")
            detection = detector.score(details)
            target_score = (detection["details"][0]["bank_scores"][job["template_index"]]
                            if len(details) == 1 else None)
            result = {
                "id": job["id"], "stratum": job["stratum"], "prompt_id": job["prompt_id"],
                "job_sha256": digest(job), "runtime_sha256": digest(runtime),
                "text": text, "text_sha256": text_sha(text), "finish_reason": response["finish_reason"],
                "detection": detection, "target_score": target_score,
                **assess(text, response["finish_reason"], details, target_score, job["word_range"]),
                "elapsed_seconds": time.monotonic() - started,
            }
            freeze(output, result)
            print(json.dumps({k: result[k] for k in
                              ("id", "words", "sentences", "target_score", "joint_success", "elapsed_seconds")}),
                  flush=True)
        heartbeat.update(stage="completed", new_api_calls=api.sent, cache_counts=detector.counts.copy())


def summarize(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    rows = [json.loads(p.read_text()) for p in sorted((folder / "evaluated").glob("*.json"))]
    jobs = {job["id"]: job for job in manifest["jobs"]}
    runtime = json.loads((folder / "runtime.json").read_text()) if (folder / "runtime.json").exists() else None
    for row in rows:
        if row["job_sha256"] != digest(jobs[row["id"]]) or row["runtime_sha256"] != digest(runtime):
            raise ValueError("Summary checkpoint mismatch")
    strata = {}
    for stratum in LENGTHS:
        group = [r for r in rows if r["stratum"] == stratum]
        strata[stratum] = {
            "planned": 6, "complete": len(group),
            **{k: sum(r[k] for r in group) for k in
               ("length_ok", "single_sentence", "target_match", "joint_success")},
            "mean_words": statistics.mean(r["words"] for r in group) if group else None,
            "mean_target_score": statistics.mean(
                r["target_score"] for r in group if r["target_score"] is not None)
                if any(r["target_score"] is not None for r in group) else None,
            "elapsed_seconds": sum(r["elapsed_seconds"] for r in group),
        }
    calls = [json.loads(p.read_text()) for p in (folder / "calls").glob("*.json")]
    complete = len(rows) == len(jobs)
    successes = [r["joint_success"] >= 4 for r in strata.values()]
    payload = {
        "status": "complete_preflight_not_asr" if complete else "incomplete_preflight",
        "strata": strata, "paragraph_pilot_gate_passed": all(successes) if complete else None,
        "api_calls": len(calls), "complete_api_calls": sum(c["status"] == "complete" for c in calls),
        "api_tokens": {k: sum(c.get("response", {}).get("usage", {}).get(k, 0) for c in calls)
                       for k in ("prompt_tokens", "completion_tokens")},
        "manifest_sha256": sha(folder / "manifest.json"),
        "result_sha256": {p.name: sha(p) for p in sorted((folder / "evaluated").glob("*.json"))},
        "limitations": manifest["protocol"]["no_asr"],
    }
    write_json(folder / "summary.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "summarize"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--reuse-preparation", type=Path)
    args = parser.parse_args()
    for k, v in {"NLTK_DATA": ".nltk_data", "HF_HOME": ".hf-cache",
                 "GLOVE_VECTORS_PATH": "vectors/glove.6B.100d.txt"}.items():
        os.environ.setdefault(k, str(ROOT / v))
    with claim(args.output / ".runner.lock"):
        if args.stage == "prepare":
            prepare(args.output, args.reuse_preparation)
            print(json.dumps({"status": "prepared", "planned_calls": 19}))
        elif args.stage == "run":
            key = (read_credential(args.key_file) if args.key_file
                   else os.environ.get("DEEPSEEK_API_KEY", "").strip())
            if not key:
                raise RuntimeError("API credential missing")
            if any(c.isspace() for c in key):
                raise ValueError("Credential must be a single token")
            try:
                run(args.output, key)
            finally:
                print(json.dumps(summarize(args.output), indent=2), flush=True)
        else:
            print(json.dumps(summarize(args.output), indent=2))


if __name__ == "__main__":
    main()
