"""Bounded, single-prompt generator comparison using pinned upstream functions."""

import argparse
import ast
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import random
import time
from types import ModuleType

from feedback2_runtime import CachedDetector, Heartbeat, claim
from prepare_confirmation_experiment import choose_prompts, normalized_hash, stream_validation
from run_confirmation_experiment import CallJournal, MODEL, digest, freeze, request_body
from run_segmentation_intervention import REVISION, ROOT, sha, text_sha, write_json
from run_swan_sentence_growth import BASELINE_REVIEW, decode_baseline_review, injection_adapter
from supervise_confirmation_experiment import read_credential

OUTPUT = ROOT / "experiments/swan_generator_comparison1"
PROVENANCE = ROOT / "experiments/swan_sentence_growth1/injection_provenance.json"
UPSTREAM_SHA = "a9465e8e8cb7e1e9560179bcd8395e9b3b12bca307b8192bcd64e7dee0726f4f"
FUNCTIONS = ("load_amr_examples", "build_system_prompt", "build_user_prompt",
             "amr_instruction_completion", "build_user_prompt_no_watermark")
ARMS = ("upstream", "local_adapted", "unwatermarked")
SLOTS = 5
TRIALS = 15
CAP = 159


def upstream_ast():
    evidence = json.loads(PROVENANCE.read_text())
    source = evidence["upstream_source"]
    if text_sha(source) != UPSTREAM_SHA or evidence["upstream_sha256"] != UPSTREAM_SHA:
        raise ValueError("Pinned upstream source checksum mismatch")
    nodes = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    return ast.Module(body=[nodes[name] for name in FUNCTIONS], type_ignores=[])


def load_upstream(bindings=None):
    """Execute only reviewed function definitions, not upstream import/HF code."""
    if bindings is None:
        from injection import injection_amr_utils as local
        bindings = {name: getattr(local, name) for name in (
            "parse_amr", "normalize_amr_variables", "generate_template_amr",
            "decide_s2match_match", "BedrockManager")}
    module = ModuleType("pinned_swan_upstream_injection")
    module.__dict__.update(json=json, random=random, logger=logging.getLogger(module.__name__), **bindings)
    exec(compile(upstream_ast(), "<pinned-swan-upstream-functions>", "exec"), module.__dict__)
    return module


class CappedJournal(CallJournal):
    def __init__(self, folder, key, heartbeat):
        super().__init__(folder, key, CAP)
        self.prior_count = len(list((folder / "calls").glob("*.json")))
        self.heartbeat = heartbeat

    def request(self, scope, body):
        path = self.folder / "calls" / f"{scope}-{self.counters.get(scope, 0):03d}.json"
        if not path.exists() and self.prior_count + self.sent >= CAP:
            raise RuntimeError("Frozen cumulative API cap reached")
        self.heartbeat.update(api_scope=scope, api_attempts=self.prior_count + self.sent)
        return super().request(scope, body)


def prepare(folder):
    if (folder / "manifest.json").exists():
        return check(folder)
    parents = [ROOT / f"experiments/{name}/{file}" for name, file in (
        ("confirmation_three_arm30", "preparation.json"), ("swan_length_probe_v2", "manifest.json"),
        ("swan_sentence_growth1", "manifest.json"))]
    historical = ROOT / "experiments/planE_e2_joint250/quality_aware_reviews.json"
    human = ROOT / "experiments/segmentation_intervention/manifest.json"
    excluded = [r["text"] for p in parents for r in json.loads(p.read_text())["prompts"]]
    excluded += [r["source"] for r in json.loads(historical.read_text())]
    excluded += [r["text"] for r in json.loads(human.read_text())["records"]]
    stream = stream_validation()
    try:
        prompts, rejected, scanned = choose_prompts(stream, 1, excluded)
    finally:
        stream.close()
    old = json.loads(parents[-1].read_text())
    paths = [ROOT / p for p in old["dependencies"]]
    paths += [*parents, historical, human, Path(__file__), PROVENANCE,
              ROOT / "third_party/SWAN/utils/text_utils.py"]
    schedule = [(slot, arm) for slot in range(SLOTS)
                for arm in ARMS[slot % 3:] + ARMS[:slot % 3]]
    manifest = {
        "status": "frozen_before_generation", "created_at": datetime.now(timezone.utc).isoformat(),
        "prompts": prompts, "scanned_rows": scanned, "rejected": rejected,
        "dataset_revision": REVISION, "upstream_source_sha256": UPSTREAM_SHA,
        "upstream_functions_ast_sha256": digest(ast.dump(upstream_ast())),
        "schedule": [{"slot": slot, "arm": arm} for slot, arm in schedule],
        "protocol": {
            "independent_prompts": 1, "arms": list(ARMS), "sentence_slots_per_arm": SLOTS,
            "api_cap": CAP, "model_requested": MODEL, "temperature": .7, "top_p": .9,
            "max_new_tokens": 120, "max_trials_per_watermarked_slot": TRIALS,
            "max_templates": 3, "retries_per_template": 5, "target_threshold": .75,
            "seed": "generator-comparison-v1:prompt_id:slot (same template seed across watermarked arms)",
            "upstream": "Exact pinned function AST, API path; no feedback correction or shortest-sentence constraint",
            "local": "Current local function, including its existing feedback and shortest-sentence constraint",
            "control": "Pinned no-watermark user prompt, common minimal one-sentence system prompt; one draw per slot",
            "context": "Each arm continues the same input prompt using only its own prior returned sentences",
            "cache": "Shared first observations under frozen runtime; separate template RNG per slot",
            "fallback": "Retain the engine's last-candidate fallback after 15 attempts; never drop failed slots",
            "quality": {"prompt": BASELINE_REVIEW, "pass": "All three ratings>=4 and no severe_problem"},
            "length": {"normal_range_words": [100, 220], "note": "Diagnostic only; never extend or trim to hit target"},
            "source_feasibility": "Quality pass AND 100..220 words AND five nonempty complete single-sentence slots",
            "feasibility_not_conditioned_on_detector": True,
            "injection_evidence": "Report acceptance and baseline detection separately, not quality-selected ASR",
            "no_attacks": True, "no_new_fpr_calibration": True,
            "limitations": [
                "N=1 paired development example, not generalization or a significance test",
                "Trials=15/templates=3 are supplied upstream parameters, not default 50/10",
                "DeepSeek API replaces the paper's generation model/backend",
                "Only injection functions are upstream; parser, normalization and matching helpers remain local",
                "Maximum token budget=120 is equal across current arms, unlike previous 50-token local run",
                "Not equal realized compute: report all rejection attempts and tokens",
                "Not a prompt-only causal ablation: the whole local feedback procedure also differs",
                "LLM source review is not independent human review or factual verification",
            ],
        },
        "dependencies": {str(p.relative_to(ROOT)): sha(p) for p in paths},
        "packages": {name: importlib.metadata.version(name) for name in old["packages"]},
    }
    freeze(folder / "manifest.json", manifest)
    return manifest


def check(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    for path, expected in manifest["dependencies"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Frozen dependency changed: {path}")
    for name, expected in manifest["packages"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"Frozen package changed: {name}")
    if digest(ast.dump(upstream_ast())) != manifest["upstream_functions_ast_sha256"]:
        raise ValueError("Upstream function extraction changed")
    return manifest


def context_for(prompt, previous):
    return prompt["generation_prompt"] + ((" " + " ".join(s["text"] for s in previous)) if previous else "")


def execute_slot(arm, slot, context, prompt_id, module, detector, api):
    """Shared by live generation and offline replay; no checkpoint writes here."""
    import nltk
    scope = f"generate-{arm}-{slot:03d}"
    seed = int(text_sha(f"generator-comparison-v1:{prompt_id}:{slot}")[:16], 16)
    events = []
    if arm == "unwatermarked":
        response = api.request(scope, request_body(
            module.build_user_prompt_no_watermark(context), .7, 120, module.build_system_prompt(), .9))
        responses = [response]
        text, attempts, template, score, accepted = response["content"].strip(), 1, None, None, None
    else:
        with injection_adapter(module, detector, api, scope, seed) as responses:
            cached_match = module.decide_s2match_match
            def record_match(amr, target, threshold):
                match, score = cached_match(amr, target, threshold)
                events.append({"amr_sha256": text_sha(amr), "template_sha256": text_sha(target),
                               "matched": bool(match), "score": float(score)})
                return match, score
            module.decide_s2match_match = record_match
            result = module.amr_instruction_completion(
                context=context, amr_bank=detector.bank, model_id=MODEL, region="us-east-1",
                max_new_tokens=120, temperature=.7, top_p=.9, max_trials=TRIALS,
                max_templates_to_try=3, max_retries_per_template=5, s2match_threshold=.75,
                amr_examples_path=str(ROOT / "third_party/SWAN/amr_bank/artifacts/normalized_amr_examples.json"))
        if arm == "upstream":
            text, attempts, template = result
            if not events or len(events) != attempts or events[-1]["template_sha256"] != text_sha(template):
                raise ValueError("Upstream match trace differs from returned result")
            score, accepted = events[-1]["score"], events[-1]["matched"]
        else:
            text, attempts, template, score, accepted = result
            score, accepted = float(score), bool(accepted)
    matching = [r for r in responses if r["content"].strip() == text]
    if len(responses) != attempts or (text and not matching):
        raise ValueError("Returned sentence differs from API trace")
    return {
        "arm": arm, "slot": slot, "seed": seed, "context_sha256": text_sha(context),
        "text": text, "text_sha256": text_sha(text), "words": len(text.split()),
        "attempts": attempts, "template": template, "target_score": score, "accepted": accepted,
        "finish_reason": matching[-1]["finish_reason"] if matching else "empty",
        "actual_sentences": len(nltk.sent_tokenize(text)), "match_events": events,
    }


def source_gates(review, sentences):
    quality = all(review[k] >= 4 for k in ("fluency", "coherence", "nonredundancy")) and not review["severe_problem"]
    words = len(" ".join(s["text"] for s in sentences).split())
    structure = len(sentences) == SLOTS and all(
        s["text"] and s["finish_reason"] == "stop" and s["actual_sentences"] == 1 for s in sentences)
    length_ok = 100 <= words <= 220
    return {"source_quality_pass": quality, "structure_pass": structure, "normal_length_pass": length_ok,
            "source_feasibility_pass": quality and structure and length_ok}


def summarize(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    rows = {}
    for arm in ARMS:
        path = folder / "evaluated" / f"{arm}.json"
        if not path.exists():
            rows[arm] = {"status": "not_evaluated", "completed_slots": len(list((folder / "sentences" / arm).glob("*.json")))}
            continue
        result = json.loads(path.read_text())
        rows[arm] = {
            "status": "complete", "completed_slots": len(result["sentences"]), "words": result["words"],
            "quality_review": result["quality_review"], "gates": result["gates"],
            "unique_sentence_fraction": result["unique_sentence_fraction"],
            "accepted_slots": sum(s["accepted"] is True for s in result["sentences"]) if arm != "unwatermarked" else None,
            "generation_calls": sum(s["attempts"] for s in result["sentences"]),
            "finish_reasons": [s["finish_reason"] for s in result["sentences"]],
            "detection": {k: result["detection"][k] for k in ("n", "green", "z")} if result["detection"] else None,
        }
    calls = [json.loads(p.read_text()) for p in (folder / "calls").glob("*.json")]
    payload = {
        "status": "complete_development_comparison" if all(r["status"] == "complete" for r in rows.values()) else "incomplete",
        "independent_prompts": 1, "arms": rows, "api_attempts": len(calls),
        "api_complete": sum(c["status"] == "complete" for c in calls),
        "api_usage": {k: sum(c.get("response", {}).get("usage", {}).get(k, 0) for c in calls)
                      for k in ("prompt_tokens", "completion_tokens")},
        "manifest_sha256": sha(folder / "manifest.json"), "no_asr": True, "no_new_fpr": True,
        "limitations": manifest["protocol"]["limitations"],
    }
    write_json(folder / "summary.json", payload)
    return payload


def run(folder, key):
    manifest = check(folder)
    with Heartbeat(folder) as heartbeat:
        api = CappedJournal(folder, key, heartbeat)
        probe = api.request("probe", request_body("Reply exactly OK.", 0, 16))
        if probe["content"].strip() != "OK" or probe["finish_reason"] != "stop":
            raise ValueError("Unexpected connectivity response")
        api.model_returned = probe["model"]
        runtime = freeze(folder / "runtime.json", {
            "manifest_sha256": sha(folder / "manifest.json"), "model_returned": probe["model"], "probe_id": probe["id"]})
        heartbeat.update(stage="loading_detector")
        detector = CachedDetector(folder, digest(runtime))
        from injection import injection_amr_utils as local
        upstream = load_upstream()
        modules = {"upstream": upstream, "local_adapted": local, "unwatermarked": upstream}
        prompt = manifest["prompts"][0]
        previous = {arm: [] for arm in ARMS}
        for job in manifest["schedule"]:
            arm, slot = job["arm"], job["slot"]
            context = context_for(prompt, previous[arm])
            heartbeat.update(stage="generation", arm=arm, slot=slot, completed_slots=sum(map(len, previous.values())))
            path = folder / "sentences" / arm / f"{slot:03d}.json"
            if path.exists():
                row = json.loads(path.read_text())
                if row["context_sha256"] != text_sha(context) or row["runtime_sha256"] != digest(runtime):
                    raise ValueError("Generation checkpoint identity mismatch")
            else:
                started = time.monotonic()
                row = execute_slot(arm, slot, context, prompt["prompt_id"], modules[arm], detector, api)
                row.update(runtime_sha256=digest(runtime), elapsed_seconds=time.monotonic() - started)
                freeze(path, row)
            previous[arm].append(row)
            summarize(folder)
            print(json.dumps({k: row[k] for k in ("arm", "slot", "words", "attempts", "accepted")}), flush=True)
        # Freeze every source judgment before scoring any of the three passages.
        for arm in ARMS:
            text = " ".join(s["text"] for s in previous[arm]).strip()
            path = folder / "quality" / f"{arm}.json"
            identity = {"text_sha256": text_sha(text), "runtime_sha256": digest(runtime)}
            if path.exists():
                saved = json.loads(path.read_text())
                if any(saved[k] != v for k, v in identity.items()):
                    raise ValueError("Quality checkpoint mismatch")
                continue
            heartbeat.update(stage="quality", arm=arm)
            response = api.request(f"review-{arm}", request_body(BASELINE_REVIEW + text, 0, 700))
            if response["finish_reason"] != "stop":
                raise ValueError("Truncated source review")
            review = decode_baseline_review(response["content"])
            freeze(path, {**identity, "review": review, "gates": source_gates(review, previous[arm])})
        for arm in ARMS:
            path = folder / "evaluated" / f"{arm}.json"
            if path.exists():
                continue
            heartbeat.update(stage="detection", arm=arm)
            text = " ".join(s["text"] for s in previous[arm]).strip()
            quality = json.loads((folder / "quality" / f"{arm}.json").read_text())
            details = detector.parse(text) if text else []
            freeze(path, {
                "arm": arm, "runtime_sha256": digest(runtime), "sentences": previous[arm],
                "text": text, "text_sha256": text_sha(text), "words": len(text.split()),
                "quality_sha256": digest(quality), "quality_review": quality["review"], "gates": quality["gates"],
                "unique_sentence_fraction": len({normalized_hash(d["text"]) for d in details}) / max(1, len(details)),
                "detection": detector.score(details) if details else None,
            })
            summarize(folder)
        heartbeat.update(stage="complete", new_api_calls=api.sent, completed_slots=15, cache_counts=detector.counts.copy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "summarize"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    for k, v in {"NLTK_DATA": ".nltk_data", "HF_HOME": ".hf-cache",
                 "GLOVE_VECTORS_PATH": "vectors/glove.6B.100d.txt"}.items():
        os.environ.setdefault(k, str(ROOT / v))
    with claim(args.output / ".runner.lock"):
        if args.stage == "prepare":
            manifest = prepare(args.output)
            print(json.dumps({"status": "prepared", "prompts": [r["prompt_id"] for r in manifest["prompts"]],
                              "api_cap": CAP, "sentence_slots": len(manifest["schedule"])}))
        elif args.stage == "run":
            key = read_credential(args.key_file) if args.key_file else os.environ.get("DEEPSEEK_API_KEY", "")
            if not key or any(c.isspace() for c in key):
                raise ValueError("Expected single nonempty API credential")
            try:
                run(args.output, key)
            finally:
                print(json.dumps(summarize(args.output), indent=2), flush=True)
        else:
            print(json.dumps(summarize(args.output), indent=2))


if __name__ == "__main__":
    main()
