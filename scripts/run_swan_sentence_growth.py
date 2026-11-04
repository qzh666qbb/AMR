"""New-document SWAN sentence-count feasibility with frozen nested prefixes.

This is a one-document development run, not a three-document ASR estimate.
The upstream injection routine is retained with isolated template RNG and
first-observation caches, so interrupted sentences can replay deterministically.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time

from feedback2_runtime import CachedDetector, Heartbeat, claim
from prepare_confirmation_experiment import attack_templates, choose_prompts, normalized_hash, stream_validation
from run_confirmation_experiment import (
    CallJournal, MODEL, digest, freeze, request_body, review_and_detect, run_attacks,
)
from run_confirmation_mechanism import lexical_tokens, punctuation_merge
from run_segmentation_intervention import ROOT, sha, text_sha, write_json
from supervise_confirmation_experiment import read_credential

OUTPUT = ROOT / "experiments/swan_sentence_growth1"
PREFIXES = {"short": 5, "medium": 20, "long": 40}
WORD_RANGES = {"short": [20, 80], "medium": [100, 220], "long": [200, 400]}
INJECTION_CAP = 1000
TOTAL_CAP = 1050
PRE_SCALAR_FIX = "2c09431ab4f57d8904a342b687d58e893c9336a9f1d1847a0bf03b1dab728e69"
BASELINE_REVIEW = (
    "Evaluate the following generated passage as standalone prose. Do not judge whether "
    "its factual claims are true. Rate fluency, coherence and nonredundancy separately "
    "from 1 (very poor) to 5 (very good). Coherence requires a connected, interpretable "
    "passage, not a sequence of disconnected claims. Nonredundancy penalizes repeated "
    "claims and empty filler. Mark severe_problem true for nonsensical or largely "
    "repetitive prose. Return only a JSON object with integer fields fluency, coherence, "
    "nonredundancy; boolean severe_problem; string reason.\n\nPASSAGE:\n"
)


class BudgetReached(RuntimeError):
    pass


def check(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    runner = str(Path(__file__).relative_to(ROOT))
    for path, expected in manifest["dependencies"].items():
        actual = sha(ROOT / path)
        if actual == expected:
            continue
        if path != runner or expected != PRE_SCALAR_FIX:
            raise ValueError(f"Frozen dependency changed: {path}")
        amendment = folder / "implementation_amendments/native_scalar_serialization.json"
        payload = {
            "original_runner_sha256": PRE_SCALAR_FIX, "runner_sha256": actual,
            "change": "Convert NumPy bool/float match outputs to native JSON bool/float before caching",
            "protocol": "No changes to prompts, seeds, thresholds, caps, cohort, or completed observations",
            "recovery": "Replay paid response and cached parse; first failed, unsaved S2Match is recomputed",
        }
        if not amendment.exists() and (
            any((folder / "injection").glob("*.json")) or any((folder / "baselines").glob("*.json"))
        ):
            raise ValueError("Scalar-fix amendment must precede completed injection")
        freeze(amendment, payload)
    for package, expected in manifest["packages"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"Frozen package changed: {package}")
    return manifest


class BoundedJournal(CallJournal):
    def __init__(self, folder, key, heartbeat):
        super().__init__(folder, key, TOTAL_CAP)
        self.heartbeat = heartbeat
        prior = list((folder / "calls").glob("*.json"))
        self.prior_total = len(prior)
        self.injection_calls = sum(p.name.startswith("inject-") for p in prior)

    def request(self, scope, body):
        path = self.folder / "calls" / f"{scope}-{self.counters.get(scope, 0):03d}.json"
        if not path.exists():
            if self.prior_total + self.sent >= TOTAL_CAP:
                raise BudgetReached("Frozen experiment-wide API cap reached")
            if scope.startswith("inject-"):
                if self.injection_calls >= INJECTION_CAP:
                    raise BudgetReached("Frozen injection cap reached")
                self.injection_calls += 1
        self.heartbeat.update(api_scope=scope, api_attempts=self.prior_total + self.sent,
                              injection_attempts=self.injection_calls)
        return super().request(scope, body)


def prepare(folder):
    if (folder / "manifest.json").exists():
        return check(folder)
    parents = [
        ROOT / "experiments/confirmation_three_arm30/preparation.json",
        ROOT / "experiments/swan_length_probe_v2/manifest.json",
        ROOT / "experiments/planE_e2_joint250/quality_aware_reviews.json",
        ROOT / "experiments/segmentation_intervention/manifest.json",
    ]
    excluded = []
    for path in parents[:2]:
        excluded.extend(r["text"] for r in json.loads(path.read_text())["prompts"])
    excluded.extend(r["source"] for r in json.loads(parents[2].read_text()))
    excluded.extend(r["text"] for r in json.loads(parents[3].read_text())["records"])
    stream = stream_validation()
    try:
        prompts, rejected, scanned = choose_prompts(stream, 1, excluded)
    finally:
        stream.close()
    old = json.loads(parents[1].read_text())
    dependencies = {name: sha(ROOT / name) for name in old["dependencies"]}
    for path in [Path(__file__), ROOT / "scripts/run_confirmation_mechanism.py",
                 ROOT / "scripts/evaluate_adaptive_retry.py", *parents]:
        dependencies[str(path.relative_to(ROOT))] = sha(path)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "frozen_before_generation",
        "prompts": prompts, "excluded_prior_prompt_count": 33,
        "scanned_rows": scanned, "rejected": rejected,
        "protocol": {
            "independent_documents": 1, "prefix_sentence_slots": PREFIXES, "target_words": WORD_RANGES,
            "sampling": "One new document; nested prefixes, NOT independent length strata",
            "injection": {
                "max_sentence_slots": 40, "max_trials_per_slot": 50,
                "max_templates": 10, "retries_per_template": 5, "max_new_tokens": 50,
                "target_threshold": .75, "temperature": .7, "top_p": .9,
                "seed": "sentence-growth-v1:prompt_id:slot",
                "upstream": "Existing amr_instruction_completion, unchanged natural-short-sentence prompt",
                "adaptations": [
                    "Separate random.Random per sentence; S2Match cannot alter template RNG",
                    "Persistent raw parser and target-match caches; first stochastic observation reused",
                    "Accept or last-candidate fallback retained; finish and actual sentence counts reported separately",
                ],
            },
            "api_caps": {"injection": INJECTION_CAP, "total": TOTAL_CAP,
                         "policy": "Cumulative across restarts; do not extend on budget stop"},
            "baseline_review_prompt": BASELINE_REVIEW,
            "baseline_gate": "All three ratings>=4, no severe_problem, target words, all returned slots nonempty, finish=stop and exactly one sentence each",
            "gate_no_detector_filter": True,
            "mechanism": "For every completed prefix, merge adjacent pairs with semicolons preserving lexical tokens",
            "mechanism_caveat": "Score stress test even if baseline quality fails; not naturalness or ASR evidence",
            "attack": {
                "k": 4, "templates": attack_templates(), "arms": list(attack_templates()),
                "execution": "Only prefixes passing frozen baseline feasibility gate; failed prefixes remain in allocation record",
                "primary": "First quality-valid candidate by index before scores; report per-prefix outcomes, no pooled ASR",
                "strong_baseline_status": "full_e2 is a historical prompt control, NOT official DIPPER/SIRA",
                "quality": "Same saved three-arm review and surface gates; no adaptive rescue",
            },
            "threshold": 2.33,
            "fpr": "No new calibration or human controls; no equal-FPR or defense claim",
            "stop_rule": "Stop after 40 allocated slots or cumulative cap; never add slots to reach word target",
            "reporting": "No population confidence intervals, significance tests, domain generalization or formal attack ranking from N=1",
        },
        "dependencies": dependencies, "packages": {
            name: importlib.metadata.version(name) for name in old["packages"]},
    }
    freeze(folder / "manifest.json", payload)
    return payload


def decode_baseline_review(content):
    content = content.strip()
    if content.startswith("```"):
        content = "\n".join(content.splitlines()[1:-1])
    row = json.loads(content)
    if not isinstance(row, dict):
        raise ValueError("Expected baseline review object")
    for key in ("fluency", "coherence", "nonredundancy"):
        if type(row.get(key)) is not int or row[key] not in range(1, 6):
            raise ValueError("Invalid baseline rating")
    if type(row.get("severe_problem")) is not bool or not isinstance(row.get("reason"), str):
        raise ValueError("Invalid baseline judgment")
    return row


def baseline_gate(review, words, stratum, sentences):
    return (
        all(review[k] >= 4 for k in ("fluency", "coherence", "nonredundancy"))
        and not review["severe_problem"]
        and WORD_RANGES[stratum][0] <= words <= WORD_RANGES[stratum][1]
        and all(s["text"] and s["finish_reason"] == "stop" and s["actual_sentences"] == 1 for s in sentences)
    )


@contextmanager
def injection_adapter(injection, detector, api, scope, seed):
    saved = {key: getattr(injection, key) for key in
             ("BedrockManager", "random", "parse_amr", "decide_s2match_match")}
    responses = []

    class LoggedManager:
        def __init__(self, **kwargs):
            pass

        def generate(self, user_text, system_text, max_tokens, temperature, top_p):
            response = api.request(scope, request_body(user_text, temperature, max_tokens, system_text, top_p))
            responses.append(response)
            return response["content"].strip()

    def parse(text):
        if len(detector.parser.tokenizer(text)["input_ids"]) > detector.parser.max_sent_len:
            raise ValueError("Refusing truncated injection AMR input")
        return detector.memo("parse", {"injection_input": text}, lambda: saved["parse_amr"](text))

    def match(amr, target, threshold):
        def compute():
            accepted, score = saved["decide_s2match_match"](amr, target, threshold)
            return [bool(accepted), float(score)]
        result = detector.memo(
            "score", {"injection_amr": amr, "target": target, "threshold": threshold},
            compute)
        if len(result) != 2 or not math.isfinite(result[1]) or not 0 <= result[1] <= 1:
            raise ValueError("Invalid injection match")
        return result

    injection.BedrockManager = LoggedManager
    injection.random = random.Random(seed)
    injection.parse_amr = parse
    injection.decide_s2match_match = match
    try:
        yield responses
    finally:
        for key, value in saved.items():
            setattr(injection, key, value)


def generate_sentence(folder, prompt, slot, previous, detector, api, runtime):
    import nltk
    from injection import injection_amr_utils as injection
    context = prompt["generation_prompt"] + ((" " + " ".join(r["text"] for r in previous)) if previous else "")
    path = folder / "injection" / f"{slot:03d}.json"
    identity = {"slot": slot, "context_sha256": text_sha(context), "runtime_sha256": digest(runtime)}
    if path.exists():
        result = json.loads(path.read_text())
        if any(result[k] != v for k, v in identity.items()):
            raise ValueError("Sentence checkpoint identity mismatch")
        return result
    started = time.monotonic()
    scope = f"inject-{slot:03d}"
    seed = int(text_sha(f"sentence-growth-v1:{prompt['prompt_id']}:{slot}")[:16], 16)
    with injection_adapter(injection, detector, api, scope, seed) as responses:
        text, attempts, template, score, accepted = injection.amr_instruction_completion(
            context=context, amr_bank=detector.bank, model_id=MODEL, region="us-east-1",
            max_new_tokens=50, temperature=.7, top_p=.9, max_trials=50,
            amr_examples_path=str(ROOT / "third_party/SWAN/amr_bank/artifacts/normalized_amr_examples.json"),
        )
    matching = [r for r in responses if r["content"].strip() == text]
    if attempts != len(responses) or (text and not matching):
        raise ValueError("Injection return differs from recorded requests")
    result = {
        **identity, "text": text, "text_sha256": text_sha(text), "attempts": attempts,
        "chosen_template": template, "s2match": score, "accepted": bool(accepted),
        "finish_reason": matching[-1]["finish_reason"] if matching else "empty",
        "actual_sentences": len(nltk.sent_tokenize(text)), "seed": seed,
        "elapsed_seconds": time.monotonic() - started,
    }
    freeze(path, result)
    print(json.dumps({"stage": "injection", "slot": slot, "words": len(text.split()),
                      "attempts": attempts, "accepted": bool(accepted)}), flush=True)
    return result


def pair_merge(sentences):
    pieces = [punctuation_merge(sentences[i:i + 2]) if len(sentences[i:i + 2]) == 2 else sentences[i]
              for i in range(0, len(sentences), 2)]
    merged = " ".join(pieces)
    if lexical_tokens(merged) != lexical_tokens(" ".join(sentences)):
        raise ValueError("Pair merging changed lexical tokens")
    return merged


def evaluate_prefix(folder, stratum, sentences, prompt, detector, api, runtime):
    import nltk
    pid = f"{prompt['prompt_id']}-{stratum}"
    path = folder / "baselines" / f"{pid}.json"
    identity = {"runtime_sha256": digest(runtime), "injection_sha256": digest(sentences)}
    text = " ".join(r["text"] for r in sentences).strip()
    if path.exists():
        baseline = json.loads(path.read_text())
        if any(baseline[k] != v for k, v in identity.items()) or baseline["text"] != text:
            raise ValueError("Prefix checkpoint mismatch")
    else:
        review = api.request(f"baseline-review-{stratum}", request_body(BASELINE_REVIEW + text, 0, 700))
        if review["finish_reason"] != "stop":
            raise ValueError("Truncated baseline review")
        judgment = decode_baseline_review(review["content"])
        words = len(text.split())
        gate = baseline_gate(judgment, words, stratum, sentences)
        details = detector.parse(text) if text else []
        baseline = {
            **identity, "prompt_id": pid, "stratum": stratum, "text": text, "text_sha256": text_sha(text),
            "sentences": sentences, "allocated_sentence_slots": len(sentences), "words": words,
            "quality_review": judgment, "feasibility_gate": gate,
            "all_injection_accepted": all(r["accepted"] for r in sentences),
            "unique_sentence_fraction": len({normalized_hash(s) for s in nltk.sent_tokenize(text)}) / max(1, len(details)),
            "detection": detector.score(details) if details else None,
        }
        freeze(path, baseline)
    mechanism_path = folder / "mechanism" / f"{pid}.json"
    if not mechanism_path.exists():
        original = [r["text"] for r in baseline["detection"]["details"]] if baseline["detection"] else []
        terminal = all(s.rstrip().endswith((".", "!", "?")) for s in original)
        if len(original) < 2 or not terminal:
            mechanism = {"status": "unavailable_terminal_punctuation", "baseline_sha256": digest(baseline)}
        else:
            merged = pair_merge(original)
            detection = detector.score(detector.parse(merged))
            n, g = baseline["detection"]["n"], baseline["detection"]["green"]
            mechanism = {
                "status": "complete", "baseline_sha256": digest(baseline),
                "text": merged, "text_sha256": text_sha(merged), "lexical_sequence_identical": True,
                "detection": detection, "expected_pair_count": (n + 1) // 2,
                "original_green_rate": g / n, "merged_green_rate": detection["green"] / detection["n"],
                "count_only_counterfactual_z": (g / n - .25) * math.sqrt(detection["n"] / (.25 * .75)),
                "z_ceiling": math.sqrt(3 * detection["n"]),
                "baseline_quality_passed": baseline["feasibility_gate"],
                "not_asr": True,
            }
        freeze(mechanism_path, mechanism)
    print(json.dumps({"stage": "prefix", "stratum": stratum, "words": baseline["words"],
                      "gate": baseline["feasibility_gate"],
                      "z": baseline["detection"]["z"] if baseline["detection"] else None}), flush=True)
    return baseline


def summarize(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    pid = manifest["prompts"][0]["prompt_id"]
    results = {}
    for stratum, slots in PREFIXES.items():
        path = folder / "baselines" / f"{pid}-{stratum}.json"
        if not path.exists():
            results[stratum] = {"status": "not_completed", "allocated_slots": slots}
            continue
        row = json.loads(path.read_text())
        result = {k: row[k] for k in ("words", "feasibility_gate", "quality_review", "unique_sentence_fraction")}
        result.update(status="baseline_complete", allocated_slots=slots,
                      accepted_slots=sum(s["accepted"] for s in row["sentences"]),
                      detection={k: row["detection"][k] for k in ("n", "green", "z")} if row["detection"] else None)
        mechanism_path = folder / "mechanism" / f"{pid}-{stratum}.json"
        if mechanism_path.exists():
            mechanism = json.loads(mechanism_path.read_text())
            result["pair_merge"] = {
                "status": mechanism["status"], "not_asr": True,
                **{k: mechanism[k] for k in ("original_green_rate", "merged_green_rate",
                                             "count_only_counterfactual_z", "z_ceiling") if k in mechanism},
                "detection": {k: mechanism["detection"][k] for k in ("n", "green", "z")}
                if "detection" in mechanism else None,
            }
        attacks = {}
        for arm in manifest["protocol"]["attack"]["arms"]:
            path = folder / "evaluated" / f"{pid}-{stratum}-{arm}.json"
            if path.exists():
                group = json.loads(path.read_text())
                attacks[arm] = {k: group[k] for k in
                                ("no_feedback_candidate", "no_feedback_success", "feedback_candidate", "feedback_success")}
                attacks[arm]["quality_valid_candidates"] = sum(r["quality_valid"] for r in group["rows"])
        result["attack_outcomes_not_population_asr"] = attacks
        result["attack_status"] = ("not_run_baseline_gate_failed" if not row["feasibility_gate"]
                                   else ("complete" if len(attacks) == 3 else "pending"))
        results[stratum] = result
    calls = [json.loads(p.read_text()) for p in (folder / "calls").glob("*.json")]
    complete = all(r["status"] == "baseline_complete" and r.get("attack_status") != "pending" for r in results.values())
    payload = {
        "status": "complete_development_run" if complete else "incomplete_development_run",
        "independent_documents": 1, "allocated_prefixes": 3, "prefixes": results,
        "completed_sentence_slots": len(list((folder / "injection").glob("*.json"))),
        "api": {
            "attempted_calls": len(calls), "complete_calls": sum(r["status"] == "complete" for r in calls),
            "injection_calls": sum(r["scope"].startswith("inject-") for r in calls),
            "usage": {k: sum(r.get("response", {}).get("usage", {}).get(k, 0) for r in calls)
                      for k in ("prompt_tokens", "completion_tokens")},
        },
        "no_population_asr": True, "no_new_fpr_calibration": True,
        "manifest_sha256": sha(folder / "manifest.json"),
        "note": "Nested prefix outcomes and punctuation stress tests are not independent ASR observations.",
    }
    write_json(folder / "summary.json", payload)
    return payload


def run(folder, key):
    manifest = check(folder)
    with Heartbeat(folder) as heartbeat:
        api = BoundedJournal(folder, key, heartbeat)
        probe = api.request("probe", request_body("Reply exactly OK.", 0, 16))
        if probe["finish_reason"] != "stop" or probe["content"].strip() != "OK":
            raise ValueError("Unexpected API smoke")
        api.model_returned = probe["model"]
        runtime = freeze(folder / "runtime.json", {
            "manifest_sha256": sha(folder / "manifest.json"), "model_returned": probe["model"], "probe_id": probe["id"]})
        detector = CachedDetector(folder, digest(runtime))
        prompt = manifest["prompts"][0]
        sentences = []
        for slot in range(40):
            heartbeat.update(stage="injection", slot=slot, cache_counts=detector.counts.copy())
            sentences.append(generate_sentence(folder, prompt, slot, sentences, detector, api, runtime))
            for stratum, count in PREFIXES.items():
                if len(sentences) == count:
                    heartbeat.update(stage="prefix", stratum=stratum)
                    evaluate_prefix(folder, stratum, sentences, prompt, detector, api, runtime)
                    summarize(folder)
        # All generation and source-feasibility decisions finish before attacks.
        for stratum in PREFIXES:
            baseline = json.loads((folder / "baselines" / f"{prompt['prompt_id']}-{stratum}.json").read_text())
            if baseline["feasibility_gate"]:
                heartbeat.update(stage="attacks", stratum=stratum)
                rows = run_attacks(folder, baseline, manifest["protocol"]["attack"]["templates"], api)
                review_and_detect(folder, baseline, rows, detector, api)
        heartbeat.update(stage="complete", new_api_calls=api.sent, cache_counts=detector.counts.copy())


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
            print(json.dumps({"status": "prepared", "prompt_ids": [r["prompt_id"] for r in manifest["prompts"]],
                              "prefixes": PREFIXES, "caps": manifest["protocol"]["api_caps"]}), flush=True)
        elif args.stage == "run":
            key = read_credential(args.key_file) if args.key_file else os.environ.get("DEEPSEEK_API_KEY", "").strip()
            if not key or any(c.isspace() for c in key):
                raise ValueError("Expected a single nonempty API credential")
            try:
                run(args.output, key)
            except BudgetReached as exc:
                write_json(args.output / "budget_stop.json", {"reason": str(exc), "status": "not_a_generation_failure"})
            finally:
                print(json.dumps(summarize(args.output), indent=2), flush=True)
        else:
            print(json.dumps(summarize(args.output), indent=2))


if __name__ == "__main__":
    main()
