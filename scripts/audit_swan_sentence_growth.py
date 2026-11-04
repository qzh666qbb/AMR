"""Offline injection-trace replay and consistency audit for sentence-growth runs."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from audit_confirmation_experiment import verify_score
from evaluate_adaptive_retry import gates
from prepare_confirmation_experiment import normalized_hash
from run_confirmation_experiment import CallJournal, MODEL, decode_reviews, digest
from run_confirmation_mechanism import lexical_tokens
from run_segmentation_intervention import ROOT, sha, text_sha, write_json
from run_swan_sentence_growth import (
    BASELINE_REVIEW, INJECTION_CAP, OUTPUT, PREFIXES, TOTAL_CAP,
    baseline_gate, check, decode_baseline_review, injection_adapter, pair_merge,
)


def audit(folder):
    import nltk
    sys.path.insert(0, str(ROOT / "third_party/SWAN"))
    from injection import injection_amr_utils as injection
    from utils.amr_utils import load_amr_bank

    inputs = {}
    def load(path):
        inputs[str(path.relative_to(folder))] = sha(path)
        return json.loads(path.read_text())

    manifest = check(folder)
    runtime = load(folder / "runtime.json")
    load(folder / "manifest.json")
    summary = load(folder / "summary.json")
    if runtime["manifest_sha256"] != sha(folder / "manifest.json"):
        raise ValueError("Manifest/runtime mismatch")
    prompt = manifest["prompts"][0]
    prior_ids = set()
    for file in ("experiments/confirmation_three_arm30/preparation.json",
                 "experiments/swan_length_probe_v2/manifest.json"):
        prior_ids.update(r["prompt_id"] for r in json.loads((ROOT / file).read_text())["prompts"])
    if prompt["prompt_id"] in prior_ids or prompt["shard_row"] < 250:
        raise ValueError("Prior cohort overlap")
    calls = [load(p) for p in sorted((folder / "calls").glob("*.json"))]
    if len(calls) > TOTAL_CAP or sum(c["scope"].startswith("inject-") for c in calls) > INJECTION_CAP:
        raise ValueError("Cumulative API cap exceeded")
    for call in calls:
        if digest(call["request"]) != call["request_sha256"]:
            raise ValueError("Request checksum mismatch")
        if call["status"] == "complete" and call["response"]["model"] != runtime["model_returned"]:
            raise ValueError("Returned model drift")

    class ReplayJournal(CallJournal):
        def request(self, scope, body):
            path = folder / "calls" / f"{scope}-{self.counters.get(scope, 0):03d}.json"
            if not path.exists():
                raise ValueError("Replay would issue a new network request")
            return super().request(scope, body)

    class ReplayDetector:
        bank = load_amr_bank(str(ROOT / "third_party/SWAN/amr_bank/banks/amr_bank_50.json"))
        # No new tokenization or inference in an artifact-consistency audit.
        parser = SimpleNamespace(tokenizer=lambda text: {"input_ids": [0]}, max_sent_len=1024)

        def memo(self, kind, key, compute):
            path = folder / "sentence_cache" / digest(runtime) / kind / f"{digest(key)}.json"
            if not path.exists():
                raise ValueError("Replay would require new model computation")
            saved = load(path)
            if saved["key"] != key or saved["value_sha256"] != digest(saved["value"]):
                raise ValueError("Corrupted cache")
            return saved["value"]

    api = ReplayJournal(folder, "", 0)
    api.model_returned = runtime["model_returned"]
    detector = ReplayDetector()

    def verify_detection(scored, text):
        verify_score(scored)
        if not scored:
            if text:
                raise ValueError("Nonempty detection input lacks scores")
            return
        if nltk.sent_tokenize(text) != [d["text"] for d in scored["details"]]:
            raise ValueError("Detection segmentation mismatch")
        for detail in scored["details"]:
            parsed = detector.memo("parse", {"text": detail["text"]}, None)
            scores = detector.memo("score", {"amr": detail["amr"]}, None)
            if (parsed != [{k: detail[k] for k in ("text", "raw_amr", "amr")}]
                    or scores != detail["bank_scores"]):
                raise ValueError("Detection output differs from cache")

    sentences = []
    for path in sorted((folder / "injection").glob("*.json")):
        row = load(path)
        slot = len(sentences)
        context = prompt["generation_prompt"] + ((" " + " ".join(s["text"] for s in sentences)) if sentences else "")
        seed = int(text_sha(f"sentence-growth-v1:{prompt['prompt_id']}:{slot}")[:16], 16)
        if (row["slot"] != slot or row["context_sha256"] != text_sha(context)
                or row["seed"] != seed or row["runtime_sha256"] != digest(runtime)):
            raise ValueError("Injection slot provenance mismatch")
        with injection_adapter(injection, detector, api, f"inject-{slot:03d}", seed) as responses:
            text, attempts, template, score, accepted = injection.amr_instruction_completion(
                context=context, amr_bank=detector.bank, model_id=MODEL, region="us-east-1",
                max_new_tokens=50, temperature=.7, top_p=.9, max_trials=50,
                amr_examples_path=str(ROOT / "third_party/SWAN/amr_bank/artifacts/normalized_amr_examples.json"))
        actual = (text, attempts, template, score, bool(accepted))
        expected = tuple(row[k] for k in ("text", "attempts", "chosen_template", "s2match", "accepted"))
        if actual != expected or attempts != len(responses) or attempts > 50:
            raise ValueError("Frozen injection trace replay diverged")
        matches = [r for r in responses if r["content"].strip() == text]
        finish = matches[-1]["finish_reason"] if matches else "empty"
        if (row["text_sha256"] != text_sha(text) or finish != row["finish_reason"]
                or row["actual_sentences"] != len(nltk.sent_tokenize(text))):
            raise ValueError("Returned sentence metadata mismatch")
        if len([c for c in calls if c["scope"] == f"inject-{slot:03d}"]) != attempts:
            raise ValueError("Unaccounted injection calls")
        sentences.append(row)
    baselines_checked, mechanism_checked, groups_checked, expected_groups = 0, 0, 0, 0
    for stratum, count in PREFIXES.items():
        pid = f"{prompt['prompt_id']}-{stratum}"
        path = folder / "baselines" / f"{pid}.json"
        if not path.exists():
            continue
        baseline = load(path)
        source = " ".join(r["text"] for r in sentences[:count]).strip()
        if (baseline["sentences"] != sentences[:count] or len(baseline["sentences"]) != count
                or baseline["injection_sha256"] != digest(sentences[:count])
                or baseline["runtime_sha256"] != digest(runtime)
                or baseline["text"] != source or baseline["text_sha256"] != text_sha(source)):
            raise ValueError("Nested prefix provenance mismatch")
        review_call = load(folder / "calls" / f"baseline-review-{stratum}-000.json")
        if review_call["request"]["messages"] != [{"role": "user", "content": BASELINE_REVIEW + source}]:
            raise ValueError("Source-review prompt mismatch")
        review = decode_baseline_review(review_call["response"]["content"])
        gate = baseline_gate(review, len(source.split()), stratum, sentences[:count])
        if review != baseline["quality_review"] or gate != baseline["feasibility_gate"]:
            raise ValueError("Source feasibility gate mismatch")
        expected_groups += 3 if gate else 0
        verify_detection(baseline["detection"], source)
        details = baseline["detection"]["details"] if baseline["detection"] else []
        if nltk.sent_tokenize(source) != [r["text"] for r in details]:
            raise ValueError("Baseline segmentation mismatch")
        unique = len({normalized_hash(s) for s in nltk.sent_tokenize(source)}) / max(1, len(details))
        if unique != baseline["unique_sentence_fraction"] or len(source.split()) != baseline["words"]:
            raise ValueError("Baseline metrics mismatch")
        report = summary["prefixes"][stratum]
        if report["feasibility_gate"] != gate or report["words"] != baseline["words"]:
            raise ValueError("Summary baseline mismatch")
        baselines_checked += 1
        mpath = folder / "mechanism" / f"{pid}.json"
        if mpath.exists():
            mechanism = load(mpath)
            if mechanism["baseline_sha256"] != digest(baseline):
                raise ValueError("Mechanism source mismatch")
            if mechanism["status"] == "complete":
                expected_text = pair_merge([r["text"] for r in details])
                if (mechanism["text"] != expected_text or mechanism["text_sha256"] != text_sha(expected_text)
                        or lexical_tokens(expected_text) != lexical_tokens(source)):
                    raise ValueError("Punctuation control changed lexical content")
                verify_detection(mechanism["detection"], expected_text)
                n, g = baseline["detection"]["n"], baseline["detection"]["green"]
                m = mechanism["detection"]["n"]
                if nltk.sent_tokenize(expected_text) != [r["text"] for r in mechanism["detection"]["details"]]:
                    raise ValueError("Mechanism segmentation mismatch")
                counterfactual = (g / n - .25) * math.sqrt(m / (.25 * .75))
                if not math.isclose(mechanism["count_only_counterfactual_z"], counterfactual):
                    raise ValueError("Counterfactual mismatch")
                if report["pair_merge"]["detection"] != {k: mechanism["detection"][k] for k in ("n", "green", "z")}:
                    raise ValueError("Summary mechanism mismatch")
            mechanism_checked += 1
        for arm in manifest["protocol"]["attack"]["arms"]:
            path = folder / "evaluated" / f"{pid}-{arm}.json"
            if not path.exists():
                if not gate and list((folder / "attacks").glob(f"{pid}-{arm}-*.json")):
                    raise ValueError("Attack generated for baseline failing the gate")
                continue
            if not gate:
                raise ValueError("Evaluated attack with failed baseline gate")
            group = load(path)
            quality = load(folder / "quality" / f"{pid}-{arm}.json")
            if group["quality_sha256"] != digest(quality) or [r["candidate"] for r in group["rows"]] != list(range(4)):
                raise ValueError("Candidate budget or quality identity mismatch")
            review_call = load(folder / "calls" / f"review-{pid}-{arm}-000.json")
            judgments = decode_reviews(review_call["response"]["content"])
            generated = []
            for row, label in zip(group["rows"], quality["rows"]):
                candidate = load(folder / "attacks" / f"{pid}-{arm}-{row['candidate']}.json")
                generated.append(candidate)
                if (row["source"] != source or any(row[k] != v for k, v in candidate.items())
                        or {k: v for k, v in row.items() if k != "detection"} != label):
                    raise ValueError("Attack source or label changed")
                review = judgments[row["candidate"]]
                surface = gates(row, [d["amr"] for d in row["details"]])
                valid = all(surface.values()) and review["score"] >= 4 and not review["major_error"] and row["finish_reason"] == "stop"
                if review != row["review"] or surface != row["surface"] or valid != row["quality_valid"]:
                    raise ValueError("Attack quality reconstruction failed")
                verify_detection(row["detection"], row["attack"])
            if digest(generated) != quality["group_sha256"]:
                raise ValueError("Attack group checksum mismatch")
            eligible = [r for r in group["rows"] if r["quality_valid"] and r["detection"]]
            first = eligible[0] if eligible else None
            best = min(eligible, key=lambda r: (r["detection"]["z"], r["candidate"])) if eligible else None
            for name, selected in (("no_feedback", first), ("feedback", best)):
                if (group[f"{name}_candidate"] != (selected["candidate"] if selected else None)
                        or group[f"{name}_success"] != bool(selected and selected["detection"]["z"] <= 2.33)):
                    raise ValueError("Selection or outcome mismatch")
            if quality["no_feedback_candidate"] != group["no_feedback_candidate"]:
                raise ValueError("No-feedback selection changed after quality")
            groups_checked += 1
    if summary["api"]["attempted_calls"] != len(calls) or summary["completed_sentence_slots"] != len(sentences):
        raise ValueError("Summary cost/count mismatch")
    for key in ("prompt_tokens", "completion_tokens"):
        actual = sum(c.get("response", {}).get("usage", {}).get(key, 0) for c in calls)
        if actual != summary["api"]["usage"][key]:
            raise ValueError("Summary token usage mismatch")
    complete = (len(sentences) == 40 and baselines_checked == 3 and mechanism_checked == 3
                and groups_checked == expected_groups)
    if (summary["status"] == "complete_development_run") != complete:
        raise ValueError("Summary completion status mismatch")
    payload = {
        "status": "passed" if complete else "passed_completed_records_only",
        "injection_traces_replayed": len(sentences), "replayed_requests": sum(api.counters.values()),
        "baselines_checked": baselines_checked, "mechanisms_checked": mechanism_checked,
        "attack_groups_checked": groups_checked, "mismatches": 0, "new_api_calls": api.sent,
        "new_parser_calls": 0, "new_s2match_calls": 0, "input_sha256": inputs,
        "audit_script_sha256": sha(Path(__file__)),
        "limitation": "Replays frozen stochastic observations, not independent parser/matcher repeatability.",
    }
    write_json(folder / "audit.json", payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "input_sha256"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    for k, v in {"NLTK_DATA": ".nltk_data", "HF_HOME": ".hf-cache",
                 "GLOVE_VECTORS_PATH": "vectors/glove.6B.100d.txt"}.items():
        os.environ.setdefault(k, str(ROOT / v))
    audit(args.output)


if __name__ == "__main__":
    main()
