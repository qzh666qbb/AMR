"""Recompute the report from saved evidence; no API, parser or detector calls.

Run from the project root:
  .venv/bin/python deliverables/swan_short_text_attack_analysis_20261103/reproduce.py
Only this deliverable directory is written. Raw text is never exported.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from scipy.stats import binomtest


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent.parent
ARMS = ("plain_paraphrase", "boundary_only", "full_e2")
MODES = ("original", "merged_sentence", "merged_rule_clause", "merged_exact_clause")
THRESHOLD = 2.33


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def digest(value):
    return text_sha(json.dumps(value, sort_keys=True))


def load(path):
    return json.loads(path.read_text())


def lexical(text):
    return re.findall(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)*", text)


def rate(k, n):
    ci = binomtest(k, n).proportion_ci(confidence_level=.95, method="exact")
    return {"count": k, "n": n, "rate": k / n, "clopper_pearson_95": [ci.low, ci.high]}


def verify_score(v):
    require(v is not None and v["details"], "Missing detection")
    for d in v["details"]:
        scores = d["bank_scores"]
        require(len(scores) == 50 and all(math.isfinite(s) and 0 <= s <= 1 for s in scores),
                "Invalid bank score vector")
        require(d["score"] == max(scores), "Invalid maximum bank score")
    n = len(v["details"])
    g = sum(d["score"] >= .65 for d in v["details"])
    z = (g - .25 * n) / math.sqrt(.1875 * n)
    require(v["n"] == n and v["green"] == g and math.isfinite(v["z"])
            and math.isclose(v["z"], z, rel_tol=0, abs_tol=1e-10), "Invalid z-score")


def summarize_rows(rows, threshold):
    detected = [r for r in rows if r["original"]["z"] > threshold]
    success = sum(r["merged_sentence"]["z"] <= threshold for r in detected)
    lost = success
    gained = sum(r["original"]["z"] <= threshold < r["merged_sentence"]["z"] for r in rows)
    return {
        "threshold": threshold,
        "escape_by_stage": {m: rate(sum(r[m]["z"] <= threshold for r in rows), len(rows))
                            for m in MODES},
        "conditional_new_evasion": rate(success, len(detected)),
        "new_evasion_over_full_cohort": rate(success, len(rows)),
        "detection_restored_among_new_evasions": sum(
            r["merged_sentence"]["z"] <= threshold < r["merged_exact_clause"]["z"]
            for r in detected),
        "discordant_pairs": {"detection_lost": lost, "detection_gained": gained},
        "mcnemar_exact_two_sided_p": binomtest(lost, lost + gained, .5).pvalue
        if lost + gained else 1.0,
    }


def analyze(root):
    mechanism = root / "experiments/confirmation_mechanism30"
    three = root / "experiments/confirmation_three_arm30"
    input_hashes = {}
    checks = []
    hash_cache = {}

    def record(path):
        value = sha(path)
        key = str(path.relative_to(root))
        require(key not in input_hashes or input_hashes[key] == value, "Input changed during audit")
        input_hashes[key] = value
        return value

    # Verify the historical manifests, including raw checkpoints, without exporting contents.
    for path in (mechanism / "summary.json", three / "final_analysis.json", three / "audit.json"):
        record(path)
        saved = load(path)
        require(saved["status"] in ("complete_verified", "complete_independently_verified"),
                "Historical run is not complete")
        for key, expected in saved["input_sha256"].items():
            p = (root / "scripts/analyze_confirmation_mechanism.py" if key == "analysis_script"
                 else root / key if key.startswith("experiments/") else path.parent / key)
            require(p.is_file(), f"Missing evidence: {p.relative_to(root)}")
            if p not in hash_cache:
                hash_cache[p] = record(p)
            require(hash_cache[p] == expected, f"SHA mismatch: {p.relative_to(root)}")
        checks.append({"manifest": str(path.relative_to(root)),
                       "entries_verified": len(saved["input_sha256"]), "mismatches": 0})

    manifest = load(mechanism / "manifest.json")
    aliases = {
        "confirmation_final_analysis": three / "final_analysis.json",
        "confirmation_runtime": three / "runtime.json",
        "segmentation_summary": root / "experiments/segmentation_intervention/summary.json",
        "segmentation_manifest": root / "experiments/segmentation_intervention/manifest.json",
        "prepare_script": root / "scripts/run_confirmation_mechanism.py",
    }
    for key, expected in manifest["dependencies"].items():
        p = aliases.get(key, root / key)
        require(record(p) == expected, f"Mechanism dependency changed: {key}")

    baselines = {}
    runtime = load(three / "runtime.json")
    for p in sorted((three / "baselines").glob("*.json")):
        b = load(p)
        require(b["text_sha256"] == text_sha(b["text"]), "Baseline text hash mismatch")
        require(b["runtime_sha256"] == digest(runtime), "Baseline runtime changed")
        require(len(b["sentences"]) == 5 and b["complete"], "Incomplete baseline")
        for slot, sentence in enumerate(b["sentences"]):
            require(sentence == load(three / "injection" / f'{b["prompt_id"]}-{slot}.json'),
                    "Injection checkpoint differs")
        require(b["all_injection_accepted"] == all(s["accepted"] for s in b["sentences"]),
                "Injection acceptance flag differs")
        verify_score(b["detection"])
        baselines[b["prompt_id"]] = b
    require(len(baselines) == 30, "Unexpected baseline count")

    expected = {r["id"]: r for r in manifest["records"]}
    seen = set()
    rows = []
    runtime_signatures = set()
    variant_count = 0
    for p in sorted((mechanism / "detected").glob("*.json")):
        d = load(p)
        r = d["record"]
        require(r["id"] not in seen and expected[r["id"]] == r, "Record differs from manifest")
        seen.add(r["id"])
        require(d["runtime"]["manifest_sha256"] == sha(mechanism / "manifest.json"),
                "Wrong mechanism runtime manifest")
        require(d["runtime_signature"] == digest(d["runtime"]), "Runtime signature differs")
        runtime_signatures.add(digest({k: v for k, v in d["runtime"].items()
                                      if k != "script_sha256"}))
        require(set(d["variants"]) == set(r["segments"]), "Variant set differs")
        for mode, v in d["variants"].items():
            verify_score(v)
            require([x["text"] for x in v["details"]] == r["segments"][mode], "Wrong segments")
            variant_count += 1
        if r["kind"] != "punctuation_control":
            continue
        b = baselines[r["prompt_id"]]
        original = r["segments"]["original"]
        require(original == [x["text"] for x in b["detection"]["details"]], "Wrong source units")
        require(r["baseline_text_sha256"] == b["text_sha256"], "Wrong source text")
        merged = "; ".join(re.sub(r'([.!?]+)(["\')\]]*)$', r"\2", s)
                           for s in original[:-1]) + "; " + original[-1]
        require(merged == r["text"] and text_sha(merged) == r["text_sha256"],
                "Punctuation intervention changed")
        tokens = lexical(b["text"])
        require(lexical(merged) == tokens and text_sha(json.dumps(tokens)) == r["lexical_sha256"],
                "Lexical sequence changed")
        for pieces in r["segments"].values():
            require(lexical(" ".join(pieces)) == tokens, "Resegmentation changed words")
        require(len(original) == len(r["segments"]["merged_exact_clause"]) == 5
                and len(r["segments"]["merged_sentence"]) == 1, "Not 5 -> 1 -> 5")
        for a, c in zip(original, r["segments"]["merged_exact_clause"]):
            require(lexical(a) == lexical(c), "Exact unit restoration failed")
        variants = {m: {k: d["variants"][m][k] for k in ("n", "green", "z")} for m in MODES}
        require(variants["original"] == r["historical_baseline"], "Historical baseline drift")
        require(variants["original"] == variants["merged_exact_clause"], "A-B-A not restored")
        rows.append({"case": f"case_{len(rows) + 1:02d}",
                     "baseline_words_whitespace": len(b["text"].split()),
                     "all_five_injection_accepted": b["all_injection_accepted"], **variants})
    require(seen == set(expected) and len(rows) == 30 and len(runtime_signatures) == 1,
            "Incomplete or mixed mechanism run")

    fixed = summarize_rows(rows, THRESHOLD)
    calibrated = summarize_rows(rows, manifest["protocol"]["calibrated_thresholds"]["sentence"])
    historical = load(mechanism / "summary.json")
    for name, current in (("fixed_2_33", fixed), ("calibrated", calibrated)):
        for m in MODES:
            require(current["escape_by_stage"][m]["count"]
                    == historical["punctuation_control"][name]["escape_rates"][m]["count"],
                    "Recomputed outcome differs")

    initially_detected = {pid for pid, b in baselines.items() if b["detection"]["z"] > THRESHOLD}
    arms = {}
    score_count = 0
    for arm in ARMS:
        successes = {role: set() for role in ("feedback", "no_feedback")}
        quality_count = 0
        lexical_same = {role: 0 for role in successes}
        for pid in sorted(baselines):
            group = load(three / "evaluated" / f"{pid}-{arm}.json")
            quality = load(three / "quality" / f"{pid}-{arm}.json")
            require(group["quality_sha256"] == digest(quality), "Quality checkpoint changed")
            require([x["candidate"] for x in group["rows"]] == list(range(4)), "Wrong K/order")
            generated = []
            for row, q in zip(group["rows"], quality["rows"]):
                a = load(three / "attacks" / f'{pid}-{arm}-{row["candidate"]}.json')
                generated.append(a)
                require(all(row[k] == v for k, v in a.items()), "Generated candidate differs")
                require({k: v for k, v in row.items() if k != "detection"} == q,
                        "Quality labels changed after scoring")
                require(row["source_sha256"] == baselines[pid]["text_sha256"]
                        and row["source"] == baselines[pid]["text"], "Wrong attack source")
                valid = (all(row["surface"].values()) and row["review"]["score"] >= 4
                         and not row["review"]["major_error"] and row["finish_reason"] == "stop")
                require(valid == row["quality_valid"], "Quality gate differs")
                quality_count += valid
                verify_score(row["detection"])
                score_count += 1
            require(digest(generated) == quality["group_sha256"], "Generated group hash differs")
            eligible = [r for r in group["rows"] if r["quality_valid"] and r["detection"]]
            first = eligible[0] if eligible else None
            best = min(eligible, key=lambda r: (r["detection"]["z"], r["candidate"])) if eligible else None
            for role, selected in (("feedback", best), ("no_feedback", first)):
                require(group[f"{role}_candidate"] == (selected["candidate"] if selected else None),
                        "Frozen candidate selection differs")
                success = bool(selected and selected["detection"]["z"] <= THRESHOLD)
                require(success == group[f"{role}_success"], "Success flag differs")
                if success:
                    successes[role].add(pid)
                if selected and lexical(selected["attack"]) == lexical(baselines[pid]["text"]):
                    lexical_same[role] += 1
            require(quality["no_feedback_candidate"] == group["no_feedback_candidate"],
                    "No-feedback selection changed")
        arms[arm] = {
            "quality_pass_candidates": quality_count, "candidate_count": 120,
            "selected_lexically_identical_to_source": lexical_same,
            **{role: {"historical_all_sources": rate(len(ids), 30),
                      "conditional_new_evasion": rate(len(ids & initially_detected), len(initially_detected))}
               for role, ids in successes.items()},
        }
    old_arms = load(three / "final_analysis.json")["arms"]
    for arm in ARMS:
        for role in ("feedback", "no_feedback"):
            require(arms[arm][role]["historical_all_sources"]["count"] == old_arms[arm][f"{role}_successes"],
                    "Three-arm reanalysis differs")

    # Supplementary historical cohort: do not pool with the confirmation cohort.
    milestone_path = root / "experiments/milestone_a/results.json"
    record(milestone_path)
    milestone = load(milestone_path)
    hrows = milestone["paragraph_metadata"]
    k4 = sum(r["quality_valid"] and r["escaped"] for r in hrows)
    require(len(hrows) == 250 and k4 == 191, "Historical K=4 count differs")
    calibration_path = root / "experiments/segmentation_intervention/summary.json"
    calibration = load(calibration_path)
    for rel in (
        "third_party/SWAN_official_2ca0c207_clean/README.md",
        "third_party/SWAN_official_2ca0c207_clean/injection/watermark_generation.py",
        "third_party/SWAN_official_2ca0c207_clean/detection/detect_from_parsed_amrs.py",
        "third_party/SWAN_official_2ca0c207_clean/detection/detection_utils.py",
        "scripts/run_confirmation_experiment.py",
        "scripts/run_segmentation_intervention.py",
        "scripts/prepare_confirmation_experiment.py",
        "experiments/swan_reframing1/BRANCH_C_SYSTEM_AUDIT.md",
    ):
        record(root / rel)

    words = [r["baseline_words_whitespace"] for r in rows]
    strict_rows = [r for r in rows if r["all_five_injection_accepted"]]
    result = {
        "analysis_type": "posthoc_read_only_reanalysis_of_frozen_runs",
        "new_api_parser_detector_calls": 0,
        "baseline": {
            "documents": 30, "sentences_per_document": 5,
            "words_whitespace": {"min": min(words), "median": statistics.median(words),
                                 "mean": statistics.mean(words), "max": max(words)},
            "total_words_whitespace": sum(words),
            "mean_words_per_original_sentence": sum(words) / (len(rows) * 5),
            "injection_max_tokens_per_sentence": runtime["injection"]["max_tokens"],
            "all_five_injection_accepted": len(strict_rows),
            "failed_injection_slots": sum(not s["accepted"] for b in baselines.values() for s in b["sentences"]),
            "model_requested": runtime["model_requested"], "model_returned": runtime["model_returned"],
        },
        "primary_punctuation_only": {
            "lexical_invariance_verified": 30, "finite_scores_verified": True,
            "original_equals_restored_n_green_z": 30,
            "fixed_2_33": fixed, "existing_calibrated_threshold": calibrated,
            "all_five_accepted_sensitivity": summarize_rows(strict_rows, THRESHOLD),
            "score_distributions": {m: dict(Counter(str(r[m]["z"]) for r in rows)) for m in MODES},
            "rows_numeric_only": rows,
        },
        "three_arms_same_30_sources": arms,
        "existing_human_test_reference": {
            mode: {k: calibration["calibrated"][mode][k]
                   for k in ("threshold", "empirical_calibration_fpr", "test_false_positives")}
            for mode in ("sentence", "clause")},
        "historical_k4_supplement": {
            "model_quality_valid_escape": rate(k4, 250),
            "original_sentence_count_4_to_6": sum(4 <= r["sentences"] <= 6 for r in hrows),
            "not_pooled_with_primary_cohort": True,
        },
    }
    audit = {
        "status": "PASS", "scope": "Saved hashes, score arithmetic, row pairing, edits and frozen selections",
        "historical_sha_checks": checks,
        "historical_sha_entries_verified": sum(c["entries_verified"] for c in checks),
        "historical_sha_unique_files": len(hash_cache),
        "mechanism_dependency_hashes_verified": len(manifest["dependencies"]),
        "mechanism_records_verified": len(seen),
        "mechanism_variants_verified": variant_count,
        "baseline_scores_verified": len(baselines),
        "three_arm_candidate_scores_verified": score_count,
        "new_inference_calls": 0,
        "limitations": [
            "No fresh AMR parsing/S2Match replay; saved outputs and arithmetic verified.",
            "Model quality labels are not independent human judgments.",
            "Confidence intervals condition on the saved cohort and fixed pipeline.",
        ],
    }
    return result, audit, input_hashes


def plot(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    bars = ax.bar(range(3), [29 / 30, 0, 29 / 30], color=["#22577a", "#c75d3a", "#22577a"], width=.55)
    ax.set_xticks(range(3), ["Original\n5 units", "Punctuation edit\n1 unit", "Exact resegmentation\n5 units"])
    ax.set_ylim(0, 1.17)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Detected documents / all 30 sources")
    ax.set_title("A. Detection disappears and returns", loc="left", fontweight="bold", pad=18)
    for bar, label in zip(bars, ("29/30", "0/30", "29/30")):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .035, label,
                ha="center", fontweight="bold")
    ax = axes[1]
    labels = ["Plain paraphrase", "Boundary prompt", "Broad syntax rewrite"]
    for j, (role, color, label) in enumerate((
        ("feedback", "#22577a", "Select with detector scores"),
        ("no_feedback", "#d59d37", "Select before detector scores"),
    )):
        stats = [result["three_arms_same_30_sources"][a][role]["conditional_new_evasion"] for a in ARMS]
        y = [v["rate"] for v in stats]
        x = [i + (j - .5) * .34 for i in range(3)]
        err = [[v["rate"] - v["clopper_pearson_95"][0] for v in stats],
               [v["clopper_pearson_95"][1] - v["rate"] for v in stats]]
        ax.bar(x, y, width=.31, color=color, label=label, yerr=err,
               error_kw={"capsize": 3, "linewidth": 1})
        for xi, v in zip(x, stats):
            ax.text(xi, v["clopper_pearson_95"][1] + .035,
                    f'{v["count"]}/29', ha="center", fontsize=10)
    ax.set_xticks(range(3), labels)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_ylim(0, 1.28)
    ax.set_yticks([0, .2, .4, .6, .8, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("New evasion / 29 originally detected sources")
    ax.set_title("B. Three attack prompts, K = 4", loc="left", fontweight="bold", pad=18)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.text(.07, .035, "Same 30 sources in both panels. Panel B uses model quality gates; error bars: exact 95% binomial CI.",
             fontsize=10, color="#555555")
    fig.subplots_adjust(left=.075, right=.98, top=.85, bottom=.2, wspace=.34)
    fig.savefig(HERE / "attack_results.png", dpi=190, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    result, audit, inputs = analyze(args.root.resolve())
    for name, value in (("metrics.json", result), ("AUDIT.json", audit),
                        ("INPUT_SHA256.json", inputs)):
        (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    plot(result)
    names = ("REPORT.md", "SOURCE_NOTES.md", "reproduce.py", "metrics.json",
             "AUDIT.json", "INPUT_SHA256.json", "attack_results.png")
    (HERE / "SHA256SUMS.txt").write_text("".join(f"{sha(HERE / n)}  {n}\n" for n in names if (HERE / n).exists()))
    print(json.dumps({"audit": audit, "primary": result["primary_punctuation_only"]["fixed_2_33"],
                      "accepted_only": result["primary_punctuation_only"]["all_five_accepted_sensitivity"],
                      "three_arms": result["three_arms_same_30_sources"]}, indent=2))


if __name__ == "__main__":
    main()
