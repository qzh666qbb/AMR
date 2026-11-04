"""Compare official SWAN bank sizes on fixed human and microfact AMRs."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import time

from feedback2_runtime import Heartbeat, claim
from run_confirmation_experiment import freeze
from run_segmentation_intervention import ROOT, sha, text_sha, write_json
from run_swan_atomic_fact_preflight import (
    MATCHER_PATH,
    configure_environment,
    count_upper_bound,
    s2_counts,
)

OUTPUT = ROOT / "experiments/swan_bank_scale_pilot1"
SWAN = ROOT / "third_party/SWAN"
BANK_SIZES = (50, 100, 500, 800)
BANK_PATHS = {
    size: SWAN / f"amr_bank/banks/amr_bank_{size}.json"
    for size in BANK_SIZES
}
HUMAN_PATH = SWAN / "amr_bank/artifacts/human_parsed_amrs_250.json"
MICROFACT_DIR = ROOT / "experiments/swan_microfact_generation1/exploratory_parser"
HUMAN_DOCUMENTS = 50
SCORE_THRESHOLD = 0.65
NULL_RATE = 0.25
Z_THRESHOLD = 2.33
WORKERS = 4
_WORKER_BANKS = None
_WORKER_COUNTS = None
_WORKER_SCORE = None


def load_banks():
    banks = {
        size: [row["normalized_template_amr"] for row in json.loads(path.read_text())]
        for size, path in BANK_PATHS.items()
    }
    if any(len(banks[size]) != size for size in BANK_SIZES):
        raise ValueError("Official bank size mismatch")
    if banks[500][:100] != banks[100] or banks[800][:500] != banks[500]:
        raise ValueError("Expected bank100 <= bank500 <= bank800 prefix topology")
    if len(set(banks[50]) & set(banks[100])) != 38:
        raise ValueError("Unexpected bank50/bank100 overlap")
    return banks


def input_documents():
    human = json.loads(HUMAN_PATH.read_text())[:HUMAN_DOCUMENTS]
    if len(human) != HUMAN_DOCUMENTS or any(not document for document in human):
        raise ValueError("Incomplete official human pilot")
    microfact = []
    for path in sorted(MICROFACT_DIR.glob("*.json")):
        saved = json.loads(path.read_text())
        microfact.append({
            "document_id": saved["prompt_id"],
            "amrs": [row["template_amr"] for row in saved["facts"]],
            "source_path": str(path.relative_to(ROOT)),
            "source_sha256": sha(path),
        })
    if len(microfact) != 5 or any(len(row["amrs"]) != 15 for row in microfact):
        raise ValueError("Expected five 15-sentence microfact documents")
    return {
        "human": [
            {
                "document_id": f"human-{index:03d}",
                "amrs": document,
                "source_path": str(HUMAN_PATH.relative_to(ROOT)),
                "source_sha256": sha(HUMAN_PATH),
            }
            for index, document in enumerate(human)
        ],
        "microfact": microfact,
    }


def dependency_hashes():
    paths = [
        Path(__file__),
        OUTPUT / "protocol.md",
        HUMAN_PATH,
        ROOT / "experiments/swan_microfact_generation1/exploratory_scale_manifest.json",
        ROOT / "experiments/swan_microfact_generation1/exploratory_scale_summary.json",
        ROOT / "experiments/swan_microfact_generation1/exploratory_scale_audit.json",
        ROOT / "scripts/feedback2_runtime.py",
        ROOT / "scripts/run_swan_atomic_fact_preflight.py",
        ROOT / "scripts/run_swan_known_positive_audit.py",
        SWAN / "utils/amr_utils.py",
        SWAN / "utils/s2match_patch.py",
        MATCHER_PATH / "amr_py3.py",
        MATCHER_PATH / "s2match.py",
        ROOT / "vectors/glove.6B.100d.txt",
    ]
    paths += list(BANK_PATHS.values())
    paths += sorted(MICROFACT_DIR.glob("*.json"))
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def package_versions():
    return {
        name: importlib.metadata.version(name)
        for name in ("nltk", "numpy", "penman", "scipy", "spacy")
    }


def inventory(documents):
    return {
        cohort: [
            {
                "document_id": row["document_id"],
                "sentences": len(row["amrs"]),
                "amr_sha256": [text_sha(amr) for amr in row["amrs"]],
                "source_path": row["source_path"],
                "source_sha256": row["source_sha256"],
            }
            for row in rows
        ]
        for cohort, rows in documents.items()
    }


def prepare(folder):
    path = folder / "manifest.json"
    if path.exists():
        return check(folder)
    banks = load_banks()
    documents = input_documents()
    bank_inventory = {
        str(size): {
            "templates": len(banks[size]),
            "sha256": sha(BANK_PATHS[size]),
            "template_sha256": [text_sha(template) for template in banks[size]],
        }
        for size in BANK_SIZES
    }
    manifest = {
        "status": "frozen_before_s2match",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "workflow": "swan_bank_scale_null_pilot",
            "bank_sizes": list(BANK_SIZES),
            "human_documents": HUMAN_DOCUMENTS,
            "human_sentences": sum(len(row["amrs"]) for row in documents["human"]),
            "microfact_documents": len(documents["microfact"]),
            "microfact_sentences": sum(len(row["amrs"]) for row in documents["microfact"]),
            "score_threshold": SCORE_THRESHOLD,
            "null_rate": NULL_RATE,
            "z_threshold": Z_THRESHOLD,
            "workers": WORKERS,
            "algorithm": (
                "Count-bound prune, official file order, stop each bank at first >=0.65; "
                "reuse first stochastic score for templates shared across banks"
            ),
            "continuation_gate": {
                "microfact_count_coverage_improvement_over_bank50": 0.10,
                "human_sentence_green_rate_max": 0.30,
                "human_document_fixed_threshold_fpr_max": 0.05,
            },
            "limits": [
                "First 50 official human documents are a calibration pilot with 2% document-FPR resolution",
                "S2Match internally reseeds; each evaluated pair retains its first observation",
                "Microfacts failed their parent text-quality gate and are auxiliary only",
                "No generation API, neural parser, watermark injection or attack is run",
            ],
        },
        "bank_inventory": bank_inventory,
        "bank_topology": {
            "bank50_bank100_overlap": len(set(banks[50]) & set(banks[100])),
            "bank100_prefix_of_bank500": banks[500][:100] == banks[100],
            "bank500_prefix_of_bank800": banks[800][:500] == banks[500],
        },
        "input_inventory": inventory(documents),
        "dependencies": dependency_hashes(),
        "packages": package_versions(),
    }
    freeze(path, manifest)
    return manifest


def check(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["protocol"]["workflow"] != "swan_bank_scale_null_pilot":
        raise ValueError("Unexpected workflow")
    for relative, expected in manifest["dependencies"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Frozen dependency changed: {relative}")
    if package_versions() != manifest["packages"]:
        raise ValueError("Frozen package versions changed")
    banks = load_banks()
    expected_banks = {
        str(size): {
            "templates": len(banks[size]),
            "sha256": sha(BANK_PATHS[size]),
            "template_sha256": [text_sha(template) for template in banks[size]],
        }
        for size in BANK_SIZES
    }
    if manifest["bank_inventory"] != expected_banks:
        raise ValueError("Bank inventory changed")
    if manifest["input_inventory"] != inventory(input_documents()):
        raise ValueError("Input AMR inventory changed")
    return manifest


def worker_init(banks):
    global _WORKER_BANKS, _WORKER_COUNTS, _WORKER_SCORE
    configure_environment()
    from run_swan_known_positive_audit import load_utils

    utils = load_utils()
    _WORKER_BANKS = banks
    _WORKER_COUNTS = {
        text_sha(template): s2_counts(template)
        for templates in banks.values()
        for template in templates
    }
    _WORKER_SCORE = utils.compute_s2match_score


def score_candidate(amr, banks, template_counts, scorer):
    candidate_counts = s2_counts(amr)
    score_cache = {}
    observations = []
    by_bank = {}
    for size in BANK_SIZES:
        templates = banks[size]
        compatible = [
            index for index, template in enumerate(templates)
            if count_upper_bound(candidate_counts, template_counts[text_sha(template)])
            >= SCORE_THRESHOLD
        ]
        calls_before = len(score_cache)
        witness = None
        for template_index, template in enumerate(templates):
            template_id = text_sha(template)
            bound = count_upper_bound(candidate_counts, template_counts[template_id])
            if bound < SCORE_THRESHOLD:
                continue
            if template_id not in score_cache:
                score = float(scorer(amr, template))
                if not math.isfinite(score) or not -1e-9 <= score <= 1 + 1e-9:
                    raise ValueError("Invalid S2Match score")
                if score > bound + 1e-6:
                    raise ValueError("S2Match score exceeds count upper bound")
                score_cache[template_id] = score
                observations.append({
                    "template_sha256": template_id,
                    "score": score,
                    "count_upper_bound": bound,
                })
            if score_cache[template_id] >= SCORE_THRESHOLD:
                witness = {
                    "template_index": template_index,
                    "template_sha256": template_id,
                    "score": score_cache[template_id],
                }
                break
        by_bank[str(size)] = {
            "green": witness is not None,
            "witness": witness,
            "count_compatible_templates": len(compatible),
            "count_compatible_fraction": len(compatible) / size,
            "new_s2match_calls": len(score_cache) - calls_before,
        }
    return {
        "amr_sha256": text_sha(amr),
        "candidate_counts": candidate_counts,
        "candidate_triples": sum(candidate_counts),
        "banks": by_bank,
        "score_observations": observations,
        "unique_s2match_calls": len(score_cache),
    }


def score_document(task):
    cohort, document_id, amrs = task
    started = time.monotonic()
    rows = [
        score_candidate(amr, _WORKER_BANKS, _WORKER_COUNTS, _WORKER_SCORE)
        for amr in amrs
    ]
    return {
        "cohort": cohort,
        "document_id": document_id,
        "sentence_count": len(amrs),
        "amr_sha256": [text_sha(amr) for amr in amrs],
        "sentences": rows,
        "unique_s2match_calls": sum(row["unique_s2match_calls"] for row in rows),
        "elapsed_seconds": time.monotonic() - started,
    }


def checkpoint_path(folder, cohort, document_id):
    return folder / "scores" / f"{cohort}-{document_id}.json"


def z_score(green, total):
    denominator = math.sqrt(total * NULL_RATE * (1 - NULL_RATE))
    return (green - NULL_RATE * total) / denominator if denominator else None


def wilson_interval(successes, total, z=1.959963984540054):
    if total == 0:
        return [None, None]
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return [center - radius, center + radius]


def cohort_metrics(documents, size):
    sentence_rows = [sentence for document in documents for sentence in document["sentences"]]
    green = sum(row["banks"][str(size)]["green"] for row in sentence_rows)
    document_scores = []
    detected = 0
    for document in documents:
        count = sum(row["banks"][str(size)]["green"] for row in document["sentences"])
        value = z_score(count, len(document["sentences"]))
        document_scores.append({
            "document_id": document["document_id"],
            "sentences": len(document["sentences"]),
            "green": count,
            "z": value,
            "detected": value > Z_THRESHOLD,
        })
        detected += value > Z_THRESHOLD
    z_values = sorted(row["z"] for row in document_scores)
    return {
        "documents": len(documents),
        "sentences": len(sentence_rows),
        "green_sentences": green,
        "sentence_green_rate": green / len(sentence_rows),
        "sentence_green_rate_wilson95": wilson_interval(green, len(sentence_rows)),
        "detected_documents": detected,
        "fixed_threshold_document_rate": detected / len(documents),
        "document_z": {
            "mean": statistics.mean(z_values),
            "median": statistics.median(z_values),
            "min": min(z_values),
            "max": max(z_values),
            "empirical_99pct_threshold_same_sample": z_values[-1],
        },
        "mean_count_compatible_fraction": statistics.mean(
            row["banks"][str(size)]["count_compatible_fraction"]
            for row in sentence_rows
        ),
        "s2match_calls": sum(
            row["banks"][str(size)]["new_s2match_calls"] for row in sentence_rows
        ),
        "document_scores": document_scores,
    }


def summarize(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    documents = {"human": [], "microfact": []}
    for cohort, rows in manifest["input_inventory"].items():
        for source in rows:
            path = checkpoint_path(folder, cohort, source["document_id"])
            if path.exists():
                documents[cohort].append(json.loads(path.read_text()))
    complete = (
        len(documents["human"]) == HUMAN_DOCUMENTS
        and len(documents["microfact"]) == 5
    )
    metrics = {}
    if complete:
        for size in BANK_SIZES:
            metrics[str(size)] = {
                "human": cohort_metrics(documents["human"], size),
                "microfact": cohort_metrics(documents["microfact"], size),
            }
    candidates = []
    if complete:
        baseline = metrics["50"]["microfact"]["mean_count_compatible_fraction"]
        for size in BANK_SIZES[1:]:
            human = metrics[str(size)]["human"]
            microfact = metrics[str(size)]["microfact"]
            improvement = microfact["mean_count_compatible_fraction"] - baseline
            gate = {
                "coverage_improvement": improvement >= 0.10,
                "sentence_green_rate": human["sentence_green_rate"] <= 0.30,
                "document_fpr": human["fixed_threshold_document_rate"] <= 0.05,
            }
            candidates.append({
                "bank_size": size,
                "microfact_coverage_improvement_over_bank50": improvement,
                "gates": gate,
                "passes": all(gate.values()),
            })
    runtime_path = folder / "runtime.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    result = {
        "status": "complete" if complete else "incomplete",
        "cohort_completion": {
            "human": len(documents["human"]),
            "microfact": len(documents["microfact"]),
        },
        "bank_topology": manifest["bank_topology"],
        "metrics": metrics,
        "candidate_banks": candidates,
        "decision": (
            "prepare_independent_large_bank_calibration"
            if any(row["passes"] for row in candidates)
            else "do_not_expand_bank_under_fixed_null"
            if complete
            else "pending"
        ),
        "total_s2match_calls": sum(
            document["unique_s2match_calls"]
            for cohort in documents.values()
            for document in cohort
        ),
        "new_s2match_calls": runtime.get("new_s2match_calls", 0),
        "new_api_calls": 0,
        "new_neural_parser_calls": 0,
        "no_watermark_or_attack_asr": True,
        "manifest_sha256": sha(folder / "manifest.json"),
    }
    write_json(folder / "summary.json", result)
    if complete:
        write_report(folder, result)
    return result


def write_report(folder, result):
    lines = [
        "# SWAN 官方 bank 规模与零假设校准 pilot",
        "",
        "状态：完整完成。使用前 50 篇官方人类 AMR 和 5 篇失败质量门的 microfact 输出；"
        "没有生成、神经解析、水印注入或攻击。",
        "",
        "| Bank | 人类句级 green | Wilson 95% | 人类文档误报 | Microfact 计数兼容比例 | Microfact green |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for size in BANK_SIZES:
        human = result["metrics"][str(size)]["human"]
        micro = result["metrics"][str(size)]["microfact"]
        low, high = human["sentence_green_rate_wilson95"]
        lines.append(
            f"| {size} | {human['green_sentences']}/{human['sentences']} "
            f"({human['sentence_green_rate']:.1%}) | [{low:.1%}, {high:.1%}] | "
            f"{human['detected_documents']}/{human['documents']} "
            f"({human['fixed_threshold_document_rate']:.1%}) | "
            f"{micro['mean_count_compatible_fraction']:.1%} | "
            f"{micro['green_sentences']}/{micro['sentences']} "
            f"({micro['sentence_green_rate']:.1%}) |"
        )
    lines += [
        "",
        "大 bank 继续候选："
        + (", ".join(str(row["bank_size"]) for row in result["candidate_banks"] if row["passes"])
           or "**无**"),
        f"。决策：`{result['decision']}`。",
        "",
        f"实际首次 S2Match 观测：**{result['total_s2match_calls']}**；"
        "生成 API 与神经 parser 调用均为 **0**。",
        "",
        "固定 `lambda=0.25` 的误报率只在 50 篇 pilot 上估计，分辨率为 2%。"
        "同一批数据上的经验 99% 阈值只作描述，不能作为独立校准证明。",
        "",
    ]
    (folder / "report.md").write_text("\n".join(lines))


def run(folder):
    check(folder)
    banks = load_banks()
    documents = input_documents()
    tasks = []
    for cohort, rows in documents.items():
        for row in rows:
            path = checkpoint_path(folder, cohort, row["document_id"])
            if not path.exists():
                tasks.append((cohort, row["document_id"], row["amrs"]))
    if not tasks:
        return summarize(folder)
    started = time.monotonic()
    new_calls = 0
    completed = (
        HUMAN_DOCUMENTS + 5 - len(tasks)
    )
    with Heartbeat(folder) as heartbeat:
        heartbeat.update(stage="loading_s2match_workers", completed_documents=completed)
        with ProcessPoolExecutor(
            max_workers=WORKERS,
            initializer=worker_init,
            initargs=(banks,),
        ) as executor:
            futures = {
                executor.submit(score_document, task): task[:2]
                for task in tasks
            }
            for future in as_completed(futures):
                cohort, document_id = futures[future]
                row = future.result()
                if row["cohort"] != cohort or row["document_id"] != document_id:
                    raise ValueError("Worker result identity mismatch")
                freeze(checkpoint_path(folder, cohort, document_id), row)
                completed += 1
                new_calls += row["unique_s2match_calls"]
                heartbeat.update(
                    stage="s2match",
                    completed_documents=completed,
                    total_documents=HUMAN_DOCUMENTS + 5,
                    new_s2match_calls=new_calls,
                )
                print(json.dumps({
                    "stage": "s2match",
                    "completed_documents": completed,
                    "total_documents": HUMAN_DOCUMENTS + 5,
                    "cohort": cohort,
                    "document_id": document_id,
                    "new_s2match_calls": row["unique_s2match_calls"],
                }), flush=True)
        total_calls = sum(
            json.loads(path.read_text())["unique_s2match_calls"]
            for path in (folder / "scores").glob("*.json")
        )
        heartbeat.update(
            stage="complete",
            completed_documents=HUMAN_DOCUMENTS + 5,
            new_s2match_calls=new_calls,
            total_s2match_calls=total_calls,
        )
    freeze(folder / "runtime.json", {
        "elapsed_seconds": time.monotonic() - started,
        "workers": WORKERS,
        "new_s2match_calls": new_calls,
        "total_s2match_calls": total_calls,
        "new_api_calls": 0,
        "new_neural_parser_calls": 0,
    })
    return summarize(folder)


def verify_sentence(saved, amr, banks, count_maps):
    if saved["amr_sha256"] != text_sha(amr):
        raise ValueError("Candidate AMR identity mismatch")
    candidate_counts = s2_counts(amr)
    if saved["candidate_counts"] != candidate_counts:
        raise ValueError("Candidate count mismatch")
    observations = {
        row["template_sha256"]: row for row in saved["score_observations"]
    }
    if len(observations) != len(saved["score_observations"]):
        raise ValueError("Duplicate S2Match observation")
    for template_id, row in observations.items():
        if not math.isfinite(row["score"]) or not 0 <= row["score"] <= 1 + 1e-9:
            raise ValueError("Invalid saved S2Match score")
        expected_bound = count_upper_bound(candidate_counts, count_maps[template_id])
        if (
            abs(row["count_upper_bound"] - expected_bound) > 1e-12
            or row["score"] > expected_bound + 1e-6
        ):
            raise ValueError("Saved S2Match/count-bound mismatch")
    for size in BANK_SIZES:
        templates = banks[size]
        feasible = [
            text_sha(template) for template in templates
            if count_upper_bound(candidate_counts, count_maps[text_sha(template)])
            >= SCORE_THRESHOLD
        ]
        result = saved["banks"][str(size)]
        if (
            result["count_compatible_templates"] != len(feasible)
            or abs(result["count_compatible_fraction"] - len(feasible) / size) > 1e-12
        ):
            raise ValueError("Count-compatible bank summary mismatch")
        if result["green"]:
            witness = result["witness"]
            template = templates[witness["template_index"]]
            template_id = text_sha(template)
            if (
                template_id != witness["template_sha256"]
                or template_id not in observations
                or witness["score"] != observations[template_id]["score"]
                or witness["score"] < SCORE_THRESHOLD
            ):
                raise ValueError("Green witness mismatch")
        elif any(template_id not in observations for template_id in feasible):
            raise ValueError("Non-green bank omitted a feasible template")
    if saved["unique_s2match_calls"] != len(observations):
        raise ValueError("S2Match call count mismatch")


def audit(folder):
    check(folder)
    banks = load_banks()
    count_maps = {
        text_sha(template): s2_counts(template)
        for templates in banks.values()
        for template in templates
    }
    documents = input_documents()
    checked_documents = 0
    checked_sentences = 0
    checked_calls = 0
    for cohort, rows in documents.items():
        for source in rows:
            saved = json.loads(
                checkpoint_path(folder, cohort, source["document_id"]).read_text()
            )
            if (
                saved["cohort"] != cohort
                or saved["document_id"] != source["document_id"]
                or saved["sentence_count"] != len(source["amrs"])
                or saved["amr_sha256"] != [text_sha(amr) for amr in source["amrs"]]
                or len(saved["sentences"]) != len(source["amrs"])
            ):
                raise ValueError("Document checkpoint identity mismatch")
            for sentence, amr in zip(saved["sentences"], source["amrs"]):
                verify_sentence(sentence, amr, banks, count_maps)
                checked_sentences += 1
                checked_calls += sentence["unique_s2match_calls"]
            if saved["unique_s2match_calls"] != sum(
                row["unique_s2match_calls"] for row in saved["sentences"]
            ):
                raise ValueError("Document S2Match call count mismatch")
            checked_documents += 1
    summary = summarize(folder)
    if (
        summary["status"] != "complete"
        or summary["total_s2match_calls"] != checked_calls
        or checked_documents != HUMAN_DOCUMENTS + 5
    ):
        raise ValueError("Summary or inventory mismatch")
    result = {
        "status": "passed",
        "mismatches": 0,
        "documents_checked": checked_documents,
        "sentences_checked": checked_sentences,
        "saved_s2match_observations_checked": checked_calls,
        "new_api_calls": 0,
        "new_neural_parser_calls": 0,
        "new_s2match_calls": 0,
        "summary_sha256": sha(folder / "summary.json"),
        "limitation": "Audits saved first S2Match observations and exact early-stop coverage; stochastic optimization is not rerun.",
    }
    write_json(folder / "audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "summarize", "audit"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    configure_environment()
    with claim(args.output / ".runner.lock"):
        if args.stage == "prepare":
            result = prepare(args.output)
            output = {
                "status": "prepared",
                "human_documents": result["protocol"]["human_documents"],
                "human_sentences": result["protocol"]["human_sentences"],
                "microfact_sentences": result["protocol"]["microfact_sentences"],
                "manifest_sha256": sha(args.output / "manifest.json"),
            }
        elif args.stage == "run":
            output = run(args.output)
        elif args.stage == "audit":
            output = audit(args.output)
        else:
            output = summarize(args.output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
