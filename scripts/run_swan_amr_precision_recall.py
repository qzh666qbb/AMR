"""Decompose deterministic AMR composition S2Match into precision/recall."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import time

from feedback2_runtime import Heartbeat, claim
from run_confirmation_experiment import freeze
from run_segmentation_intervention import ROOT, sha, text_sha, write_json
from run_swan_known_positive_audit import load_utils
import run_swan_amr_composition_decomposition as composition


OUTPUT = ROOT / "experiments/swan_amr_precision_recall1"
SOURCE = ROOT / "experiments/swan_amr_composition_decomposition1"
VECTORS_PATH = ROOT / "vectors/glove.6B.100d.txt"
REPEATS = 5
EXPECTED_TARGETS = 108
EXPECTED_S2MATCH = EXPECTED_TARGETS * 3 * REPEATS
SIGN_ALPHA = 0.025


def configure_environment():
    os.environ.setdefault("GLOVE_VECTORS_PATH", str(VECTORS_PATH))


def dependency_paths():
    return [
        Path(__file__),
        ROOT / "scripts/audit_swan_amr_precision_recall.py",
        OUTPUT / "protocol.md",
        VECTORS_PATH,
        ROOT / "third_party/SWAN/utils/amr_utils.py",
        ROOT / "third_party/SWAN/utils/s2match_patch.py",
        ROOT
        / "third_party/SWAN/amr-metric-suite/py3-Smatch-and-S2match/smatch/s2match.py",
        SOURCE / "protocol.md",
        SOURCE / "manifest.json",
        SOURCE / "summary.json",
        SOURCE / "audit.json",
        SOURCE / "recovery_verification.json",
    ]


def dependencies():
    return {
        str(path.relative_to(ROOT)): sha(path)
        for path in dependency_paths()
    }


def package_versions():
    return {
        name: importlib.metadata.version(name)
        for name in ("penman", "numpy", "scipy")
    }


def target_inventory():
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    targets = []
    for source in manifest["sources"]:
        checkpoint = json.loads(
            composition.checkpoint_path(SOURCE, source).read_text()
        )
        loaded = composition.load_graphs(source)
        sentences = loaded[2]
        for target, sentence in zip(checkpoint["targets"], sentences):
            if not target["accepted"]:
                continue
            template = sentence["chosen_template"]
            targets.append(
                {
                    "prompt_id": source["prompt_id"],
                    "pair_index": source["pair_index"],
                    "slot": target["slot"],
                    "composition_checkpoint": str(
                        composition.checkpoint_path(
                            SOURCE,
                            source,
                        ).relative_to(ROOT)
                    ),
                    "composition_checkpoint_sha256": sha(
                        composition.checkpoint_path(SOURCE, source)
                    ),
                    "target_template": template,
                    "target_template_sha256": text_sha(template),
                    "original_amr_sha256": target["amr_sha256"],
                    "composition_amr_sha256": source[
                        "composed_amr_sha256"
                    ],
                    "reparse_amr_sha256": source["reparsed_amr_sha256"],
                }
            )
    if len(targets) != EXPECTED_TARGETS:
        raise ValueError("Accepted target inventory mismatch")
    return targets


def prepare(folder=OUTPUT):
    configure_environment()
    path = folder / "manifest.json"
    if path.exists():
        return check(folder)
    targets = target_inventory()
    manifest = {
        "status": "frozen_before_precision_recall_s2match",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "workflow": "swan_amr_composition_precision_recall",
            "targets": EXPECTED_TARGETS,
            "conditions": ["original", "composition", "reparse"],
            "repeats": REPEATS,
            "new_s2match_observations": EXPECTED_S2MATCH,
            "argument_order": "(condition_graph, target_template)",
            "primary_sign_test_alpha": SIGN_ALPHA,
            "new_api_calls": 0,
            "new_parser_calls": 0,
        },
        "targets": targets,
        "dependencies": dependencies(),
        "packages": package_versions(),
    }
    freeze(path, manifest)
    return manifest


def check(folder=OUTPUT):
    configure_environment()
    manifest = json.loads((folder / "manifest.json").read_text())
    if (
        manifest["protocol"]["workflow"]
        != "swan_amr_composition_precision_recall"
        or manifest["protocol"]["new_s2match_observations"]
        != EXPECTED_S2MATCH
    ):
        raise ValueError("Unexpected workflow")
    for relative, expected in manifest["dependencies"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Frozen dependency changed: {relative}")
    if package_versions() != manifest["packages"]:
        raise ValueError("Frozen package versions changed")
    if manifest["targets"] != target_inventory():
        raise ValueError("Target inventory changed")
    return manifest


def checkpoint_path(folder, target):
    return (
        folder
        / "scores"
        / (
            f"{target['prompt_id']}-{target['pair_index']}-"
            f"{target['slot']}.json"
        )
    )


def target_graphs(target):
    checkpoint = json.loads(
        (ROOT / target["composition_checkpoint"]).read_text()
    )
    source_manifest = json.loads((SOURCE / "manifest.json").read_text())
    source = next(
        row
        for row in source_manifest["sources"]
        if row["prompt_id"] == target["prompt_id"]
        and row["pair_index"] == target["pair_index"]
    )
    details = composition.load_graphs(source)[1]
    source_slot = source["source_slots"].index(target["slot"])
    graphs = {
        "original": details[source_slot]["amr"],
        "composition": checkpoint["composed_amr"],
        "reparse": checkpoint["reparsed_amr"],
    }
    if (
        text_sha(graphs["original"]) != target["original_amr_sha256"]
        or text_sha(graphs["composition"])
        != target["composition_amr_sha256"]
        or text_sha(graphs["reparse"]) != target["reparse_amr_sha256"]
    ):
        raise ValueError("Target graph identity mismatch")
    return graphs


def validate_metric(metric):
    if (
        len(metric) != 3
        or not all(
            isinstance(value, (int, float))
            and math.isfinite(value)
            and 0 <= value <= 1
            for value in metric
        )
    ):
        raise ValueError("Invalid precision/recall/F1 result")
    precision, recall, f1 = metric
    expected = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    if not math.isclose(f1, expected, abs_tol=1e-12):
        raise ValueError("F1 does not reconstruct")


def score_target(folder, target, utils, heartbeat, completed):
    path = checkpoint_path(folder, target)
    if path.exists():
        return json.loads(path.read_text()), 0
    graphs = target_graphs(target)
    heartbeat.update(
        stage="precision_recall",
        completed_targets=completed,
        total_targets=EXPECTED_TARGETS,
        prompt_id=target["prompt_id"],
        pair_index=target["pair_index"],
        slot=target["slot"],
    )
    started = time.monotonic()
    conditions = {}
    calls = 0
    for name, graph in graphs.items():
        observations = []
        for _ in range(REPEATS):
            values = tuple(
                float(value)
                for value in utils.compute_s2match_from_strings(
                    graph,
                    target["target_template"],
                    vectors_file=str(VECTORS_PATH),
                    similarity_function="cosine",
                    cutoff=0.5,
                    diffsense=0.5,
                    mwp="split",
                    do_not_mark_quotes=False,
                    verbose=False,
                )
            )
            validate_metric(values)
            observations.append(
                {
                    "precision": values[0],
                    "recall": values[1],
                    "f1": values[2],
                }
            )
            calls += 1
        conditions[name] = {
            "observations": observations,
            "median": {
                metric: statistics.median(
                    row[metric] for row in observations
                )
                for metric in ("precision", "recall", "f1")
            },
        }
    checkpoint = {
        **{
            key: target[key]
            for key in (
                "prompt_id",
                "pair_index",
                "slot",
                "target_template_sha256",
                "original_amr_sha256",
                "composition_amr_sha256",
                "reparse_amr_sha256",
            )
        },
        "manifest_sha256": sha(folder / "manifest.json"),
        "conditions": conditions,
        "new_s2match_observations": calls,
        "elapsed_seconds": time.monotonic() - started,
    }
    if calls != 15:
        raise ValueError("Per-target S2Match budget mismatch")
    freeze(path, checkpoint)
    return checkpoint, calls


def sign_test(differences):
    losses = sum(value < 0 for value in differences)
    gains = sum(value > 0 for value in differences)
    ties = len(differences) - losses - gains
    n = losses + gains
    probability = (
        sum(math.comb(n, value) for value in range(losses, n + 1))
        / 2**n
        if n
        else 1.0
    )
    median_difference = statistics.median(differences)
    return {
        "differences": differences,
        "median_difference": median_difference,
        "losses": losses,
        "ties": ties,
        "gains": gains,
        "non_ties": n,
        "one_sided_sign_p": probability,
        "confirmed_negative": bool(
            median_difference < 0 and probability <= SIGN_ALPHA
        ),
    }


def summarize(folder=OUTPUT):
    manifest = json.loads((folder / "manifest.json").read_text())
    checkpoints = [
        json.loads(checkpoint_path(folder, target).read_text())
        for target in manifest["targets"]
    ]
    documents = []
    for prompt_id in composition.EXPECTED_IDS:
        rows = [
            row for row in checkpoints if row["prompt_id"] == prompt_id
        ]
        if not rows:
            raise ValueError("Missing document targets")
        values = {}
        for condition in ("original", "composition", "reparse"):
            for metric in ("precision", "recall", "f1"):
                values[f"{condition}_{metric}"] = statistics.mean(
                    row["conditions"][condition]["median"][metric]
                    for row in rows
                )
        values.update(
            composition_precision_change=(
                values["composition_precision"]
                - values["original_precision"]
            ),
            composition_recall_change=(
                values["composition_recall"]
                - values["original_recall"]
            ),
            precision_minus_recall_change=(
                (
                    values["composition_precision"]
                    - values["original_precision"]
                )
                - (
                    values["composition_recall"]
                    - values["original_recall"]
                )
            ),
            reparse_precision_change=(
                values["reparse_precision"]
                - values["composition_precision"]
            ),
            reparse_recall_change=(
                values["reparse_recall"]
                - values["composition_recall"]
            ),
            reparse_f1_change=(
                values["reparse_f1"] - values["composition_f1"]
            ),
        )
        documents.append(
            {
                "prompt_id": prompt_id,
                "accepted_targets": len(rows),
                **values,
            }
        )
    precision_loss = sign_test([
        row["composition_precision_change"] for row in documents
    ])
    precision_specific = sign_test([
        row["precision_minus_recall_change"] for row in documents
    ])
    supported = bool(
        precision_loss["confirmed_negative"]
        and precision_specific["confirmed_negative"]
    )
    result = {
        "status": "complete_precision_recall_decomposition",
        "integrity": {
            "documents": len(documents),
            "targets": len(checkpoints),
            "new_s2match_observations": sum(
                row["new_s2match_observations"] for row in checkpoints
            ),
            "new_api_calls": 0,
            "new_parser_calls": 0,
        },
        "primary": {
            "composition_precision_loss": precision_loss,
            "precision_specific_dilution": precision_specific,
            "decision": (
                "precision_normalization_dilution_supported"
                if supported
                else "precision_recall_mechanism_inconclusive"
            ),
        },
        "condition_medians": {
            condition: {
                metric: statistics.median([
                    row["conditions"][condition]["median"][metric]
                    for row in checkpoints
                ])
                for metric in ("precision", "recall", "f1")
            }
            for condition in ("original", "composition", "reparse")
        },
        "descriptive_document_changes": {
            key: {
                "median": statistics.median([row[key] for row in documents]),
                "negative": sum(row[key] < 0 for row in documents),
                "zero": sum(row[key] == 0 for row in documents),
                "positive": sum(row[key] > 0 for row in documents),
            }
            for key in (
                "composition_recall_change",
                "reparse_precision_change",
                "reparse_recall_change",
                "reparse_f1_change",
            )
        },
        "documents": documents,
        "limitations": [
            "Precision and recall are S2Match alignment metrics.",
            "Deterministic composition is not a gold AMR annotation.",
            "The low-quality source cohort was previously observed.",
            "No recall equivalence or noninferiority claim is made.",
            "Not a natural-text ASR, TPR, or FPR estimate.",
        ],
        "manifest_sha256": sha(folder / "manifest.json"),
    }
    write_json(folder / "summary.json", result)
    write_report(folder, result)
    return result


def write_report(folder, result):
    medians = result["condition_medians"]
    primary = result["primary"]
    lines = [
        "# SWAN AMR-composition precision/recall decomposition",
        "",
        "| Condition | Precision | Recall | F1 |",
        "|---|---:|---:|---:|",
    ]
    for condition in ("original", "composition", "reparse"):
        values = medians[condition]
        lines.append(
            f"| {condition} | {values['precision']:.4f} | "
            f"{values['recall']:.4f} | {values['f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "- Composition precision change, median: "
            f"**{primary['composition_precision_loss']['median_difference']:.4f}** "
            f"(p={primary['composition_precision_loss']['one_sided_sign_p']:.6g}).",
            "- Precision change minus recall change, median: "
            f"**{primary['precision_specific_dilution']['median_difference']:.4f}** "
            f"(p={primary['precision_specific_dilution']['one_sided_sign_p']:.6g}).",
            f"- Decision: `{primary['decision']}`.",
            "- No API, parser, generation, injection, or attack call was run.",
            "",
        ]
    )
    (folder / "report.md").write_text("\n".join(lines))


def run(folder=OUTPUT):
    manifest = check(folder)
    utils = load_utils()
    new_calls = 0
    with Heartbeat(folder) as heartbeat:
        for index, target in enumerate(manifest["targets"]):
            scored = score_target(
                folder,
                target,
                utils,
                heartbeat,
                index,
            )
            calls = scored[1]
            new_calls += calls
            heartbeat.update(
                stage="precision_recall",
                completed_targets=index + 1,
                total_targets=EXPECTED_TARGETS,
                new_s2match_calls=new_calls,
            )
        heartbeat.update(
            stage="complete",
            completed_targets=EXPECTED_TARGETS,
            total_targets=EXPECTED_TARGETS,
            new_s2match_calls=new_calls,
        )
    result = summarize(folder)
    if (
        result["integrity"]["new_s2match_observations"]
        != EXPECTED_S2MATCH
    ):
        raise ValueError("Frozen S2Match budget mismatch")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "summarize"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    configure_environment()
    with claim(args.output / ".runner.lock"):
        if args.stage == "prepare":
            manifest = prepare(args.output)
            result = {
                "status": "prepared",
                "targets": len(manifest["targets"]),
                "manifest_sha256": sha(args.output / "manifest.json"),
            }
        elif args.stage == "run":
            result = run(args.output)
        else:
            result = summarize(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
