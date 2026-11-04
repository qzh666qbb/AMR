"""Decompose SWAN pair losses with deterministic AMR composition."""

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
from run_segmentation_intervention import amr_body, ROOT, sha, text_sha, write_json
from run_swan_known_positive_audit import load_utils


OUTPUT = ROOT / "experiments/swan_amr_composition_decomposition1"
SOURCE = ROOT / "experiments/confirmation_three_arm30"
ORACLE = ROOT / "experiments/swan_oracle_composition1"
BANK_PATH = ROOT / "third_party/SWAN/amr_bank/banks/amr_bank_50.json"
VECTORS_PATH = ROOT / "vectors/glove.6B.100d.txt"
EXPECTED_IDS = [f"c4-validation-{index}" for index in range(250, 280)]
REPEATS = 5
GREEN_THRESHOLD = 0.65
SIGN_ALPHA = 0.025
EXPECTED_S2MATCH = 5100


def configure_environment():
    os.environ.setdefault("GLOVE_VECTORS_PATH", str(VECTORS_PATH))


def rename_graph(graph, prefix):
    variables = set(graph.variables())
    mapping = {
        variable: f"{prefix}{index}"
        for index, variable in enumerate(sorted(variables))
    }
    triples = [
        (
            mapping[source],
            role,
            mapping[target] if target in variables else target,
        )
        for source, role, target in graph.triples
    ]
    return triples, mapping[graph.top]


def compose_raw_amrs(first_raw, second_raw):
    import penman

    first = penman.decode(amr_body(first_raw))
    second = penman.decode(amr_body(second_raw))
    first_triples, first_root = rename_graph(first, "a")
    second_triples, second_root = rename_graph(second, "b")
    graph = penman.Graph(
        [
            ("m", ":instance", "multi-sentence"),
            ("m", ":snt1", first_root),
            ("m", ":snt2", second_root),
            *first_triples,
            *second_triples,
        ],
        top="m",
    )
    return penman.encode(graph, indent=4)


def transform_amr(raw_amr, utils):
    import penman

    graph = amr_body(raw_amr)
    penman.decode(graph)
    graph = utils.normalize_amr_variables(graph)
    graph = utils.generate_template_amr(graph)
    graph = utils.normalize_amr_variables(graph)
    penman.decode(graph)
    return graph


def graph_triples(amr):
    import penman

    return len(penman.decode(amr).triples)


def dependency_paths():
    return [
        Path(__file__),
        ROOT / "scripts/audit_swan_amr_composition_decomposition.py",
        OUTPUT / "protocol.md",
        BANK_PATH,
        VECTORS_PATH,
        ROOT / "third_party/SWAN/utils/amr_utils.py",
        ROOT / "third_party/SWAN/utils/s2match_patch.py",
        ROOT
        / "third_party/SWAN/amr-metric-suite/py3-Smatch-and-S2match/smatch/s2match.py",
        ORACLE / "manifest.json",
        ORACLE / "summary.json",
        ORACLE / "audit.json",
        ORACLE / "recovery_verification.json",
        SOURCE / "final_analysis.json",
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


def source_inventory(utils):
    bank_rows = json.loads(BANK_PATH.read_text())
    bank = [row["normalized_template_amr"] for row in bank_rows]
    if len(bank) != 50 or len(set(bank)) != 50:
        raise ValueError("Expected 50 unique bank templates")
    template_indices = {template: index for index, template in enumerate(bank)}
    oracle_manifest = json.loads((ORACLE / "manifest.json").read_text())
    sources = []
    for oracle_source in oracle_manifest["sources"]:
        prompt_id = oracle_source["prompt_id"]
        if prompt_id not in EXPECTED_IDS:
            continue
        baseline_path = SOURCE / "baselines" / f"{prompt_id}.json"
        baseline = json.loads(baseline_path.read_text())
        for unit in oracle_source["pair_units"]:
            pair_index = unit["pair_index"]
            slots = unit["source_slots"]
            parser_path = (
                ORACLE / "pair_parser" / f"{prompt_id}-{pair_index}.json"
            )
            score_path = (
                ORACLE / "pair_scores" / f"{prompt_id}-{pair_index}.json"
            )
            parsed = json.loads(parser_path.read_text())
            scored = json.loads(score_path.read_text())
            components = []
            raw_amrs = []
            for slot in slots:
                sentence = baseline["sentences"][slot]
                detail = baseline["detection"]["details"][slot]
                template = sentence["chosen_template"]
                template_index = template_indices.get(template)
                if (
                    template_index is None
                    or not math.isclose(
                        sentence["s2match"],
                        detail["bank_scores"][template_index],
                        abs_tol=1e-12,
                    )
                ):
                    raise ValueError("Target template provenance mismatch")
                components.append(
                    {
                        "slot": slot,
                        "text_sha256": text_sha(sentence["text"]),
                        "raw_amr_sha256": text_sha(detail["raw_amr"]),
                        "amr_sha256": text_sha(detail["amr"]),
                        "chosen_template_sha256": text_sha(template),
                        "chosen_template_index": template_index,
                        "accepted": sentence["accepted"],
                        "saved_target_score": sentence["s2match"],
                        "saved_bank_max": detail["score"],
                    }
                )
                raw_amrs.append(detail["raw_amr"])
            composed_raw = compose_raw_amrs(*raw_amrs)
            composed_amr = transform_amr(composed_raw, utils)
            if (
                parsed["source_slots"] != slots
                or scored["parser_sha256"] != sha(parser_path)
                or scored["detail"]["amr"] != parsed["detail"]["amr"]
            ):
                raise ValueError("Reparse provenance mismatch")
            sources.append(
                {
                    "prompt_id": prompt_id,
                    "pair_index": pair_index,
                    "source_slots": slots,
                    "baseline_path": str(baseline_path.relative_to(ROOT)),
                    "baseline_sha256": sha(baseline_path),
                    "pair_parser_path": str(parser_path.relative_to(ROOT)),
                    "pair_parser_sha256": sha(parser_path),
                    "pair_score_path": str(score_path.relative_to(ROOT)),
                    "pair_score_sha256": sha(score_path),
                    "components": components,
                    "composed_raw_sha256": text_sha(composed_raw),
                    "composed_amr_sha256": text_sha(composed_amr),
                    "reparsed_amr_sha256": text_sha(parsed["detail"]["amr"]),
                }
            )
    if (
        len(sources) != 60
        or [row["prompt_id"] for row in sources[::2]] != EXPECTED_IDS
        or sum(
            component["accepted"]
            for source in sources
            for component in source["components"]
        )
        != 108
    ):
        raise ValueError("Frozen pair cohort mismatch")
    return bank, sources


def prepare(folder=OUTPUT):
    configure_environment()
    path = folder / "manifest.json"
    if path.exists():
        return check(folder)
    utils = load_utils()
    bank, sources = source_inventory(utils)
    manifest = {
        "status": "frozen_before_new_s2match",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "workflow": "swan_deterministic_amr_composition_decomposition",
            "documents": 30,
            "pair_units": 60,
            "targets": 120,
            "accepted_targets": 108,
            "composition": (
                "multi-sentence root with :snt1/:snt2 over disjoint raw "
                "constituent AMRs, then detector transformation"
            ),
            "target_score_repeats": REPEATS,
            "graph_similarity_repeats": REPEATS,
            "composition_bank_observations": 3000,
            "total_new_s2match_observations": EXPECTED_S2MATCH,
            "green_threshold": GREEN_THRESHOLD,
            "primary_sign_test_alpha": SIGN_ALPHA,
            "new_api_calls": 0,
            "new_parser_calls": 0,
        },
        "bank_template_sha256": [text_sha(template) for template in bank],
        "sources": sources,
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
        != "swan_deterministic_amr_composition_decomposition"
        or manifest["protocol"]["total_new_s2match_observations"]
        != EXPECTED_S2MATCH
    ):
        raise ValueError("Unexpected workflow")
    for relative, expected in manifest["dependencies"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Frozen dependency changed: {relative}")
    if package_versions() != manifest["packages"]:
        raise ValueError("Frozen package versions changed")
    utils = load_utils()
    bank, sources = source_inventory(utils)
    if (
        manifest["bank_template_sha256"]
        != [text_sha(template) for template in bank]
        or manifest["sources"] != sources
    ):
        raise ValueError("Frozen source inventory changed")
    return manifest


def checkpoint_path(folder, source):
    return (
        folder
        / "scores"
        / f"{source['prompt_id']}-{source['pair_index']}.json"
    )


def load_graphs(source):
    baseline = json.loads((ROOT / source["baseline_path"]).read_text())
    parsed = json.loads((ROOT / source["pair_parser_path"]).read_text())
    scored = json.loads((ROOT / source["pair_score_path"]).read_text())
    details = [
        baseline["detection"]["details"][slot]
        for slot in source["source_slots"]
    ]
    sentences = [
        baseline["sentences"][slot]
        for slot in source["source_slots"]
    ]
    return baseline, details, sentences, parsed, scored


def median(values):
    return statistics.median(values)


def validate_scores(values, expected_length):
    if (
        len(values) != expected_length
        or not all(
            isinstance(value, (int, float))
            and math.isfinite(value)
            and 0 <= value <= 1
            for value in values
        )
    ):
        raise ValueError("Invalid S2Match score vector")


def run_pair(folder, source, bank, utils, heartbeat, completed):
    path = checkpoint_path(folder, source)
    if path.exists():
        return json.loads(path.read_text()), 0
    _baseline, details, sentences, parsed, scored = load_graphs(source)
    composed_raw = compose_raw_amrs(
        details[0]["raw_amr"],
        details[1]["raw_amr"],
    )
    composed_amr = transform_amr(composed_raw, utils)
    if (
        text_sha(composed_raw) != source["composed_raw_sha256"]
        or text_sha(composed_amr) != source["composed_amr_sha256"]
        or text_sha(parsed["detail"]["amr"])
        != source["reparsed_amr_sha256"]
    ):
        raise ValueError("Graph identity changed")
    heartbeat.update(
        stage="composition_bank",
        completed_pairs=completed,
        total_pairs=60,
        prompt_id=source["prompt_id"],
        pair_index=source["pair_index"],
    )
    started = time.monotonic()
    bank_scores = [
        float(utils.compute_s2match_score(composed_amr, template))
        for template in bank
    ]
    validate_scores(bank_scores, 50)
    target_rows = []
    calls = 50
    for component, detail, sentence in zip(
        source["components"],
        details,
        sentences,
    ):
        template = sentence["chosen_template"]
        conditions = {
            "original": detail["amr"],
            "composition": composed_amr,
            "reparse": parsed["detail"]["amr"],
        }
        scores = {}
        for condition, graph in conditions.items():
            values = [
                float(utils.compute_s2match_score(graph, template))
                for _ in range(REPEATS)
            ]
            validate_scores(values, REPEATS)
            scores[condition] = {
                "observations": values,
                "median": median(values),
            }
            calls += REPEATS
        target_rows.append(
            {
                **component,
                "scores": scores,
            }
        )
    similarity = [
        float(
            utils.compute_s2match_score(
                composed_amr,
                parsed["detail"]["amr"],
            )
        )
        for _ in range(REPEATS)
    ]
    validate_scores(similarity, REPEATS)
    calls += REPEATS
    checkpoint = {
        "prompt_id": source["prompt_id"],
        "pair_index": source["pair_index"],
        "source_slots": source["source_slots"],
        "manifest_sha256": sha(folder / "manifest.json"),
        "composed_raw_amr": composed_raw,
        "composed_amr": composed_amr,
        "reparsed_amr": parsed["detail"]["amr"],
        "triple_counts": {
            "original_sum": sum(
                graph_triples(detail["amr"]) for detail in details
            ),
            "composition": graph_triples(composed_amr),
            "reparse": graph_triples(parsed["detail"]["amr"]),
        },
        "bank": {
            "composition_scores": bank_scores,
            "composition_max": max(bank_scores),
            "composition_green": max(bank_scores) >= GREEN_THRESHOLD,
            "saved_original_max": [detail["score"] for detail in details],
            "saved_original_green": [
                detail["score"] >= GREEN_THRESHOLD for detail in details
            ],
            "saved_reparse_max": scored["detail"]["score"],
            "saved_reparse_green": scored["green"],
        },
        "targets": target_rows,
        "composition_reparse_similarity": {
            "observations": similarity,
            "median": median(similarity),
        },
        "new_s2match_observations": calls,
        "elapsed_seconds": time.monotonic() - started,
    }
    if calls != 85:
        raise ValueError("Per-pair S2Match budget mismatch")
    freeze(path, checkpoint)
    return checkpoint, calls


def sign_test_loss(differences):
    losses = sum(value < 0 for value in differences)
    gains = sum(value > 0 for value in differences)
    ties = len(differences) - losses - gains
    n = losses + gains
    probability = (
        sum(math.comb(n, k) for k in range(losses, n + 1)) / 2**n
        if n
        else 1.0
    )
    median_difference = median(differences)
    return {
        "differences": differences,
        "median_difference": median_difference,
        "losses": losses,
        "ties": ties,
        "gains": gains,
        "non_ties": n,
        "one_sided_sign_p": probability,
        "confirmed_loss": bool(
            median_difference < 0 and probability <= SIGN_ALPHA
        ),
    }


def summarize(folder=OUTPUT):
    manifest = json.loads((folder / "manifest.json").read_text())
    checkpoints = [
        json.loads(checkpoint_path(folder, source).read_text())
        for source in manifest["sources"]
    ]
    if len(checkpoints) != 60:
        raise ValueError("Incomplete pair checkpoints")
    document_targets = {prompt_id: [] for prompt_id in EXPECTED_IDS}
    all_targets = []
    for checkpoint in checkpoints:
        for target in checkpoint["targets"]:
            row = {
                "prompt_id": checkpoint["prompt_id"],
                "pair_index": checkpoint["pair_index"],
                "slot": target["slot"],
                "accepted": target["accepted"],
                "original": target["scores"]["original"]["median"],
                "composition": target["scores"]["composition"]["median"],
                "reparse": target["scores"]["reparse"]["median"],
            }
            all_targets.append(row)
            if row["accepted"]:
                document_targets[row["prompt_id"]].append(row)
    document_rows = []
    for prompt_id in EXPECTED_IDS:
        rows = document_targets[prompt_id]
        if not rows:
            raise ValueError("Document has no accepted pair target")
        values = {
            condition: statistics.mean(row[condition] for row in rows)
            for condition in ("original", "composition", "reparse")
        }
        document_rows.append(
            {
                "prompt_id": prompt_id,
                "accepted_targets": len(rows),
                **values,
                "composition_minus_original": (
                    values["composition"] - values["original"]
                ),
                "reparse_minus_composition": (
                    values["reparse"] - values["composition"]
                ),
            }
        )
    composition_test = sign_test_loss(
        [row["composition_minus_original"] for row in document_rows]
    )
    reparse_test = sign_test_loss(
        [row["reparse_minus_composition"] for row in document_rows]
    )
    composition_loss = composition_test["confirmed_loss"]
    reparse_loss = reparse_test["confirmed_loss"]
    if composition_loss and reparse_loss:
        decision = "composition_dilution_plus_reparse_loss"
    elif composition_loss:
        decision = "composition_dilution_only"
    elif reparse_loss:
        decision = "reparse_loss_only"
    else:
        decision = "no_confirmed_loss"
    accepted = [row for row in all_targets if row["accepted"]]
    result = {
        "status": "complete_composition_decomposition",
        "integrity": {
            "documents": len(document_rows),
            "pairs": len(checkpoints),
            "targets": len(all_targets),
            "accepted_targets": len(accepted),
            "new_s2match_observations": sum(
                row["new_s2match_observations"] for row in checkpoints
            ),
            "new_api_calls": 0,
            "new_parser_calls": 0,
        },
        "primary": {
            "composition_matcher_dilution": composition_test,
            "additional_reparse_loss": reparse_test,
            "decision": decision,
        },
        "target_scores": {
            condition: {
                "median": median([row[condition] for row in accepted]),
                "mean": statistics.mean(row[condition] for row in accepted),
                "green": sum(
                    row[condition] >= GREEN_THRESHOLD for row in accepted
                ),
                "denominator": len(accepted),
            }
            for condition in ("original", "composition", "reparse")
        },
        "bank_descriptive": {
            "original_green_units": sum(
                sum(row["bank"]["saved_original_green"])
                for row in checkpoints
            ),
            "original_units": 120,
            "composition_green_units": sum(
                row["bank"]["composition_green"] for row in checkpoints
            ),
            "composition_units": 60,
            "reparse_green_units": sum(
                row["bank"]["saved_reparse_green"] for row in checkpoints
            ),
            "reparse_units": 60,
        },
        "graph_descriptive": {
            key: {
                "median": median([
                    row["triple_counts"][key] for row in checkpoints
                ]),
                "mean": statistics.mean([
                    row["triple_counts"][key] for row in checkpoints
                ]),
            }
            for key in ("original_sum", "composition", "reparse")
        },
        "composition_reparse_similarity": {
            "median": median([
                row["composition_reparse_similarity"]["median"]
                for row in checkpoints
            ]),
            "mean": statistics.mean([
                row["composition_reparse_similarity"]["median"]
                for row in checkpoints
            ]),
        },
        "documents": document_rows,
        "limitations": [
            "Deterministic composition is not a gold AMR annotation.",
            "Additional reparse loss still includes matcher response.",
            "Source prose is low quality and previously observed.",
            "S2Match internally reseeds; all five observations are retained.",
            "Not a natural-text ASR, TPR, or FPR estimate.",
        ],
        "manifest_sha256": sha(folder / "manifest.json"),
    }
    write_json(folder / "summary.json", result)
    write_report(folder, result)
    return result


def write_report(folder, result):
    primary = result["primary"]
    scores = result["target_scores"]
    bank = result["bank_descriptive"]
    lines = [
        "# SWAN deterministic-AMR composition decomposition",
        "",
        "| Condition | Accepted target median | Target green | Bank green units |",
        "|---|---:|---:|---:|",
        f"| Original constituents | {scores['original']['median']:.4f} | "
        f"{scores['original']['green']}/108 | "
        f"{bank['original_green_units']}/120 |",
        f"| Deterministic composition | "
        f"{scores['composition']['median']:.4f} | "
        f"{scores['composition']['green']}/108 | "
        f"{bank['composition_green_units']}/60 |",
        f"| Merged-text reparse | {scores['reparse']['median']:.4f} | "
        f"{scores['reparse']['green']}/108 | "
        f"{bank['reparse_green_units']}/60 |",
        "",
        f"- Composition minus original median document difference: "
        f"**{primary['composition_matcher_dilution']['median_difference']:.4f}** "
        f"(one-sided sign p="
        f"{primary['composition_matcher_dilution']['one_sided_sign_p']:.6g}).",
        f"- Reparse minus composition median document difference: "
        f"**{primary['additional_reparse_loss']['median_difference']:.4f}** "
        f"(one-sided sign p="
        f"{primary['additional_reparse_loss']['one_sided_sign_p']:.6g}).",
        f"- Decision: `{primary['decision']}`.",
        "- No API, parser, generation, injection, or attack call was run.",
        "",
    ]
    (folder / "report.md").write_text("\n".join(lines))


def run(folder=OUTPUT):
    manifest = check(folder)
    utils = load_utils()
    bank = [
        row["normalized_template_amr"]
        for row in json.loads(BANK_PATH.read_text())
    ]
    new_calls = 0
    with Heartbeat(folder) as heartbeat:
        for index, source in enumerate(manifest["sources"]):
            _checkpoint, calls = run_pair(
                folder,
                source,
                bank,
                utils,
                heartbeat,
                index,
            )
            new_calls += calls
            heartbeat.update(
                stage="scoring",
                completed_pairs=index + 1,
                total_pairs=60,
                new_s2match_calls=new_calls,
            )
        heartbeat.update(
            stage="complete",
            completed_pairs=60,
            total_pairs=60,
            new_s2match_calls=new_calls,
        )
    result = summarize(folder)
    if result["integrity"]["new_s2match_observations"] != EXPECTED_S2MATCH:
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
                "sources": len(manifest["sources"]),
                "manifest_sha256": sha(args.output / "manifest.json"),
            }
        elif args.stage == "run":
            result = run(args.output)
        else:
            result = summarize(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
