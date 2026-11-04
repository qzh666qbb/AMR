"""Freeze and verify the final SWAN segmentation-audit preregistration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess

from scipy.stats import beta

from run_confirmation_experiment import digest, freeze
from run_segmentation_intervention import ROOT, sha


OUTPUT = ROOT / "experiments/swan_final_confirmation1"
SWAN_COMMIT = "2ca0c2071c8d0c6bb3739b2b2bcfb34cfe63eb2a"
SWORDSTAMP_COMMIT = "b7925dc76d56403ee94f80c9b7737c6d10306088"
DATASET_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
SAMPLE_SIZE = 104
SOURCE_ALLOCATION = 250
CALIBRATION_SIZE = 250
TEST_SIZE = 250
PRIMARY_ALPHA = 0.025
FAVORABLE_DISCORDANCE = 0.25
REVERSE_DISCORDANCE = 0.05


def binomial_upper_tail(n: int, threshold: int, probability: float) -> float:
    return sum(
        math.comb(n, count)
        * probability**count
        * (1 - probability) ** (n - count)
        for count in range(threshold, n + 1)
    )


def exact_mcnemar_critical_value(
    discordant: int,
    alpha: float,
) -> int:
    for favorable in range(discordant + 1):
        if binomial_upper_tail(discordant, favorable, 0.5) <= alpha:
            return favorable
    return discordant + 1


def unconditional_mcnemar_power(
    sample_size: int,
    favorable_probability: float,
    reverse_probability: float,
    alpha: float,
) -> float:
    discordance_probability = favorable_probability + reverse_probability
    conditional_favorable = (
        favorable_probability / discordance_probability
    )
    power = 0.0
    for discordant in range(sample_size + 1):
        discordant_mass = (
            math.comb(sample_size, discordant)
            * discordance_probability**discordant
            * (1 - discordance_probability)
            ** (sample_size - discordant)
        )
        critical = exact_mcnemar_critical_value(discordant, alpha)
        if critical <= discordant:
            power += discordant_mass * binomial_upper_tail(
                discordant,
                critical,
                conditional_favorable,
            )
    return power


def clopper_pearson_interval(
    successes: int,
    trials: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(
            beta.ppf(
                1 - alpha / 2,
                successes + 1,
                trials - successes,
            )
        )
    )
    return lower, upper


def maximum_clopper_pearson_width(trials: int) -> dict:
    values = []
    for successes in range(trials + 1):
        lower, upper = clopper_pearson_interval(successes, trials)
        values.append((upper - lower, successes, lower, upper))
    width, successes, lower, upper = max(values)
    return {
        "trials": trials,
        "successes_at_maximum": successes,
        "lower": lower,
        "upper": upper,
        "maximum_width": width,
    }


def git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def dependency_paths(folder: Path) -> list[Path]:
    paths = {
        Path(__file__),
        ROOT / "scripts/swan_final_structural.py",
        ROOT / "scripts/test_swan_final_confirmation.py",
        ROOT / "任务规划.md",
        folder / "protocol.md",
        folder / "claims.json",
        ROOT / "experiments/literature_novelty_audit1/report.md",
        ROOT / "experiments/swan_sentence_growth1/injection_provenance.json",
        ROOT / "third_party/SWAN/injection/injection_amr_utils.py",
        ROOT / "third_party/SWAN/utils/amr_utils.py",
        ROOT / "third_party/SWAN/utils/text_utils.py",
        ROOT / "third_party/SWAN/amr_bank/banks/amr_bank_50.json",
        ROOT
        / "third_party/SWAN/amr_bank/artifacts/normalized_amr_examples.json",
        ROOT
        / "third_party/SWAN/amr-metric-suite/py3-Smatch-and-S2match"
        / "smatch/s2match.py",
        ROOT / "third_party/SwordStamp/attacks/simple/structural.py",
        ROOT / "third_party/SwordStamp/LICENSE",
        ROOT / "vectors/glove.6B.100d.txt",
    }
    paths.update(
        path
        for directory in (
            ROOT / "models/model_parse_xfm_bart_large-v0_1_0",
            ROOT / "models/bart-large-tokenizer",
            ROOT / ".nltk_data/tokenizers/punkt_tab/english",
        )
        for path in directory.glob("*")
        if path.is_file()
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing preregistration dependencies: "
            + ", ".join(str(path) for path in sorted(missing))
        )
    return sorted(paths)


def package_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "transformers",
            "amrlib",
            "penman",
            "spacy",
            "nltk",
            "numpy",
            "scipy",
            "requests",
        )
    }


def frozen_design(folder: Path) -> dict:
    claims = json.loads((folder / "claims.json").read_text())
    if claims["paper_track"] != "swan_specific_segmentation_audit":
        raise ValueError("Unexpected paper track")
    power = unconditional_mcnemar_power(
        SAMPLE_SIZE,
        FAVORABLE_DISCORDANCE,
        REVERSE_DISCORDANCE,
        PRIMARY_ALPHA,
    )
    precision = maximum_clopper_pearson_width(SAMPLE_SIZE)
    return {
        "status": "preregistered_before_source_acquisition",
        "protocol": {
            "workflow": "swan_final_segmentation_audit",
            "paper_track": claims["paper_track"],
            "sample_size_minimum": SAMPLE_SIZE,
            "source_allocation": SOURCE_ALLOCATION,
            "human_calibration": CALIBRATION_SIZE,
            "human_test": TEST_SIZE,
            "dataset": {
                "name": "allenai/c4",
                "configuration": "realnewslike",
                "revision": DATASET_REVISION,
                "watermarked_source_start_shard": 6,
            },
            "baseline": {
                "sentence_slots": 8,
                "word_range": [100, 220],
                "minimum_detector_units": 8,
                "minimum_accepted_target_slots": 5,
                "target_s2match_threshold": 0.75,
                "maximum_templates_per_slot": 2,
                "maximum_attempts_per_template": 5,
                "maximum_attempts_per_slot": 10,
            },
            "detector": {
                "bank_size": 50,
                "green_s2match_threshold": 0.65,
                "null_probability": 0.25,
                "official_z_threshold": 2.33,
                "calibration_false_positives_maximum": 2,
                "calibration_denominator": CALIBRATION_SIZE,
                "minimum_original_tpr": 0.80,
            },
            "conditions": {
                "C0": "original",
                "C1": "swordstamp_merge_adjacent_ratio_1.0",
                "C2": "swordstamp_boundary_exchange_ratio_1.0",
                "C3": "swordstamp_uniform_rechunk_16_words",
                "C4a": "strict_semicolon_merge_0.25",
                "C4b": "strict_semicolon_merge_0.50",
                "C4c": "strict_semicolon_merge_0.75",
                "C4d": "strict_semicolon_merge_1.00",
                "C5": "C4d_with_original_detector_spans",
            },
            "s2match_repeats": 5,
            "remote_generation_request_cap": 20417,
            "cross_watermark_extension": False,
            "dipper_baseline": False,
            "defense_experiment": False,
        },
        "power": {
            "test": "one_sided_unconditional_exact_mcnemar",
            "sample_size": SAMPLE_SIZE,
            "alpha": PRIMARY_ALPHA,
            "favorable_discordance_probability": (
                FAVORABLE_DISCORDANCE
            ),
            "reverse_discordance_probability": REVERSE_DISCORDANCE,
            "net_difference": (
                FAVORABLE_DISCORDANCE - REVERSE_DISCORDANCE
            ),
            "power": power,
            "binomial_precision": precision,
        },
        "external_versions": {
            "swan_upstream_commit": SWAN_COMMIT,
            "swordstamp_commit": SWORDSTAMP_COMMIT,
            "swordstamp_local_head": git_head(
                ROOT / "third_party/SwordStamp"
            ),
            "workspace_head": git_head(ROOT),
        },
        "protocol_sha256": sha(folder / "protocol.md"),
        "claims_sha256": sha(folder / "claims.json"),
        "claims_digest": digest(claims),
        "dependencies": {
            str(path.relative_to(ROOT)): sha(path)
            for path in dependency_paths(folder)
        },
        "packages": package_versions(),
        "prohibitions": [
            "source_acquisition_before_manifest",
            "attack_score_visible_to_quality_raters",
            "post_hoc_sample_replacement",
            "post_hoc_threshold_or_quality_gate_change",
            "adaptive_attack_rescue",
            "cross_watermark_extension",
            "defense_branch",
        ],
    }


def check(folder: Path = OUTPUT) -> dict:
    path = folder / "manifest.json"
    saved = json.loads(path.read_text())
    expected = frozen_design(folder)
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError(f"Frozen manifest differs at {key}")
    return saved


def prepare(folder: Path = OUTPUT) -> dict:
    path = folder / "manifest.json"
    if path.exists():
        return check(folder)
    forbidden = (
        "source_manifest.json",
        "human_control_manifest.json",
        "runtime.json",
        "quality_packet.json",
        "quality_labels.json",
        "summary.json",
    )
    present = [name for name in forbidden if (folder / name).exists()]
    if present:
        raise RuntimeError(
            "Final artifacts exist before preregistration: "
            + ", ".join(present)
        )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **frozen_design(folder),
    }
    freeze(path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("prepare", "check"),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = prepare(args.output) if args.stage == "prepare" else check(
        args.output
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "manifest": str(args.output / "manifest.json"),
                "power": result["power"]["power"],
                "maximum_cp_width": result["power"][
                    "binomial_precision"
                ]["maximum_width"],
                "dependencies": len(result["dependencies"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
