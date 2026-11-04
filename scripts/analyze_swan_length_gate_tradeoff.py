"""Rescore every saved length-confirmation2 candidate without network access."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Callable, Iterable

import run_swan_positive_collection as swan


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/swan_positive_deepseek_length_confirmation2"
OUTPUT = ROOT / "experiments/swan_length_gate_tradeoff1"
PROTOCOL = OUTPUT / "protocol.md"
ATTEMPTS = OUTPUT / "attempts.jsonl"
SUMMARY = OUTPUT / "summary.json"
REPORT = OUTPUT / "report.md"
MANIFEST = OUTPUT / "manifest.json"
S2MATCH_RUNTIME = ROOT / "third_party/swan_s2match_runtime_711a231d"
S2MATCH_FILE = S2MATCH_RUNTIME / "s2match.py"
S2MATCH_PROVENANCE = S2MATCH_RUNTIME / "provenance.json"
VECTORS = ROOT / "vectors/glove.6B.100d.txt"
SOURCE_MANIFEST = SOURCE / "manifest.json"
SOURCE_ATTEMPTS = SOURCE / "generation_attempts"

EXPECTED_TRACES = 606
EXPECTED_CANDIDATES = 605
EXPECTED_HISTORICAL_EVALUATIONS = 359
THRESHOLD = 0.75
ABS_TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def digest(value) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, payload) -> None:
    write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
    )


def package_versions() -> dict:
    return {
        name: importlib.metadata.version(name)
        for name in ("numpy", "scipy")
    }


def attempt_inventory() -> tuple[list[tuple[Path, dict]], dict]:
    candidates = []
    identities = []
    censored = []
    for path in sorted(SOURCE_ATTEMPTS.glob("*.json")):
        row = load_json(path)
        relative = str(path.relative_to(ROOT))
        file_hash = sha256(path)
        identities.append(
            {
                "path": relative,
                "sha256": file_hash,
                "call_id": row.get("call_id"),
            }
        )
        if row.get("status") == "transport_failed_or_uncertain":
            if "candidate" in row or "candidate_template_amr" in row:
                raise ValueError(
                    f"Uncertain trace unexpectedly has a candidate: {path}"
                )
            censored.append(
                {
                    "path": relative,
                    "sha256": file_hash,
                    "call_id": row.get("call_id"),
                    "error_type": row.get("error_type"),
                }
            )
            continue
        candidates.append((path, row))
    if len(identities) != EXPECTED_TRACES:
        raise ValueError(
            f"Expected {EXPECTED_TRACES} traces, got {len(identities)}"
        )
    if len(candidates) != EXPECTED_CANDIDATES:
        raise ValueError(
            f"Expected {EXPECTED_CANDIDATES} candidates, "
            f"got {len(candidates)}"
        )
    if len(censored) != EXPECTED_TRACES - EXPECTED_CANDIDATES:
        raise ValueError("Unexpected censored trace count")
    if len({row["call_id"] for _, row in candidates}) != len(candidates):
        raise ValueError("Generation attempt call IDs are not unique")
    if len({row["call_id"] for _, row in candidates}) + len(censored) != len(
        {
            identity["call_id"]
            for identity in identities
        }
    ):
        raise ValueError("Trace call IDs are not unique")
    for path, row in candidates:
        if path.stem != row["call_id"]:
            raise ValueError(f"Attempt filename mismatch: {path}")
        if row.get("status") != "complete":
            raise ValueError(f"Attempt is not complete: {path}")
        required = (
            "allocation_id",
            "slot",
            "attempt",
            "template_index",
            "template",
            "template_sha256",
            "candidate",
            "candidate_sha256",
            "candidate_gate",
            "candidate_template_amr",
            "candidate_template_amr_sha256",
            "threshold",
            "score",
            "accepted",
            "s2match_evaluated",
        )
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Attempt lacks {missing}: {path}")
        if row["threshold"] != THRESHOLD:
            raise ValueError(f"Attempt threshold changed: {path}")
    return candidates, {
        "trace_count": len(identities),
        "candidate_count": len(candidates),
        "censored_count": len(censored),
        "aggregate_sha256": digest(identities),
        "files": identities,
        "censored": censored,
    }


class S2MatchEngine:
    """Exact frozen matcher with a read-only in-memory GloVe cache."""

    def __init__(self) -> None:
        runtime = str(S2MATCH_RUNTIME)
        if runtime not in sys.path:
            sys.path.insert(0, runtime)
        import s2match

        if Path(s2match.__file__).resolve() != S2MATCH_FILE.resolve():
            raise RuntimeError("Loaded an ambiguous S2Match module")
        self.module = s2match
        loader = s2match.load_vecs
        self.vectors = loader(str(VECTORS))

        def cached_loader(path: str):
            if Path(path).resolve() != VECTORS.resolve():
                raise ValueError("Unexpected vector path")
            return self.vectors

        s2match.load_vecs = cached_loader

    def metric(self, candidate: str, target: str, seed: int) -> tuple:
        with swan.fixed_score_seed(seed):
            values = self.module.compute_s2match_from_strings(
                candidate,
                target,
                vectors_file=str(VECTORS),
                similarity_function="cosine",
                cutoff=0.5,
                diffsense=0.5,
                mwp="split",
                do_not_mark_quotes=False,
                verbose=False,
            )
        return tuple(float(value) for value in values)

    def graph_counts(self, source: str) -> dict:
        graph = self.module.amr.AMR.parse_AMR_line(source, False)
        instances, attributes, relations = graph.get_triples()
        return {
            "instances": len(instances),
            "attributes": len(attributes),
            "relations": len(relations),
            "triples": len(instances) + len(attributes) + len(relations),
        }


def validate_metric(metric: tuple) -> None:
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
    reconstructed = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    if not math.isclose(f1, reconstructed, abs_tol=ABS_TOLERANCE):
        raise ValueError("F1 does not reconstruct from precision and recall")


def length_class(word_count: int) -> str:
    if word_count < 7:
        return "under_7"
    if word_count > 12:
        return "over_12"
    return "target_7_12"


def enrich_attempt(
    source_path: Path,
    row: dict,
    metric_fn: Callable[[str, str, int], tuple],
    count_fn: Callable[[str], dict],
) -> dict:
    gate = row["candidate_gate"]
    required_gate = (
        "word_count",
        "word_range_pass",
        "sentence_units",
        "single_sentence_pass",
        "terminal_punctuation_pass",
        "candidate_gate_pass",
    )
    if any(key not in gate for key in required_gate):
        raise ValueError(f"Incomplete candidate gate: {source_path}")
    expected_gate = all(
        gate[key]
        for key in (
            "word_range_pass",
            "single_sentence_pass",
            "terminal_punctuation_pass",
        )
    )
    if bool(gate["candidate_gate_pass"]) != expected_gate:
        raise ValueError(f"Candidate gate does not reconstruct: {source_path}")

    seed = swan.score_seed(
        row["allocation_id"],
        row["slot"],
        row["attempt"],
    )
    metric = metric_fn(
        row["candidate_template_amr"],
        row["template"],
        seed,
    )
    validate_metric(metric)
    precision, recall, f1 = metric
    threshold_pass = f1 >= THRESHOLD
    evaluated = bool(row["s2match_evaluated"])

    if evaluated:
        if row["score_seed"] != seed:
            raise ValueError(f"Historical score seed mismatch: {source_path}")
        score_delta = f1 - float(row["score"])
        if not math.isclose(
            f1,
            float(row["score"]),
            abs_tol=ABS_TOLERANCE,
        ):
            raise ValueError(
                f"Historical S2Match score mismatch: {source_path}"
            )
        if bool(row["accepted"]) != threshold_pass:
            raise ValueError(
                f"Historical acceptance does not reconstruct: {source_path}"
            )
        historical_score = float(row["score"])
    else:
        if (
            row["score_seed"] is not None
            or float(row["score"]) != 0.0
            or bool(row["accepted"])
        ):
            raise ValueError(
                f"Invalid gate-rejection sentinel: {source_path}"
            )
        historical_score = None
        score_delta = None

    candidate_counts = count_fn(row["candidate_template_amr"])
    target_counts = count_fn(row["template"])
    candidate_triples = candidate_counts["triples"]
    target_triples = target_counts["triples"]
    matched_from_precision = precision * candidate_triples
    matched_from_recall = recall * target_triples
    if not math.isclose(
        matched_from_precision,
        matched_from_recall,
        abs_tol=1e-9,
    ):
        raise ValueError(f"S2Match matched mass is inconsistent: {source_path}")

    return {
        "call_id": row["call_id"],
        "source_trace": str(source_path.relative_to(ROOT)),
        "source_trace_sha256": sha256(source_path),
        "allocation_id": row["allocation_id"],
        "slot": row["slot"],
        "attempt": row["attempt"],
        "template_index": row["template_index"],
        "template_sha256": row["template_sha256"],
        "candidate_sha256": row["candidate_sha256"],
        "candidate_template_amr_sha256": row[
            "candidate_template_amr_sha256"
        ],
        "word_count": gate["word_count"],
        "length_class": length_class(gate["word_count"]),
        "word_range_pass": bool(gate["word_range_pass"]),
        "sentence_units": gate["sentence_units"],
        "single_sentence_pass": bool(gate["single_sentence_pass"]),
        "terminal_punctuation_pass": bool(
            gate["terminal_punctuation_pass"]
        ),
        "candidate_gate_pass": bool(gate["candidate_gate_pass"]),
        "historical_s2match_evaluated": evaluated,
        "historical_score": historical_score,
        "historical_accepted": bool(row["accepted"]),
        "score_seed": seed,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "score_delta_from_history": score_delta,
        "threshold": THRESHOLD,
        "threshold_pass": threshold_pass,
        "gate_rejected_threshold_pass": bool(
            not gate["candidate_gate_pass"] and threshold_pass
        ),
        "candidate_graph": candidate_counts,
        "target_graph": target_counts,
        "candidate_minus_target_triples": (
            candidate_triples - target_triples
        ),
        "candidate_to_target_triple_ratio": (
            candidate_triples / target_triples
        ),
        "matched_triple_mass": (
            matched_from_precision + matched_from_recall
        )
        / 2,
        "precision_minus_recall": precision - recall,
    }


def quantile(values: Iterable[float], probability: float):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def distribution(values: Iterable[float]) -> dict:
    values = list(values)
    if not values:
        return {
            "min": None,
            "q25": None,
            "median": None,
            "mean": None,
            "q75": None,
            "max": None,
        }
    return {
        "min": min(values),
        "q25": quantile(values, 0.25),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "q75": quantile(values, 0.75),
        "max": max(values),
    }


def group_summary(rows: list[dict]) -> dict:
    count = len(rows)
    passes = sum(row["threshold_pass"] for row in rows)
    gate_passes = sum(row["candidate_gate_pass"] for row in rows)
    hidden = sum(row["gate_rejected_threshold_pass"] for row in rows)
    return {
        "attempts": count,
        "candidate_gate_passes": gate_passes,
        "threshold_passes": passes,
        "threshold_pass_rate": passes / count if count else None,
        "gate_rejected_threshold_passes": hidden,
        "word_count": distribution(row["word_count"] for row in rows),
        "candidate_triples": distribution(
            row["candidate_graph"]["triples"] for row in rows
        ),
        "target_triples": distribution(
            row["target_graph"]["triples"] for row in rows
        ),
        "candidate_minus_target_triples": distribution(
            row["candidate_minus_target_triples"] for row in rows
        ),
        "precision": distribution(row["precision"] for row in rows),
        "recall": distribution(row["recall"] for row in rows),
        "f1": distribution(row["f1"] for row in rows),
        "precision_minus_recall": distribution(
            row["precision_minus_recall"] for row in rows
        ),
        "precision_below_recall": sum(
            row["precision"] < row["recall"] for row in rows
        ),
    }


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2 + 1
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def pearson(first: list[float], second: list[float]):
    if len(first) != len(second) or len(first) < 2:
        return None
    first_mean = statistics.fmean(first)
    second_mean = statistics.fmean(second)
    first_centered = [value - first_mean for value in first]
    second_centered = [value - second_mean for value in second]
    denominator = math.sqrt(
        sum(value * value for value in first_centered)
        * sum(value * value for value in second_centered)
    )
    if denominator == 0:
        return None
    return sum(
        left * right
        for left, right in zip(first_centered, second_centered)
    ) / denominator


def spearman(rows: list[dict], first: Callable, second: Callable):
    left = [float(first(row)) for row in rows]
    right = [float(second(row)) for row in rows]
    return pearson(average_ranks(left), average_ranks(right))


def grouped(rows: list[dict], key: Callable) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return dict(groups)


def within_template_contrasts(rows: list[dict]) -> dict:
    comparisons = []
    for template_index, template_rows in sorted(
        grouped(rows, lambda row: row["template_index"]).items()
    ):
        classes = grouped(template_rows, lambda row: row["length_class"])
        eligible = classes.get("target_7_12", [])
        long_rows = classes.get("over_12", [])
        if not eligible or not long_rows:
            continue
        comparison = {
            "template_index": template_index,
            "template_sha256": template_rows[0]["template_sha256"],
            "target_7_12_attempts": len(eligible),
            "over_12_attempts": len(long_rows),
        }
        for field, getter in (
            ("precision", lambda row: row["precision"]),
            ("recall", lambda row: row["recall"]),
            ("f1", lambda row: row["f1"]),
            (
                "candidate_triples",
                lambda row: row["candidate_graph"]["triples"],
            ),
        ):
            eligible_median = statistics.median(
                getter(row) for row in eligible
            )
            long_median = statistics.median(
                getter(row) for row in long_rows
            )
            comparison[f"target_7_12_{field}_median"] = eligible_median
            comparison[f"over_12_{field}_median"] = long_median
            comparison[
                f"target_7_12_minus_over_12_{field}_median"
            ] = eligible_median - long_median
        comparisons.append(comparison)
    return {
        "templates_with_both_groups": len(comparisons),
        "comparisons": comparisons,
        "median_target_7_12_minus_over_12": {
            field: (
                statistics.median(
                    row[
                        f"target_7_12_minus_over_12_{field}_median"
                    ]
                    for row in comparisons
                )
                if comparisons
                else None
            )
            for field in ("precision", "recall", "f1", "candidate_triples")
        },
    }


def observed_slot_opportunities(rows: list[dict]) -> dict:
    slots = []
    for slot_id, slot_rows in sorted(
        grouped(
            rows,
            lambda row: (
                row["allocation_id"],
                row["slot"],
            ),
        ).items()
    ):
        ordered = sorted(slot_rows, key=lambda row: row["attempt"])
        first_threshold = next(
            (row for row in ordered if row["threshold_pass"]),
            None,
        )
        historical = next(
            (row for row in ordered if row["historical_accepted"]),
            None,
        )
        slots.append(
            {
                "allocation_id": slot_id[0],
                "slot": slot_id[1],
                "observed_candidate_attempts": len(ordered),
                "historical_accepted": historical is not None,
                "historical_accept_call_id": (
                    historical["call_id"] if historical else None
                ),
                "any_observed_threshold_pass": first_threshold is not None,
                "first_threshold_pass_call_id": (
                    first_threshold["call_id"]
                    if first_threshold
                    else None
                ),
                "first_threshold_pass_attempt": (
                    first_threshold["attempt"]
                    if first_threshold
                    else None
                ),
                "first_threshold_pass_gate_pass": (
                    first_threshold["candidate_gate_pass"]
                    if first_threshold
                    else None
                ),
                "first_threshold_pass_word_count": (
                    first_threshold["word_count"]
                    if first_threshold
                    else None
                ),
                "first_threshold_pass_f1": (
                    first_threshold["f1"] if first_threshold else None
                ),
            }
        )
    recovered = [
        row
        for row in slots
        if row["any_observed_threshold_pass"]
        and not row["first_threshold_pass_gate_pass"]
    ]
    return {
        "observed_slots": len(slots),
        "historically_accepted_slots": sum(
            row["historical_accepted"] for row in slots
        ),
        "slots_with_any_observed_threshold_pass": sum(
            row["any_observed_threshold_pass"] for row in slots
        ),
        "slots_whose_first_observed_threshold_pass_was_gate_rejected": len(
            recovered
        ),
        "recovered_slot_prefix_details": recovered,
        "slots": slots,
        "interpretation": (
            "This is restricted to each observed slot prefix. It does not "
            "simulate the unobserved later slots that an earlier acceptance "
            "would have caused."
        ),
    }


def build_summary(rows: list[dict], inventory: dict | None = None) -> dict:
    if inventory is None:
        inventory = {
            "trace_count": len(rows),
            "candidate_count": len(rows),
            "censored_count": 0,
        }
    evaluated = [
        row for row in rows if row["historical_s2match_evaluated"]
    ]
    gate_rejected = [
        row for row in rows if not row["candidate_gate_pass"]
    ]
    hidden = [
        row for row in rows if row["gate_rejected_threshold_pass"]
    ]
    max_delta = max(
        abs(row["score_delta_from_history"])
        for row in evaluated
    )
    gate_failure_combinations = grouped(
        gate_rejected,
        lambda row: "+".join(
            name
            for name, passed in (
                ("word_range", row["word_range_pass"]),
                ("single_sentence", row["single_sentence_pass"]),
                (
                    "terminal_punctuation",
                    row["terminal_punctuation_pass"],
                ),
            )
            if not passed
        ),
    )
    by_template = grouped(rows, lambda row: row["template_index"])
    by_slot = grouped(
        rows,
        lambda row: f"{row['allocation_id']}/slot-{row['slot']:02d}",
    )
    by_word_count = grouped(rows, lambda row: row["word_count"])
    by_length = grouped(rows, lambda row: row["length_class"])
    by_gate = grouped(
        rows,
        lambda row: "pass" if row["candidate_gate_pass"] else "fail",
    )

    correlations = {
        "word_count_vs_candidate_triples": spearman(
            rows,
            lambda row: row["word_count"],
            lambda row: row["candidate_graph"]["triples"],
        ),
        "word_count_vs_precision": spearman(
            rows,
            lambda row: row["word_count"],
            lambda row: row["precision"],
        ),
        "word_count_vs_recall": spearman(
            rows,
            lambda row: row["word_count"],
            lambda row: row["recall"],
        ),
        "word_count_vs_f1": spearman(
            rows,
            lambda row: row["word_count"],
            lambda row: row["f1"],
        ),
        "candidate_minus_target_triples_vs_precision": spearman(
            rows,
            lambda row: row["candidate_minus_target_triples"],
            lambda row: row["precision"],
        ),
        "candidate_minus_target_triples_vs_recall": spearman(
            rows,
            lambda row: row["candidate_minus_target_triples"],
            lambda row: row["recall"],
        ),
        "candidate_minus_target_triples_vs_f1": spearman(
            rows,
            lambda row: row["candidate_minus_target_triples"],
            lambda row: row["f1"],
        ),
    }

    return {
        "status": "complete_retrospective_offline_diagnostic",
        "scope": {
            "source_experiment": str(SOURCE.relative_to(ROOT)),
            "causal_prompt_effect_identifiable": False,
            "iid_inference_permitted": False,
            "reason": (
                "All candidates share one prompt condition and attempts are "
                "clustered under adaptive stopping."
            ),
        },
        "integrity": {
            "source_traces": inventory["trace_count"],
            "candidate_attempts": len(rows),
            "censored_uncertain_traces": inventory["censored_count"],
            "historical_s2match_evaluations": len(evaluated),
            "newly_observed_gate_rejected_s2match_scores": (
                len(rows) - len(evaluated)
            ),
            "unique_candidate_s2match_observations": len(rows),
            "final_materialization_s2match_computations": len(rows),
            "model_calls": 0,
            "generation_calls": 0,
            "neural_parser_calls": 0,
            "network_calls": 0,
            "detector_calls": 0,
            "attack_calls": 0,
        },
        "historical_reproduction": {
            "expected": EXPECTED_HISTORICAL_EVALUATIONS,
            "reproduced": len(evaluated),
            "maximum_absolute_f1_delta": max_delta,
            "absolute_tolerance": ABS_TOLERANCE,
            "all_exact_within_tolerance": (
                len(evaluated) == EXPECTED_HISTORICAL_EVALUATIONS
                and max_delta <= ABS_TOLERANCE
            ),
        },
        "gate_effect": {
            "historical_gate_passes": sum(
                row["candidate_gate_pass"] for row in rows
            ),
            "historical_accepted_attempts": sum(
                row["historical_accepted"] for row in rows
            ),
            "all_candidate_threshold_passes": sum(
                row["threshold_pass"] for row in rows
            ),
            "gate_rejected_attempts": len(gate_rejected),
            "gate_rejected_threshold_passes": len(hidden),
            "gate_rejected_threshold_pass_rate": (
                len(hidden) / len(gate_rejected)
                if gate_rejected
                else None
            ),
            "gate_failure_combinations": {
                name: len(group_rows)
                for name, group_rows in sorted(
                    gate_failure_combinations.items()
                )
            },
            "gate_rejected_threshold_pass_details": [
                {
                    "call_id": row["call_id"],
                    "word_count": row["word_count"],
                    "length_class": row["length_class"],
                    "template_index": row["template_index"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1"],
                    "candidate_triples": row["candidate_graph"]["triples"],
                    "target_triples": row["target_graph"]["triples"],
                }
                for row in hidden
            ],
            "observed_slot_opportunities": observed_slot_opportunities(rows),
            "adaptive_counterfactual_warning": (
                "Candidate-level threshold passes do not reconstruct how "
                "later attempts, slots, or documents would have evolved if "
                "the gate had accepted an earlier candidate."
            ),
        },
        "all_attempts": group_summary(rows),
        "by_length_class": {
            name: group_summary(by_length.get(name, []))
            for name in ("under_7", "target_7_12", "over_12")
        },
        "by_candidate_gate": {
            name: group_summary(by_gate.get(name, []))
            for name in ("pass", "fail")
        },
        "by_word_count": {
            str(name): group_summary(group_rows)
            for name, group_rows in sorted(by_word_count.items())
        },
        "by_allocation_slot": {
            name: group_summary(group_rows)
            for name, group_rows in sorted(by_slot.items())
        },
        "by_template": [
            {
                "template_index": name,
                "template_sha256": group_rows[0]["template_sha256"],
                **group_summary(group_rows),
            }
            for name, group_rows in sorted(by_template.items())
        ],
        "descriptive_spearman_rho": correlations,
        "within_template_target_vs_long": within_template_contrasts(rows),
        "decision": (
            "gate_directly_removed_observed_threshold_candidates"
            if hidden
            else "no_observed_threshold_candidate_removed_by_gate"
        ),
    }


def format_number(value, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def build_report(summary: dict) -> str:
    integrity = summary["integrity"]
    reproduction = summary["historical_reproduction"]
    gate = summary["gate_effect"]
    slot_effect = gate["observed_slot_opportunities"]
    correlations = summary["descriptive_spearman_rho"]
    contrasts = summary["within_template_target_vs_long"]
    lines = [
        "# SWAN length-gate tradeoff diagnostic",
        "",
        "Status: complete retrospective offline diagnostic.",
        "",
        "## Integrity",
        "",
        (
            f"All {integrity['candidate_attempts']} saved candidate AMRs "
            "were rescored "
            "against their recorded target templates. "
            f"{reproduction['reproduced']}/"
            f"{reproduction['expected']} historical scores reproduced "
            "within absolute tolerance "
            f"{reproduction['absolute_tolerance']:.0e}; the maximum "
            "absolute F1 delta was "
            f"{reproduction['maximum_absolute_f1_delta']:.3g}."
        ),
        "",
        (
            f"The source inventory contains {integrity['source_traces']} "
            "attempt traces: "
            f"{integrity['candidate_attempts']} complete candidates and "
            f"{integrity['censored_uncertain_traces']} uncertain trace with "
            "no candidate AMR."
        ),
        "",
        (
            f"The final materialization pass computed "
            f"{integrity['unique_candidate_s2match_observations']} unique "
            "candidate scores. Deterministic implementation reruns are "
            "documented in the protocol and are not additional samples. "
            "There were zero model, generation, neural-parser, network, "
            "detector, or attack calls."
        ),
        "",
        "## Candidate-level gate effect",
        "",
        "| Quantity | Count |",
        "|---|---:|",
        f"| Candidate attempts | {integrity['candidate_attempts']} |",
        (
            "| Historical candidate-gate passes | "
            f"{gate['historical_gate_passes']} |"
        ),
        (
            "| Historical accepted attempts | "
            f"{gate['historical_accepted_attempts']} |"
        ),
        (
            "| Threshold passes after rescoring every candidate | "
            f"{gate['all_candidate_threshold_passes']} |"
        ),
        f"| Gate-rejected attempts | {gate['gate_rejected_attempts']} |",
        (
            "| Gate-rejected candidates with F1 >= 0.75 | "
            f"{gate['gate_rejected_threshold_passes']} |"
        ),
        (
            "| Observed slots with any threshold pass | "
            f"{slot_effect['slots_with_any_observed_threshold_pass']} / "
            f"{slot_effect['observed_slots']} |"
        ),
        (
            "| Slot prefixes whose first threshold pass was gate-rejected | "
            f"{slot_effect['slots_whose_first_observed_threshold_pass_was_gate_rejected']} |"
        ),
        "",
        (
            "The candidate and slot-prefix rows are direct filtering "
            "quantities at their respective levels. Of the 359 gate-eligible "
            f"candidates, {gate['historical_gate_passes'] - gate['historical_accepted_attempts']} "
            "still failed S2Match. Of the 246 gate-rejected candidates, "
            f"{gate['gate_rejected_attempts'] - gate['gate_rejected_threshold_passes']} "
            "also fell below 0.75. The gate therefore contributed to the "
            "low yield but does not explain most candidate failures."
        ),
        "",
        "## Length and S2Match",
        "",
        (
            "| Length class | n | Gate pass | F1 >= .75 | "
            "Median candidate triples | Precision | Recall | F1 |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "under_7": "<7",
        "target_7_12": "7-12",
        "over_12": ">12",
    }
    for name in ("under_7", "target_7_12", "over_12"):
        group = summary["by_length_class"][name]
        lines.append(
            f"| {labels[name]} | {group['attempts']} | "
            f"{group['candidate_gate_passes']} | "
            f"{group['threshold_passes']} | "
            f"{format_number(group['candidate_triples']['median'], 1)} | "
            f"{format_number(group['precision']['median'])} | "
            f"{format_number(group['recall']['median'])} | "
            f"{format_number(group['f1']['median'])} |"
        )
    lines.extend(
        [
            "",
            "Descriptive Spearman correlations over all attempts:",
            "",
            (
                "- word count vs candidate triples: "
                f"`{format_number(correlations['word_count_vs_candidate_triples'])}`"
            ),
            (
                "- word count vs precision / recall / F1: "
                f"`{format_number(correlations['word_count_vs_precision'])}` / "
                f"`{format_number(correlations['word_count_vs_recall'])}` / "
                f"`{format_number(correlations['word_count_vs_f1'])}`"
            ),
            (
                "- candidate-minus-target triples vs precision / recall / F1: "
                f"`{format_number(correlations['candidate_minus_target_triples_vs_precision'])}` / "
                f"`{format_number(correlations['candidate_minus_target_triples_vs_recall'])}` / "
                f"`{format_number(correlations['candidate_minus_target_triples_vs_f1'])}`"
            ),
            "",
            (
                f"Across {contrasts['templates_with_both_groups']} templates "
                "observed in both the 7-12 and >12 groups, the median "
                "within-template 7-12 minus >12 contrast was "
                f"`{format_number(contrasts['median_target_7_12_minus_over_12']['precision'])}` "
                "for precision, "
                f"`{format_number(contrasts['median_target_7_12_minus_over_12']['recall'])}` "
                "for recall, and "
                f"`{format_number(contrasts['median_target_7_12_minus_over_12']['f1'])}` "
                "for F1."
            ),
            "",
            (
                "The >12-word group has a median of "
                f"{summary['by_length_class']['over_12']['candidate_triples']['median']:.0f} "
                "candidate triples versus "
                f"{summary['by_length_class']['target_7_12']['candidate_triples']['median']:.0f} "
                "in the 7-12 group. Its median precision is lower by "
                f"{summary['by_length_class']['target_7_12']['precision']['median'] - summary['by_length_class']['over_12']['precision']['median']:.4f}, "
                "while median recall is lower by only "
                f"{summary['by_length_class']['target_7_12']['recall']['median'] - summary['by_length_class']['over_12']['recall']['median']:.4f}. "
                "Together with the graph-excess correlations, this is "
                "consistent with whole-candidate-graph precision dilution."
            ),
            "",
            "## Interpretation boundary",
            "",
            f"Decision: `{summary['decision']}`.",
            "",
            (
                "This diagnostic can measure the gate's direct filtering of "
                "already observed candidates and whether graph-size changes "
                "co-occur with precision dilution. It cannot establish that "
                "the appended prompt caused those patterns, compare DeepSeek "
                "with Claude, show that the model is irrelevant, or estimate "
                "an ungated document completion rate. Accepting an earlier "
                "rejected candidate would have changed later adaptive "
                "attempts and slots. The reported correlations and medians "
                "are descriptive."
            ),
            "",
            (
                "`swan_positive_deepseek_length_confirmation3` was not "
                "executed."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_manifest(inventory: dict, output_hashes: dict) -> dict:
    source_manifest = load_json(SOURCE_MANIFEST)
    frozen_dependencies = source_manifest["dependencies"]
    expected_runtime_hash = frozen_dependencies[
        "third_party/swan_s2match_runtime_711a231d/provenance.json"
    ]
    expected_vector_hash = frozen_dependencies[
        "vectors/glove.6B.100d.txt"
    ]
    if sha256(S2MATCH_PROVENANCE) != expected_runtime_hash:
        raise ValueError("S2Match provenance differs from source experiment")
    if sha256(VECTORS) != expected_vector_hash:
        raise ValueError("GloVe vectors differ from source experiment")
    return {
        "schema_version": 1,
        "status": "complete_retrospective_offline_diagnostic",
        "protocol_sha256": sha256(PROTOCOL),
        "source_experiment": str(SOURCE.relative_to(ROOT)),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "source_attempt_inventory": inventory,
        "method": {
            "argument_order": "(candidate_template_amr, target_template)",
            "threshold": THRESHOLD,
            "absolute_reproduction_tolerance": ABS_TOLERANCE,
            "score_seed": (
                "run_swan_positive_collection.score_seed("
                "allocation_id, slot, attempt)"
            ),
            "vectors_loaded_once_read_only": True,
            "unique_candidate_s2match_observations": EXPECTED_CANDIDATES,
            "final_materialization_s2match_computations": (
                EXPECTED_CANDIDATES
            ),
            "new_model_calls": 0,
            "new_parser_calls": 0,
            "new_network_calls": 0,
        },
        "dependencies": {
            str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__)),
            str(PROTOCOL.relative_to(ROOT)): sha256(PROTOCOL),
            str(
                Path(swan.__file__).resolve().relative_to(ROOT)
            ): sha256(Path(swan.__file__).resolve()),
            str(S2MATCH_FILE.relative_to(ROOT)): sha256(S2MATCH_FILE),
            str(S2MATCH_PROVENANCE.relative_to(ROOT)): sha256(
                S2MATCH_PROVENANCE
            ),
            str(VECTORS.relative_to(ROOT)): sha256(VECTORS),
        },
        "packages": package_versions(),
        "outputs": output_hashes,
    }


def write_attempts(rows: list[dict]) -> None:
    write_text(
        ATTEMPTS,
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n"
            for row in rows
        ),
    )


def run() -> dict:
    outputs = (ATTEMPTS, SUMMARY, REPORT, MANIFEST)
    existing = [path.exists() for path in outputs]
    if any(existing):
        if not all(existing):
            raise RuntimeError("Partial diagnostic output exists")
        return check()
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    inventory_rows, inventory = attempt_inventory()
    engine = S2MatchEngine()
    rows = []
    for index, (path, source_row) in enumerate(inventory_rows, start=1):
        rows.append(
            enrich_attempt(
                path,
                source_row,
                engine.metric,
                engine.graph_counts,
            )
        )
        if index % 50 == 0 or index == len(inventory_rows):
            print(
                json.dumps(
                    {
                        "stage": "offline_s2match",
                        "completed": index,
                        "total": len(inventory_rows),
                    }
                ),
                flush=True,
            )
    summary = build_summary(rows, inventory)
    if not summary["historical_reproduction"][
        "all_exact_within_tolerance"
    ]:
        raise ValueError("Historical S2Match reproduction failed")
    write_attempts(rows)
    write_json(SUMMARY, summary)
    write_text(REPORT, build_report(summary))
    manifest = build_manifest(
        inventory,
        {
            str(ATTEMPTS.relative_to(ROOT)): sha256(ATTEMPTS),
            str(SUMMARY.relative_to(ROOT)): sha256(SUMMARY),
            str(REPORT.relative_to(ROOT)): sha256(REPORT),
        },
    )
    write_json(MANIFEST, manifest)
    return summary


def check() -> dict:
    for path in (PROTOCOL, ATTEMPTS, SUMMARY, REPORT, MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = load_json(MANIFEST)
    if manifest["status"] != "complete_retrospective_offline_diagnostic":
        raise ValueError("Unexpected manifest status")
    for relative, expected in manifest["dependencies"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Frozen dependency changed: {relative}")
    for relative, expected in manifest["outputs"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Diagnostic output changed: {relative}")
    if package_versions() != manifest["packages"]:
        raise ValueError("Diagnostic package versions changed")
    _, inventory = attempt_inventory()
    if inventory != manifest["source_attempt_inventory"]:
        raise ValueError("Source attempt inventory changed")
    summary = load_json(SUMMARY)
    if (
        summary["integrity"]["source_traces"] != EXPECTED_TRACES
        or summary["integrity"]["candidate_attempts"]
        != EXPECTED_CANDIDATES
        or summary["integrity"]["censored_uncertain_traces"]
        != EXPECTED_TRACES - EXPECTED_CANDIDATES
        or not summary["historical_reproduction"][
            "all_exact_within_tolerance"
        ]
    ):
        raise ValueError("Diagnostic summary integrity failed")
    attempt_rows = [
        json.loads(line)
        for line in ATTEMPTS.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(attempt_rows) != EXPECTED_CANDIDATES:
        raise ValueError("Attempt output row count changed")
    if len({row["call_id"] for row in attempt_rows}) != EXPECTED_CANDIDATES:
        raise ValueError("Attempt output identities are not unique")
    source_hashes = {
        item["path"]: item["sha256"]
        for item in inventory["files"]
    }
    if any(
        source_hashes.get(row["source_trace"])
        != row["source_trace_sha256"]
        for row in attempt_rows
    ):
        raise ValueError("Attempt output source lineage changed")
    rebuilt_summary = build_summary(attempt_rows, inventory)
    if rebuilt_summary != summary:
        raise ValueError("Diagnostic summary does not reconstruct")
    if build_report(summary) != REPORT.read_text(encoding="utf-8"):
        raise ValueError("Diagnostic report does not reconstruct")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "check"))
    args = parser.parse_args()
    result = run() if args.stage == "run" else check()
    print(
        json.dumps(
            {
                "status": result["status"],
                "source_traces": result["integrity"]["source_traces"],
                "candidate_attempts": result["integrity"][
                    "candidate_attempts"
                ],
                "historical_reproduction": result[
                    "historical_reproduction"
                ]["all_exact_within_tolerance"],
                "gate_rejected_threshold_passes": result["gate_effect"][
                    "gate_rejected_threshold_passes"
                ],
                "decision": result["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
