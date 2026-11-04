"""Audit one frozen system under the WP2-v3 structural contract."""

import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import resource
import signal
import time

from audit_certified_resegmentation_wp2_v2_design import (
    SYSTEMS,
    frozen_identities,
    nearest_rank_quantiles,
    official_tokenizer_text,
    tokenizer_manifest,
    untruncated_token_count,
    write_jsonl,
    load_tokenizer,
)
from certified_resegmentation_wp2_v3 import (
    SparsePairwiseProblem,
    canonicalize_structured,
    exact_sparse_pairwise_certificate,
)


PRIMARY_CANDIDATE = "STRUCTURAL_SEPARATOR_NATIVE_512"
STRICT_COMPARATOR = "STRICT_NATIVE_512"
NATIVE_TOKEN_LIMIT = 512
SCALE_TIMEOUT_SECONDS = 70
SCALE_WALL_GATE_SECONDS = 60
SCALE_RSS_GATE_BYTES = 4 * 1024 ** 3
SCALE_SPAN_GATE = 2_000_000
SCALE_PAIR_GATE = 50_000_000


def edge_whitespace_counts(units):
    leading = sum(unit != unit.lstrip() for unit in units)
    trailing = sum(unit != unit.rstrip() for unit in units)
    missing_cross = sum(
        left == left.rstrip() and right == right.lstrip()
        for left, right in zip(units, units[1:])
    )
    return leading, trailing, missing_cross


def audit_identity(item, system_id, count_tokens, sentence_splitter):
    record = item["record"]
    output = {
        "cohort": item["cohort"],
        "role": item["role"],
        "row_index": record.row_index,
        "raw_text_sha256": record.raw_text_sha256,
        "raw_field_type": record.raw_field_type,
        "contains_raw_text": False,
    }
    try:
        document = canonicalize_structured(
            item["raw_value"],
            sentence_splitter=(
                sentence_splitter
                if isinstance(item["raw_value"], str)
                else None
            ),
        )
    except (TypeError, ValueError) as error:
        output.update({
            "structural_tape_sha256": None,
            "unit_count": None,
            "atom_count": None,
            "max_native_tokens": None,
            "units_with_leading_whitespace": None,
            "units_with_trailing_whitespace": None,
            "missing_cross_unit_whitespace": None,
            "candidate_eligible": {
                PRIMARY_CANDIDATE: False,
                STRICT_COMPARATOR: False,
            },
            "candidate_exclusion_codes": {
                PRIMARY_CANDIDATE: ["invalid_p0"],
                STRICT_COMPARATOR: ["invalid_p0"],
            },
            "p0_error_type": type(error).__name__,
        })
        return output, None

    token_counts = [
        count_tokens(official_tokenizer_text(
            system_id, unit, index
        ))
        for index, unit in enumerate(document.original_units)
    ]
    primary_codes = []
    if document.unit_count < 2:
        primary_codes.append("unit_count_below_2")
    if max(token_counts) > NATIVE_TOKEN_LIMIT:
        primary_codes.append("unit_above_512_native_tokens")
    strict_codes = list(primary_codes)
    if any(unit != unit.strip() for unit in document.original_units):
        strict_codes.append("unit_edge_whitespace")

    leading, trailing, missing_cross = edge_whitespace_counts(
        document.original_units
    )
    output.update({
        "structural_tape_sha256": document.structural_tape_sha256,
        "unit_count": document.unit_count,
        "atom_count": document.atom_count,
        "max_native_tokens": max(token_counts),
        "units_with_leading_whitespace": leading,
        "units_with_trailing_whitespace": trailing,
        "missing_cross_unit_whitespace": missing_cross,
        "candidate_eligible": {
            PRIMARY_CANDIDATE: not primary_codes,
            STRICT_COMPARATOR: not strict_codes,
        },
        "candidate_exclusion_codes": {
            PRIMARY_CANDIDATE: sorted(set(primary_codes)),
            STRICT_COMPARATOR: sorted(set(strict_codes)),
        },
        "p0_error_type": None,
    })
    if document.render_partition(
        document.original_boundaries
    ) != document.original_units:
        raise AssertionError("structured adapter failed k=0")
    if document.partition_atoms(
        document.original_boundaries
    ) != document.atom_payloads:
        raise AssertionError("structured adapter changed P0 atoms")
    return output, document


def summarize(rows):
    output = {}
    for cohort in (
        "detector_fit",
        "certificate_calibration",
        "independent_test",
        "positive",
    ):
        selected = [row for row in rows if row["cohort"] == cohort]
        candidates = {}
        for candidate in (PRIMARY_CANDIDATE, STRICT_COMPARATOR):
            reasons = Counter(
                code
                for row in selected
                for code in row["candidate_exclusion_codes"][candidate]
            )
            candidates[candidate] = {
                "eligible": sum(
                    row["candidate_eligible"][candidate] for row in selected
                ),
                "excluded": sum(
                    not row["candidate_eligible"][candidate]
                    for row in selected
                ),
                "exclusion_code_counts": dict(sorted(reasons.items())),
            }
        analyzable = [
            row for row in selected if row["atom_count"] is not None
        ]
        output[cohort] = {
            "selected": len(selected),
            "candidates": candidates,
            "strict_only_attrition": sum(
                row["candidate_eligible"][PRIMARY_CANDIDATE]
                and not row["candidate_eligible"][STRICT_COMPARATOR]
                for row in selected
            ),
            "raw_field_type_counts": dict(sorted(Counter(
                row["raw_field_type"] for row in selected
            ).items())),
            "distribution": {
                key: nearest_rank_quantiles([
                    row[key] for row in analyzable
                    if row[key] is not None
                ])
                for key in (
                    "atom_count",
                    "unit_count",
                    "max_native_tokens",
                    "missing_cross_unit_whitespace",
                )
            },
        }
    return output


def primary_gates(system_id, cohorts):
    gates = {
        "certificate_calibration_eligible_ge_399": (
            cohorts["certificate_calibration"]["candidates"][
                PRIMARY_CANDIDATE
            ]["eligible"] >= 399
        ),
        "independent_test_eligible_ge_400": (
            cohorts["independent_test"]["candidates"][
                PRIMARY_CANDIDATE
            ]["eligible"] >= 400
        ),
        "positive_eligible_ge_200": (
            cohorts["positive"]["candidates"][
                PRIMARY_CANDIDATE
            ]["eligible"] >= 200
        ),
    }
    if system_id == "simmark":
        gates["detector_fit_eligible_ge_200"] = (
            cohorts["detector_fit"]["candidates"][
                PRIMARY_CANDIDATE
            ]["eligible"] >= 200
        )
    return gates


def synthetic_pair_score(left, middle, right):
    return float(
        ((left + 3) * 31 + (middle + 5) * 17 + (right + 7) * 13)
        % 97
    )


def integration_audit(rows_and_documents, system_id, count_tokens):
    selected = []
    for cohort in (
        "detector_fit",
        "certificate_calibration",
        "independent_test",
        "positive",
    ):
        eligible = [
            (row, document)
            for row, document in rows_and_documents
            if (
                row["cohort"] == cohort
                and row["candidate_eligible"][PRIMARY_CANDIDATE]
            )
        ]
        selected.extend(sorted(
            eligible,
            key=lambda pair: (
                pair[0]["atom_count"],
                pair[0]["raw_text_sha256"],
            ),
        )[:5])

    results = []
    for row, document in selected:
        def span_allowed(start, end):
            rendered = document.preprocess_unit(
                system_id, start, end
            )
            text = official_tokenizer_text(
                system_id, rendered, int(start > 0)
            )
            return count_tokens(text) <= NATIVE_TOKEN_LIMIT

        result = exact_sparse_pairwise_certificate(
            SparsePairwiseProblem(
                token_count=document.atom_count,
                original_boundaries=document.original_boundaries,
                max_boundary_moves=min(
                    3, document.unit_count - 1
                ),
                span_allowed=span_allowed,
                pair_scorer=synthetic_pair_score,
            )
        )
        results.append({
            "cohort": row["cohort"],
            "row_index": row["row_index"],
            "raw_text_sha256": row["raw_text_sha256"],
            "atom_count": document.atom_count,
            "unit_count": document.unit_count,
            "threat_set_size": result.threat_set_size,
            "reachable_state_count": result.reachable_state_count,
            "transitions_evaluated": result.transitions_evaluated,
            "unique_span_checks": result.unique_span_checks,
            "unique_pair_scores": result.unique_pair_scores,
        })
    return {
        "selection": "five smallest eligible documents per cohort",
        "documents": len(results),
        "all_nonempty": all(
            result["threat_set_size"] >= 1 for result in results
        ),
        "results": results,
    }


def scale_audit(document):
    started = time.perf_counter()
    previous_handler = signal.getsignal(signal.SIGALRM)

    def timeout_handler(_signum, _frame):
        raise TimeoutError("sparse scale audit exceeded hard timeout")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(SCALE_TIMEOUT_SECONDS)
    try:
        result = exact_sparse_pairwise_certificate(
            SparsePairwiseProblem(
                token_count=document.atom_count,
                original_boundaries=document.original_boundaries,
                max_boundary_moves=3,
                span_allowed=lambda start, end: True,
                pair_scorer=synthetic_pair_score,
            )
        )
        elapsed = time.perf_counter() - started
        peak_rss_bytes = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        gates = {
            "wall_seconds_le_60": elapsed <= SCALE_WALL_GATE_SECONDS,
            "peak_rss_le_4_gib": peak_rss_bytes <= SCALE_RSS_GATE_BYTES,
            "unique_spans_le_2000000": (
                result.unique_span_checks <= SCALE_SPAN_GATE
            ),
            "unique_pairs_le_50000000": (
                result.unique_pair_scores <= SCALE_PAIR_GATE
            ),
        }
        return {
            "status": "pass" if all(gates.values()) else "stop",
            "completed": True,
            "elapsed_seconds": elapsed,
            "peak_rss_bytes": peak_rss_bytes,
            "token_count": document.atom_count,
            "unit_count": document.unit_count,
            "gates": gates,
            "result": result.to_dict(),
        }
    except TimeoutError as error:
        elapsed = time.perf_counter() - started
        peak_rss_bytes = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        return {
            "status": "stop",
            "completed": False,
            "elapsed_seconds": elapsed,
            "peak_rss_bytes": peak_rss_bytes,
            "token_count": document.atom_count,
            "unit_count": document.unit_count,
            "gates": {
                "wall_seconds_le_60": False,
                "peak_rss_le_4_gib": (
                    peak_rss_bytes <= SCALE_RSS_GATE_BYTES
                ),
                "unique_spans_le_2000000": None,
                "unique_pairs_le_50000000": None,
            },
            "error": str(error),
            "result": None,
        }
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def main():
    import nltk
    from nltk.tokenize import sent_tokenize

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-manifest", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    args = parser.parse_args()

    config = SYSTEMS[args.system]
    if nltk.__version__ != config["nltk_version"]:
        raise ValueError("NLTK version mismatch")
    nltk.data.path.insert(0, str(args.nltk_data))
    tokenizer_files = tokenizer_manifest(args.model_dir, args.system)
    tokenizer = load_tokenizer(args.model_dir, args.system)
    tokenizer_calls = 0

    def count_tokens(text):
        nonlocal tokenizer_calls
        tokenizer_calls += 1
        return untruncated_token_count(tokenizer, text)

    identities, split, revision = frozen_identities(
        args.root, args.system
    )
    rows = []
    rows_and_documents = []
    largest_document = None
    for item in identities:
        row, document = audit_identity(
            item, args.system, count_tokens, sent_tokenize
        )
        rows.append(row)
        rows_and_documents.append((row, document))
        if (
            document is not None
            and row["candidate_eligible"][PRIMARY_CANDIDATE]
            and (
                largest_document is None
                or (
                    document.atom_count,
                    document.unit_count,
                ) > (
                    largest_document.atom_count,
                    largest_document.unit_count,
                )
            )
        ):
            largest_document = document

    cohorts = summarize(rows)
    gates = primary_gates(args.system, cohorts)
    integration = integration_audit(
        rows_and_documents, args.system, count_tokens
    )
    if largest_document is None:
        raise ValueError("no eligible document for scale audit")
    scale = scale_audit(largest_document)
    status = (
        "algorithm_ready"
        if all(gates.values())
        and integration["all_nonempty"]
        and scale["status"] == "pass"
        else "stop"
    )

    args.row_manifest.parent.mkdir(parents=True, exist_ok=True)
    row_sha256 = write_jsonl(args.row_manifest, rows)
    output = {
        "schema_version": 1,
        "created_on": "2026-09-22",
        "audit_mode": "wp2_v3_structural_readiness",
        "system_id": args.system,
        "status": status,
        "data_revision": revision,
        "primary_candidate": PRIMARY_CANDIDATE,
        "strict_comparator": STRICT_COMPARATOR,
        "parameters": {
            "native_token_limit": NATIVE_TOKEN_LIMIT,
            "virtual_separator": "one ASCII space",
            "max_boundary_moves": 3,
            "scale_timeout_seconds": SCALE_TIMEOUT_SECONDS,
            "scale_wall_gate_seconds": SCALE_WALL_GATE_SECONDS,
            "scale_rss_gate_bytes": SCALE_RSS_GATE_BYTES,
            "scale_span_gate": SCALE_SPAN_GATE,
            "scale_pair_gate": SCALE_PAIR_GATE,
        },
        "identity_attrition": {
            "null_input_count": split.attrition.null_input_count,
            "null_unique_count": split.attrition.null_unique_count,
            "null_duplicate_count": split.attrition.null_duplicate_count,
            "positive_input_count": split.attrition.positive_input_count,
            "positive_retained_count": (
                split.attrition.positive_retained_count
            ),
            "positive_overlap_count": split.attrition.positive_overlap_count,
        },
        "cohorts": cohorts,
        "primary_gates": gates,
        "native_span_integration": integration,
        "scale_audit": scale,
        "row_manifest": {
            "path": args.row_manifest.name,
            "rows": len(rows),
            "sha256": row_sha256,
            "contains_raw_text": False,
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "model_max_length": tokenizer.model_max_length,
            "local_files_only": True,
            "file_sha256": tokenizer_files,
            "calls": tokenizer_calls,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "nltk": nltk.__version__,
        },
        "execution_attestation": {
            "model_loads": 0,
            "watermark_detector_calls": 0,
            "detector_score_reads": 0,
            "watermark_generation_calls": 0,
            "provider_api_calls": 0,
            "tokenizer_loads": 1,
        },
        "contains_raw_text": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "system_id": args.system,
        "status": status,
        "primary_gates": gates,
        "integration_documents": integration["documents"],
        "scale_status": scale["status"],
        "scale_completed": scale["completed"],
        "scale_elapsed_seconds": scale["elapsed_seconds"],
        "row_manifest_sha256": row_sha256,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
