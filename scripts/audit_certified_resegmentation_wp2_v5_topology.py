"""Run the model-free native-span and reachable-pair gate for WP2-v5."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

from audit_certified_resegmentation_wp2_v2_design import (
    NATIVE_TOKEN_LIMIT,
    SYSTEMS,
    load_dataset,
    load_tokenizer,
    official_tokenizer_text,
    tokenizer_manifest,
    untruncated_token_count,
)
from certified_resegmentation_wp2_adapter import hash_raw_value
from certified_resegmentation_wp2_v3 import (
    SparsePairwiseProblem,
    canonicalize_structured,
)
from certified_resegmentation_wp2_v4 import (
    exact_grouped_sparse_pairwise_certificate,
)


PROTOCOL_SHA256 = (
    "1af73c61817afce878d88ce29c5f626a981b2382af92ded634331cf9663c2c24"
)
SELECTION_SHA256 = (
    "413bdd37780b9c3e9e41807fcf80b2e8843951fb26af4570d81312544649e399"
)
SPAN_PER_DOCUMENT_CAP = 250_000
PAIR_PER_DOCUMENT_CAP = 5_000_000
SPAN_PER_SYSTEM_CAP = 1_000_000
PAIR_PER_SYSTEM_CAP = 20_000_000
CACHE_CAP_BYTES = 50 * 1024 ** 3
ESTIMATED_PAIR_BYTES = 64
ESTIMATED_FEATURE_BYTES = {
    "semstamp": 128,
    "simmark": 3_200,
}


class TopologyCapExceeded(RuntimeError):
    def __init__(self, gate, observed_lower_bound):
        super().__init__(
            f"{gate} exceeded with lower bound {observed_lower_bound}"
        )
        self.gate = gate
        self.observed_lower_bound = observed_lower_bound


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def git_head(path):
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def audit_document(document, system_id, tokenizer):
    span_calls = 0
    legal_spans = 0
    legal_input_tokens = 0
    feature_token_counts = {}
    pair_calls = 0

    def span_allowed(start, end):
        nonlocal span_calls, legal_spans, legal_input_tokens
        span_calls += 1
        if span_calls > SPAN_PER_DOCUMENT_CAP:
            raise TopologyCapExceeded(
                "spans_per_document",
                span_calls,
            )
        rendered = document.preprocess_unit(system_id, start, end)
        tokenizer_text = official_tokenizer_text(
            system_id,
            rendered,
            int(start > 0),
        )
        token_count = untruncated_token_count(
            tokenizer,
            tokenizer_text,
        )
        allowed = token_count <= NATIVE_TOKEN_LIMIT
        if allowed:
            legal_spans += 1
            legal_input_tokens += token_count
            text_sha = sha256_bytes(rendered.encode("utf-8"))
            previous = feature_token_counts.setdefault(
                text_sha,
                token_count,
            )
            if previous != token_count:
                raise AssertionError("feature-text SHA256 collision")
        return allowed

    def count_pair(*_coordinates):
        nonlocal pair_calls
        if len(_coordinates) != 3:
            raise AssertionError("pair scorer received the wrong arity")
        pair_calls += 1
        if pair_calls > PAIR_PER_DOCUMENT_CAP:
            raise TopologyCapExceeded(
                "pairs_per_document",
                pair_calls,
            )
        return 0.0

    started = time.perf_counter()
    result = exact_grouped_sparse_pairwise_certificate(
        SparsePairwiseProblem(
            token_count=document.atom_count,
            original_boundaries=document.original_boundaries,
            max_boundary_moves=min(3, document.unit_count - 1),
            span_allowed=span_allowed,
            pair_scorer=count_pair,
        )
    )
    elapsed = time.perf_counter() - started
    if result.unique_span_checks != span_calls:
        raise AssertionError("span callback count differs from v4 result")
    if result.unique_pair_scores != pair_calls:
        raise AssertionError("pair callback count differs from v4 result")
    return {
        "completed": True,
        "elapsed_seconds": elapsed,
        "unique_span_checks": span_calls,
        "legal_reachable_spans": legal_spans,
        "unique_feature_texts": len(feature_token_counts),
        "legal_span_input_tokens": legal_input_tokens,
        "deduplicated_feature_input_tokens": sum(
            feature_token_counts.values()
        ),
        "reachable_pair_keys": pair_calls,
        "candidate_boundary_positions": (
            result.candidate_boundary_positions
        ),
        "reachable_state_count": result.reachable_state_count,
        "transitions_evaluated": result.transitions_evaluated,
        "peak_layer_states": result.peak_layer_states,
        "threat_set_size": result.threat_set_size,
    }


def main():
    import nltk
    from nltk.tokenize import sent_tokenize

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    os.environ["NLTK_DATA"] = str(args.nltk_data)
    nltk.data.path.insert(0, str(args.nltk_data))
    config = SYSTEMS[args.system]
    if nltk.__version__ != config["nltk_version"]:
        raise ValueError("NLTK version mismatch")

    protocol_bytes = args.protocol.read_bytes()
    selection_bytes = args.selection.read_bytes()
    if sha256_bytes(protocol_bytes) != PROTOCOL_SHA256:
        raise ValueError("WP2-v5 protocol SHA256 mismatch")
    if sha256_bytes(selection_bytes) != SELECTION_SHA256:
        raise ValueError("WP2-v5 selection SHA256 mismatch")
    selection = json.loads(selection_bytes)
    selected = selection["systems"][args.system]["documents"]
    if len(selected) != 12:
        raise ValueError("selection must contain exactly 12 documents")

    data_root = args.root / config["data_repo_dir"]
    revision = git_head(data_root)
    if revision != config["data_revision"]:
        raise ValueError("data revision mismatch")
    dataset = load_dataset(data_root / config["null_path"])
    tokenizer_files = tokenizer_manifest(args.model_dir, args.system)
    tokenizer = load_tokenizer(args.model_dir, args.system)

    started = time.perf_counter()
    documents = []
    totals = {
        "unique_span_checks": 0,
        "legal_reachable_spans": 0,
        "unique_feature_texts_upper_bound": 0,
        "deduplicated_feature_input_tokens_upper_bound": 0,
        "reachable_pair_keys": 0,
    }
    status = "pass"
    stop_reason = None
    for frozen in selected:
        raw_value = dataset[frozen["row_index"]]["text"]
        if hash_raw_value(raw_value) != frozen["raw_text_sha256"]:
            raise ValueError("selected raw-value SHA256 mismatch")
        document = canonicalize_structured(
            raw_value,
            sentence_splitter=(
                sent_tokenize if isinstance(raw_value, str) else None
            ),
        )
        if (
            document.structural_tape_sha256
            != frozen["structural_tape_sha256"]
            or document.atom_count != frozen["atom_count"]
            or document.unit_count != frozen["unit_count"]
        ):
            raise ValueError("selected structural identity mismatch")

        identity = {
            "selection_slot": frozen["selection_slot"],
            "row_index": frozen["row_index"],
            "raw_text_sha256": frozen["raw_text_sha256"],
            "structural_tape_sha256": (
                frozen["structural_tape_sha256"]
            ),
            "atom_count": document.atom_count,
            "unit_count": document.unit_count,
            "contains_raw_text": False,
        }
        try:
            topology = audit_document(document, args.system, tokenizer)
        except TopologyCapExceeded as error:
            topology = {
                "completed": False,
                "stop_gate": error.gate,
                "observed_lower_bound": error.observed_lower_bound,
            }
            identity["topology"] = topology
            documents.append(identity)
            status = "stop"
            stop_reason = str(error)
            break
        identity["topology"] = topology
        documents.append(identity)

        for key in (
            "unique_span_checks",
            "legal_reachable_spans",
            "reachable_pair_keys",
        ):
            totals[key] += topology[key]
        totals["unique_feature_texts_upper_bound"] += topology[
            "unique_feature_texts"
        ]
        totals[
            "deduplicated_feature_input_tokens_upper_bound"
        ] += topology["deduplicated_feature_input_tokens"]
        if totals["unique_span_checks"] > SPAN_PER_SYSTEM_CAP:
            status = "stop"
            stop_reason = "spans_per_system exceeded"
            break
        if totals["reachable_pair_keys"] > PAIR_PER_SYSTEM_CAP:
            status = "stop"
            stop_reason = "pairs_per_system exceeded"
            break

    estimated_cache_bytes = (
        totals["reachable_pair_keys"] * ESTIMATED_PAIR_BYTES
        + totals["unique_feature_texts_upper_bound"]
        * ESTIMATED_FEATURE_BYTES[args.system]
    )
    gates = {
        "all_12_documents_completed": len(documents) == 12 and all(
            row["topology"]["completed"] for row in documents
        ),
        "spans_per_system_le_1000000": (
            totals["unique_span_checks"] <= SPAN_PER_SYSTEM_CAP
        ),
        "pairs_per_system_le_20000000": (
            totals["reachable_pair_keys"] <= PAIR_PER_SYSTEM_CAP
        ),
        "estimated_cache_le_50_gib": (
            estimated_cache_bytes <= CACHE_CAP_BYTES
        ),
    }
    if not all(gates.values()):
        status = "stop"
    output = {
        "schema_version": 1,
        "created_on": "2026-09-22",
        "audit_mode": "wp2_v5_model_free_topology",
        "system": args.system,
        "status": status,
        "stop_reason": stop_reason,
        "parameters": {
            "max_boundary_moves": 3,
            "native_token_limit": NATIVE_TOKEN_LIMIT,
            "span_per_document_cap": SPAN_PER_DOCUMENT_CAP,
            "pair_per_document_cap": PAIR_PER_DOCUMENT_CAP,
            "span_per_system_cap": SPAN_PER_SYSTEM_CAP,
            "pair_per_system_cap": PAIR_PER_SYSTEM_CAP,
            "cache_cap_bytes": CACHE_CAP_BYTES,
            "estimated_pair_bytes": ESTIMATED_PAIR_BYTES,
            "estimated_feature_bytes": (
                ESTIMATED_FEATURE_BYTES[args.system]
            ),
        },
        "inputs": {
            "protocol_sha256": sha256_bytes(protocol_bytes),
            "selection_sha256": sha256_bytes(selection_bytes),
            "data_revision": revision,
            "tokenizer_files": tokenizer_files,
        },
        "documents": documents,
        "totals": {
            **totals,
            "estimated_cache_bytes": estimated_cache_bytes,
        },
        "gates": gates,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_bytes": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "nltk": nltk.__version__,
            "tokenizer_class": type(tokenizer).__name__,
        },
        "execution_attestation": {
            "model_loads": 0,
            "tokenizer_loads": 1,
            "model_score_reads": 0,
            "calibration_score_reads": 0,
            "test_score_reads": 0,
            "positive_score_reads": 0,
            "watermark_generation_calls": 0,
            "provider_api_calls": 0,
        },
        "contains_raw_text": False,
    }
    encoded = (
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("ascii")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(json.dumps({
        "status": status,
        "stop_reason": stop_reason,
        "documents_completed": sum(
            row["topology"]["completed"] for row in documents
        ),
        "totals": output["totals"],
        "gates": gates,
        "output_sha256": sha256_bytes(encoded),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
