"""Run the preregistered count-neutral SWAN A/B confirmation.

Preparation is detector blind. Detection and mechanism stages are local and
never read an API credential.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
from typing import Callable, Optional
import unicodedata

import prepare_swan_positive_collection as allocator
from run_segmentation_intervention import amr_body
import run_swan_amr_composition_decomposition as composition
import run_swan_positive_collection as swan
import run_swan_positive_deepseek_confirmation as positive


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/swan_count_neutral_confirmation1"
PROTOCOL = OUTPUT / "protocol.md"
PAIR_MANIFEST = OUTPUT / "pair_manifest.json"
CONTROL_SNAPSHOT = OUTPUT / "control_prior_identity_snapshot.json"
CONTROL_MANIFEST = OUTPUT / "control_manifest.json"
CALIBRATION = OUTPUT / "calibration.json"
TEST_RESULT = OUTPUT / "test_result.json"
SUMMARY = OUTPUT / "summary.json"
REPORT = OUTPUT / "report.md"

DATASET_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
CONTROL_SHARD = 10
SHARD_COUNT = 512
CALIBRATION_CONTROLS = 250
TEST_CONTROLS = 250
SOURCE_SENTENCES = 16
DETECTOR_UNITS = 15
MIN_WORDS = 100
MAX_WORDS = 220
GREEN_THRESHOLD = 0.65
NULL_GREEN_RATE = 0.25
PUBLIC_Z_THRESHOLD = 2.33
CALIBRATION_FPR_TARGET = 0.01
TEST_FPR_CEILING = 0.05
MECHANISM_REPEATS = 5
BOUNDARY_SALT = "swan-count-neutral-confirmation1:boundaries"
TOKEN_RE = positive.WORD_RE
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "expl"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def freeze_json(path: Path, payload) -> dict:
    if path.exists():
        saved = load_json(path)
        if saved != payload:
            raise ValueError(f"Frozen artifact differs: {path}")
        return saved
    write_json(path, payload)
    return payload


def lexical_sequence(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def punctuation_multiset(text: str) -> list[list[object]]:
    counts = Counter(
        character
        for character in text
        if unicodedata.category(character).startswith("P")
    )
    return [[key, counts[key]] for key in sorted(counts)]


def sentence_profile(sentence: str, nlp) -> dict:
    doc = nlp(sentence)
    roots = [
        token
        for token in doc
        if token.dep_ == "ROOT" and token.pos_ in {"VERB", "AUX"}
    ]
    subjects = [
        token
        for token in doc
        if token.dep_ in SUBJECT_DEPS
    ]
    return {
        "finite_root": bool(roots),
        "explicit_subject": bool(subjects),
        "root_lemmas": sorted({token.lemma_.casefold() for token in roots}),
        "subject_lemmas": sorted(
            {token.lemma_.casefold() for token in subjects}
        ),
    }


def first_alpha(text: str) -> str:
    return next((character for character in text if character.isalpha()), "")


def merge_units(sentences: list[str], boundary: int) -> dict:
    if boundary not in range(len(sentences) - 1):
        raise ValueError("Boundary index is out of range")
    left = sentences[boundary]
    right = sentences[boundary + 1]
    if not left.endswith(".") or left.endswith(".."):
        raise ValueError("Merge boundary is not period final")
    merged = left[:-1] + "; " + right
    units = []
    sources = []
    index = 0
    while index < len(sentences):
        if index == boundary:
            units.append(merged)
            sources.append([index, index + 1])
            index += 2
        else:
            units.append(sentences[index])
            sources.append([index])
            index += 1
    return {
        "merge_boundary": boundary,
        "units": units,
        "unit_source_sentence_indices": sources,
        "text": " ".join(units),
    }


def usable_boundaries(
    sentences: list[str],
    nlp,
    tokenize: Callable[[str], list[str]],
) -> list[dict]:
    profiles = [sentence_profile(sentence, nlp) for sentence in sentences]
    candidates = []
    for index, (left, right) in enumerate(zip(sentences, sentences[1:])):
        mergeable_period = left.endswith(".") and not left.endswith("..")
        merged = (
            merge_units(sentences, index)
            if mergeable_period
            else None
        )
        right_initial = first_alpha(right)
        checks = {
            "single_terminal_period": mergeable_period,
            "right_initial_uppercase": bool(
                right_initial and right_initial.isupper()
            ),
            "minimum_four_words_each": (
                len(lexical_sequence(left)) >= 4
                and len(lexical_sequence(right)) >= 4
            ),
            "finite_roots": (
                profiles[index]["finite_root"]
                and profiles[index + 1]["finite_root"]
            ),
            "explicit_subjects": (
                profiles[index]["explicit_subject"]
                and profiles[index + 1]["explicit_subject"]
            ),
            "tokenizer_merges_pair": bool(
                merged is not None
                and len(tokenize(merged["text"])) == len(sentences) - 1
                and tokenize(merged["text"]) == merged["units"]
            ),
        }
        if all(checks.values()):
            candidates.append(
                {
                    "boundary_index": index,
                    "left_profile": profiles[index],
                    "right_profile": profiles[index + 1],
                    "checks": checks,
                }
            )
    return candidates


def select_boundary_pair(
    allocation_id: str,
    text_hash: str,
    boundaries: list[dict],
) -> dict:
    indices = [row["boundary_index"] for row in boundaries]
    pairs = [
        [left, right]
        for position, left in enumerate(indices)
        for right in indices[position + 1 :]
        if abs(left - right) > 1
    ]
    if not pairs:
        raise ValueError("No two nonadjacent usable boundaries")
    selector = int(
        hashlib.sha256(
            f"{BOUNDARY_SALT}:{allocation_id}:{text_hash}".encode("utf-8")
        ).hexdigest(),
        16,
    )
    selected = pairs[selector % len(pairs)]
    if (selector // len(pairs)) % 2:
        selected = list(reversed(selected))
    return {
        "candidate_pairs": len(pairs),
        "selected_unordered_pair": sorted(selected),
        "a": selected[0],
        "b": selected[1],
        "selector_sha256": hashlib.sha256(
            f"{BOUNDARY_SALT}:{allocation_id}:{text_hash}".encode("utf-8")
        ).hexdigest(),
    }


def build_variants(
    allocation_id: str,
    sentences: list[str],
    nlp,
    tokenize: Callable[[str], list[str]],
) -> dict:
    if len(sentences) != SOURCE_SENTENCES:
        raise ValueError("Expected exactly 16 source sentences")
    original = " ".join(sentences)
    if tokenize(original) != sentences:
        raise ValueError("Tokenizer does not recover source sentences")
    boundaries = usable_boundaries(sentences, nlp, tokenize)
    selection = select_boundary_pair(
        allocation_id,
        text_sha256(original),
        boundaries,
    )
    variants = {
        "A": merge_units(sentences, selection["a"]),
        "B": merge_units(sentences, selection["b"]),
    }
    for arm, variant in variants.items():
        if (
            len(variant["units"]) != DETECTOR_UNITS
            or tokenize(variant["text"]) != variant["units"]
        ):
            raise ValueError(f"{arm} detector unit invariant failed")
    invariant = {
        "source_units": len(sentences),
        "a_units": len(variants["A"]["units"]),
        "b_units": len(variants["B"]["units"]),
        "case_sensitive_lexical_sequence_equal": (
            lexical_sequence(variants["A"]["text"])
            == lexical_sequence(variants["B"]["text"])
            == lexical_sequence(original)
        ),
        "punctuation_multiset_equal": (
            punctuation_multiset(variants["A"]["text"])
            == punctuation_multiset(variants["B"]["text"])
        ),
        "nonadjacent_boundaries": (
            abs(selection["a"] - selection["b"]) > 1
        ),
        "a_to_b_one_split_one_merge": (
            variants["A"]["merge_boundary"] == selection["a"]
            and variants["B"]["merge_boundary"] == selection["b"]
        ),
    }
    if not all(
        value is True
        for key, value in invariant.items()
        if key not in {"source_units", "a_units", "b_units"}
    ):
        raise ValueError("A/B identity invariant failed")
    return {
        "original_sentences": sentences,
        "original_text": original,
        "original_text_sha256": text_sha256(original),
        "word_count": len(lexical_sequence(original)),
        "usable_boundaries": boundaries,
        "selection": selection,
        "variants": {
            arm: {
                **variant,
                "text_sha256": text_sha256(variant["text"]),
                "lexical_sha256": digest(
                    lexical_sequence(variant["text"])
                ),
                "punctuation_multiset": punctuation_multiset(
                    variant["text"]
                ),
            }
            for arm, variant in variants.items()
        },
        "invariants": invariant,
    }


def pair_dependencies() -> dict:
    paths = [
        Path(__file__),
        PROTOCOL,
        positive.PROTOCOL,
        positive.SOURCE_MANIFEST,
        positive.MANIFEST,
        positive.quality_paths("main")["decision"],
    ]
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for path in paths
    }


def prepare_pairs() -> dict:
    if PAIR_MANIFEST.exists():
        return check_pairs()
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    positive.check()
    decision = positive.quality_decision("main")
    if not positive.phase_terminal("main"):
        raise RuntimeError("Upstream main generation is not terminal")

    import nltk
    import spacy

    nlp = spacy.load("en_core_web_sm")
    quality = {
        row["allocation_id"]: row
        for row in decision["documents"]
    }
    documents = []
    records = []
    for source in positive.phase_sources("main"):
        allocation_id = source["allocation_id"]
        document_path = positive.document_path(source)
        document = load_json(document_path)
        quality_row = quality.get(allocation_id)
        quality_pass = bool(
            quality_row and quality_row["quality_pass"]
        )
        row = {
            "allocation_id": allocation_id,
            "allocation_ordinal": source["allocation_ordinal"],
            "source_document": str(document_path.relative_to(ROOT)),
            "source_document_sha256": sha256(document_path),
            "injection_status": document["status"],
            "all_slots_accepted": document["all_slots_accepted"],
            "quality_pass": quality_pass,
            "construction_status": "not_attempted_quality_failed",
            "construction": None,
        }
        if quality_pass:
            try:
                construction = build_variants(
                    allocation_id,
                    list(document["slot_texts"]),
                    nlp,
                    nltk.sent_tokenize,
                )
                row.update(
                    construction_status="constructed",
                    construction=construction,
                )
                records.append(
                    {
                        "id": allocation_id,
                        "cohort": "positive",
                        "allocation_id": allocation_id,
                        "source_document_sha256": sha256(document_path),
                        "target_templates": document["slot_templates"],
                        **construction,
                    }
                )
            except ValueError as exc:
                if str(exc) != "No two nonadjacent usable boundaries":
                    raise
                row["construction_status"] = (
                    "failed_no_nonadjacent_boundary_pair"
                )
        documents.append(row)

    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": (
            "pairs_frozen_before_detection"
            if records
            else "construction_gate_failed"
        ),
        "workflow": "swan_count_neutral_confirmation1",
        "selection_detector_blind": True,
        "upstream_main_allocations": positive.MAIN_DOCUMENTS,
        "upstream_quality_pass_documents": decision[
            "quality_pass_documents"
        ],
        "constructed_pairs": len(records),
        "documents": documents,
        "records": records,
        "dependencies": pair_dependencies(),
        "resource_counts": {
            "api_calls": 0,
            "parser_calls": 0,
            "s2match_calls": 0,
            "detector_calls": 0,
        },
    }
    return freeze_json(PAIR_MANIFEST, payload)


def validate_construction(record: dict) -> None:
    variants = record["variants"]
    original = record["original_text"]
    if (
        len(record["original_sentences"]) != SOURCE_SENTENCES
        or len(variants["A"]["units"]) != DETECTOR_UNITS
        or len(variants["B"]["units"]) != DETECTOR_UNITS
        or lexical_sequence(variants["A"]["text"])
        != lexical_sequence(variants["B"]["text"])
        or lexical_sequence(variants["A"]["text"])
        != lexical_sequence(original)
        or punctuation_multiset(variants["A"]["text"])
        != punctuation_multiset(variants["B"]["text"])
        or abs(
            variants["A"]["merge_boundary"]
            - variants["B"]["merge_boundary"]
        )
        <= 1
    ):
        raise ValueError(f"Invalid frozen construction: {record['id']}")


def check_pairs() -> dict:
    manifest = load_json(PAIR_MANIFEST)
    if (
        manifest["workflow"] != "swan_count_neutral_confirmation1"
        or manifest["upstream_main_allocations"] != positive.MAIN_DOCUMENTS
        or manifest["constructed_pairs"] != len(manifest["records"])
        or manifest["selection_detector_blind"] is not True
    ):
        raise ValueError("Pair manifest contract differs")
    for relative, expected in manifest["dependencies"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Frozen dependency changed: {relative}")
    for record in manifest["records"]:
        validate_construction(record)
    return manifest


def shard_name() -> str:
    return (
        f"realnewslike/c4-train.{CONTROL_SHARD:05d}-of-"
        f"{SHARD_COUNT:05d}.json.gz"
    )


def shard_url() -> str:
    return (
        "https://huggingface.co/datasets/allenai/c4/resolve/"
        f"{DATASET_REVISION}/{shard_name()}"
    )


def stream_control_rows(
    transport: Optional[Callable[[str], object]] = None,
):
    if transport is None:
        import requests

        def transport(url: str):
            return requests.get(
                url,
                headers={
                    "User-Agent": "SWAN-count-neutral-controls/1.0"
                },
                stream=True,
                timeout=(20, 90),
            )

    response = transport(shard_url())
    response.raise_for_status()
    response.raw.decode_content = False
    with response:
        with gzip.GzipFile(fileobj=response.raw) as handle:
            for row_index, line in enumerate(handle):
                yield row_index, json.loads(line)


def build_control_snapshot() -> dict:
    paths, scanned = allocator.discover_history_artifacts(
        root=ROOT,
        excluded_tree=OUTPUT,
    )
    snapshot = allocator.build_prior_identity_snapshot(
        paths,
        root=ROOT,
        all_files_scanned=scanned,
    )
    snapshot["selection"]["excluded_tree"] = str(OUTPUT.relative_to(ROOT))
    snapshot["selection"]["purpose"] = (
        "Exclude all historical source, window, and URL identities before "
        "allocating independent count-neutral human controls."
    )
    return snapshot


def first_control_window(
    sentences: list[str],
    allocation_id: str,
    seen_text: set[str],
    nlp,
    tokenize: Callable[[str], list[str]],
) -> Optional[dict]:
    for start in range(max(0, len(sentences) - SOURCE_SENTENCES + 1)):
        window = sentences[start : start + SOURCE_SENTENCES]
        if len(window) != SOURCE_SENTENCES:
            break
        text = " ".join(window)
        words = len(lexical_sequence(text))
        identity = allocator.normalized_hash(text)
        if (
            not MIN_WORDS <= words <= MAX_WORDS
            or identity in seen_text
        ):
            continue
        try:
            construction = build_variants(
                allocation_id,
                window,
                nlp,
                tokenize,
            )
        except ValueError as exc:
            if str(exc) == "No two nonadjacent usable boundaries":
                continue
            raise
        return {
            "window_start_sentence": start,
            "normalized_window_sha256": identity,
            **construction,
        }
    return None


def prepare_controls(
    transport: Optional[Callable[[str], object]] = None,
) -> dict:
    if CONTROL_MANIFEST.exists():
        return check_controls()
    pairs = check_pairs()
    if not pairs["records"]:
        raise RuntimeError("Positive construction gate failed")
    if CONTROL_SNAPSHOT.exists():
        snapshot = load_json(CONTROL_SNAPSHOT)
    else:
        snapshot = freeze_json(
            CONTROL_SNAPSHOT,
            build_control_snapshot(),
        )

    import nltk
    import spacy

    nlp = spacy.load("en_core_web_sm")
    seen_text = set(snapshot["normalized_text_sha256"])
    seen_url = set(snapshot["url_sha256"])
    records = []
    screened = []
    required = CALIBRATION_CONTROLS + TEST_CONTROLS
    for row_index, row in stream_control_rows(transport):
        raw = row.get("text", "")
        url = row.get("url", "")
        raw = raw if isinstance(raw, str) else ""
        url = url if isinstance(url, str) else ""
        raw_identity = allocator.normalized_hash(raw)
        url_identity = allocator.normalized_url_hash(url)
        checks = {
            "nonempty": bool(allocator.canonical(raw)),
            "historically_unseen_raw": raw_identity not in seen_text,
            "historically_unseen_url": (
                not url or url_identity not in seen_url
            ),
        }
        window = None
        if all(checks.values()):
            ordinal = len(records)
            cohort = (
                "human_calibration"
                if ordinal < CALIBRATION_CONTROLS
                else "human_test"
            )
            cohort_ordinal = (
                ordinal
                if cohort == "human_calibration"
                else ordinal - CALIBRATION_CONTROLS
            )
            allocation_id = f"cn-{cohort}-{cohort_ordinal:03d}"
            window = first_control_window(
                nltk.sent_tokenize(raw),
                allocation_id,
                seen_text,
                nlp,
                nltk.sent_tokenize,
            )
            checks["eligible_window"] = window is not None
            if window is not None:
                record = {
                    "id": allocation_id,
                    "allocation_id": allocation_id,
                    "allocation_ordinal": ordinal,
                    "cohort": cohort,
                    "cohort_ordinal": cohort_ordinal,
                    "dataset_locator": f"{shard_name()}:{row_index}",
                    "shard_row": row_index,
                    "source_text_sha256": text_sha256(raw),
                    "normalized_source_sha256": raw_identity,
                    "url_sha256": url_identity,
                    **window,
                }
                records.append(record)
                seen_text.add(raw_identity)
                seen_text.add(record["normalized_window_sha256"])
                if url:
                    seen_url.add(url_identity)
        screened.append(
            {
                "shard_row": row_index,
                "source_text_sha256": text_sha256(raw),
                "normalized_source_sha256": raw_identity,
                "url_sha256": url_identity,
                "checks": checks,
                "selected": window is not None,
            }
        )
        if len(records) == required:
            break
    if len(records) != required:
        raise RuntimeError("C4 shard ended before 500 controls were allocated")

    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "controls_frozen_before_detection",
        "dataset": {
            "name": "allenai/c4",
            "configuration": "realnewslike",
            "revision": DATASET_REVISION,
            "shard": shard_name(),
            "shard_url": shard_url(),
        },
        "selection": (
            "First eligible 16-sentence window per unseen row in source "
            "order; first eligible window in sentence-start order."
        ),
        "prior_identity_snapshot_sha256": sha256(CONTROL_SNAPSHOT),
        "pair_manifest_sha256": sha256(PAIR_MANIFEST),
        "cohort_counts": {
            "human_calibration": CALIBRATION_CONTROLS,
            "human_test": TEST_CONTROLS,
        },
        "rows_scanned": len(screened),
        "screened_rows": screened,
        "records": records,
        "resource_counts": {
            "public_dataset_http_requests": 1,
            "api_calls": 0,
            "parser_calls": 0,
            "s2match_calls": 0,
        },
    }
    return freeze_json(CONTROL_MANIFEST, payload)


def check_controls() -> dict:
    controls = load_json(CONTROL_MANIFEST)
    snapshot = load_json(CONTROL_SNAPSHOT)
    if (
        controls["status"] != "controls_frozen_before_detection"
        or controls["prior_identity_snapshot_sha256"]
        != sha256(CONTROL_SNAPSHOT)
        or controls["pair_manifest_sha256"] != sha256(PAIR_MANIFEST)
        or controls["cohort_counts"]
        != {
            "human_calibration": CALIBRATION_CONTROLS,
            "human_test": TEST_CONTROLS,
        }
        or len(controls["records"])
        != CALIBRATION_CONTROLS + TEST_CONTROLS
    ):
        raise ValueError("Control manifest contract differs")
    prior = set(snapshot["normalized_text_sha256"])
    prior_url = set(snapshot["url_sha256"])
    seen = set()
    seen_url = set()
    for record in controls["records"]:
        validate_construction(record)
        identities = {
            record["normalized_source_sha256"],
            record["normalized_window_sha256"],
        }
        if identities & (prior | seen):
            raise ValueError("Control identity is not independent")
        if (
            record["url_sha256"] != allocator.normalized_url_hash("")
            and record["url_sha256"] in prior_url | seen_url
        ):
            raise ValueError("Control URL identity is not independent")
        seen.update(identities)
        if record["url_sha256"] != allocator.normalized_url_hash(""):
            seen_url.add(record["url_sha256"])
    return controls


def check() -> dict:
    return {
        "pairs": check_pairs(),
        "controls": check_controls(),
    }


def records_for_cohort(cohort: str) -> tuple[list[dict], Path]:
    if cohort == "positive":
        manifest = check_pairs()
        return manifest["records"], PAIR_MANIFEST
    controls = check_controls()
    return [
        record
        for record in controls["records"]
        if record["cohort"] == cohort
    ], CONTROL_MANIFEST


def detector_runtime(manifest_path: Path, device: str) -> dict:
    packages = {}
    for name in (
        "amrlib",
        "nltk",
        "numpy",
        "penman",
        "scipy",
        "spacy",
        "torch",
        "transformers",
    ):
        packages[name] = importlib.metadata.version(name)
    return {
        "workflow": "swan_count_neutral_confirmation1",
        "script_sha256": sha256(Path(__file__)),
        "protocol_sha256": sha256(PROTOCOL),
        "record_manifest_sha256": sha256(manifest_path),
        "official_swan_commit": swan.EXPECTED_COMMIT,
        "bank_sha256": sha256(
            swan.OFFICIAL_ROOT / "amr_bank/banks/amr_bank_50.json"
        ),
        "parser_model_sha256": sha256(
            swan.PARSER_MODEL / "pytorch_model.bin"
        ),
        "parser_tokenizer_sha256": sha256(
            swan.PARSER_TOKENIZER / "tokenizer.json"
        ),
        "vectors_sha256": sha256(swan.GLOVE_VECTORS),
        "device": device,
        "green_threshold": GREEN_THRESHOLD,
        "null_green_rate": NULL_GREEN_RATE,
        "s2match_seeded_per_unit_template": True,
        "packages": packages,
    }


def detection_seed(record_id: str, text: str, bank_index: int) -> int:
    return int(
        hashlib.sha256(
            (
                "swan-count-neutral-confirmation1:detector:"
                f"{record_id}:{text_sha256(text)}:{bank_index}"
            ).encode("utf-8")
        ).hexdigest()[:8],
        16,
    )


def transform_raw(raw: str, module) -> str:
    import penman

    graph = amr_body(raw)
    penman.decode(graph)
    graph = module.normalize_amr_variables(graph)
    graph = module.generate_template_amr(graph)
    graph = module.normalize_amr_variables(graph)
    penman.decode(graph)
    return graph


def z_score(green: int, units: int = DETECTOR_UNITS) -> float:
    return (
        green - NULL_GREEN_RATE * units
    ) / math.sqrt(units * NULL_GREEN_RATE * (1 - NULL_GREEN_RATE))


def detect_record(
    record: dict,
    module,
    utils_globals: dict,
    bank: list[str],
    runtime: dict,
) -> dict:
    parsed = {}
    for arm in ("A", "B"):
        variant = record["variants"][arm]
        for unit in variant["units"]:
            if unit in parsed:
                continue
            raw = module.parse_amr(unit)
            if not raw:
                raise ValueError("AMR parser returned an empty graph")
            graph = transform_raw(raw, module)
            scores = []
            for bank_index, target in enumerate(bank):
                with swan.fixed_score_seed(
                    detection_seed(record["id"], unit, bank_index)
                ):
                    score = float(
                        utils_globals["compute_s2match_score"](
                            graph,
                            target,
                        )
                    )
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Invalid detector S2Match score")
                scores.append(score)
            parsed[unit] = {
                "text": unit,
                "text_sha256": text_sha256(unit),
                "raw_amr": raw,
                "raw_amr_sha256": text_sha256(raw),
                "amr": graph,
                "amr_sha256": text_sha256(graph),
                "bank_scores": scores,
                "score": max(scores),
                "green": max(scores) >= GREEN_THRESHOLD,
            }
    arms = {}
    for arm in ("A", "B"):
        details = [
            {
                **parsed[unit],
                "source_sentence_indices": source_indices,
            }
            for unit, source_indices in zip(
                record["variants"][arm]["units"],
                record["variants"][arm][
                    "unit_source_sentence_indices"
                ],
            )
        ]
        green = sum(row["green"] for row in details)
        arms[arm] = {
            "n": len(details),
            "green": green,
            "green_rate": green / len(details),
            "z": z_score(green, len(details)),
            "details": details,
        }
        if arms[arm]["n"] != DETECTOR_UNITS:
            raise ValueError("Count-neutral detector unit count changed")
    return {
        "schema_version": 1,
        "status": "complete",
        "id": record["id"],
        "cohort": record["cohort"],
        "record_sha256": digest(record),
        "runtime": runtime,
        "runtime_sha256": digest(runtime),
        "arms": arms,
        "resource_counts": {
            "parser_calls": len(parsed),
            "s2match_calls": len(parsed) * len(bank),
            "api_calls": 0,
        },
    }


def detection_path(cohort: str, record_id: str) -> Path:
    return OUTPUT / "detected" / cohort / f"{record_id}.json"


def detect(
    cohort: str,
    offset: int,
    limit: int,
    device: str,
) -> dict:
    if offset < 0 or limit < 1:
        raise ValueError("Detection offset/limit is invalid")
    if device != "cpu":
        raise ValueError(
            "The frozen official runtime is CPU-only for this experiment"
        )
    records, manifest_path = records_for_cohort(cohort)
    selected = records[offset : offset + limit]
    if not selected:
        return {
            "status": "no_records_in_requested_slice",
            "cohort": cohort,
            "available_records": len(records),
        }
    runtime = detector_runtime(manifest_path, device)
    bank_rows = load_json(
        swan.OFFICIAL_ROOT / "amr_bank/banks/amr_bank_50.json"
    )
    bank = [row["normalized_template_amr"] for row in bank_rows]
    if len(bank) != 50 or len(set(bank)) != 50:
        raise ValueError("Official bank50 changed")

    completed = 0
    failed = 0
    with positive.patched_swan_runtime() as module:
        utils_globals = module.parse_amr.__globals__
        for record in selected:
            path = detection_path(cohort, record["id"])
            if path.exists():
                saved = load_json(path)
                if (
                    saved.get("record_sha256") != digest(record)
                    or saved.get("runtime_sha256") != digest(runtime)
                ):
                    raise ValueError(
                        f"Frozen detection differs: {path.name}"
                    )
                completed += saved["status"] == "complete"
                failed += saved["status"] != "complete"
                continue
            try:
                payload = detect_record(
                    record,
                    module,
                    utils_globals,
                    bank,
                    runtime,
                )
                completed += 1
            except Exception as exc:
                payload = {
                    "schema_version": 1,
                    "status": "failed",
                    "id": record["id"],
                    "cohort": cohort,
                    "record_sha256": digest(record),
                    "runtime": runtime,
                    "runtime_sha256": digest(runtime),
                    "error_type": type(exc).__name__,
                    "resource_counts": {
                        "api_calls": 0,
                    },
                }
                failed += 1
            freeze_json(path, payload)
    return {
        "status": "detection_slice_terminal",
        "cohort": cohort,
        "selected": len(selected),
        "complete": completed,
        "failed": failed,
        "offset": offset,
        "limit": limit,
    }


def terminal_detection_rows(cohort: str) -> list[dict]:
    records, _ = records_for_cohort(cohort)
    rows = []
    for record in records:
        path = detection_path(cohort, record["id"])
        if not path.is_file():
            raise RuntimeError(
                f"{cohort} detection is incomplete: {record['id']}"
            )
        row = load_json(path)
        if row["record_sha256"] != digest(record):
            raise ValueError("Detection record identity changed")
        rows.append(row)
    return rows


def calibrate() -> dict:
    if CALIBRATION.exists():
        return load_json(CALIBRATION)
    rows = terminal_detection_rows("human_calibration")
    complete = [row for row in rows if row["status"] == "complete"]
    if len(complete) != CALIBRATION_CONTROLS:
        payload = {
            "schema_version": 1,
            "status": "calibration_failed_incomplete_detection",
            "expected": CALIBRATION_CONTROLS,
            "complete": len(complete),
            "failed": len(rows) - len(complete),
            "control_manifest_sha256": sha256(CONTROL_MANIFEST),
        }
        return freeze_json(CALIBRATION, payload)
    candidates = [PUBLIC_Z_THRESHOLD] + [
        z_score(green)
        for green in range(DETECTOR_UNITS + 1)
        if z_score(green) > PUBLIC_Z_THRESHOLD
    ]
    candidates = sorted(set(candidates))
    selected = None
    diagnostics = []
    for threshold in candidates:
        arm_counts = {
            arm: sum(row["arms"][arm]["z"] > threshold for row in complete)
            for arm in ("A", "B")
        }
        eligible = all(
            count / CALIBRATION_CONTROLS <= CALIBRATION_FPR_TARGET
            for count in arm_counts.values()
        )
        diagnostics.append(
            {
                "threshold": threshold,
                "strict_exceedances": arm_counts,
                "eligible": eligible,
            }
        )
        if eligible and selected is None:
            selected = threshold
    if selected is None:
        raise RuntimeError("No attainable calibration threshold")
    payload = {
        "schema_version": 1,
        "status": "calibration_frozen_without_positive_or_test_access",
        "control_manifest_sha256": sha256(CONTROL_MANIFEST),
        "records": len(complete),
        "arms_per_record": 2,
        "target_fpr_each_arm": CALIBRATION_FPR_TARGET,
        "public_threshold": PUBLIC_Z_THRESHOLD,
        "selected_common_threshold": selected,
        "strict_rule": "z > threshold",
        "diagnostics": diagnostics,
    }
    return freeze_json(CALIBRATION, payload)


def wilson(successes: int, total: int, z: float = 1.95996398454) -> dict:
    if total == 0:
        return {
            "successes": successes,
            "total": total,
            "rate": None,
            "ci95": [None, None],
        }
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            rate * (1 - rate) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return {
        "successes": successes,
        "total": total,
        "rate": rate,
        "ci95": [max(0.0, center - margin), min(1.0, center + margin)],
    }


def exact_directional_p(losses: int, gains: int) -> float:
    discordant = losses + gains
    if discordant == 0 or losses <= gains:
        return 1.0
    return sum(
        math.comb(discordant, value)
        for value in range(losses, discordant + 1)
    ) / 2**discordant


def arm_rates(rows: list[dict], threshold: float) -> dict:
    complete = [row for row in rows if row["status"] == "complete"]
    return {
        arm: wilson(
            sum(row["arms"][arm]["z"] > threshold for row in complete),
            len(complete),
        )
        for arm in ("A", "B")
    }


def paired_result(rows: list[dict], threshold: float) -> dict:
    complete = [row for row in rows if row["status"] == "complete"]
    detected = {
        arm: [
            row["arms"][arm]["z"] > threshold
            for row in complete
        ]
        for arm in ("A", "B")
    }
    losses = sum(
        left and not right
        for left, right in zip(detected["A"], detected["B"])
    )
    gains = sum(
        not left and right
        for left, right in zip(detected["A"], detected["B"])
    )
    a_positive = sum(detected["A"])
    return {
        "threshold": threshold,
        "complete_pairs": len(complete),
        "failed_pairs": len(rows) - len(complete),
        "arm_rates": arm_rates(rows, threshold),
        "a_detected_b_not": losses,
        "a_not_b_detected": gains,
        "both_detected": sum(
            left and right
            for left, right in zip(detected["A"], detected["B"])
        ),
        "neither_detected": sum(
            not left and not right
            for left, right in zip(detected["A"], detected["B"])
        ),
        "a_to_b_escape_rate": wilson(losses, a_positive),
        "one_sided_exact_p": exact_directional_p(losses, gains),
        "mean_green_change_b_minus_a": (
            statistics.mean(
                row["arms"]["B"]["green"] - row["arms"]["A"]["green"]
                for row in complete
            )
            if complete
            else None
        ),
        "mean_z_change_b_minus_a": (
            statistics.mean(
                row["arms"]["B"]["z"] - row["arms"]["A"]["z"]
                for row in complete
            )
            if complete
            else None
        ),
    }


def test() -> dict:
    if TEST_RESULT.exists():
        return load_json(TEST_RESULT)
    calibration = calibrate()
    if calibration["status"] != (
        "calibration_frozen_without_positive_or_test_access"
    ):
        raise RuntimeError("Calibration did not produce a valid threshold")
    positive_rows = terminal_detection_rows("positive")
    control_rows = terminal_detection_rows("human_test")
    threshold = calibration["selected_common_threshold"]
    control_complete = [
        row for row in control_rows if row["status"] == "complete"
    ]
    control_rates = arm_rates(control_rows, threshold)
    fpr_valid = bool(
        len(control_complete) == TEST_CONTROLS
        and all(
            control_rates[arm]["rate"] <= TEST_FPR_CEILING
            for arm in ("A", "B")
        )
    )
    calibrated_positive = paired_result(positive_rows, threshold)
    public_positive = paired_result(
        positive_rows,
        PUBLIC_Z_THRESHOLD,
    )
    directional = bool(
        fpr_valid
        and calibrated_positive["a_detected_b_not"]
        > calibrated_positive["a_not_b_detected"]
        and calibrated_positive["one_sided_exact_p"] <= 0.05
    )
    pair_manifest = check_pairs()
    payload = {
        "schema_version": 1,
        "status": "confirmatory_test_frozen",
        "calibration_sha256": sha256(CALIBRATION),
        "pair_manifest_sha256": sha256(PAIR_MANIFEST),
        "control_manifest_sha256": sha256(CONTROL_MANIFEST),
        "attrition": {
            "allocated_main_documents": pair_manifest[
                "upstream_main_allocations"
            ],
            "quality_pass_documents": pair_manifest[
                "upstream_quality_pass_documents"
            ],
            "constructed_pairs": pair_manifest["constructed_pairs"],
            "positive_detection_complete": sum(
                row["status"] == "complete" for row in positive_rows
            ),
            "positive_detection_failed": sum(
                row["status"] != "complete" for row in positive_rows
            ),
        },
        "calibrated_threshold": threshold,
        "human_test": {
            "expected_records": TEST_CONTROLS,
            "complete_records": len(control_complete),
            "failed_records": len(control_rows) - len(control_complete),
            "arm_fpr": control_rates,
            "fpr_ceiling_each_arm": TEST_FPR_CEILING,
            "detector_valid": fpr_valid,
        },
        "positive_public_threshold": public_positive,
        "positive_calibrated_threshold": calibrated_positive,
        "directional_claim_rule": (
            "valid human test FPR, A-to-B losses exceed reverse gains, "
            "and one-sided exact p <= 0.05"
        ),
        "directional_count_neutral_effect_confirmed": directional,
        "claim_boundary": (
            "The A/B orientation was fixed by a detector-blind hash. "
            "Construction and detection attrition remain in the denominators."
        ),
    }
    return freeze_json(TEST_RESULT, payload)


def detail_for_sources(detection: dict, arm: str, indices: list[int]) -> dict:
    matches = [
        detail
        for detail in detection["arms"][arm]["details"]
        if detail["source_sentence_indices"] == indices
    ]
    if len(matches) != 1:
        raise ValueError("Cannot recover AMR for source sentence indices")
    return matches[0]


def validate_metric(values: tuple[float, float, float]) -> None:
    if (
        len(values) != 3
        or not all(math.isfinite(value) and 0 <= value <= 1 for value in values)
    ):
        raise ValueError("Invalid precision/recall/F1")
    precision, recall, f1 = values
    expected = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    if not math.isclose(f1, expected, abs_tol=1e-12):
        raise ValueError("F1 does not reconstruct")


def mechanism_seed(
    record_id: str,
    arm: str,
    slot: int,
    condition: str,
    repeat: int,
) -> int:
    return int(
        hashlib.sha256(
            (
                "swan-count-neutral-confirmation1:mechanism:"
                f"{record_id}:{arm}:{slot}:{condition}:{repeat}"
            ).encode("utf-8")
        ).hexdigest()[:8],
        16,
    )


def mechanism_metrics(
    graph: str,
    target: str,
    utils_globals: dict,
    record_id: str,
    arm: str,
    slot: int,
    condition: str,
) -> dict:
    observations = []
    scorer = utils_globals["compute_s2match_from_strings"]
    for repeat in range(MECHANISM_REPEATS):
        with swan.fixed_score_seed(
            mechanism_seed(record_id, arm, slot, condition, repeat)
        ):
            values = tuple(
                float(value)
                for value in scorer(
                    graph,
                    target,
                    vectors_file=str(swan.GLOVE_VECTORS),
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
    return {
        "observations": observations,
        "median": {
            metric: statistics.median(
                row[metric] for row in observations
            )
            for metric in ("precision", "recall", "f1")
        },
    }


def mechanism_record(
    record: dict,
    detection: dict,
    module,
    utils_globals: dict,
) -> dict:
    arms = {}
    for arm, opposite in (("A", "B"), ("B", "A")):
        boundary = record["variants"][arm]["merge_boundary"]
        slots = [boundary, boundary + 1]
        merged = detail_for_sources(detection, arm, slots)
        components = [
            detail_for_sources(detection, opposite, [slot])
            for slot in slots
        ]
        composed_raw = composition.compose_raw_amrs(
            components[0]["raw_amr"],
            components[1]["raw_amr"],
        )
        composed = transform_raw(composed_raw, module)
        targets = []
        for slot, component in zip(slots, components):
            target = record["target_templates"][slot]
            conditions = {
                "original": component["amr"],
                "composition": composed,
                "reparse": merged["amr"],
            }
            targets.append(
                {
                    "slot": slot,
                    "target_template_sha256": text_sha256(target),
                    "conditions": {
                        name: mechanism_metrics(
                            graph,
                            target,
                            utils_globals,
                            record["id"],
                            arm,
                            slot,
                            name,
                        )
                        for name, graph in conditions.items()
                    },
                }
            )
        arms[arm] = {
            "merge_boundary": boundary,
            "source_slots": slots,
            "component_raw_amr_sha256": [
                row["raw_amr_sha256"] for row in components
            ],
            "composed_raw_amr": composed_raw,
            "composed_raw_amr_sha256": text_sha256(composed_raw),
            "composed_amr": composed,
            "composed_amr_sha256": text_sha256(composed),
            "reparsed_amr_sha256": merged["amr_sha256"],
            "targets": targets,
        }
    return {
        "schema_version": 1,
        "status": "complete",
        "id": record["id"],
        "record_sha256": digest(record),
        "detection_sha256": sha256(
            detection_path("positive", record["id"])
        ),
        "arms": arms,
        "resource_counts": {
            "new_api_calls": 0,
            "new_parser_calls": 0,
            "new_s2match_observations": (
                2 * 2 * 3 * MECHANISM_REPEATS
            ),
        },
    }


def mechanism(offset: int, limit: int) -> dict:
    if not TEST_RESULT.is_file():
        raise RuntimeError("Confirmatory test must be frozen first")
    if offset < 0 or limit < 1:
        raise ValueError("Mechanism offset/limit is invalid")
    records = check_pairs()["records"]
    selected = records[offset : offset + limit]
    if not selected:
        return {
            "status": "no_records_in_requested_slice",
            "available_records": len(records),
        }
    completed = 0
    with positive.patched_swan_runtime() as module:
        utils_globals = module.parse_amr.__globals__
        for record in selected:
            detection_path_value = detection_path("positive", record["id"])
            detection = load_json(detection_path_value)
            if detection["status"] != "complete":
                continue
            path = OUTPUT / "mechanism" / f"{record['id']}.json"
            payload = mechanism_record(
                record,
                detection,
                module,
                utils_globals,
            )
            freeze_json(path, payload)
            completed += 1
    return {
        "status": "mechanism_slice_terminal",
        "selected": len(selected),
        "complete": completed,
        "offset": offset,
        "limit": limit,
    }


def mechanism_summary() -> dict:
    records = check_pairs()["records"]
    rows = []
    missing = []
    for record in records:
        detection = load_json(detection_path("positive", record["id"]))
        if detection["status"] != "complete":
            continue
        path = OUTPUT / "mechanism" / f"{record['id']}.json"
        if not path.is_file():
            missing.append(record["id"])
            continue
        rows.append(load_json(path))
    if missing:
        raise RuntimeError(
            "Mechanism is incomplete: " + ", ".join(missing[:5])
        )
    targets = [
        target
        for row in rows
        for arm in ("A", "B")
        for target in row["arms"][arm]["targets"]
    ]
    means = {
        condition: {
            metric: (
                statistics.mean(
                    target["conditions"][condition]["median"][metric]
                    for target in targets
                )
                if targets
                else None
            )
            for metric in ("precision", "recall", "f1")
        }
        for condition in ("original", "composition", "reparse")
    }
    return {
        "complete_documents": len(rows),
        "targets": len(targets),
        "new_parser_calls": 0,
        "new_s2match_observations": sum(
            row["resource_counts"]["new_s2match_observations"]
            for row in rows
        ),
        "mean_target_metrics": means,
        "mean_changes": {
            f"{condition}_minus_original_{metric}": (
                means[condition][metric] - means["original"][metric]
                if targets
                else None
            )
            for condition in ("composition", "reparse")
            for metric in ("precision", "recall", "f1")
        },
    }


def render_report(summary: dict) -> str:
    test_row = summary["test"]
    attrition = test_row["attrition"]
    calibrated = test_row["positive_calibrated_threshold"]
    return "\n".join(
        [
            "# Count-neutral SWAN confirmation",
            "",
            f"Status: `{summary['status']}`.",
            "",
            f"- Main allocations: **{attrition['allocated_main_documents']}**",
            f"- Quality pass: **{attrition['quality_pass_documents']}**",
            f"- Constructed A/B pairs: **{attrition['constructed_pairs']}**",
            (
                "- Complete positive detections: "
                f"**{attrition['positive_detection_complete']}**"
            ),
            (
                "- Calibrated threshold: "
                f"**{test_row['calibrated_threshold']:.6f}**"
            ),
            (
                "- A-detected to B-not-detected: "
                f"**{calibrated['a_detected_b_not']}**"
            ),
            (
                "- Reverse transitions: "
                f"**{calibrated['a_not_b_detected']}**"
            ),
            (
                "- Directional effect confirmed: "
                f"**{test_row['directional_count_neutral_effect_confirmed']}**"
            ),
            "",
            (
                "A/B labels and both boundary positions were fixed without "
                "detector scores. DeepSeek is a non-equivalent generation "
                "backend."
            ),
            "",
        ]
    )


def summarize() -> dict:
    if not TEST_RESULT.is_file():
        raise RuntimeError("Confirmatory test is not frozen")
    test_row = load_json(TEST_RESULT)
    mechanism_row = mechanism_summary()
    payload = {
        "schema_version": 1,
        "status": "complete_count_neutral_confirmation",
        "pair_manifest_sha256": sha256(PAIR_MANIFEST),
        "control_manifest_sha256": sha256(CONTROL_MANIFEST),
        "calibration_sha256": sha256(CALIBRATION),
        "test_result_sha256": sha256(TEST_RESULT),
        "test": test_row,
        "mechanism": mechanism_row,
        "claim_boundary": (
            "Report the preregistered gates exactly, including null results, "
            "attrition, and independent-test FPR."
        ),
    }
    frozen = freeze_json(SUMMARY, payload)
    report = render_report(frozen)
    if REPORT.exists() and REPORT.read_text(encoding="utf-8") != report:
        raise ValueError("Frozen report differs")
    if not REPORT.exists():
        REPORT.write_text(report, encoding="utf-8")
    return frozen


def count_detection_files(cohort: str) -> dict:
    records, _ = records_for_cohort(cohort)
    paths = [detection_path(cohort, row["id"]) for row in records]
    terminal = [path for path in paths if path.is_file()]
    complete = sum(
        load_json(path).get("status") == "complete"
        for path in terminal
    )
    return {
        "expected": len(records),
        "terminal": len(terminal),
        "complete": complete,
        "failed": len(terminal) - complete,
    }


def status() -> dict:
    if not PAIR_MANIFEST.is_file():
        return {
            "status": "awaiting_positive_main_quality",
            "next": "prepare-pairs",
            "api_calls": 0,
        }
    pairs = check_pairs()
    if not pairs["records"]:
        return {
            "status": "construction_gate_failed",
            "constructed_pairs": 0,
            "api_calls": 0,
        }
    if not CONTROL_MANIFEST.is_file():
        return {
            "status": "pairs_frozen_controls_pending",
            "constructed_pairs": len(pairs["records"]),
            "next": "prepare-controls",
            "api_calls": 0,
        }
    counts = {
        cohort: count_detection_files(cohort)
        for cohort in (
            "human_calibration",
            "positive",
            "human_test",
        )
    }
    mechanism_expected = counts["positive"]["complete"]
    mechanism_complete = len(
        list((OUTPUT / "mechanism").glob("*.json"))
    )
    return {
        "status": (
            "complete"
            if SUMMARY.is_file()
            else "mechanism_pending"
            if TEST_RESULT.is_file()
            else "test_pending"
            if CALIBRATION.is_file()
            else "calibration_detection_pending"
        ),
        "constructed_pairs": len(pairs["records"]),
        "detection": counts,
        "calibration_frozen": CALIBRATION.is_file(),
        "test_frozen": TEST_RESULT.is_file(),
        "mechanism": {
            "expected": mechanism_expected,
            "complete": mechanism_complete,
        },
        "summary_frozen": SUMMARY.is_file(),
        "api_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare-pairs",
            "prepare-controls",
            "check",
            "detect",
            "calibrate",
            "test",
            "mechanism",
            "summarize",
            "status",
        ),
    )
    parser.add_argument(
        "--cohort",
        choices=("positive", "human_calibration", "human_test"),
        default="positive",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--device",
        choices=("cpu",),
        default="cpu",
    )
    args = parser.parse_args()
    os.environ.setdefault("NLTK_DATA", str(swan.NLTK_DATA))
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
    os.environ.setdefault("GLOVE_VECTORS_PATH", str(swan.GLOVE_VECTORS))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.stage == "prepare-pairs":
        result = prepare_pairs()
    elif args.stage == "prepare-controls":
        result = prepare_controls()
    elif args.stage == "check":
        result = check()
    elif args.stage == "detect":
        result = detect(
            args.cohort,
            args.offset,
            args.limit,
            args.device,
        )
    elif args.stage == "calibrate":
        result = calibrate()
    elif args.stage == "test":
        result = test()
    elif args.stage == "mechanism":
        result = mechanism(args.offset, args.limit)
    elif args.stage == "summarize":
        result = summarize()
    else:
        result = status()
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
