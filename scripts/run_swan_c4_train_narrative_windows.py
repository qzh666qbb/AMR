"""Preflight frozen narrative windows in held-out C4 RealNews train shards."""

import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path

from feedback2_runtime import Heartbeat, claim
from prepare_confirmation_experiment import normalized_hash
from run_confirmation_experiment import freeze, request_body
from run_segmentation_intervention import REVISION, ROOT, sha, text_sha, write_json
import run_swan_natural_narrative_windows as narrative
import run_swan_natural_mixed_windows as mixed
from supervise_confirmation_experiment import read_credential


OUTPUT = ROOT / "experiments/swan_c4_train_narrative_windows3"
PRIOR = ROOT / "experiments/swan_natural_mixed_windows1"
SHARD_INDICES = tuple(range(4))
SHARD_COUNT = 512
FIRST_SHARD_SKIP = 5250
MAX_ELIGIBLE_ROWS = 100000
DOCUMENTS = 5
API_CAP = 6
MAX_ACQUISITION_ATTEMPTS = 3


def configure_environment():
    narrative.configure_environment()


def shard_name(index):
    return (
        f"realnewslike/c4-train.{index:05d}-of-{SHARD_COUNT:05d}.json.gz"
    )


def stream_train_shard(index):
    import requests

    url = (
        "https://huggingface.co/datasets/allenai/c4/resolve/"
        f"{REVISION}/{shard_name(index)}"
    )
    with requests.get(url, stream=True, timeout=(20, 60)) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        with gzip.GzipFile(fileobj=response.raw) as handle:
            for line in handle:
                yield json.loads(line)


def prior_identities():
    manifest = json.loads((PRIOR / "manifest.json").read_text())
    return (
        {
            source["normalized_source_sha256"]
            for source in manifest["sources"]
        },
        {
            source["url"]
            for source in manifest["sources"]
            if source["url"]
        },
    )


def source_attempt(folder):
    path = folder / "source_acquisition.json"
    journal = (
        json.loads(path.read_text())
        if path.exists()
        else {"attempts": []}
    )
    if len(journal["attempts"]) >= MAX_ACQUISITION_ATTEMPTS:
        raise RuntimeError("Frozen source acquisition attempt cap reached")
    attempt = {
        "ordinal": len(journal["attempts"]) + 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "in_flight",
        "shards_opened": 0,
    }
    journal["attempts"].append(attempt)
    write_json(path, journal)
    return path, journal, attempt


def acquire_sources(folder, tokenize, nlp):
    path, journal, attempt = source_attempt(folder)
    prior_text, prior_url = prior_identities()
    seen_text = set(prior_text)
    seen_url = set(prior_url)
    selected = []
    screening = {
        "excluded_prefix_rows": 0,
        "eligible_rows_scanned": 0,
        "duplicate_rows": 0,
        "base_window_rows": 0,
        "selected_documents": 0,
    }
    scanned_through = {}
    reached_limit = False
    stream = None
    try:
        for shard_index in SHARD_INDICES:
            stream = stream_train_shard(shard_index)
            attempt["shards_opened"] += 1
            for row_index, row in enumerate(stream):
                scanned_through[str(shard_index)] = row_index
                text = row.get("text", "")
                url = row.get("url", "")
                identity = normalized_hash(text)
                duplicate = (
                    identity in seen_text
                    or bool(url and url in seen_url)
                )
                seen_text.add(identity)
                if url:
                    seen_url.add(url)
                if shard_index == 0 and row_index < FIRST_SHARD_SKIP:
                    screening["excluded_prefix_rows"] += 1
                    continue
                if screening["eligible_rows_scanned"] >= MAX_ELIGIBLE_ROWS:
                    reached_limit = True
                    break
                screening["eligible_rows_scanned"] += 1
                if duplicate:
                    screening["duplicate_rows"] += 1
                    continue
                base_exists = mixed.find_window(text, tokenize) is not None
                screening["base_window_rows"] += bool(base_exists)
                if not base_exists:
                    continue
                window = narrative.find_window(text, tokenize, nlp)
                if window is None:
                    continue
                selected.append(
                    {
                        "prompt_id": (
                            f"c4-train-{shard_index:05d}-{row_index:08d}"
                        ),
                        "shard_index": shard_index,
                        "shard_row": row_index,
                        "url": url,
                        "url_sha256": text_sha(url),
                        "source": text,
                        "source_sha256": text_sha(text),
                        "normalized_source_sha256": identity,
                        "window": window,
                    }
                )
                if len(selected) == DOCUMENTS:
                    reached_limit = True
                    break
            stream.close()
            stream = None
            if reached_limit:
                break
    except Exception as exc:
        attempt.update(
            status="failed",
            error_type=type(exc).__name__,
            finished_at=datetime.now(timezone.utc).isoformat(),
            scanned_through=scanned_through,
            manifest_written=False,
        )
        write_json(path, journal)
        raise
    finally:
        if stream is not None:
            stream.close()
    screening["selected_documents"] = len(selected)
    attempt.update(
        status="complete",
        finished_at=datetime.now(timezone.utc).isoformat(),
        scanned_through=scanned_through,
        eligible_rows_scanned=screening["eligible_rows_scanned"],
        selected_documents=len(selected),
    )
    write_json(path, journal)
    return selected, screening, scanned_through


def package_versions():
    return narrative.package_versions()


def dependency_paths():
    paths = set(narrative.dependency_paths())
    paths.update(
        {
            Path(__file__),
            ROOT / "scripts/audit_swan_c4_train_narrative_windows.py",
            OUTPUT / "protocol.md",
            ROOT / "scripts/run_segmentation_intervention.py",
            PRIOR / "manifest.json",
            PRIOR / "summary.json",
            PRIOR / "audit.json",
            PRIOR / "recovery_verification.json",
        }
    )
    return sorted(paths)


def dependencies():
    return {
        str(path.relative_to(ROOT)): sha(path)
        for path in dependency_paths()
    }


def prepare(folder=OUTPUT):
    configure_environment()
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        return check(folder)
    import nltk
    import spacy

    nlp = spacy.load("en_core_web_sm")
    sources, screening, scanned_through = acquire_sources(
        folder,
        nltk.sent_tokenize,
        nlp,
    )
    acquisition_path = folder / "source_acquisition.json"
    attempts = json.loads(acquisition_path.read_text())["attempts"]
    manifest = {
        "status": "frozen_after_train_scan_before_api_or_amr",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "allenai/c4",
            "configuration": "realnewslike",
            "revision": REVISION,
            "shards": [shard_name(index) for index in SHARD_INDICES],
            "first_shard_excluded_rows": FIRST_SHARD_SKIP,
            "maximum_eligible_rows": MAX_ELIGIBLE_ROWS,
            "scanned_through": scanned_through,
            "selection": "first five frozen train narrative windows",
            "source_acquisition_sha256": sha(acquisition_path),
            "dataset_stream_http_requests": sum(
                row["shards_opened"] for row in attempts
            ),
        },
        "screening": screening,
        "sources": sources,
        "protocol": {
            "workflow": "swan_c4_train_narrative_window_preflight",
            "target_documents": DOCUMENTS,
            "window_sentences": narrative.WINDOW_SENTENCES,
            "window_total_word_range": [
                narrative.MIN_TOTAL_WORDS,
                narrative.MAX_TOTAL_WORDS,
            ],
            "short_slot_word_range": [
                narrative.MIN_SHORT_WORDS,
                narrative.MAX_SHORT_WORDS,
            ],
            "minimum_short_slots": narrative.MIN_SHORT_SLOTS,
            "narrative_filters": {
                "maximum_source_lines": narrative.MAX_SOURCE_LINES,
                "minimum_multi_sentence_lines": (
                    narrative.MIN_MULTI_SENTENCE_LINES
                ),
                "minimum_verbal_roots": narrative.MIN_VERBAL_ROOTS,
                "minimum_unique_prefix2": narrative.MIN_UNIQUE_PREFIX2,
                "maximum_prefix3_frequency": (
                    narrative.MAX_PREFIX3_FREQUENCY
                ),
                "maximum_pair_jaccard": narrative.MAX_PAIR_JACCARD,
                "minimum_adjacent_overlap": (
                    narrative.MIN_ADJACENT_OVERLAP
                ),
            },
            "quality": (
                "frozen narrative window pass and all style dimensions >=4"
            ),
            "graph_scale": {
                "minimum_compatible_slots": mixed.MIN_SCALE_COMPATIBLE,
                "minimum_compatible_fraction": mixed.MIN_SCALE_FRACTION,
                "median_triples_range": [
                    mixed.MIN_MEDIAN_TRIPLES,
                    mixed.MAX_MEDIAN_TRIPLES,
                ],
                "bank_sha256": sha(mixed.BANK_PATH),
                "s2match_optimization": False,
            },
            "continuation_gate": {
                "minimum_joint_documents": mixed.MIN_JOINT_DOCUMENTS,
                "denominator": DOCUMENTS,
            },
            "api_cap": API_CAP,
            "new_generation_calls": 0,
            "no_watermark_injection_detection_or_attack": True,
        },
        "dependencies": dependencies(),
        "packages": package_versions(),
    }
    freeze(manifest_path, manifest)
    return manifest


def check(folder=OUTPUT):
    configure_environment()
    import nltk
    import spacy

    manifest = json.loads((folder / "manifest.json").read_text())
    if (
        manifest["protocol"]["workflow"]
        != "swan_c4_train_narrative_window_preflight"
    ):
        raise ValueError("Unexpected workflow")
    if sha(folder / "source_acquisition.json") != manifest["dataset"][
        "source_acquisition_sha256"
    ]:
        raise ValueError("Source acquisition journal changed")
    for relative, expected in manifest["dependencies"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Frozen dependency changed: {relative}")
    if package_versions() != manifest["packages"]:
        raise ValueError("Frozen package versions changed")
    nlp = spacy.load("en_core_web_sm")
    for source in manifest["sources"]:
        if (
            source["shard_index"] not in SHARD_INDICES
            or (
                source["shard_index"] == 0
                and source["shard_row"] < FIRST_SHARD_SKIP
            )
            or text_sha(source["source"]) != source["source_sha256"]
            or normalized_hash(source["source"])
            != source["normalized_source_sha256"]
        ):
            raise ValueError("Frozen source identity changed")
        rebuilt = narrative.find_window(
            source["source"],
            nltk.sent_tokenize,
            nlp,
        )
        if rebuilt != source["window"]:
            raise ValueError("Frozen narrative window changed")
    return manifest


def summarize(folder=OUTPUT):
    manifest = json.loads((folder / "manifest.json").read_text())
    selected = len(manifest["sources"])
    if selected < DOCUMENTS:
        result = {
            "status": "complete_corpus_gate_failed",
            "selected_documents": selected,
            "target_documents": DOCUMENTS,
            "screening": manifest["screening"],
            "quality_pass": 0,
            "graph_pass": 0,
            "joint_pass": 0,
            "continuation_gate": False,
            "decision": "stop_c4_train_narrative_window_route",
            "api_attempts": len(list((folder / "calls").glob("*.json"))),
            "neural_parser_sentences": 0,
            "new_s2match_calls": 0,
            "new_watermark_injection_detection_or_attack_calls": 0,
            "manifest_sha256": sha(folder / "manifest.json"),
        }
        write_json(folder / "summary.json", result)
        write_report(folder, result)
        return result
    rows = []
    for source in manifest["sources"]:
        generated = json.loads(
            (folder / "generated" / f"{source['prompt_id']}.json").read_text()
        )
        review = json.loads(
            (folder / "reviews" / f"{source['prompt_id']}.json").read_text()
        )
        parser_path = folder / "parser" / f"{source['prompt_id']}.json"
        graph = (
            mixed.graph_document(json.loads(parser_path.read_text()))
            if parser_path.exists()
            else None
        )
        rows.append(
            {
                "prompt_id": source["prompt_id"],
                "shard_index": source["shard_index"],
                "shard_row": source["shard_row"],
                "words": generated["structure"]["words"],
                "short_slots": generated["structure"]["short_slots"],
                "narrative": source["window"]["narrative"],
                "quality_gates": review["gates"],
                "quality_pass": review["quality_pass"],
                "graph": graph,
                "joint_pass": bool(
                    review["quality_pass"] and graph and graph["pass"]
                ),
            }
        )
    quality_pass = sum(row["quality_pass"] for row in rows)
    parser_docs = sum(row["graph"] is not None for row in rows)
    complete = len(rows) == DOCUMENTS and parser_docs == quality_pass
    joint = sum(row["joint_pass"] for row in rows)
    result = {
        "status": "complete" if complete else "incomplete",
        "selected_documents": selected,
        "target_documents": DOCUMENTS,
        "screening": manifest["screening"],
        "documents": rows,
        "quality_pass": quality_pass,
        "graph_pass": sum(
            bool(row["graph"] and row["graph"]["pass"]) for row in rows
        ),
        "joint_pass": joint,
        "continuation_gate": bool(
            complete and joint >= mixed.MIN_JOINT_DOCUMENTS
        ),
        "decision": (
            "c4_train_narrative_windows_support_partial_injection_preflight"
            if complete and joint >= mixed.MIN_JOINT_DOCUMENTS
            else "stop_c4_train_narrative_window_route"
            if complete
            else "pending"
        ),
        "api_attempts": len(list((folder / "calls").glob("*.json"))),
        "neural_parser_sentences": sum(
            len(json.loads(path.read_text())["sentences"])
            for path in (folder / "parser").glob("*.json")
        ),
        "new_s2match_calls": 0,
        "new_watermark_injection_detection_or_attack_calls": 0,
        "manifest_sha256": sha(folder / "manifest.json"),
    }
    write_json(folder / "summary.json", result)
    if complete:
        write_report(folder, result)
    return result


def write_report(folder, result):
    lines = ["# SWAN C4-train narrative-window preflight", ""]
    if result["selected_documents"] < DOCUMENTS:
        lines.extend(
            [
                f"Corpus gate: **{result['selected_documents']}/{DOCUMENTS}** "
                "documents found.",
                "",
                f"- Eligible rows scanned: "
                f"**{result['screening']['eligible_rows_scanned']}**.",
                f"- Base-window rows before narrative filters: "
                f"**{result['screening']['base_window_rows']}**.",
                "- No quality API, AMR parser, S2Match, injection, detection, "
                "or attack call was run.",
                f"- Decision: `{result['decision']}`.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"Quality-valid windows: **{result['quality_pass']}/"
                f"{DOCUMENTS}**; joint quality and graph pass: "
                f"**{result['joint_pass']}/{DOCUMENTS}**.",
                "",
                "| Document | Words | Short slots | Verbal roots | "
                "Adjacent overlap | Quality | Scale-compatible | Joint |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in result["documents"]:
            graph = row["graph"]
            lines.append(
                f"| {row['prompt_id']} | {row['words']} | "
                f"{row['short_slots']} | "
                f"{row['narrative']['verbal_roots']} | "
                f"{row['narrative']['adjacent_overlap_pairs']} | "
                f"{'yes' if row['quality_pass'] else 'no'} | "
                f"{graph['scale_compatible'] if graph else 'n/a'} | "
                f"{'yes' if row['joint_pass'] else 'no'} |"
            )
        lines.extend(
            [
                "",
                f"- API calls: **{result['api_attempts']}**; local AMR parser "
                f"sentences: **{result['neural_parser_sentences']}**; "
                "S2Match calls: **0**.",
                "- No injection, detection, attack, TPR, FPR, or ASR was run.",
                f"- Decision: `{result['decision']}`.",
                "",
            ]
        )
    (folder / "report.md").write_text("\n".join(lines))


def run(folder, key):
    manifest = check(folder)
    mixed.freeze_documents(folder, manifest)
    if len(manifest["sources"]) < DOCUMENTS:
        return summarize(folder)
    with Heartbeat(folder) as heartbeat:
        api = mixed.Journal(folder, key, heartbeat)
        probe = api.request(
            "probe",
            request_body("Reply exactly OK.", 0, 16),
        )
        if probe["finish_reason"] != "stop" or probe["content"].strip() != "OK":
            raise ValueError("Unexpected connectivity response")
        api.model_returned = probe["model"]
        runtime = freeze(
            mixed.runtime_path(folder),
            {
                "manifest_sha256": sha(folder / "manifest.json"),
                "model_returned": probe["model"],
                "probe_id": probe["id"],
            },
        )
        mixed.review_all(folder, manifest, api, runtime, heartbeat)
        parser_calls = mixed.parse_quality_documents(
            folder,
            manifest,
            runtime,
            heartbeat,
        )
        heartbeat.update(
            stage="complete",
            new_api_calls=api.sent,
            new_neural_parser_calls=parser_calls,
            new_s2match_calls=0,
        )
    return summarize(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "summarize"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    configure_environment()
    with claim(args.output / ".runner.lock"):
        if args.stage == "prepare":
            manifest = prepare(args.output)
            result = {
                "status": "prepared",
                "selected_documents": len(manifest["sources"]),
                "scanned_through": manifest["dataset"]["scanned_through"],
                "manifest_sha256": sha(args.output / "manifest.json"),
            }
        elif args.stage == "run":
            key = (
                read_credential(args.key_file)
                if args.key_file
                else os.environ.get("DEEPSEEK_API_KEY", "")
            )
            try:
                result = run(args.output, key)
            finally:
                if (args.output / "manifest.json").exists():
                    print(json.dumps(summarize(args.output), indent=2), flush=True)
        else:
            result = summarize(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
