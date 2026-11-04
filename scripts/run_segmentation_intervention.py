"""Prepare and run paired sentence/clause SWAN detection without text generation."""

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
SHARD = "realnewslike/c4-train.00000-of-00512.json.gz"
PRE_METADATA_FIX_SHA = "5a25d772dcee30334e43640bcdcadd91c62fc2af1323207dc571047ff7c7e72b"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def amr_body(raw):
    """Match SWAN finalize_amr: metadata can contain unprefixed text lines."""
    lines = raw.strip().splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip().startswith("(")), None)
    if start is None:
        raise ValueError("AMR output contains no graph")
    return "\n".join(line for line in lines[start:] if not line.strip().startswith("#"))


def compatible_runtime(saved):
    """Permit only the audited metadata extraction fix, not arbitrary versions."""
    runtime = saved["runtime"]
    if text_sha(json.dumps(runtime, sort_keys=True)) != saved["runtime_signature"]:
        raise ValueError("Runtime signature does not match its contents")
    if runtime["script_sha256"] not in (PRE_METADATA_FIX_SHA, sha(Path(__file__))):
        raise ValueError("Unrecognized detector implementation")
    if runtime["script_sha256"] == PRE_METADATA_FIX_SHA:
        for variant in saved["variants"].values():
            for detail in variant["details"]:
                previous = "\n".join(line for line in detail["raw_amr"].splitlines()
                                     if not line.strip().startswith("#"))
                if previous.strip() != amr_body(detail["raw_amr"]).strip():
                    raise ValueError("Old metadata extraction affected this result; rerun in a fresh folder")
    return {k: v for k, v in runtime.items() if k != "script_sha256"}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def partition(text, boundaries, min_words=4):
    """Split at character offsets without deleting lexical content or punctuation."""
    cuts = [0]
    for boundary in sorted(set(boundaries)):
        if 0 < boundary < len(text) and len(text[cuts[-1]:boundary].split()) >= min_words:
            cuts.append(boundary)
    if len(cuts) > 1 and len(text[cuts[-1]:].split()) < min_words:
        cuts.pop()
    pieces = [text[a:b].strip() for a, b in zip(cuts, cuts[1:] + [len(text)])]
    if re.sub(r"\s+", "", "".join(pieces)) != re.sub(r"\s+", "", text):
        raise ValueError("Segmentation changed non-whitespace content")
    return pieces


def segment(text, sent_tokenize, nlp):
    sentences = sent_tokenize(text)
    clauses = []
    for sentence in sentences:
        doc = nlp(sentence)
        boundaries = []
        for token in doc:
            if token.text == ";":
                boundaries.append(token.idx + len(token.text))
            if token.dep_ == "conj" and token.pos_ in ("VERB", "AUX"):
                children = list(token.children)
                if any(child.dep_ in ("nsubj", "nsubjpass", "csubj") for child in children):
                    boundaries.append(min(t.idx for t in token.subtree))
        clauses.extend(partition(sentence, boundaries))
    return {"sentence": sentences, "clause": clauses}


def human_records(count):
    import requests

    url = f"https://huggingface.co/datasets/allenai/c4/resolve/{REVISION}/{SHARD}"
    with requests.get(url, stream=True, timeout=(20, 60)) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        with gzip.GzipFile(fileobj=response.raw) as stream:
            for index, line in enumerate(stream):
                # Match the historical loader's skip, without claiming identical
                # human text until original artifact identities can be verified.
                if index >= 250:
                    yield index, json.loads(line)
                    count -= 1
                    if count == 0:
                        return
    raise ValueError("Not enough reference records")


def prepare(args):
    import nltk
    import spacy

    output = args.output / "manifest.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen manifest: {output}")
    nlp = spacy.load("en_core_web_sm")
    reviews_path = ROOT / "experiments/planE_e2_joint250/quality_aware_reviews.json"
    reviews = json.loads(reviews_path.read_text())
    groups = {}
    for row in reviews:
        groups.setdefault(row["index"], []).append(row)
    indices = sorted(groups, key=lambda index: text_sha(f"segmentation-v1:{index}"))[:args.attack_count]
    if len(indices) != args.attack_count:
        raise ValueError("Insufficient attack paragraphs")
    records = []
    for index in indices:
        pool = groups[index]
        row = min([r for r in pool if r["quality_valid"]] or pool,
                  key=lambda r: (r["z_score"], r["candidate"]))
        records.append({
            "id": f"attack-{index}", "cohort": "attack", "index": index,
            "candidate": row["candidate"], "position": row["position"],
            "quality_valid": row["quality_valid"], "historical_z": row["z_score"],
            "text": row["attack"], "text_sha256": text_sha(row["attack"]),
            "source_sha256": text_sha(row["source"]),
            "segments": segment(row["attack"], nltk.sent_tokenize, nlp),
        })
    human_hashes = set()
    excluded = {text_sha(r[field]) for r in reviews for field in ("source", "attack")}
    if args.human_count:
        for ordinal, (index, row) in enumerate(human_records(2 * args.human_count)):
            text = " ".join(nltk.sent_tokenize(row["text"])[:5])
            digest = text_sha(text)
            if not text or digest in human_hashes or digest in excluded:
                raise ValueError("Empty or duplicate reference; define a new sampling protocol")
            human_hashes.add(digest)
            cohort = "human_calibration" if ordinal < args.human_count else "human_test"
            records.append({
                "id": f"{cohort}-{index}", "cohort": cohort, "shard_row": index,
                "text": text, "text_sha256": digest, "raw_text_sha256": text_sha(row["text"]),
                "segments": segment(text, nltk.sent_tokenize, nlp),
            })
    protocol = {
        "version": 1, "attack_sampling": "First 30 by SHA256(segmentation-v1:index); no success filtering",
        "selection": "Quality then min(z,candidate); min-score fallback remains quality-invalid",
        "segmentation": "NLTK punkt_tab sentences; split semicolon and verbal conj with explicit subject; min 4 words",
        "text_policy": "No lexical edits; references frozen to first 5 original sentences before either segmentation",
        "human_source": {"dataset": "allenai/c4", "revision": REVISION, "shard": SHARD, "skip": 250},
        "human_reference_note": "New provenance-pinned samples, not asserted to be recovered historical references",
        "calibration": "Separate cohorts; later choose threshold on calibration only for <=1% empirical FPR; strict z>threshold",
        "reporting": "Pilot paired scores only until both human cohorts are fully detected and calibrated",
        "attack_count": args.attack_count, "human_count_per_split": args.human_count,
        "spacy": spacy.__version__, "spacy_model": nlp.meta["version"], "nltk": nltk.__version__,
        "reviews_sha256": sha(reviews_path), "script_sha256": sha(Path(__file__)),
    }
    write_json(output, {"protocol": protocol, "records": records})
    print(json.dumps({"records": len(records), "changed_attack_segmentations": sum(
        r["cohort"] == "attack" and r["segments"]["sentence"] != r["segments"]["clause"] for r in records),
        "output": str(output)}, indent=2), flush=True)


def detect(args):
    import amrlib
    import penman
    import torch

    sys.path.insert(0, str(ROOT / "third_party/SWAN"))
    from utils.amr_utils import (
        normalize_amr_variables, generate_template_amr,
        load_amr_bank, compute_s2match_score,
    )

    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records = [r for r in manifest["records"] if r["cohort"] == args.cohort][args.offset:args.offset + args.limit]
    if not records:
        raise ValueError("No records selected")
    bank_path = ROOT / "third_party/SWAN/amr_bank/banks/amr_bank_50.json"
    bank = load_amr_bank(str(bank_path))
    if len(bank) != 50:
        raise ValueError("Expected the official 50-template bank")
    vector_path = Path(os.environ["GLOVE_VECTORS_PATH"])
    runtime = {
        "manifest_sha256": sha(manifest_path), "bank_sha256": sha(bank_path),
        "vectors_sha256": sha(vector_path), "device": args.device, "num_beams": 4,
        "batch_size": 1, "lambda": .25, "match_threshold": .65,
        "model_files": {p.name: sha(p) for p in sorted(args.model_dir.glob("*")) if p.is_file()},
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "amrlib", "transformers", "numpy", "scipy", "spacy", "en_core_web_sm", "nltk")},
        "s2match_randomness": "Official random restarts reseed internally; not deterministic",
        "script_sha256": sha(Path(__file__)),
    }
    signature = text_sha(json.dumps(runtime, sort_keys=True))
    torch.set_num_threads(4)
    parser = amrlib.load_stog_model(model_dir=str(args.model_dir), device=args.device, batch_size=1,
                                   tok_name_or_path=str(args.tokenizer_dir))
    for record in records:
        destination = args.output / "detected" / f'{record["id"]}.json'
        if destination.exists():
            saved = json.loads(destination.read_text())
            if compatible_runtime(saved) != {k: v for k, v in runtime.items() if k != "script_sha256"}:
                raise ValueError(f"Runtime changed; choose a fresh output directory: {destination}")
            print(f'skip complete {record["id"]}', flush=True)
            continue
        started = time.monotonic()
        parsed_cache = {}
        scored_cache = {}
        variants = {}
        for mode, pieces in record["segments"].items():
            details = []
            for piece in pieces:
                if piece not in parsed_cache:
                    length = len(parser.tokenizer(piece)["input_ids"])
                    if length > parser.max_sent_len:
                        raise ValueError(f'{record["id"]}: tokenizer truncation would occur')
                    raw = parser.parse_sents([piece])[0]
                    if not raw:
                        raise ValueError(f'{record["id"]}: AMR parse failure')
                    graph = amr_body(raw)
                    penman.decode(graph)
                    final = normalize_amr_variables(generate_template_amr(normalize_amr_variables(graph)))
                    penman.decode(final)
                    parsed_cache[piece] = {"raw_amr": raw, "amr": final, "input_tokens": length}
                final = parsed_cache[piece]["amr"]
                if final not in scored_cache:
                    scores = [float(compute_s2match_score(final, template)) for template in bank]
                    if not all(math.isfinite(s) and 0 <= s <= 1 for s in scores):
                        raise ValueError("Invalid S2Match score")
                    scored_cache[final] = {"score": max(scores), "bank_scores": scores}
                details.append({"text": piece, **parsed_cache[piece], **scored_cache[final]})
            n = len(details)
            if not n:
                raise ValueError("Empty segmentation")
            green = sum(r["score"] >= .65 for r in details)
            variants[mode] = {"n": n, "green": green, "green_rate": green / n,
                              "z": (green - .25 * n) / math.sqrt(n * .25 * .75), "details": details}
        result = {"record": record, "variants": variants, "runtime": runtime,
                  "runtime_signature": signature, "elapsed_seconds": time.monotonic() - started}
        write_json(destination, result)
        print(json.dumps({"id": record["id"], "elapsed": result["elapsed_seconds"],
                          "scores": {mode: {k: v[k] for k in ("n", "green", "z")} for mode, v in variants.items()}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "detect"))
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/segmentation_intervention")
    parser.add_argument("--attack-count", type=int, default=30)
    parser.add_argument("--human-count", type=int, default=250, help="Per calibration/test split; 0 for attack-only smoke")
    parser.add_argument("--cohort", choices=("attack", "human_calibration", "human_test"), default="attack")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/model_parse_xfm_bart_large-v0_1_0")
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "models/bart-large-tokenizer")
    args = parser.parse_args()
    if args.attack_count < 1 or args.human_count < 0 or args.limit < 1 or args.offset < 0:
        parser.error("Counts must be positive (human-count and offset may be zero)")
    os.environ.setdefault("NLTK_DATA", str(ROOT / ".nltk_data"))
    os.environ.setdefault("GLOVE_VECTORS_PATH", str(ROOT / "vectors/glove.6B.100d.txt"))
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
    (prepare if args.stage == "prepare" else detect)(args)


if __name__ == "__main__":
    main()
