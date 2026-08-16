# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
parse_machine_text.py

Stage 1 of the two-stage detection pipeline: parallel AMR parsing of
machine-generated text. Sentence-tokenizes each paragraph, parses to AMR,
applies normalize → template → normalize, and saves as JSON.

Supports GPU-parallel parsing across multiple GPUs, single-GPU, or CPU-only mode.
By default, auto-detects available GPUs and falls back to CPU if none are found.

Usage:
    # Auto-detect GPUs (or fall back to CPU)
    python detection/parse_machine_text.py output/watermarked

    # Use specific GPUs
    python detection/parse_machine_text.py output/watermarked --gpu_ids 0,1,2,3

    # CPU-only
    python detection/parse_machine_text.py output/watermarked --gpu_ids cpu
"""

import argparse
import os
import sys
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datasets import load_from_disk
from nltk.tokenize import sent_tokenize
import amrlib
import multiprocessing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.amr_utils import (
    remove_amr_comments,
    normalize_amr_variables,
    generate_template_amr
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help="Path to HF dataset containing machine paragraphs.")
    parser.add_argument('--gpu_ids', default=None,
                        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3'). "
                             "If not specified, uses all available GPUs. Use 'cpu' for CPU-only mode.")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for amrlib.load_stog_model().")
    parser.add_argument('--output_filename', type=str, default="parsed_amrs.json",
                        help="Name of the output JSON file. Will be saved in the same directory as dataset_path.")
    parser.add_argument('--paraphrased', action='store_true',
                        help="If set, parse the 'para_text' column instead of 'text'.")
    return parser.parse_args()

def finalize_amr(raw_amr: str) -> str:
    if not raw_amr or not raw_amr.strip():
        return ""
    lines = raw_amr.strip().split('\n')
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('('):
            start_idx = i
            break
    if start_idx is None:
        return ""
    amr_str = '\n'.join(lines[start_idx:])
    amr_str = remove_amr_comments(amr_str)
    cand_norm = normalize_amr_variables(amr_str)
    cand_templ = generate_template_amr(cand_norm)
    cand_final = normalize_amr_variables(cand_templ)
    return cand_final.strip()

def bulk_parse_paragraphs(paragraphs, device="cpu", batch_size=32):
    print(f"[{device}] Loading stog model (batch_size={batch_size})...")
    stog = amrlib.load_stog_model(device=device, batch_size=batch_size)

    results = []
    for paragraph in paragraphs:
        sents = sent_tokenize(paragraph)
        raw_amrs = stog.parse_sents(sents)
        final_amrs = [finalize_amr(a) for a in raw_amrs]
        results.append(final_amrs)
    return results

def resolve_device_list(gpu_ids_arg):
    """Resolve --gpu_ids argument to a list of device strings."""
    import torch
    if gpu_ids_arg is not None and gpu_ids_arg.strip().lower() == "cpu":
        return ["cpu"]
    if gpu_ids_arg is not None:
        return [f"cuda:{int(x)}" for x in gpu_ids_arg.split(',') if x.strip()]
    # Auto-detect: use all available GPUs, fall back to CPU
    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ["cpu"]

def parallel_parse_docs(paragraphs, devices, batch_size):
    n = len(paragraphs)
    g = len(devices)
    if n == 0:
        return []

    # Single device: no need for multiprocessing
    if g == 1:
        return bulk_parse_paragraphs(paragraphs, device=devices[0], batch_size=batch_size)

    chunk_size = (n + g - 1) // g
    shards = []
    for i, device in enumerate(devices):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        shard_paras = paragraphs[start:end]
        if shard_paras:
            shards.append((device, shard_paras))

    results_list = [None] * len(shards)
    with ProcessPoolExecutor(max_workers=len(shards)) as executor:
        futs = []
        for idx, (device, shard_docs) in enumerate(shards):
            futs.append(executor.submit(bulk_parse_paragraphs,
                                        shard_docs,
                                        device=device,
                                        batch_size=batch_size))
        for i, f in enumerate(as_completed(futs)):
            results_list[i] = f.result()

    # Merge results in shard order
    merged = []
    for (device, shard_docs), shard_amrs in zip(shards, results_list):
        merged.extend(shard_amrs)
    return merged

def main():
    multiprocessing.set_start_method("spawn", force=True)
    args = parse_args()

    # 1) Load dataset
    ds = load_from_disk(args.dataset_path)

    # 2) Decide which column to parse
    if args.paraphrased:
        col_name = "para_text"
        if col_name not in ds.column_names:
            raise ValueError(f"[ERROR] Dataset does not have '{col_name}' column.")
    else:
        col_name = "text"
        if col_name not in ds.column_names:
            raise ValueError(f"[ERROR] Dataset does not have '{col_name}' column.")

    paragraphs = ds[col_name]
    print(f"Loaded {len(paragraphs)} paragraphs from column '{col_name}' in '{args.dataset_path}'")

    # 3) Resolve devices
    devices = resolve_device_list(args.gpu_ids)
    print(f"Parsing {len(paragraphs)} paragraphs on {devices} (batch_size={args.batch_size})...")

    # 4) Parse in parallel
    final_machine_amrs = parallel_parse_docs(
        paragraphs=paragraphs,
        devices=devices,
        batch_size=args.batch_size
    )
    assert len(final_machine_amrs) == len(paragraphs), \
        "Mismatch in # paragraphs vs. parsed AMRs"

    # 5) Save output in same directory as dataset_path
    dataset_dir = os.path.abspath(args.dataset_path)
    if os.path.isdir(dataset_dir):
        out_dir = dataset_dir
    else:
        out_dir = os.path.dirname(dataset_dir)

    out_path = os.path.join(out_dir, args.output_filename)
    with open(out_path, 'w') as f:
        json.dump(final_machine_amrs, f, indent=2)

    print(f"\n[DONE] Parsed AMRs from '{col_name}' column. Results saved to:\n  {out_path}")

if __name__ == "__main__":
    main()
