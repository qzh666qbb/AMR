# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
parse_human_text.py

Parse human paragraphs into AMR graphs for use in watermark detection.
Each paragraph is sentence-tokenized, truncated to 5 sentences, and parsed into
normalized template AMRs. Output is saved as a single JSON file.

Supports GPU-parallel parsing across multiple GPUs, single-GPU, or CPU-only mode.
By default, auto-detects available GPUs and falls back to CPU if none are found.

Usage:
    # Auto-detect GPUs (or fall back to CPU)
    python detection/parse_human_text.py --dataset_path data/c4-human

    # Use specific GPUs
    python detection/parse_human_text.py --dataset_path data/c4-human --gpu_ids 0,1

    # CPU-only
    python detection/parse_human_text.py --dataset_path data/c4-human --gpu_ids cpu
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
    generate_template_amr,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True, help='HF dataset of human paragraphs (use utils/load_c4_realnews_data.py to download)')
    parser.add_argument('--num_paragraphs', type=int, default=250, help='Number of human paragraphs to parse')
    parser.add_argument('--gpu_ids', default=None,
                        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3'). "
                             "If not specified, uses all available GPUs. Use 'cpu' for CPU-only mode.")
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Model batch size for amrlib')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output JSON path. '
                             'Defaults to amr_bank/artifacts/human_parsed_amrs_<N>.json '
                             'where N is the number of parsed paragraphs.')
    return parser.parse_args()

def finalize_amr(raw_amr: str) -> str:
    """
    Convert raw_amr to final form:
      1) Trim to from the first '('
      2) Remove comments
      3) normalize -> template -> normalize
    """
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
    """
    Parse a list of paragraphs on a single device. Return list-of-lists of final AMRs.
    """
    print(f"[{device}] Loading stog model (batch_size={batch_size})...")
    stog = amrlib.load_stog_model(device=device, batch_size=batch_size)

    results = []
    for paragraph in paragraphs:
        sents = sent_tokenize(paragraph)
        # Truncate to 5 sentences total
        sents = sents[:5]
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
    """
    Distribute paragraphs across multiple devices. Return list-of-lists of final AMRs.
    """
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

    results_list = [None]*len(shards)
    with ProcessPoolExecutor(max_workers=len(shards)) as executor:
        futs = []
        for idx, (device, shard_docs) in enumerate(shards):
            futs.append(executor.submit(bulk_parse_paragraphs, shard_docs, device, batch_size))
        for i, f in enumerate(as_completed(futs)):
            results_list[i] = f.result()

    # Merge results in shard order
    merged = []
    for (device, shard_docs), shard_amrs in zip(shards, results_list):
        merged.extend(shard_amrs)

    return merged

def main():
    args = parse_args()

    # 1) Load human dataset
    ds = load_from_disk(args.dataset_path)
    n = args.num_paragraphs
    human_paragraphs = ds['text'][:n]
    assert len(human_paragraphs) == n, \
        f"Expected exactly {n} paragraphs, found {len(human_paragraphs)}"

    print(f"Loaded {len(human_paragraphs)} human paragraphs from '{args.dataset_path}'")

    # 2) Parse on device(s)
    devices = resolve_device_list(args.gpu_ids)
    print(f"Parsing {n} paragraphs on {devices} (batch_size={args.batch_size})...")
    final_human_amrs = parallel_parse_docs(human_paragraphs, devices, args.batch_size)
    assert len(final_human_amrs) == n

    # 3) Save parsed AMRs
    if args.output_path is not None:
        outfile = args.output_path
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        outfile = os.path.join(project_root, "amr_bank", "artifacts",
                               f"human_parsed_amrs_{n}.json")
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, 'w') as f:
        json.dump(final_human_amrs, f)
    print(f"[Saved] {n} paragraphs of human AMRs to '{outfile}'")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
