"""Run frozen final-baseline semantics over disjoint source partitions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import traceback

from allocate_swan_final_sources import OUTPUT, check_allocations, check_runtime
from feedback2_runtime import CachedDetector, claim
from run_confirmation_experiment import digest, freeze
from run_segmentation_intervention import ROOT, sha
from run_swan_final_baselines import (
    PaidCallJournal,
    configure_environment,
    load_generation_assets,
    load_model_probe,
    run_baseline,
)
from supervise_confirmation_experiment import read_credential


DEFAULT_WORKERS = 4
AMENDMENT = "parallel_baseline_scheduler.json"


def checkpoint_inventory(folder: Path) -> dict:
    patterns = {
        "calls": "calls/*.json",
        "attempts": "baseline_attempts/*.json",
        "slots": "baseline_slots/*.json",
        "baselines": "baselines/*.json",
    }
    result = {}
    for label, pattern in patterns.items():
        paths = sorted(folder.glob(pattern))
        entries = [
            {
                "path": str(path.relative_to(folder)),
                "sha256": sha(path),
            }
            for path in paths
        ]
        result[label] = {
            "count": len(entries),
            "digest": digest(entries),
        }
    return result


def amendment_payload(folder: Path, workers: int) -> dict:
    if workers < 2:
        raise ValueError("Parallel scheduler requires at least two workers")
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_parallel_resume",
        "scope": "execution_scheduler_only",
        "scientific_protocol_change": False,
        "runtime_sha256": sha(folder / "runtime.json"),
        "model_probe_sha256": sha(folder / "model_probe.json"),
        "frozen_baseline_runner_sha256": sha(
            ROOT / "scripts/run_swan_final_baselines.py"
        ),
        "parallel_launcher_sha256": sha(Path(__file__)),
        "parallel_test_sha256": sha(
            ROOT / "scripts/test_swan_final_parallel.py"
        ),
        "workers": workers,
        "partition": (
            "source ordinal modulo worker count; each source retains its "
            "original eight-slot sequential order"
        ),
        "worker_new_call_cap": (
            "number of assigned sources times 80; global frozen cap remains "
            "enforced by the baseline call journal"
        ),
        "semantic_functions": [
            "run_swan_final_baselines.run_baseline",
            "run_swan_final_baselines.PaidCallJournal",
        ],
        "pre_resume_checkpoints": checkpoint_inventory(folder),
    }


def freeze_amendment(folder: Path, workers: int) -> dict:
    check_runtime(folder)
    check_allocations(folder)
    load_model_probe(folder)
    path = folder / "implementation_amendments" / AMENDMENT
    if path.exists():
        return check_amendment(folder, workers)
    payload = amendment_payload(folder, workers)
    return freeze(path, payload)


def check_amendment(folder: Path, workers: int) -> dict:
    path = folder / "implementation_amendments" / AMENDMENT
    saved = json.loads(path.read_text())
    stable = {
        "status": "frozen_before_parallel_resume",
        "scope": "execution_scheduler_only",
        "scientific_protocol_change": False,
        "runtime_sha256": sha(folder / "runtime.json"),
        "model_probe_sha256": sha(folder / "model_probe.json"),
        "frozen_baseline_runner_sha256": sha(
            ROOT / "scripts/run_swan_final_baselines.py"
        ),
        "parallel_launcher_sha256": sha(Path(__file__)),
        "parallel_test_sha256": sha(
            ROOT / "scripts/test_swan_final_parallel.py"
        ),
        "workers": workers,
    }
    for key, expected in stable.items():
        if saved.get(key) != expected:
            raise ValueError(f"Parallel amendment changed at {key}")
    return saved


def assigned_sources(sources: list[dict], worker: int, workers: int) -> list[dict]:
    if workers < 1 or worker not in range(workers):
        raise ValueError("Invalid worker partition")
    return [
        source
        for index, source in enumerate(sources)
        if index % workers == worker
    ]


def worker_main(
    folder_text: str,
    key: str,
    worker: int,
    workers: int,
    queue,
) -> None:
    folder = Path(folder_text)
    try:
        configure_environment()
        source_manifest, _ = check_allocations(folder)
        probe = load_model_probe(folder)
        amendment = check_amendment(folder, workers)
        runtime_signature = digest(
            {
                "runtime_sha256": amendment["runtime_sha256"],
                "model_probe_sha256": amendment["model_probe_sha256"],
            }
        )
        detector = CachedDetector(folder, runtime_signature)
        bank, examples = load_generation_assets(detector)
        sources = assigned_sources(
            source_manifest["sources"],
            worker,
            workers,
        )
        api = PaidCallJournal(
            folder,
            key,
            max_new_calls=len(sources) * 8 * 10,
            expected_model=probe["model_returned"],
        )
        completed = 0
        for source in sources:
            with claim(
                folder / "locks" / f'{source["source_id"]}.lock'
            ):
                run_baseline(
                    folder,
                    source,
                    detector,
                    api,
                    bank,
                    examples,
                )
            completed += 1
            print(
                json.dumps(
                    {
                        "stage": "parallel_baseline",
                        "worker": worker,
                        "source_id": source["source_id"],
                        "worker_completed": completed,
                        "worker_total": len(sources),
                        "new_paid_calls": api.new_calls,
                    }
                ),
                flush=True,
            )
        queue.put(
            {
                "worker": worker,
                "status": "complete",
                "completed": completed,
                "new_paid_calls": api.new_calls,
                "cache_counts": detector.counts,
            }
        )
    except BaseException as exc:
        queue.put(
            {
                "worker": worker,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise


def run_parallel(
    folder: Path,
    key: str,
    workers: int,
) -> list[dict]:
    freeze_amendment(folder, workers)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(
            target=worker_main,
            args=(str(folder), key, worker, workers, queue),
            name=f"swan-final-worker-{worker}",
        )
        for worker in range(workers)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    results = [queue.get() for _ in processes]
    failures = [
        result for result in results if result["status"] != "complete"
    ]
    if failures or any(process.exitcode != 0 for process in processes):
        raise RuntimeError(
            "Parallel baseline worker failed:\n"
            + "\n".join(
                result.get("traceback", json.dumps(result))
                for result in failures
            )
        )
    return sorted(results, key=lambda row: row["worker"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("freeze-amendment", "run", "check"),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()

    if args.stage == "freeze-amendment":
        result = freeze_amendment(args.output, args.workers)
    elif args.stage == "check":
        result = check_amendment(args.output, args.workers)
    else:
        if args.key_file is None:
            parser.error("run requires --key-file")
        result = run_parallel(
            args.output,
            read_credential(args.key_file),
            args.workers,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
