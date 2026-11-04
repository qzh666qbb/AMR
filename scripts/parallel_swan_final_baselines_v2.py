"""Eight-worker scheduler amendment for the frozen final SWAN baselines."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import traceback

from allocate_swan_final_sources import OUTPUT, check_allocations, check_runtime
from feedback2_runtime import CachedDetector, claim
from parallel_swan_final_baselines import (
    assigned_sources,
    checkpoint_inventory,
)
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


WORKERS = 8
AMENDMENT = "parallel_baseline_scheduler_w8.json"
PREVIOUS_AMENDMENT = "parallel_baseline_scheduler.json"


def amendment_payload(folder: Path) -> dict:
    previous = folder / "implementation_amendments" / PREVIOUS_AMENDMENT
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_eight_worker_resume",
        "scope": "execution_scheduler_only",
        "scientific_protocol_change": False,
        "reason": (
            "Four-worker execution used substantially fewer than the host's "
            "14 logical CPUs. Eight workers change only source scheduling."
        ),
        "runtime_sha256": sha(folder / "runtime.json"),
        "model_probe_sha256": sha(folder / "model_probe.json"),
        "previous_scheduler_amendment_sha256": sha(previous),
        "frozen_baseline_runner_sha256": sha(
            ROOT / "scripts/run_swan_final_baselines.py"
        ),
        "launcher_sha256": sha(Path(__file__)),
        "test_sha256": sha(
            ROOT / "scripts/test_swan_final_parallel_v2.py"
        ),
        "workers": WORKERS,
        "partition": (
            "source ordinal modulo 8; each source retains its original "
            "eight-slot sequential order"
        ),
        "semantic_functions": [
            "run_swan_final_baselines.run_baseline",
            "run_swan_final_baselines.PaidCallJournal",
        ],
        "pre_resume_checkpoints": checkpoint_inventory(folder),
    }


def freeze_amendment(folder: Path) -> dict:
    check_runtime(folder)
    check_allocations(folder)
    load_model_probe(folder)
    path = folder / "implementation_amendments" / AMENDMENT
    if path.exists():
        return check_amendment(folder)
    previous = folder / "implementation_amendments" / PREVIOUS_AMENDMENT
    if not previous.exists():
        raise RuntimeError("The four-worker scheduler amendment is missing")
    return freeze(path, amendment_payload(folder))


def check_amendment(folder: Path) -> dict:
    path = folder / "implementation_amendments" / AMENDMENT
    saved = json.loads(path.read_text())
    expected = {
        "status": "frozen_before_eight_worker_resume",
        "scope": "execution_scheduler_only",
        "scientific_protocol_change": False,
        "runtime_sha256": sha(folder / "runtime.json"),
        "model_probe_sha256": sha(folder / "model_probe.json"),
        "previous_scheduler_amendment_sha256": sha(
            folder / "implementation_amendments" / PREVIOUS_AMENDMENT
        ),
        "frozen_baseline_runner_sha256": sha(
            ROOT / "scripts/run_swan_final_baselines.py"
        ),
        "launcher_sha256": sha(Path(__file__)),
        "test_sha256": sha(
            ROOT / "scripts/test_swan_final_parallel_v2.py"
        ),
        "workers": WORKERS,
    }
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError(f"Eight-worker amendment changed at {key}")
    return saved


def worker_main(folder_text: str, key: str, worker: int, queue) -> None:
    folder = Path(folder_text)
    try:
        configure_environment()
        source_manifest, _ = check_allocations(folder)
        probe = load_model_probe(folder)
        amendment = check_amendment(folder)
        detector = CachedDetector(
            folder,
            digest(
                {
                    "runtime_sha256": amendment["runtime_sha256"],
                    "model_probe_sha256": amendment[
                        "model_probe_sha256"
                    ],
                }
            ),
        )
        bank, examples = load_generation_assets(detector)
        sources = assigned_sources(
            source_manifest["sources"],
            worker,
            WORKERS,
        )
        api = PaidCallJournal(
            folder,
            key,
            max_new_calls=len(sources) * 8 * 10,
            expected_model=probe["model_returned"],
        )
        for completed, source in enumerate(sources, start=1):
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
            print(
                json.dumps(
                    {
                        "stage": "parallel_baseline_w8",
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
                "completed": len(sources),
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


def run_parallel(folder: Path, key: str) -> list[dict]:
    freeze_amendment(folder)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(
            target=worker_main,
            args=(str(folder), key, worker, queue),
            name=f"swan-final-w8-{worker}",
        )
        for worker in range(WORKERS)
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
            "Eight-worker baseline scheduler failed:\n"
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
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze-amendment":
        result = freeze_amendment(args.output)
    elif args.stage == "check":
        result = check_amendment(args.output)
    else:
        if args.key_file is None:
            parser.error("run requires --key-file")
        result = run_parallel(
            args.output,
            read_credential(args.key_file),
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
