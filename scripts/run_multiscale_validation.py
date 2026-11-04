"""Execute one frozen group of the cached MSA development validation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "experiments/swan_multiscale_attention_development1/validation1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_inputs(protocol):
    checks = dict(protocol["source_sha256"])
    checks[protocol["split_path"]] = protocol["split_sha256"]
    checks[protocol["input_audit_path"]] = protocol["input_audit_sha256"]
    for name, expected in checks.items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f"Frozen input changed: {name}")
    audit = json.loads((ROOT / protocol["input_audit_path"]).read_text())
    for side in ("positive", "negative"):
        for source in audit[side]["sources"]:
            if sha(source["path"]) != source["sha256"]:
                raise RuntimeError(f"Cache changed: {source['path']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("attention", "gate", "baselines"), required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    protocol_path = args.output / "RUN_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text())
    verify_inputs(protocol)
    if args.group == "baselines":
        matrix = [(mode, protocol["seeds"][0]) for mode in protocol["deterministic_modes"]]
    else:
        matrix = [(args.group, seed) for seed in protocol["seeds"]]
    # A group rerun requires a separate output root, never partial overwriting.
    for mode, seed in matrix:
        if (args.output / "runs" / f"{mode}_{seed}").exists():
            raise FileExistsError(f"Run already exists: {mode}_{seed}")
    group_start = time.perf_counter()
    for mode, seed in matrix:
        name = f"{mode}_{seed}"
        run_dir = args.output / "runs" / name
        log_path = args.output / "logs" / f"{name}.log"
        log_path.parent.mkdir(exist_ok=True)
        command = [
            sys.executable, str(ROOT / "scripts/run_multiscale_fusion.py"),
            "--mode", "fit", "--fusion", mode,
            "--split", str(ROOT / protocol["split_path"]),
            "--output", str(run_dir), "--epochs", str(protocol["epochs"]),
            "--batch-size", str(protocol["batch_size"]), "--seed", str(seed),
            "--boot", str(protocol["run_bootstrap_repetitions"]),
        ]
        started = datetime.now(timezone.utc).isoformat()
        begin = time.perf_counter()
        with log_path.open("x") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        metadata = {
            "run": name, "group": args.group, "started_utc": started,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - begin,
            "returncode": result.returncode, "command": command,
            "protocol_sha256": sha(protocol_path),
            "orchestrator_sha256": sha(__file__), "log_sha256": sha(log_path),
            "device": "CPU", "torch_threads_per_process": 2,
        }
        with (log_path.with_suffix(".metadata.json")).open("x") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(f"{name} exit={result.returncode} wall_s={metadata['elapsed_seconds']:.2f}",
              flush=True)
        if result.returncode:
            raise RuntimeError(f"Failed run; inspect {log_path}")
    payload = {
        "group": args.group, "completed_runs": len(matrix),
        "elapsed_seconds": time.perf_counter() - group_start,
        # On macOS ru_maxrss is bytes; this is max child RSS, not summed RSS.
        "maximum_child_peak_rss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "rss_platform": sys.platform, "protocol_sha256": sha(protocol_path),
    }
    with (args.output / f"{args.group}_EXECUTION.json").open("x") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
