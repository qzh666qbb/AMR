"""Wait for the staged worker and run its previously frozen independent audit."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time
import traceback


def main(root, out):
    deadline = time.monotonic() + 24 * 3600
    while time.monotonic() < deadline:
        if any((out / name).exists() for name in ("HALT.json", "STOP.json")):
            print("Worker stopped; final audit not run.", flush=True)
            return
        heartbeat = out / "heartbeat.json"
        ledger = out / "private/RESOURCE_LEDGER.json"
        if (out / "COMPLETE.json").exists() and heartbeat.exists() and ledger.exists():
            if json.loads(heartbeat.read_text()).get("phase") == "complete":
                break
        time.sleep(2)
    else:
        raise RuntimeError("Worker completion wait budget reached")
    source = root / "scripts/audit_multiscale_transfer.py"
    spec = importlib.util.spec_from_file_location("independent_transfer_audit", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module.audit(root, out)
    except BaseException as error:
        module.load_transfer(out).atomic(out / "FINAL_AUDIT.json", {
            "passed": False, "reason_type": type(error).__name__,
            "freeze_sha256": module.load_transfer(out).sha(out / "FREEZE.json"),
            "private_traceback": "private/audit_worker.log"})
        traceback.print_exc()
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    main(args.root.resolve(), args.out.resolve())
