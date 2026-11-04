"""Versioned v2 repair; reuse the frozen at-most-once runtime without editing v1."""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import shlex
import subprocess
import sys

import run_semantic_fact_carrier_pilot as runtime
from semantic_fact_carrier_v2 import (
    ALPHABET_VERSION, SYMBOLS, generation_messages, paraphrase_messages, plan,
)

ROOT = runtime.ROOT
OUT = ROOT / "experiments/semantic_fact_carrier_pilot2"
PRIVATE = Path.home() / ".cache/amr/semantic_fact_carrier_pilot2"
SOURCES = [
    "scripts/run_semantic_fact_carrier_pilot.py",
    "src/core/semantic_fact_carrier.py",
    "src/core/semantic_fact_carrier_v2.py",
    "scripts/run_semantic_fact_carrier_pilot2.py",
    "scripts/test_semantic_fact_carrier_pilot2.py",
]
AUTHORIZATION = (
    "User 2026-10-26 explicitly instructed: execute carrier v2 repair, "
    "new 96 originals plus 96 paraphrases, model development assessment; "
    "fact-layer diagnosis after carrier verification."
)


def configure():
    runtime.OUT, runtime.PRIVATE = OUT, PRIVATE
    runtime.ALPHABET_VERSION, runtime.SYMBOLS = ALPHABET_VERSION, SYMBOLS
    runtime.plan = plan
    runtime.generation_messages = generation_messages
    runtime.paraphrase_messages = paraphrase_messages


configure()


def prepare(out, private):
    if (out / "FREEZE.json").exists() or (private / "FREEZE.json").exists():
        raise RuntimeError("Existing freeze is immutable; do not overwrite or resume v1")
    required = ["PROTOCOL.md", "PRICE_SOURCE.json", "TEST_RESULT.json"]
    if any(not (out / name).is_file() for name in required):
        raise RuntimeError("Protocol, price source and passed test evidence required")
    tests = runtime.load(out / "TEST_RESULT.json")
    hashes = {name: runtime.sha(ROOT / name) for name in SOURCES}
    if tests["status"] != "PASS" or tests["tests_run"] < 12 or tests["code_sha256"] != hashes:
        raise RuntimeError("Meaningful v2 tests missing or source drift after tests")
    rows = plan()
    if len(rows) != 96 or Counter(r["intended_symbol"] for r in rows) != dict.fromkeys("ABCD", 24):
        raise RuntimeError("Incorrect fixed denominator")
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    runtime.atomic(private / "PLAN.json", [
        {"row": row, "original_messages": generation_messages(row)} for row in rows])
    runtime.atomic(out / "ALPHABET.json", {
        "version": ALPHABET_VERSION, "symbols": SYMBOLS, "fields": runtime.FIELDS,
        "same_direction_status_mapping_as_v1": True, "erasure_is_failure": True,
        "narrator_level_current_plan_and_nonoccurrence_required": True,
    })
    runtime.atomic(out / "PLAN_METADATA.json", {
        "split": "repair_development_only", "originals": 96, "paraphrases": 96,
        "scenes": 24, "symbols_per_scene": list(SYMBOLS),
        "planned_ids": [row["item_id"] for row in rows],
        "scene_families_reused_from_v1": True, "new_generation_receipts_required": True,
        "held_out_generalization_claimed": False,
        "one_sample_per_request_no_resampling": True,
        "paraphrase_source": "literal complete original only; no intended target",
    })
    freeze = {
        "created_at": runtime.now(), "version": out.name, "alphabet": ALPHABET_VERSION,
        "api_url": runtime.API_URL, "parameters": runtime.PARAMETERS,
        "budget": runtime.BUDGET, "gates": runtime.GATES,
        "private_plan_sha256": runtime.sha(private / "PLAN.json"),
        "code_sha256": hashes,
        "public_sha256": {name: runtime.sha(out / name)
                          for name in [*required, "ALPHABET.json", "PLAN_METADATA.json"]},
        "authorization": AUTHORIZATION,
        "formal_human_gate_required": True,
        "model_diagnostic_is_not_formal_human_gate": True,
        "historical_runs_resumed": False,
    }
    runtime.atomic(private / "FREEZE.json", freeze)
    runtime.atomic(out / "FREEZE.json", freeze)
    return {"status": "FROZEN", "version": out.name, "planned_calls": 192}


def start(out, private, access):
    runtime.precheck(out, private, access)
    config = runtime.load(access)
    directory = private / "supervisor"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / "supervisord.conf"
    if (directory / "supervisord.pid").exists():
        raise RuntimeError("Supervisor exists; inspect its state")
    command = [config["python"], str(Path(__file__).resolve()), "run",
               "--out", str(out), "--private", str(private), "--access-config", str(access)]
    path.write_text(
        f"[unix_http_server]\nfile={directory}/supervisor.sock\nchmod=0600\n"
        f"[supervisord]\nlogfile={directory}/daemon.log\npidfile={directory}/supervisord.pid\n"
        f"childlogdir={directory}\n"
        "[rpcinterface:supervisor]\n"
        "supervisor.rpcinterface_factory=supervisor.rpcinterface:make_main_rpcinterface\n"
        f"[supervisorctl]\nserverurl=unix://{directory}/supervisor.sock\n"
        f"[program:fact_carrier_v2]\ncommand={shlex.join(command)}\ndirectory={ROOT}\n"
        "autostart=true\nautorestart=unexpected\nexitcodes=0,2\nstartsecs=0\nstartretries=0\n"
        f"stdout_logfile={directory}/runner.log\nstderr_logfile={directory}/error.log\n"
        "stopasgroup=true\nkillasgroup=true\n")
    os.chmod(path, 0o600)
    subprocess.run([config["python"], "-m", "supervisor.supervisord", "-c", str(path)],
                   check=True, capture_output=True)
    return {"status": "SUPERVISOR_STARTED", "program": "fact_carrier_v2"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "verify", "precheck", "start",
                                           "run", "status"])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--private", type=Path, default=PRIVATE)
    parser.add_argument("--access-config", type=Path, default=Path("/tmp/amr_fact_carrier_access.json"))
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.out, args.private)
        elif args.command == "verify":
            result = {"status": "PASS", "checks": runtime.verify(args.out, args.private)}
        elif args.command == "start":
            result = start(args.out, args.private, args.access_config)
        elif args.command in ("precheck", "run"):
            result = getattr(runtime, args.command)(args.out, args.private, args.access_config)
        else:
            result = runtime.load(args.out / "heartbeat.json")
        print(runtime.encode(result).decode())
    except runtime.TerminalHalt as error:
        print(runtime.encode({"status": "TERMINAL_HALT", "reason": str(error)}).decode())
        raise SystemExit(2)


if __name__ == "__main__":
    main()
