"""Advance WP2-v18 from readiness to execution only after an audit pass."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import tempfile
import time


READINESS_PROGRAM = "wp2-v18-b1-readiness"
EXECUTION_PROGRAM = "wp2-v18-b1"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    payload = (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def supervisor_status(program):
    completed = subprocess.run(
        ["supervisorctl", "status", program],
        check=False,
        capture_output=True,
        text=True,
    )
    output = (completed.stdout + completed.stderr).strip()
    fields = output.split()
    state = fields[1] if len(fields) >= 2 else "UNKNOWN"
    return state, output


def write_result(path, status, **fields):
    atomic_write_json(path, {
        "schema_version": 1,
        "stage": "WP2-v18-B1-gate-supervisor",
        "status": status,
        "updated_at": utc_now(),
        "contains_raw_text": False,
        **fields,
    })


def run(args):
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    readiness_path = output_dir / "P0_READINESS.json"
    audit_path = output_dir / "P0_AUDIT.json"
    result_path = output_dir / "B1_GATE_RESULT.json"
    failure_path = output_dir / "LAST_FAILURE.json"
    deadline = time.monotonic() + args.timeout_hours * 3600
    while not readiness_path.is_file():
        state, detail = supervisor_status(READINESS_PROGRAM)
        write_result(
            result_path,
            "waiting",
            readiness_supervisor_state=state,
            readiness_supervisor_detail=detail,
        )
        if state in {"FATAL", "UNKNOWN"} or (
            state in {"EXITED", "STOPPED"}
            and failure_path.is_file()
        ):
            write_result(
                result_path,
                "stop",
                terminal_reason="readiness_process_failed",
                readiness_supervisor_state=state,
                readiness_supervisor_detail=detail,
            )
            return 2
        if time.monotonic() >= deadline:
            write_result(
                result_path,
                "stop",
                terminal_reason="readiness_timeout",
                readiness_supervisor_state=state,
            )
            return 2
        time.sleep(args.poll_seconds)

    readiness = read_json(readiness_path)
    if readiness.get("status") != "go":
        write_result(
            result_path,
            "stop",
            terminal_reason="readiness_not_go",
            readiness_status=readiness.get("status"),
        )
        return 0

    audit = subprocess.run(
        [
            str(args.python),
            str(root / "scripts/audit_certified_resegmentation_wp2_v18_b1.py"),
            "--phase",
            "readiness",
            "--root",
            str(root),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if audit.returncode != 0 or not audit_path.is_file():
        write_result(
            result_path,
            "stop",
            terminal_reason="readiness_auditor_failed",
            audit_returncode=audit.returncode,
            audit_stderr=audit.stderr[-2000:],
        )
        return 2
    audit_result = read_json(audit_path)
    if audit_result.get("status") != "pass":
        write_result(
            result_path,
            "stop",
            terminal_reason="readiness_audit_not_pass",
            audit_status=audit_result.get("status"),
        )
        return 0

    execution_state, execution_detail = supervisor_status(EXECUTION_PROGRAM)
    if execution_state != "STOPPED":
        write_result(
            result_path,
            "stop",
            terminal_reason="execution_program_not_stopped",
            execution_supervisor_state=execution_state,
            execution_supervisor_detail=execution_detail,
        )
        return 2
    started = subprocess.run(
        ["supervisorctl", "start", EXECUTION_PROGRAM],
        check=False,
        capture_output=True,
        text=True,
    )
    if started.returncode != 0:
        write_result(
            result_path,
            "stop",
            terminal_reason="execution_start_failed",
            start_returncode=started.returncode,
            start_stderr=started.stderr[-2000:],
        )
        return 2
    write_result(
        result_path,
        "go",
        terminal_reason="readiness_audit_pass_execution_started",
        readiness_sha256=readiness["workload_rows"]["sha256"],
        p0_audit_status=audit_result["status"],
        execution_supervisor_output=started.stdout.strip(),
    )
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=6.0)
    return parser.parse_args()


def main():
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
