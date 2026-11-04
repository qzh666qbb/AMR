"""Persist repair2 through fresh CUDA inference and numerical archival."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/tmp/amr_blind_eval_20261023")
from remote_access import ssh, upload_files
from scripts.analyze_swan_blind_eval import save_json

OUT = ROOT / "experiments/swan_multiview_development1/final_completion2"
ALLOWLIST = ("EVAL_FREEZE.json", "READINESS.json", "READINESS_STOP.json",
             "DOCUMENT_SCORES.json", "CALIBRATION.json", "RESULTS.json",
             "INJECTION_DIAGNOSTIC.json", "REPORT.md", "heartbeat.json",
             "HALT.json", "DEADLINE_STOP.json")


def now():
    return datetime.now(timezone.utc).isoformat()


def remaining():
    c = json.loads((OUT / "CONTROL.json").read_text())
    return (datetime.fromisoformat(c["deadline_at"])-datetime.now(timezone.utc)).total_seconds()


def emit(stage, **fields):
    value = {"at": now(), "stage": stage, "remaining_seconds": max(0, remaining()), **fields}
    save_json(OUT / "WORKFLOW_STATUS.json", value)
    print(json.dumps(value), flush=True)


def launch():
    paths = [ROOT / "scripts" / name for name in (
        "run_swan_last_round.py", "run_swan_blind_eval.py",
        "analyze_swan_last_round.py", "guard_swan_last_round.py")]
    paths += [OUT / name for name in (
        "CONTROL.json", "SOURCE_FREEZE.json", "GENERATION_FREEZE.json",
        "CANDIDATE_SELECTION.json", "INPUT_AUDIT.json", "private/DOCUMENTS.json")]
    upload_files(paths, names=[str(p.relative_to(ROOT)) for p in paths])
    save_json(OUT / "REMOTE_LAUNCH_DISPATCH.json", {"at": now(), "identity": "repair2"})
    text = ssh(r"""
cd /workspace/swan-elastic-reconstruct && .venv/bin/python - <<'RPY'
import json,subprocess
from pathlib import Path
p=Path('/tmp/swan_blind_eval_control/supervisord.conf')
out=Path('experiments/swan_multiview_development1/final_completion2')
if (out/'EVAL_FREEZE.json').exists() or (out/'HALT.json').exists():
    raise RuntimeError('Repair2 already dispatched')
state=subprocess.run(['/usr/bin/supervisorctl','-c',str(p),'status'],capture_output=True,text=True)
for line in state.stdout.splitlines():
    if any(line.startswith(n+' ') for n in ('evaluation','final_confirmation')) \
            and any(t in line for t in ('RUNNING','STARTING')):
        raise RuntimeError('Old GPU job still active')
if '[program:repair_confirmation2]' in p.read_text():
    raise RuntimeError('Repair2 program already registered; no redispatch')
s=p.read_text()+'''
[program:repair_confirmation2]
directory=/workspace/swan-elastic-reconstruct
command=/workspace/swan-elastic-reconstruct/.venv/bin/python /workspace/swan-elastic-reconstruct/scripts/run_swan_last_round.py --phase workflow --output /workspace/swan-elastic-reconstruct/experiments/swan_multiview_development1/final_completion2 --cache /tmp/swan_last_round_confirmation2
autostart=true
autorestart=false
startsecs=1
stopwaitsecs=20
stopasgroup=true
killasgroup=true
stdout_logfile=/tmp/swan_blind_eval_control/repair_confirmation2.log
stderr_logfile=/tmp/swan_blind_eval_control/repair_confirmation2.error.log
environment=PYTHONUNBUFFERED="1"

[program:repair_deadline_guard2]
directory=/workspace/swan-elastic-reconstruct
command=/workspace/swan-elastic-reconstruct/.venv/bin/python /workspace/swan-elastic-reconstruct/scripts/guard_swan_last_round.py --control /workspace/swan-elastic-reconstruct/experiments/swan_multiview_development1/final_completion2/CONTROL.json --supervisorctl /usr/bin/supervisorctl --config /tmp/swan_blind_eval_control/supervisord.conf --jobs repair_confirmation2
autostart=true
autorestart=false
startsecs=1
stdout_logfile=/tmp/swan_blind_eval_control/repair_deadline_guard2.log
stderr_logfile=/tmp/swan_blind_eval_control/repair_deadline_guard2.error.log
'''
p.write_text(s)
for command in ('reread','update'):
    r=subprocess.run(['/usr/bin/supervisorctl','-c',str(p),command],capture_output=True,text=True,timeout=15)
    if r.returncode:
        raise RuntimeError('Supervisor update failed')
print(json.dumps({'registered':True,'program':'repair_confirmation2'}))
RPY
""", timeout=55)
    receipt = json.loads(text)
    if not receipt.get("registered"):
        raise RuntimeError("Remote program registration missing")
    save_json(OUT / "REMOTE_LAUNCH_RECEIPT.json", {"at": now(), **receipt})


def retrieve():
    text = ssh(r"""
cd /workspace/swan-elastic-reconstruct && .venv/bin/python - <<'RPY'
from pathlib import Path
import base64,hashlib,json
p=Path('experiments/swan_multiview_development1/final_completion2')
out={}
for name in %s:
    if (p/name).exists():
        b=(p/name).read_bytes()
        out[name]={'data':base64.b64encode(b).decode(),'sha256':hashlib.sha256(b).hexdigest()}
print(json.dumps(out))
RPY
""" % repr(ALLOWLIST), timeout=45)
    for name, record in json.loads(text).items():
        if name not in ALLOWLIST:
            raise RuntimeError("Unknown remote aggregate")
        data = base64.b64decode(record["data"])
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise RuntimeError("Remote aggregate SHA mismatch")
        path = OUT / name
        temp = path.with_suffix(path.suffix+".transfer")
        temp.write_bytes(data)
        temp.replace(path)


def finish(reason):
    if (OUT / "REMOTE_LAUNCH_RECEIPT.json").exists():
        result = ssh(r"""
cd /workspace/swan-elastic-reconstruct && .venv/bin/python - <<'RPY'
import json,subprocess,time
ctl=['/usr/bin/supervisorctl','-c','/tmp/swan_blind_eval_control/supervisord.conf']
subprocess.run(ctl+['stop','repair_deadline_guard2'],capture_output=True,text=True,timeout=30)
for attempt in range(10):
    state=subprocess.run(ctl+['status','repair_confirmation2'],capture_output=True,text=True,timeout=15)
    if not any(t in state.stdout for t in ('RUNNING','STARTING')):
        break
    time.sleep(1)
print(json.dumps({'repair_confirmation2_inactive':not any(t in state.stdout for t in ('RUNNING','STARTING')),
                  'status':state.stdout.strip()}))
RPY
""", timeout=50)
        state = json.loads(result)
        save_json(OUT / "REMOTE_FINAL_STATUS.json", state)
        if not state["repair_confirmation2_inactive"]:
            raise RuntimeError("Remote scoring still active; cannot close")
    from scripts.archive_swan_repair2 import archive
    archive(reason)
    subprocess.run([str(ROOT / ".venv/bin/supervisorctl"), "-c",
                    "/tmp/amr_blind_eval_20261023/repair_supervisord.conf",
                    "stop", "repair_deadline_guard2"], capture_output=True, timeout=30)
    emit("closed", reason=reason)


def follow():
    emit("bounded_surface_repair")
    while not (OUT / "INPUT_AUDIT.json").exists():
        if remaining() <= 0:
            finish("deadline_during_repair")
            return
        time.sleep(15)
    from scripts.repair_swan_last_round_inputs import verify
    verify()
    from scripts.audit_swan_repair2 import audit as audit_integrity
    audit_integrity()
    import scripts.follow_swan_last_round as legacy
    legacy.OUT = OUT
    legacy.quality_audit()
    audit = json.loads((OUT / "INPUT_AUDIT.json").read_text())
    if not audit["new_evaluation_ready"]:
        finish("repair_input_readiness_stop")
        return
    if remaining() <= 0:
        finish("deadline_after_repair")
        return
    if not (OUT / "REMOTE_LAUNCH_RECEIPT.json").exists():
        if (OUT / "REMOTE_LAUNCH_DISPATCH.json").exists():
            raise RuntimeError("Uncertain remote dispatch requires state audit")
        emit("fresh_cuda_freeze_and_readiness")
        launch()
    while True:
        try:
            retrieve()
        except Exception as exc:
            emit("aggregate_transfer_pending", exception_type=type(exc).__name__)
            if remaining() <= 0:
                finish("deadline_transfer_limit")
                return
            time.sleep(20)
            continue
        for name, reason in (
            ("HALT.json", "repair_execution_halt"),
            ("READINESS_STOP.json", "repair_cuda_readiness_stop"),
            ("DEADLINE_STOP.json", "deadline")):
            if (OUT / name).exists():
                finish(reason)
                return
        hb = OUT / "heartbeat.json"
        if hb.exists() and json.loads(hb.read_text())["state"] == "complete":
            finish("confirmation_complete")
            return
        if remaining() <= 0:
            finish("deadline")
            return
        emit("confirmation_running")
        time.sleep(30)


if __name__ == "__main__":
    try:
        follow()
    except Exception as exc:
        save_json(OUT / "WORKFLOW_STOP.json", {
            "at": now(), "exception_type": type(exc).__name__, "permanent": True})
        raise
