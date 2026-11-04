"""Stop only the completed diagnostic's supervisors and verify sealed receipts."""
from __future__ import annotations

import argparse
import configparser
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

import run_semantic_cycle2 as old
import run_semantic_readout as task
from remote_execution import Remote
from run_semantic_sequence_cycle import atomic, sha


def close(helper):
    task.verify()
    result = json.loads((task.OUT / "RESULT_AUDIT.json").read_text())
    assert result["Go"]
    assert result["result_sha256"] == sha(task.OUT / "RESULT.json")
    assert result["order_result_sha256"] == sha(task.OUT / "ORDER_RESULT.json")
    execution = json.loads((task.OUT / "EXECUTION_CLOSEOUT.json").read_text())
    assert execution["completed_requests"] == 228
    assert execution["remote_supervisor_shutdown"]

    calls = task.PRIVATE / "calls"
    intents = {p.name.removesuffix(".intent.json"): p for p in calls.glob("*.intent.json")}
    responses = {p.name.removesuffix(".response.json"): p for p in calls.glob("*.response.json")}
    assert set(intents) == set(responses) and len(responses) == 228
    receipt_manifest = {}
    reasons = {}
    for cid, path in responses.items():
        intent, response = json.loads(intents[cid].read_text()), json.loads(path.read_text())
        assert response["complete"] is True
        assert response["request_sha256"] == intent["request_sha256"]
        reason = response["finish_reason"]
        reasons[reason] = reasons.get(reason, 0) + 1
        receipt_manifest[cid] = {"intent_sha256": sha(intents[cid]), "response_sha256": sha(path)}
    task.integrity()
    atomic(task.OUT / "RECEIPT_AUDIT.json", {
        "Go": True, "completed_requests": 228, "unresolved_dispatches": 0,
        "complete_receipts": True, "request_identity_verified": True,
        "finish_reason_counts": reasons, "private_receipt_sha256": receipt_manifest,
        "response_text_exported": False,
    })

    config_path = task.PRIVATE / "local.conf"
    config = configparser.ConfigParser(interpolation=None)
    config.read(config_path)
    pidfile = Path(config["supervisord"]["pidfile"])
    socket = Path(config["unix_http_server"]["file"])
    pid = int(pidfile.read_text().strip()) if pidfile.exists() else None
    ctl = str(task.ROOT / ".venv-swan-positive/bin/supervisorctl")
    if socket.exists():
        checked = subprocess.run([ctl, "-c", str(config_path), "status", "readout"],
                                 capture_output=True, text=True, timeout=15)
        # Supervisor returns 3 for a normal EXITED process.
        assert checked.returncode in (0, 3)
        assert "EXITED" in checked.stdout and "RUNNING" not in checked.stdout
    config["program:readout"]["autostart"] = "false"
    config["program:readout"]["autorestart"] = "false"
    with config_path.open("w") as stream:
        config.write(stream)
    if socket.exists():
        stopped = subprocess.run([ctl, "-c", str(config_path), "shutdown"],
                                 capture_output=True, text=True, timeout=15)
        assert stopped.returncode == 0
    for _ in range(10):
        if not socket.exists():
            break
        time.sleep(1)
    local_alive = False
    if pid is not None:
        try:
            os.kill(pid, 0)
            # A zombie cannot restart a worker.
            record = subprocess.run(["ps", "-p", str(pid), "-o", "stat="],
                                    capture_output=True, text=True, timeout=5).stdout.strip()
            local_alive = bool(record and not record.startswith("Z"))
        except ProcessLookupError:
            pass
    assert not socket.exists() and not local_alive

    access = Remote(old.remote_module(helper))
    code = """
import json,os,pathlib
p=pathlib.Path(TASK_ROOT)
alive=False
pidfile=p/'supervisord.pid'
if pidfile.exists():
    pid=int(pidfile.read_text().strip())
    proc=pathlib.Path('/proc')/str(pid)
    alive=proc.exists() and TASK_ROOT.encode() in (proc/'cmdline').read_bytes()
workers=0
for item in pathlib.Path('/proc').iterdir():
    if not item.name.isdigit() or int(item.name)==os.getpid():continue
    try:argv=(item/'cmdline').read_bytes().split(b'\\x00')
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    if any(x.endswith(b'/semantic_readout_gpu.py') for x in argv) and any(TASK_ROOT.encode() in x for x in argv):
        workers+=1
print(json.dumps({'remote_supervisor_socket_exists':(p/'supervisor.sock').exists(),
                  'remote_supervisor_process_active':alive,'remote_inference_worker_count':workers}))
""".replace("TASK_ROOT", repr(task.REMOTE))
    remote = json.loads(access.read(task.GPU + " -c " + shlex.quote(code), timeout=35))
    assert not remote["remote_supervisor_socket_exists"]
    assert not remote["remote_supervisor_process_active"]
    assert remote["remote_inference_worker_count"] == 0
    closeout = {
        "Go": True, "at": old.now(), "completed_requests": 228, "unresolved_dispatches": 0,
        "local_supervisor_shutdown": True, "local_autostart_disabled": True,
        "local_autorestart_disabled": True, "remote_supervisor_shutdown": True,
        **remote, "private_process_configuration_sha256": sha(config_path),
        "sealed_versions": 72, "old_cycle2": "NO_GO_19_OF_24", "human_completed": False,
        "scope": "independent diagnostic computation and archival closeout",
    }
    atomic(task.OUT / "CLOSEOUT.json", closeout)
    task.status("COMPLETE", pairs=12, versions=72, independent_audit="Go",
                local_supervisor_shutdown=True, remote_supervisor_shutdown=True,
                human="NOT_EVALUATED", formal_G1_G2_G3_G4="NOT_EVALUATED")
    print(json.dumps(closeout))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-helper", type=Path, required=True)
    close(parser.parse_args().remote_helper)
