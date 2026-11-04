"""One-off remote completion listener and SHA-verified public artifact download.

The helper supplies authentication outside this script. No endpoints, credentials
or private source/response text are saved in the public deliverable.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys


def collect(helper, remote_root, output):
    # Linux inotify blocks on filesystem events, rather than sleep/poll loops.
    # Terminal and HALT are atomic writes in this existing directory.
    watcher = r'''
from pathlib import Path
import base64,ctypes,json,os,select,subprocess,sys
root=Path(REMOTE_ROOT)
study=root/"evaluation1"
libc=ctypes.CDLL(None,use_errno=True)
fd=libc.inotify_init1(0)
if fd<0:
    raise OSError(ctypes.get_errno(),"inotify_init1")
watch=libc.inotify_add_watch(fd,os.fsencode(study),0x00000008|0x00000080|0x00000100)
if watch<0:
    raise OSError(ctypes.get_errno(),"inotify_add_watch")
pidfd=None
try:
    pidfd=os.pidfd_open(int((root/"study.pid").read_text()))
except ProcessLookupError:
    if not (study/"TERMINAL.json").exists() and not (study/"HALT.json").exists():
        raise RuntimeError("Study process exited without a terminal marker")
while not (study/"TERMINAL.json").exists():
    if (study/"HALT.json").exists():
        print(json.dumps({"status":"halted","audit":json.loads((study/"HALT.json").read_text())}))
        sys.exit(2)
    ready,_,_=select.select([fd]+([pidfd] if pidfd is not None else []),[],[])
    if fd in ready:
        os.read(fd,65536)
    if pidfd in ready and not (study/"TERMINAL.json").exists():
        print(json.dumps({"status":"process_exited","halt_exists":(study/"HALT.json").exists()}))
        sys.exit(2)
os.close(fd)
if pidfd is not None:
    os.close(pidfd)
if not (root/"reviews1/TERMINAL.json").exists():
    raise RuntimeError("Review terminal is still missing")
report=root/"final_report"
if not report.exists():
    command=[str(root.parent/".venv/bin/python"),str(root/"scripts/report_embedding_amr_study.py"),
             "--inputs",str(root/"experiments/embedding_amr_benchmark_20261104"),
             "--study",str(study),"--reviews",str(root/"reviews1"),"--output",str(report)]
    completed=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if completed.returncode:
        print(json.dumps({"status":"report_failed","returncode":completed.returncode,
                          "error":completed.stderr[-2500:]}))
        sys.exit(3)
print(json.dumps({"status":"complete","files":{
    p.name:base64.b64encode(p.read_bytes()).decode() for p in report.iterdir() if p.is_file()}}))
'''.replace("REMOTE_ROOT", repr(remote_root))
    completed = subprocess.run([sys.executable, str(helper)],
                               input="python3 -c " + shlex.quote(watcher),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # The helper already redacts connection metadata; do not persist its stderr.
    if completed.returncode:
        print(completed.stdout)
        try:
            event = json.loads(completed.stdout)
            status = event.get("status", "listener_failed")
        except (ValueError, TypeError):
            status = "listener_failed"
        note = {
            "halted": "评估触发停止门，未生成完整检测结果。",
            "process_exited": "评估进程已退出，尚无成功终态。",
            "report_failed": "评估已有终态，但报告生成或复核失败。",
        }.get(status, "结果监听连接中断，远端任务状态需要核查。")
        (output.parent / "当前状态.md").write_text(
            "# 句向量辅助 AMR：当前状态\n\n" + note
            + "\n\n先前完成时间预估暂不适用。停止通知已返回助手处理，"
            "不会将部分结果记作完整评估。\n\n"
            "[模型评审结果](模型评审结果.json) · [实现与运行说明](实现与运行说明.md)\n")
        raise RuntimeError("Completion listener stopped; return code " + str(completed.returncode))
    value = json.loads(completed.stdout)
    assert value["status"] == "complete"
    files = {name: base64.b64decode(data) for name, data in value["files"].items()}
    seal = json.loads(files["SHA256.json"])
    assert set(files) == set(seal) | {"SHA256.json"}
    assert all(hashlib.sha256(files[name]).hexdigest() == sha for name, sha in seal.items())
    assert all(Path(name).name == name for name in files)
    output.mkdir()
    for name, data in files.items():
        (output / name).write_bytes(data)
    audit = json.loads(files["最终审计.json"])
    (output.parent / "当前状态.md").write_text(
        "# 句向量辅助 AMR：当前状态\n\n"
        "完整复评已结束，校准、测试、API 评审及报告下载均完成 SHA 校验。\n\n"
        "[查看完整评估报告](完整复评/完整评估报告.md)\n\n"
        + audit["interpretation"] + "\n\n"
        "本次为公开 Bank50 的历史短文本复评；机械编辑条件仅作为标点结构压力测试。\n")
    print(json.dumps({"status": "complete", "files": sorted(files), "directory": str(output)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--remote-helper", type=Path, required=True)
    p.add_argument("--remote-root", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    collect(a.remote_helper, a.remote_root, a.output)
