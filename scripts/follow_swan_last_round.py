"""Persist the one authorized completion through generation, GPU and archive."""
from __future__ import annotations

from datetime import datetime, timezone
import base64
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
from scripts.analyze_swan_blind_eval import save_json, ATTACKS
from scripts.recalibrate_swan_5pct import threshold_from_calibration, statistic
from src.core.swan_blind_protocol import digest

EXP = ROOT / "experiments/swan_multiview_development1"
OUT = EXP / "final_completion1"
OLD = EXP / "blind_eval1/run2"
PYTHON = ROOT / ".venv/bin/python"
ALLOWLIST = (
    "CONTROL.json", "EVAL_FREEZE.json", "READINESS.json", "READINESS_STOP.json",
    "DOCUMENT_SCORES.json", "CALIBRATION.json", "RESULTS.json",
    "INJECTION_DIAGNOSTIC.json", "REPORT.md", "heartbeat.json",
    "HALT.json", "DEADLINE_STOP.json", "GUARD_HEARTBEAT.json", "GUARD_STOP_OUTCOME.json",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def remaining():
    c = json.loads((OUT / "CONTROL.json").read_text())
    return (datetime.fromisoformat(c["deadline_at"]) - datetime.now(timezone.utc)).total_seconds()


def emit(stage, **fields):
    save_json(OUT / "WORKFLOW_STATUS.json", {"at": now(), "stage": stage,
              "remaining_seconds": max(0, remaining()), **fields})
    print(json.dumps({"at": now(), "stage": stage, **fields}), flush=True)


def local(script, *args):
    r = subprocess.run([str(PYTHON), str(ROOT / "scripts" / script), *map(str, args)],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("Local stage failed: " + script)


def select_candidate():
    if (OUT / "CANDIDATE_SELECTION.json").exists():
        return
    if list((OUT / "private/calls").glob("*")):
        raise RuntimeError("Selection attempted after generation")
    frozen = json.loads((OLD / "FREEZE.json").read_text())
    previous = json.loads((OLD / "RESULTS.json").read_text())
    scored = json.loads((OLD / "DOCUMENT_SCORES.json").read_text())
    if previous["complete_methods"] != previous["planned_methods"]:
        raise RuntimeError("All old methods required for candidate selection")
    if previous["freeze_sha256"] != digest(frozen) or scored["freeze_sha256"] != digest(frozen):
        raise RuntimeError("Old snapshot provenance mismatch")
    ranking = []
    for method in ("S6", "S8", "S10", "S12"):
        data = [r for r in scored["documents"] if r["method"] == method]
        unit = previous["methods"][method]["unit_threshold"]
        negatives = [r for r in data if r["arm"] == "paired_negative"
                     and r["split"] == "calibration"]
        threshold, groups = threshold_from_calibration(negatives, unit)
        if threshold is None or previous["methods"][method]["availability"] != "available":
            ranking.append({"method": method, "available": False})
            continue
        fprs, tprs, completion = {}, {}, []
        for condition in ATTACKS:
            test = [r for r in data if r["split"] == "test" and r["attack"] == condition]
            for label, arm in (("FPR", "paired_negative"), ("TPR", "positive")):
                subset = [r for r in test if r["arm"] == arm]
                rate = sum(statistic(r, unit) is not None
                           and statistic(r, unit) > threshold for r in subset) / len(subset)
                (fprs if label == "FPR" else tprs)[condition] = rate
                if arm == "positive":
                    completion.append(sum(r["complete"] for r in subset)/len(subset))
        ranking.append({
            "method": method, "available": True,
            "worst_condition_fpr": max(fprs.values()),
            "mean_merge_comma_space_tpr": sum(tprs[c] for c in ("merge", "comma", "space"))/3,
            "mean_positive_complete_rate": sum(completion)/len(completion),
            "candidate_cost": frozen["candidate_totals"][method],
            "unit_threshold": unit, "unified_document_threshold": threshold,
            "conditions_FPR": fprs, "conditions_TPR": tprs,
        })
    eligible = [r for r in ranking if r["available"]]
    if not eligible:
        raise RuntimeError("No calibrated old window candidate")
    eligible.sort(key=lambda r: (
        r["worst_condition_fpr"], -r["mean_merge_comma_space_tpr"],
        -r["mean_positive_complete_rate"], r["candidate_cost"], int(r["method"][1:])))
    save_json(OUT / "CANDIDATE_SELECTION.json", {
        "schema": "swan-last-round-candidate-selection/v1", "created_at": now(),
        "method": eligible[0]["method"], "ranking": ranking,
        "selection_rule": "min-worst-FPR/max-format-TPR/max-positive-completion/min-cost/min-window",
        "old_freeze_sha256": digest(frozen),
        "old_results_sha256": hashlib.sha256((OLD / "RESULTS.json").read_bytes()).hexdigest(),
        "old_scores_sha256": hashlib.sha256((OLD / "DOCUMENT_SCORES.json").read_bytes()).hexdigest(),
        "before_new_responses_or_test_scores": True,
        "development_selection_only": True, "new_threshold_must_be_recalibrated": True,
    })


def quality_audit():
    """Read-only matching and cost summaries; never select samples or thresholds."""
    docs = json.loads((OUT/"private/DOCUMENTS.json").read_text())["documents"]
    groups = {}
    for source, arm, split in (
        ("human", "negative", "calibration"), ("human", "negative", "test"),
        ("generated", "negative", "calibration"), ("generated", "negative", "test"),
        ("generated", "positive", "test"),
    ):
        subset = [d for d in docs if
                  (d["source"], d["arm"], d["split"]) == (source, arm, split)]
        lengths = [len(d["text"].split()) for d in subset]
        groups[f"{source}:{arm}:{split}"] = {
            "documents": len(subset), "input_eligible": sum(d["eligible"] for d in subset),
            "mean_words_all": sum(lengths)/len(lengths),
            "minimum_words_all": min(lengths), "maximum_words_all": max(lengths),
            "requested_generated_sentences": 16 if source == "generated" else None,
        }
    positive = {d["pair_id"]: d for d in docs if d["arm"] == "positive"}
    negative = {d["pair_id"]: d for d in docs if d["source"] == "generated"
                and d["arm"] == "negative" and d["split"] == "test" and d["pair_id"]}
    length_gaps = [abs(len(p["text"].split())-len(negative[k]["text"].split()))
                   for k, p in positive.items()]
    from collections import Counter
    models, finishes, usage = Counter(), Counter(), Counter()
    for path in (OUT/"private/calls").glob("*.receipt.json"):
        r = json.loads(path.read_text())
        models[r.get("response_model") or "unavailable"] += 1
        finishes[r.get("finish_reason") or r.get("error") or "unavailable"] += 1
        usage.update({k: int(v) for k, v in (r.get("usage") or {}).items()
                      if k in ("prompt_tokens", "completion_tokens", "total_tokens")
                      and isinstance(v, int)})
    save_json(OUT/"INPUT_QUALITY_AUDIT.json", {
        "groups": groups, "topic_pairs": len(positive),
        "paired_mean_absolute_word_count_gap": sum(length_gaps)/len(length_gaps),
        "source_article_pairing_exact": all(
            p["source_article_sha256"] == negative[k]["source_article_sha256"]
            for k, p in positive.items()),
        "response_models": dict(models), "finish_or_error_counts": dict(finishes),
        "reported_api_token_usage": dict(usage),
        "independent_semantic_or_human_quality_review": False,
        "used_to_select_samples_or_thresholds": False,
        "all_failed_generation_intents_retained": True,
    })


def launch_remote():
    frozen_files = [
        ROOT / "scripts/run_swan_last_round.py", ROOT / "scripts/analyze_swan_last_round.py",
        OUT / "CONTROL.json", OUT / "SOURCE_FREEZE.json", OUT / "GENERATION_FREEZE.json",
        OUT / "CANDIDATE_SELECTION.json", OUT / "INPUT_AUDIT.json", OUT / "private/DOCUMENTS.json",
    ]
    upload_files(frozen_files, names=[str(p.relative_to(ROOT)) for p in frozen_files])
    # One process owns both readiness and actual fresh inference, with a group stop.
    result = ssh(r"""
cd /workspace/swan-elastic-reconstruct && .venv/bin/python - <<'RPY'
import json,subprocess
from pathlib import Path
p=Path('/tmp/swan_blind_eval_control/supervisord.conf')
old=Path('experiments/swan_multiview_development1/blind_eval1/run2')
if json.loads((old/'heartbeat.json').read_text())['state']!='complete' or (old/'HALT.json').exists():
    raise RuntimeError('Old queue incomplete')
state=subprocess.run(['/usr/bin/supervisorctl','-c',str(p),'status','evaluation'],capture_output=True,text=True)
if 'RUNNING' in state.stdout or 'STARTING' in state.stdout:
    raise RuntimeError('Old evaluation still owns runtime')
s=p.read_text()
if '[program:final_confirmation]' not in s:
    s+='''
[program:final_confirmation]
directory=/workspace/swan-elastic-reconstruct
command=/workspace/swan-elastic-reconstruct/.venv/bin/python /workspace/swan-elastic-reconstruct/scripts/run_swan_last_round.py --phase workflow
autostart=true
autorestart=false
startsecs=1
stopwaitsecs=20
stopasgroup=true
killasgroup=true
stdout_logfile=/tmp/swan_blind_eval_control/final_confirmation.log
stderr_logfile=/tmp/swan_blind_eval_control/final_confirmation.error.log
environment=PYTHONUNBUFFERED="1"
'''
    p.write_text(s)
for c in ('reread','update','status'):
    r=subprocess.run(['/usr/bin/supervisorctl','-c',str(p),c],capture_output=True,text=True,timeout=15)
    print(json.dumps({'command':c,'returncode':r.returncode,'status':r.stdout}))
RPY
""", timeout=55)
    if '"remote_command_exit"' in result:
        raise RuntimeError("Remote launch failed")
    save_json(OUT / "REMOTE_LAUNCH_RECEIPT.json", {"at": now(), "launched_once": True})


def retrieve():
    command = r"""
cd /workspace/swan-elastic-reconstruct && .venv/bin/python - <<'RPY'
from pathlib import Path
import base64,hashlib,json
p=Path('experiments/swan_multiview_development1/final_completion1')
names=%s
out={}
for n in names:
    if (p/n).exists():
        b=(p/n).read_bytes()
        out[n]={'base64':base64.b64encode(b).decode(),'sha256':hashlib.sha256(b).hexdigest()}
print(json.dumps(out))
RPY
""" % repr(ALLOWLIST)
    returned = json.loads(ssh(command, timeout=45))
    for name, record in returned.items():
        if name not in ALLOWLIST:
            raise RuntimeError("Unknown exported resource")
        content = base64.b64decode(record["base64"])
        if hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise RuntimeError("Aggregate transfer SHA mismatch")
        # Local controller and guard have their own mutable status.
        target = OUT / ("REMOTE_" + name if name in (
            "CONTROL.json", "GUARD_HEARTBEAT.json", "GUARD_STOP_OUTCOME.json") else name)
        temp = target.with_suffix(target.suffix + ".transfer")
        temp.write_bytes(content)
        temp.replace(target)


def archive(reason):
    local("archive_swan_last_round.py", "--reason", reason)
    emit("closed", reason=reason, line_closed=True, no_successor=True)


def follow():
    try:
        emit("waiting_for_completed_old_windows")
        while True:
            if remaining() <= 0:
                archive("deadline_before_old_windows_complete")
                return
            # The existing follower is the sole old snapshot writer.
            halt = OLD / "HALT.json"
            if halt.exists():
                archive("old_queue_permanent_halt")
                return
            hb = OLD / "heartbeat.json"
            if hb.exists() and json.loads(hb.read_text())["state"] == "complete":
                result = OLD / "RESULTS.json"
                if result.exists():
                    r = json.loads(result.read_text())
                    if r["complete_methods"] == r["planned_methods"]:
                        break
            time.sleep(min(30, max(.1, remaining())))
        source_audit = OUT / "SOURCE_OVERLAP_AUDIT.json"
        if not source_audit.exists() or not json.loads(source_audit.read_text())["ready"] \
                or (OUT / "SOURCE_STOP.json").exists():
            archive("source_readiness_stop")
            return
        select_candidate()
        emit("single_generation_batch")
        if not (OUT / "INPUT_AUDIT.json").exists():
            local("swan_last_round_inputs.py", "--phase", "generate")
        quality_audit()
        if remaining() <= 0:
            archive("deadline_after_generation")
            return
        if not (OUT / "REMOTE_LAUNCH_RECEIPT.json").exists():
            emit("fresh_confirmation_readiness")
            launch_remote()
        while True:
            try:
                retrieve()
            except Exception as exc:
                emit("aggregate_transfer_pending", exception_type=type(exc).__name__)
                if remaining() <= 0:
                    archive("deadline")
                    return
                time.sleep(min(30, max(.1, remaining())))
                continue
            for marker, reason in (
                ("HALT.json", "confirmation_permanent_halt"),
                ("READINESS_STOP.json", "confirmation_readiness_stop"),
                ("DEADLINE_STOP.json", "deadline"),
            ):
                if (OUT / marker).exists():
                    archive(reason)
                    return
            hb = OUT / "heartbeat.json"
            if hb.exists() and json.loads(hb.read_text())["state"] == "complete":
                archive("confirmation_complete")
                return
            if remaining() <= 0:
                retrieve()
                archive("deadline")
                return
            emit("confirmation_running")
            time.sleep(min(30, max(.1, remaining())))
    except Exception as exc:
        save_json(OUT / "WORKFLOW_STOP.json", {
            "at": now(), "permanent": True, "exception_type": type(exc).__name__,
            "reason": "frozen_stage_failure_no_rescue", "no_successor": True})
        archive("workflow_stop")
        raise


if __name__ == "__main__":
    follow()
