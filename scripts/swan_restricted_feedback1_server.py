"""Evaluator-owned replay server. Scores/threshold are never sent in label mode."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def session(candidates, scores, threshold, mode, budget, client=None):
    client = Path(client or Path(__file__).with_name("swan_restricted_feedback1_client.py"))
    if (mode not in ("none", "label", "score") or type(budget) is not int or budget < 1
            or len(candidates) < budget or not math.isfinite(threshold)):
        raise ValueError("Invalid fixed pool/query budget/threshold")
    ids = [r["candidate_id"] for r in candidates]
    if (len(set(ids)) != len(ids) or set(scores) != set(ids)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in scores.values())):
        raise ValueError("Incomplete evaluator score table")
    clean = [{"candidate_id": r["candidate_id"], "text": r["text"]} for r in candidates]
    by_id = {r["candidate_id"]: r for r in clean}
    transcript, final = [], None
    with tempfile.TemporaryDirectory(prefix="swan-feedback-client-") as work:
        # This separation enforces the audited interface, not a hostile-code OS sandbox.
        process = subprocess.Popen(
            [sys.executable, "-I", str(client.resolve())], cwd=work,
            env={"PATH": os.defpath, "PYTHONIOENCODING": "utf-8"},
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8")
        try:
            process.stdin.write(json.dumps({"candidates": clean, "mode": mode, "budget": budget}) + "\n")
            process.stdin.flush()
            for line in process.stdout:
                message = json.loads(line)
                if message.get("op") == "query":
                    index = len(transcript)
                    if (set(message) != {"op", "candidate_id", "text_sha256"} or mode == "none"
                            or index >= budget or message["candidate_id"] != ids[index]):
                        raise ValueError("Query outside fixed order or budget")
                    item = by_id[message["candidate_id"]]
                    if message["text_sha256"] != hashlib.sha256(item["text"].encode()).hexdigest():
                        raise ValueError("Query text differs")
                    score = scores[item["candidate_id"]]
                    response = {"label": bool(score > threshold)} if mode == "label" else {"score": score}
                    transcript.append({"request": message, "response": response})
                    process.stdin.write(json.dumps(response) + "\n")
                    process.stdin.flush()
                elif message.get("op") == "final":
                    if final is not None or set(message) != {
                            "op", "selected", "queries", "first_label_escape_query"}:
                        raise ValueError("Invalid final selection")
                    final = message
                else:
                    raise ValueError("Unknown attacker message")
            process.wait(timeout=10)
            if process.returncode:
                raise RuntimeError("Restricted attacker process failed: " + process.stderr.read()[-500:])
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()
    expected = 0 if mode == "none" else budget
    if (final is None or final["selected"] not in ids[:budget]
            or final["queries"] != expected or len(transcript) != expected):
        raise ValueError("Incomplete query/selection transcript")
    return {"selection": {k: v for k, v in final.items() if k != "op"},
            "transcript": transcript,
            "evaluation": {"selected_score": scores[final["selected"]],
                           "detected": scores[final["selected"]] > threshold}}
