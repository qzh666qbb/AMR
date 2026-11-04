"""Independent offline audit for the local Qwen SWAN preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/swan_positive_local_qwen_preflight1"
AUDIT = OUTPUT / "audit.json"
REPORT = OUTPUT / "report.md"
EXPECTED_ALLOCATIONS = {f"pilot-{index:03d}" for index in range(5)}


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_frozen(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"Frozen local audit differs: {path}")
    path.write_text(text, encoding="utf-8")


def audit() -> dict:
    manifest = load_json(OUTPUT / "manifest.json")
    summary = load_json(OUTPUT / "summary.json")
    calls = {
        path.stem: load_json(path)
        for path in sorted((OUTPUT / "local_calls").glob("*.json"))
    }
    attempts = {
        path.stem: (path, load_json(path))
        for path in sorted(
            (OUTPUT / "generation_attempts").glob("*.json")
        )
    }
    slots = {
        path.stem: (path, load_json(path))
        for path in sorted((OUTPUT / "generation_slots").glob("*.json"))
    }
    dependency_checks = {
        relative: (
            (ROOT / relative).is_file()
            and sha256(ROOT / relative) == expected
        )
        for relative, expected in manifest["dependencies"].items()
    }

    allocation_rows = []
    all_trace_links = True
    all_identity_links = True
    all_scores_valid = True
    all_rejected = True
    all_fallbacks = True
    all_call_records_local = True
    all_scores = []
    for allocation_id in sorted(EXPECTED_ALLOCATIONS):
        prefix = f"{allocation_id}-slot-00-attempt-"
        call_rows = [
            value
            for key, value in calls.items()
            if key.startswith(prefix)
        ]
        trace_rows = [
            (path, value)
            for key, (path, value) in attempts.items()
            if key.startswith(prefix)
        ]
        slot_key = f"{allocation_id}-slot-00"
        slot_path, slot = slots[slot_key]
        trace_values = [value for path, value in trace_rows if path.is_file()]
        scores = [row["score"] for row in trace_values]
        all_scores.extend(scores)
        all_trace_links &= (
            slot["attempt_trace_sha256"]
            == [sha256(path) for path, value in trace_rows if value]
        )
        all_identity_links &= all(
            trace["candidate"] == call["response"]["text"]
            and trace["candidate_sha256"]
            == call["response"]["text_sha256"]
            and trace["call_sha256"]
            == sha256(OUTPUT / "local_calls" / f"{call_id}.json")
            for call_id, call in calls.items()
            for trace in trace_values
            if trace["call_id"] == call_id
        )
        all_scores_valid &= all(
            isinstance(score, (int, float)) and 0 <= score <= 1
            for score in scores
        )
        all_rejected &= (
            all(row["accepted"] is False for row in trace_values)
            and slot["accepted"] is False
        )
        all_fallbacks &= slot["fallback"] is True
        all_call_records_local &= all(
            row["request"]["backend"]
            == "local_qwen2.5-0.5b-instruct"
            and row["status"] == "complete"
            for row in call_rows
        )
        allocation_rows.append(
            {
                "allocation_id": allocation_id,
                "calls": len(call_rows),
                "attempts": len(trace_rows),
                "slot_sha256": sha256(slot_path),
                "maximum_score": max(scores),
                "mean_score": statistics.fmean(scores),
                "accepted": slot["accepted"],
                "fallback": slot["fallback"],
            }
        )

    expected_decision = (
        "prepare_separate_local_qwen_pilot"
        if sum(row["accepted"] for row in allocation_rows) >= 3
        else "stop_local_qwen_route"
    )
    forbidden_text = ("api.txt", "DEEPSEEK_API_KEY", "sk-")
    artifact_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for folder in (
            OUTPUT / "local_calls",
            OUTPUT / "generation_attempts",
            OUTPUT / "generation_slots",
        )
        for path in folder.glob("*.json")
    )
    checks = {
        "manifest_dependencies_match": all(
            dependency_checks.values()
        ),
        "exact_five_allocations": (
            set(slots) == {
                f"{allocation_id}-slot-00"
                for allocation_id in EXPECTED_ALLOCATIONS
            }
        ),
        "exact_50_calls_and_attempts": (
            len(calls) == 50 and len(attempts) == 50
        ),
        "ten_attempts_per_slot": all(
            row["calls"] == row["attempts"] == 10
            for row in allocation_rows
        ),
        "all_call_records_local_and_complete": all_call_records_local,
        "all_candidate_call_links_match": all_identity_links,
        "all_slot_trace_hashes_match": all_trace_links,
        "all_scores_in_unit_interval": all_scores_valid,
        "all_attempts_and_slots_rejected": all_rejected,
        "all_slots_are_explicit_fallbacks": all_fallbacks,
        "summary_counts_match": (
            summary["completed_slots"] == 5
            and summary["accepted_slots"] == 0
            and summary["attempts"] == 50
            and summary["local_generation_calls"] == 50
        ),
        "stop_decision_recomputed": (
            summary["decision"] == expected_decision
            == "stop_local_qwen_route"
            and summary["continue_gate"]
            == {"required": 3, "observed": 0, "passed": False}
        ),
        "no_remote_or_downstream_artifacts": (
            not (OUTPUT / "calls").exists()
            and not (OUTPUT / "detection").exists()
            and not (OUTPUT / "attacks").exists()
        ),
        "no_secret_marker_in_generated_artifacts": not any(
            marker in artifact_text for marker in forbidden_text
        ),
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "status": (
            "verified_stop_local_qwen_route"
            if passed
            else "local_preflight_audit_failed"
        ),
        "all_checks_pass": passed,
        "checks": checks,
        "allocations": allocation_rows,
        "score_distribution": {
            "count": len(all_scores),
            "minimum": min(all_scores),
            "maximum": max(all_scores),
            "mean": statistics.fmean(all_scores),
            "median": statistics.median(all_scores),
            "accepted_at_0_75": sum(score >= 0.75 for score in all_scores),
        },
        "decision": expected_decision,
        "resource_counts": summary["resource_counts"],
        "claim_boundary": summary["claim_boundary"],
    }


def report_text(result: dict) -> str:
    score = result["score_distribution"]
    rows = "\n".join(
        (
            f"- `{row['allocation_id']}`: max={row['maximum_score']:.4f}, "
            f"mean={row['mean_score']:.4f}, accepted={row['accepted']}"
        )
        for row in result["allocations"]
    )
    return f"""# Local Qwen SWAN feasibility preflight

The independent offline audit passed.

- Backend: local Qwen2.5-0.5B-Instruct
- Frozen sources: 5
- Tested slots: first slot of each source
- Local generations: 50
- Parser calls: 50
- S2Match calls: 50
- Accepted slots: 0/5
- Continue gate: at least 3/5
- Decision: `stop_local_qwen_route`
- DeepSeek and Bedrock calls: 0

Per-source score summary:

{rows}

Across all 50 candidates, the maximum S2Match score was
{score["maximum"]:.4f}, the mean was {score["mean"]:.4f}, and no candidate
reached 0.75. Every returned slot is explicitly marked as a rejected fallback.

This is backend-feasibility evidence only. It does not establish that local
generation is generally incapable of SWAN injection, does not evaluate
detector TPR, and produces no attack or ASR result. The frozen Qwen route must
not be extended or retuned on these observed sources.
"""


def main() -> None:
    result = audit()
    if not result["all_checks_pass"]:
        raise ValueError("Local Qwen preflight audit failed")
    write_frozen(
        AUDIT,
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
    )
    write_frozen(REPORT, report_text(result))
    print(
        json.dumps(
            {
                "status": result["status"],
                "score_distribution": result["score_distribution"],
                "decision": result["decision"],
                "resource_counts": result["resource_counts"],
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
