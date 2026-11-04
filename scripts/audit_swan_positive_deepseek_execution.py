"""Independent audit of the completed DeepSeek SWAN backend preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/swan_positive_deepseek_preflight1"
AUDIT = OUTPUT / "execution_audit.json"
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


def freeze_text(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"Frozen DeepSeek execution audit differs: {path}")
    path.write_text(text, encoding="utf-8")


def audit() -> dict:
    manifest = load_json(OUTPUT / "manifest.json")
    budget = load_json(OUTPUT / "budget_estimate.json")
    authorization = load_json(OUTPUT / "budget_authorization_v2.json")
    amendment_v2 = load_json(OUTPUT / "protocol_amendment_v2.json")
    amendment_v3 = load_json(OUTPUT / "protocol_amendment_v3.json")
    amendment_v4 = load_json(OUTPUT / "protocol_amendment_v4.json")
    summary = load_json(OUTPUT / "summary.json")
    calls = {
        path.stem: (path, load_json(path))
        for path in sorted((OUTPUT / "deepseek_calls").glob("*.json"))
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
    dependencies_match = all(
        (ROOT / relative).is_file()
        and sha256(ROOT / relative) == expected
        for relative, expected in manifest["dependencies"].items()
    )

    generation_calls = {
        key: row
        for key, row in calls.items()
        if key != "model-probe"
    }
    total_cost = sum(
        item[1]["cost"]["actual"]["conservative_peak_cost_usd"]
        for item in calls.values()
    )
    all_scores = [item[1]["score"] for item in attempts.values()]
    accepted_attempts = [
        item[1] for item in attempts.values() if item[1]["accepted"]
    ]
    allocation_rows = []
    trace_hashes_match = True
    identity_links_match = True
    for allocation_id in sorted(EXPECTED_ALLOCATIONS):
        prefix = f"{allocation_id}-slot-00-attempt-"
        traces = [
            (path, row)
            for key, (path, row) in attempts.items()
            if key.startswith(prefix)
        ]
        slot_path, slot = slots[f"{allocation_id}-slot-00"]
        trace_hashes_match &= (
            slot["attempt_trace_sha256"]
            == [sha256(item[0]) for item in traces]
        )
        for trace_item in traces:
            trace = trace_item[1]
            call_path, call = calls[trace["call_id"]]
            identity_links_match &= (
                trace["candidate"] == call["response"]["text"]
                and trace["call_sha256"] == sha256(call_path)
                and trace["candidate_sha256"]
                == hashlib.sha256(
                    call["response"]["text"].encode("utf-8")
                ).hexdigest()
            )
        scores = [item[1]["score"] for item in traces]
        allocation_rows.append(
            {
                "allocation_id": allocation_id,
                "attempts": len(traces),
                "accepted": slot["accepted"],
                "fallback": slot["fallback"],
                "final_score": slot["score"],
                "maximum_score": max(scores),
                "mean_score": statistics.fmean(scores),
                "slot_sha256": sha256(slot_path),
            }
        )

    superseded_failure = (
        OUTPUT
        / "superseded_generation_failures/pilot-000-slot-00.json"
    )
    serialized = "\n".join(
        item[0].read_text(encoding="utf-8", errors="replace")
        for item in calls.values()
    )
    checks = {
        "base_dependencies_match": dependencies_match,
        "authorization_bound_to_base_artifacts": (
            authorization["manifest_sha256"] == sha256(OUTPUT / "manifest.json")
            and authorization["budget_estimate_sha256"]
            == sha256(OUTPUT / "budget_estimate.json")
            and authorization["max_cost_usd"] == 0.08
            and authorization["max_model_calls"] == 51
            and authorization["credential_risk_acknowledged"] is True
            and authorization["credential_rotated"] is False
        ),
        "amendment_chain_matches": (
            amendment_v2["base_manifest_sha256"]
            == sha256(OUTPUT / "manifest.json")
            and amendment_v3["v2_amendment_sha256"]
            == sha256(OUTPUT / "protocol_amendment_v2.json")
            and amendment_v4["v3_amendment_sha256"]
            == sha256(OUTPUT / "protocol_amendment_v3.json")
        ),
        "superseded_failure_preserved": (
            superseded_failure.is_file()
            and sha256(superseded_failure)
            == amendment_v4["observed_failure"]["failure_sha256"]
            and not (
                OUTPUT / "generation_failures/pilot-000-slot-00.json"
            ).exists()
        ),
        "exact_call_counts": (
            len(calls) == 42
            and len(generation_calls) == 41
            and len(attempts) == 41
            and len(slots) == 5
        ),
        "all_calls_complete_and_model_stable": all(
            row["status"] == "complete"
            and row["response"]["model"] == "deepseek-flash"
            and row["transport_attempts"] == 1
            for row in (item[1] for item in calls.values())
        ),
        "all_candidate_call_links_match": identity_links_match,
        "all_slot_trace_hashes_match": trace_hashes_match,
        "scores_valid_and_acceptance_exact": (
            all(
                isinstance(score, (int, float)) and 0 <= score <= 1
                for score in all_scores
            )
            and len(accepted_attempts) == 1
            and accepted_attempts[0]["score"] >= 0.75
            and all(
                row["score"] < 0.75
                for row in (item[1] for item in attempts.values())
                if not row["accepted"]
            )
        ),
        "summary_recomputed": (
            summary["status"] == "complete"
            and summary["completed_slots"] == 5
            and summary["accepted_slots"] == 1
            and summary["attempts"] == 41
            and summary["deepseek_calls"] == 42
            and summary["decision"] == "stop_deepseek_route"
            and summary["continue_gate"]
            == {"required": 3, "observed": 1, "passed": False}
        ),
        "actual_cost_within_bounds": (
            total_cost <= authorization["max_cost_usd"]
            and total_cost
            <= budget["conservative_total_bound_usd"]
        ),
        "no_detector_or_attack_artifacts": (
            not (OUTPUT / "detection").exists()
            and not (OUTPUT / "attacks").exists()
        ),
        "credential_absent_from_journals": (
            "sk-" not in serialized
            and "api.txt" not in serialized
            and "DEEPSEEK_API_KEY" not in serialized
        ),
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "status": (
            "verified_stop_deepseek_route"
            if passed
            else "deepseek_execution_audit_failed"
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
            "accepted_at_0_75": len(accepted_attempts),
        },
        "usage": {
            "probe_calls": 1,
            "generation_calls": len(generation_calls),
            "total_calls": len(calls),
            "conservative_peak_cost_usd": round(total_cost, 9),
            "approved_cost_usd": authorization["max_cost_usd"],
        },
        "decision": "stop_deepseek_route",
        "claim_boundary": summary["claim_boundary"],
    }


def report_text(result: dict) -> str:
    score = result["score_distribution"]
    usage = result["usage"]
    rows = "\n".join(
        (
            f"- `{row['allocation_id']}`: attempts={row['attempts']}, "
            f"max={row['maximum_score']:.4f}, "
            f"accepted={row['accepted']}"
        )
        for row in result["allocations"]
    )
    return f"""# DeepSeek SWAN backend preflight

The independent execution audit passed.

- Backend: `deepseek-flash` (DeepSeek-V4.1-Flash)
- Frozen sources and slots: 5 first slots
- Probe calls: {usage["probe_calls"]}
- Generation calls: {usage["generation_calls"]}
- Total calls: {usage["total_calls"]}
- Conservative peak-price cost: ${usage["conservative_peak_cost_usd"]:.6f}
- Approved ceiling: ${usage["approved_cost_usd"]:.2f}
- Accepted slots: 1/5
- Continue gate: at least 3/5
- Decision: `stop_deepseek_route`

Per-source result:

{rows}

Across {score["count"]} generated candidates, the maximum S2Match score was
{score["maximum"]:.4f}, the mean was {score["mean"]:.4f}, and exactly one
candidate reached 0.75. Four rejected slots are explicit fallbacks.

This is backend-feasibility evidence only. It is not an official Claude
reproduction, not a complete SWAN positive set, and contains no detector or
attack result. The frozen DeepSeek route must not be extended or retuned on
these observed sources.
"""


def main() -> None:
    result = audit()
    if not result["all_checks_pass"]:
        failed = [key for key, value in result["checks"].items() if not value]
        raise ValueError(f"DeepSeek execution audit failed: {failed}")
    freeze_text(
        AUDIT,
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
    )
    freeze_text(REPORT, report_text(result))
    print(
        json.dumps(
            {
                "status": result["status"],
                "score_distribution": result["score_distribution"],
                "usage": result["usage"],
                "decision": result["decision"],
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
