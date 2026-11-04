"""Versioned offline audit for the post-P0 Branch C supersession chain."""
import argparse
import hashlib
import json
from pathlib import Path

import audit_swan_reframing as legacy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/swan_reframing1"
REPORT = "experiments/literature_novelty_audit1/report.md"
AUTHORITY = "experiments/literature_novelty_audit1/BRANCH_C_UPDATE.md"
AUDIT_PATH = OUT / "audit_v2.json"

P0_REPORT_SHA256 = "35bf187df6429211ba328a284ae1b6a5bf245c5792e9b2709e97ca45fe0ed779"
SUPERSEDED_REPORT_SHA256 = "f20a36b77357a7754dc7b541377e0672a6b9dfa016264a624238afda7d297c3b"
PRE_REFRESH_AUTHORITY_SHA256 = (
    "319d7e86a6b2da5f311852d257344617fb9d79cb281e0ebc166d217c6cf826fe"
)
AUTHORITY_SHA256 = "f12c955623a3612982aeacd15c771f7f9a7f75f7c322fc86bedbcb52030f8631"
NOTICE = (
    "> **2026-09-19 BRANCH C UPDATE / SUPERSEDED POSITIONING：**\n"
    "> 本文件保留 2026-09-16 时点的检索、版本和历史选题决策。final1 后续在\n"
    "> 生成质量 pilot 停止，故下文“攻击 + 机制审计”完整论文定位及自然文本\n"
    "> 机制归因不再是当前结论。当前文献矩阵、可写主张和防御处理以\n"
    "> [`BRANCH_C_UPDATE.md`](BRANCH_C_UPDATE.md) 为准；旧数值和文字保留用于\n"
    "> 追溯，不授权新增实验。\n\n"
)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def build():
    legacy_result = legacy.audit()
    snapshot = json.loads((OUT / "pre_reset_hashes.json").read_text())
    snapshot_rows = [row for row in snapshot["files"] if row["path"] == REPORT]

    report_bytes = (ROOT / REPORT).read_bytes()
    report_text = report_bytes.decode()
    notice_count = report_text.count(NOTICE)
    reconstructed = report_text.replace(NOTICE, "", 1).encode()
    authority_path = (ROOT / REPORT).parent / "BRANCH_C_UPDATE.md"
    authority_bytes = authority_path.read_bytes()
    endpoint_ids = [row["id"] for row in legacy_result["endpoint_gates"]]

    checks = {
        "legacy_audit_retains_expected_failure": legacy_result["ok"] is False,
        "legacy_unexpected_change_is_exactly_superseded_report":
            legacy_result["unexpected_historical_changes"] == [REPORT],
        "p0_snapshot_has_exactly_one_report_row": len(snapshot_rows) == 1,
        "p0_snapshot_report_sha256_matches":
            len(snapshot_rows) == 1
            and snapshot_rows[0]["sha256"] == P0_REPORT_SHA256,
        "superseded_report_sha256_matches":
            sha256_bytes(report_bytes) == SUPERSEDED_REPORT_SHA256,
        "superseded_notice_occurs_once": notice_count == 1,
        "removing_notice_reconstructs_p0_bytes":
            notice_count == 1
            and sha256_bytes(reconstructed) == P0_REPORT_SHA256,
        "notice_links_relative_authority":
            "[`BRANCH_C_UPDATE.md`](BRANCH_C_UPDATE.md)" in NOTICE
            and authority_path.resolve() == (ROOT / AUTHORITY).resolve(),
        "authority_sha256_matches":
            sha256_bytes(authority_bytes) == AUTHORITY_SHA256,
        "authority_declares_current_position":
            "状态：**当前文献定位依据。**" in authority_bytes.decode(),
        "authority_declares_final_refresh":
            "2026-10-02 已按冻结协议完成该次刷新" in authority_bytes.decode()
            and "swan_branch_c_final_literature_refresh_20261002"
            in authority_bytes.decode(),
        "all_82_preserved_originals_verified":
            legacy_result["historical_preservation"][
                "preserved_originals_verified"
            ] == 82,
        "all_16_endpoint_gates_reused":
            endpoint_ids == [f"CL{number:02d}" for number in range(1, 17)],
        "no_bad_endpoints": legacy_result["bad_endpoints"] == [],
        "no_new_model_parser_s2match_detector_observations":
            legacy_result["new_model_parser_s2match_detector_observations"] == 0,
    }
    return {
        "schema_version": 2,
        "ok": all(checks.values()),
        "scope": (
            "Versioned reconciliation of the legacy P0 reframing audit with "
            "the later Branch C literature supersession; no experiment rerun."
        ),
        "checks": checks,
        "supersession": {
            "historical_report": REPORT,
            "p0_sha256": P0_REPORT_SHA256,
            "current_sha256": SUPERSEDED_REPORT_SHA256,
            "notice_utf8_sha256": sha256_bytes(NOTICE.encode()),
            "notice_bytes": len(NOTICE.encode()),
            "reconstructed_p0_bytes": len(reconstructed),
            "authority": AUTHORITY,
            "pre_refresh_authority_sha256": PRE_REFRESH_AUTHORITY_SHA256,
            "authority_sha256": AUTHORITY_SHA256,
        },
        "legacy_audit": {
            "ok": legacy_result["ok"],
            "expected_unexpected_historical_changes":
                legacy_result["unexpected_historical_changes"],
            "historical_preservation": legacy_result["historical_preservation"],
            "approved_document_errata":
                legacy_result["approved_document_errata"],
            "text_scan": legacy_result["text_scan"],
            "endpoint_gates": legacy_result["endpoint_gates"],
            "bad_endpoints": legacy_result["bad_endpoints"],
            "new_model_parser_s2match_detector_observations":
                legacy_result[
                    "new_model_parser_s2match_detector_observations"
                ],
        },
        "limitations": [
            "The legacy audit and its allowlist remain unchanged and still return ok=false.",
            "This audit validates an exact notice insertion and authority chain; it does not authorize new experiments.",
            "The 2026-10-02 authority transition records the single final literature refresh; it adds no empirical endpoint.",
            "No API, neural parser, S2Match, detector, or sealed main source is accessed.",
        ],
    }


def frozen_bytes(path, expected):
    if not path.is_file() or path.read_bytes() != expected:
        raise ValueError(f"Frozen audit differs from offline replay: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=AUDIT_PATH)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="require the frozen audit to match a fresh offline replay",
    )
    args = parser.parse_args()
    payload = build()
    encoded = json_bytes(payload)
    if args.verify_only:
        frozen_bytes(args.output, encoded)
    else:
        with args.output.open("xb") as handle:
            handle.write(encoded)
    print(json.dumps({
        "ok": payload["ok"],
        "legacy_ok": payload["legacy_audit"]["ok"],
        "expected_legacy_difference":
            payload["legacy_audit"]["expected_unexpected_historical_changes"],
        "preserved_originals": payload["legacy_audit"][
            "historical_preservation"
        ]["preserved_originals_verified"],
        "endpoint_gates": len(payload["legacy_audit"]["endpoint_gates"]),
        "verified_only": args.verify_only,
    }, ensure_ascii=False))
    raise SystemExit(0 if payload["ok"] else 1)
