"""Offline audit for the frozen count-neutral boundary-exchange experiment."""

import argparse
import json
import os
from pathlib import Path
import socket
import sys

from run_segmentation_intervention import ROOT, sha, write_json
from run_swan_count_neutral_boundary import (
    build_manifest,
    render_report,
    summarize,
)


class NetworkDisabled:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("Network access is disabled during offline audit")


def freeze(path, payload):
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f"Frozen audit artifact differs: {path}")
    else:
        write_json(path, payload)


def audit(output):
    import nltk
    import spacy

    manifest_path = output / "manifest.json"
    summary_path = output / "summary.json"
    report_path = output / "report.md"
    protocol_path = output / "protocol.md"
    for path in (manifest_path, summary_path, report_path, protocol_path):
        if not path.exists():
            raise FileNotFoundError(path)

    saved_manifest = json.loads(manifest_path.read_text())
    saved_summary = json.loads(summary_path.read_text())
    saved_report = report_path.read_text()
    original_socket = socket.socket
    socket.socket = NetworkDisabled
    try:
        nlp = spacy.load("en_core_web_sm")
        rebuilt_manifest = build_manifest(nlp, nltk.sent_tokenize)
    finally:
        socket.socket = original_socket

    rebuilt_summary = summarize(rebuilt_manifest)
    rebuilt_report = render_report(rebuilt_summary)
    detected_files = sorted((output / "detected").glob("*.json"))
    checks = {
        "manifest_exact": rebuilt_manifest == saved_manifest,
        "summary_exact": rebuilt_summary == saved_summary,
        "report_exact": rebuilt_report == saved_report,
        "document_count_30": len(saved_manifest["documents"]) == 30,
        "record_count_0": len(saved_manifest["records"]) == 0,
        "split_coverage_0": (
            saved_manifest["coverage"]["documents_with_split_candidate"] == 0
        ),
        "exchange_coverage_0": (
            saved_manifest["coverage"]["eligible_detected_exchanges"] == 0
        ),
        "no_detected_outputs": not detected_files,
        "amrlib_not_loaded": "amrlib" not in sys.modules,
        "swan_amr_utils_not_loaded": "utils.amr_utils" not in sys.modules,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Audit failed: {failed}")

    stable_paths = [
        protocol_path,
        manifest_path,
        summary_path,
        report_path,
    ]
    audit_payload = {
        "status": "pass",
        "checks": checks,
        "coverage": saved_manifest["coverage"],
        "detected_files": [],
        "stable_artifacts": {
            str(path.relative_to(ROOT)): sha(path) for path in stable_paths
        },
        "audit_script_sha256": sha(Path(__file__)),
    }
    audit_path = output / "audit.json"
    freeze(audit_path, audit_payload)

    recovery_payload = {
        "status": "pass",
        "network_disabled": True,
        "manifest_recomputed": True,
        "amr_parser_loaded": False,
        "s2match_loaded": False,
        "stable_artifacts": {
            **audit_payload["stable_artifacts"],
            str(audit_path.relative_to(ROOT)): sha(audit_path),
        },
    }
    freeze(output / "recovery_verification.json", recovery_payload)
    print(json.dumps(audit_payload, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "experiments/swan_count_neutral_boundary1",
    )
    args = parser.parse_args()
    os.environ.setdefault("NLTK_DATA", str(ROOT / ".nltk_data"))
    audit(args.output)


if __name__ == "__main__":
    main()
