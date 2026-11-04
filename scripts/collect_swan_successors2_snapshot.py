"""Materialize byte-exact aggregate audit artifacts and check deployment identities."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/swan_research_completion_20261102/sequential_successors2"


def main(source):
    payload = json.loads(Path(source).read_text())
    stamp = datetime.fromtimestamp(payload["at_unix"], timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = OUT / ("snapshot_" + stamp)
    folder.mkdir(mode=0o700)
    hashes = {}
    for relative, record in payload.pop("exports").items():
        path = Path(relative)
        assert not path.is_absolute() and ".." not in path.parts and path.suffix == ".json"
        raw = base64.b64decode(record["base64"], validate=True)
        assert hashlib.sha256(raw).hexdigest() == record["sha256"]
        target = folder / path
        target.parent.mkdir(mode=0o700, exist_ok=True)
        target.write_bytes(raw)
        hashes[str(target.relative_to(OUT))] = record["sha256"]
    scripts = {}
    for label, filename in (
        ("calibration", "deploy_swan_calibration_successor2.py"),
        ("normal", "deploy_swan_normal_detection2.py"),
        ("endpoint", "deploy_swan_endpoint_successors2.py"),
        ("multiparser", "deploy_swan_endpoint_successors2.py"),
    ):
        digest = hashlib.sha256((ROOT / "scripts" / filename).read_bytes()).hexdigest()
        frozen = json.loads((folder / label / "FREEZE.json").read_text())
        assert frozen["files"]["DEPLOYMENT_SOURCE.py"] == digest
        scripts[label] = {"script": filename, "sha256": digest}
    (OUT / "LATEST_STATUS.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    audit = {"status": "PASS", "at_unix": payload["at_unix"], "snapshot": folder.name,
             "byte_exact_files_verified": len(hashes), "files": hashes,
             "deployment_sources_match": scripts, "all_experiments_complete": False,
             "sealed_source_or_generated_text_exported": False}
    (OUT / "AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "snapshot": folder.name,
                      "exported_files": len(hashes), "jobs": payload["jobs"],
                      "gpu": payload["gpu"], "free_bytes": payload["free_bytes"]}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
