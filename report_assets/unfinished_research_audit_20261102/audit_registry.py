"""Verify this curated index and record source identity; never run experiments."""
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPECTED = {
    "swan_open": 8,
    "unverified_extension": 4,
    "semstamp_open": 2,
    "completed_comparison": 8,
    "withdrawn_conclusion": 1,
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    with (HERE / "REGISTRY.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids)), "duplicate research-question ID"
    counts = dict(Counter(row["group"] for row in rows))
    assert counts == EXPECTED, counts
    sources = sorted({
        name for row in rows for name in row["source_paths"].split("|")
    })
    hashes = {}
    for name in sources:
        path = ROOT / name
        assert path.is_file(), f"missing source: {name}"
        hashes[name] = {"sha256": sha(path), "bytes": path.stat().st_size}
    write_json(HERE / "SOURCE_SHA256.json", hashes)
    missing = []
    checked = 0
    for target in re.findall(r"\]\(([^)]+)\)", (HERE / "REPORT.md").read_text()):
        if target == "AUDIT.json":
            continue  # This check creates the artifact.
        parsed = urlsplit(target)
        if parsed.scheme not in ("", "file"):
            continue
        path = Path(unquote(parsed.path))
        if not path.is_absolute():
            path = HERE / path
        checked += 1
        if not path.exists():
            missing.append(target)
    assert not missing, missing
    result = {
        "status": "PASS",
        "date": "2026-11-02",
        "registry_rows": len(rows),
        "counts": counts,
        "swan_related_open_questions": 12,
        "including_semstamp_open_questions": 14,
        "unique_source_files": len(sources),
        "local_report_links_checked": checked,
        "missing_links": missing,
        "registry_sha256": sha(HERE / "REGISTRY.tsv"),
        "report_sha256": sha(HERE / "REPORT.md"),
        "source_manifest_sha256": sha(HERE / "SOURCE_SHA256.json"),
        "scope": "Curated source/status audit, not a rerun or independent metric recomputation.",
    }
    write_json(HERE / "AUDIT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
