"""Locate frozen certificate inputs by SHA without reading text into reports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

SKIP = {".git", "node_modules", "__pycache__", ".venv", "python_env", "responses",
        "raw_graphs", "candidate_graphs"}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def scan(expected, roots):
    """Named files plus small HF blobs; report exact matches, never approximate."""
    wanted = set(expected.values())
    names = {Path(p).name for p in expected}
    hits = {digest: [] for digest in wanted}
    checked, errors, seen = 0, [], set()
    for label, root in roots.items():
        root = Path(root).resolve()
        if not root.is_dir():
            errors.append({"root": label, "reason": "root_missing"})
            continue
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP)
            for name in sorted(files):
                path = Path(directory) / name
                try:
                    stat = path.stat()
                    key = (stat.st_dev, stat.st_ino)
                    if key in seen:
                        continue
                    if name not in names and not (
                            "blobs" in path.parts and stat.st_size < 10 * 1024**2):
                        continue
                    seen.add(key)
                    digest = sha(path)
                    checked += 1
                    if digest in hits:
                        hits[digest].append({"root": label,
                                            "relative_path": str(path.relative_to(root)),
                                            "bytes": stat.st_size})
                except OSError as exc:
                    errors.append({"root": label, "relative_path": str(path.relative_to(root)),
                                   "reason": type(exc).__name__})
    rows = [{"frozen_path": path, "expected_sha256": digest,
             "matches": hits[digest], "recovered": bool(hits[digest])}
            for path, digest in sorted(expected.items())]
    return {"schema": "completion-asset-recovery/v1", "at_unix": time.time(),
            "files_hashed": checked, "rows": rows, "errors": errors,
            "resolved": sum(r["recovered"] for r in rows), "requested": len(rows),
            "scope": "named files and small HF blobs in listed roots; archives not unpacked",
            "source_text_emitted": False, "files_modified": False,
            "scientific_result": None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--root", action="append", required=True, help="label=absolute_path")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = dict(r.split("=", 1) for r in args.root)
    result = scan(json.loads(args.expected.read_text()), roots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError("Choose a new audit output; preserve earlier evidence")
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("requested", "resolved", "files_hashed", "errors")}))
