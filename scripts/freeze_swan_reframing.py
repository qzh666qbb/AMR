"""Preserve the pre-reframing inventory; never import experimental runners."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/swan_reframing1"
TREES = ("experiments", "archive", "artifacts", "scripts", "paper")
EXCLUDED_PARTS = {".git", "__pycache__", ".secrets", ".pytest_cache"}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def create(path, value):
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded(value))


def paths():
    found = set(ROOT.glob("*.md"))
    for tree in TREES:
        found.update((ROOT / tree).rglob("*"))
    return sorted(
        path for path in found
        if path.is_file() and not path.is_symlink()
        and OUTPUT not in path.parents
        and not EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts)
        and path.suffix not in {".pyc", ".tmp"}
        and path.name != ".DS_Store"
    )


def should_copy(path):
    relative = path.relative_to(ROOT)
    return (
        path.parent == ROOT
        or relative.parts[0] == "archive" and path.suffix in {".md", ".tex"}
        or relative.as_posix() in {
            "experiments/swan_amr_composition_decomposition1/report.md",
            "experiments/swan_amr_precision_recall1/report.md",
            "experiments/swan_length_gate_tradeoff1/report.md",
        }
    )


def freeze():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "pre_reset_hashes.json").exists():
        raise FileExistsError("Pre-reset snapshot exists; use check, never replace")
    rows = []
    copies = {}
    inventory = paths()
    for index, path in enumerate(inventory, 1):
        before = path.stat()
        digest = sha(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (
            after.st_size, after.st_mtime_ns
        ):
            raise RuntimeError(f"File changed while hashing: {path}")
        relative = path.relative_to(ROOT).as_posix()
        rows.append({"path": relative, "bytes": after.st_size, "sha256": digest})
        if should_copy(path):
            copy = OUTPUT / "originals" / relative
            copy.parent.mkdir(parents=True, exist_ok=True)
            if copy.exists():
                raise FileExistsError(copy)
            shutil.copyfile(path, copy)
            if sha(copy) != digest:
                raise ValueError(f"Snapshot copy mismatch: {relative}")
            copies[relative] = digest
        if index % 5000 == 0:
            print(f"hashed {index}/{len(inventory)}", flush=True)
    aggregate = hashlib.sha256(encoded(rows).encode()).hexdigest()
    git_status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout
    create(OUTPUT / "pre_reset_hashes.json", {
        "schema_version": 1, "files": rows, "aggregate_sha256": aggregate,
        "copied_originals": copies,
    })
    create(OUTPUT / "pre_reset_inventory.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "meaning": "State at this continuation, after the prior cleanup; not the state before that cleanup.",
        "root": str(ROOT), "included_trees": TREES,
        "root_markdown_included": True,
        "exclusions": [
            "credentials, git internals, symlinks, temporary files, bytecode",
            "model/vector/third-party/environment trees: separately frozen by historical manifests",
            "new swan_reframing1 tree",
        ],
        "files": len(rows), "bytes": sum(row["bytes"] for row in rows),
        "copies": len(copies), "aggregate_sha256": aggregate,
        "git_status_sha256": hashlib.sha256(git_status).hexdigest(),
        "new_model_parser_s2match_detector_calls": 0,
    })
    return {"files": len(rows), "copies": len(copies), "aggregate_sha256": aggregate}


def check():
    hashes = json.loads((OUTPUT / "pre_reset_hashes.json").read_text())
    if hashlib.sha256(encoded(hashes["files"]).encode()).hexdigest() != hashes["aggregate_sha256"]:
        raise ValueError("Inventory checksum mismatch")
    for relative, expected in hashes["copied_originals"].items():
        if sha(OUTPUT / "originals" / relative) != expected:
            raise ValueError(f"Preserved original changed: {relative}")
    changes = []
    for row in hashes["files"]:
        path = ROOT / row["path"]
        if not path.is_file() or sha(path) != row["sha256"]:
            changes.append(row["path"])
    return {
        "preserved_originals_verified": len(hashes["copied_originals"]),
        "inventoried_files": len(hashes["files"]),
        "changes_since_snapshot": changes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "check"])
    args = parser.parse_args()
    print(encoded(freeze() if args.stage == "freeze" else check()))
