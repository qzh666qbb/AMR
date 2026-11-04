"""Build and verify the exact S2Match runtime required by official SWAN."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/amr-metric-suite_official_clean"
OUTPUT = ROOT / "third_party/swan_s2match_runtime_711a231d"
PATCH = (
    ROOT
    / "third_party/SWAN_official_2ca0c207_clean/utils/s2match_patch.py"
)
EXPECTED_COMMIT = "711a231d3600139662fe048460e10d773ff8e214"
EXPECTED_ORIGIN = "https://github.com/flipz357/amr-metric-suite.git"
SOURCE_DIR = UPSTREAM / "py3-Smatch-and-S2match/smatch"
COPIED_FILES = ("__init__.py", "amr_py3.py", "helpers.py")
MANIFEST = OUTPUT / "provenance.json"


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(UPSTREAM), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def source_evidence() -> dict:
    if git("rev-parse", "HEAD") != EXPECTED_COMMIT:
        raise ValueError("S2Match upstream commit changed")
    if git("config", "--get", "remote.origin.url") != EXPECTED_ORIGIN:
        raise ValueError("S2Match upstream origin changed")
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("S2Match upstream checkout is not clean")
    source_files = {
        name: sha256(SOURCE_DIR / name)
        for name in (*COPIED_FILES, "s2match.py")
    }
    return {
        "repository": EXPECTED_ORIGIN.removesuffix(".git"),
        "commit": EXPECTED_COMMIT,
        "tree_oid": git("rev-parse", f"{EXPECTED_COMMIT}^{{tree}}"),
        "local_root": str(UPSTREAM.relative_to(ROOT)),
        "clean": True,
        "source_file_sha256": source_files,
        "swan_patch_path": str(PATCH.relative_to(ROOT)),
        "swan_patch_sha256": sha256(PATCH),
    }


def expected_outputs() -> dict[str, bytes]:
    outputs = {
        name: (SOURCE_DIR / name).read_bytes()
        for name in COPIED_FILES
    }
    upstream = (SOURCE_DIR / "s2match.py").read_bytes().rstrip(b"\n")
    patch = PATCH.read_bytes().rstrip(b"\n")
    outputs["s2match.py"] = upstream + b"\n\n" + patch + b"\n"
    return outputs


def build() -> dict:
    evidence = source_evidence()
    outputs = expected_outputs()
    if OUTPUT.exists():
        return check()
    OUTPUT.mkdir(parents=True)
    for name in COPIED_FILES:
        shutil.copyfile(SOURCE_DIR / name, OUTPUT / name)
    (OUTPUT / "s2match.py").write_bytes(outputs["s2match.py"])
    manifest = {
        "schema_version": 1,
        "status": "built_from_frozen_upstream_plus_official_swan_patch",
        "source": evidence,
        "construction": (
            "Copy __init__.py, amr_py3.py, and helpers.py; append the exact "
            "official SWAN s2match_patch.py bytes to upstream s2match.py."
        ),
        "runtime_file_sha256": {
            name: sha256(OUTPUT / name)
            for name in sorted(outputs)
        },
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return check()


def check() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    evidence = source_evidence()
    outputs = expected_outputs()
    if manifest["source"] != evidence:
        raise ValueError("Frozen S2Match source evidence changed")
    for name, content in outputs.items():
        path = OUTPUT / name
        if not path.is_file() or path.read_bytes() != content:
            raise ValueError(f"S2Match runtime differs: {name}")
        if manifest["runtime_file_sha256"][name] != sha256(path):
            raise ValueError(f"S2Match runtime hash differs: {name}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build", "check"))
    args = parser.parse_args()
    result = build() if args.stage == "build" else check()
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
