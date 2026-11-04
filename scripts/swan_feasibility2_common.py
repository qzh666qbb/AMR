"""Provenance primitives for the normal-length SWAN feasibility pilot."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/swan_normal_length_feasibility2"
EXPERIMENT = "swan_normal_length_feasibility2"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def digest(value):
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def freeze(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if load(path) != value:
            raise ValueError(f"Frozen artifact differs: {path}")
        return value
    fd, temporary = tempfile.mkstemp(prefix="freeze-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return value


@contextmanager
def single_writer(folder=OUT):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".writer.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def seed(*parts):
    return int(digest([EXPERIMENT, *parts])[:8], 16)


def assert_hashes(mapping, root=ROOT):
    for relative, expected in mapping.items():
        path = Path(relative)
        path = path if path.is_absolute() else Path(root) / path
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Frozen hash differs: {relative}")
