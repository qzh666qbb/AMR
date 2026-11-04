"""Process leases, heartbeats and versioned sentence caches for new experiments."""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
import threading
import time
import uuid

from run_confirmation_experiment import Detector, digest, freeze
from run_segmentation_intervention import write_json


@contextmanager
def claim(path):
    """Kernel-managed leases release on process death, without deleting lock files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Task already claimed: {path.name}") from None
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class Heartbeat:
    def __init__(self, folder, interval=15):
        self.folder, self.interval = folder, interval
        self.started = time.monotonic()
        self.run_id = uuid.uuid4().hex
        self.state = {"stage": "starting"}
        self.stop = threading.Event()
        self.guard = threading.Lock()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def update(self, **state):
        with self.guard:
            self.state = {**self.state, **state}

    def emit(self):
        with self.guard:
            payload = {
                **self.state, "pid": os.getpid(), "run_id": self.run_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - self.started,
            }
        write_json(self.folder / "run_logs" / f"{self.run_id}.json", payload)

    def loop(self):
        while not self.stop.wait(self.interval):
            self.emit()

    def __enter__(self):
        self.emit()
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set()
        self.thread.join()
        self.update(status="failed" if exc_type else "complete",
                    error_type=exc_type.__name__ if exc_type else None)
        self.emit()


class CachedDetector(Detector):
    def __init__(self, folder, runtime_signature):
        self.cache = folder / "sentence_cache" / runtime_signature
        self.counts = {"parse_hits": 0, "parse_misses": 0, "score_hits": 0, "score_misses": 0}
        super().__init__()

    def memo(self, kind, key, compute):
        path = self.cache / kind / f"{digest(key)}.json"
        with claim(path.with_suffix(".lock")):
            if path.exists():
                saved = json.loads(path.read_text())
                if saved["key"] != key or saved["value_sha256"] != digest(saved["value"]):
                    raise ValueError("Sentence cache identity or payload mismatch")
                self.counts[f"{kind}_hits"] += 1
                return saved["value"]
            start = time.monotonic()
            value = compute()
            freeze(path, {"key": key, "value": value, "value_sha256": digest(value),
                          "elapsed_seconds": time.monotonic() - start})
            self.counts[f"{kind}_misses"] += 1
            return value

    def parse(self, text):
        import nltk
        details = []
        for sentence in nltk.sent_tokenize(text):
            items = self.memo("parse", {"text": sentence}, lambda: super(CachedDetector, self).parse(sentence))
            if len(items) != 1 or items[0]["text"] != sentence:
                raise ValueError("Sentence cache segmentation mismatch")
            details.extend(items)
        if not details:
            raise ValueError("Empty paragraph")
        return details

    def score(self, details):
        matched = []
        for item in details:
            scores = self.memo("score", {"amr": item["amr"]}, lambda: [
                float(self.utils.compute_s2match_score(item["amr"], template)) for template in self.bank])
            if len(scores) != 50 or not all(math.isfinite(v) and 0 <= v <= 1 for v in scores):
                raise ValueError("Invalid cached bank scores")
            matched.append({**item, "bank_scores": scores, "score": max(scores)})
        n = len(matched)
        if not n:
            raise ValueError("Cannot score an empty paragraph")
        green = sum(item["score"] >= .65 for item in matched)
        return {"n": n, "green": green, "z": (green - .25 * n) / math.sqrt(n * .25 * .75),
                "details": matched}
