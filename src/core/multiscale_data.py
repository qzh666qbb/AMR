"""Strict adapter from saved FUSION_RECOVERY vectors to score-level MSA.

Historical rows use carrier-slot spans. These are oracle-aligned diagnostics,
not a blind document detector; joining caches cannot recover lost positions.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(path):
    path = Path(path)
    files = [path] if path.is_file() else sorted(path.glob("s*/FUSION_RECOVERY.json"))
    if not files:
        raise ValueError(f"No score files: {path}")
    indexed, sources = {}, []
    for source in files:
        payload = json.loads(source.read_text())
        rows = payload["per_instance"]
        if not isinstance(rows, list):
            raise ValueError(f"per_instance must be a list: {source}")
        for row in rows:
            if (not isinstance(row.get("doc"), str) or not row["doc"]
                    or not isinstance(row.get("attack"), str) or not row["attack"]
                    or type(row.get("slot")) is not int or row["slot"] < 0):
                raise ValueError(f"Invalid instance key in {source}")
            key = (row["doc"], row["attack"], row["slot"])
            if key in indexed:
                raise ValueError(f"Duplicate instance key: {key}")
            indexed[key] = row
        sources.append({"path": str(source.resolve()), "sha256": sha256_file(source),
                        "rows": len(rows)})
    if not indexed:
        raise ValueError(f"No score rows: {path}")
    return indexed, sources


def _vector(row, field, templates):
    if row is None:
        return [math.nan] * templates
    vector = row.get(field)
    if not isinstance(vector, list) or len(vector) != templates:
        raise ValueError(f"{field} must contain exactly {templates} scores")
    result = []
    for value in vector:
        if value is None:
            result.append(math.nan)
        elif (isinstance(value, bool) or not isinstance(value, (int, float))
              or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"Invalid {field} score")
        else:
            result.append(float(value))
    return result


@dataclass
class CachedScales:
    keys: list
    scores: torch.Tensor
    sources: list
    missing_short_rows: int
    missing_main_rows: int

    @property
    def documents(self):
        return sorted({key[0] for key in self.keys})

    def indices(self, documents):
        documents = set(documents)
        return [i for i, key in enumerate(self.keys) if key[0] in documents]

    def audit(self):
        finite = torch.isfinite(self.scores)
        return {
            "instances": len(self.keys), "documents": len(self.documents),
            "scales": ["S6", "S12", "Ssub"], "templates": self.scores.shape[2],
            "missing_short_rows": self.missing_short_rows,
            "missing_main_rows": self.missing_main_rows,
            "available_rows_by_scale": finite.any(2).sum(0).tolist(),
            "all_missing_rows": int((~finite.any(2).any(1)).sum()),
            "sources": self.sources,
            "unit_origin": "historical_oracle_carrier_slot_alignment",
            "bank_order": "inherited_from_legacy_cache_not_independently_verified",
        }


def load_cached_scales(main_path, short_path, n_templates=100):
    main, main_sources = load_rows(main_path)
    short, short_sources = load_rows(short_path)
    # Union, not intersection: missing views must remain visible.
    keys = sorted(set(main) | set(short))
    scores = [[_vector(short.get(k), "vec_d1", n_templates),
               _vector(main.get(k), "vec_d1", n_templates),
               _vector(main.get(k), "vec_d2", n_templates)] for k in keys]
    return CachedScales(
        keys, torch.tensor(scores, dtype=torch.float32), main_sources + short_sources,
        len(set(main) - set(short)), len(set(short) - set(main)))


def validate_split(split, positive, negative):
    names = ("train", "validation", "calibration", "test")
    if set(split) != set(names):
        raise ValueError(f"Split must have exactly {names}")
    seen = set()
    for name in names:
        docs = split[name]
        if (not isinstance(docs, list) or not docs
                or any(not isinstance(doc, str) for doc in docs)
                or len(docs) != len(set(docs))):
            raise ValueError(f"Invalid {name} document list")
        if seen & set(docs):
            raise ValueError("Source-document leakage across partitions")
        seen.update(docs)
    if seen != set(positive.documents) or seen != set(negative.documents):
        raise ValueError("Split must cover both cohorts' source documents exactly")
    if set(positive.keys) != set(negative.keys):
        raise ValueError("Paired cohorts have different union keys; audit before training")
    return split
