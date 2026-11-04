"""Binary sidecars and deterministic reference features for WP2-v6."""

import math
from pathlib import Path
import struct


MAGIC = b"WP2V6F1\0"
SEMSTAMP_MODE = 5
SIMMARK_MODE = 6
SIMMARK_DIMENSION = 768


def as_float32(value):
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def synthetic_relation():
    return bytes(
        int(((previous + 1) * 5 + (current + 3) * 3) % 4 == 0)
        for previous in range(8)
        for current in range(8)
    )


def synthetic_region(start, end):
    return ((start + 3) * 5 + (end + 7) * 3) % 8


def synthetic_embedding(start, end):
    raw = (
        float((start * 13 + end * 7) % 29 + 1),
        float((start * 5 + end * 17) % 31 + 1),
        float((start * 19 + end * 3) % 37 + 1),
        float((start * 11 + end * 23) % 41 + 1),
    )
    norm = math.sqrt(sum(value * value for value in raw))
    values = [as_float32(value / norm) for value in raw]
    values.extend([0.0] * (SIMMARK_DIMENSION - len(values)))
    return tuple(values)


def cosine_soft_count(left, right):
    if len(left) != SIMMARK_DIMENSION or len(right) != SIMMARK_DIMENSION:
        raise ValueError("SimMark embeddings must contain 768 values")
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if not left_norm > 0 or not right_norm > 0:
        raise ValueError("SimMark embeddings must have nonzero norm")
    cosine = dot / (left_norm * right_norm)
    distance = (
        0.0
        if 0.68 <= cosine <= 0.76
        else min(abs(cosine - 0.68), abs(cosine - 0.76))
    )
    return math.exp(-250.0 * distance)


def feature_pair_score(mode, features, relation, left, middle, right):
    previous = features[(left, middle)]
    current = features[(middle, right)]
    if mode == SEMSTAMP_MODE:
        return float(relation[previous * 8 + current])
    if mode == SIMMARK_MODE:
        return cosine_soft_count(previous, current)
    raise ValueError("unsupported feature mode")


def encode_sidecar(mode, token_count, records, relation=None):
    if mode not in {SEMSTAMP_MODE, SIMMARK_MODE}:
        raise ValueError("mode must be 5 or 6")
    ordered = sorted(records, key=lambda row: (row[0], row[1]))
    if ordered != list(records):
        raise ValueError("records must be strictly sorted")
    if len({(row[0], row[1]) for row in ordered}) != len(ordered):
        raise ValueError("records contain duplicate spans")

    dimension = 1 if mode == SEMSTAMP_MODE else SIMMARK_DIMENSION
    output = bytearray(MAGIC)
    output.extend(struct.pack(
        "<IIQI",
        mode,
        token_count,
        len(ordered),
        dimension,
    ))
    if mode == SEMSTAMP_MODE:
        relation = synthetic_relation() if relation is None else bytes(relation)
        if len(relation) != 64 or any(value not in {0, 1} for value in relation):
            raise ValueError("SemStamp relation must contain 64 binary values")
        output.extend(relation)

    for start, end, allowed, feature in ordered:
        if not (0 <= start < end <= token_count):
            raise ValueError("invalid span endpoints")
        if not isinstance(allowed, bool):
            raise TypeError("allowed must be bool")
        output.extend(struct.pack("<IIB", start, end, allowed))
        if not allowed:
            if feature is not None:
                raise ValueError("illegal spans cannot carry features")
            continue
        if mode == SEMSTAMP_MODE:
            if not isinstance(feature, int) or not 0 <= feature <= 7:
                raise ValueError("SemStamp region must be in [0, 7]")
            output.extend(struct.pack("<B", feature))
        else:
            if len(feature) != SIMMARK_DIMENSION:
                raise ValueError("SimMark feature dimension must be 768")
            output.extend(struct.pack(
                f"<{SIMMARK_DIMENSION}f",
                *feature,
            ))
    return bytes(output)


def write_sidecar(path, mode, token_count, records, relation=None):
    encoded = encode_sidecar(
        mode,
        token_count,
        records,
        relation=relation,
    )
    Path(path).write_bytes(encoded)
    return encoded


def serialize_core_problem(problem, mode, sidecar_path):
    normalization = (
        len(problem.original_boundaries)
        if problem.normalization is None
        else float(problem.normalization)
    )
    fields = [
        problem.token_count,
        len(problem.original_boundaries),
        problem.max_boundary_moves,
        mode,
        2,
        format(float(normalization), ".17g"),
        format(float(problem.bias), ".17g"),
        format(float(problem.threshold), ".17g"),
        *problem.original_boundaries,
        str(Path(sidecar_path)),
    ]
    return ("1\n" + " ".join(map(str, fields)) + "\n").encode("ascii")
