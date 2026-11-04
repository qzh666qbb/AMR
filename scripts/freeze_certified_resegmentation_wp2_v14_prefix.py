"""Freeze the public WP2-v13 committed prefix for v14 replay checking."""

import argparse
import hashlib
import json
from pathlib import Path


V13_LOG_SHA256 = "c803a91e4f5be5ce645e70d8765913e913571d34d85dc9fda85ed1724fb7f46a"
PREFIX_ATTEMPTS = 372


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def freeze_prefix(log_path):
    if sha256_file(log_path) != V13_LOG_SHA256:
        raise ValueError("v13 generation log SHA mismatch")
    source_rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        if "attempt_index" in row:
            source_rows.append(row)
    if len(source_rows) != PREFIX_ATTEMPTS:
        raise ValueError("v13 committed log row count mismatch")
    rows = []
    expected_eligible_count = 0
    for index, source in enumerate(source_rows):
        codes = source.get("exclusion_codes")
        eligible = source.get("eligible")
        if (
            source.get("attempt_index") != index
            or type(eligible) is not bool
            or not isinstance(codes, list)
            or codes != sorted(set(codes))
            or eligible is not (not codes)
            or not is_sha256(source.get("output_sha256"))
        ):
            raise ValueError(f"invalid public v13 row {index}")
        expected_eligible_count += int(eligible)
        if source.get("eligible_count") != expected_eligible_count:
            raise ValueError(f"v13 eligible count drift at row {index}")
        rows.append({
            "attempt_index": index,
            "output_raw_text_sha256": source["output_sha256"],
            "eligible": eligible,
            "exclusion_codes": codes,
            "eligible_count": expected_eligible_count,
        })
    if expected_eligible_count != 326:
        raise ValueError("v13 terminal committed eligible count mismatch")
    return rows


def canonical_jsonl(rows):
    return (
        "\n".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"))
            for row in rows
        )
        + "\n"
    ).encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v13-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = freeze_prefix(args.v13_log)
    encoded = canonical_jsonl(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(json.dumps({
        "rows": len(rows),
        "eligible": sum(row["eligible"] for row in rows),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "contains_raw_text": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
