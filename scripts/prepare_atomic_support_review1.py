"""Generate two private offline full-universe reviewer pages from sealed inputs."""
import argparse
import hashlib
import json
from pathlib import Path


def build(data, output, allow_synthetic=False):
    if output.exists():
        raise ValueError("A new immutable review directory is required")
    if (data["schema"] != "atomic-support-private-review/v1"
            or data["scores_included"] is not False
            or data["predicted_alignment_included"] is not False):
        raise ValueError("Unblinded or foreign source packet")
    if len({r["row_id"] for r in data["rows"]}) != len(data["rows"]):
        raise ValueError("Duplicate row identity")
    pairs = 0
    for row in data["rows"]:
        events = {e["event_id"]: e for e in row["events"]}
        seen = set()
        for pair in row["pairs"]:
            key = pair["candidate_index"], pair["event_id"]
            if key in seen or pair["event_id"] not in events:
                raise ValueError("Duplicate or foreign pair")
            seen.add(key)
            if (events[pair["event_id"]]["candidate_index"] != pair["event_candidate_index"]
                    or not 0 <= pair["candidate_index"] < len(row["candidate_texts"])):
                raise ValueError("Candidate origin mismatch")
        pairs += len(seen)
    if not allow_synthetic and (len(data["rows"]) != 84 or pairs != 23229):
        raise ValueError("Expected the entire frozen development universe")
    template = Path(__file__).with_name("atomic_support_review1.html").read_text()
    # Escaping '<' prevents any source sentence from ending a script element.
    embedded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    output.mkdir(parents=True, mode=0o700)
    hashes = {}
    for role in ("reviewer1", "reviewer2"):
        content = template.replace("__ROLE__", role).replace("__DATA__", embedded)
        path = output / (role + ".html")
        path.write_text(content)
        path.chmod(0o600)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"rows": len(data["rows"]), "pairs_per_reviewer": pairs, "reviewers": 2,
            "actual_human_answers": 0, "source_freeze_sha256": data["freeze_sha256"],
            "prediction_seal_sha256": data["prediction_seal_sha256"],
            "files": hashes, "synthetic_test": allow_synthetic}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(json.loads(args.source.read_text()), args.output), indent=2))
