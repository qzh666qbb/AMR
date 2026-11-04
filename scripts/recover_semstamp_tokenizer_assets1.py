"""Restore only byte-identical public tokenizer assets into a new recovery folder."""
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/swan_research_completion_20261102/certificate_asset_recovery1"
REPO = "AbeHou/SemStamp-c4-sbert"
REVISION = "a77dce8530e100ac1101981e4be390546c1cb282"


def main():
    expected = json.loads((OUT / "EXPECTED.json").read_text())
    rows = []
    for original, digest in expected.items():
        if not original.startswith("models/semstamp-c4-sbert-"):
            continue
        filename = Path(original).name
        destination = OUT / "recovered" / original
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{filename}"
        if destination.exists():
            data = destination.read_bytes()
        else:
            response = requests.get(url, timeout=(20, 45))
            response.raise_for_status()
            data = response.content
        actual = hashlib.sha256(data).hexdigest()
        if actual != digest:
            raise RuntimeError("Public asset does not match frozen SHA: " + filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(data)
        rows.append({"original_path": original, "recovered_path": str(destination.relative_to(OUT)),
                     "sha256": actual, "bytes": len(data), "url": url})
    audit = {"schema": "exact-public-asset-recovery/v1", "assets": rows,
             "repository": REPO, "revision": REVISION, "recovered": len(rows),
             "sealed_sources_replaced": False, "detector_calls": 0}
    path = OUT / "TOKENIZER_RECOVERY.json"
    encoded = json.dumps(audit, indent=2) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise RuntimeError("Recovery manifest changed")
    path.write_text(encoded)
    print(json.dumps({"recovered": len(rows), "sha_exact": True}))


if __name__ == "__main__":
    main()
