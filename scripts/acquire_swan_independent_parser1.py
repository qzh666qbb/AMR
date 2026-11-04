"""Fetch a pinned author's AMRBART checkpoint; validate LFS and git object hashes."""
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
REPO = "xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing-v2"
REVISION = "12f7e78e3dfa6f37ddbbf77c4c152b4fc39bc512"
OUT = ROOT / "models/amrbart-amr3-v2-12f7e78e3dfa"


def main():
    OUT.mkdir(exist_ok=True)
    response = requests.get(f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}",
                            params={"recursive": "true"}, timeout=(20, 45))
    response.raise_for_status()
    tree = response.json()
    if len(tree) != 8 or any(r["type"] != "file" or "/" in r["path"] for r in tree):
        raise RuntimeError("Pinned repository file set differs")
    rows = []
    for item in tree:
        target = OUT / item["path"]
        part = target.with_suffix(target.suffix + ".part")
        if not target.exists():
            offset = part.stat().st_size if part.exists() else 0
            if offset > item["size"]:
                raise RuntimeError("Partial download exceeds pinned size")
            if offset < item["size"]:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with requests.get(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{item['path']}",
                                  headers=headers, stream=True, timeout=(20, 90)) as stream:
                    stream.raise_for_status()
                    if offset and (stream.status_code != 206 or not
                                   stream.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                        raise RuntimeError("Server did not honor resumed byte range")
                    with part.open("ab" if offset else "wb") as output:
                        for chunk in stream.iter_content(4 * 1024**2):
                            output.write(chunk)
                if part.stat().st_size != item["size"]:
                    raise RuntimeError("Downloaded size differs")
            candidate = part
        else:
            candidate = target
        sha256 = hashlib.sha256()
        git_sha = hashlib.sha1(f"blob {item['size']}\0".encode())
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
                sha256.update(chunk)
                git_sha.update(chunk)
        if candidate.stat().st_size != item["size"]:
            raise RuntimeError("Asset size differs")
        if "lfs" in item:
            if sha256.hexdigest() != item["lfs"]["oid"]:
                raise RuntimeError("LFS SHA256 differs")
        elif git_sha.hexdigest() != item["oid"]:
            raise RuntimeError("Git blob SHA differs")
        if candidate == part:
            part.replace(target)
        rows.append({"file": item["path"], "bytes": item["size"], "sha256": sha256.hexdigest(),
                     "upstream_object_id": item.get("lfs", {}).get("oid", item["oid"])})
        print(json.dumps({"verified": item["path"], "bytes": item["size"]}), flush=True)
    value = {"repository": REPO, "revision": REVISION, "files": rows,
             "decoder_source_verified": False, "inference_executed": False,
             "status": "CHECKPOINT_ACQUIRED_ONLY"}
    manifest = OUT / "ASSET_MANIFEST.json"
    encoded = json.dumps(value, indent=2) + "\n"
    if manifest.exists() and manifest.read_text() != encoded:
        raise RuntimeError("Asset manifest changed")
    manifest.write_text(encoded)


if __name__ == "__main__":
    main()
