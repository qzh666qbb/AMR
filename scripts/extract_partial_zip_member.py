"""Extract one ZIP member using a local prefix and HTTP byte-range fallback.

zipfile verifies the member CRC. This does not verify the whole archive SHA.
"""

import argparse
import io
import json
from pathlib import Path
import shutil
import zipfile

import requests

from run_segmentation_intervention import sha, write_json


class PrefixRangeReader(io.RawIOBase):
    def __init__(self, prefix, url):
        self.local = prefix.open("rb")
        self.prefix_size = prefix.stat().st_size
        self.session = requests.Session()
        self.url = url
        response = self.session.head(url, allow_redirects=True, timeout=(20, 30))
        response.raise_for_status()
        self.size = int(response.headers["Content-Length"])
        self.etag = response.headers.get("ETag")
        self.position = 0
        self.remote_bytes = 0
        if self.prefix_size > self.size:
            raise ValueError("Local prefix is larger than remote archive")

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.position + size)
        if end <= self.position:
            return b""
        chunks = []
        if self.position < self.prefix_size:
            self.local.seek(self.position)
            block = self.local.read(min(end, self.prefix_size) - self.position)
            if not block:
                raise OSError("Local prefix truncated")
            chunks.append(block)
            self.position += len(block)
        if self.position < end:
            headers = {"Range": f"bytes={self.position}-{end - 1}"}
            # Some archive servers reject ranges on a connection reused after HEAD.
            response = requests.get(self.url, headers=headers, timeout=(20, 60))
            response.raise_for_status()
            if self.etag and response.headers.get("ETag") != self.etag:
                raise OSError("Remote archive changed during extraction")
            expected = f"bytes {self.position}-{end - 1}/{self.size}"
            if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                raise OSError("Server did not honor the exact range")
            block = response.content
            if len(block) != end - self.position:
                raise OSError("Incomplete range")
            chunks.append(block)
            self.remote_bytes += len(block)
            self.position = end
        return b"".join(chunks)

    def close(self):
        self.local.close()
        self.session.close()
        super().close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--member", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    with PrefixRangeReader(args.prefix, args.url) as reader, zipfile.ZipFile(reader) as archive:
        info = archive.getinfo(args.member)
        with archive.open(info) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        if temporary.stat().st_size != info.file_size:
            raise OSError("Extracted size mismatch")
        temporary.replace(args.output)
        result = {"url": args.url, "etag": reader.etag, "archive_bytes": reader.size,
                  "local_prefix_bytes": reader.prefix_size, "remote_bytes": reader.remote_bytes,
                  "member": args.member, "crc32": f"{info.CRC:08x}", "size": info.file_size,
                  "sha256": sha(args.output), "validation": "ZIP member CRC and size; not full archive SHA"}
        write_json(args.output.with_suffix(".provenance.json"), result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
