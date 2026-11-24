#!/usr/bin/env python3
"""Fetch exactly four pinned public judge heads; no credentials or source override."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.parse
import urllib.request

REPOSITORY = "https://huggingface.co/krummholz-ai/krummholz-harness"
REVISION = "30f793a03d7ba5ca0f123f275171f04ac04aab4d"
HEAD_PATHS = (
    "judge/models/request.kzj",
    "judge/models/reply.kzj",
    "judge/models/request.kze",
    "judge/models/reply.kze",
)
ROOT = Path(__file__).resolve().parent / "vendor" / "krummholz-harness"


class HttpsRedirectsOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        if urllib.parse.urlsplit(new_url).scheme != "https":
            raise RuntimeError("Refusing a non-HTTPS dependency redirect")
        return super().redirect_request(request, fp, code, message, headers, new_url)


def valid_existing(path, size, digest):
    if not path.is_file() or path.stat().st_size != size:
        return False
    return hashlib.sha256(path.read_bytes()).hexdigest() == digest


def main():
    if len(sys.argv) != 1:
        raise ValueError("This script accepts no arguments or source overrides")
    manifest = json.loads((ROOT / "PROVENANCE.json").read_text())
    if manifest.get("repository") != REPOSITORY or manifest.get("revision") != REVISION:
        raise ValueError("The dependency manifest does not match the approved public source pin")
    opener = urllib.request.build_opener(HttpsRedirectsOnly())
    for relative in HEAD_PATHS:
        entries = [entry for entry in manifest["files"] if entry.get("path") == relative]
        if len(entries) != 1:
            raise ValueError("Missing or duplicated pinned dependency: " + relative)
        entry = entries[0]
        size, digest = entry["size_bytes"], entry["sha256"]
        if not isinstance(size, int) or not 0 < size < 2 * 1024 * 1024:
            raise ValueError("Unexpected judge-head size: " + relative)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid SHA-256 pin: " + relative)
        destination = ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink() or ROOT.resolve() not in destination.resolve().parents:
            raise ValueError("Dependency destination must remain inside the bundle")
        if valid_existing(destination, size, digest):
            print("Verified existing: " + relative)
            continue

        # All source paths come from HEAD_PATHS above, never arbitrary manifest
        # entries, user input, environment variables, or an authentication token.
        url = REPOSITORY + "/resolve/" + REVISION + "/" + relative
        request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
        temporary = None
        try:
            with opener.open(request, timeout=30) as response:
                if response.status != 200:
                    raise RuntimeError("Unexpected HTTP status for " + relative)
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != size:
                    raise RuntimeError("Content-Length does not match the pin: " + relative)
                with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".judge-", suffix=".part", delete=False) as out:
                    temporary = Path(out.name)
                    received = 0
                    hasher = hashlib.sha256()
                    while True:
                        chunk = response.read(min(256 * 1024, size - received + 1))
                        if not chunk:
                            break
                        received += len(chunk)
                        if received > size:
                            raise RuntimeError("Dependency exceeds the pinned size: " + relative)
                        hasher.update(chunk)
                        out.write(chunk)
                    if received != size or hasher.hexdigest() != digest:
                        raise RuntimeError("Dependency failed size/SHA-256 verification: " + relative)
                    out.flush()
                    os.fsync(out.fileno())
            os.replace(temporary, destination)
            temporary = None
            print("Fetched and verified: " + relative)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print("PASS: all four pinned public judge heads are present and verified.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("Judge-head acquisition failed: " + str(error), file=sys.stderr)
        sys.exit(1)
