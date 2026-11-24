#!/usr/bin/env python3
"""Verify the unchanged pinned upstream bundle. No network or dependencies."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent / "vendor" / "krummholz-harness"
manifest = json.loads((root / "PROVENANCE.json").read_text())
if manifest["revision"] != "30f793a03d7ba5ca0f123f275171f04ac04aab4d":
    sys.exit("FAIL: unexpected upstream revision in PROVENANCE.json")
failures = []
for item in manifest["files"]:
    path = root / item["path"]
    if not path.is_file():
        failures.append(item["path"] + ": missing")
        continue
    content = path.read_bytes()
    if len(content) != item["size_bytes"]:
        failures.append(item["path"] + ": size differs")
    elif hashlib.sha256(content).hexdigest() != item["sha256"]:
        failures.append(item["path"] + ": digest differs")
if failures:
    print("FAIL: upstream bundle verification:\n" + "\n".join(failures), file=sys.stderr)
    print("The four binary judge heads are public download dependencies, not checked-in files. "
          "Run python3 fetch_heads.py from native/ to fetch or repair their pinned copies. "
          "Restore any other changed/missing file from source control.", file=sys.stderr)
    sys.exit(1)
print(f'PASS: {len(manifest["files"])} source/dependency files match the recorded upstream revision.')
