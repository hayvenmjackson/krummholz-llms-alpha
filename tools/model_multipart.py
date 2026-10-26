"""Offline byte-preserving model packaging. No download, inference or release approval.

Split only into a new directory. Reassemble only with an independently trusted
manifest hash, then publish a verified whole file atomically without overwriting.
Python 3.12+, standard library only. Payload reads are bounded to 8 MiB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile

BUFFER = 8 * 1024 * 1024
DEFAULT_PART_BYTES = 1536 * 1024 * 1024
MAX_PART_BYTES = 2 * 1024**3 - 1  # GitHub requires strictly below 2 GiB.
MAX_WHOLE_BYTES = 1024**4
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_PARTS = 999
MANIFEST_NAME = "multipart-manifest.json"
HEX = re.compile(r"[0-9a-f]{64}\Z")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def digest(value):
    if not isinstance(value, str) or not HEX.fullmatch(value):
        raise ValueError("Expected lowercase SHA256")
    return value


def filename(value):
    if (not isinstance(value, str) or not NAME.fullmatch(value) or value.endswith(".")
            or value.split(".")[0].upper() in RESERVED):
        raise ValueError("Unsafe portable filename")
    return value


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("Invalid bounded integer")
    return value


def plain_path(value):
    """Do not follow symlinks/junctions, including an existing parent directory."""
    path = Path(os.path.abspath(value))
    for component in (path, *path.parents):
        if component.is_symlink() or (hasattr(component, "is_junction") and component.is_junction()):
            raise ValueError("Symlink/junction paths are not allowed")
    return path


def open_regular(path):
    path = plain_path(path)
    flags = (os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_NONBLOCK", 0))  # Reject FIFOs via fstat without blocking on open.
    fd = os.open(path, flags)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("Expected a regular file")
    return os.fdopen(fd, "rb")


def unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def serialized(value):
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(data) > MAX_MANIFEST_BYTES:
        raise ValueError("Manifest exceeds size limit")
    return data


def part_name(name, index, count):
    return f"{name}.part-{index:05d}-of-{count:05d}"


def validate_manifest(manifest):
    if (not isinstance(manifest, dict) or set(manifest) != {
            "schema_version", "format", "file", "part_bytes", "parts", "provenance",
            "release_authorized", "quality_accepted"}
            or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
            or manifest["format"] != "raw-concatenation-v1"
            or manifest["release_authorized"] is not False or manifest["quality_accepted"] is not False
            or not isinstance(manifest["provenance"], dict)):
        raise ValueError("Unknown manifest schema or unsupported approval claim")
    whole = manifest["file"]
    if not isinstance(whole, dict) or set(whole) != {"name", "bytes", "sha256"}:
        raise ValueError("Invalid whole-file pin")
    filename(whole["name"])
    integer(whole["bytes"], 1, MAX_WHOLE_BYTES)
    digest(whole["sha256"])
    part_bytes = integer(manifest["part_bytes"], 1, MAX_PART_BYTES)
    count = (whole["bytes"] + part_bytes - 1) // part_bytes
    if count > MAX_PARTS or not isinstance(manifest["parts"], list) or len(manifest["parts"]) != count:
        raise ValueError("Invalid number of ordered parts")
    for index, part in enumerate(manifest["parts"], 1):
        if not isinstance(part, dict) or set(part) != {"index", "name", "offset", "bytes", "sha256"}:
            raise ValueError("Invalid part schema")
        offset = (index - 1) * part_bytes
        size = min(part_bytes, whole["bytes"] - offset)
        if (type(part["index"]) is not int or part["index"] != index
                or part["name"] != part_name(whole["name"], index, count)
                or type(part["offset"]) is not int or part["offset"] != offset
                or type(part["bytes"]) is not int or part["bytes"] != size):
            raise ValueError("Part order, filename, offset or size changed")
        digest(part["sha256"])
    return manifest


def load_manifest(path, expected_sha256):
    digest(expected_sha256)
    with open_regular(path) as source:
        if os.fstat(source.fileno()).st_size > MAX_MANIFEST_BYTES:
            raise ValueError("Oversized manifest")
        data = source.read(MAX_MANIFEST_BYTES + 1)
    if len(data) > MAX_MANIFEST_BYTES or hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Manifest SHA256 mismatch")
    return validate_manifest(json.loads(data.decode("utf-8"), object_pairs_hook=unique_json,
                                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON"))))


def require_space(parent, count):
    if shutil.disk_usage(parent).free < count + 16 * 1024 * 1024:
        raise OSError("Insufficient free disk space for the complete output plus safety margin")


def write_exclusive(path, data):
    with path.open("xb") as target:
        target.write(data)
        target.flush()
        os.fsync(target.fileno())


def split(source_path, output_dir, expected_bytes, expected_sha256,
          part_bytes=DEFAULT_PART_BYTES, provenance=None):
    integer(expected_bytes, 1, MAX_WHOLE_BYTES)
    digest(expected_sha256)
    integer(part_bytes, 1, MAX_PART_BYTES)
    count = (expected_bytes + part_bytes - 1) // part_bytes
    if count > MAX_PARTS:
        raise ValueError("Too many parts")
    source_path, output_dir = plain_path(source_path), plain_path(output_dir)
    filename(source_path.name)
    if output_dir.exists() or not output_dir.parent.is_dir():
        raise FileExistsError("Package directory must be new, with an existing parent")
    if provenance is None:
        provenance = {}
    if not isinstance(provenance, dict):
        raise ValueError("Provenance must be a JSON object")
    serialized(provenance)
    require_space(output_dir.parent, expected_bytes)
    with open_regular(source_path) as source:
        if os.fstat(source.fileno()).st_size != expected_bytes:
            raise ValueError("Source size differs from the trusted pin")
        output_dir.mkdir()  # Exclusive ownership; never remove pre-existing files.
        whole_hash, parts = hashlib.sha256(), []
        for index in range(1, count + 1):
            size = min(part_bytes, expected_bytes - (index - 1) * part_bytes)
            name = part_name(source_path.name, index, count)
            pending = output_dir / (name + ".partial")
            part_hash, copied = hashlib.sha256(), 0
            with pending.open("xb") as target:
                while copied < size:
                    block = source.read(min(BUFFER, size - copied))
                    if not block:
                        raise ValueError("Source truncated during packaging")
                    target.write(block)
                    part_hash.update(block)
                    whole_hash.update(block)
                    copied += len(block)
                target.flush()
                os.fsync(target.fileno())
            # A hard link atomically exposes only a fully written part and never overwrites.
            os.link(pending, output_dir / name)
            pending.unlink()  # Only this invocation's exclusively created file.
            parts.append({"index": index, "name": name, "offset": (index - 1) * part_bytes,
                          "bytes": size, "sha256": part_hash.hexdigest()})
        if source.read(1) or whole_hash.hexdigest() != expected_sha256:
            raise ValueError("Whole-source SHA256/size mismatch; package has no valid manifest")
    manifest = {"schema_version": 1, "format": "raw-concatenation-v1",
                "file": {"name": source_path.name, "bytes": expected_bytes, "sha256": expected_sha256},
                "part_bytes": part_bytes, "parts": parts, "provenance": provenance,
                "release_authorized": False, "quality_accepted": False}
    data = serialized(validate_manifest(manifest))
    pending = output_dir / (MANIFEST_NAME + ".partial")
    write_exclusive(pending, data)
    os.link(pending, output_dir / MANIFEST_NAME)
    pending.unlink()
    return {"manifest": str(output_dir / MANIFEST_NAME), "manifest_sha256": hashlib.sha256(data).hexdigest(),
            "parts": count, "bytes": expected_bytes, "release_authorized": False}


def copy_verified_parts(manifest, directory, target=None):
    whole = hashlib.sha256()
    for part in manifest["parts"]:
        with open_regular(directory / part["name"]) as source:
            if os.fstat(source.fileno()).st_size != part["bytes"]:
                raise ValueError("Part size mismatch: " + part["name"])
            copied, part_hash = 0, hashlib.sha256()
            while copied < part["bytes"]:
                block = source.read(min(BUFFER, part["bytes"] - copied))
                if not block:
                    raise ValueError("Part truncated during reading")
                part_hash.update(block)
                whole.update(block)
                if target is not None:
                    target.write(block)
                copied += len(block)
            if source.read(1) or part_hash.hexdigest() != part["sha256"]:
                raise ValueError("Part SHA256/size mismatch: " + part["name"])
    if whole.hexdigest() != manifest["file"]["sha256"]:
        raise ValueError("Whole-file SHA256 mismatch")


def verify(manifest_path, expected_manifest_sha256):
    manifest_path = plain_path(manifest_path)
    manifest = load_manifest(manifest_path, expected_manifest_sha256)
    copy_verified_parts(manifest, manifest_path.parent)
    return {"verified": True, "file": manifest["file"], "parts": len(manifest["parts"]),
            "release_authorized": False}


def assemble(manifest_path, output_path, expected_manifest_sha256):
    manifest_path, output_path = plain_path(manifest_path), plain_path(output_path)
    manifest = load_manifest(manifest_path, expected_manifest_sha256)
    if output_path.exists() or not output_path.parent.is_dir():
        raise FileExistsError("Whole-file output must be absent, with an existing parent")
    if output_path.name != manifest["file"]["name"]:
        raise ValueError("Output filename must match the trusted manifest")
    require_space(output_path.parent, manifest["file"]["bytes"])
    fd, temporary = tempfile.mkstemp(prefix=".krummholz-reassembly-", suffix=".partial", dir=output_path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as target:
            copy_verified_parts(manifest, manifest_path.parent, target)
            target.flush()
            os.fsync(target.fileno())
        # Same-directory hard-link creation is atomic and fails if another file appeared.
        # Unsupported filesystems fail closed; there is no overwrite/copy fallback.
        os.link(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)  # Exactly the uniquely created owned temp, never inputs.
    return {"verified": True, "output": str(output_path), "file": manifest["file"],
            "release_authorized": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("split")
    pack.add_argument("--source", type=Path, required=True)
    pack.add_argument("--output-dir", type=Path, required=True)
    pack.add_argument("--expected-bytes", type=int, required=True)
    pack.add_argument("--expected-sha256", required=True)
    pack.add_argument("--part-bytes", type=int, default=DEFAULT_PART_BYTES)
    pack.add_argument("--provenance", type=Path, help="Small JSON metadata file; this cannot authorize release")
    for command in ("verify", "assemble"):
        sub = commands.add_parser(command)
        sub.add_argument("--manifest", type=Path, required=True)
        sub.add_argument("--manifest-sha256", required=True, help="Trusted digest obtained separately from the downloaded bundle")
        if command == "assemble":
            sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "split":
            metadata = None
            if args.provenance:
                with open_regular(args.provenance) as source:
                    data = source.read(MAX_MANIFEST_BYTES + 1)
                if len(data) > MAX_MANIFEST_BYTES:
                    raise ValueError("Provenance too large")
                metadata = json.loads(data.decode("utf-8"), object_pairs_hook=unique_json)
            result = split(args.source, args.output_dir, args.expected_bytes, args.expected_sha256,
                           args.part_bytes, metadata)
        elif args.command == "verify":
            result = verify(args.manifest, args.manifest_sha256)
        else:
            result = assemble(args.manifest, args.output, args.manifest_sha256)
    except (OSError, ValueError, TypeError, RecursionError) as error:
        parser.exit(1, f"Packaging refused: {error}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
