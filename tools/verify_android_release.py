"""Check release identity, offline permissions, model pin, signing and ELF alignment."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import zipfile

SIGNER = "a597e224548e8aef87a825ccda56bac2036ab183aca1d26e486661ae92542ccb"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--apk", type=Path, required=True)
    p.add_argument("--build-tools", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    def run(tool: str, *values: str) -> str:
        return subprocess.check_output([str(args.build_tools / tool), *values], text=True, stderr=subprocess.STDOUT)
    signing = run("apksigner", "verify", "--verbose", "--print-certs", str(args.apk))
    certs = re.findall(r"certificate SHA-256 digest: ([0-9a-fA-F]{64})", signing)
    if certs != [SIGNER] or "Number of signers: 1" not in signing:
        raise RuntimeError("APK does not have the pinned single alpha signing identity")
    badging = run("aapt", "dump", "badging", str(args.apk))
    if not all(value in badging for value in ("name='ai.krummholz.llm'", "versionCode='1'", "versionName='0.1.0-alpha.1'")):
        raise RuntimeError("wrong alpha package or version")
    permissions = run("aapt", "dump", "permissions", str(args.apk))
    if "uses-permission:" in permissions:
        raise RuntimeError("offline alpha must not request Android permissions")
    manifest = run("aapt", "dump", "xmltree", str(args.apk), "AndroidManifest.xml")
    if re.search(r"android:debuggable[^\n]*0xffffffff", manifest):
        raise RuntimeError("public alpha is debuggable")
    if not re.search(r"android:allowBackup[^\n]*0x0(?:\s|$)", manifest):
        raise RuntimeError("automatic backups are not explicitly disabled")
    run("zipalign", "-c", "-P", "16", "4", str(args.apk))
    libraries = []
    with zipfile.ZipFile(args.apk) as archive:
        names = archive.namelist()
        if any(name.endswith(".gguf") for name in names):
            raise RuntimeError("model belongs in a separate download, not the public APK")
        approved = json.loads(archive.read("assets/model-manifest.json"))
        if not approved.get("approved") or not re.fullmatch(r"[0-9a-f]{64}", approved.get("sha256", "")):
            raise RuntimeError("APK lacks an approved exact model pin")
        for name in names:
            if not name.startswith("lib/") or not name.endswith(".so"):
                continue
            data = archive.read(name)
            if data[:6] != b"\x7fELF\x02\x01":
                raise RuntimeError("unexpected native library format")
            offset = struct.unpack_from("<Q", data, 32)[0]
            entry_size, count = struct.unpack_from("<HH", data, 54)
            alignments = []
            for i in range(count):
                entry = struct.unpack_from("<IIQQQQQQ", data, offset + i * entry_size)
                if entry[0] == 1:
                    alignments.append(entry[7])
                    if entry[7] < 16384:
                        raise RuntimeError(f"{name} lacks 16KB LOAD alignment")
            if not alignments:
                raise RuntimeError("no native LOAD segments")
            libraries.append({"path": name, "load_alignments": alignments,
                              "sha256": hashlib.sha256(data).hexdigest()})
        for abi in ("arm64-v8a", "x86_64"):
            if f"lib/{abi}/libkrummholz_llm.so" not in names:
                raise RuntimeError(f"missing inference engine for {abi}")
    digest = hashlib.sha256(args.apk.read_bytes()).hexdigest()
    report = {"apk": args.apk.name, "bytes": args.apk.stat().st_size, "sha256": digest,
              "signer_sha256": SIGNER, "package": "ai.krummholz.llm", "version": "0.1.0-alpha.1",
              "build": 1, "offline_no_permissions": True, "debuggable": False,
              "model_sha256": approved["sha256"], "zip_16kb_alignment": True, "native_libraries": libraries}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
