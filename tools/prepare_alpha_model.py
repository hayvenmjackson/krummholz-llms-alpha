"""Reproduce the alpha's quantized model and run actual bounded CPU checks in CI."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from krummholz_llm.runner import run_request, verify_runtime  # noqa: E402

ENGINE_COMMIT = "620d2225cc436ab0b001e3200e590e3b286ef9ab"
RUNTIME_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b11669/llama-b11669-bin-ubuntu-x64.tar.gz"
RUNTIME_SHA = "cf07c1c112d709bd67e522017471de57ed0ec8bea031d732c5422351c07f65ac"
RUNTIME_BYTES = 17851611
NAME = "Pleias-RAG-350M-Q4_K_M.gguf"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, size: int, digest: str) -> None:
    partial = target.with_suffix(target.suffix + ".partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Krummholz-alpha-build/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
        count = 0
        while chunk := response.read(1024 * 1024):
            count += len(chunk)
            if count > size:
                raise RuntimeError("download exceeds pinned size")
            output.write(chunk)
    if count != size or sha(partial) != digest:
        raise RuntimeError("download differs from pinned size or SHA-256")
    partial.replace(target)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work, output = args.work.resolve(), args.output.resolve()
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads((ROOT / "config/models.json").read_text(encoding="utf-8"))["artifacts"][0]
    source_file = work / source["filename"]
    archive = work / "llama-b11669-linux-x64.tar.gz"
    download(source["download_url"], source_file, source["bytes"], source["sha256"])
    download(RUNTIME_URL, archive, RUNTIME_BYTES, RUNTIME_SHA)
    runtime = work / "runtime"
    runtime.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(runtime, filter="data")
    quantizers = list(runtime.rglob("llama-quantize"))
    completions = list(runtime.rglob("llama-completion"))
    if len(quantizers) != 1 or len(completions) != 1:
        raise RuntimeError("unexpected runtime archive layout")
    quantizer, engine = quantizers[0], completions[0]
    os.environ["LD_LIBRARY_PATH"] = str(engine.parent)
    verify_runtime(engine)
    partial = output / (NAME + ".partial")
    command = [str(quantizer), "--max-buffer-size", "16", str(source_file), str(partial), "Q4_K_M", "2"]
    subprocess.run(command, check=True, timeout=600)
    with partial.open("rb") as model:
        if model.read(4) != b"GGUF":
            raise RuntimeError("quantizer did not produce GGUF")
    if not 100_000_000 < partial.stat().st_size < source["bytes"]:
        raise RuntimeError("unexpected quantized artifact size")
    model_file = output / NAME
    partial.replace(model_file)
    manifest = {
        "name": "Pleias RAG 350M · Q4_K_M",
        "filename": NAME,
        "bytes": model_file.stat().st_size,
        "sha256": sha(model_file),
        "download_url": "https://github.com/hayvenmjackson/krummholz-llms-alpha/releases/download/v0.1.0-alpha.1/" + NAME,
        "license": "Apache-2.0",
        "upstream": source["repository"],
        "upstream_revision": source["revision"],
        "upstream_sha256": source["sha256"],
        "quantization": "Q4_K_M",
        "quantizer_tag": "b11669",
        "quantizer_commit": ENGINE_COMMIT,
        "quantizer_archive_sha256": RUNTIME_SHA,
        "quantizer_options": ["--max-buffer-size", "16", "Q4_K_M", "2"],
        "krummholz_training_completed": False,
    }
    write_json(output / "model-manifest.json", manifest)
    # These authored examples establish execution and framing, not model quality.
    results = {"scope": "Three fictional software smoke cases; not a representative model evaluation.",
               "source_sha256": source["sha256"], "quantized_sha256": manifest["sha256"], "cases": []}
    expectations = {"acadia": "30", "katahdin": "90", "atlantic": "paper lantern"}
    for label, candidate in (("upstream_bf16", source_file), ("q4_k_m", model_file)):
        for profile, expected in expectations.items():
            data = json.loads((ROOT / "data" / profile / "example.json").read_text(encoding="utf-8"))
            result = run_request(data, profile, engine, candidate, context=4096,
                                 output_tokens=1024, timeout=180, threads=2, batch_size=64)
            result.update({"profile": profile, "model_variant": label,
                           "expected_fixture_text_present": expected in result.get("answer", "").lower()})
            results["cases"].append(result)
            print(json.dumps({k: result.get(k) for k in ("profile", "model_variant", "status", "elapsed_seconds", "expected_fixture_text_present")}))
    results["all_fictional_smokes_passed"] = all(
        r["status"] == "needs_human_review" and r["expected_fixture_text_present"] and r.get("citations")
        for r in results["cases"])
    write_json(output / "cpu-smoke-results.json", results)
    (output / "SHA256SUMS").write_text(f"{manifest['sha256']}  {NAME}\n", encoding="ascii")
    return 0 if results["all_fictional_smokes_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
