"""Prepare pinned 1B precision candidates and retain fictional evaluation evidence.

An Actions artifact is not an approved model or a release. A failed parser,
tokenizer parity check, or existing fixture gate makes this command fail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from krummholz_llm.protocol import build_prompt, validate_request
from krummholz_llm.runner import runtime_environment, verify_runtime
from prepare_alpha_model import (
    ENGINE_COMMIT, RUNTIME_BYTES, RUNTIME_SHA, RUNTIME_URL, download, sha, write_json,
)
from diagnose_pleias_framing import (
    EOS, EXPECTATIONS, TOKENIZER_BYTES, TOKENIZER_SHA, completion, encoded_string,
    exact_read, gguf_string, layout, number, patch_pre, token_ids,
)

SOURCE_ID = "pleias-rag-1b-upstream-bf16-gguf"  # Legacy manifest identity; actual weights are 16-bit.
SOURCE_REVISION = "443722afa1237b00a2e18d9eb2f1f1dd9dc37ad6"
SOURCE_SHA = "c6e7da82daf0e45b4946de035e6dcc7e0eebdd11433cbb66cd8a4567bfcb67e7"
SOURCE_BYTES = 2_393_458_272
PUBLISHER_REVISION = "b8c3f1ab2c13e1d44305eee46662310cdbfd72a1"
PRECISIONS = ("F16", "Q8_0", "Q4_K_M")


def framed_prompt(text: str) -> tuple[str, str]:
    """HF adds EOS itself; llama receives exactly one literal leading EOS.

    Accept the protocol helper both before and after its one-EOS correction,
    but never silently repair duplicate/internal EOS or a missing final LF.
    """
    raw = text[len(EOS):] if text.startswith(EOS) else text
    if EOS in raw or not raw.startswith("<|query_start|>") or not raw.endswith("<|language_start|>\n"):
        raise ValueError("unexpected publisher prompt framing; require one prefix EOS and final LF")
    return raw, EOS + raw


def tokenizer_metadata(path: Path) -> dict:
    with path.open("rb") as source:
        values = layout(source)["values"]
    if values.get("tokenizer.ggml.model") != "gpt2" or values.get("tokenizer.ggml.eos_token_id") != 2:
        raise ValueError("unexpected pinned publisher tokenizer identity")
    for key in ("tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token"):
        if values.get(key, False) is not False:
            raise ValueError("automatic special-token insertion must be disabled")
    pre = values.get("tokenizer.ggml.pre")
    if pre is not None and pre != "gpt-2":
        raise ValueError("existing pre-tokenizer is not the verified gpt-2 value")
    return {key: values.get(key) for key in (
        "general.file_type", "tokenizer.ggml.model", "tokenizer.ggml.pre",
        "tokenizer.ggml.eos_token_id", "tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token",
    )}


def tensor_type_counts(path: Path) -> dict:
    # Numeric IDs are from ggml/include/ggml.h at the pinned ENGINE_COMMIT.
    names = {0: "F32", 1: "F16", 8: "Q8_0", 12: "Q4_K", 14: "Q6_K", 30: "BF16"}
    counts = {}
    with path.open("rb") as source:
        metadata = layout(source)
        source.seek(metadata["kv_end"])
        for _ in range(metadata["tensors"]):
            gguf_string(source)
            dimensions = number(source, "I")
            if dimensions > 4:
                raise ValueError("unexpected GGUF tensor rank")
            exact_read(source, 8 * dimensions)
            kind = number(source, "I")
            number(source, "Q")
            label = names.get(kind, f"GGML_TYPE_{kind}")
            counts[label] = counts.get(label, 0) + 1
    return counts


def corrected_copy(source: Path, target: Path) -> dict:
    original = tokenizer_metadata(source)
    if target.exists() or source.resolve() == target.resolve():
        raise ValueError("corrected model requires a new output path")
    if original["tokenizer.ggml.pre"] is None:
        report = patch_pre(source, target)
    else:
        shutil.copyfile(source, target)
        source_hash, target_hash = sha(source), sha(target)
        if source_hash != target_hash:
            raise ValueError("already-correct model copy changed bytes")
        report = {"source_sha256": source_hash, "derived_sha256": target_hash,
                  "tensor_payload_bytes_unchanged": True, "metadata_already_correct": True,
                  "only_metadata_change": {}}
    corrected = tokenizer_metadata(target)
    if corrected["tokenizer.ggml.pre"] != "gpt-2":
        raise ValueError("corrected metadata check failed")
    report.update({"original_metadata": original, "corrected_metadata": corrected})
    return report


def pinned_source() -> dict:
    source = json.loads((ROOT / "config/models.json").read_text(encoding="utf-8"))["artifacts"][1]
    expected = {"id": SOURCE_ID, "revision": SOURCE_REVISION, "sha256": SOURCE_SHA,
                "bytes": SOURCE_BYTES, "filename": "Pleias-RAG-1B.gguf",
                "repository": "PleIAs/Pleias-RAG-1B-gguf",
                "source_repository": "PleIAs/Pleias-RAG-1B", "source_revision": PUBLISHER_REVISION,
                "publisher_license": "Apache-2.0",
                "download_url": f"https://huggingface.co/PleIAs/Pleias-RAG-1B-gguf/resolve/{SOURCE_REVISION}/Pleias-RAG-1B.gguf"}
    if any(source.get(key) != value for key, value in expected.items()):
        raise ValueError("config/models.json artifact 1 differs from the reviewed 1B source pin")
    return source


def runtime(work: Path, output: Path) -> tuple[Path, Path, Path]:
    archive = work / "llama-b11669-linux-x64.tar.gz"
    download(RUNTIME_URL, archive, RUNTIME_BYTES, RUNTIME_SHA)
    destination = work / "runtime"
    destination.mkdir()
    with tarfile.open(archive) as incoming:
        incoming.extractall(destination, filter="data")
    executables = []
    for name in ("llama-completion", "llama-tokenize", "llama-quantize"):
        matches = list(destination.rglob(name))
        if len(matches) != 1 or not matches[0].is_file():
            raise ValueError("unexpected pinned runtime archive layout")
        executables.append(matches[0])
    engine, tokenize, quantizer = executables
    os.environ["LD_LIBRARY_PATH"] = str(engine.parent)
    verify_runtime(engine)
    write_json(output / "runtime-provenance.json", {
        "tag": "b11669", "commit": ENGINE_COMMIT, "archive_url": RUNTIME_URL,
        "archive_bytes": RUNTIME_BYTES, "archive_sha256": RUNTIME_SHA,
        "executables": {path.name: {"bytes": path.stat().st_size, "sha256": sha(path)} for path in executables},
    })
    return engine, tokenize, quantizer


def fixture_cases() -> dict:
    cases = {}
    for profile in EXPECTATIONS:
        fixture = ROOT / "data" / profile / "example.json"
        data = json.loads(fixture.read_text(encoding="utf-8"))
        request = validate_request(data, profile)
        raw, framed = framed_prompt(build_prompt(request, 4096, 1024).text)
        cases[profile] = {"request": request, "raw": raw, "framed": framed,
                          "fixture_sha256": sha(fixture), "fixture": data}
    return cases


def existing_gate(cases: list[dict]) -> bool:
    return (len(cases) == len(EXPECTATIONS)
            and {row["profile"] for row in cases} == set(EXPECTATIONS)
            and all(row.get("existing_smoke_gate_passed") is True for row in cases))


def prepare(model: str, precision: str, work: Path, output: Path, timeout: int) -> int:
    import tokenizers
    if model != "1b" or precision not in PRECISIONS:
        raise ValueError("only the reviewed 1B precision candidates are supported")
    if tokenizers.__version__ != "0.22.2":
        raise ValueError("install exactly tokenizers==0.22.2")
    source = pinned_source()
    filename = f"Pleias-RAG-1B-{precision}.gguf"
    report = {"scope": "Three fictional software smoke cases; not a representative quality evaluation.",
              "model": model, "precision": precision, "approved": False,
              "release_authorized_by_this_run": False, "status": "preparing",
              "source": source, "parity": [], "cases": []}
    report_path = output / "candidate-results.json"
    write_json(report_path, report)
    source_file = work / source["filename"]
    download(source["download_url"], source_file, source["bytes"], source["sha256"])
    source_types = tensor_type_counts(source_file)
    if not source_types.get("F16") or set(source_types) - {"F16", "F32"}:
        raise ValueError("pinned original weights are not the expected F16/F32 tensor mix; inspect before labeling")
    write_json(output / "upstream-tensor-types.json", source_types)
    engine, tokenize, quantizer = runtime(work, output)
    model_file = output / filename
    corrected = model_file if precision == "F16" else work / "Pleias-RAG-1B-corrected-16bit.gguf"
    patch = corrected_copy(source_file, corrected)
    write_json(output / "metadata-correction.json", patch)
    if precision != "F16":
        partial = model_file.with_suffix(model_file.suffix + ".partial")
        command = [str(quantizer), "--max-buffer-size", "16", str(corrected), str(partial), precision, "2"]
        with (output / "quantizer.stdout.txt").open("w", encoding="utf-8") as stdout, \
                (output / "quantizer.stderr.txt").open("w", encoding="utf-8") as stderr:
            subprocess.run(command, stdout=stdout, stderr=stderr, check=True, timeout=600,
                           env=runtime_environment())
        if not 100_000_000 < partial.stat().st_size < corrected.stat().st_size:
            raise ValueError("unexpected quantized candidate size")
        metadata = tokenizer_metadata(partial)
        if metadata["tokenizer.ggml.pre"] != "gpt-2":
            raise ValueError("quantizer did not retain the verified tokenizer metadata")
        partial.replace(model_file)
    manifest = {
        "approved": False, "name": "Pleias RAG 1B · " + ("original 16-bit" if precision == "F16" else precision),
        "filename": filename, "bytes": model_file.stat().st_size, "sha256": sha(model_file),
        "download_url": "", "license": source["publisher_license"], "upstream": source["repository"],
        "upstream_revision": source["revision"], "upstream_bytes": source["bytes"],
        "upstream_sha256": source["sha256"], "publisher_repository": source["source_repository"],
        "publisher_revision": source["source_revision"], "precision_candidate": precision,
        "quantization_performed": precision != "F16", "quantization": None if precision == "F16" else precision,
        "quantizer_tag": None if precision == "F16" else "b11669",
        "quantizer_commit": None if precision == "F16" else ENGINE_COMMIT,
        "quantizer_archive_sha256": None if precision == "F16" else RUNTIME_SHA,
        "quantizer_options": [] if precision == "F16" else ["--max-buffer-size", "16", precision, "2"],
        "metadata_correction": patch, "tokenizer_metadata": tokenizer_metadata(model_file),
        "upstream_tensor_type_counts": source_types, "candidate_tensor_type_counts": tensor_type_counts(model_file),
        "runtime_prompt": "Exactly one literal prefix EOS; publisher final LF preserved via --binary-file.",
        "krummholz_training_completed": False,
        "note": "Evaluation candidate only. No release or Android provisioning is authorized by this manifest.",
    }
    write_json(output / "model-manifest.json", manifest)
    (output / "SHA256SUMS").write_text(f"{manifest['sha256']}  {filename}\n", encoding="ascii")
    tokenizer_file = output / "publisher-tokenizer.json"
    tokenizer_url = f"https://huggingface.co/PleIAs/Pleias-RAG-1B/resolve/{PUBLISHER_REVISION}/tokenizer.json"
    download(tokenizer_url, tokenizer_file, TOKENIZER_BYTES, TOKENIZER_SHA)
    publisher = tokenizers.Tokenizer.from_file(str(tokenizer_file))
    report.update({"model_sha256": manifest["sha256"], "tokenizers_version": tokenizers.__version__,
                   "tokenizer_url": tokenizer_url, "tokenizer_bytes": TOKENIZER_BYTES,
                   "tokenizer_sha256": TOKENIZER_SHA, "status": "checking_tokenizer"})
    cases = fixture_cases()
    probes = {profile: case["raw"] for profile, case in cases.items()}
    probes["unicode_digits"] = "<|query_start|>What's 123456789? café, naïve, 東京 🌲<|query_end|>\n<|language_start|>\n"
    variants = [("candidate", model_file)]
    if precision != "F16":
        variants.insert(0, ("corrected_upstream_16bit", corrected))
    for label, candidate in variants:
        for probe, raw in probes.items():
            raw, prompt = framed_prompt(raw)
            expected = publisher.encode(raw, add_special_tokens=True).ids
            if not expected or expected[0] != 2 or expected.count(2) != 1:
                raise ValueError("publisher tokenizer did not add exactly one expected EOS")
            actual = token_ids(tokenize, candidate, prompt)
            report["parity"].append({"model_variant": label, "probe": probe,
                "matches_publisher": actual == expected, "publisher_ids": expected, "actual_ids": actual,
                "runtime_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()})
            write_json(report_path, report)
    report["tokenizer_parity_passed"] = (len(report["parity"]) == len(variants) * len(probes)
                                         and all(row["matches_publisher"] for row in report["parity"]))
    report["status"] = "running_fictional_cases" if report["tokenizer_parity_passed"] else "tokenizer_parity_failed"
    write_json(report_path, report)
    if not report["tokenizer_parity_passed"]:
        return 2
    for profile, case in cases.items():
        write_json(output / f"{profile}.fixture.json", case["fixture"])
        result = completion(engine, model_file, case["framed"], case["request"],
                            EXPECTATIONS[profile], output, profile, timeout)
        result.update({"profile": profile, "fixture_sha256": case["fixture_sha256"],
                       "raw_stdout_retained": (output / f"{profile}.stdout.txt").is_file(),
                       "raw_stderr_retained": (output / f"{profile}.stderr.txt").is_file()})
        report["cases"].append(result)
        write_json(output / f"{profile}.answer.json", result)
        write_json(report_path, report)
        print(json.dumps({key: result.get(key) for key in
                         ("profile", "status", "existing_smoke_gate_passed", "elapsed_seconds")}))
    report["all_fictional_smokes_passed"] = existing_gate(report["cases"])
    report["status"] = "smoke_gates_passed_pending_review" if report["all_fictional_smokes_passed"] else "fixture_gate_failed"
    write_json(report_path, report)
    return 0 if report["all_fictional_smokes_passed"] else 2


def self_test() -> None:
    raw = "<|query_start|>Fictional café 🌲?<|query_end|>\n<|language_start|>\n"
    assert framed_prompt(raw) == framed_prompt(EOS + raw) == (raw, EOS + raw)
    for bad in (EOS + EOS + raw, raw[:-1], raw.replace("café", EOS)):
        try:
            framed_prompt(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("bad framing was accepted")
    passed = [{"profile": profile, "existing_smoke_gate_passed": True} for profile in EXPECTATIONS]
    assert existing_gate(passed) and not existing_gate([]) and not existing_gate(passed[:-1])
    assert not existing_gate([dict(row, existing_smoke_gate_passed=False) for row in passed])
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        payload = bytes(range(32))
        for label, pre, bos in (("missing", None, False), ("correct", "gpt-2", False),
                                ("wrong", "llama3", False), ("auto_bos", "gpt-2", True)):
            items = [("tokenizer.ggml.model", 8, encoded_string("gpt2")),
                     ("tokenizer.ggml.eos_token_id", 4, struct.pack("<I", 2)),
                     ("tokenizer.ggml.add_bos_token", 7, struct.pack("<?", bos))]
            if pre is not None:
                items.append(("tokenizer.ggml.pre", 8, encoded_string(pre)))
            header = b"GGUF" + struct.pack("<IQQ", 3, 1, len(items))
            for key, kind, value in items:
                header += encoded_string(key) + struct.pack("<I", kind) + value
            header += encoded_string("fictional.weight") + struct.pack("<IQIQ", 1, 8, 0, 0)
            source, target = folder / f"{label}.gguf", folder / f"{label}-copy.gguf"
            source.write_bytes(header + b"\0" * (-len(header) % 32) + payload)
            if label in ("wrong", "auto_bos"):
                try:
                    corrected_copy(source, target)
                except ValueError:
                    pass
                else:
                    raise AssertionError("unverified tokenizer metadata was accepted")
                continue
            correction = corrected_copy(source, target)
            assert correction["tensor_payload_bytes_unchanged"] is True
            assert tensor_type_counts(target) == {"F32": 1}
            with target.open("rb") as handle:
                metadata = layout(handle)
                assert metadata["values"]["tokenizer.ggml.pre"] == "gpt-2"
                handle.seek(metadata["data_start"])
                assert handle.read() == payload
    pinned_source()
    print("candidate framing, metadata, payload preservation, source pin and gate checks passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("1b",))
    parser.add_argument("--quantization", choices=PRECISIONS)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not all((args.model, args.quantization, args.work, args.output)) or not 1 <= args.timeout <= 180:
        parser.error("model, quantization, work and output required; timeout must be 1-180 seconds")
    work, output = args.work.resolve(), args.output.resolve()
    if work == output or work in output.parents or output in work.parents:
        parser.error("work and output must be separate, non-nested directories")
    if any(path.exists() and any(path.iterdir()) for path in (work, output)):
        parser.error("work and output must be empty; retained evidence will never be overwritten")
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    try:
        return prepare(args.model, args.quantization, work, output, args.timeout)
    except Exception as error:
        write_json(output / "preparation-error.json", {"status": "preparation_failed", "approved": False,
                   "error_type": type(error).__name__, "reason": str(error)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
