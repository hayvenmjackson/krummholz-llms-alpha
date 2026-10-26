"""Bounded, fictional-only tokenizer/framing A/B checks; never a release gate override.

Requires tokenizers==0.22.2 for the pinned publisher tokenizer. Reuses the verified
BF16 model, Q4 model, and b11669 runtime produced by prepare_alpha_model.py. It
downloads only the small, hash-pinned tokenizer JSON. No network inference occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from krummholz_llm.protocol import OutputError, build_prompt, parse_answer, validate_request
from krummholz_llm.runner import ANSI, runtime_environment, verify_runtime
from prepare_alpha_model import NAME, download, sha, write_json

TOKENIZER_URL = "https://huggingface.co/PleIAs/Pleias-RAG-350M/resolve/db001b29a33532583d14979c3c38ef04b6d59352/tokenizer.json"
TOKENIZER_SHA = "dd7acf61196c61741cf95cc0f0a2b920443e44094438d325ab6e65a048fe8be4"
TOKENIZER_BYTES = 2777269
EOS = "<|end_of_text|>"
VARIANTS = (("baseline", False, False), ("eos_only", False, True),
            ("pre_only", True, False), ("pre_and_eos", True, True))
EXPECTATIONS = {"acadia": "30", "katahdin": "90", "atlantic": "paper lantern"}


def publisher_raw_prompt(framed: str) -> str:
    """HF's post-processor supplies EOS; remove the production adapter's one copy.

    This also supports historical prompts before that production fix, so the
    four diagnostic variants remain controlled after build_prompt is corrected.
    """
    raw = framed[len(EOS):] if framed.startswith(EOS) else framed
    if not raw.startswith("<|query_start|>") or not raw.endswith("<|language_start|>\n"):
        raise ValueError("unexpected production framing; do not silently alter diagnostic input")
    return raw


def exact_read(handle, size: int) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise ValueError("truncated GGUF header")
    return value


def number(handle, fmt: str):
    return struct.unpack("<" + fmt, exact_read(handle, struct.calcsize("<" + fmt)))[0]


def gguf_string(handle) -> str:
    size = number(handle, "Q")
    if size > 16 * 1024 * 1024:
        raise ValueError("unexpected GGUF string size")
    return exact_read(handle, size).decode("utf-8")


def skip_value(handle, kind: int):
    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
              7: "?", 10: "Q", 11: "q", 12: "d"}
    if kind in scalar:
        return number(handle, scalar[kind])
    if kind == 8:
        return gguf_string(handle)
    if kind == 9:
        subtype, count = number(handle, "I"), number(handle, "Q")
        if count > 10_000_000 or subtype == 9:
            raise ValueError("unexpected GGUF array")
        if subtype in scalar:
            handle.seek(struct.calcsize("<" + scalar[subtype]) * count, 1)
        else:
            for _ in range(count):
                skip_value(handle, subtype)
        return None
    raise ValueError("unknown GGUF metadata type")


def layout(handle) -> dict:
    if exact_read(handle, 4) != b"GGUF" or number(handle, "I") != 3:
        raise ValueError("expected GGUF v3")
    tensors, count = number(handle, "Q"), number(handle, "Q")
    if count > 10000 or tensors > 100000:
        raise ValueError("unexpected GGUF counts")
    values = {}
    for _ in range(count):
        key = gguf_string(handle)
        if key in values:
            raise ValueError("duplicate GGUF key")
        values[key] = skip_value(handle, number(handle, "I"))
    kv_end = handle.tell()
    for _ in range(tensors):
        gguf_string(handle)
        dims = number(handle, "I")
        if dims > 4:
            raise ValueError("unexpected GGUF tensor rank")
        exact_read(handle, 8 * dims + 4 + 8)
    info_end = handle.tell()
    alignment = values.get("general.alignment", 32)
    if not isinstance(alignment, int) or not 1 <= alignment <= 4096 or alignment & (alignment - 1):
        raise ValueError("unexpected GGUF alignment")
    return {"tensors": tensors, "count": count, "values": values, "kv_end": kv_end,
            "info_end": info_end, "data_start": (info_end + alignment - 1) // alignment * alignment,
            "alignment": alignment}


def encoded_string(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def patch_pre(source: Path, target: Path) -> dict:
    """Add only the missing pre-tokenizer key, retaining every tensor payload byte.

    Tensor offsets are relative to the aligned data section, so copying tensor
    descriptors verbatim and recomputing header padding preserves their meaning.
    The official tokenize CLI ignores --override-kv; a patched *copy* also lets
    that exact runtime check input token IDs without loading tensor weights.
    """
    if source.resolve() == target.resolve() or target.exists():
        raise ValueError("metadata diagnostic requires a new output path")
    partial = target.with_suffix(target.suffix + ".partial")
    with source.open("rb") as incoming:
        meta = layout(incoming)
        values = meta["values"]
        if "tokenizer.ggml.pre" in values:
            raise ValueError("source already has pre metadata; inspect it before adapting this diagnostic")
        if values.get("tokenizer.ggml.add_bos_token", False) or values.get("tokenizer.ggml.add_eos_token", False):
            raise ValueError("unexpected automatic special-token setting")
        if values.get("tokenizer.ggml.model") != "gpt2" or values.get("tokenizer.ggml.eos_token_id") != 2:
            raise ValueError("unexpected pinned tokenizer identity")
        incoming.seek(24)
        metadata = exact_read(incoming, meta["kv_end"] - 24)
        descriptors = exact_read(incoming, meta["info_end"] - meta["kv_end"])
        addition = encoded_string("tokenizer.ggml.pre") + struct.pack("<I", 8) + encoded_string("gpt-2")
        payload = hashlib.sha256()
        with partial.open("xb") as outgoing:
            outgoing.write(b"GGUF" + struct.pack("<IQQ", 3, meta["tensors"], meta["count"] + 1))
            outgoing.write(metadata + addition + descriptors)
            outgoing.write(b"\0" * (-outgoing.tell() % meta["alignment"]))
            incoming.seek(meta["data_start"])
            for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                payload.update(chunk)
                outgoing.write(chunk)
    with partial.open("rb") as check:
        derived = layout(check)
        if derived["values"]["tokenizer.ggml.pre"] != "gpt-2":
            raise ValueError("metadata patch verification failed")
        check.seek(derived["data_start"])
        copied = hashlib.file_digest(check, "sha256").hexdigest()
    if copied != payload.hexdigest():
        raise ValueError("tensor payload changed during metadata copy")
    partial.replace(target)
    return {"source_sha256": sha(source), "derived_sha256": sha(target),
            "tensor_payload_sha256": copied, "tensor_payload_bytes_unchanged": True,
            "only_metadata_change": {"tokenizer.ggml.pre": "gpt-2"}}


def token_ids(executable: Path, model: Path, prompt: str) -> list[int]:
    result = subprocess.run([str(executable), "-m", str(model), "--stdin", "--ids", "--no-bos", "--no-escape"],
                            input=prompt, capture_output=True, text=True, encoding="utf-8",
                            timeout=60, check=True, env=runtime_environment())
    candidates = re.findall(r"^\s*\[(?:\s*\d+\s*,?)*\]\s*$", result.stdout, re.MULTILINE)
    if len(candidates) != 1:
        raise ValueError("tokenizer did not return exactly one ID list")
    ids = json.loads(candidates[0])
    if not ids or any(type(value) is not int for value in ids):
        raise ValueError("invalid token ID output")
    return ids


def completion(executable: Path, model: Path, prompt: str, request, expected: str,
               output: Path, name: str, timeout: int) -> dict:
    prompt_file = output / (name + ".prompt.txt")
    prompt_file.write_bytes(prompt.encode("utf-8"))
    command = [str(executable), "-m", str(model), "--binary-file", str(prompt_file),
               "-c", "4096", "-n", "1024", "-ngl", "0", "-t", "2", "-b", "64", "-ub", "64",
               "--temp", "0", "--seed", "42", "--special", "--no-display-prompt", "--no-conversation",
               "--no-context-shift", "--simple-io", "--no-escape", "--no-warmup", "--no-perf"]
    started = time.monotonic()
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                encoding="utf-8", errors="strict", timeout=timeout,
                                env=runtime_environment())
    except subprocess.TimeoutExpired:
        return {"status": "runtime_timeout", "elapsed_seconds": round(time.monotonic() - started, 3)}
    raw = ANSI.sub("", result.stdout)
    # These files contain ONLY authored fictional fixtures. Keeping raw diagnostics
    # here does not enable displaying the model's intermediate text in the app.
    (output / (name + ".stdout.txt")).write_text(raw, encoding="utf-8")
    (output / (name + ".stderr.txt")).write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        answer = {"status": "runtime_error", "exit_code": result.returncode}
    else:
        try:
            answer = parse_answer(raw, request)
        except OutputError as error:
            answer = {"status": "output_rejected", "reason": str(error)}
    answer.update({"elapsed_seconds": round(time.monotonic() - started, 3),
                   "expected_fixture_text_present": expected in answer.get("answer", "").lower(),
                   "authorized_actions": []})
    answer["existing_smoke_gate_passed"] = bool(answer["status"] == "needs_human_review"
                                                and answer["expected_fixture_text_present"]
                                                and answer.get("citations"))
    return answer


def self_test() -> None:
    """Exercise metadata relocation and payload preservation without any model."""
    raw_prompt = "<|query_start|>test<|query_end|>\n<|language_start|>\n"
    assert publisher_raw_prompt(raw_prompt) == raw_prompt
    assert publisher_raw_prompt(EOS + raw_prompt) == raw_prompt
    try:
        publisher_raw_prompt(EOS + EOS + raw_prompt)
    except ValueError:
        pass
    else:
        raise AssertionError("double EOS must not be silently repaired")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        items = [("tokenizer.ggml.model", 8, encoded_string("gpt2")),
                 ("tokenizer.ggml.eos_token_id", 4, struct.pack("<I", 2))]
        header = b"GGUF" + struct.pack("<IQQ", 3, 1, len(items))
        for key, kind, value in items:
            header += encoded_string(key) + struct.pack("<I", kind) + value
        header += encoded_string("test.weight") + struct.pack("<IQIQ", 1, 8, 0, 0)
        payload = bytes(range(32))
        source, derived = root / "source.gguf", root / "derived.gguf"
        source.write_bytes(header + b"\0" * (-len(header) % 32) + payload)
        report = patch_pre(source, derived)
        assert report["tensor_payload_sha256"] == hashlib.sha256(payload).hexdigest()
        with source.open("rb") as handle:
            assert "tokenizer.ggml.pre" not in layout(handle)["values"]
        with derived.open("rb") as handle:
            info = layout(handle)
            assert info["values"]["tokenizer.ggml.pre"] == "gpt-2"
            handle.seek(info["data_start"])
            assert handle.read() == payload
        try:
            patch_pre(source, derived)
        except ValueError:
            pass
        else:
            raise AssertionError("existing output was not rejected")
        try:
            layout(io.BytesIO(b"GGUF"))
        except ValueError:
            pass
        else:
            raise AssertionError("truncated input was not rejected")
    print("metadata relocation, payload integrity, overwrite guard and truncation checks passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--parity-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not all((args.work, args.model_output, args.output)) or not 1 <= args.timeout <= 300:
        parser.error("work, model-output and output are required; timeout must be 1-300 seconds")
    import tokenizers
    if tokenizers.__version__ != "0.22.2":
        raise RuntimeError("install exactly tokenizers==0.22.2 for this diagnostic")
    work, model_output, output = args.work.resolve(), args.model_output.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads((ROOT / "config/models.json").read_text(encoding="utf-8"))["artifacts"][0]
    source_file, q4_file = work / source["filename"], model_output / NAME
    manifest = json.loads((model_output / "model-manifest.json").read_text(encoding="utf-8"))
    if sha(source_file) != source["sha256"] or sha(q4_file) != manifest["sha256"]:
        raise RuntimeError("existing model hash does not match its pinned manifest")
    engines = list((work / "runtime").rglob("llama-completion"))
    tokenizers_cli = list((work / "runtime").rglob("llama-tokenize"))
    if len(engines) != 1 or len(tokenizers_cli) != 1:
        raise RuntimeError("expected exactly one pinned completion/tokenize executable")
    engine, tokenize = engines[0], tokenizers_cli[0]
    os.environ["LD_LIBRARY_PATH"] = str(engine.parent)
    verify_runtime(engine)
    tokenizer_file = output / "publisher-tokenizer.json"
    download(TOKENIZER_URL, tokenizer_file, TOKENIZER_BYTES, TOKENIZER_SHA)
    publisher = tokenizers.Tokenizer.from_file(str(tokenizer_file))
    report = {"scope": "Fictional tokenizer/framing diagnostic, not representative quality evaluation.",
              "tokenizer_sha256": TOKENIZER_SHA, "tokenizers_version": tokenizers.__version__,
              "trailing_newline": "Preserved verbatim with --binary-file; legacy -f dropped one LF.",
              "variants": {}, "parity": [], "cases": []}
    models = {}
    for label, original in (("upstream_bf16", source_file), ("q4_k_m", q4_file)):
        patched = work / (label + "-diagnostic-pre-gpt2.gguf")
        report["variants"][label] = patch_pre(original, patched)
        models[label] = (original, patched)
    cases = {}
    for profile in EXPECTATIONS:
        data = json.loads((ROOT / "data" / profile / "example.json").read_text(encoding="utf-8"))
        request = validate_request(data, profile)
        prompt = publisher_raw_prompt(build_prompt(request, 4096, 1024).text)
        # Production now supplies the HF postprocessor's initial EOS explicitly.
        # This diagnostic still compares raw publisher text plus controlled variants.
        if prompt.startswith(EOS):
            prompt = prompt[len(EOS):]
        if EOS in prompt:
            raise ValueError("unexpected duplicate EOS in publisher prompt")
        cases[profile] = request, prompt
    # Extra probes exercise Unicode, contractions and digit boundaries without inference.
    probes = {key: value[1] for key, value in cases.items()}
    probes["unicode_digits"] = "<|query_start|>What's 123456789? café, naïve, 東京 🌲<|query_end|>\n<|language_start|>\n"
    for label, (original, patched) in models.items():
        for profile, raw_prompt in probes.items():
            expected_ids = publisher.encode(raw_prompt, add_special_tokens=True).ids
            for variant, fixed_pre, initial_eos in VARIANTS:
                prompt = (EOS if initial_eos else "") + raw_prompt
                actual = token_ids(tokenize, patched if fixed_pre else original, prompt)
                report["parity"].append({"model_variant": label, "probe": profile, "framing": variant,
                                         "matches_publisher": actual == expected_ids,
                                         "publisher_ids": expected_ids, "actual_ids": actual})
        # Distinguish the old runner's missing LF from the four controlled variants.
        legacy = token_ids(tokenize, original, cases["acadia"][1][:-1])
        report["parity"].append({"model_variant": label, "probe": "legacy_acadia_dropped_lf",
                                 "framing": "legacy", "actual_ids": legacy,
                                 "matches_publisher": legacy == publisher.encode(cases["acadia"][1]).ids})
    write_json(output / "framing-diagnostic.json", report)
    parity_ok = all(row["matches_publisher"] for row in report["parity"] if row["framing"] == "pre_and_eos")
    report["corrected_tokenizer_parity_passed"] = parity_ok
    # Run all controlled variants even when the parity hypothesis is disproved:
    # diagnostics must expose a failed hypothesis rather than call it corrected.
    if not args.parity_only:
        # Produce the hypothesized correction for BOTH model precisions first,
        # so an interrupted run still preserves the most actionable evidence.
        for variant, fixed_pre, initial_eos in VARIANTS[-1:] + VARIANTS[:-1]:
            for label, (original, patched) in models.items():
                for profile, (request, raw_prompt) in cases.items():
                    name = f"{label}-{variant}-{profile}"
                    result = completion(engine, patched if fixed_pre else original,
                                        (EOS if initial_eos else "") + raw_prompt, request,
                                        EXPECTATIONS[profile], output, name, args.timeout)
                    result.update({"model_variant": label, "framing": variant, "profile": profile})
                    report["cases"].append(result)
                    write_json(output / "framing-diagnostic.json", report)
                    print(json.dumps({key: result.get(key) for key in
                                      ("model_variant", "framing", "profile", "status", "existing_smoke_gate_passed", "elapsed_seconds")}))
    corrected = [row for row in report["cases"] if row["framing"] == "pre_and_eos"]
    report["corrected_existing_smokes_passed"] = bool(corrected) and all(row["existing_smoke_gate_passed"] for row in corrected)
    report["release_authorized_by_this_diagnostic"] = False
    write_json(output / "framing-diagnostic.json", report)
    return 0 if parity_ok and (args.parity_only or report["corrected_existing_smokes_passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
