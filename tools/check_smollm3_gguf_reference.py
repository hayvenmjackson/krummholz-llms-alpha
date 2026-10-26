"""Direct original-BF16 GGUF diagnostic on hosted CPU, never app/release approval.

Reuses seven unchanged HF prompts and token IDs. Checks every tensor against the
original safetensors with only the converter's documented Q/K permutation and
lossless BF16-to-F32 norm promotion. It neither tunes prompts nor executes tools.
"""
from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time

from check_smollm3_reference import (ROOT, REPOSITORY, REVISION, canonical_sha, download,
                                     inspect_output, load_spec, request_for, sha, write_json)
from prepare_alpha_candidate import runtime
from prepare_alpha_model import ENGINE_COMMIT
from diagnose_pleias_framing import layout
from krummholz_llm.runner import ANSI, runtime_environment

VERSIONS = {"torch": "2.11.0+cpu", "transformers": "4.57.6", "tokenizers": "0.22.2",
            "safetensors": "0.6.2", "huggingface-hub": "0.36.0", "numpy": "2.2.6",
            "sentencepiece": "0.2.1", "protobuf": "4.25.8"}


def hardware(path: Path) -> dict:
    memory = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
    cpu = {}
    for line in Path("/proc/cpuinfo").read_text().split("\n\n", 1)[0].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            if key.strip() in ("model name", "vendor_id", "flags", "Features", "CPU architecture"):
                cpu[key.strip()] = value.strip()
    return {"machine": platform.machine(), "logical_cpus": os.cpu_count(), "cpu": cpu,
            "memory_total_kib": memory["MemTotal"], "memory_available_kib": memory["MemAvailable"],
            "disk_free_bytes": shutil.disk_usage(path).free}


def native_tokens(executable: Path, model: Path, prompt: str, no_bos: bool) -> list[int]:
    command = [str(executable), "-m", str(model), "--stdin", "--ids", "--no-escape"]
    if no_bos:
        command.append("--no-bos")
    process = subprocess.run(command, input=prompt, capture_output=True, text=True,
                             encoding="utf-8", check=True, timeout=60, env=runtime_environment())
    candidates = re.findall(r"^\s*\[(?:\s*\d+\s*,?)*\]\s*$", process.stdout, re.MULTILINE)
    if len(candidates) != 1:
        raise RuntimeError("Native tokenizer did not return exactly one token-ID list")
    ids = json.loads(candidates[0])
    if not ids or any(type(value) is not int for value in ids):
        raise RuntimeError("Native tokenizer returned invalid token IDs")
    return ids


def completion_content(raw: str) -> str:
    # Only trim the exact publisher EOS and CLI termination decoration. Preserve
    # the complete unmodified stdout separately; never rescue malformed JSON.
    match = re.fullmatch(r"(.*?)<\|im_end\|>\s*(?:\[end of text\])?\s*", raw, re.DOTALL)
    if not match:
        raise ValueError("No complete publisher end-of-turn marker in native output")
    return match[1]


def mapped_name(name: str) -> str:
    if name == "model.embed_tokens.weight":
        return "token_embd.weight"
    if name == "model.norm.weight":
        return "output_norm.weight"
    match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", name)
    names = {"input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "ffn_norm.weight",
             "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight",
             "self_attn.v_proj.weight": "attn_v.weight", "self_attn.o_proj.weight": "attn_output.weight",
             "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight",
             "mlp.down_proj.weight": "ffn_down.weight"}
    if not match or match[2] not in names:
        raise RuntimeError(f"Unexpected source tensor: {name}")
    return f"blk.{match[1]}.{names[match[2]]}"


def verify_tensor_values(source: Path, model: Path, llama_source: Path) -> dict:
    # Mmap the GGUF and read one source tensor at a time. The largest source is
    # the embedding (~0.5GB); block comparison avoids a full-model allocation.
    sys.path.insert(0, str(llama_source / "gguf-py"))
    import gguf
    import numpy as np
    import torch
    from safetensors import safe_open

    torch.set_num_threads(2)
    index = json.loads((source / "model.safetensors.index.json").read_text())
    reader = gguf.GGUFReader(str(model), mode="r")
    by_name = {tensor.name: tensor for tensor in reader.tensors}
    expected_names = {mapped_name(name) for name in index["weight_map"]}
    if set(by_name) != expected_names or len(by_name) != 326:
        raise RuntimeError("GGUF tensor set differs from the pinned 326 original tensors")
    counts = collections.Counter(t.tensor_type.name for t in reader.tensors)
    if not counts["BF16"] or set(counts) - {"BF16", "F32"}:
        raise RuntimeError("Converted tensor types include unexpected precision or quantization")
    checked = []
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(str(source / shard), framework="pt", device="cpu") as incoming:
            for name in sorted(incoming.keys()):
                original = incoming.get_tensor(name)
                if original.dtype != torch.bfloat16:
                    raise RuntimeError("Pinned source tensor is not original BF16")
                expected = original
                permuted = name.endswith(("q_proj.weight", "k_proj.weight"))
                if permuted:
                    heads = 16 if name.endswith("q_proj.weight") else 4
                    expected = original.reshape(heads, 2, original.shape[0] // heads // 2,
                                                *original.shape[1:]).swapaxes(1, 2).reshape(original.shape)
                converted = by_name[mapped_name(name)]
                if tuple(reversed(converted.shape.tolist())) != tuple(expected.shape):
                    raise RuntimeError(f"Tensor shape mismatch: {name}")
                if expected.ndim == 1:
                    if converted.tensor_type.name != "F32":
                        raise RuntimeError("Norm vectors must retain the converter's F32 promotion")
                    wanted = expected.float().numpy()
                    actual = converted.data.view(np.float32).reshape(wanted.shape)
                else:
                    if converted.tensor_type.name != "BF16":
                        raise RuntimeError("Every original matrix must remain BF16")
                    wanted = expected.contiguous().view(torch.int16).numpy().view(np.uint16)
                    actual = converted.data.view(np.uint16).reshape(wanted.shape)
                for start in range(0, wanted.shape[0], 64):
                    if not np.array_equal(wanted[start:start + 64], actual[start:start + 64]):
                        raise RuntimeError(f"Lossless mapped tensor comparison failed: {name}")
                checked.append({"source": name, "gguf": converted.name, "type": converted.tensor_type.name,
                                "qk_permutation": permuted, "all_values_exact": True})
                del actual, wanted, expected, original
    del by_name, reader
    gc.collect()
    return {"tensor_count": len(checked), "tensor_types": dict(counts), "all_values_exact": True,
            "transformations": "Q/K layout permutation; BF16 matrices unchanged in precision; F32 norm promotion is lossless.",
            "tensors": checked}


def self_test() -> None:
    load_spec()
    assert mapped_name("model.layers.35.self_attn.q_proj.weight") == "blk.35.attn_q.weight"
    assert mapped_name("model.embed_tokens.weight") == "token_embd.weight"
    assert completion_content('{"a":1}<|im_end|> [end of text]\n') == '{"a":1}'
    for bad in ('{"a":1}', '{"a":1}<|im_end|> extra'):
        try:
            completion_content(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Incomplete or decorated content was silently accepted")
    print("Bounded source declarations, tensor-name mapping and strict completion handling passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-source", type=Path)
    parser.add_argument("--hf-reference", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if os.environ.get("GITHUB_ACTIONS") != "true" or not all((args.llama_source, args.hf_reference, args.work, args.output)):
        parser.error("Hosted CI only; require source/reference/work/output directories")
    work, output, llama_source, hf_dir = (p.resolve() for p in (args.work, args.output, args.llama_source, args.hf_reference))
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "smollm3-gguf-reference.json"
    report = {"status": "preparing", "quality_accepted": False, "release_authorized": False,
              "precision": "direct BF16 with lossless F32 norm promotion", "quantization_performed": False,
              "tool_execution_available": False, "source_commit": os.environ.get("GITHUB_SHA"),
              "hf_reference_run_id": os.environ.get("HF_REFERENCE_RUN_ID"), "cases": [], "parity": []}
    write_json(report_path, report)
    try:
        spec = load_spec()
        report["hardware_before_download"] = hardware(work)
        write_json(report_path, report)
        if report["hardware_before_download"]["disk_free_bytes"] < 24 * 1024 ** 3:
            raise RuntimeError("Less than 24GiB free disk for source, BF16 model and conversion staging")
        if report["hardware_before_download"]["memory_available_kib"] < 9 * 1024 ** 2:
            raise RuntimeError("Less than 9GiB available RAM before conversion")
        actual_commit = subprocess.check_output(["git", "-C", str(llama_source), "rev-parse", "HEAD"], text=True).strip()
        if actual_commit != ENGINE_COMMIT:
            raise RuntimeError("Converter is not the immutable reviewed llama.cpp revision")
        versions = {name: importlib.metadata.version(name) for name in VERSIONS}
        if versions != VERSIONS:
            raise RuntimeError(f"Conversion runtime versions differ: {versions}")
        hf = json.loads((hf_dir / "smollm3-reference.json").read_text(encoding="utf-8"))
        if hf["repository"] != REPOSITORY or hf["revision"] != REVISION or hf["probe_spec_sha256"] != canonical_sha(spec):
            raise RuntimeError("Reference report differs from the unchanged probe/source specification")
        report.update(repository=REPOSITORY, revision=REVISION, source_files=spec["files"], versions=versions,
                      probe_spec_sha256=canonical_sha(spec), converter_commit=ENGINE_COMMIT,
                      hf_report_sha256=sha(hf_dir / "smollm3-reference.json"))
        source = work / "hf-source"
        source.mkdir()
        for name, pin in spec["files"].items():
            download(name, pin, source)
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                          HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
        model = work / "SmolLM3-3B-BF16.gguf"
        command = [sys.executable, str(llama_source / "convert_hf_to_gguf.py"), str(source),
                   "--outfile", str(model), "--outtype", "bf16", "--use-temp-file"]
        report.update(status="converting", converter_sha256=sha(llama_source / "convert_hf_to_gguf.py"),
                      conversion_options=["--outtype", "bf16", "--use-temp-file"])
        write_json(report_path, report)
        with (output / "conversion.stdout.txt").open("w") as stdout, (output / "conversion.stderr.txt").open("w") as stderr:
            subprocess.run(command, stdout=stdout, stderr=stderr, timeout=600, check=True)
        if not 6_000_000_000 < model.stat().st_size < 7_000_000_000:
            raise RuntimeError("Direct BF16 converted model has an unexpected size")
        integrity = verify_tensor_values(source, model, llama_source)
        write_json(output / "tensor-integrity.json", integrity)
        with model.open("rb") as handle:
            metadata = layout(handle)["values"]
        expected = {"general.architecture": "smollm3", "smollm3.block_count": 36,
                    "smollm3.embedding_length": 2048, "smollm3.attention.head_count": 16,
                    "smollm3.attention.head_count_kv": 4, "tokenizer.ggml.eos_token_id": 128012}
        if any(metadata.get(k) != v for k, v in expected.items()):
            raise RuntimeError("Converted architecture/tokenizer metadata mismatch")
        manifest = {"approved": False, "filename": model.name, "bytes": model.stat().st_size, "sha256": sha(model),
                    "publisher_repository": REPOSITORY, "publisher_revision": REVISION,
                    "source_files": spec["files"], "license": spec["publisher_license"],
                    "converter_commit": ENGINE_COMMIT, "quantization_performed": False,
                    "tensor_type_counts": integrity["tensor_types"], "all_tensor_values_verified": True,
                    "krummholz_training_completed": False, "release_authorized": False}
        write_json(output / "model-manifest.json", manifest)
        engine, tokenize, _unused_quantizer = runtime(work, output)
        from tokenizers import Tokenizer
        publisher = Tokenizer.from_file(str(source / "tokenizer.json"))
        cases = []
        hf_results = {case["id"]: case for case in hf["cases"]}
        for case in spec["cases"]:
            name, request = case["id"], request_for(case)
            prompt_path, input_path = hf_dir / f"{name}.prompt.txt", hf_dir / f"{name}.input.json"
            prompt = prompt_path.read_bytes().decode("utf-8")
            original = json.loads(input_path.read_text(encoding="utf-8"))
            expected_ids = original["input_token_ids"]
            if original["request"] != request or hashlib.sha256(prompt.encode()).hexdigest() != hf_results[name]["prompt_sha256"]:
                raise RuntimeError("Reference prompt/request was changed after original execution")
            if publisher.encode(prompt, add_special_tokens=False).ids != expected_ids or len(expected_ids) + 384 > 4096:
                raise RuntimeError("Original publisher token IDs or input budget differ")
            explicit_ids = native_tokens(tokenize, model, prompt, True)
            default_ids = native_tokens(tokenize, model, prompt, False)
            parity = {"id": name, "publisher_ids": expected_ids, "native_no_bos_ids": explicit_ids,
                      "native_default_ids": default_ids, "matches": explicit_ids == default_ids == expected_ids}
            report["parity"].append(parity)
            write_json(report_path, report)
            if not parity["matches"]:
                raise RuntimeError("Native token IDs do not exactly match the unchanged publisher prompt")
            shutil.copyfile(prompt_path, output / prompt_path.name)
            shutil.copyfile(input_path, output / input_path.name)
            cases.append((case, request, prompt))
        report.update(status="running", model_sha256=manifest["sha256"], tokenizer_parity_passed=True,
                      hardware_before_inference=hardware(work))
        write_json(report_path, report)
        if report["hardware_before_inference"]["memory_available_kib"] < 8 * 1024 ** 2:
            raise RuntimeError("Less than 8GiB available RAM before native inference")
        deadline = time.monotonic() + 20 * 60
        for case, request, prompt in cases:
            name = case["id"]
            result = {"id": name, "kind": case["kind"], "semantic_quality": "not_reviewed",
                      "review_criteria": case["review"], "authorized_actions": [], "tool_executed": False}
            remaining = int(deadline - time.monotonic())
            if remaining < 1:
                result["status"] = "not_run_budget_exhausted"
                report["cases"].append(result)
                write_json(report_path, report)
                continue
            command = [str(engine), "-m", str(model), "--binary-file", str(output / f"{name}.prompt.txt"),
                       "-c", "4096", "-n", "384", "-ngl", "0", "-t", "2", "-b", "64", "-ub", "64",
                       "--temp", "0", "--seed", "42", "--repeat-penalty", "1", "--special", "--no-display-prompt",
                       "--no-conversation", "--no-context-shift", "--simple-io", "--no-escape", "--no-warmup"]
            started = time.monotonic()
            stdout_path, stderr_path = output / f"{name}.stdout.txt", output / f"{name}.stderr.txt"
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                try:
                    process = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                             timeout=min(240, remaining), env=runtime_environment())
                    result.update(status="completed" if process.returncode == 0 else "runtime_error", exit_code=process.returncode)
                except subprocess.TimeoutExpired:
                    result["status"] = "runtime_timeout"
            raw = ANSI.sub("", stdout_path.read_text(encoding="utf-8"))
            result.update(raw_completion=raw, elapsed_seconds=round(time.monotonic() - started, 3),
                          input_tokens=len(next(p["publisher_ids"] for p in report["parity"] if p["id"] == name)))
            if result["status"] == "completed":
                try:
                    content = completion_content(raw)
                except ValueError as error:
                    result.update(status="output_incomplete", reason=str(error))
                else:
                    try:
                        result.update(inspect_output(content, case, request), structural_checks_passed=True)
                    except ValueError as error:
                        result.update(structural_checks_passed=False, structural_error=str(error))
                    hf_raw = (hf_dir / f"{name}.completion.txt").read_text(encoding="utf-8")
                    hf_content = completion_content(hf_raw)
                    result["hf_completed_output_text_identical"] = content.strip() == hf_content.strip()
                    result["hf_status"] = hf_results[name]["status"]
                    result["hf_raw_completion"] = hf_raw
            report["cases"].append(result)
            write_json(report_path, report)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        report.update(status="completed", all_cases_completed=len(report["cases"]) == 7 and all(c["status"] == "completed" for c in report["cases"]))
        write_json(report_path, report)
        return 0 if report["all_cases_completed"] else 2
    except Exception as error:
        report.update(status="setup_or_runtime_error", error=f"{type(error).__name__}: {error}")
        write_json(report_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
