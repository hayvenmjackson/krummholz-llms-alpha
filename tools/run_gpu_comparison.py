"""Bounded, remote-only original-BF16 pilot; no training, tools or semantic grading.

The coordinator never imports torch. One isolated process owns each model, in
sequence. Every scheduled case/variant retains a record, even after setup fails.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_smollm3_pilot as pilot
from tools import check_smollm3_reference as reference
from tools.validate_pilot_eval import build_report, canonical_sha256, check_result, load_json

CONFIG = ROOT / "config/reference-comparators.json"
CONFIG_SHA = "e4dc6673e9fe036bcf62fd1fb07d3dddb994d53d0c93c8f9f8603cbd2ed33ad1"
GIB = 1024 ** 3


def inputs():
    config = load_json(CONFIG)
    if canonical_sha256(config) != CONFIG_SHA:
        raise RuntimeError("Comparison configuration changed; freeze and review a new run version")
    suite, qa = pilot.frozen_inputs()
    if config["suite_sha256"] != pilot.SUITE_SHA or config["qa_reference_spec_sha256"] != pilot.SPEC_SHA:
        raise RuntimeError("Pilot/reference version mismatch")
    return config, suite, qa


def budget_seconds(hourly_usd, already_spent_usd, reserve_usd, limits):
    values = (hourly_usd, already_spent_usd, reserve_usd)
    if any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
        raise ValueError("Billing inputs must be finite numbers")
    if hourly_usd <= 0 or already_spent_usd < 0 or reserve_usd < limits["minimum_reserve_usd"]:
        raise ValueError("Positive all-in hourly rate, nonnegative prior spend and at least $2 reserve required")
    available = limits["budget_usd"] - already_spent_usd - reserve_usd
    seconds = min(limits["total_seconds"], math.floor(available / hourly_usd * 3600))
    if seconds < 120:
        raise ValueError("Insufficient remaining allocation for even a bounded setup attempt")
    return seconds


def variants(config, model):
    result = []
    for row in config["variants"]:
        generation = {"max_new_tokens": config["limits"]["max_new_tokens"], "do_sample": row["do_sample"],
                      "num_beams": 1, "use_cache": True, "eos_token_id": model["eos_token_ids"],
                      "pad_token_id": model["pad_token_id"], "repetition_penalty": 1.0,
                      "no_repeat_ngram_size": 0}
        generation.update(model["primary_sampling"] if row["do_sample"] else
                          {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0})
        result.append({**row, "generation": generation})
    return result


def scheduled(config, suite, model):
    return [(variant, case) for variant in variants(config, model) for case in suite["cases"]]


def result_error(status, reason):
    return {"status": status, "reason": reason, "authorized_actions": []}


def attempt_dir(output, model_id, variant_id, case_id):
    return output / model_id / variant_id / case_id


def initial_responses(config, suite, model, variant, run_id):
    return {"suite_sha256": config["suite_sha256"], "run_id": f"{run_id}-{model['id']}-{variant['id']}",
            "records": [{"case_id": case["id"], "result": result_error("not_run", "Scheduled; not started"),
                         "semantic_reviews": []} for case in suite["cases"]]}


def finalize(output, config, suite, run_id, reasons):
    """Reconstruct all 216 outcomes, including a missing/crashed model worker."""
    summary = []
    for model in config["models"]:
        for variant in variants(config, model):
            evidence = initial_responses(config, suite, model, variant, run_id)
            for record, case in zip(evidence["records"], suite["cases"], strict=True):
                folder = attempt_dir(output, model["id"], variant["id"], case["id"])
                path = folder / "result.json"
                if path.is_file():
                    record["result"] = load_json(path)
                else:
                    active = (folder / "attempt.json").is_file()
                    record["result"] = result_error("runtime_timeout" if active and reasons.get(model["id"], {}).get("timed_out")
                                                    else "runtime_error" if active else "not_run",
                                                    reasons.get(model["id"], {}).get("reason", "Worker did not produce a result"))
                    reference.write_json(path, record["result"])
            folder = output / model["id"] / variant["id"]
            reference.write_json(folder / "responses.json", evidence)
            report = build_report(suite, evidence)
            reference.write_json(folder / "deterministic-report.json", report)
            summary.append({"model": model["id"], "variant": variant["id"], "tier": variant["tier"],
                            "seed": variant["seed"], "scheduled": len(evidence["records"]),
                            "outcomes": dict(collections.Counter(row["result"]["status"] for row in evidence["records"])),
                            "semantic_review": "not_reviewed", "approved_for_application": False})
    reference.write_json(output / "summary.json", {"scheduled_attempts": 216, "groups": summary,
                         "selection_policy": "All variants retained separately; no best-of selection",
                         "semantic_review": "not_reviewed", "approved_for_application": False})
    reference.write_json(output / "artifact-hashes.json", pilot.inventory(output))
    return summary


def gpu_preflight():
    raw = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,driver_version",
                          "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=15).stdout
    lines = [line.split(",") for line in raw.strip().splitlines()]
    if len(lines) != 1 or len(lines[0]) != 5:
        raise RuntimeError("Exactly one visible 24/48 GiB-class GPU is required")
    index, name, total, free, driver = [part.strip() for part in lines[0]]
    total, free = int(total), int(free)
    if not (22 * 1024 <= total <= 25 * 1024 or 44 * 1024 <= total <= 49 * 1024) or free < 21 * 1024:
        raise RuntimeError(f"Expected idle 24/48 GiB-class GPU; found {total} MiB total/{free} MiB free")
    return {"index": index, "name": name, "total_mib": total, "free_mib": free, "driver": driver}


def versions(config):
    found = {name: importlib.metadata.version(name) for name in config["versions"]}
    if found != config["versions"] or ".".join(platform.python_version_tuple()[:2]) != config["python_minor"]:
        raise RuntimeError(f"Pinned Python/runtime mismatch: Python {platform.python_version()}, {found}")
    return found


def effective_memory(host_available_kib, cgroup_root=Path("/sys/fs/cgroup"), membership=Path("/proc/self/cgroup")):
    """Bound host MemAvailable by container allowances and clean inactive cache."""
    memberships = membership.read_text().splitlines() if membership.is_file() else []
    groups = []
    for line in memberships:
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        if fields[0] == "0" and fields[1] == "":
            groups.append(("v2", cgroup_root, fields[2], "memory.max", "memory.current"))
        elif "memory" in fields[1].split(","):
            groups.append(("v1", cgroup_root / "memory", fields[2], "memory.limit_in_bytes", "memory.usage_in_bytes"))
    # Cgroup namespaces may expose their root as / even when membership is host-relative.
    if not groups:
        if (cgroup_root / "memory.max").is_file():
            groups.append(("v2", cgroup_root, "/", "memory.max", "memory.current"))
        elif (cgroup_root / "memory/memory.limit_in_bytes").is_file():
            groups.append(("v1", cgroup_root / "memory", "/", "memory.limit_in_bytes", "memory.usage_in_bytes"))
    bounds, seen = [], set()
    for version, root, relative, limit_name, usage_name in groups:
        root = root.resolve()
        leaf = (root / relative.lstrip("/")).resolve()
        if not leaf.is_relative_to(root) or not leaf.is_dir():
            leaf = root
        current = leaf
        while current.is_relative_to(root):
            limit_file, usage_file = current / limit_name, current / usage_name
            if limit_file.is_file() and str(limit_file) not in seen:
                seen.add(str(limit_file))
                raw = limit_file.read_text().strip()
                # v1 uses a very large signed integer for an unlimited allowance.
                limit = None if raw == "max" or int(raw) >= 1 << 60 else int(raw)
                usage = int(usage_file.read_text().strip())
                if usage < 0 or (limit is not None and limit < 0):
                    raise RuntimeError("Invalid cgroup memory accounting")
                stats = {}
                if (current / "memory.stat").is_file():
                    for line in (current / "memory.stat").read_text().splitlines():
                        key, value = line.split()
                        stats[key] = int(value)
                if version == "v2":
                    cache = {key: stats.get(key, 0) for key in ("inactive_file", "file_dirty", "file_writeback")}
                else:
                    cache = {key: stats.get("total_" + key, stats.get(key, 0)) for key in ("inactive_file", "dirty", "writeback")}
                if any(value < 0 for value in cache.values()):
                    raise RuntimeError("Invalid cgroup file-cache accounting")
                # Do not credit active pages, anonymous memory, dirty pages or writeback.
                # Counting dirty/writeback twice when they overlap is conservative.
                inactive, dirty, writeback = cache.values()
                reclaimable = max(0, min(usage, inactive) - dirty - writeback)
                available = None if limit is None else min(limit, max(0, limit - usage + reclaimable))
                bounds.append({"version": version, "path": str(current), "limit_bytes": limit,
                               "usage_bytes": usage, "file_cache_counters": cache,
                               "clean_inactive_file_bytes": reclaimable,
                               "raw_available_bytes": None if limit is None else max(0, limit - usage),
                               "available_bytes": available})
            if current == root:
                break
            current = current.parent
    available = min([host_available_kib * 1024] + [row["available_bytes"] for row in bounds if row["available_bytes"] is not None])
    return {"host_available_kib": host_available_kib, "effective_available_kib": available // 1024,
            "cgroup_bounds": bounds, "cgroup_accounting_visible": bool(bounds)}


def pinned_download(model, name, pin, work, deadline):
    """Public HTTPS only; no Hugging Face authentication or unpinned cache lookup."""
    target = work / name
    if target.exists():
        reference.verify(target, pin)
        return "verified_cache"
    partial = work / (name + ".partial")
    request = urllib.request.Request(f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{name}",
                                     headers={"User-Agent": "Krummholz-BF16-pilot/0.1"})
    digest, count = hashlib.sha256(), 0
    # Exclusive creation preserves interrupted evidence and never overwrites a file.
    with urllib.request.urlopen(request, timeout=30) as response, partial.open("xb") as dest:
        if not response.url.startswith("https://"):
            raise RuntimeError("Publisher download redirected outside HTTPS")
        while chunk := response.read(1024 * 1024):
            if time.monotonic() >= deadline:
                raise TimeoutError("Model allocation expired during source download")
            count += len(chunk)
            if count > pin["bytes"]:
                raise RuntimeError("Source download exceeded pinned byte size")
            dest.write(chunk)
            digest.update(chunk)
    if count != pin["bytes"] or digest.hexdigest() != pin["sha256"]:
        raise RuntimeError(f"Pinned source size/hash mismatch: {name}")
    partial.replace(target)
    return "downloaded_and_verified"


def validate_source_metadata(model, work):
    config = load_json(work / "config.json")
    if (config.get("model_type") != model["model_type"] or config.get("architectures") != [model["architecture"]]
            or config.get("dtype", config.get("torch_dtype")) != "bfloat16"
            or config.get("auto_map") or config.get("quantization_config")):
        raise RuntimeError("Unexpected model architecture, dtype, remote code or quantization")
    weight_map = load_json(work / "model.safetensors.index.json")["weight_map"]
    shards = {name for name in model["files"] if name.endswith(".safetensors")}
    if not weight_map or set(weight_map.values()) != shards:
        raise RuntimeError("Weight index references a missing/unpinned shard")
    generation = load_json(work / "generation_config.json")
    if generation["eos_token_id"] != model["eos_token_ids"]:
        raise RuntimeError("Publisher EOS configuration changed")
    return config


def completion_result(generated, tokenizer, model, request, timed_out=False):
    if timed_out:
        return result_error("runtime_timeout", "Generation exceeded the declared case deadline")
    if not generated or generated[-1] not in model["eos_token_ids"]:
        return result_error("output_rejected", "No publisher EOS before the 384-token output ceiling")
    content = tokenizer.decode(generated[:-1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    try:
        return pilot.parsed_result(content, request)
    except (ValueError, TypeError) as error:
        return result_error("output_rejected", str(error))


def model_worker(work, output, model_id, seconds):
    if os.environ.get("KRUMMHOLZ_GPU_WORKER") != "1" or platform.system() != "Linux" or not os.environ.get("RUNPOD_POD_ID"):
        raise RuntimeError("Model inference is restricted to the coordinator's remote Runpod worker")
    started = time.monotonic()
    deadline = started + seconds
    config, suite, qa = inputs()
    model_spec = next(model for model in config["models"] if model["id"] == model_id)
    model_output = output / model_id
    progress = model_output / "progress.json"
    reference.write_json(progress, {"phase": "verifying_sources"})
    versions_found = versions(config)
    work.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for name, pin in model_spec["files"].items():
        receipts[name] = {**pin, "verification": pinned_download(model_spec, name, pin, work, deadline)}
    validate_source_metadata(model_spec, work)
    reference.write_json(model_output / "verified-source.json", {"repository": model_spec["repository"],
                         "revision": model_spec["revision"], "files": receipts})
    pilot.configure_offline()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, StoppingCriteria, StoppingCriteriaList, set_seed
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("A single CUDA GPU with native BF16 support is required")
    if torch.cuda.mem_get_info()[0] < 21 * GIB:
        raise RuntimeError("Less than 21 GiB free GPU memory before model loading")
    for name in model_spec["files"]:
        if name.endswith(".safetensors"):
            with safe_open(work / name, framework="pt", device="cpu") as tensors:
                if any(tensors.get_slice(key).get_dtype() != "BF16" for key in tensors.keys()):
                    raise RuntimeError("Original weight tensor is not BF16")
    tokenizer = AutoTokenizer.from_pretrained(str(work), local_files_only=True, trust_remote_code=False, use_fast=True)
    if not tokenizer.is_fast or hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != model_spec["template_sha256"]:
        raise RuntimeError("Loaded tokenizer/publisher template mismatch")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    modules = ("transformers.generation.utils", "transformers.tokenization_utils_base", "transformers.utils.chat_template_utils",
               f"transformers.models.{model_spec['model_type']}.modeling_{model_spec['model_type']}")
    metadata = {"repository": model_spec["repository"], "revision": model_spec["revision"], "versions": versions_found,
                "python": platform.python_version(), "hardware": pilot.hardware(), "gpu": gpu_preflight(),
                "cuda": torch.version.cuda, "capability": list(torch.cuda.get_device_capability()),
                "template_kwargs": model_spec["template_kwargs"], "template_sha256": model_spec["template_sha256"],
                "runtime_file_sha256": {name: reference.sha(Path(importlib.import_module(name).__file__)) for name in modules},
                "attention_implementation": "sdpa", "dtype": "bfloat16", "semantic_review": "not_reviewed",
                "approved_for_application": False, "authorized_actions": [], "tool_execution_available": False}
    metadata["memory_preflight"] = effective_memory(metadata["hardware"]["mem_available_kib"])
    reference.write_json(model_output / "model.json", metadata)
    if not metadata["memory_preflight"]["cgroup_accounting_visible"]:
        raise RuntimeError("Container memory accounting is unavailable; cannot verify its RAM allowance")
    if metadata["memory_preflight"]["effective_available_kib"] < 24 * 1024 * 1024:
        raise RuntimeError("Less than 24 GiB effective host/container RAM before model loading")
    reference.write_json(progress, {"phase": "loading_model"})
    load_started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(str(work), local_files_only=True, trust_remote_code=False,
             use_safetensors=True, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to("cuda:0").eval()
    if any(p.device.type != "cuda" or p.dtype != torch.bfloat16 for p in model.parameters()):
        raise RuntimeError("Original BF16 GPU parameter dtype/device changed")
    metadata.update(load_seconds=round(time.monotonic() - load_started, 3),
                    parameter_count=sum(p.numel() for p in model.parameters()),
                    setup_seconds=round(time.monotonic() - started, 3))
    reference.write_json(model_output / "model.json", metadata)

    class Deadline(StoppingCriteria):
        def __init__(self, until):
            self.until, self.expired = until, False
        def __call__(self, input_ids, scores, **kwargs):
            self.expired = time.monotonic() >= self.until
            return self.expired

    class Journal:
        def __init__(self, folder):
            self.folder, self.prompt_seen, self.tokens = folder, False, []
        def put(self, values):
            values = values.tolist()
            if not self.prompt_seen:
                self.prompt_seen = True
                return
            if values and isinstance(values[0], list):
                if len(values) != 1:
                    raise RuntimeError("Unexpected generation batch")
                values = values[0]
            self.tokens.extend(values)
            with (self.folder / "token-journal.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(values) + "\n")
                stream.flush()
            partial = self.folder / "partial-completion.txt.partial"
            partial.write_bytes(tokenizer.decode(self.tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False).encode())
            partial.replace(self.folder / "partial-completion.txt")
        def end(self):
            pass

    for variant, case in scheduled(config, suite, model_spec):
        if deadline - time.monotonic() < 15:
            break
        folder = attempt_dir(output, model_id, variant["id"], case["id"])
        until = min(deadline - 3, time.monotonic() + config["limits"]["case_seconds"])
        reference.write_json(progress, {"phase": "case", "variant": variant["id"], "case_id": case["id"],
                                       "hard_deadline_monotonic": until + 5})
        attempt = {"case_id": case["id"], "variant": variant, "request_sha256": canonical_sha256(case["request"]),
                   "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "semantic_review": "not_reviewed"}
        reference.write_json(folder / "attempt.json", attempt)
        case_started = time.monotonic()
        try:
            messages = pilot.messages_for(case["request"], qa)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **model_spec["template_kwargs"])
            encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt", return_token_type_ids=False)
            ids = encoded["input_ids"][0].tolist()
            direct = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **model_spec["template_kwargs"])
            if ids != direct or len(ids) + config["limits"]["max_new_tokens"] > config["limits"]["context_tokens"]:
                raise ValueError("Publisher tokenization parity or fixed context budget failed")
            (folder / "prompt.txt").write_bytes(prompt.encode())
            reference.write_json(folder / "input.json", {"request": case["request"], "messages": messages, "input_token_ids": ids})
            generation = GenerationConfig(**variant["generation"])
            attempt.update(input_tokens=len(ids), prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                           actual_generation_config=generation.to_dict())
            reference.write_json(folder / "attempt.json", attempt)
            set_seed(variant["seed"])
            stopper = Deadline(until)
            encoded = {name: tensor.to("cuda:0") for name, tensor in encoded.items()}
            torch.cuda.synchronize()
            generation_started = time.monotonic()
            with torch.inference_mode():
                sequence = model.generate(**encoded, generation_config=generation, streamer=Journal(folder),
                                          stopping_criteria=StoppingCriteriaList([stopper]))
            torch.cuda.synchronize()
            generated = sequence[0, len(ids):].tolist()
            (folder / "completion.txt").write_bytes(tokenizer.decode(generated, skip_special_tokens=False,
                                                    clean_up_tokenization_spaces=False).encode())
            reference.write_json(folder / "generated-token-ids.json", generated)
            result = completion_result(generated, tokenizer, model_spec, case["request"], stopper.expired)
            attempt.update(generation_seconds=round(time.monotonic() - generation_started, 3), generated_tokens=len(generated),
                           stopped_on_eos=bool(generated and generated[-1] in model_spec["eos_token_ids"]),
                           deterministic=check_result(case, result), peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        except Exception as error:
            result = result_error("runtime_error", f"{type(error).__name__}: {error}")
            reference.write_json(folder / "result.json", result)
            attempt.update(elapsed_seconds=round(time.monotonic() - case_started, 3), exception=result["reason"])
            reference.write_json(folder / "attempt.json", attempt)
            # A failed CUDA operation may poison this process; never disguise a retry as the same run.
            raise
        attempt["elapsed_seconds"] = round(time.monotonic() - case_started, 3)
        reference.write_json(folder / "attempt.json", attempt)
        reference.write_json(folder / "result.json", result)
        reference.write_json(progress, {"phase": "between_cases"})
    reference.write_json(progress, {"phase": "finished"})
    return 0


def watch_worker(command, env, log_path, progress_path, seconds):
    started = time.monotonic()
    timed_out, reason = False, "Worker exited without producing all results"
    with log_path.open("xb") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                now = time.monotonic()
                progress = load_json(progress_path) if progress_path.is_file() else {}
                if now - started >= seconds or now >= progress.get("hard_deadline_monotonic", math.inf):
                    timed_out, reason = True, "Model allocation or active case watchdog deadline reached"
                    process.kill()
                    break
                time.sleep(min(0.5, max(0.01, seconds - (now - started))))
            process.wait(timeout=5)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    return {"exit_code": process.returncode, "timed_out": timed_out, "reason": reason,
            "elapsed_seconds": round(time.monotonic() - started, 3)}


def worker_environment(parent_environment):
    """Model dependencies need public source files, never cloud/HF credentials."""
    credentials = {"RUNPOD_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}
    env = {name: value for name, value in parent_environment.items() if name.upper() not in credentials}
    env.update(KRUMMHOLZ_GPU_WORKER="1", HF_HUB_DISABLE_TELEMETRY="1",
               HF_HUB_DISABLE_IMPLICIT_TOKEN="1", TOKENIZERS_PARALLELISM="false")
    return env


def execute(work, output, hourly_usd, already_spent_usd, reserve_usd):
    started = time.monotonic()
    config, suite, _ = inputs()
    allowed = budget_seconds(hourly_usd, already_spent_usd, reserve_usd, config["limits"])
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Output must be new/empty; previous evidence cannot be overwritten")
    if work.resolve() == output.resolve() or work.resolve() in output.resolve().parents or output.resolve() in work.resolve().parents:
        raise RuntimeError("Use separate non-nested model-cache and evidence directories")
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    run_id = "gpu-comparison-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    reasons = {}
    manifest = {"run_id": run_id, "config_sha256": CONFIG_SHA, "suite_sha256": pilot.SUITE_SHA,
                "qa_reference_spec_sha256": pilot.SPEC_SHA, "scheduled_attempts": 216,
                "source_files_sha256": {name: reference.sha(ROOT / name) for name in (
                    "tools/run_gpu_comparison.py", "tools/run_smollm3_pilot.py", "tools/check_smollm3_reference.py",
                    "tools/validate_pilot_eval.py", "config/reference-comparators.json", "data/pilot_eval_cases.json")},
                "billing": {"confirmed_all_in_hourly_usd": hourly_usd, "prior_spend_usd": already_spent_usd,
                            "reserve_usd": reserve_usd, "total_cap_usd": 20, "allowed_seconds": allowed,
                            "provider_billing_stops_automatically": False},
                "limits": config["limits"], "variants": config["variants"], "approved_for_application": False,
                "semantic_review": "not_reviewed", "training_performed": False}
    for model in config["models"]:
        for variant, case in scheduled(config, suite, model):
            attempt_dir(output, model["id"], variant["id"], case["id"]).mkdir(parents=True)
    reference.write_json(output / "run.json", manifest)
    fatal = None
    try:
        if platform.system() != "Linux" or not os.environ.get("RUNPOD_POD_ID"):
            raise RuntimeError("Actual inference is remote Runpod only; use --plan or unit tests locally")
        manifest["versions"] = versions(config)
        manifest["hardware"] = pilot.hardware()
        manifest["gpu"] = gpu_preflight()
        manifest["memory_preflight"] = effective_memory(manifest["hardware"]["mem_available_kib"])
        if not manifest["memory_preflight"]["cgroup_accounting_visible"]:
            raise RuntimeError("Container memory accounting is unavailable; cannot verify its RAM allowance")
        if manifest["memory_preflight"]["effective_available_kib"] < 24 * 1024 * 1024:
            raise RuntimeError("At least 24 GiB effective available host/container RAM is required")
        missing = sum(pin["bytes"] for model in config["models"] for name, pin in model["files"].items()
                      if not (work / model["id"] / name).is_file())
        if shutil.disk_usage(work).free < missing + 8 * GIB:
            raise RuntimeError("Insufficient free disk for pinned source files plus 8 GiB reserve")
        reference.write_json(output / "python-packages.json", dict(sorted((dist.metadata["Name"], dist.version)
                             for dist in importlib.metadata.distributions() if dist.metadata["Name"])))
        reference.write_json(output / "run.json", manifest)
        for index, model in enumerate(config["models"]):
            remaining = started + allowed - config["limits"]["reporting_reserve_seconds"] - time.monotonic()
            seconds = remaining / (len(config["models"]) - index)
            if seconds < 30:
                reasons[model["id"]] = {"reason": "Overall time/budget allocation exhausted before model setup"}
                continue
            env = worker_environment(os.environ)
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", model["id"], "--work", str(work / model["id"]),
                       "--output", str(output), "--seconds", str(seconds)]
            reasons[model["id"]] = watch_worker(command, env, output / model["id"] / "worker.log",
                                                output / model["id"] / "progress.json", seconds)
            reference.write_json(output / model["id"] / "execution.json", reasons[model["id"]])
    except Exception as error:
        fatal = f"{type(error).__name__}: {error}"
        for model in config["models"]:
            reasons.setdefault(model["id"], {"reason": "Setup failed: " + fatal})
    finally:
        manifest.update(elapsed_seconds=round(time.monotonic() - started, 3), execution=reasons, setup_error=fatal)
        manifest["billing"]["estimated_workload_usd"] = round(manifest["elapsed_seconds"] / 3600 * hourly_usd, 4)
        reference.write_json(output / "run.json", manifest)
        summary = finalize(output, config, suite, run_id, reasons)
    complete = sum(group["outcomes"].get("completed", 0) + group["outcomes"].get("output_rejected", 0) for group in summary)
    print(json.dumps({"run_id": run_id, "attempts_with_finished_generation": complete, "scheduled": 216,
                      "semantic_review": "not_reviewed", "setup_error": fatal,
                      "reminder": "This process does not stop the paid pod; retain evidence and stop it promptly."}))
    return 0 if complete == 216 and fatal is None else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--worker", choices=("olmo3-7b", "qwen3-8b"), help=argparse.SUPPRESS)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hourly-usd", type=float)
    parser.add_argument("--already-spent-usd", type=float)
    parser.add_argument("--reserve-usd", type=float, default=2.0)
    parser.add_argument("--seconds", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.plan:
        config, suite, _ = inputs()
        print(json.dumps({"configuration_sha256": CONFIG_SHA, "cases": len(suite["cases"]), "attempts": 216,
                          "models": [{"id": m["id"], "repository": m["repository"], "revision": m["revision"],
                                      "download_bytes": sum(p["bytes"] for p in m["files"].values()),
                                      "variants": variants(config, m)} for m in config["models"]],
                          "limits": config["limits"], "inference_started": False}, indent=2))
        return 0
    if args.work is None or args.output is None:
        parser.error("--work and --output are required")
    if args.worker:
        if args.seconds is None or not math.isfinite(args.seconds) or not 0 < args.seconds <= 7200:
            parser.error("Invalid worker allocation")
        return model_worker(args.work.resolve(), args.output.resolve(), args.worker, args.seconds)
    if args.hourly_usd is None or args.already_spent_usd is None:
        parser.error("--hourly-usd and --already-spent-usd require current confirmed billing inputs")
    return execute(args.work.resolve(), args.output.resolve(), args.hourly_usd, args.already_spent_usd, args.reserve_usd)


if __name__ == "__main__":
    raise SystemExit(main())
