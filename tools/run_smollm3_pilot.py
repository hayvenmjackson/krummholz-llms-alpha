"""Hosted-CI source-QA pilot using the frozen original BF16 reference configuration.

Every shard preserves nine attempts, including setup failures and partial raw
generation. This runner neither grades semantics nor executes actions.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import check_smollm3_reference as reference
from tools.validate_pilot_eval import build_report, canonical_sha256, check_result, load_json, validate_suite

SUITE = ROOT / "data/pilot_eval_cases.json"
SUITE_SHA = "6eda5672c65edf015409198eb21ed9f13ca346fde9d914f51497a38c3065a541"
SPEC_SHA = "8503c0a847c05828517f4b160a0b194659020ea27cd55ca8d4cfa7034e5cf8c6"
SHARDS = 4
MAX_NEW_TOKENS = 384
CASE_SECONDS = 240
RUN_SECONDS = 32 * 60  # Includes download/setup; leaves room in the 40-minute CI job.
EOS = 128012
PAD = 128004


def frozen_inputs():
    suite, spec = load_json(SUITE), reference.load_spec()
    checked = validate_suite(suite)
    if not checked["valid"] or checked["suite_sha256"] != SUITE_SHA:
        raise RuntimeError("Frozen pilot suite changed; preserve this run's evaluation version")
    if canonical_sha256(spec) != SPEC_SHA:
        raise RuntimeError("Frozen reference configuration changed; do not tune it on pilot answers")
    return suite, spec


def shard_cases(suite, shard):
    if type(shard) is not int or not 0 <= shard < SHARDS:
        raise ValueError("shard must be 0, 1, 2 or 3")
    cases = suite["cases"][shard::SHARDS]
    if len(cases) != 9:
        raise ValueError("Each fixed round-robin shard must contain nine cases")
    return cases


def messages_for(request, spec):
    # Deliberately excludes the profile rubric, expected answer and required IDs.
    return [{"role": "system", "content": spec["qa_system"]},
            {"role": "user", "content": json.dumps(request, ensure_ascii=False)}]


def parsed_result(raw, request):
    """Retain the model's answer/list exactly; no inserted citation markers."""
    inspected = reference.inspect_output(raw, {"kind": "qa"}, request)
    answer = inspected["parsed"]
    return {"status": "completed", "answer": answer["answer"],
            "citations": answer["citations"], "insufficient_evidence": answer["insufficient_evidence"],
            "citation_presentation": "separate_list", "authorized_actions": []}


def run_child(command, seconds, log_path):
    """A killed native operation cannot poison the next case's process."""
    started = time.monotonic()
    with log_path.open("xb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=seconds)
            timed_out = False
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
            code, timed_out = process.returncode, True
    return {"exit_code": code, "timed_out": timed_out,
            "elapsed_seconds": round(time.monotonic() - started, 3)}


def inventory(folder):
    return {path.relative_to(folder).as_posix(): {"bytes": path.stat().st_size, "sha256": reference.sha(path)}
            for path in sorted(folder.rglob("*")) if path.is_file() and path.name != "artifact-hashes.json"}


def hardware():
    memory = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
    cpu = {}
    for line in Path("/proc/cpuinfo").read_text().split("\n\n", 1)[0].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            if key.strip() in ("model name", "vendor_id", "flags", "Features", "CPU architecture"):
                cpu[key.strip()] = value.strip()
    return {"machine": platform.machine(), "logical_cpus": os.cpu_count(), "cpu": cpu,
            "mem_total_kib": memory["MemTotal"], "mem_available_kib": memory["MemAvailable"]}


def runtime_versions(spec):
    versions = {name: importlib.metadata.version(name) for name in spec["versions"]}
    if versions != spec["versions"]:
        raise RuntimeError(f"Pinned runtime version mismatch: {versions}")
    return versions


def configure_offline():
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      HF_HUB_DISABLE_IMPLICIT_TOKEN="1", TOKENIZERS_PARALLELISM="false")


def case_worker(work, output, case_id):
    """Private CI subprocess; source hashes were checked once by its coordinator."""
    if os.environ.get("KRUMMHOLZ_PILOT_WORKER") != "1":
        raise RuntimeError("A model worker must be launched by the verified shard coordinator")
    suite, spec = frozen_inputs()
    case = next((row for row in suite["cases"] if row["id"] == case_id), None)
    if case is None:
        raise RuntimeError("Unknown frozen pilot case")
    receipt = load_json(work / "verified-source.json")
    if receipt != {"spec_sha256": SPEC_SHA, "files": spec["files"]}:
        raise RuntimeError("Missing source verification receipt")
    versions = runtime_versions(spec)
    configure_offline()
    import resource
    import torch
    from transformers import AutoTokenizer, SmolLM3ForCausalLM, set_seed

    torch.set_num_threads(spec["limits"]["threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    set_seed(spec["limits"]["seed"])
    machine = hardware()
    if machine["mem_available_kib"] < 9 * 1024 * 1024:
        raise RuntimeError("Less than 9 GiB available for original BF16 CPU inference")
    tokenizer = AutoTokenizer.from_pretrained(str(work), local_files_only=True, trust_remote_code=False, use_fast=True)
    if not tokenizer.is_fast or tokenizer.chat_template != (work / "chat_template.jinja").read_text(encoding="utf-8"):
        raise RuntimeError("Loaded tokenizer differs from the pinned publisher template")
    request = case["request"]
    messages = messages_for(request, spec)
    fixed_date = dt.datetime.fromisoformat(spec["frozen_template_date"])
    template_args = {"enable_thinking": False, "strftime_now": lambda fmt: fixed_date.strftime(fmt)}
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_args)
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt", return_token_type_ids=False)
    input_ids = encoded["input_ids"][0].tolist()
    direct_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **template_args)
    if (input_ids != direct_ids or len(input_ids) + MAX_NEW_TOKENS > 4096
            or "Reasoning Mode: /no_think" not in prompt or fixed_date.strftime("%d %B %Y") not in prompt):
        raise RuntimeError("Publisher framing, frozen date or token budget mismatch")
    reference.write_json(output / "input.json", {"request": request, "messages": messages, "input_token_ids": input_ids})
    (output / "prompt.txt").write_bytes(prompt.encode("utf-8"))
    modules = ("transformers.models.smollm3.modeling_smollm3", "transformers.generation.utils",
               "transformers.tokenization_utils_base", "transformers.utils.chat_template_utils")
    evidence = {"case_id": case_id, "status": "loading_model", "request_sha256": canonical_sha256(request),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(), "input_tokens": len(input_ids),
                "hardware": machine, "versions": versions, "python": platform.python_version(),
                "runtime_file_sha256": {name: reference.sha(Path(importlib.import_module(name).__file__)) for name in modules},
                "semantic_review": "not_reviewed", "authorized_actions": [], "tool_execution_available": False}
    reference.write_json(output / "worker.json", evidence)
    load_start = time.monotonic()
    model = SmolLM3ForCausalLM.from_pretrained(str(work), local_files_only=True, trust_remote_code=False,
             use_safetensors=True, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.eval()
    if any(p.device.type != "cpu" or p.dtype != torch.bfloat16 for p in model.parameters()):
        raise RuntimeError("Original BF16 CPU dtype changed unexpectedly")
    evidence.update(status="generating", load_seconds=round(time.monotonic() - load_start, 3),
                    parameter_count=sum(p.numel() for p in model.parameters()))
    reference.write_json(output / "worker.json", evidence)

    class Journal:
        """A fictional diagnostic token journal, never a streamed app answer."""
        def __init__(self):
            self.prompt_seen, self.tokens = False, []
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
            with (output / "token-journal.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(values) + "\n")
                stream.flush()
            partial = output / "partial-completion.txt.partial"
            partial.write_bytes(tokenizer.decode(self.tokens, skip_special_tokens=False,
                                clean_up_tokenization_spaces=False).encode("utf-8"))
            partial.replace(output / "partial-completion.txt")
        def end(self):
            pass

    started = time.monotonic()
    set_seed(spec["limits"]["seed"])
    with torch.inference_mode():
        sequence = model.generate(**encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1,
            temperature=1.0, top_p=1.0, top_k=0, use_cache=True, eos_token_id=EOS, pad_token_id=PAD,
            streamer=Journal())
    generated = sequence[0, len(input_ids):].tolist()
    raw = tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    (output / "completion.txt").write_bytes(raw.encode("utf-8"))
    reference.write_json(output / "generated-token-ids.json", generated)
    evidence.update(status="generated", generation_seconds=round(time.monotonic() - started, 3),
                    generated_tokens=len(generated), stopped_on_eos=bool(generated and generated[-1] == EOS),
                    process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if evidence["stopped_on_eos"]:
        content = tokenizer.decode(generated[:-1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        try:
            result = parsed_result(content, request)
            evidence["deterministic"] = check_result(case, result)
        except (ValueError, TypeError) as error:
            result = {"status": "output_rejected", "reason": str(error), "authorized_actions": []}
    else:
        result = {"status": "output_rejected", "reason": "No publisher EOS before the 384-token output limit",
                  "authorized_actions": []}
    reference.write_json(output / "worker.json", evidence)
    reference.write_json(output / "result.json", result)
    return 0


def execute_shard(work, output, shard):
    started = time.monotonic()
    suite, spec = frozen_inputs()
    cases = shard_cases(suite, shard)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Output directory must be empty; do not overwrite a previous attempt")
    output.mkdir(parents=True, exist_ok=True)
    if work.exists() and any(work.iterdir()):
        raise RuntimeError("Use a fresh owned source directory for each shard")
    work.mkdir(parents=True, exist_ok=True)
    run_id = f"smollm3-pilot-{os.environ.get('GITHUB_RUN_ID', 'unknown')}-{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}"
    responses = {"suite_sha256": SUITE_SHA, "run_id": run_id, "records": [
        {"case_id": case["id"], "result": {"status": "not_run", "authorized_actions": [], "reason": "Pending setup"},
         "semantic_reviews": []} for case in cases]}
    manifest = {"schema_version": 1, "run_id": run_id, "shard": shard, "shard_count": SHARDS,
                "case_ids": [case["id"] for case in cases], "status": "preparing", "quality_accepted": False,
                "semantic_review": "not_reviewed", "source_commit": os.environ.get("GITHUB_SHA"),
                "suite_sha256": SUITE_SHA, "spec_sha256": SPEC_SHA,
                "repository": reference.REPOSITORY, "revision": reference.REVISION, "source_files": spec["files"],
                "runner_sha256": reference.sha(Path(__file__)),
                "reference_helper_sha256": reference.sha(Path(reference.__file__)),
                "validator_sha256": reference.sha(ROOT / "tools/validate_pilot_eval.py"),
                "context_tokens": 4096, "pilot_output_ceiling": 1024, "actual_max_new_tokens": MAX_NEW_TOKENS,
                "case_wall_seconds_ceiling": CASE_SECONDS, "coordinator_wall_seconds_budget": RUN_SECONDS,
                "timeout_scope": "Case wall time includes imports, tokenizer/model load and generation",
                "generation": "Original BF16 CPU, SDPA, greedy, seed42, fresh process/context; frozen publisher template, thinking disabled",
                "inference_dtype": "bfloat16", "quantization": None, "trust_remote_code": False,
                "frozen_template_date": spec["frozen_template_date"], "tool_execution_available": False,
                "cases": []}

    def save():
        reference.write_json(output / "responses.json", responses)
        reference.write_json(output / "run.json", manifest)

    save()
    try:
        for name, pin in spec["files"].items():
            reference.download(name, pin, work)
        config = load_json(work / "config.json")
        index = load_json(work / "model.safetensors.index.json")
        if (config.get("model_type") != "smollm3" or config.get("architectures") != ["SmolLM3ForCausalLM"]
                or "auto_map" in config or config.get("torch_dtype") != "bfloat16"
                or set(index["weight_map"].values()) != {name for name in spec["files"] if name.endswith(".safetensors")}):
            raise RuntimeError("Unexpected standard architecture or checkpoint shard index")
        reference.write_json(work / "verified-source.json", {"spec_sha256": SPEC_SHA, "files": spec["files"]})
        manifest.update(status="running", versions=runtime_versions(spec), hardware=hardware(),
                        architecture_config=config, source_verification="Exact bytes and SHA256 checked once per shard")
        save()
        os.environ["KRUMMHOLZ_PILOT_WORKER"] = "1"
        for position, (case, record) in enumerate(zip(cases, responses["records"])):
            folder = output / case["id"]
            folder.mkdir()
            remaining = started + RUN_SECONDS - time.monotonic()
            # Equal-share the remaining budget only when needed; still attempt every case.
            budget = max(1.0, min(CASE_SECONDS, (remaining - 15) / (len(cases) - position)))
            detail = {"case_id": case["id"], "attempted": True, "status": "running",
                      "case_wall_seconds_limit": round(budget, 3), "request_sha256": canonical_sha256(case["request"])}
            manifest["cases"].append(detail)
            record["result"] = {"status": "not_run", "reason": "Worker started; result not yet captured", "authorized_actions": []}
            save()
            try:
                execution = run_child([sys.executable, "-B", str(Path(__file__).resolve()), "--work", str(work),
                                       "--output", str(folder), "--case", case["id"]], budget, folder / "worker.log")
                detail.update(execution)
                if execution["timed_out"]:
                    result = {"status": "runtime_timeout", "reason": "Bounded worker terminated; partial raw evidence is diagnostic only",
                              "authorized_actions": []}
                elif execution["exit_code"] != 0:
                    result = {"status": "runtime_error", "reason": "Worker failed; see retained worker.log", "authorized_actions": []}
                else:
                    result = load_json(folder / "result.json")
                result.update(elapsed_seconds=execution["elapsed_seconds"], exit_code=execution["exit_code"])
            except Exception as error:
                result = {"status": "runtime_error", "reason": f"{type(error).__name__}: {error}", "authorized_actions": []}
            record["result"] = result
            detail.update(status=result["status"], deterministic=check_result(case, result))
            save()
            print(json.dumps({"case_id": case["id"], "status": result["status"], "seconds": result.get("elapsed_seconds")}), flush=True)
        manifest["status"] = "completed"
    except Exception as error:
        manifest.update(status="setup_error", error=f"{type(error).__name__}: {error}")
        for record in responses["records"]:
            if record["result"]["status"] == "not_run":
                record["result"]["reason"] = "Shard setup failed; see retained run.json"
    finally:
        manifest.update(elapsed_seconds=round(time.monotonic() - started, 3),
                        attempted_cases=sum(row["attempted"] for row in manifest["cases"]),
                        completed_answers=sum(row["result"]["status"] == "completed" for row in responses["records"]))
        save()
        reference.write_json(output / "artifact-hashes.json", inventory(output))
    # A finished collection is not a model-quality pass; retain output rejection/timeouts as results.
    return 0 if manifest["status"] == "completed" and manifest["attempted_cases"] == 9 else 2


def merge_shards(input_dir, output):
    suite, _ = frozen_inputs()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Merged output must be a new directory")
    output.mkdir(parents=True, exist_ok=True)
    rows, manifests, run_ids, configurations = {}, {}, set(), set()
    for path in sorted(input_dir.rglob("responses.json")):
        response, manifest = load_json(path), load_json(path.with_name("run.json"))
        if response["suite_sha256"] != SUITE_SHA or manifest["suite_sha256"] != SUITE_SHA or manifest["spec_sha256"] != SPEC_SHA:
            raise RuntimeError("Shard frozen suite/configuration mismatch")
        if manifest["actual_max_new_tokens"] != MAX_NEW_TOKENS or manifest["context_tokens"] != 4096:
            raise RuntimeError("Shard inference bounds differ from the frozen pilot run")
        configurations.add(canonical_sha256({key: manifest[key] for key in
            ("source_commit", "runner_sha256", "reference_helper_sha256", "validator_sha256")}))
        shard = manifest["shard"]
        if shard in manifests or manifest["case_ids"] != [c["id"] for c in shard_cases(suite, shard)]:
            raise RuntimeError("Duplicate shard or changed case allocation")
        expected = set(manifest["case_ids"])
        if len(response["records"]) != 9 or {r["case_id"] for r in response["records"]} != expected:
            raise RuntimeError("Missing, duplicate or wrong-case shard records")
        hashes = load_json(path.with_name("artifact-hashes.json"))
        for relative, pin in hashes.items():
            target = (path.parent / relative).resolve()
            if not target.is_relative_to(path.parent.resolve()):
                raise RuntimeError("Artifact hash path escapes shard directory")
            reference.verify(target, pin)
        if "responses.json" not in hashes or "run.json" not in hashes:
            raise RuntimeError("Shard evidence is absent from its hash manifest")
        run_ids.add(response["run_id"])
        if manifest["run_id"] != response["run_id"]:
            raise RuntimeError("Shard run identities differ")
        manifests[shard] = manifest
        rows.update({r["case_id"]: r for r in response["records"]})
    if len(run_ids) > 1 or len(configurations) > 1:
        raise RuntimeError("Merge requires one identified run; cannot mix retries")
    missing_shards = sorted(set(range(SHARDS)) - set(manifests))
    fallback_run_id = f"smollm3-pilot-{os.environ.get('GITHUB_RUN_ID', 'unavailable')}-{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}"
    responses = {"suite_sha256": SUITE_SHA, "run_id": next(iter(run_ids), fallback_run_id), "records": [
        rows.get(case["id"], {"case_id": case["id"], "result": {"status": "not_run", "authorized_actions": [],
          "reason": "Shard artifact missing; this case has no captured attempt"}, "semantic_reviews": []})
        for case in suite["cases"]]}
    reference.write_json(output / "responses.json", responses)
    reference.write_json(output / "deterministic-report.json", build_report(suite, responses))
    summary = {"run_id": responses["run_id"], "suite_sha256": SUITE_SHA, "spec_sha256": SPEC_SHA,
               "shards": manifests, "missing_shards": missing_shards, "case_count": 36,
               "run_identity_from": "verified_shard_artifacts" if run_ids else "collector_environment_no_shard_evidence",
               "semantic_review": "not_reviewed", "quality_accepted": False,
               "actual_max_new_tokens": MAX_NEW_TOKENS, "pilot_output_ceiling": 1024}
    reference.write_json(output / "run.json", summary)
    reference.write_json(output / "artifact-hashes.json", inventory(output))
    return 2 if missing_shards else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shard", type=int, choices=range(SHARDS))
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument("--merge", type=Path, help="Combine captured shard evidence; no inference")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        suite, _ = frozen_inputs()
        assert len({c["id"] for shard in range(SHARDS) for c in shard_cases(suite, shard)}) == 36
        import unittest
        result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromName("tests.test_smollm3_pilot"))
        return 0 if result.wasSuccessful() else 1
    if args.merge and args.output:
        return merge_shards(args.merge.resolve(), args.output.resolve())
    if not args.work or not args.output or os.environ.get("GITHUB_ACTIONS") != "true":
        parser.error("Inference is hosted-CI only; use --self-test or --merge locally")
    if args.case:
        return case_worker(args.work.resolve(), args.output.resolve(), args.case)
    if args.shard is None:
        parser.error("--shard is required")
    return execute_shard(args.work.resolve(), args.output.resolve(), args.shard)


if __name__ == "__main__":
    raise SystemExit(main())
