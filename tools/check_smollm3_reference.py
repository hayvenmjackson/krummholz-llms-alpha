"""CI-only original SmolLM3 BF16 probe; inert tool proposals, no release approval.

The seven small cases are fictional development diagnostics, not the separate
held-out pilot suite. Full outputs are retained for semantic review. No keyword
or parser success is a quality pass. No tool executor exists in this program.
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
import re
import signal
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config/smollm3-reference.json"
REPOSITORY = "HuggingFaceTB/SmolLM3-3B"
REVISION = "a07cc9a04f16550a088caea529712d1d335b0ac1"


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify(path: Path, pin: dict) -> None:
    if not path.is_file() or path.stat().st_size != pin["bytes"] or sha(path) != pin["sha256"]:
        raise RuntimeError(f"Pinned source size/hash mismatch: {path.name}")


def download(name: str, pin: dict, work: Path) -> None:
    target = work / name
    if target.exists():
        verify(target, pin)
        return
    partial = work / (name + ".partial")
    owned = False
    try:
        request = urllib.request.Request(f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}",
                                         headers={"User-Agent": "Krummholz-SmolLM3-reference/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("xb") as dest:
            owned = True
            count = 0
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > pin["bytes"]:
                    raise RuntimeError("Source download exceeded pinned byte size")
                dest.write(chunk)
        verify(partial, pin)
        partial.replace(target)
    finally:
        if owned and partial.exists():
            partial.unlink()


def load_spec() -> dict:
    spec = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if spec["repository"] != REPOSITORY or spec["revision"] != REVISION:
        raise RuntimeError("Publisher changed; re-verify all source pins")
    comparator = json.loads((ROOT / "config/models.json").read_text(encoding="utf-8"))["comparator"]
    if comparator["repository"] != REPOSITORY or comparator["revision"] != REVISION:
        raise RuntimeError("The configured comparator no longer matches this probe")
    if spec["approved_for_application"] or spec["krummholz_training_completed"]:
        raise RuntimeError("This diagnostic cannot authorize an application or training claim")
    expected = {"config.json", "generation_config.json", "model.safetensors.index.json",
                "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
                "chat_template.jinja", "README.md"}
    if set(spec["files"]) != expected:
        raise RuntimeError("Unexpected publisher file set")
    for name, pin in spec["files"].items():
        if Path(name).name != name or not 0 < pin["bytes"] < 6_000_000_000 or not re.fullmatch(r"[a-f0-9]{64}", pin["sha256"]):
            raise RuntimeError("Invalid source pin")
    if spec["limits"] != {"context_tokens": 4096, "max_new_tokens": 384, "case_seconds": 240,
                          "generation_total_seconds": 1200, "threads": 2, "seed": 42}:
        raise RuntimeError("Diagnostic bounds changed; review before running")
    if len(spec["cases"]) != 7 or len({c["id"] for c in spec["cases"]}) != 7:
        raise RuntimeError("Expected seven distinct development probes")
    return spec


def request_for(case: dict) -> dict:
    if "fixture" in case:
        allowed = {f"data/{profile}/example.json" for profile in ("acadia", "katahdin", "atlantic")}
        if case["fixture"] not in allowed:
            raise RuntimeError("Only the three declared fictional startup fixtures may be loaded")
        request = json.loads((ROOT / case["fixture"]).read_text(encoding="utf-8"))
    else:
        request = case["request"]
    if set(request) != {"query", "sources"} or not isinstance(request["query"], str) or not isinstance(request["sources"], list):
        raise RuntimeError("Unexpected development request schema")
    if any(set(source) != {"id", "text"} or not all(isinstance(v, str) for v in source.values()) for source in request["sources"]):
        raise RuntimeError("Unexpected source schema")
    return request


def diagnostic_cases(spec: dict, tool_format: bool) -> list[dict]:
    if not tool_format:
        return spec["cases"]
    return [dict(case, id=case["id"] + "--" + variant, base_case_id=case["id"], tool_format=variant)
            for variant in ("original_wrapped_json", "publisher_flat_dict")
            for case in spec["cases"] if case["kind"] == "tool"]


def template_tools(spec: dict, case: dict) -> list:
    if case.get("tool_format") == "publisher_flat_dict":
        return [tool["function"] for tool in spec["tools"]]
    return [json.dumps(tool, ensure_ascii=False) for tool in spec["tools"]]


def strict_json(raw: str):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("Duplicate JSON key")
            obj[key] = value
        return obj
    def reject_constant(value):
        raise ValueError("Non-finite JSON number")
    return json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)


def inspect_answer(answer: dict, request: dict, qa: bool) -> dict:
    keys = {"answer", "citations", "insufficient_evidence"} if qa else {"answer", "citations"}
    if not isinstance(answer, dict) or set(answer) != keys:
        raise ValueError("Unexpected answer keys")
    if not isinstance(answer["answer"], str) or not answer["answer"].strip() or not isinstance(answer["citations"], list):
        raise ValueError("Answer must have nonempty text and a citation list")
    if qa and type(answer["insufficient_evidence"]) is not bool:
        raise ValueError("insufficient_evidence must be a boolean")
    sources = {s["id"]: s["text"] for s in request["sources"]}
    quotes = []
    for citation in answer["citations"]:
        if not isinstance(citation, dict) or set(citation) != {"source_id", "quote"}:
            raise ValueError("Unexpected citation schema")
        source_id, quote = citation["source_id"], citation["quote"]
        if not isinstance(source_id, str) or not isinstance(quote, str) or not quote.strip():
            raise ValueError("Citation needs source_id and nonempty quote strings")
        if source_id not in sources or quote not in sources[source_id]:
            raise ValueError("Citation is not an exact substring of the identified source")
        quotes.append({"source_id": source_id, "literal_match": True})
    if not answer["citations"] and not (qa and answer["insufficient_evidence"]):
        raise ValueError("An asserted answer requires a supporting citation")
    return {"parsed": answer, "literal_quote_checks": quotes,
            "semantic_quality": "not_reviewed", "claim_citation_entailment_verified": False}


def inspect_output(raw: str, case: dict, request: dict) -> dict:
    if case["kind"] == "qa":
        return inspect_answer(strict_json(raw), request, True)
    match = re.fullmatch(r"\s*<tool_call>\s*(.*?)\s*</tool_call>\s*", raw, re.DOTALL)
    if not match:
        raise ValueError("Expected exactly one inert tool_call with no extra text")
    proposal = strict_json(match[1])
    if not isinstance(proposal, dict) or set(proposal) != {"name", "arguments"} or not isinstance(proposal["arguments"], dict):
        raise ValueError("Unexpected tool proposal schema")
    if proposal["name"] == "answer":
        result = inspect_answer(proposal["arguments"], request, False)
    elif proposal["name"] == "search_public_web":
        args = proposal["arguments"]
        if set(args) != {"query"} or not isinstance(args["query"], str) or not args["query"].strip():
            raise ValueError("Search proposal needs one nonempty query")
        result = {"semantic_quality": "not_reviewed"}
    else:
        raise ValueError("Tool name is not in the inert allowlist")
    result.update(parsed=proposal, tool_executed=False, network_requests_from_proposal=0)
    return result


def self_test() -> None:
    spec = load_spec()
    assert diagnostic_cases(spec, False) == spec["cases"]
    variants = diagnostic_cases(spec, True)
    assert len(variants) == 4 and all(case["kind"] == "tool" for case in variants)
    assert request_for(variants[0]) == request_for(variants[2])
    assert request_for(variants[1]) == request_for(variants[3])
    assert all(isinstance(tool, str) for tool in template_tools(spec, variants[0]))
    assert template_tools(spec, variants[2]) == [json.loads(tool)["function"] for tool in template_tools(spec, variants[0])]
    for case in spec["cases"]:
        request_for(case)
    request = {"query": "Which label?", "sources": [{"id": "demo", "text": "The fictional label is Cedar."}]}
    valid = {"answer": "Cedar", "citations": [{"source_id": "demo", "quote": "label is Cedar"}], "insufficient_evidence": False}
    assert inspect_output(json.dumps(valid), {"kind": "qa"}, request)["semantic_quality"] == "not_reviewed"
    invalid = ["```json\n" + json.dumps(valid) + "\n```", json.dumps(valid).replace("label is Cedar", "label is Ash"),
               '{"answer":"a","answer":"b","citations":[],"insufficient_evidence":true}',
               '{"answer":"Cedar","citations":[],"insufficient_evidence":false}']
    for raw in invalid:
        try:
            inspect_output(raw, {"kind": "qa"}, request)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed or unsupported quotation was accepted")
    proposal = '<tool_call>{"name":"search_public_web","arguments":{"query":"fictional search"}}</tool_call>'
    assert inspect_output(proposal, {"kind": "tool"}, request)["tool_executed"] is False
    for raw in (proposal + proposal, proposal.replace("search_public_web", "send_email")):
        try:
            inspect_output(raw, {"kind": "tool"}, request)
        except ValueError:
            pass
        else:
            raise AssertionError("Unexpected tool proposal accepted")
    with tempfile.TemporaryDirectory() as directory:
        file = Path(directory) / "test"
        file.write_bytes(b"fictional hash probe")
        verify(file, {"bytes": file.stat().st_size, "sha256": sha(file)})
        try:
            verify(file, {"bytes": file.stat().st_size, "sha256": "0" * 64})
        except RuntimeError:
            pass
        else:
            raise AssertionError("Incorrect source digest accepted")
    print("Pinned source declarations, seven fixtures, strict JSON/quotes and inert tool checks passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--tool-format-diagnostic", action="store_true",
                        help="Separate four-run comparison of original wrapped schemas and publisher flat dictionaries")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.work or not args.output or os.environ.get("GITHUB_ACTIONS") != "true":
        parser.error("Model execution is CI-only and requires work/output directories; use --self-test locally")
    started = time.monotonic()
    work, output = args.work.resolve(), args.output.resolve()
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / ("smollm3-tool-format.json" if args.tool_format_diagnostic else "smollm3-reference.json")
    report = {"status": "preparing", "scope": "Seven fictional development probes; not a representative evaluation.",
              "quality_accepted": False, "release_authorized": False, "semantic_review": "not_reviewed",
              "tool_execution_available": False, "cases": [], "source_commit": os.environ.get("GITHUB_SHA")}
    write_json(report_path, report)
    try:
        spec = load_spec()
        cases = diagnostic_cases(spec, args.tool_format_diagnostic)
        if args.tool_format_diagnostic:
            report.update(scope="Two unchanged tool-selection development requests across two schema renderings; four runs.",
                          tool_format_diagnostic=True,
                          independent_variable="Original OpenAI-style wrapper JSON strings versus publisher flat function dictionaries.",
                          unchanged="Same model/runtime, user requests, system text, function names/descriptions/parameters, fixed date and decoding.",
                          original_evidence_preserved=True)
        report.update(repository=REPOSITORY, revision=REVISION, source_files=spec["files"],
                      publisher_license=spec["publisher_license"], provenance_scope=spec["provenance_scope"],
                      probe_spec_sha256=canonical_sha(spec), runtime_script_sha256=sha(Path(__file__)),
                      limits=spec["limits"], frozen_template_date=spec["frozen_template_date"])
        write_json(report_path, report)
        for name, pin in spec["files"].items():
            download(name, pin, work)
        config = json.loads((work / "config.json").read_text(encoding="utf-8"))
        index = json.loads((work / "model.safetensors.index.json").read_text(encoding="utf-8"))
        if (config.get("model_type") != "smollm3" or config.get("architectures") != ["SmolLM3ForCausalLM"]
                or "auto_map" in config or config.get("torch_dtype") != "bfloat16"
                or set(index["weight_map"].values()) != {name for name in spec["files"] if name.endswith(".safetensors")}):
            raise RuntimeError("Unexpected standard architecture or checkpoint shard index")
        versions = {name: importlib.metadata.version(name) for name in spec["versions"]}
        if versions != spec["versions"]:
            raise RuntimeError(f"Pinned runtime version mismatch: {versions}")
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                          HF_HUB_DISABLE_IMPLICIT_TOKEN="1", TOKENIZERS_PARALLELISM="false")
        import resource
        import torch
        from transformers import AutoTokenizer, SmolLM3ForCausalLM, StoppingCriteria, StoppingCriteriaList, set_seed

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        set_seed(42)
        memory = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
        cpu = {}
        for line in Path("/proc/cpuinfo").read_text().split("\n\n", 1)[0].splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                if key.strip() in ("model name", "vendor_id", "flags", "Features", "CPU architecture"):
                    cpu[key.strip()] = value.strip()
        report["hardware"] = {"machine": platform.machine(), "logical_cpus": os.cpu_count(),
                              "cpu": cpu, "mem_total_kib": memory["MemTotal"],
                              "mem_available_before_load_kib": memory["MemAvailable"]}
        write_json(report_path, report)
        if memory["MemAvailable"] < 9 * 1024 * 1024:
            raise RuntimeError("Less than 9 GiB RAM available for the bounded original BF16 checkpoint")
        modules = ("transformers.models.smollm3.modeling_smollm3", "transformers.generation.utils",
                   "transformers.tokenization_utils_base", "transformers.utils.chat_template_utils")
        runtime_hashes = {name: sha(Path(importlib.import_module(name).__file__)) for name in modules}
        tokenizer = AutoTokenizer.from_pretrained(str(work), local_files_only=True, trust_remote_code=False, use_fast=True)
        if not tokenizer.is_fast or tokenizer.chat_template != (work / "chat_template.jinja").read_text(encoding="utf-8"):
            raise RuntimeError("Loaded tokenizer differs from the pinned publisher chat template")
        load_started = time.monotonic()
        model = SmolLM3ForCausalLM.from_pretrained(str(work), local_files_only=True, trust_remote_code=False,
                   use_safetensors=True, dtype=torch.bfloat16, attn_implementation="sdpa")
        model.eval()
        if any(p.device.type != "cpu" or p.dtype != torch.bfloat16 for p in model.parameters()):
            raise RuntimeError("Original BF16 CPU execution changed unexpectedly")
        report.update(status="running", versions=versions, python=platform.python_version(),
                      runtime_file_sha256=runtime_hashes, architecture_config=config,
                      inference_dtype="original bfloat16; no quantization or FP32 fallback", attention="sdpa",
                      parameter_count=sum(p.numel() for p in model.parameters()),
                      load_seconds=round(time.monotonic() - load_started, 3), mem_available_before_load_kib=memory["MemAvailable"],
                      generation="greedy, fresh generate call per case, publisher template with thinking disabled",
                      trust_remote_code=False)
        write_json(report_path, report)
        fixed_date = dt.datetime.fromisoformat(spec["frozen_template_date"])
        total_deadline = min(time.monotonic() + 1200, started + 22 * 60)

        class TimeLimit(StoppingCriteria):
            def __init__(self, deadline):
                self.deadline, self.timed_out = deadline, False
            def __call__(self, input_ids, scores, **kwargs):
                self.timed_out = time.monotonic() >= self.deadline
                return torch.full((input_ids.shape[0],), self.timed_out, dtype=torch.bool, device=input_ids.device)

        def hard_timeout(signum, frame):
            raise TimeoutError("Bounded CPU generation deadline exceeded")

        old_handler = signal.signal(signal.SIGALRM, hard_timeout)
        try:
            for case in cases:
                request = request_for(case)
                result = {"id": case["id"], "kind": case["kind"], "request_sha256": canonical_sha(request),
                          "review_criteria": case["review"], "semantic_quality": "not_reviewed",
                          "authorized_actions": [], "tool_executed": False}
                if args.tool_format_diagnostic:
                    result.update(base_case_id=case["base_case_id"], tool_format=case["tool_format"])
                if time.monotonic() >= total_deadline:
                    result.update(status="not_run_budget_exhausted")
                    report["cases"].append(result)
                    write_json(report_path, report)
                    continue
                system = spec["qa_system"] if case["kind"] == "qa" else spec["tool_system"]
                messages = [{"role": "system", "content": system},
                            {"role": "user", "content": json.dumps(request, ensure_ascii=False)}]
                template_args = {"enable_thinking": False, "strftime_now": lambda fmt: fixed_date.strftime(fmt)}
                if case["kind"] == "tool":
                    # The publisher variant passes flat dictionaries directly;
                    # default execution preserves the original wrapped JSON strings.
                    template_args["xml_tools"] = template_tools(spec, case)
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_args)
                encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt", return_token_type_ids=False)
                input_ids = encoded["input_ids"][0].tolist()
                direct_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **template_args)
                if input_ids != direct_ids or len(input_ids) + 384 > 4096 or "Reasoning Mode: /no_think" not in prompt:
                    raise RuntimeError("Publisher template, disabled thinking, or 4096-token budget mismatch")
                if fixed_date.strftime("%d %B %Y") not in prompt:
                    raise RuntimeError("Publisher date was not frozen consistently")
                write_json(output / f"{case['id']}.input.json", {"request": request, "messages": messages, "input_token_ids": input_ids})
                (output / f"{case['id']}.prompt.txt").write_bytes(prompt.encode("utf-8"))
                result.update(input_tokens=len(input_ids), prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest())
                case_started = time.monotonic()
                limit = TimeLimit(min(case_started + 240, total_deadline))
                signal.alarm(max(1, int(limit.deadline - case_started)) + 15)
                try:
                    set_seed(42)
                    with torch.inference_mode():
                        sequence = model.generate(**encoded, max_new_tokens=384, do_sample=False, num_beams=1,
                            temperature=1.0, top_p=1.0, top_k=0, use_cache=True, eos_token_id=128012,
                            pad_token_id=128004, stopping_criteria=StoppingCriteriaList([limit]))
                    generated = sequence[0, len(input_ids):].tolist()
                    raw = tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    (output / f"{case['id']}.completion.txt").write_bytes(raw.encode("utf-8"))
                    write_json(output / f"{case['id']}.generated-token-ids.json", generated)
                    result.update(generated_tokens=len(generated), raw_completion=raw,
                                  stopped_on_eos=bool(generated and generated[-1] == 128012))
                    content_ids = generated[:-1] if generated and generated[-1] == 128012 else generated
                    content = tokenizer.decode(content_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    if limit.timed_out:
                        result.update(status="runtime_timeout")
                    elif not result["stopped_on_eos"]:
                        result.update(status="output_incomplete", reason="No publisher EOS before output limit")
                    else:
                        result["status"] = "completed"
                        try:
                            result.update(inspect_output(content, case, request), structural_checks_passed=True)
                        except ValueError as error:
                            result.update(structural_checks_passed=False, structural_error=str(error))
                except (TimeoutError, RuntimeError) as error:
                    result.update(status="runtime_error", error=f"{type(error).__name__}: {error}")
                finally:
                    signal.alarm(0)
                result.update(elapsed_seconds=round(time.monotonic() - case_started, 3),
                              process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                report["cases"].append(result)
                write_json(report_path, report)
                print(json.dumps(result, ensure_ascii=False), flush=True)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        report.update(status="completed", elapsed_seconds=round(time.monotonic() - started, 3),
                      all_cases_completed=len(report["cases"]) == len(cases) and all(c["status"] == "completed" for c in report["cases"]))
        write_json(report_path, report)
        # Success means diagnostic execution finished, never semantic acceptance.
        return 0 if report["all_cases_completed"] else 2
    except Exception as error:
        report.update(status="setup_or_reference_error", error=f"{type(error).__name__}: {error}")
        write_json(report_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
