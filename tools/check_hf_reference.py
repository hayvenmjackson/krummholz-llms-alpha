"""Compare original pinned Pleias safetensors with the unchanged fictional smoke gate.

This is a CI-only CPU diagnostic, not model approval. It retains complete fictional
completions for semantic review and never reads user prompts, documents or credentials.
Official interfaces: https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/llama
and https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from krummholz_llm.protocol import OutputError, build_prompt, parse_answer, validate_request

EOS = "<|end_of_text|>"
CONTEXT = 4096
NEW_TOKENS = 1024
SEED = 42
EXPECTATIONS = {"acadia": "30", "katahdin": "90", "atlantic": "paper lantern"}
VERSIONS = {"torch": "2.9.1+cpu", "transformers": "4.57.1", "tokenizers": "0.22.2",
            "safetensors": "0.6.2", "huggingface-hub": "0.36.0"}
# Small-file bytes were verified directly at each immutable publisher revision.
# The tokenizer hash also matches the independent GGUF framing diagnostic.
SHARED_FILES = {
    "tokenizer.json": (2777269, "dd7acf61196c61741cf95cc0f0a2b920443e44094438d325ab6e65a048fe8be4"),
    "tokenizer_config.json": (4897, "644736bd4531d14d845a0999ff13acd39ee9aa98b5f353c771aab04af5c01fb4"),
    "special_tokens_map.json": (535, "a93a1d2a4eae63e912e314d9a65cdc035224f53a440834fab20303f1fc2f36bd"),
    "generation_config.json": (111, "63b76107a660687d9591beb7db277a7fb50862f00e805b19c021a266bbaef051"),
}
MODELS = {
    "350m": {"artifact_id": "pleias-rag-350m-upstream-bf16-gguf",
             "repository": "PleIAs/Pleias-RAG-350M",
             "revision": "db001b29a33532583d14979c3c38ef04b6d59352",
             "weight_bytes": 706875632,
             "config": (654, "591da78c94e3662c2bc2cf00fd8410603e33ee5f144cc4d8c5dca637ea463314")},
    "1b": {"artifact_id": "pleias-rag-1b-upstream-bf16-gguf",
           "repository": "PleIAs/Pleias-RAG-1B",
           "revision": "b8c3f1ab2c13e1d44305eee46662310cdbfd72a1",
           "weight_bytes": 2390960584,
           "config": (654, "1d16d6cc2be6f6f749d5fb55af97094ae7ac977f13d1bd8b2d6e652872aeadef")},
}


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify_file(path: Path, size: int, digest: str) -> None:
    if not path.is_file() or path.stat().st_size != size or sha(path) != digest:
        raise RuntimeError(f"Pinned source size/hash mismatch: {path.name}")


def download(url: str, path: Path, size: int, digest: str) -> None:
    if path.exists():
        verify_file(path, size, digest)
        return
    partial = path.with_suffix(path.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "Krummholz-HF-reference/0.1"})
    created = False
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("xb") as output:
            created = True
            count = 0
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > size:
                    raise RuntimeError("Download exceeded its pinned byte size")
                output.write(chunk)
        verify_file(partial, size, digest)
        partial.replace(path)
    finally:
        if created and partial.exists():
            partial.unlink()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_pins(model: str) -> tuple[dict, dict[str, tuple[int, str]]]:
    selected = MODELS[model]
    manifest = json.loads((ROOT / "config/models.json").read_text(encoding="utf-8"))
    matches = [item for item in manifest["artifacts"] if item["id"] == selected["artifact_id"]]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one configured source artifact")
    source = matches[0]
    if source["source_repository"] != selected["repository"] or source["source_revision"] != selected["revision"]:
        raise RuntimeError("Configured publisher changed; re-verify every metadata pin")
    files = dict(SHARED_FILES)
    files["config.json"] = selected["config"]
    files["model.safetensors"] = (selected["weight_bytes"], source["source_safetensors_sha256"])
    return source, files


def publisher_prompt(framed_prompt: str) -> str:
    # Python/Android direct-runtime prompts now include one initial EOS. The original
    # HF tokenizer adds this itself. Remove exactly one, preserve the final newline,
    # and reject duplicate framing rather than silently normalizing it away.
    raw = framed_prompt[len(EOS):] if framed_prompt.startswith(EOS) else framed_prompt
    if raw.startswith(EOS) or not raw.startswith("<|query_start|>") or not raw.endswith("<|language_start|>\n"):
        raise RuntimeError("Unexpected source prompt framing")
    return raw


def self_test() -> None:
    for name in MODELS:
        _, files = source_pins(name)
        assert len(files) == 6 and files["model.safetensors"][0] > 0
    raw = "<|query_start|>Fictional probe?<|query_end|>\n<|language_start|>\n"
    assert publisher_prompt(raw) == raw
    assert publisher_prompt(EOS + raw) == raw
    for invalid in (EOS + EOS + raw, raw[:-1]):
        try:
            publisher_prompt(invalid)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Invalid framing was accepted")
    with tempfile.TemporaryDirectory() as temporary:
        candidate = Path(temporary) / "fixture"
        candidate.write_bytes(b"fictional source verification probe")
        verify_file(candidate, candidate.stat().st_size, sha(candidate))
        for size, digest in ((candidate.stat().st_size + 1, sha(candidate)), (candidate.stat().st_size, "0" * 64)):
            try:
                verify_file(candidate, size, digest)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Invalid source pin was accepted")
    print("source selection, exact hashes and single-EOS framing self-checks passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), default="1b")
    parser.add_argument("--work", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case-timeout", type=int, default=480)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.work or not args.output or not 60 <= args.case_timeout <= 600:
        parser.error("work/output are required and case-timeout must be 60-600 seconds")
    if os.environ.get("GITHUB_ACTIONS") != "true":
        parser.error("Large checkpoint execution is CI-only; use --self-test locally")
    work, output = args.work.resolve(), args.output.resolve()
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    report = {"scope": "Three authored fictional software smoke cases, not representative quality validation.",
              "model": args.model, "backend": "original HF safetensors / Transformers / CPU",
              "release_authorized_by_this_diagnostic": False, "cases": [], "status": "preparing"}
    report_path = output / "hf-reference.json"
    write_json(report_path, report)
    try:
        source, files = source_pins(args.model)
        report.update({"source_repository": source["source_repository"], "source_revision": source["source_revision"],
                       "source_safetensors_sha256": source["source_safetensors_sha256"],
                       "source_files": {name: {"bytes": size, "sha256": digest} for name, (size, digest) in files.items()}})
        for name, (size, digest) in files.items():
            url = f"https://huggingface.co/{source['source_repository']}/resolve/{source['source_revision']}/{name}"
            download(url, work / name, size, digest)
        config_data = json.loads((work / "config.json").read_text(encoding="utf-8"))
        if (config_data.get("model_type") != "llama" or config_data.get("architectures") != ["LlamaForCausalLM"]
                or "auto_map" in config_data or config_data.get("max_position_embeddings") != CONTEXT):
            raise RuntimeError("Only the pinned standard 4096-context Llama architecture is allowed")
        versions = {name: importlib.metadata.version(name) for name in VERSIONS}
        if versions != VERSIONS:
            raise RuntimeError(f"Reference dependency versions differ: {versions}")
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                          HF_HUB_DISABLE_IMPLICIT_TOKEN="1", TOKENIZERS_PARALLELISM="false")
        import torch
        from tokenizers import Tokenizer
        from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, StoppingCriteria, StoppingCriteriaList, set_seed

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        set_seed(SEED)
        tokenizer = AutoTokenizer.from_pretrained(str(work), local_files_only=True, trust_remote_code=False, use_fast=True)
        publisher = Tokenizer.from_file(str(work / "tokenizer.json"))
        if not tokenizer.is_fast or tokenizer.convert_tokens_to_ids(EOS) != 2:
            raise RuntimeError("Unexpected publisher tokenizer")
        answer_end = tokenizer.convert_tokens_to_ids("<|answer_end|>")
        if answer_end != 65535:
            raise RuntimeError("Unexpected publisher answer-end token")
        report.update({"versions": versions, "python": platform.python_version(), "architecture_config": config_data,
                       "seed": SEED, "threads": 2, "context_tokens": CONTEXT, "max_new_tokens": NEW_TOKENS,
                       "generation": "greedy, num_beams=1, fresh generate call per fixture",
                       "inference_dtype": "float32 (original BF16 values upcast; no quantization)",
                       "attention": "eager", "case_timeout_seconds": args.case_timeout,
                       "stop_token_ids": [2, answer_end], "trust_remote_code": False})
        write_json(report_path, report)
        load_started = time.monotonic()
        config = LlamaConfig.from_pretrained(str(work), local_files_only=True)
        model = LlamaForCausalLM.from_pretrained(str(work), config=config, local_files_only=True,
                    use_safetensors=True, trust_remote_code=False, dtype=torch.float32, attn_implementation="eager")
        model.eval()
        if any(parameter.device.type != "cpu" for parameter in model.parameters()):
            raise RuntimeError("Reference execution must remain on CPU")
        report.update({"load_seconds": round(time.monotonic() - load_started, 3),
                       "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "status": "running"})
        write_json(report_path, report)

        class TimeLimit(StoppingCriteria):
            def __init__(self, seconds: int):
                self.deadline = time.monotonic() + seconds
                self.timed_out = False
            def __call__(self, input_ids, scores, **kwargs):
                self.timed_out = time.monotonic() >= self.deadline
                return torch.full((input_ids.shape[0],), self.timed_out, dtype=torch.bool, device=input_ids.device)

        def hard_timeout(signum, frame):
            raise TimeoutError("The HF reference exceeded its hard per-case CPU deadline")

        previous_handler = signal.signal(signal.SIGALRM, hard_timeout)
        try:
            for profile, expected in EXPECTATIONS.items():
                data = json.loads((ROOT / "data" / profile / "example.json").read_text(encoding="utf-8"))
                request = validate_request(data, profile)
                raw_prompt = publisher_prompt(build_prompt(request, CONTEXT, NEW_TOKENS).text)
                encoded = tokenizer(raw_prompt, add_special_tokens=True, return_tensors="pt", return_token_type_ids=False)
                ids = encoded["input_ids"][0].tolist()
                reference_ids = publisher.encode(raw_prompt, add_special_tokens=True).ids
                explicit_ids = tokenizer.encode(EOS + raw_prompt, add_special_tokens=False)
                if ids != reference_ids or ids != explicit_ids or ids[:2] == [2, 2] or ids[0] != 2:
                    raise RuntimeError("HF input differs from exact publisher single-EOS tokenization")
                if len(ids) + NEW_TOKENS > CONTEXT:
                    raise RuntimeError("Actual tokens exceed the fixed 4096-context budget")
                (output / f"{profile}.prompt.txt").write_bytes(raw_prompt.encode("utf-8"))
                write_json(output / f"{profile}.input.json", {"request": data, "input_token_ids": ids,
                           "publisher_tokenizer_parity": True, "initial_eos_count": 1})
                case = {"profile": profile, "input_tokens": len(ids), "publisher_tokenizer_parity": True}
                started = time.monotonic()
                limit = TimeLimit(args.case_timeout)
                signal.alarm(args.case_timeout + 30)
                try:
                    set_seed(SEED)
                    with torch.inference_mode():
                        sequence = model.generate(**encoded, max_new_tokens=NEW_TOKENS, do_sample=False,
                            num_beams=1, use_cache=True, eos_token_id=[2, answer_end], pad_token_id=3,
                            stopping_criteria=StoppingCriteriaList([limit]))
                    generated = sequence[0, len(ids):].tolist()
                    raw = tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    (output / f"{profile}.completion.txt").write_bytes(raw.encode("utf-8"))
                    write_json(output / f"{profile}.generated-token-ids.json", {"generated_token_ids": generated})
                    case["generated_tokens"] = len(generated)
                    if limit.timed_out:
                        case.update(status="runtime_timeout", reason="Bounded per-case generation deadline")
                    else:
                        try:
                            case.update(parse_answer(raw, request))
                        except OutputError as error:
                            case.update(status="output_rejected", reason=str(error))
                except (TimeoutError, RuntimeError) as error:
                    case.update(status="runtime_error", reason=str(error))
                finally:
                    signal.alarm(0)
                case.update(elapsed_seconds=round(time.monotonic() - started, 3), authorized_actions=[])
                case["expected_fixture_text_present"] = expected in case.get("answer", "").lower()
                case["existing_smoke_gate_passed"] = bool(case.get("status") == "needs_human_review"
                            and case["expected_fixture_text_present"] and case.get("citations"))
                report["cases"].append(case)
                write_json(report_path, report)
                print(json.dumps({key: case.get(key) for key in ("profile", "status", "generated_tokens",
                                      "elapsed_seconds", "existing_smoke_gate_passed")}))
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
        report["all_existing_smokes_passed"] = len(report["cases"]) == 3 and all(case["existing_smoke_gate_passed"] for case in report["cases"])
        report["status"] = "completed"
        write_json(report_path, report)
        return 0 if report["all_existing_smokes_passed"] else 2
    except Exception as error:
        report.update(status="setup_or_reference_error", error=f"{type(error).__name__}: {error}")
        write_json(report_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
