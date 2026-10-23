"""Run an allowlisted, hashed Pleias GGUF with local llama-completion (no server)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any

from .protocol import InputError, OutputError, build_prompt, parse_answer, validate_request

ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def runtime_environment() -> dict[str, str]:
    """Do not let ambient llama/HF/cloud settings change this local-only invocation."""
    allowed = {
        "PATH", "SYSTEMROOT", "WINDIR", "SYSTEMDRIVE", "TEMP", "TMP",
        "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "PATHEXT",
        "LANG", "LC_ALL", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH",
    }
    result = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    result["HF_HUB_OFFLINE"] = "1"
    result["TRANSFORMERS_OFFLINE"] = "1"
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(path: Path) -> dict[str, Any]:
    manifest = json.loads((ROOT / "config" / "models.json").read_text(encoding="utf-8"))
    if not path.is_file():
        raise InputError("model must be an existing local GGUF file; downloads are not implemented")
    size = path.stat().st_size
    candidates = [item for item in manifest["artifacts"] if item["bytes"] == size]
    if not candidates:
        raise InputError("model file size does not match an approved baseline artifact")
    digest = file_sha256(path)
    matches = [item for item in candidates if item["sha256"] == digest]
    if len(matches) != 1:
        raise InputError("model SHA-256 does not match an approved baseline artifact")
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise InputError("model does not have a GGUF header")
    return matches[0]


def verify_runtime(path: Path) -> None:
    if not path.is_file():
        raise InputError("engine must be an existing local executable")
    if path.name.lower() not in {"llama-completion", "llama-completion.exe"}:
        raise InputError("use llama-completion; the current llama-cli uses a different chat/server path")
    # The separate provisioning step is responsible for checking release provenance.
    # This harness never downloads, installs, or invokes a shell.
    result = subprocess.run(
        [str(path), "--help"], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=30, shell=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        env=runtime_environment(),
    )
    help_text = result.stdout + result.stderr
    required = ("--special", "--no-display-prompt", "--no-conversation", "--no-context-shift")
    if result.returncode != 0 or any(flag not in help_text for flag in required):
        raise InputError("runtime help does not confirm required raw-output/context-control flags")


def run_request(
    data: dict[str, Any], profile: str, executable: Path, model: Path,
    context: int = 1024, output_tokens: int = 128, timeout: int = 120,
    threads: int = 2, batch_size: int = 16,
) -> dict[str, Any]:
    request = validate_request(data, profile)
    prompt = build_prompt(request, context, output_tokens)
    if not 1 <= timeout <= 600:
        raise InputError("timeout must be between 1 and 600 seconds")
    if not 1 <= threads <= 8 or not 1 <= batch_size <= 128:
        raise InputError("threads must be 1-8 and batch size must be 1-128")
    # Prompt exists briefly in a local temporary file; it is removed on success/error.
    # Deletion is not a forensic secure-erasure promise. No client data is used in smoke tests.
    with tempfile.TemporaryDirectory(prefix="krummholz-llm-") as temporary:
        prompt_path = Path(temporary) / "prompt.txt"
        prompt_path.write_text(prompt.text, encoding="utf-8")
        command = [
            str(executable), "-m", str(model), "-f", str(prompt_path),
            "-c", str(context), "-n", str(output_tokens), "-ngl", "0",
            "-t", str(threads), "-b", str(batch_size), "-ub", str(batch_size),
            "--temp", "0", "--seed", "42", "--special", "--no-display-prompt",
            "--no-conversation", "--no-context-shift", "--simple-io",
            "--no-escape", "--no-warmup", "--no-perf",
        ]
        start = time.monotonic()
        try:
            completed = subprocess.run(
                command, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=timeout, shell=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                env=runtime_environment(),
            )
        except subprocess.TimeoutExpired:
            return {"status": "runtime_timeout", "authorized_actions": [],
                    "elapsed_seconds": round(time.monotonic() - start, 3)}
        elapsed = round(time.monotonic() - start, 3)
    if completed.returncode != 0:
        # Do not echo stderr: runtime diagnostics could include prompt/source content.
        return {"status": "runtime_error", "exit_code": completed.returncode,
                "elapsed_seconds": elapsed, "authorized_actions": []}
    try:
        result = parse_answer(ANSI.sub("", completed.stdout), request)
    except OutputError as exc:
        result = {"status": "output_rejected", "reason": str(exc), "authorized_actions": []}
    result["elapsed_seconds"] = elapsed
    result["budget"] = {
        "context_tokens": context, "max_output_tokens": output_tokens,
        "prompt_utf8_byte_upper_bound": prompt.utf8_byte_upper_bound,
        "special_token_reserve": prompt.reserve_tokens,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=("acadia", "katahdin", "atlantic"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--engine", "--llama-cli", dest="engine", type=Path,
                        help="Path to llama-completion.exe (legacy option name accepted)")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, help="Explicitly save output locally; omitted by default")
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="Validate/framing only; no runtime or weights")
    args = parser.parse_args()
    try:
        data = json.loads(args.input.read_text(encoding="utf-8-sig"))
        # Reject bad requests before loading any model or starting a process.
        prompt = build_prompt(validate_request(data, args.profile), args.context, args.max_output_tokens)
        if args.dry_run:
            result = {"status": "dry_run_validated", "profile": args.profile,
                      "prompt_utf8_byte_upper_bound": prompt.utf8_byte_upper_bound,
                      "inference_performed": False, "authorized_actions": []}
        else:
            if args.engine is None or args.model is None:
                raise InputError("--engine and --model are required for inference")
            executable, model = args.engine.resolve(), args.model.resolve()
            baseline = verify_model(model)
            verify_runtime(executable)
            result = run_request(data, args.profile, executable, model, args.context,
                                 args.max_output_tokens, args.timeout, args.threads, args.batch_size)
            result["baseline"] = baseline["id"]
        result["weights_status"] = "unchanged_upstream_baseline_not_krummholz_trained"
    except (InputError, OSError, ValueError, subprocess.SubprocessError) as exc:
        result = {"status": "request_rejected", "reason": str(exc), "authorized_actions": []}
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if result["status"] in {"needs_human_review", "dry_run_validated"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
