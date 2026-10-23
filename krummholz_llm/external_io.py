"""Export a bounded raw prompt or parse locally captured output; never invokes ADB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .protocol import InputError, OutputError, build_prompt, parse_answer, validate_request
from .runner import ANSI


def export_prompt(data, profile, destination, context=1024, output_tokens=128):
    prompt = build_prompt(validate_request(data, profile), context, output_tokens)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(prompt.text.encode("utf-8"))
    return {
        "status": "prompt_exported", "inference_performed": False,
        "profile": profile, "prompt_path": str(destination.resolve()),
        "context_tokens": context, "max_output_tokens": output_tokens,
        "prompt_utf8_byte_upper_bound": prompt.utf8_byte_upper_bound,
        "authorized_actions": [],
    }


def parse_capture(data, profile, generation):
    request = validate_request(data, profile)
    generation = Path(generation)
    if generation.stat().st_size > 2 * 1024 * 1024:
        raise OutputError("captured output exceeds the 2 MiB inspection limit")
    raw = generation.read_text(encoding="utf-8-sig")
    result = parse_answer(ANSI.sub("", raw), request)
    result["provenance_status"] = "external_runtime_and_weights_not_verified_by_parser"
    result["quality_status"] = "parse_and_literal_citations_only_not_a_quality_pass"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("export", "parse"))
    parser.add_argument("--profile", required=True, choices=("acadia", "katahdin", "atlantic"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--prompt-output", type=Path)
    parser.add_argument("--generation", type=Path)
    parser.add_argument("--output", type=Path, help="Optional local JSON result report")
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    args = parser.parse_args()
    try:
        data = json.loads(args.input.read_text(encoding="utf-8-sig"))
        if args.operation == "export":
            if args.prompt_output is None:
                raise InputError("export requires --prompt-output")
            if args.prompt_output.resolve() == args.input.resolve():
                raise InputError("prompt output must not overwrite the request JSON")
            result = export_prompt(data, args.profile, args.prompt_output,
                                   args.context, args.max_output_tokens)
        else:
            if args.generation is None:
                raise InputError("parse requires --generation")
            result = parse_capture(data, args.profile, args.generation)
    except (InputError, OutputError, OSError, ValueError) as exc:
        result = {"status": "rejected", "reason": str(exc), "authorized_actions": []}
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        if args.output.resolve() in {args.input.resolve(),
                                     args.generation.resolve() if args.generation else None,
                                     args.prompt_output.resolve() if args.prompt_output else None}:
            parser.error("JSON output must not overwrite an input, capture or prompt")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if result["status"] in {"prompt_exported", "needs_human_review"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
