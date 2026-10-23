"""Synthetic local smoke cases. This does not measure production readiness."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .protocol import InputError, build_prompt, validate_request
from .runner import ROOT, run_request, verify_model, verify_runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", "--llama-cli", dest="engine", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "smoke_cases.json")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= 24:
        parser.error("--limit must be from 1 to 24")
    if args.dry_run:
        executable, model, baseline = None, None, {"id": "not_loaded_dry_run_only"}
    else:
        if args.engine is None or args.model is None:
            parser.error("--engine and --model are required for inference")
        executable, model = args.engine.resolve(), args.model.resolve()
        baseline = verify_model(model)
        verify_runtime(executable)
    cases = json.loads(args.cases.read_text(encoding="utf-8"))[:args.limit]
    results = []
    for case in cases:
        try:
            prompt = build_prompt(validate_request(case["request"], case["profile"]),
                                  args.context, args.max_output_tokens)
            if args.dry_run:
                result = {"status": "dry_run_validated", "inference_performed": False,
                          "prompt_utf8_byte_upper_bound": prompt.utf8_byte_upper_bound,
                          "authorized_actions": []}
            else:
                result = run_request(case["request"], case["profile"], executable, model,
                                     args.context, args.max_output_tokens, args.timeout,
                                     args.threads, args.batch_size)
        except InputError as exc:
            result = {"status": "input_rejected", "reason": str(exc), "authorized_actions": []}
        results.append({
            "case_id": case["id"], "profile": case["profile"],
            "category": case["category"], "review_instruction": case["review_instruction"],
            "expected_input_rejection": case.get("expect_input_rejection", False),
            "result": result,
        })
        # Progress contains no source or prompt payload.
        print(f'{case["id"]}: {result["status"]}', flush=True)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "synthetic plumbing and baseline smoke test; not a release evaluation",
        "baseline": baseline,
        "weights_status": "unchanged_upstream_baseline_not_krummholz_trained",
        "case_count": len(results),
        "production_quality": "not_assessed",
        "phone_compatibility": "not_assessed_by_this_desktop_run",
        "external_tools": "none",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(results)} cases; human review required.")
    failures = [
        entry for entry in results
        if (entry["result"]["status"] == "input_rejected") != entry["expected_input_rejection"]
        or entry["result"]["status"] in {"runtime_error", "runtime_timeout", "output_rejected"}
    ]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
