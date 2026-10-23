# Krummholz LLMs [ALPHA]

[![Offline harness checks](https://github.com/hayvenmjackson/krummholz-llms-alpha/actions/workflows/harness-checks.yml/badge.svg)](https://github.com/hayvenmjackson/krummholz-llms-alpha/actions/workflows/harness-checks.yml)

Source and development evidence for Acadia, Katahdin and Atlantic, the Krummholz
LLM offerings. Initial implementation is underway; device testing is the next
milestone.

## Android builds

No LLM Android APK has been built yet. Verified APKs will be attached to GitHub
Releases with their version, checksum and testing status.

## Current construction milestone

Status: a local inference/evaluation harness around unchanged upstream model weights.
This is not a Krummholz-trained model, a phone benchmark, or a production service.
No client data, cloud inference, external tools or paid compute are used here.

The first quality baseline is Pleias-RAG-1B. Its smaller 350M sibling is the first
runtime smoke test. The 24 authored fictional cases cover Acadia, Katahdin and
Atlantic; they are deliberately small plumbing checks, not representative release
evaluations or a client-ready synthetic dataset.

## Run offline tests

From this directory, using Python 3.10 or later:

    python -m unittest discover -s tests -v
    python -m krummholz_llm.smoke --dry-run --output artifacts/dry-run.json
    python -m krummholz_llm.runner --dry-run --profile acadia --input data/acadia/example.json

Python's standard library is sufficient. Nothing is installed by these commands.
Dry runs do not invoke a model, runtime process, network or external service.
GitHub Actions repeats these checks on each push and pull request. The linked
workflow logs provide a public record of the result for each source revision.

## Run a baseline only after memory and provisioning checks

The provisioned b11669 runtime uses llama-completion.exe for raw completion.
The newer llama-cli.exe has a different chat/server path and is intentionally
rejected. Models must match the exact sizes and hashes in config/models.json.
Provisioning and any quantization happen separately; this harness never downloads.

    python -m krummholz_llm.runner --profile acadia --input data/acadia/example.json --engine runtime/llama-b11669/llama-completion.exe --model models/350m_test.gguf --context 512 --max-output-tokens 64 --threads 2 --batch-size 16 --timeout 60

On a memory-constrained machine, begin with one bounded inference case. Passing
unit tests does not establish available inference memory. A separately verified,
derived Q4_K_M artifact must be added to the model manifest before use.
Quantization changes weights numerically;
it is not task training and needs its own quality comparison. The first tiny
64-token run may truncate before a complete answer; output rejection is an honest
failure, not a reason to display unfinished reasoning.

On a suitable machine, after the bounded first run has been checked:

    python -m krummholz_llm.smoke --engine runtime/llama-b11669/llama-completion.exe --model models/350m_test.gguf --output artifacts/smoke-350m.json --limit 24 --context 1024 --max-output-tokens 128

Use the accepted 1B model path with the same cases for the quality baseline.
A limited run can use --limit 1. Runtime threads, batch size, context, output
length and timeout are bounded. Defaults are two CPU threads, batch/microbatch 16,
1024 context tokens and 128 output tokens.

## What the boundaries do

- Requests contain only query and sources. There is no tool execution endpoint.
- Each source ID must belong to the selected profile namespace. This helps catch
  accidental profile mixing; it is NOT production tenant authentication.
- Reserved Pleias tokens, unsupported fields, control characters, excessive source
  counts and oversized prompts are rejected before inference.
- The prompt uses the publisher's query/source protocol, not a generic chat template.
- A conservative UTF-8 byte bound plus output/reserve budgets is used for the
  pinned byte-level BPE checkpoints. It overestimates token use. It is not an
  arbitrary-model tokenizer or an invitation to extend the 4096-token context.
- The local subprocess receives a restricted environment and no shell, no model
  URL and no credentials. This is an application boundary, not an OS network sandbox.
- The raw completion process is invoked with special-token output, prompt display
  disabled, no escape rewriting, no warmup and no context shift.
- Only a complete final-answer block is returned. Intermediate reasoning is not
  exposed in the result. Malformed citations, nonexistent source references and
  quotes absent from their cited source reject the output.
- Literal quote matching does not establish that the full answer is true or that a
  quote entails a claim. Every accepted parse is marked needs_human_review.
- There is no self-reported probability threshold or claim of universal safety.

## Data retention and review

Inference uses a temporary local prompt file and removes it afterward, including
on normal errors/timeouts. This is not a secure-erasure guarantee. Responses are
printed to the invoking terminal; they are saved to a report only when an output
path is explicitly supplied. Smoke reports contain fictional data.

Do not enter client information into this baseline. The host terminal, crash
reporting, runtime behavior and operating-system caches have not been audited as
a production no-retention system. Future HTML interfaces must escape all model
text, even when the citation parser accepts it.

If a case fails, fix a feasible code defect and rerun the relevant test. Do not
paper over unverified model quality, physical device constraints or third-party
requirements with an optimistic status.

## Evidence and project decisions

See config/models.json for exact upstream revisions, hashes and provenance limits,
config/profiles.json for the intended profile boundaries, and
docs/ADR-001-local-first-baseline.md for implementation scope and release gates.
Models and runtime binaries are separate artifacts, ignored by Git.

For a paired Android device, docs/android-first-run.md contains the staged manual
procedure. The external_io module can export a bounded raw prompt and parse a
captured completion locally. It never invokes ADB, provisions files or runs a
model, and its parser does not attest the provenance of an imported capture.

This milestone does not include Krummholz training, an S26 performance benchmark,
a production deployment, or autonomous actions.
