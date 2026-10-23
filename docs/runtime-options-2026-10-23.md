# Runtime provisioning evidence — October 23, 2026

An initial desktop quantization attempt was stopped by a memory guard before
completion. That output is incomplete and must not be registered, loaded or
published. Do not lower the memory guard to force a completion.

The official b11669 release has an Android arm64 CPU archive:

- Release source commit: `620d2225cc436ab0b001e3200e590e3b286ef9ab`.
- Filename: `llama-b11669-bin-android-arm64.tar.gz`.
- Size: 75,478,449 bytes.
- Published SHA-256: `640f870f795b3b2939ed40d1a31ce529ecc4865abfc5d0f5153168bc4b82e5c8`.
- URL: https://github.com/ggml-org/llama.cpp/releases/download/b11669/llama-b11669-bin-android-arm64.tar.gz
- Metadata: https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b11669
- Android instructions: https://github.com/ggml-org/llama.cpp/blob/b11669/docs/android.md

This is a provisioning candidate, not an installed or tested phone runtime. First
verify the phone's ABI and available memory, the archive hash and its actual tool
contents. Prefer the generic arm64 CPU build until the device's chipset and driver
support are confirmed. Phone-side quantization or a better-memory host can retain
the official 350M source lineage. Any resulting GGUF needs a recorded exact hash,
size, quantizer version/options and inference/quality checks before registration.

A Hugging Face API search for Pleias-RAG-350M and GGUF found no published quantized
350M RAG GGUF at review time. Two tempting substitutes were investigated and are
not approved for this harness:

- `brendanddev/Pleias-RAG-1B-Q4_K_M-GGUF` at revision
  `cba15b678ec012ecb352ac91865379f5f63b5be1`: 744,851,264 bytes, SHA-256
  `00e5a805e8e4ca4f573b3ed003f4157c42dee5475918d9a1a413e946ef5c0e56`.
  Its card describes custom tokenizer changes and a prompt format that differs
  from the original Pleias RAG protocol. It is larger and adds compatibility and
  provenance uncertainty; do not substitute it silently.
- Official `PleIAs/Pleias-SLM-RAG` at revision
  `454280a16b0afce3e59cfd45870ffad9fb3888d4` contains a 343,655,872-byte Q8 model
  called Pleias Redline, a Baguettotron fine-tune. It uses ChatML and a different
  source/citation protocol. It is not a quantized copy of Pleias-RAG-350M.

Read-only metadata and model cards were inspected. No alternate model or Android
runtime was downloaded by this review.
