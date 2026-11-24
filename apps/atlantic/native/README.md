# Atlantic native host boundary

This is a working, dependency-light C++17 boundary around Krummholz's published
harness. It also exposes a C ABI for Rust, Swift, and Kotlin/JNI integration.
It is **not a bundled llama.cpp runtime or a working model-inference build**.
The model and embedding callbacks still need production implementations.

## What is implemented

Each synchronous request follows this sequence:

1. Validate the complete input, UTF-8, and size limits.
2. Run the pinned input rules with `SafetyGate::before`.
3. Require `EmbeddingJudge::request_verdict` to return `Safe`.
4. Generate privately, applying the pinned `ReasoningBudget` to every token.
5. Run output rules and the embedding reply judge with `SafetyGate::after`.
6. Return only the checked final text or a fixed refusal/status message.

`SafetyGate::before` alone does not run the request embedding judge. This host
adds that mandatory call. There is no raw-generation accessor, UI token stream,
rules-only fallback, or switch that disables a required check. Reasoning remains
private. An unfinished reasoning segment becomes the upstream fixed abstention,
which also goes through the output checks. Missing/invalid embeddings, missing
assets, callback failure, interrupted generation, and malformed output fail
closed. Cancellation withholds all partial text.

Default settings preserve the pinned harness: temperature 0.7 and a 1,024-token
reasoning budget. Host limits additionally cap complete input at 64 KiB, generated
text at 1 MiB, and generation at 4,096 tokens. These are implementation defaults,
not measured device limits or quality recommendations.

## Upstream bundle and attribution

The checked-in headers, rules, documentation, reports and license in
`vendor/krummholz-harness` are unchanged copies from:

- Repository: https://huggingface.co/krummholz-ai/krummholz-harness
- Revision: `30f793a03d7ba5ca0f123f275171f04ac04aab4d`
- License: Apache-2.0, included at `vendor/krummholz-harness/LICENSE`.
- Four public trained head files total about 2.69 MB. They are download
  dependencies fetched from that immutable upstream release, not checked-in
  binary assets. No language-model or embedding backbone weights are included.

`PROVENANCE.json` records the SHA-256 and size of the source files and four binary
dependencies. Run `python3 fetch_heads.py` to acquire only the four exact pinned
public heads, then `python3 verify_vendor.py` to check the complete bundle.
The acquisition script requires no account/key, accepts no source override,
checks exact bytes and digest, and installs each verified file atomically.
It skips an existing file only if both its size and SHA-256 match.
These development checks are not runtime signature verification. The application installer/loader must
verify its signed manifest and every installed asset before passing bundle paths
to this host. Native code and callbacks are trusted application components.

The upstream judge README attributes judge training data to NVIDIA's Aegis AI
Content Safety Dataset 2.0 (CC BY 4.0) and Krummholz prompt sets. Preserve that
attribution in distributions. The intended embedding backbone is BAAI
bge-small-en-v1.5 (MIT). Its separate model files, license, immutable revision,
digest, and verified conversion still need to be selected and packaged.

## Native integration requirements

Use `include/atlantic_host.h` for the C ABI or `include/atlantic_host.hpp` for C++.
No Python runtime is used by the application library. Python is used only for
development-time dependency acquisition and vendor verification.

The model adapter must load a verified compatible Atlantic GGUF, use its exact
chat template, and emit one actual tokenizer token per sink call. It must return
normal completion only after an actual end-of-generation, stop promptly when the
sink returns false, and poll the cancellation callback during long prefill or
decode work. It must not display/log tokens or append unchecked history, memory,
knowledge-store excerpts, or search data. Assemble the entire inference input
before calling the host; preserve that same checked semantic input in the model
template. This boundary does not implement retrieval or tool execution.

The embedding adapter must provide the exact bge-small-en-v1.5 first-token-pool
vectors, 384 dimensions, for each supplied window. The upstream judge handles
normalization, finite/norm checks, and windows. A different embedding model or
pooling method invalidates the trained heads. Returning an arbitrary vector to
make the check pass is never a release implementation.

Run the host on a native worker thread. The C ABI is synchronous, owns its result
objects, and does not call into UI code. Keep the host, cancellation object, and
callback user data alive until a request returns. Use one cancellation object per
request. Do not retain any callback pointers after a callback returns. A second
request on the same host returns `busy`; callbacks must cooperate for cancellation
to interrupt long native operations. The host cannot preempt a stuck callback.

The remaining application wiring is:

- Compile/link llama.cpp and the verified encoder for each target architecture.
- Implement and parity-test the two callbacks against the pinned harness setup.
- Link this library through Tauri/Rust and native mobile adapters.
- Connect only the final C ABI result to chat rendering.
- Run real Atlantic reply evaluation and device memory, latency, and interruption
  tests before enabling consumer inference.

All five platforms are design targets. Only the Linux C++ build below has been
compiled here; this directory is not evidence of Windows, macOS, iOS, or Android
build completion.

## Build and verify

From this directory in a Linux developer shell, with a C++17 compiler, make, ar,
and Python 3, first acquire the four pinned public head dependencies:

```sh
python3 fetch_heads.py
make all test
```

`make fetch-heads` is an optional equivalent for the acquisition step. Building
the library does not automatically download anything. Network access is required
only when a head file is absent or fails its expected size/checksum.

The static library and test binary are placed under `/tmp/atlantic-native-build`.
Override `BUILD_DIR` for a different development output directory. A direct test
command, without make, is:

```sh
python3 fetch_heads.py
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror -pthread \
  src/atlantic_host.cpp src/atlantic_ffi.cpp tests/host_test.cpp \
  -o /tmp/atlantic-host-test
/tmp/atlantic-host-test vendor/krummholz-harness
python3 verify_vendor.py
```

The tests create clearly synthetic heads, embeddings, and model completions only
inside a temporary test directory. They exercise the ordering of checks, hidden
reasoning removal, split reasoning markers, late-window screening, unavailable
judges, malformed input/output, generation errors, cancellation, reentrancy, and
the C ABI. A separate assertion loads the real pinned head files and verifies
that missing encoder output prevents generation. Synthetic fixtures are never
used by the consumer application.

Verification performed on November 17, 2026: 57 boundary assertions passed;
17 source/dependency files matched their recorded hashes, including the four
locally present public head files; the C header compiled as C11.
The five upstream suites also passed in the separate source checkout: screen
(46 rules), gate, judge, reasoning budget, and sampling. **No actual Atlantic or
bge model inference, encoder parity, or safety-efficacy evaluation was performed.**

Upstream limits still apply: the published embedding judge is English only, its
reply head was evaluated on Aegis replies rather than Atlantic replies, and a
passing verdict does not prove that an answer is correct or safe. This host does
not implement dynamic weight training, knowledge-store storage, search, signed
downloads, or mobile scheduling; those are separate application components.
