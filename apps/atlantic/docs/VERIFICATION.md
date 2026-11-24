# Verification record

Date: November 17, 2026. This record describes the design foundation, not a consumer release.

## Executed checks

| Check | Result | What it establishes |
| --- | --- | --- |
| TypeScript type checking and Vite production build | Passed | The current interface and retrieval implementation compile and bundle. |
| Core tests | 7 passed | Exact knowledge checksum, corruption rejection, source attribution, BM25 parity against the existing Python store, invalid-source rejection, resume-response validation and download calculations. |
| Standalone preview workflow | 1 passed | The actual embedded bundle renders in JSDOM, verifies and searches the real corpus, exposes attribution, completes memory add/edit/keep/forget/undo, and accurately labels unavailable generation and download features. All fetch calls are forbidden in this check. |
| Existing dynamic weights suite | 26 passed | Includes six new regression cases for candidate preservation and the distinction between memory and trainable examples. |
| Linux C++ boundary | 57 synthetic assertions passed | Guard ordering, private reasoning, checked final output, malformed data, unavailable encoder, cancellation, reentrancy and the C ABI contract. |
| Native dependency verification | 17 files passed | The copied harness files and downloaded public trained heads match their recorded immutable source hashes. Binary heads remain dependencies of the existing HF release and are not republished in the Git branch. |
| C header compatibility | Compiled as C11 | The public ABI header can be consumed by C tooling. |
| Upstream harness suites | 5 passed in the separate upstream source checkout | Screen with 46 rules, gate, judge, reasoning budget and sampling tests. |

The synthetic native tests do not run an Atlantic model or the BGE encoder, and do not measure the quality or effectiveness of the trained judge. The preview workflow uses DOM and dialog polyfills; it is not a browser rendering, accessibility or physical-device test. Third-party runtime notices are retained in the source and standalone artifact.

## Source pins

| Component | Immutable revision or digest |
| --- | --- |
| Existing GitHub source baseline | `86334a760ef3b2c1a57f22e6b347c390d4feaeee` |
| Atlantic Lite v8.1 HF repository | `2bd4f6a56740ce8c1afe7ee61d3b41001433410e` |
| Atlantic v8 HF repository | `12f9aef52e11595a055f51ad553f9a077ffaea45` |
| Native harness HF repository | `30f793a03d7ba5ca0f123f275171f04ac04aab4d` |
| Existing knowledge file SHA-256 | `ed60b93d3f527b6a4f2ce6efed2e4f5d8b1fe0eddf48aede5d2c83e0e8891f29` |

The model file checksums, byte counts and immutable download paths are in `src/catalog.ts`. The 17 native dependency hashes are in `native/vendor/krummholz-harness/PROVENANCE.json`. The four small public judge-head binaries are fetched by `native/fetch_heads.py` when preparing the native tests or app package. No chat-model GGUF or encoder GGUF was downloaded for this design task. The current model cards cite llama.cpp `e574a17`; resolving and validating its complete commit remains a release requirement.

## Limits of this verification

The connected-browser interface was unavailable from this conversation. Cloud browser control timed out, native desktop input reported an unavailable accessibility provider, and local headless Chromium could not create its required socket. A successful visual review is therefore not claimed. The responsive rules and touch target changes have source review and DOM coverage only.

Rust, Xcode and the full platform SDK toolchains were not available in this environment. The Tauri source has not been compiled. The C++ library has only been compiled on Linux. Windows, macOS, Linux consumer packaging, iOS and Android app execution remain unverified.

The following are intentionally incomplete and unavailable in the preview:

- Actual GGUF generation, exact encoder integration and encoder parity.
- A signed release catalog and production runtime asset validation.
- Native background model transfers and interruption/recovery tests.
- The native encrypted knowledge, memory and review service binding.
- Real adapter training, held-out evaluation, activation and rollback.
- Live Brave API search, provider credentials and service deployment.
- Platform permissions, signing, installers, store submission and physical-device qualification.

The preview is explicitly labeled and never fabricates completion, model installation, a live download or a trained update. Browser localStorage is used only for entered demonstration memories and preferences; use synthetic examples when evaluating it.

## Reproducing the interface checks

From `apps/atlantic`, with the prerequisites in the README, run in PowerShell:

```powershell
npm ci
npm test
npm run build
npm run standalone
npm run test:preview
```

From the repository root:

```powershell
python -m unittest tests.test_dynamic_weights
```

The separately tested Linux C++ build procedure and its integration requirements are in `native/README.md`.
