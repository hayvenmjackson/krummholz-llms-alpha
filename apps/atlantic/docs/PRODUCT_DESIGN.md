# Atlantic: product and implementation design

Date: November 17, 2026. Owner: Krummholz AI LLC. Working app name: Atlantic.

This document proposes the complete consumer product and records the current implementation boundary. The target is Windows, macOS, Linux, iOS and Android. The design preview, retrieval port and native boundary are initial implementation work. Production inference, training and platform packages are not completed here.

## 1. The product

A person installs an ordinary app, chooses the recommended download, and talks to an Atlantic model. The app owns acquisition, compatibility checks, the knowledge store, personal memory, the harness, the judge, the gates and optional web search. The user never needs a Hugging Face account, terminal, Ollama installation or Docker runtime for public offline model packs.

Use the website's dark forest background, restrained green accent, exact K mark and joined krummholzAI wordmark. The full brand type system is Libre Franklin for display text and DM Sans for body text. Bundle appropriately licensed font files for release. The current preview uses local system fallbacks and makes no remote font request.

The app is local by default. No account is required for local chat. Search is a clearly separate network feature. No model request silently moves to a hosted provider because the device is too small, offline or slow.

## 2. Five screens

| Screen | What the person does | What must remain accurate |
| --- | --- | --- |
| Chat | Ask a question, choose an installed model, attach an optional supported file, inspect sources | Show generation/checking progress while the native core buffers the answer. Never show hidden reasoning or an unreviewed draft. |
| Knowledge | Search and read the bundled reference collection, inspect personal sources and source dates | A search hit is not a verified claim. Full passages retain original links, revisions and licenses. |
| What I learned today | Review explicit memory and correction proposals, accept, edit, skip, forget and inspect history | A saved fact is memory. An accepted training example is not an applied weight update. |
| Models | See the recommended model, actual download size, compatibility, progress and installed versions | Only verified installed packs are selectable for inference. A downloaded model is not necessarily compatible. |
| Settings | Choose network behavior, storage, accessibility and learning preferences | Network and learning status must reflect actual native capabilities. |

Use a sidebar on desktop and large, labeled bottom navigation on phones. The main chat view remains uncluttered. Version hashes, quantization details and evaluation reports belong behind Details, while download size, readiness and source age stay visible.

First run checks available memory, disk, architecture and backend capability. It offers one recommended package with a clear explanation. If the device cannot support it, show a useful incompatibility result before transferring gigabytes. Hardware checks are estimates until a bounded load/prefill/self-test succeeds; advertised device RAM alone is insufficient.

## 3. Model packages and hardware

Current metadata was retrieved directly from Hugging Face on November 17, 2026. The exact pins are in `src/catalog.ts`.

| Package | Exact weight bytes | Display size | Intended first use |
| --- | ---: | ---: | --- |
| Atlantic Lite v8.1 Q4_K_M | 2,741,241,728 | 2.74 GB | Initial phone and ordinary-computer candidate, subject to app/device validation |
| Atlantic v8 Q4_K_M | 16,742,036,672 | 16.74 GB | High-memory desktop/workstation candidate; current card targets 24 GB GPU memory |
| Existing knowledge pack | 4,975,273 | 4.98 MB | Included immediately |
| Lite image projector | 672,423,040 | 672 MB | Optional later image-input package |
| Atlantic image projector | 927,606,912 | 928 MB | Optional later image-input package |

The judge also needs the exact BGE small encoder, described upstream as approximately 133 MB, plus approximately 2.7 MB of trained heads. The encoder GGUF still needs an approved immutable revision and checksum. Runtime libraries and platform packaging add their own size; do not advertise an exact installer size before building it.

Both models can be downloaded and selected on suitable devices. Normally load one chat model at a time, with the lightweight judge resident or scheduled to fit memory. Mobile defaults to Lite. Supporting iOS does not imply that a 16.74 GB model works on every iPhone. The app can support all five operating systems while offering different compatible model choices.

The cards distinguish full-precision and quantized quality. Lite's Q4_K_M file is wrong on 24 of 229 no-tool fact questions in the published evaluation; full precision is wrong on 5. Full Atlantic Q4_K_M is wrong on 35 of 229. These are author-reported small-suite results, not measurements through this app or a general accuracy guarantee. Q5_K_M is published but not evaluated in the cited cards. Keep each file's actual evaluation attached to that file. Do not promise a blanket sub-1-percent hallucination rate or label a larger file automatically better for every task.

## 4. Native architecture

Recommended shell: Tauri 2 and a shared React/TypeScript interface. Tauri documents support for all five requested systems, with Rust application logic and Swift/Kotlin integration where needed. Use the existing C++17 harness directly within a common native host. The C++ host links llama.cpp as a library; Rust exposes only narrow guarded operations to the interface.

This combines the team's existing web development strengths with a native core. Flutter is a credible alternative but adds a different UI stack without eliminating native inference, training or download work. The important decision is the common native boundary, not a framework label.

| Layer | Responsibility | Current state |
| --- | --- | --- |
| React interface | Navigation, local review UX, readable sources, model status | Working design preview |
| Tauri/Rust | OS integration, permissions, secure storage, orchestration and package lifecycle | Shell configuration and honest status command |
| Shared C++ host | The only path from a model draft to visible output | Initial guarded boundary with synthetic tests |
| llama.cpp | GGUF loading, tokenization, sampling, native generation and embedding | Integration required, current cards cite e574a17 |
| Knowledge and memory | Dated reference corpus, source ledger, user-reviewed facts and examples | Existing Python contracts; retrieval port in preview; production store binding needed |
| Platform adapters | Background transfers, credential vaults, resource measurement, lifecycle | Specified, not implemented |

Use in-process native integration on phones. A desktop sidecar may isolate failures, but the public command contract remains identical and no raw model server is exposed. Avoid making desktop subprocess behavior a dependency of the iOS design. CPU work stays off the UI thread. Backend acceleration is selected only after validation for the actual model architecture and operating system.

The model cards cite llama.cpp `e574a17` on November 15, 2026. Resolve that short pin to a full immutable commit and verify its trust/provenance before a reproducible production build. A current upstream Android or iPhone example establishes a platform path, not Atlantic-specific correctness or speed.

## 5. Harness, judge and gates

Use the current HF harness at revision `30f793a03d7ba5ca0f123f275171f04ac04aab4d`, not the older Android-only examples from GitHub main. It contains 46 request/reply rules, the combined gate, reasoning budget, sampling helper and trained linear/embedding judge heads. The embedding judge requires BAAI/bge-small-en-v1.5 with its exact tokenizer, first-token pooling and 384-dimensional output. The host must supply that embedding implementation.

Mandatory guarded turn:

1. Verify model, rules, judge heads, encoder, knowledge pack and compatibility records.
2. Retrieve approved evidence through bounded operations and assemble the complete inference input, including history, memory and source excerpts. Treat external content as data and retain a source ledger. A preliminary user-message check may run before retrieval, but cannot replace checking the complete input.
3. Run the request rules and the separate embedding request verdict on that complete input. `SafetyGate.before()` alone does not run the embedding verdict. Do not append unchecked content after this point; any further tool round must assemble and check a new complete input.
4. Generate in native code while enforcing cancellation, resource limits and the 1,024-token reasoning budget.
5. Keep hidden reasoning and all draft text inside the host.
6. Run output rules and the reply judge on the visible answer.
7. Return only the final approved answer or the fixed unavailable/abstention/refusal response.

No missing-judge fallback becomes rules-only chat. No user-facing debug toggle bypasses the gate. A spinner can say Looking in your library, Thinking or Checking the answer. That is progress; it is not a stream of potentially rejected answer tokens.

The gate is one component of quality and safety, not proof that every accepted answer is true. Grounding needs source membership, citation checks, numerical checks where applicable, and honest uncertainty. The current upstream judge is English-only, and its reply head was evaluated on Aegis replies rather than an independently established Atlantic reply suite. App-level release evaluation must cover the whole combination.

## 6. Knowledge store and personal memory

Preserve the existing store, rather than replacing it with a vague Memory checkbox. The current dated collection has 2,359 passages from 141 articles, mostly covering history, censorship and sovereignty. It is a useful real pack, not a complete general knowledge base. Expand it with separately licensed subject packs as a later content channel.

The reference pack is read-only and versioned. Each passage retains title, section, source URL, revision, license and date. The app verifies the raw file checksum before indexing. Search results are discovery aids. A fetched full passage is evidence with a source identity, but semantic support still needs checking. Display the corpus date when a user asks a current question. A November snapshot cannot establish today's officeholder, price or weather.

Personal facts remain in a separate, user-controlled store. Approved behavior examples and correction pairs are separate again. Imported documents can support retrieval when the user chooses them, but should not quietly become training examples. Conversations, local notes, licensed reference data and live web results have different provenance and retention rules.

Production storage should use native SQLite with platform-protected encryption keys and explicit export/delete controls. Keep this out of browser localStorage in the release app. The preview's small localStorage memory flow is intentionally a design demonstration, clearly marked as such.

## 7. What I learned today and dynamic weights

Every review item has a type, origin, creation date, proposed action and state. The screen should answer: What is being kept? Why? What changes if I accept it? Can I reverse it?

| Item | Accepting it does | Release wording |
| --- | --- | --- |
| Personal fact | Saves a retrievable fact with provenance | Remembered |
| Behavior or correction | Adds an explicit approved training example | Approved for learning |
| Candidate adapter | Starts the required evaluation path | Checking a proposed update |
| Adapter that passed checks | Atomically activates a compatible signed artifact | Dynamic weights updated |
| Rejected or failed candidate | Leaves the current working adapter in place | Current version kept |

Reuse the existing capture rules, review limit, signed manifests, compatible stacks, candidate-versus-active gates and canary/replay machinery. The existing three-review-cycles-per-UTC-day policy should be represented consistently rather than silently changed by a local-date UI.

The existing roadmap supports desktop or explicitly opted-in hosted training first. Mobile training is research with exit criteria. llama.cpp's ability to load an inference adapter does not establish an online trainer for these models. Local memory can work on phones while the mobile weight-training feature remains unavailable; if full on-device training on every platform is mandatory for release, that becomes a release gate rather than an invented capability.

Use an isolated candidate adapter, evaluate it against unchanged held-out checks and appropriate replay data, and activate only after passing the configured gates. Keep the last known good compatible version. Record the base-model hash, training-example lineage and gate receipt for every activation.

Forgetting a fact removes its future retrieval. Forgetting an example already used for training additionally invalidates or rebuilds affected adapters. Do not claim that deleting a database row unlearns model weights. Immediate rollback to an unaffected adapter is the practical safe path while rebuilding.

Two small source fixes accompany this foundation: invalid edits no longer remove the pending candidate, and a facts-only review no longer requests training.

## 8. Downloads that recover cleanly

The app manages a small signed catalog separately from large model files. Each entry binds exact bytes, SHA-256, immutable source revision, compatible engine/model versions, license, architecture and optional dependencies. The current catalog is checked source metadata, not yet an operational signing or update service.

Stage the product:

1. App, current harness, judge and knowledge store. The library can be used immediately.
2. Complete verified Lite package. Unlock local chat only after required components and self-tests pass.
3. Optional full Atlantic. Keep Lite usable while this downloads.
4. Optional vision projector and selected knowledge expansions.

Write downloads into a private partial file with durable progress. Range resumption must match file identity, Content-Range and expected total bytes. If a server returns the complete file instead of the requested range, restart the partial file rather than append duplicate bytes. Finish with a whole-file hash and atomic rename. A resumed segment is never treated as a complete installed model. Retry boundedly with backoff and retain useful progress after interruption.

Reserve enough disk for old and incoming versions at the same time. If the platform copies rather than renames, budget for that extra copy. Do not delete the working model before its replacement verifies. Use large sensible chunks if chunk-level checks are available; arbitrary bytes of an incomplete ordinary GGUF do not produce a usable partial model.

Native transfer adapters: background URLSession on iOS, user-initiated transfer jobs on newer Android with an appropriate older-version fallback, and a native HTTP transfer worker on desktop. iOS controls background execution; force quitting the app can cancel a background transfer. Persist state so reopening offers recovery, without promising the OS will always finish the job while closed. Wi-Fi and cellular preferences belong in the first download flow.

Ideal transfer estimates, excluding overhead and verification:

| Connection | Lite 2.74 GB | Atlantic 16.74 GB |
| --- | ---: | ---: |
| 100 Mbps | 3.7 minutes | 22.3 minutes |
| 50 Mbps | 7.3 minutes | 44.6 minutes |
| 10 Mbps | 36.6 minutes | 3.72 hours |

These are bytes divided by throughput, not observed benchmark times. Staging makes the first useful experience smaller and avoids repeating failed transfers. It cannot make a slow connection move 16.74 GB instantly. No external media is required.

## 9. Search partner

Brave is the leading integrated provider because it offers a documented web-search API. Keep a provider interface so the choice can change without rewriting the app. DuckDuckGo can be offered as an explicit Open in browser search; no current official full-results API suitable for this integration was established in this review. Do not make an undocumented scraper the reliability foundation.

For a nontechnical user, a small authenticated Krummholz broker can hold the service key and enforce quotas. It receives only the approved search query, not the entire chat or local documents. This is a proposed service, not one deployed by this task. Advanced per-user keys can be an optional alternative. Never embed a reusable company key in an app binary.

The UI explains exactly when a query leaves the device and which services receive it. Local inference remains local. Ordinary Brave API terms/privacy do not justify a zero-retention claim: the current privacy notice allows query retention up to 90 days. Search-result persistence and use for model improvement also depend on the contracted rights. Ordinary current terms restrict both. Keep provider-retention and training-eligibility flags on evidence and exclude unlicensed search material from persistent learning. Any broader rights need an appropriate agreement before activation.

The runtime tool broker should expose bounded search and fetch operations, not arbitrary shell or network tools. Treat retrieved pages as untrusted data, preserve citation provenance and reject attempts to turn source content into instructions. Live search, a billing account and provider keys are not connected in the preview.

## 10. Validation and delivery

One codebase produces five builds, with genuine platform-specific integration and testing. Desktop bundles can use the appropriate signed Windows and macOS distributions and a tested Linux package. Android uses APK/AAB distribution. iOS uses a signed app through the applicable Apple distribution path. A desktop executable inside a ZIP is not an iPhone application.

Before a consumer beta, demonstrate on physical devices: a complete current-model chat; exact encoder parity; full gate and tool behavior; low-memory handling; cancellation and lifecycle recovery; interrupted-download resumption; corrupt-pack rejection; local facts and review persistence; a real evaluated adapter activation and rollback where offered; and accessible navigation with screen readers and keyboard/touch input.

Define supported device classes from measured peak memory, prefill latency, generation speed, sustained temperature and battery use. Test the active combination of chat model, context, judge, knowledge index and GPU backend. Do not infer device support merely from weight file size.

The first vertical slice should connect the current Lite model and exact encoder to the native host on one Windows machine and one representative phone, with the real knowledge pack and review service. Build support for all five remains part of the architecture; each platform becomes releasable only after its own native integration, signing and physical-device verification. No store submission, paid service or production deployment was performed for this design.

## Sources

- Model cards and immutable file metadata: https://huggingface.co/krummholz-ai/atlantic-lite-4b-v8.1-GGUF and https://huggingface.co/krummholz-ai/atlantic-27b-v8-GGUF
- Current native harness: https://huggingface.co/krummholz-ai/krummholz-harness/tree/30f793a03d7ba5ca0f123f275171f04ac04aab4d
- Existing knowledge, dynamic weights and Android source: https://github.com/hayvenmjackson/krummholz-llms-alpha/tree/86334a760ef3b2c1a57f22e6b347c390d4feaeee
- Tauri platform architecture: https://v2.tauri.app/ and https://v2.tauri.app/develop/plugins/develop-mobile/
- llama.cpp Android: https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md
- llama.cpp iPhone: https://github.com/ggml-org/llama.cpp/blob/master/examples/llama.swiftui/README.md
- Apple background transfers: https://developer.apple.com/documentation/foundation/downloading-files-in-the-background.md
- Android user-initiated transfer jobs: https://developer.android.com/develop/background-work/background-tasks/uidt
- Brave API terms: https://api-dashboard.search.brave.com/documentation/resources/terms-of-service
- Brave API privacy notice: https://api-dashboard.search.brave.com/documentation/resources/privacy-notice
