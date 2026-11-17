# Atlantic app

An installable, local model workspace for Atlantic and Atlantic Lite, designed for Windows, macOS, Linux, iOS and Android.

**Status: design and integration foundation, November 17, 2026. This is not a finished inference app or a signed installer.** The interface explicitly identifies itself as a preview. It does not pretend to generate model replies, download weights, run the real embedding encoder or train adapters.

## Included now

- Responsive React interface in the existing Krummholz forest-green visual language, with the real K mark.
- Chat, Knowledge, Learned today, Models and Settings screens.
- The actual existing knowledge pack, prepared without changing its bytes or attribution: 2,359 passages, 141 titles, approximately 5 MB. Its current subject coverage is history, censorship and sovereignty, not a general encyclopedia.
- SHA-256 verification before loading the pack and a BM25 port checked against the Python source.
- A working preview memory-review flow, stored locally in the browser for design testing. This is not the production personal-data store.
- Pinned current Atlantic and Lite file sizes, hashes, revisions and a staged-download contract.
- A native C++ host boundary built around the current published harness. Synthetic callback tests exercise the boundary; actual GGUF generation and BGE encoder parity are not tested by those checks.
- A Tauri 2 shell configuration with explicit unavailable runtime capability reporting. Rust and platform SDK builds remain to be performed.
- Two regression fixes in the existing Python review code: invalid edits preserve pending candidates; facts-only reviews do not request weight training.

The full product and implementation design is in [PRODUCT_DESIGN.md](docs/PRODUCT_DESIGN.md). Source pins and the actual verification record are in [VERIFICATION.md](docs/VERIFICATION.md). Data and trademark notices are in [ATTRIBUTION.md](docs/ATTRIBUTION.md).

## Developer preview

From this repository's `apps/atlantic` directory, using Node.js 22.18 or later and Python 3 for the parity test, run these in PowerShell:

```powershell
npm install
npm test
npm run build
npm run dev
```

The development server binds only to loopback. The app does not require users to install Node, Python or any command-line program in the planned consumer distribution.

`npm run standalone` after building creates `atlantic-design-preview.html`. It embeds the actual corpus and logo, works without a service after opening, and verifies the embedded corpus bytes. This is a review artifact, not a five-platform installer. The preview uses browser local storage only for explicitly entered demo preferences and memories. Use synthetic examples; production encryption and OS key handling are separate work.

Native boundary test instructions are in `native/README.md`. The four small judge-head binaries stay at their existing pinned public HF release and are fetched with checksum verification during development or packaging; they are not republished in this Git branch. Native packaging requires Tauri's Rust prerequisites and each platform's SDK. The current environment did not have Rust, Xcode or the required platform build setup, so the shell is provided as source without a claim of a successful native build.

## Release requirements

The app must not expose a normal chat completion until its exact model, encoder, rule pack, judge heads and runtime all verify and the complete guarded turn passes. Implement one common native host on all platforms. Do not make Python, Ollama, a localhost model server or Docker a consumer dependency. The model selector only lists verified, installed, compatible packs.

The production requirements include authenticated resumable model acquisition; native secure storage; conversation and source provenance; the real memory/review service; adapter training, evaluation and rollback where supported; a privacy-disclosed search integration; signed builds; accessibility; and physical-device tests. A directory or a configuration for a platform is not evidence that the app runs there.
