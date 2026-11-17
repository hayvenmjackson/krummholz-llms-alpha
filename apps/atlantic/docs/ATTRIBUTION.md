# Attribution and data notices

The app reuses Krummholz AI's published model and runtime work. The K mark and Krummholz AI name remain trademarks of Krummholz AI LLC. A code license does not grant trademark rights.

## Knowledge pack

The application prepares an unchanged copy of `data/library/krummholz-gates-library-v0.json`. This is data separately licensed under CC BY-SA 4.0. The pack contains 2,359 passages from 141 article titles, dated November 9, 2026. Each passage retains its title, section, revision identifier, original URL and license. The interface displays that attribution when opening a passage. Follow the original URL for its article history and contributors.

License: https://creativecommons.org/licenses/by-sa/4.0/

The original text has not been rewritten to match app punctuation rules. Preserving quoted source content takes precedence. The app's generated framing and code follow the repository style rules.

The source JSON file has SHA-256 `ed60b93d3f527b6a4f2ce6efed2e4f5d8b1fe0eddf48aede5d2c83e0e8891f29`. This file checksum is distinct from the canonical body checksum embedded in the corpus. The preview checks the exact file bytes before indexing.

## Models

Atlantic Lite v8.1 is based on Qwen3.5-4B. Atlantic v8 is based on Qwen3.8-27B. Both model cards declare Apache 2.0. Their base-model and dataset notices remain authoritative; an app wrapper does not replace them. No model weights are copied into this source branch.

- https://huggingface.co/krummholz-ai/atlantic-lite-4b-v8.1-GGUF
- https://huggingface.co/krummholz-ai/atlantic-27b-v8-GGUF

## Harness and judge

The current native headers and rules are vendored unchanged with their license and source provenance under `native/vendor`. The four binary judge heads remain at their existing immutable public Hugging Face release; `native/fetch_heads.py` obtains and verifies them for development and packaging. They are not copied into Git history. Preserve the Apache 2.0 license and NVIDIA Aegis 2.0 attribution in the built application. The separately required BAAI/bge-small-en-v1.5 encoder has an MIT license. The actual encoder GGUF is not bundled in this foundation.

https://huggingface.co/krummholz-ai/krummholz-harness

## Interface dependencies

React, React DOM and Scheduler: MIT. Vite: MIT. TypeScript: Apache 2.0. Lucide: ISC, with MIT notices for icons derived from Feather. Full runtime license text is retained in `THIRD_PARTY_NOTICES.txt` and embedded in the standalone preview. Consult the exact installed packages for additional build-tool notices when building a release. The design preview uses system font fallbacks and loads no fonts or images from a remote service.
