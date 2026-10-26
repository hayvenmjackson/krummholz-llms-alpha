# Byte-preserving distribution of the BF16 candidate

Prepared 26 October 2026. Tooling is implemented and tested with small synthetic files. **No model was packaged, uploaded, approved for release or imported into Android by this task.**

The verified conversion candidate is `SmolLM3-3B-BF16.gguf`, **6,158,339,904 bytes**, SHA256 `8bb048dbbc726c8e10bdf41575f5209b9b68a27b0ec15390b5031846047cec9b`. Its [conversion evidence](https://github.com/hayvenmjackson/krummholz-llms-alpha/actions/runs/40324601321) records unchanged tensor values, BF16/F32 tensors and no quantization. Multipart packaging concatenates the same bytes; it does not change precision or model behavior.

## Distribution choice

[GitHub's official release limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases) require every asset to be strictly below 2 GiB. Use four raw parts with a 1,610,612,736-byte (1.5 GiB) maximum:

| Part | Byte offset | Bytes |
| --- | ---: | ---: |
| 1 | 0 | 1,610,612,736 |
| 2 | 1,610,612,736 | 1,610,612,736 |
| 3 | 3,221,225,472 | 1,610,612,736 |
| 4 | 4,831,838,208 | 1,326,501,696 |

Publish the manifest and license/attribution/conversion information alongside the four parts only after the separate release gates pass. Record the exact manifest SHA256 in the trusted release metadata and eventual signed app. Per-part hashes alone are insufficient: the tool also checks the whole file and the manifest itself.

A single file in a Krummholz-controlled Hugging Face model repository is a simpler future phone download/import path. [Hugging Face's current official storage guidance](https://huggingface.co/docs/hub/en/storage-limits) supports files much larger than this candidate, but account quota and access must be checked before choosing that route. Public free storage is best effort; no paid plan or free capacity is assumed. Pin the host repository's immutable commit, exact filename, byte size and SHA256. Do not treat an arbitrary community reupload as the verified conversion. Nothing was uploaded or provisioned here.

## Pinned license evidence

The [publisher's pinned model card](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/a07cc9a04f16550a088caea529712d1d335b0ac1/README.md) declares Apache-2.0 in its metadata and License section. Its exact hash remains `24f16f90b883af3618425a589783a944356a7fc2e3cfa69b579c9ff06963e178`. The complete file inventory at that revision contains no separate `LICENSE` or `NOTICE`; direct requests for both returned 404. That absence is recorded, not replaced with a fabricated upstream notice.

[Apache 2.0 section 4](https://www.apache.org/licenses/LICENSE-2.0) requires a license copy, preservation of relevant notices and prominent notices of modifications; upstream NOTICE material must be preserved when present. The exact [license copy](licenses/SmolLM3-Apache-2.0.txt) was retrieved from Apache and hashed. [Machine-readable evidence](smollm3-distribution-evidence.json) records the publisher and license pins. A release must retain the source attribution/model-card provenance and plainly identify conversion from the pinned safetensors source to GGUF and byte splitting. This is an unchanged upstream comparator, not Krummholz-trained weights. The model license does not establish that every inherited training source is public domain or independently rights-cleared.

## Local commands

The [standard-library tool](../tools/model_multipart.py) performs no network requests, model loading or publication. Run from the repository root on Python 3.12+. The example assumes a previously verified GGUF is already present; it does not download one.

```text
python -B tools/model_multipart.py split --source models/SmolLM3-3B-BF16.gguf --output-dir artifacts/smollm3-bf16-parts --expected-bytes 6158339904 --expected-sha256 8bb048dbbc726c8e10bdf41575f5209b9b68a27b0ec15390b5031846047cec9b --provenance docs/smollm3-distribution-evidence.json
```

The output directory must not exist. Success prints the manifest SHA256. A failed split deliberately leaves its newly created partial directory for inspection; no valid manifest is published when the whole-source pin fails. The tool never recursively removes a directory. Choose a fresh directory for a subsequent attempt after investigating a failure.

Recipients obtain the expected manifest digest from trusted release metadata, not by hashing an untrusted downloaded manifest and declaring it trusted. Replace `TRUSTED_MANIFEST_SHA256` below with that value. Place all parts beside the manifest. The output's parent directory must exist, and the output file must not.

```text
python -B tools/model_multipart.py verify --manifest artifacts/smollm3-bf16-parts/multipart-manifest.json --manifest-sha256 TRUSTED_MANIFEST_SHA256
python -B tools/model_multipart.py assemble --manifest artifacts/smollm3-bf16-parts/multipart-manifest.json --manifest-sha256 TRUSTED_MANIFEST_SHA256 --output models/SmolLM3-3B-BF16.gguf
python -B -m unittest tests.test_model_multipart -v
```

Payload reads are bounded to 8 MiB; no 6 GB in-memory copy is created. Assembly verifies every part's size/hash while writing a unique sibling temporary file, verifies the complete digest, flushes it, then atomically creates the final filename using a no-overwrite hard link. Only the invocation's owned temporary file is removed. Existing files, parts and unrelated temporary files are preserved. Filesystems without hard-link support fail closed; use a supported local filesystem such as NTFS or ext4. Crash leftovers retain a `.partial` name and must not be imported. Checksums detect changes; they do not authenticate a malicious replacement when the trusted manifest pin is also replaced.

Tests cover exact bytes and boundaries, corrupt/missing/truncated/extended parts, whole-file mismatch despite valid part hashes, manifest tampering and duplicate keys, order/offset/path attacks, 64-bit offsets, limited reads, disk space refusal, unsupported atomic publish and concurrent destination creation. Symlink rejection is tested where the operating system permits creating a test symlink.

## Phone integration status

Android multipart import is implemented as reusable infrastructure, but actual part pins are not yet embedded and this candidate remains unapproved. A working download/import alone does not establish model quality, runtime compatibility or device acceptance.

The signed APK's existing top-level `filename`, `bytes` and `sha256` fields pin the whole file. Its optional `multipart` object uses `schema_version: 1`, `format: "raw-concatenation-v1"`, the exact `manifest_sha256`, `part_bytes: 1610612736`, and an ordered `parts` list. Each list entry contains integer `index`, canonical `name`, integer `offset`, integer `bytes`, and exact `sha256`, copied from the packaging output. Actual hashes must be generated from the verified artifact; no placeholder hash enables release.

Users select all four local files through SAF in any order; they do not supply a manifest or run Python. The app maps bounded metadata to embedded canonical names, rejects missing/duplicate/unknown parts, and requires regular local file descriptors with exact sizes. It streams into a unique app-owned temporary file, verifies each part and the whole, then activates atomically. Provider metadata/open calls have 30-second cancellation watchdogs. A provider that ignores Android's cancellation transport cannot be force-terminated by the app; this remains a device/provider acceptance concern. The optional single-whole-file content-provider path remains available.

Fourteen JVM verifier tests pass. Seven synthetic Android instrumentation cases exercise real provider metadata, regular descriptors, pipe rejection, cancellation, whole-file compatibility and preservation of the previous model; these compile but execution remains pending. They use isolated storage and seven fictional bytes, without changing production pins or invoking a model. Existing actual-model tests use a test-only provider receiving an explicitly registered read-only descriptor, so the production importer need not accept `file://` paths.

Parts plus the assembled/imported model require roughly **12.32 GB** of storage while both copies coexist, plus normal app/filesystem headroom. An optional whole-file host removes user reassembly steps but an external download plus private app import can still require two copies.

Packaging success is not a semantic-quality pass, a mobile performance result or release authorization. The source-QA pilot and phone acceptance gates remain in force.
