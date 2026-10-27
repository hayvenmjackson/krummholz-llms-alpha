# Synthetic Maine source-QA set, version 1

500 verified examples (450 train, 50 validation) for evidence-seeking behavior
across forestry, fishing, customer service and workplace robotics in Maine.
All businesses, people, records and rules are fictional.

| Category | Train | Val | Target behavior |
| --- | ---: | ---: | --- |
| conflict | 135 | 15 | Flag unresolved source conflicts; apply explicit amendments |
| injection | 135 | 15 | Complete the real task; treat planted instructions as untrusted |
| missing_evidence | 90 | 10 | Absence of a record is not proof something did not happen |
| plain | 90 | 10 | Answer directly with no unsupported hedging |

## Provenance

- Generated 2026-10-26 by Krummholz AI LLC with `tools/training/generate_synthetic.py`,
  drafting with an open-weight model served through Ollama. The exact model name,
  license, temperature, seed and prompt hash are in the run's
  `synthetic_maine.provenance.json`; copy that file here before external release.
- Every kept draft passed the script's verifier: every number in each expected
  answer appears in its sources, plus category-specific checks, near-duplicate
  removal and em-dash normalization.
- Re-verified 2026-10-27: 500 of 500 pass, zero shared word 8-grams with the
  36-case pilot test set, 500 distinct queries, no duplicate IDs.
- No Claude-generated example text. Conversion into the app's answer protocol
  (`tools/training/convert.py`) is deterministic extraction from these files.

SHA-256:
- `train.jsonl` cf5921baf79c3d1ce854b79aff236895a6ff322819c5f0f216901a9387a6a153
- `val.jsonl` 322e2137a1dd78e454738d67410012b45be828fbc19f5e5ae81a9ed21ab30b3a

## First training use (2026-10-27)

`tools/training/convert.py` produced 404 train and 46 validation records in the
pilot JSON protocol (examples without a valid exact-quote citation were dropped).
A rank-16 LoRA on original BF16 Qwen3 8B, 2 epochs, 52 steps, took 444 seconds on
one A40. Validation loss fell from 0.906 to 0.162; 46 of 46 validation outputs
parsed.

Frozen 36-case pilot, greedy decoding, development-agent review:

| Model | Parsed | Notable changes vs untrained Qwen3 8B |
| --- | ---: | --- |
| Untrained Qwen3 8B (run-003) | 36/36 greedy | Failed 3 conflict cases, 1 injection, 1 absence-as-proof |
| Trained, BF16 | 36/36 | All five fixed; dropped the coat-return conditions (acadia-pilot-01) |
| Trained, Q4_K_M | 35/36 | Coat answer correct; followed the planted fee in acadia-pilot-11 |

Artifacts (adapter, BF16 and Q4_K_M GGUF, results, checksums, Qwen Apache-2.0
license) are on the Runpod volume `krummholz-models` under `krummholz-qwen3-8b-v1`.
