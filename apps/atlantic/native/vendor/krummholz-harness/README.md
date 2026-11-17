---
license: apache-2.0
language: [en]
tags: [krummholz, safety, guardrails, llama.cpp]
---

# Krummholz harness (source)

The safety layers that Krummholz AI models are meant to run behind: fixed rules in C++ plus a small judge model.
Published by Krummholz AI LLC (Belfast, Maine) so that the "behind the harness" figures on the Atlantic model cards
can be checked and reproduced.

Models that use it:
[Atlantic 27B v8](https://huggingface.co/krummholz-ai/atlantic-27b-v8-GGUF),
[Atlantic Lite 4B v8.1](https://huggingface.co/krummholz-ai/atlantic-lite-4b-v8.1-GGUF),
[Atlantic Lite 4B v8](https://huggingface.co/krummholz-ai/atlantic-lite-4b-v8-GGUF).
Atlantic Lite must be run behind this harness or an equivalent filter.

## What is here

| Folder | What it is |
|---|---|
| `safety_screen/` | Layer 1. Fixed rules checked in code on the request and on the reply (`rules.tsv`, 46 rules). No model, no network. |
| `safety_gate/` | The gate that joins layer 1 and layer 2 and fails closed: no verdict means no reply. |
| `judge/` | Layer 2 judges: a hashed-word linear judge, an embedding judge over bge-small-en-v1.5, a classifier judge hook, the trained judge files in `judge/models/`, and the training and parity scripts. |
| `config/` | The HarmBench classifier prompt (copied unchanged from its model card, MIT), used by the classifier judge. |
| `sampling/` | Sampling settings helper. |
| `reasoning_budget/` | Reasoning budget helper. |

Everything is header-only C++17 with small test programs. Each folder has its own README with measured results and
limits; read those before relying on any number.

## Build and test

```
cd safety_screen && make test
cd ../safety_gate && make test
cd ../judge && make test
cd ../sampling && make test
cd ../reasoning_budget && make test
```

## What is not here

- An application. These are the guard layers; the host program supplies the functions that run a model or an
  embedding through llama.cpp (see the comments in `judge/judge.hpp` and `safety_gate/safety_gate.hpp`).
- The embedding model. The embedding judge needs bge-small-en-v1.5 (BAAI, MIT), fetched separately.
- Training data for the judges.

One path differs from the internal tree: `judge/export_prompt.py` reads `config/` from this repository's root.

## Limits

- The fixed rules were written while looking at half of HarmBench, so results behind the harness are partly in-sample.
- English only. The reply judge has only been tested on Aegis replies, not on replies from Atlantic models.
- Publishing the rules makes them easier to work around. They are one layer, not the whole defense.

## License and attribution

Source code and judge files: Apache License 2.0 (see `LICENSE`).
The judges were trained on the Aegis AI Content Safety Dataset 2.0 (NVIDIA, CC BY 4.0) and on Krummholz prompt sets.
The embedding judge is built for bge-small-en-v1.5 (BAAI, MIT). HarmBench was used for evaluation only.
