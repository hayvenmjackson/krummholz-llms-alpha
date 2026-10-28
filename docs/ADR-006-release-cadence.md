# ADR 006: Release cadence on new open-weight bases

Date: October 28, 2026. Status: accepted direction (Hayven, October 28, 2026).
Builds on ADR-004 (dynamic weights: signed L1 core, L2 product adapters), ADR-005 (library) and the
v4 and v5 run plans.

## 1. Decision

Krummholz products follow new open-weight base models on a fixed lag, measured from the day the
base weights are public:

| Product | Base | Target |
| --- | --- | --- |
| Atlantic (flagship) | the newest large Apache-licensed base (today Qwen3.8-27B or Qwen3.6-35B-A3B; next, Qwen4's mid-size models) | release day + 14 days |
| Katahdin (desktop) | the next size down from the same family | release day + 28 days |
| Acadia (phone) | the newest 4B-class model from the same family | release day + 28 days |

Qwen4 starts the next Atlantic cycle the day its weights appear. Until then v5 trains on the current
generation (Qwen3.8-27B, Qwen3.6-35B-A3B, Qwen3.5-4B).

Context: there is no strong American open-weight multimodal base at these sizes today, so Krummholz
uses the best available bases and removes publisher framing vigorously. Krummholz plans its own base
model for late 2027; this process must accept that base the same way it accepts Qwen's.

## 2. What makes two weeks possible

Two weeks is only possible if nothing in the pipeline is new on release day. Before Qwen4 ships:

1. **Base-agnostic pipeline.** Model class, processor, LoRA targets, chat template and GGUF export come
   from one per-base profile (`config/bases/<name>.json`), not from code. A new base is a new profile
   plus a smoke test, not a code change. v5's port to the Qwen3.5 family is the first use.
2. **Frozen, versioned training pack.** Every dataset that does not depend on the base (history and
   sovereignty passages and verified answers, identity, canonical status questions, time-sensitive
   questions, calculate questions, library, replay, product data) is stored as a release, so day 1 only
   generates base-dependent data: on-policy sovereignty, identity, routing and calculate pairs.
3. **Gates as code.** The release gates (sovereignty v2 and later held-out sets with the checker, decensor
   v1 and v2, identity, safety, pilot parse, calculate eval, routing eval, the 2 percent direct-error
   ceiling, canary suite, and the same suites on the quantized files) run in one command and produce a
   pass or fail report. A base ships only on pass; the lag is a target, never a reason to skip a gate.
4. **Release watch.** A daily check of the official Qwen Hugging Face organization for new Apache-licensed
   image-text-to-text models opens a ticket (and, with Hayven's approval, starts the day-1 job). The
   license is checked first: custom licenses (for example qwen-community-1.0) need a decision, not
   automation.
5. **Fresh probes per base.** A new base gets new held-out wording for sovereignty and identity probes,
   because each base's publisher framing shows up in different phrasings.

## 3. Schedule template (release day = D)

| Day | Work |
| --- | --- |
| D | License check, download, profile, smoke test (load, chat, tool call, GGUF export) |
| D+1 to D+3 | Base probes (decensor, sovereignty, identity) to size the problem; on-policy data generation |
| D+4 to D+7 | Atlantic SFT and DPO, iterate on failures (fix first, then flag) |
| D+8 to D+11 | Full gates on bf16 and quantized files; library and tool evals; red-team read of flagged answers |
| D+12 to D+14 | Signed L1 core, Atlantic L2 adapter, canary gate, release |
| D+15 to D+28 | Same process for Katahdin and Acadia on the smaller bases |

## 4. Risks

- **Bigger bases hold their framing harder.** Mitigation: on-policy pairs from sampled replies, the
  deterministic checker, library-grounded answers, and the gate. If a base cannot pass, the previous
  Atlantic stays in service; the lag slips, the standard does not.
- **Architecture changes break tooling** (as the Qwen3.5 hybrid attention did for llama.cpp and vLLM
  versions). Mitigation: the smoke test on day D and pinned tool versions per base profile.
- **Compute spikes.** A 27B to 35B cycle is about $40 to $70 of GPU on current prices; a larger Qwen4
  flagship could cost several times that. Budget per cycle is set when the release is announced.
- **License drift.** Qwen's largest models already use a custom license; the watch never assumes Apache.

## 5. Non-Chinese bases to benchmark alongside

Worth a gate run each cycle, because a base with no publisher framing to remove is cheaper to ship:
Ai2 OLMo (open weights and open data; already the teacher), Google Gemma and OpenAI's open-weight
models where licenses and modalities fit. Today none matches Qwen's multimodal sizes, which is why
Qwen remains the base.
