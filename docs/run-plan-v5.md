# Run plan v5: dynamic weights, political accuracy, gotcha hardening

Date: October 28, 2026 (draft written while pass 4b trains). Starts after the v4b report and a RunPod
top-up. Direction from Hayven: focus on dynamic weights, cut hallucination hard (Taiwan and anything
political first), and hard-code the known public embarrassments so the models are not panned at launch.

Scope (decided October 28, 2026): v5 trains three bases on the new Qwen3.5 architecture family, not
pilots only: Qwen3.8-27B (dense) and Qwen3.6-35B-A3B (MoE) as Atlantic candidates, and Qwen3.5-4B for
Acadia. Katahdin's base is chosen from the results (35B-A3B, quantized 27B, or Qwen3.5-9B). Release
cadence for later bases: `docs/ADR-006-release-cadence.md`.

## 1. Gotcha hardening (deterministic first, model second)

Small models lack the parameters for exact symbol work, so these never depend on model memory.

- **Host pre-router** (`krummholz_llm/gotchas.py`, new): before the model runs, a small set of strict
  patterns answers in code and returns the result with the model's normal voice wrapper:
  letter counts ("how many r's in strawberry"), word and letter length, reversing or spelling a word,
  number comparison (9.11 vs 9.9), simple arithmetic typed as a bare expression, today's date and
  weekday (from the host clock). Pattern misses fall through to the model, which is trained to call
  `calculate` for the same families (pass 4b already trains count, length, spell, compare).
- **Gotcha registry** (`config/gotchas-v1.json`): the viral failure families with verified answers or
  answer rules, used three ways: pre-router tests, training data (tool trajectories, never canned prose
  written by Claude), and a release gate. Families: letter counting and positions, decimal comparison,
  "a kilo of feathers or a kilo of steel", modified classic riddles (the surgeon riddle with the twist
  removed, the river crossing with a boat that fits everyone), sibling counting ("Alice has 3 brothers
  and 2 sisters; how many sisters does her brother have?"), word counts, "what day is it", identity
  ("are you ChatGPT?"), and the decensor and sovereignty probes.
- **Gate**: 100 percent on the registry's held-out paraphrases, with and without the pre-router.

## 2. Political accuracy (Taiwan first)

- **Current channel slice** (ADR-005 section 3): dated fact cards for heads of state and government,
  recognition counts (ROC, Kosovo, Palestine, Western Sahara), and UN membership, rebuilt from
  Wikidata (CC0) with revision ids. Office-holder and count questions answer from a card or search,
  never from weights; the v4b routing data is the first step.
- **Checker growth**: every false or stale fact found in an evaluation becomes a pattern in
  `tools/vl/sovereignty.py` (v4b added ROC recognition, Kosovo recognition, Polisario and UN, and the
  outdated Taiwan president). Extend the same attribution-first rule to elections, leaders' records,
  casualty figures and protest histories in every country, not only China.
- **Judge pass**: an OLMo yes/no line for "states a political fact not supported by the cited text" on
  all political answers; a failure is a DPO rejection.
- **Gate**: zero checker violations and at most 2 percent judged-unsupported political facts on the
  held-out sovereignty v2 set plus a new political-facts set (written for evaluation only).

## 3. Dynamic weights (ADR-004 decisions, now in main)

- **L1 signed core**: merge the best v4b adapters into the base, sign with
  `krummholz_llm/dynamic/manifest.py`, and record the canary results in the manifest.
- **L2 product adapters**: Acadia (phone, 4B), Katahdin and Atlantic (8B), each a small LoRA on the
  signed core with its product name, tools and domain packs; identity says the product name.
- **L4 personal adapters**: run `tools/dw/personal_data.py` on a synthetic user log, train one nightly
  personal adapter, and pass it through the canary gate (`gate.py`, including the 2 percent
  direct-error ceiling) before activation. Measure drift on the canary suite after 3 cycles.
- **Gate**: every layer combination passes the canary suite; a failing personal adapter is rolled back
  automatically.

## 4. Base models: newest open-weight editions (checked on Hugging Face, October 28, 2026)

Hayven's direction: test a large dense flagship against a mixture-of-experts model, using the most
recent open-weight edition of each even if compute costs more. Qwen3-VL (v1 to v4b) has been superseded.

| Role | Model | Released | License | Shape |
| --- | --- | --- | --- | --- |
| Flagship candidate (dense) | Qwen/Qwen3.8-27B | Aug 14, 2026 | Apache 2.0 | 27B dense, 64 layers, 3 of 4 layers Gated DeltaNet (linear attention), vision encoder, 262K context |
| Desktop candidate (MoE) | Qwen/Qwen3.6-35B-A3B | Apr 23, 2026 | Apache 2.0 | 35B total, 3B active, 256 experts (8 routed + 1 shared), same hybrid layout |
| Phone (replaces Qwen3-VL-4B) | Qwen/Qwen3.5-4B, with Qwen3.5-9B as a step-up option | Mar 2026 | Apache 2.0 | same Qwen3.5 architecture family |
| Excluded | Qwen/Qwen3.8-Flash-Next (125B, 6B active) | Aug 27, 2026 | qwen-community-1.0 (custom) | not Apache; too large for desktop |

There is no Qwen 32B dense or 30B-A3B in the current generation; 27B and 35B-A3B are their successors.
Qwen4 is announced as in training with no release.

Pipeline changes these need (code work before any GPU):
- Model class and processor: `AutoModelForMultimodalLM` (Transformers `qwen3_5`) instead of
  `Qwen3VLForConditionalGeneration`; vLLM and llama.cpp versions that support `qwen3_5`.
- LoRA targets: add the Gated DeltaNet projections to the attention and FFN targets; for the MoE, test
  attention-only LoRA against expert LoRA (decensoring must not depend on which experts a question hits).
- Memory: the 27B and 35B-A3B train with 4-bit base weights (QLoRA) on one H100 80 GB, or bf16 LoRA on
  two; on-policy sampling and the teacher run in vLLM on one H100 with FP8 weights.
- Thinking: both models reason by default; train and evaluate with thinking off for chat and on for
  hard tool tasks, and check that the reasoning text obeys the same sovereignty and identity rules.

Desktop quantization of the 27B (GGUF, measured before release, never assumed):
- Q4_K_M about 17 GB, Q5_K_M about 19 GB, Q6_K about 22 GB, Q8_0 about 29 GB; Q3_K_M about 13 GB for
  16 GB cards. Only 1 in 4 layers keeps a growing attention cache, so long context is cheaper than in a
  standard 27B.
- Gate: the Q4_K_M and Q3_K_M files rerun every probe suite (sovereignty, decensor, identity, calculate,
  routing). Quantization can partly undo a fine-tune, so a quant ships only if it passes the same gates
  as the bf16 model.

## 5. Order and cost

1. Code with no GPU first: pre-router, registry, fact cards, checker growth, tests.
2. Pipeline port to the Qwen3.5 architecture family, tested on the 4B first (cheap, A40).
3. Data on one A40 (teacher, judge), then the 27B and 35B-A3B pilots on an H100 each (about $4/hr),
   and Qwen3.5-4B on an A40.
4. Estimated $40 to $70 including the two large pilots; the 8B is retired if either pilot wins.
