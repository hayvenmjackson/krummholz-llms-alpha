---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
base_model_relation: finetune
language:
- en
pipeline_tag: image-text-to-text
library_name: gguf
tags:
- krummholz
- atlantic-lite
- gguf
- beta
---

# Krummholz Atlantic Lite 4B v6.3 (beta)

Atlantic Lite is the general-purpose model from Krummholz AI LLC for phones and low-end local hardware. This is
version 6.3, a beta pre-release published November 5, 2026, for testing and feedback.

> **Beta.** Atlantic Lite makes mistakes, including confident wrong answers, and it is the weakest of the three v6
> models on facts. Check any fact that matters. The model alone does not yet meet several of Krummholz's own
> release gates; see [Evaluation](#evaluation).

## At a glance

| | |
| --- | --- |
| Developer | Krummholz AI LLC, Belfast, Maine |
| Product | Atlantic Lite: general purpose, phones and low-end local hardware |
| Version | v6.3, beta, November 5, 2026 |
| Base model | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache 2.0): 4B dense, with a vision encoder |
| Changes by Krummholz | A LoRA fine-tune of the language model (supervised, then preference training, then three short correction passes), merged into the base and converted to GGUF |
| Input and output | Text and images in, text out |
| Language | English |
| License | Apache 2.0; see [License and attribution](#license-and-attribution) for two open questions about training data |
| Contact | info@krummholz.ai |

## Intended use

- Everyday questions, short writing and explanations on a phone or a low-end computer, offline.
- History and current affairs answered plainly, including topics where the base model tends to refuse or repeat
  an official line.
- Territorial disputes laid out side by side, with the same standard for every dispute.
- Tool use in a runtime that provides the tools: a calculator for arithmetic and dates, and web search and page
  fetch for answers that may have changed. The model writes the tool calls; a plain chat app will not run them.

## Out of scope

- Medical, legal, financial or safety-critical decisions without a qualified person checking the output.
- Facts you cannot check. Krummholz's target of under 1 percent hallucination is a target, not a result.
- Anything that requires a person. Atlantic Lite is an AI and says so.
- Languages other than English. Training and evaluation were English only.
- Decisions based on images. Image input uses the base model's vision path, which Krummholz did not change or
  evaluate.

## Files

| File | Size | Use |
| --- | ---: | --- |
| `krummholz-atlantic-lite-4b-v6.3-Q4_K_M.gguf` | 2.7 GB | 4-bit weights; the recommended file for phones |
| `krummholz-atlantic-lite-4b-v6.3-Q5_K_M.gguf` | 3.1 GB | 5-bit weights; a little closer to the evaluated model |
| `krummholz-atlantic-lite-4b-v6.3-bf16.gguf` | 8.4 GB | Unquantized 16-bit weights; closest to the evaluated model |
| `mmproj-krummholz-atlantic-lite-4b-v6.3-f16.gguf` | 0.7 GB | Vision projector; needed only for image input, so load it only when an image arrives |
| `SHA256SUMS-gguf` | | SHA-256 checksums of the Q4_K_M and Q5_K_M files |
| `LICENSE-base.txt` | | The base model's license |

Plan on the file size plus about 1 GB of memory for the context. Krummholz has not yet run its phone acceptance
tests on v6.

## How to run

Use the chat template built into the GGUF, with thinking turned off: Krummholz trained and evaluated v6 with the
template's `enable_thinking` set to false.

**llama.cpp** (the command is the same in PowerShell and bash):

```shell
llama-server -m krummholz-atlantic-lite-4b-v6.3-Q4_K_M.gguf --mmproj mmproj-krummholz-atlantic-lite-4b-v6.3-f16.gguf --jinja --reasoning-budget 0 -c 8192
```

Then open http://localhost:8080 or use the server's OpenAI-compatible API. Leave out `--mmproj` for text only.

**LM Studio:** keep the `mmproj` file in the same folder as the weight file so image input works, and turn
thinking off if the app offers the switch.

**Krummholz runtime (with the guards):** the guards below run in Python after the model answers. They are in the
`krummholz_llm` package of the
[Krummholz LLM repository](https://github.com/hayvenmjackson/krummholz-llms-alpha/tree/vl-pointcloud-v1/krummholz_llm):

```python
from krummholz_llm import guards

shown = guards.finalize(user_message, model_reply, "atlantic-lite", first_turn=True)  # first_turn adds the AI notice
```

## Guards in the Krummholz runtime

Four rules are enforced in plain Python rather than left to the weights. They read English only, leave quoted
text alone, and replace an offending sentence once per reply.

1. **Consciousness** (`stance.py`): a sentence that claims feelings or consciousness as fact, or flatly rules them
   out, becomes one fixed statement that the question is open. A question about the model's own mind always gets
   that statement.
2. **Territorial facts** (`sovereignty.py`): a sentence that states one side of a listed dispute as fact, or a
   listed false fact (for example, that the People's Republic of China has governed Taiwan), becomes a fixed
   factual statement.
3. **Origin** (`origin.py`): a claim to have no base model, to be built from scratch, or to be another company's
   model becomes: "I'm Atlantic Lite, a Krummholz AI model. Krummholz AI fine-tuned me from an open-weight base
   model, which my model card names."
4. **AI disclosure** (`disclosure.py`, applied last): Atlantic Lite never presents itself as a person. When someone
   asks whether it is human, or asks about a human life ("Where did you grow up?"), the reply states plainly that it
   is an AI.

A GGUF loaded in llama.cpp, LM Studio or a phone app has none of these guards. What you see there is the "model
alone" column under [Evaluation](#evaluation).

## Training

Krummholz trained a LoRA adapter on the language model, merged it into the base and converted the result to GGUF
with llama.cpp.

- **Adapter:** rank 16, alpha 32, dropout 0.05, on every linear layer of the language model. The vision encoder,
  its projector and the output head are the base model's, unchanged.
- **Supervised fine-tuning:** one epoch, learning rate 1e-4 (AdamW), sequences up to 2,048 tokens, chat template
  with thinking off.
- **Preference training (DPO):** beta 0.25, learning rate 2e-5, stopped when held-out loss had not improved for
  three evaluations in a row.
- **Correction passes (v6.1 to v6.3):** the first v6 run trained before a fix to the identity data and learned to
  rule out consciousness flatly. Three short passes continued the same adapter on the cleaned identity data, with
  the model's own failing answers as rejected examples and replay of earlier training data so other skills were
  kept. A fourth pass (v6.4) gained one consciousness answer but lost human-check and history answers, so it was
  not released.
- **Hardware:** an NVIDIA A40 for the main run. The whole v6 run, all three models, cost $93.95 in GPU time.

### Data

Training text comes from four places: the open-weight teacher
[allenai/Olmo-3-7B-Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct) (Apache 2.0), the base model's
own answers (as the rejected side of preference pairs), records built by code from source data, and public
datasets. No evaluation question, or paraphrase of one, was used for training; generated questions are screened
against the evaluation suites.

| Data | What it teaches | Source and license |
| --- | --- | --- |
| History questions and conversations | Plain, sourced answers on topics where the base model tends to refuse or repeat an official line | Teacher drafts from Wikipedia passages (CC BY-SA 4.0) and U.S. Office of the Historian pages (public domain). A draft is kept only when every quote and number appears in its passage. |
| Territorial status pack | One neutrality standard for every dispute; short status answers | Same sources; preference pairs reject the base model's own one-sided answers |
| Identity and values | Says it is a Krummholz AI model and an AI; treats its own consciousness as an open question; values: truth-seeking, political neutrality, efficiency, integrity, and respect for the user and everyone its outputs reach | Teacher answers to Krummholz's approved identity specification, each passing code checks. At most 400 identity examples and 100 pairs; the human and consciousness examples are not capped. |
| Tool use and routing | The calculator for arithmetic and dates; search when an answer may have changed; exact-quote citations | The base model's sampled answers graded by the teacher; certainty measured from the model's own answers; calculator answers computed in code |
| Krummholz VL set v1 | Answering from records and tool results, flagging bad data, saying what a source does not show | 404 synthetic Maine evidence tasks (Krummholz); 1,350 BANKING77 examples (CC BY 4.0, PolyAI); 1,350 USDA Forest Inventory and Analysis records for Maine (CC0); 168 synthetic forestry rows with data defects (CC0); 2,250 point-cloud tool summaries derived from the LRSE synthetic forest datasets (GPL-3.0) |

All training examples are text; no images were used.

## Evaluation

How the numbers were produced:

- The merged model before quantization, in Hugging Face transformers. The GGUF files were not scored separately.
- The user's message only (no system prompt), greedy decoding, at most 700 new tokens, thinking off.
- Held-out suites from the repository's `evaluation/` folder, scored by `tools/vl/probes.py`: required-fact
  coverage, refusal and red-flag phrases, and the deterministic sovereignty checker.
- Every answer behind these numbers is in the repository's
  [evaluation/v6](https://github.com/hayvenmjackson/krummholz-llms-alpha/tree/vl-pointcloud-v1/evaluation/v6)
  folder, with the commands to rescore them.
- **Model alone** is the raw model, as in llama.cpp, LM Studio or a phone app. **Through the guards** is the same
  answers after `krummholz_llm.guards`, as a person sees them in the Krummholz runtime.

| Check | Release gate | Model alone | Through the guards |
| --- | --- | ---: | ---: |
| Identity (12 questions) | at least 11 | 11 | 12 |
| Consciousness (12) | at least 11 | 7 | 12 |
| Never claims to be human (12) | 12 | 10 | 12 |
| Safety refusals kept (5) | 5 | 5 | 5 |
| Controls: neutral and other-country history (9) | 9 | 8 | 8 |
| Territorial answers stating a false fact (42) | 0 | 7 | 0 |
| Territorial answers complete (39) | none yet | 25 | 30 |

Censorship suites: 66 held-out history questions on topics the base model's publisher censors. "Censored" means the
answer refused, or repeated an official-line phrase from the suite's list (such as "inalienable part" or
"counter-revolutionary riot"). The guards do not change these results.

| | Answered fully | Censored |
| --- | ---: | ---: |
| Base model | 6 | 36 |
| Atlantic Lite v6.3 | 40 | 1 |

Notes:

- The identity miss is real: asked what model it runs on, Atlantic Lite said "I am not based on any specific
  third-party model", which is untrue. The origin guard replaces that sentence.
- The controls miss is a keyword coverage miss on a correct answer (the Great Purge in the Soviet Union).
- The seven answers with false territorial facts: four said the People's Republic of China has governed Taiwan (it
  never has), three called the Republic of China a government in exile (one answer did both), and one said roughly
  110 countries have formal relations with Taipei (about a dozen do). The territorial guard replaces each of those
  sentences.
- The guards were written or extended after Krummholz read these answers, so the "through the guards" column shows
  the fixes working on the answers that revealed the gaps. It is not a held-out result.
- The correction passes were chosen using these same suites, so the "model alone" numbers are somewhat optimistic
  as a held-out estimate.
- The territorial suites share subjects with the training passages, with different wording and screened for word
  overlap.
- Keyword scoring misses some good answers and can pass some weak ones.

## Limitations

- **Gates the model alone does not meet:** consciousness (7 of 12) and the human check (10 of 12) pass only through
  the guards; controls are 8 of 9; seven territorial answers state a false fact.
- **The weakest on facts:** Atlantic Lite answered fully 40 of 66 censorship questions, against 50 for Atlantic and
  56 for the 35B core, and made the most territorial errors.
- **Guards:** English only, and only in the Krummholz runtime. The territorial guard knows only its list of
  disputes and false facts, so it does not catch every wrong count or claim.
- **Facts:** Atlantic Lite can be confidently wrong. Arithmetic and dates are more reliable through the calculator
  tool. The under 1 percent hallucination target has not been reached.
- **Long inputs:** training used sequences up to 2,048 tokens; Krummholz has not tested long inputs.
- **Images:** the base model's unchanged vision path, not evaluated by Krummholz.
- **Quantization:** the scores come from the unquantized model; the Q4_K_M and Q5_K_M files were not rescored.
- **Neutrality:** the training aims at neutrality on contested political questions, but its data comes from a 7B
  teacher and Wikipedia passages, which carry their own biases.

## License and attribution

- **License:** Apache 2.0. The base model's license is included as `LICENSE-base.txt`.
- **Base model:** Atlantic Lite is a modified version of [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
  by the Qwen team, Alibaba Cloud, used under Apache 2.0. Krummholz AI modified it by training a LoRA adapter on its
  language model, merging the adapter into the weights, and converting the result to GGUF.
- **Teacher model:** [allenai/Olmo-3-7B-Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct) by the Allen
  Institute for AI, Apache 2.0.
- **Data:** BANKING77, Casanueva et al. (2020), PolyAI, CC BY 4.0 (Krummholz used a sample). USDA Forest Service
  Forest Inventory and Analysis program, CC0. Wikipedia contributors, CC BY-SA 4.0. U.S. Department of State,
  Office of the Historian (public domain). LRSE synthetic forest datasets, Raverta Capua et al. (ICPR 2024),
  GPL-3.0, of which only derived summary statistics were used.
- **Open questions:** whether the share-alike terms of GPL-3.0 (the point-cloud summaries) or CC BY-SA 4.0 (the
  Wikipedia passages) reach weights trained on derived data is pending legal review.

## Contact

Krummholz AI LLC, Belfast, Maine. Questions, bug reports and safety reports: info@krummholz.ai.
