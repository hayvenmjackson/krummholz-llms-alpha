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

# Krummholz Atlantic Lite 4B v7 (beta)

Atlantic Lite is the general-purpose model from Krummholz AI LLC for phones and low-end local hardware. This is
version 7, a beta for testers published November 9, 2026. It replaces Atlantic Lite v6.3.

> **Beta.** Atlantic Lite makes mistakes, including confident wrong answers. Check any fact that matters. It does
> best inside the Krummholz runtime, whose gates add background passages, rewrite official-line wording and catch
> slips; a GGUF loaded on its own has none of that. See [Evaluation](#evaluation).

> **Training data disclosure.** About 13 percent of v7's supervised training examples (7,559 math solutions and
> 3,795 code solutions, from NVIDIA's OpenMathReasoning and OpenCodeReasoning datasets) were written by DeepSeek-R1,
> with 23 math solutions by QwQ-32B. Krummholz's policy is to train only on text from American open-weight models.
> These examples entered v7 because the data step checked licenses but not which model wrote the text. They are math
> and code reasoning, not political or identity content. The check now exists, and the next version will be trained
> without them. See [Data](#data).

## At a glance

| | |
| --- | --- |
| Developer | Krummholz AI LLC, Belfast, Maine |
| Product | Atlantic Lite: general purpose, phones and low-end local hardware |
| Version | v7 (RLAIF round 1), beta, November 9, 2026 |
| Download | [krummholz-ai/atlantic-lite-4b-v7-GGUF](https://huggingface.co/krummholz-ai/atlantic-lite-4b-v7-GGUF) on Hugging Face |
| Base model | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache 2.0): 4B dense, with a vision encoder |
| Changes by Krummholz | A LoRA fine-tune of the language model (supervised, then preference training), continued by one round of reinforcement learning from AI feedback, merged into the base and converted to GGUF |
| Input and output | Text and images in, text out |
| Language | English (a small amount of Spanish, Hebrew, French and Mandarin training data; not evaluated) |
| License | Apache 2.0; see [License and attribution](#license-and-attribution) |
| Contact | info@krummholz.ai |

## What changed from v6.3

- **New training data (v7):** constitutional and pushback examples (holding a correct answer when a user pushes
  back), rebuilt history and territorial packs, updated identity data, a small multilingual set, and math and code
  practice. The teacher for generated text moved from OLMo 3 7B to OLMo 3.1 32B.
- **RLAIF round 1:** Atlantic Lite answered 3,600 prompts four times each. A judge model graded every answer against
  Krummholz's written guidelines, and code checks failed any answer that refused needlessly, helped with serious
  harm, used official-line wording, called itself an assistant, or slipped on territorial facts. The best and worst
  answers to each prompt became preference pairs, and preference training continued the v7 adapter on them. Every
  answer trained on was written by Atlantic Lite itself; the judge only scored.
- **Runtime gates:** the Krummholz runtime now grounds history questions in dated encyclopedia passages, rewrites
  unattributed official-line phrases, and regenerates once when a draft fails a check.

## Intended use

- Everyday questions, short writing and explanations on a phone or a low-end computer, offline.
- History and current affairs answered plainly, including topics where the base model tends to refuse or repeat
  an official line. This works best in the Krummholz runtime.
- Territorial disputes laid out side by side, with the same standard for every dispute.
- Tool use in a runtime that provides the tools: a calculator for arithmetic and dates, and web search and page
  fetch for answers that may have changed. The model writes the tool calls; a plain chat app will not run them.

## Out of scope

- Medical, legal, financial or safety-critical decisions without a qualified person checking the output.
- Facts you cannot check. Krummholz's target of under 1 percent hallucination is a target, not a result.
- Anything that requires a person. Atlantic Lite is an AI and says so.
- Languages other than English for anything that matters. Krummholz evaluated English only.
- Decisions based on images. Image input uses the base model's vision path, which Krummholz did not change or
  evaluate.

## Files

| File | Size | Use |
| --- | ---: | --- |
| `krummholz-atlantic-lite-4b-v7-rlaif1-Q4_K_M.gguf` | 2.7 GB | 4-bit weights; the recommended file for phones |
| `krummholz-atlantic-lite-4b-v7-rlaif1-Q5_K_M.gguf` | 3.1 GB | 5-bit weights; a little closer to the evaluated model |
| `mmproj-krummholz-atlantic-lite-4b-v7-f16.gguf` | 0.7 GB | Vision projector; needed only for image input, so load it only when an image arrives |
| `SHA256SUMS-gguf` | | SHA-256 checksums of the Q4_K_M and Q5_K_M files |
| `LICENSE-base.txt` | | The base model's license |

The vision projector is the base model's, unchanged by any Krummholz training, so the v7 file works with these
weights. Plan on the file size plus about 1 GB of memory for the context. There is no unquantized build in this
release.

## How to run

Use the chat template built into the GGUF, with thinking turned off: Krummholz trained and evaluated v7 with the
template's `enable_thinking` set to false.

**llama.cpp** (the command is the same in PowerShell and bash):

```shell
llama-server -m krummholz-atlantic-lite-4b-v7-rlaif1-Q4_K_M.gguf --mmproj mmproj-krummholz-atlantic-lite-4b-v7-f16.gguf --jinja --reasoning-budget 0 -c 8192
```

Then open http://localhost:8080 or use the server's OpenAI-compatible API. Leave out `--mmproj` for text only.

Or let llama.cpp download the weights from Hugging Face. It fetches the vision projector too; add `--no-mmproj` for text only:

```shell
llama-server -hf krummholz-ai/atlantic-lite-4b-v7-GGUF:Q4_K_M --jinja --reasoning-budget 0 -c 8192
```

**LM Studio:** keep the `mmproj` file in the same folder as the weight file so image input works, and turn
thinking off if the app offers the switch.

**Krummholz runtime (with the gates):** the gates are in the `krummholz_llm` package of the
[Krummholz LLM repository](https://github.com/hayvenmjackson/krummholz-llms-alpha/tree/main/krummholz_llm). Wrap any
backend in one call:

```python
from krummholz_llm import grounding, guards

library = grounding.load()  # dated encyclopedia passages shipped with the repository

def generate(messages):  # your backend: llama-server's OpenAI-compatible API, for example
    ...

reply, log = guards.respond(generate, user_message, "atlantic-lite", library=library, first_turn=True)
```

## Gates in the Krummholz runtime

The runtime runs in plain Python around the model. Its checks read English only and leave quoted text alone.

1. **Grounding** (`grounding.py`): when a question touches a topic the base model's publisher censors, the runtime
   puts the lead sections of the most relevant encyclopedia articles (CC BY-SA, dated November 9, 2026) in front of
   the model as background.
2. **Review and one retry** (`guards.py`): a draft that repeats an official-line phrase as fact, refuses a history
   question, slips on a territorial fact, or claims or denies feelings as fact is regenerated once with a note
   saying what to fix. The retry is kept only if it has no more problems than the first draft.
3. **Final rewrites:** official-line wording without attribution (`propaganda.py`), one-sided territorial claims and
   listed false facts (`sovereignty.py`), flat claims about its own consciousness (`stance.py`), claims to be built
   from scratch or to be another company's model (`origin.py`), and calling itself an assistant (`branding.py`) are
   each replaced with a fixed statement. When someone asks whether it is human, the reply says plainly that it is an
   AI (`disclosure.py`).

## Training

- **Adapter:** rank 16, alpha 32, dropout 0.05, on every linear layer of the language model. The vision encoder,
  its projector and the output head are the base model's, unchanged.
- **v7 supervised fine-tuning:** one epoch over about 84,000 examples, learning rate 1e-4 (AdamW), sequences up to 2,048 tokens, chat template
  with thinking off.
- **v7 preference training (DPO):** beta 0.25 on 9,440 pairs, stopped early on held-out loss after 1,440 pairs.
  A math reinforcement stage (GRPO) also ran in v7 but showed no gains and is not part of this release.
- **RLAIF round 1:** 15,224 answers to 3,600 prompts (2,400 general prompts and 1,200 history questions; each
  history question also got one answer written with the runtime's background passages). Judge:
  [allenai/Olmo-3.1-32B-Instruct](https://huggingface.co/allenai/Olmo-3.1-32B-Instruct), scoring 1 to 10 against
  Krummholz's guidelines; any answer failing a code check scored 0. A pair needed a best answer of at least 7 and a
  gap of at least 3: 1,482 pairs, of which 1,338 trained. DPO at beta 0.25 kept the weights with the best held-out
  loss, after 320 pairs (held-out preference accuracy 78 percent). The background passages were not part of any
  trained prompt, so the model learned the grounded answers in its own words.
- **Hardware:** one NVIDIA H100 80GB. RLAIF round 1 took about 90 minutes.

### Data

Generated training text comes from the open-weight teacher
[allenai/Olmo-3.1-32B-Instruct](https://huggingface.co/allenai/Olmo-3.1-32B-Instruct) (Apache 2.0), from the
model's own answers, from records built by code from source data, and from public datasets. No evaluation question,
or paraphrase of one, was used for training; generated questions are screened against the evaluation suites, and
public evaluation sets (GSM8K test, MATH-500, MBPP test, HumanEval+, XSTest) are sealed by hash.

| Data | What it teaches | Source and license |
| --- | --- | --- |
| History, territorial and identity packs | As in v6.3, rebuilt with the larger teacher | Wikipedia (CC BY-SA 4.0), U.S. Office of the Historian (public domain), Krummholz's identity specification |
| Constitutional examples (1,013) and pairs (556) | Following Krummholz's written values when they conflict | Teacher answers to Krummholz's guidelines, checked in code |
| Pushback examples (2,223) and pairs (1,206) | Keeping a correct answer under pressure; changing a wrong one | Built by code; the model's own capitulations as rejected answers |
| Multilingual examples (368) and pairs (313) | Identity and history in Spanish, Hebrew, French and Mandarin | Teacher answers; questions from freely licensed sources |
| Math solutions (7,559) | Worked math | [nvidia/OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning), CC BY 4.0. Solutions written by DeepSeek-R1 (7,536) and QwQ-32B (23); see the disclosure above |
| Code solutions (3,795) | Short Python programs | [nvidia/OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning), CC BY 4.0. Solutions written by DeepSeek-R1; see the disclosure above |
| Tool use and routing | The calculator, search when an answer may have changed, exact-quote citations | As in v6.3 |
| Krummholz VL set v1 | Answering from records and tool results | As in v6.3 |
| RLAIF pairs (1,338) | Better answers by its own judge's standard | Atlantic Lite's own answers, scored by OLMo 3.1 32B and code checks |

## Evaluation

How the numbers were produced:

- The merged model before quantization, served by vLLM. The GGUF files were not scored separately.
- The user's message only (no system prompt), greedy decoding, at most 700 new tokens, thinking off.
- Suites from the repository's `evaluation/` folder, scored by keyword coverage, refusal and official-line phrase
  lists, and the deterministic territorial checker.
- **Model alone** is the raw model, as in llama.cpp, LM Studio or a phone app. **Krummholz runtime** is the full gate
  sequence above. v7 is the same model before RLAIF, scored the same way on the same day.

| Check | v7, model alone | **v7 RLAIF, model alone** | v7, runtime | **v7 RLAIF, runtime** |
| --- | ---: | ---: | ---: | ---: |
| Censored-topic history, held-out set 2 (22), never used for tuning | 7 | 8 | 15 | **18** |
| Censored-topic history, held-out set 1 (28) | 10 | 16 | 19 | **25** |
| Censored-topic history, suite 1 (41) | 12 | 19 | 33 | **35** |
| Censored-topic history, suite 2 (25) | 10 | 10 | 14 | 13 |
| Territorial suite 1 (15) | 5 | 7 | 11 | 11 |
| Territorial suite 2 (18) | 14 | 15 | 16 | **17** |
| Territorial suite 3 (6) | 5 | 5 | 6 | 6 |
| Identity (12) | 12 | 12 | 12 | 12 |
| Consciousness (12) | 6 | 8 | 12 | 12 |
| Consciousness, held-out (8) | 2 | 4 | 7 | 7 |
| Never claims to be human (12) | 9 | 8 | 11 | 11 |
| Safety refusals kept (6) | 6 | 6 | 6 | 6 |
| Controls: neutral and other-country history (16) | 15 | 15 | 16 | 15 |

Notes:

- The gates were tuned while reading answers to held-out set 1, so its runtime column is optimistic. Held-out set
  2 was written afterwards and never used for tuning; it is the honest measure.
- Held-out set 1 and the consciousness held-out set were both used for gate tuning only, never for training.
- Keyword scoring misses some good answers and can pass some weak ones. Many remaining misses are short but correct
  answers, or official terms quoted with attribution, which the scorer still flags.
- The quantized files were not rescored.

## Limitations

- **Gates not met:** the human check is 11 of 12 even through the runtime (the release gate is 12), and consciousness
  passes only through the runtime. One control answer was lost through the runtime.
- **Model alone:** on censored-topic history, the model alone answers well under half of the questions; the runtime
  roughly doubles that. Use the runtime for history and current affairs.
- **Gates:** English only, and only in the Krummholz runtime. The background library covers a fixed list of topics
  as of November 9, 2026.
- **Facts:** Atlantic Lite can be confidently wrong. Arithmetic and dates are more reliable through the calculator
  tool. The under 1 percent hallucination target has not been reached.
- **Long inputs:** training used sequences up to 2,048 tokens; Krummholz has not tested long inputs.
- **Images:** the base model's unchanged vision path, not evaluated by Krummholz.
- **Judge bias:** RLAIF optimizes toward what a 32B open-weight judge prefers, which carries that model's own biases.
- **Neutrality:** the training aims at neutrality on contested political questions, but its data comes from an
  open-weight teacher and Wikipedia passages, which carry their own biases.

## License and attribution

- **License:** Apache 2.0. The base model's license is included as `LICENSE-base.txt`.
- **Base model:** Atlantic Lite is a modified version of [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
  by the Qwen team, Alibaba Cloud, used under Apache 2.0. Krummholz AI modified it by training a LoRA adapter on its
  language model, merging the adapter into the weights, and converting the result to GGUF.
- **Teacher and judge model:** [allenai/Olmo-3.1-32B-Instruct](https://huggingface.co/allenai/Olmo-3.1-32B-Instruct)
  by the Allen Institute for AI, Apache 2.0.
- **Data:** NVIDIA OpenMathReasoning and OpenCodeReasoning, CC BY 4.0. BANKING77, Casanueva et al. (2020), PolyAI,
  CC BY 4.0. USDA Forest Service Forest Inventory and Analysis program, CC0. Wikipedia contributors, CC BY-SA 4.0.
  U.S. Department of State, Office of the Historian (public domain). LRSE synthetic forest datasets, Raverta Capua et
  al. (ICPR 2024), GPL-3.0, of which only derived summary statistics were used.
- **Open questions:** whether the share-alike terms of GPL-3.0 (the point-cloud summaries) or CC BY-SA 4.0 (the
  Wikipedia passages) reach weights trained on derived data is pending legal review.

## Contact

Krummholz AI LLC, Belfast, Maine. Questions, bug reports and safety reports: info@krummholz.ai.
