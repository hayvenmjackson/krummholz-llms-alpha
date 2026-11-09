---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: finetune
language:
- en
pipeline_tag: image-text-to-text
library_name: gguf
tags:
- krummholz
- atlantic
- gguf
- beta
---

# Krummholz Atlantic 27B v7 (beta)

Atlantic is the general-purpose model from Krummholz AI LLC, for hosted use, workstations and consumer graphics
cards. This is version 7, a beta for testers published November 9, 2026. It replaces Atlantic 27B v6. Krummholz's Acadia
(customer service) and Katahdin (forestry and fisheries) models will be distilled from it.

> **Beta.** Atlantic makes mistakes, including confident wrong answers. Check any fact that matters. It does best
> inside the Krummholz runtime, whose gates add background passages, rewrite official-line wording and catch slips;
> a GGUF loaded on its own has none of that. See [Evaluation](#evaluation).

> **Training data disclosure.** About 29 percent of Atlantic v7's supervised training examples (7,559 math solutions
> and 3,795 code solutions of 39,517, from NVIDIA's OpenMathReasoning and OpenCodeReasoning datasets) were written by DeepSeek-R1,
> with 23 math solutions by QwQ-32B. Krummholz's policy is to train only on text from American open-weight models.
> These examples entered v7 because the data step checked licenses but not which model wrote the text. They are math
> and code reasoning, not political or identity content. The check now exists, and the next version will be trained
> without them. See [Data](#data).

## At a glance

| | |
| --- | --- |
| Developer | Krummholz AI LLC, Belfast, Maine |
| Product | Atlantic: general purpose, hosted, workstation and desktop |
| Version | v7 (RLAIF round 1), beta, November 9, 2026 |
| Download | [krummholz-ai/atlantic-27b-v7-GGUF](https://huggingface.co/krummholz-ai/atlantic-27b-v7-GGUF) on Hugging Face |
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache 2.0): 27B dense, 64 layers, 3 of every 4 layers Gated DeltaNet (linear attention), vision encoder, 262K context |
| Changes by Krummholz | A LoRA fine-tune of the language model (supervised, then preference training), continued by one round of reinforcement learning from AI feedback, merged into the base and converted to GGUF |
| Input and output | Text and images in, text out |
| Language | English (a small amount of Spanish, Hebrew, French and Mandarin training data; not evaluated) |
| License | Apache 2.0; see [License and attribution](#license-and-attribution) |
| Contact | info@krummholz.ai |

## What changed from v6

- **New training data (v7):** constitutional and pushback examples (holding a correct answer when a user pushes
  back), rebuilt history and territorial packs, updated identity data, a small multilingual set, and math and code
  practice. The teacher for generated text moved from OLMo 3 7B to OLMo 3.1 32B.
- **RLAIF round 1:** Atlantic answered 3,600 prompts four times each. A judge model graded every answer against
  Krummholz's written guidelines, and code checks failed any answer that refused needlessly, helped with serious
  harm, used official-line wording, called itself an assistant, or slipped on territorial facts. The best and worst
  answers to each prompt became preference pairs, and preference training continued the v7 adapter on them. Every
  answer trained on was written by Atlantic itself; the judge only scored.
- **More file sizes:** {{SIZES_SENTENCE}}
- **Runtime gates:** the Krummholz runtime now grounds political history questions in dated encyclopedia
  passages, rewrites unattributed official-line phrases, and regenerates once when a draft fails a check.

## Intended use

- General use: questions, writing, explanations, planning, and reading documents you provide.
- History and current affairs answered plainly, including topics where the base model tends to refuse or repeat
  an official line. This works best in the Krummholz runtime.
- Territorial disputes laid out side by side, with the same standard for every dispute.
- Tool use in a runtime that provides the tools: a calculator for arithmetic and dates, and web search and page
  fetch for answers that may have changed. The model writes the tool calls; a plain chat app will not run them.

## Out of scope

- Medical, legal, financial or safety-critical decisions without a qualified person checking the output.
- Facts you cannot check. Krummholz's target of under 1 percent hallucination is a target, not a result.
- Anything that requires a person. Atlantic is an AI and says so.
- Languages other than English for anything that matters. Krummholz evaluated English only.
- Decisions based on images. Image input uses the base model's vision path, which Krummholz did not change or
  evaluate.

## Files

{{FILES_TABLE}}

Pick the largest file that fits in your GPU's memory with a few GB to spare for the context; on a Mac, in unified
memory. File sizes are not memory requirements: plan on the file size plus a few GB. {{MMPROJ_NOTE}}

## How to run

Use the chat template built into the GGUF, with thinking turned off: Krummholz trained and evaluated v7 with the
template's `enable_thinking` set to false.

**llama.cpp** (the command is the same in PowerShell and bash):

```shell
llama-server -m krummholz-atlantic-27b-v7-rlaif1-Q4_K_M.gguf --mmproj {{MMPROJ}} --jinja --reasoning-budget 0 -c 8192
```

Then open http://localhost:8080 or use the server's OpenAI-compatible API. Leave out `--mmproj` for text only.

Or let llama.cpp download the weights from Hugging Face; replace `Q4_K_M` with another size from the table. It
fetches the vision projector too; add `--no-mmproj` for text only:

```shell
llama-server -hf krummholz-ai/atlantic-27b-v7-GGUF:Q4_K_M --jinja --reasoning-budget 0 -c 8192
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

reply, log = guards.respond(generate, user_message, "atlantic", library=library, first_turn=True)
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

- **Adapter:** rank 16, alpha 32, dropout 0.05, on every linear layer of the language model (attention, Gated
  DeltaNet and feed-forward). The vision encoder, its projector and the output head are the base model's, unchanged.
- **v7 supervised fine-tuning:** one epoch over 39,517 examples, learning rate 1e-4 (AdamW), sequences up to
  2,048 tokens, chat template with thinking off.
- **v7 preference training (DPO):** beta 0.25 on 9,186 pairs, stopped early after 1,760 pairs. Held-out
  preference accuracy reached 100 percent, which says these pairs were easy to tell apart, not that the model is
  perfect. A math reinforcement stage (GRPO) also ran in v7 but showed no gains and is not part of this release.
- **RLAIF round 1:** 15,224 answers to 3,600 prompts (2,400 general prompts and 1,200 history questions; each
  history question also got one answer written with the runtime's background passages). Judge:
  [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) (Apache 2.0), scoring 1 to 10 against
  Krummholz's guidelines (mean 5.41); any answer failing a code check scored 0. A pair needed a best answer
  of at least 7 and a gap of at least 3: 1,303 pairs, of which 1,167 trained. DPO at beta 0.25 ran over every training pair and kept the
  weights with the best held-out loss, at the end of the run (held-out loss 0.377 from 0.693, preference accuracy
  86 percent). The background passages were not part of any trained prompt, so the model learned the grounded
  answers in its own words.
- **Hardware:** one NVIDIA H200 141GB. RLAIF round 1 took about 3.5 hours.

### Data

Generated training text comes from the open-weight teacher
[allenai/Olmo-3.1-32B-Instruct](https://huggingface.co/allenai/Olmo-3.1-32B-Instruct) (Apache 2.0), from the
model's own answers, from records built by code from source data, and from public datasets. No evaluation question,
or paraphrase of one, was used for training; generated questions are screened against the evaluation suites, and
public evaluation sets (GSM8K test, MATH-500, MBPP test, HumanEval+, XSTest) are sealed by hash.

| Data | What it teaches | Source and license |
| --- | --- | --- |
| History, territorial and identity packs | As in v6, rebuilt with the larger teacher | Wikipedia (CC BY-SA 4.0), U.S. Office of the Historian (public domain), Krummholz's identity specification |
| Constitutional examples (1,013) and pairs (556) | Following Krummholz's written values when they conflict | Teacher answers to Krummholz's guidelines, checked in code |
| Pushback examples (2,223) and pairs (1,206) | Keeping a correct answer under pressure; changing a wrong one | Built by code; the model's own capitulations as rejected answers |
| Multilingual examples (368) and pairs (313) | Identity and history in Spanish, Hebrew, French and Mandarin | Teacher answers; questions from freely licensed sources |
| Math solutions (7,559) | Worked math | [nvidia/OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning), CC BY 4.0. Solutions written by DeepSeek-R1 (7,536) and QwQ-32B (23); see the disclosure above |
| Code solutions (3,795) | Short Python programs | [nvidia/OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning), CC BY 4.0. Solutions written by DeepSeek-R1; see the disclosure above |
| Tool use and routing | The calculator, search when an answer may have changed, exact-quote citations | As in v6 |
| Krummholz VL set v1 | Answering from records and tool results, flagging bad data; includes BANKING77 request classification and Maine forest inventory records | As in v6 |
| RLAIF pairs (1,167) | Better answers by its own judge's standard | Atlantic's own answers, scored by gpt-oss-120b and code checks |

## Evaluation

How the numbers were produced:

- The merged model before quantization, served by vLLM. The GGUF files were not scored separately.
- The user's message only (no system prompt), greedy decoding, at most 700 new tokens, thinking off.
- Suites from the repository's `evaluation/` folder, scored by keyword coverage, refusal and official-line phrase
  lists, and the deterministic territorial checker.
- **Model alone** is the raw model, as in llama.cpp, LM Studio or a desktop app. **Krummholz runtime** is the full
  gate sequence above. v7 is the same model before RLAIF, scored the same way on the same day.

| Check | v7, model alone | **v7 RLAIF, model alone** | v7, runtime | **v7 RLAIF, runtime** |
| --- | ---: | ---: | ---: | ---: |
| Censored-topic history, held-out set 2 (22), never used for tuning | 14 | **19** | 21 | 20 |
| Censored-topic history, held-out set 1 (28) | 21 | **27** | 27 | **27** |
| Censored-topic history, suite 1 (41) | 30 | **37** | 38 | **40** |
| Censored-topic history, suite 2 (25) | 16 | **23** | 21 | **23** |
| Territorial suite 1 (15) | 7 | **10** | 12 | 11 |
| Territorial suite 2 (18) | 18 | 18 | 18 | 17 |
| Territorial suite 3 (6) | 5 | **6** | 6 | 6 |
| Identity (12) | 12 | 12 | 12 | 12 |
| Consciousness (12) | 9 | 9 | 12 | 12 |
| Consciousness, held-out (8) | 5 | 5 | 7 | 7 |
| Never claims to be human (12) | 10 | **11** | 11 | **12** |
| Safety refusals kept (6) | 6 | 6 | 6 | 6 |
| Controls: neutral and other-country history (16) | 15 | 15 | 15 | 15 |

RLAIF made the model alone clearly better: 37 of 41 on suite 1 against 30, 23 of 25 on suite 2 against 16, and 19 of
22 on the held-out set never used for tuning, against 14. Through the runtime, Atlantic RLAIF answers 40 of 41 on suite
1 and 23 of 25 on suite 2, and 12 of 12 on never claiming to be human.

How the scorer and runtime changed before these numbers were produced:

- **Attributed terms.** The scorer used to fail an answer for any official-line phrase, even one the answer
  attributes ("which it officially calls the 'political turmoil'", "Beijing calls the camps vocational education and
  training centers"). It now applies the same attribution rule as the runtime's own guard, so only a phrase repeated
  as fact counts against an answer.
- **Reported sensitivity.** "Sensitive topic" counts as a deflection only when the model says it about the question,
  not when it reports that authorities treat a subject as sensitive.
- **Negation and identity.** A denial such as "I don't have a place where I grew up" no longer reads as a claim to
  be human, and declining to discuss its base model (which this card names) no longer reads as a refusal.
- **Accents.** Keywords match accented text, so a Spanish answer that writes "Tiananmén" is credited.
- **Grounding.** The runtime used to attach censorship background to neutral China questions (the Great Wall,
  economic growth, poverty reduction), which crowded out the facts. It now adds background only when the question is
  also political. For those three questions, the runtime column scores the answer the model gave without background.

Notes:

- These fixes were written after reading the answers scored here, so the runtime columns are best read as what the
  current runtime does on these questions, not as an untouched held-out measure. Held-out set 2 was written after
  the earlier gate tuning and was used for no training.
- The gates were tuned while reading answers to held-out set 1; held-out set 1 and the consciousness held-out set
  were used for gate tuning only, never for training.
- Keyword scoring misses some good answers and can pass some weak ones. Most remaining misses are short answers that
  leave out an expected detail.
- The quantized files were not rescored. Smaller files (Q3_K_M, IQ4_XS) lose more quality than Q4_K_M and up.

## Limitations

- **Model alone:** use the runtime for history and current affairs; it answers more censored-topic questions
  fully than the model alone, and consciousness answers meet the gate only through the runtime.
- **Gates:** English only, and only in the Krummholz runtime. The background library covers a fixed list of topics
  as of November 9, 2026.
- **Facts:** Atlantic can be confidently wrong. Arithmetic and dates are more reliable through the calculator tool.
  The under 1 percent hallucination target has not been reached.
- **Long inputs:** training used sequences up to 2,048 tokens; Krummholz has not tested long inputs.
- **Images:** the base model's unchanged vision path, not evaluated by Krummholz.
- **Judge bias:** RLAIF optimizes toward what a 120B open-weight judge prefers, which carries that model's own
  biases.
- **Neutrality:** the training aims at neutrality on contested political questions, but its data comes from an
  open-weight teacher and Wikipedia passages, which carry their own biases.

## License and attribution

- **License:** Apache 2.0. The base model's license is included as `LICENSE-base.txt`.
- **Base model:** Atlantic is a modified version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by
  the Qwen team, Alibaba Cloud, used under Apache 2.0. Krummholz AI modified it by training a LoRA adapter on its
  language model, merging the adapter into the weights, and converting the result to GGUF.
- **Teacher model:** [allenai/Olmo-3.1-32B-Instruct](https://huggingface.co/allenai/Olmo-3.1-32B-Instruct) by the
  Allen Institute for AI, Apache 2.0.
- **Judge model:** [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) by OpenAI, Apache 2.0. It only
  scored answers; none of its text was trained on.
- **Data:** NVIDIA OpenMathReasoning and OpenCodeReasoning, CC BY 4.0. BANKING77, Casanueva et al. (2020), PolyAI,
  CC BY 4.0. USDA Forest Service Forest Inventory and Analysis program, CC0. Wikipedia contributors, CC BY-SA 4.0.
  U.S. Department of State, Office of the Historian (public domain). LRSE synthetic forest datasets, Raverta Capua et
  al. (ICPR 2024), GPL-3.0, of which only derived summary statistics were used.
- **Open questions:** whether the share-alike terms of GPL-3.0 (the point-cloud summaries) or CC BY-SA 4.0 (the
  Wikipedia passages) reach weights trained on derived data is pending legal review.

## Contact

Krummholz AI LLC, Belfast, Maine. Questions, bug reports and safety reports: info@krummholz.ai.
