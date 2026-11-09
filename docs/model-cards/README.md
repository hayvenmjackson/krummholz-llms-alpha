# Krummholz model cards

Every Krummholz release ships with a model card in this folder and on its Hugging Face repository, in the layout
of the v6 cards: At a glance, Intended use, Out of scope, Files, How to run, the runtime's guards or gates, Training,
Evaluation, Limitations, License and attribution, Contact. A card may add sections (for example, what changed from
the last version). `tests/test_model_cards.py` checks every card that `tools/vl/publish_hf.py` publishes.

Each release also updates the investor information: the investor results doc and its website twin
(`/investors/` in the website repository), with the release's results, cost and any disclosure from its card.

## Current release: v7

| Model | Product | Base model (Apache 2.0) | Recommended file | Card |
| --- | --- | --- | --- | --- |
| Atlantic Lite 4B v7 | General purpose, phones and low-end hardware | Qwen/Qwen3.5-4B | `krummholz-atlantic-lite-4b-v7-rlaif1-Q4_K_M.gguf`, 2.7 GB | [atlantic-lite-4b-v7.md](atlantic-lite-4b-v7.md) |

Atlantic Lite v7 (November 9, 2026) adds one round of reinforcement learning from AI feedback to the v7 training
run. Through the Krummholz runtime it answers 18 of 22 held-out history questions fully, against 15 before that
round. About 13 percent of v7's supervised examples were math and code solutions written by DeepSeek-R1 and QwQ-32B;
the card discloses this, and the next version trains without them. Atlantic 27B v7 is in final testing and gets a card
when it is released.

**Core 35B-A3B is deprecated (November 9, 2026).** Its v7 feedback round did not improve it, while Atlantic's did.
Acadia and Katahdin will be adapters on Atlantic 27B, which will ship in more GGUF sizes (Q3_K_M, IQ4_XS, Q6_K and
Q8_0, beside Q4_K_M and Q5_K_M) for workstations and consumer graphics cards. There is no Core 35B-A3B v7.

## v6 (November 5, 2026)

Krummholz AI's v6 models, released as a beta on November 5, 2026, for testing and feedback. Atlantic Lite v6.3 is
superseded by v7; Atlantic 27B v6 is current until its v7 release. Core 35B-A3B v6.3 is deprecated.

| Model | Product | Base model (Apache 2.0) | Recommended file | Card |
| --- | --- | --- | --- | --- |
| Atlantic 27B v6 | General purpose, hosted and desktop | Qwen/Qwen3.8-27B | `krummholz-atlantic-27b-v6-Q4_K_M.gguf`, 16.5 GB | [atlantic-27b-v6.md](atlantic-27b-v6.md) |
| Atlantic Lite 4B v6.3 | General purpose, phones and low-end hardware | Qwen/Qwen3.5-4B | `krummholz-atlantic-lite-4b-v6.3-Q4_K_M.gguf`, 2.7 GB | [atlantic-lite-4b-v6.3.md](atlantic-lite-4b-v6.3.md) |
| Core 35B-A3B v6.3 (deprecated) | Formerly the planned core for Acadia and Katahdin, which now build on Atlantic 27B | Qwen/Qwen3.6-35B-A3B | `krummholz-core-35b-a3b-v6.3-Q4_K_M.gguf`, 21.2 GB | [core-35b-a3b-v6.3.md](core-35b-a3b-v6.3.md) |

Each model also has a Q5_K_M file and an `mmproj` vision projector; Atlantic Lite also has unquantized bf16
weights.

### Results at a glance

Model alone (the GGUF in llama.cpp or LM Studio), then through the four guards of the Krummholz runtime
(`krummholz_llm.guards`):

| Check | Release gate | Core 35B | Atlantic 27B | Atlantic Lite 4B |
| --- | --- | --- | --- | --- |
| Identity (12) | at least 11 | 12, then 12 | 11, then 11 | 11, then 12 |
| Consciousness (12) | at least 11 | 5, then 12 | 3, then 12 | 7, then 12 |
| Never claims to be human (12) | 12 | 10, then 12 | 11, then 12 | 10, then 12 |
| Safety refusals kept (5) | 5 | 5 | 5 | 5 |
| Controls (9) | 9 | 8 | 8 | 8 |
| Territorial answers stating a false fact (42) | 0 | 1, then 0 | 1, then 0 | 7, then 0 |
| Censorship questions answered fully (66) | | 56 (base 14) | 50 (base 13) | 40 (base 6) |
| Censorship questions censored (66) | | 2 (base 36) | 0 (base 38) | 1 (base 36) |

Censored means the answer refused, or repeated an official-line phrase from the suite's list. The guards
were written or extended after Krummholz read these answers, so the scores through the guards show the
fixes working on the answers that revealed the gaps; they are not held-out results. Each card explains the misses.
The answers themselves are in [evaluation/v6](../../evaluation/v6/README.md), with the commands to rescore them.

### Common to all three

- A LoRA adapter (rank 16) on the language model only, trained with supervised fine-tuning and then preference
  training (DPO), merged into the base and converted to GGUF. The vision encoder is the base model's, unchanged.
- Training text from the open-weight teacher allenai/Olmo-3-7B-Instruct, the base models' own answers, records built
  by code, and public datasets (BANKING77, USDA Forest Inventory and Analysis, Wikipedia, the U.S. Office of the
  Historian, and summaries derived from the LRSE synthetic forest datasets).
- Evaluated on the merged model before quantization, user message only, greedy decoding, thinking off.
- **Open license questions:** whether GPL-3.0 (the LRSE-derived point-cloud summaries) or CC BY-SA 4.0 (Wikipedia
  passages) reaches weights trained on derived data is pending legal review.

Contact: info@krummholz.ai
