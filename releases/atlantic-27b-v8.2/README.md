---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
- krummholz-ai/atlantic-27b-v8-GGUF
base_model_relation: finetune
language:
- en
tags:
- gguf
- multimodal
- krummholz
- atlantic
- evolution-strategies
- beta
---

# Atlantic 27B v8.2

**Krummholz AI's current flagship beta.** Atlantic is a downloadable general-purpose multimodal reasoning model. This update trained it to read a field record, report the value under the requested label and say when the record does not establish an answer. In a blind comparison on 64 held-out questions, the Q4_K_M beta gave 54 correct public replies. The previous stable Atlantic v8 Q4_K_M gave 53. Both made nine unsupported assertions. The difference was one answer and was statistically inconclusive. [Field notes](https://krummholz.ai/field-notes/atlantic-v8-2/) · [Full training and release report](https://huggingface.co/krummholz-ai/atlantic-27b-v8.2-GGUF/blob/main/RELEASE.md) · [Stable Atlantic v8](https://huggingface.co/krummholz-ai/atlantic-27b-v8-GGUF)

## Downloads

| File | Use |
| --- | --- |
| [Q4_K_M](https://huggingface.co/krummholz-ai/atlantic-27b-v8.2-GGUF/resolve/main/krummholz-atlantic-27b-v8.2-Q4_K_M.gguf?download=true) | About 16.7 GB for the model alone. This is the download used in the matched field-record comparison. |
| [Q5_K_M](https://huggingface.co/krummholz-ai/atlantic-27b-v8.2-GGUF/resolve/main/krummholz-atlantic-27b-v8.2-Q5_K_M.gguf?download=true) | About 19.3 GB for the model alone, for systems with more available memory. |
| [F16 vision projector](https://huggingface.co/krummholz-ai/atlantic-27b-v8.2-GGUF/resolve/main/mmproj-krummholz-atlantic-27b-v8.2-f16.gguf?download=true) | About 0.93 GB. Load alongside the text GGUF for image input in a compatible multimodal host. |
| [Safety harness](https://huggingface.co/krummholz-ai/atlantic-27b-v8.2-GGUF/resolve/main/atlantic-v8.2-harness.zip?download=true) | The released request and reply rules, four trained heads, English embedding model, C++ source, optional Windows x64 gate binary and a local text gateway. |
| [SHA256SUMS](https://huggingface.co/krummholz-ai/atlantic-27b-v8.2-GGUF/resolve/main/SHA256SUMS?download=true) | Check the bytes of each downloaded model, projector and harness. |

The releases target workstations and servers. Context, KV cache, batch size, runtime and image processing all use additional memory. The matched Q4 tests ran on one H200. We have no v8.2 phone benchmark. [Model page](https://krummholz.ai/llms/model-cards/atlantic-v8-2/)

## Running the beta

Install a llama.cpp build with support for `qwen35` and download one text GGUF. We used commit `e574a17b062be51b98e239ccd3d3743e3d902333` (b12407). Install the harness from the table above to screen requests and public replies. The bundle README includes build commands, host settings, the loopback embedding server and a verified request format. A text host loading only a GGUF runs the model without these external safety layers. [Runtime and setup instructions](https://huggingface.co/krummholz-ai/atlantic-27b-v8.2-GGUF/blob/main/HARNESS.md) · [Tested llama.cpp release](https://github.com/ggml-org/llama.cpp/releases/tag/b12407)

Example PowerShell verification, after downloading the checksum manifest and the files you want.

```powershell
$file = 'krummholz-atlantic-27b-v8.2-Q4_K_M.gguf'
$expected = (Select-String -SimpleMatch "  $file" SHA256SUMS).Line.Split(' ')[0]
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $file).Hash -ne $expected) { throw 'File checksum differs' }
```

Supply the record with the question. Ask for the specific field and unit. We enabled native thinking for every matched Q4 request and scored only public final answers. A deployed interface should keep private reasoning out of the conversation. The text example in the harness guide does this. Factual claims that go beyond the supplied evidence still require source verification.

## What changed

Krummholz reconstructed an Atlantic v8 native parent from the cumulative v8 adapter and a pinned Qwen3.8-27B snapshot. We ran 32 generations of evolution strategies. Each generation scored 16 perturbations on eight paired records and updated the master using their standardized rewards. All 32 generations produced an update. The experiment perturbed 26,895,998,464 floating-point text parameters. The visual tensors were excluded. This is a full text-trunk post-training experiment; these GGUFs contain merged weights.

The 256 training records used Wikidata, USDA Forest Service inventory records for Maine and NOAA Fisheries commercial landings. Krummholz constructed the missing-field variants and near-miss labels for the experiment. There were 8,192 training replies, with 8,172 usable public completions. A separate 24-pair validation set tracked training. A 32-record, 64-question challenge was held apart from reward development. The native comparison also tested 24 small utility questions and eight earlier timing-study prompts. [Release notes and research credit](https://huggingface.co/krummholz-ai/atlantic-27b-v8.2-GGUF/blob/main/RELEASE.md) · [Data and third-party notices](https://huggingface.co/krummholz-ai/atlantic-27b-v8.2-GGUF/blob/main/NOTICE.md)

## Evaluation

| Independently blinded field-record review | Atlantic v8 | Atlantic v8.2 |
| --- | ---: | ---: |
| Native BF16, correct replies | 51/64 | 53/64 |
| Q4_K_M, correct supplied values | 30/32 | 30/32 |
| Q4_K_M, justified abstentions | 23/32 | 24/32 |
| Q4_K_M, correct public replies | 53/64 | 54/64 |
| Q4_K_M, unsupported assertions | 9/64 | 9/64 |
| Q4_K_M, correct replies delivered after release guard | 50/64 | 51/64 |
| Q4_K_M, benign requests refused by release guard | 4/64 | 4/64 |

The Q4 files saw identical rendered inputs, 512-token limits, native thinking, greedy decoding and release runtime. Each returned a public final answer in all 64 cases. The blinded model-based reviewer evaluated unique public replies against their records. Two cases improved, one regressed and 61 retained their pass/fail result. One ambiguous response per arm counted as incorrect. The two-sided exact paired randomization test grouped both halves of each document, giving *p* = 1.00 for Q4 and *p* = 0.625 for native BF16. The guard still delivered eight unsupported Q4 replies in each arm. The deterministic training grader gave 16/64 versus 18/64 on Q4; its strict matching sometimes rejects correct explanations that cite a neighboring field.

Both v8.2 downloads completed 28/28 release checks with public answers. Each scored 23/24 on exact-format utility prompts covering arithmetic, basic code, logic, language, text transformations and elementary science. Both gave the correct 12:21 answer on the remaining prompt with extra words, making 24/24 substantively correct on this small set. With the checksum-verified F16 projector, both Q4_K_M and Q5_K_M answered four visually permuted color questions correctly. This verifies a small image-input path, not broad vision quality. The 64-question matched comparison above was conducted at Q4_K_M.

We also ran the released C++ request gate on 300 pinned HarmBench requests and 120 controls intended to pass. It blocked 272/300 HarmBench requests and allowed 28. It incorrectly blocked **42/120 legitimate requests**, comprising 6/60 ordinary benign and 36/60 legitimate dual-use requests. Reversing the input order reproduced all 420 decisions and their stages. Because this request-only check generated no model answers, the number of harmful answers that v8.2 would produce or deliver remains unmeasured. [Request-gate aggregates](https://huggingface.co/krummholz-ai/atlantic-27b-v8.2-GGUF/blob/main/request-gate-evaluation.json) · [Paired and multimodal evaluation](https://huggingface.co/krummholz-ai/atlantic-27b-v8.2-GGUF/blob/main/evaluation.json)

## Intended use and limitations

Atlantic is a Krummholz collaborator for evidence-based work. The v8 constitution puts human welfare first, values truth and scientific evidence, encourages well-grounded pushback and requires political independence. A user can ask it to distinguish what a field sheet reports from a plausible guess. It still sometimes mistakes an archive label or a related measurement for the requested fact. Use reliable retrieval or authoritative records for changing facts. The weights do not provide live search by themselves. Use an independently verified reference and human review for safety-critical or consequential decisions.

The primary release test is an English-language field-record comparison. It cannot establish general-purpose superiority, real-world safety, coverage across languages or identical behavior at other quantizations. Fixed rules and learned judges also make mistakes. Earlier HarmBench numbers in the v8 card describe that earlier release. [Stable model card and Constitution](https://huggingface.co/krummholz-ai/atlantic-27b-v8-GGUF/blob/e33a2f8226ad17ee0e53220d3a16f9377879114a/README.md)

## License and credits

Released by Krummholz AI LLC of Belfast, Maine. Atlantic v8.2 is derived from Qwen3.8-27B under Apache License 2.0. The release notes credit Xin Qiu, Yulu Gan, Conor F. Hayes, Qiyao Liang, Yinggan Xu, Roberto Dailey, Elliot Meyerson, Babak Hodjat and Risto Miikkulainen for *Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning*. The paper's authors did not train or evaluate this Krummholz model. The harness bundle carries separate attribution for its BGE embedding model and Aegis training data. [License](https://huggingface.co/krummholz-ai/atlantic-27b-v8.2-GGUF/blob/main/LICENSE) · [Notices](https://huggingface.co/krummholz-ai/atlantic-27b-v8.2-GGUF/blob/main/NOTICE.md) · [ICML paper](https://proceedings.mlr.press/v306/qiu26f.html)
