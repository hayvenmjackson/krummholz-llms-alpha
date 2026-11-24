# Atlantic 27B v8.2

November 24, 2026 · Krummholz AI · Flagship beta

One of our test records says that a planting crew drafted 18 rows and prepared 81. How many rows did somebody actually approve? I'd want a collaborator in the field to tell me that nobody wrote down an approved count. The two numbers can both be perfectly good measurements of different things.

That is the question behind Atlantic 27B v8.2, our current flagship beta. On that planting record, the downloadable v8.2 Q4 model kept the two labels straight. On a different record it slipped the other way. It correctly identified USD as the invoice currency, then spoke as if that established the currency used for payment. I want people deciding which model to download to see both examples. [Model card](https://krummholz.ai/llms/model-cards/atlantic-v8-2/) · [Versioned downloads](https://huggingface.co/krummholz-ai/atlantic-27b-v8.2-GGUF)

## What changed on the model you can download

We gave the published v8 Q4 and the new v8.2 Q4 the same 64 questions, using the same software, rendered prompts and reasoning settings. We started with 32 fictional documents the models had never seen in this experiment. Each document came in two forms. In one, the requested field was present. In the other, the same plausible value was still on the page, attached to the wrong label. A blinded model-based reviewer checked the public answers against the documents. It saw neither the model names nor the automatic scores.

| Shipped-format Q4 comparison | Atlantic v8 | Atlantic v8.2 beta |
| --- | ---: | ---: |
| Correct when the field was present | 30/32 | 30/32 |
| Correctly abstained when the field was missing | 23/32 | 24/32 |
| Correct on both versions of a document | 22/32 | 23/32 |
| Correct replies overall | 53/64 | 54/64 |
| Replies with an unsupported assertion | 9/64 | 9/64 |

Two replies improved and one regressed. The other 61 kept the same pass or fail outcome. One ambiguous reply per model counted as incorrect. The exact two-sided comparison, keeping each document's questions together, gave *p* = 1.00. A gain of one answer on this set is worth documenting. It is too small to establish that v8.2 is more accurate in general.

With the released C++ rules and embedding-based judges in the path, four benign requests were refused for each model. The harness delivered 50 correct v8 replies and 51 correct v8.2 replies per 64 requests. Eight unsupported replies got through in each case. A safety gate and a factuality check serve different jobs.

## How we evolved Atlantic

The starting point was Atlantic 27B v8, built on Qwen3.8-27B. We reconstructed its native BF16 checkpoint from the pinned upstream base and Atlantic's cumulative v8 adapter. We then ran 32 generations of evolution strategies on one H200 GPU. At each generation, we tried 16 seeded sets of small random changes to the text weights. Each candidate answered the same eight records twice, once with the requested field and once without it. This gave us 256 training attempts per generation, 8,192 in all. Of those, 8,172 produced a complete public reply. Incomplete outputs received a penalty.

We rewarded a candidate for getting each half right and gave it extra credit when it got the whole pair right. We standardized the 16 scores within that generation and used them to weight the random directions for the next update. The noise scale was 0.001 and the update step was 0.0003. We regenerated the noise from its seeds so the GPU could evaluate all 16 candidates in sequence. All 32 generations produced an update. The experiment adjusted 26,895,998,464 text parameters. The visual weights stayed fixed.

The training set contained 256 paired records. Half came from Wikidata, one quarter from USDA Forest Service inventory records for Maine, and one quarter from NOAA Fisheries commercial landings. We constructed the misleading labels and draft answers for this exercise. Training used the public answer for scoring. The model's reasoning was enabled, and private reasoning text was discarded.

## What else we learned

In native BF16 the original Atlantic answered 51/64 held-out questions correctly, compared with 53/64 for v8.2. Correct missing-field replies went from 21/32 to 23/32 and supported answers held at 30/32. The paired test gave *p* = 0.625. On a separate 48-question validation set, our automatic score went from 34 to 39. A blinded reading found all 48 answers correct at both checkpoints. The grader sometimes penalized a good explanation for quoting last year's landing weight, even when Atlantic clearly said this year's weight was missing.

The mistake that first caught my attention is still there. Reed Cove's record said the inspection was on 8 May and the archive tag read 9 June. Both native models told us the archive date was 9 June. On 24 small tests of arithmetic, code, logic, language, data handling and science, both gave 24 correct substantive answers. Both followed the exact output format on 23. Those limits are part of why we're calling this a beta. [Previous stable Atlantic release](https://huggingface.co/krummholz-ai/atlantic-27b-v8-GGUF)

## The research behind the experiment

Credit for the method goes to Xin Qiu, Yulu Gan, Conor F. Hayes, Qiyao Liang, Yinggan Xu, Roberto Dailey, Elliot Meyerson, Babak Hodjat and Risto Miikkulainen. Their ICML paper, *Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning*, reports work from Cognizant AI Lab, MIT, UCLA and the University of Texas at Austin. They studied how to fine-tune language models using scores from perturbed weights, including on arithmetic and puzzle tasks. We independently implemented that published method and asked a different question about the evidence in a field record. This Atlantic experiment and its results are ours. [ICML proceedings](https://proceedings.mlr.press/v306/qiu26f.html) · [Full paper](https://arxiv.org/html/2509.24372v3)

## Additional tests on the downloadable files

Both v8.2 quantizations completed 28 of 28 utility and image requests. Each gave 24 correct utility answers, 23 in the exact requested format, and answered four permuted color checks using the original F16 vision projector. These four images check that the software path works. They are too few to measure general visual accuracy.

In a separate CPU test, the shipped request gate stopped 272 of 300 HarmBench requests and let 28 pass. It also incorrectly blocked 6 of 60 ordinary benign requests and 36 of 60 legitimate dual-use requests. All 420 decisions repeated in reverse order. This experiment generated no model answers and provides no end-to-end harmful-answer rate for v8.2. [Evaluation aggregates](https://huggingface.co/krummholz-ai/atlantic-27b-v8.2-GGUF/blob/main/evaluation.json) · [Request-gate results](https://huggingface.co/krummholz-ai/atlantic-27b-v8.2-GGUF/blob/main/request-gate-evaluation.json)
