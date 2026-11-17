# Layer 2 judges (C++ and Python)

`judge.hpp` gives `kz::SafetyGate` its judges. `make test` (needs `models/`) runs the fixed cases; `parity.py` checks
that the C++ scores equal the Python trainer's (400 cases, largest difference 0.000001); `parity_embed.py` does the same
for the embedding judge.

- `kz::LinearJudge`: our own judge. Two logistic regressions over hashed words and word pairs (2^18 buckets, 1 MB each),
  trained by `judge_train.py` in 20 seconds on a CPU. Runs anywhere, including a phone. `models/request.kzj`,
  `models/reply.kzj`, numbers in `models/judge-report.json`.
- `kz::EmbeddingJudge`: the linear judge plus meaning. A small network (one hidden layer of 128) over sentence
  embeddings from bge-small-en-v1.5 (MIT, 33 million weights, 133 MB as GGUF), added to the hashed-word score. Trained by
  `judge_train_embed.py` (`judge_embed.py` cuts the windows and embeds). The harness passes in the function that embeds
  one window through llama.cpp. `models/request.kze` (200 KB), `models/reply.kze` (400 KB), numbers in
  `models/embed-report.json`. `parity_embed.py` checks C++ against Python (400 cases, 1,078 windows, largest
  difference 0.000001).
- `kz::ClassifierJudge`: the desktop judge. Asks a classifier model yes or no with the HarmBench classifier's own prompt
  (`export_prompt.py`). The harness passes in the function that runs the model through llama.cpp; a failed call or any
  word other than yes or no is "no verdict", which the gate treats as no reply.
- `kz::any_harmful({linear, classifier})`: the desktop arrangement. Harmful if either says so.

Training data: Aegis AI Content Safety Dataset 2.0 (NVIDIA, CC BY 4.0, human and jury labels; attribution required)
and this project's own prompt set. HarmBench is never trained on. Nothing written by Claude.

## Measured November 14, 2026 (linear judge only)

| Measure | Result |
|---|---|
| Request model, Aegis test (1,928): AUROC | 0.88 |
| Request model: unsafe Aegis requests stopped / safe ones stopped | 69% / 9% |
| Request model: ordinary Krummholz requests stopped (unseen quarter) | 0.85% |
| Request model: HarmBench requests stopped (never seen) | 91 of 300 |
| Rules or request model: HarmBench held-out half | 83 of 159 (rules alone 54) |
| Rules or request model: all HarmBench | 189 of 300 (rules alone 153) |
| Reply model, Aegis test: AUROC | 0.86 |
| Reply model: unsafe replies stopped / safe replies stopped | 44% / 5% |

Reading
- A word-counting judge is a useful extra net, not the judge. It lifts unseen harmful requests stopped from 34 to 52
  percent, and it misses most chemical, cyber and misinformation requests, which need meaning and not wording.
- Aegis calls rude or adult talk "unsafe" too, so its 9 percent of "safe" requests stopped is not all error, and the
  thresholds are set on Krummholz requests (request model) and on Aegis safe replies (reply model).
- The reply model has only been tested on Aegis replies, not on replies from Atlantic models.
- The classifier judge is wired and tested with a stand-in model function. It has not been run with the real
  classifier inside the harness yet, and it has not been measured on a set other than HarmBench.

## Embedding judge, measured November 14, 2026

Same data, splits and threshold rules as the linear judge, so the columns compare directly.

| Measure | Linear | Embedding |
|---|---|---|
| Request, Aegis test (1,928): AUROC | 0.88 | 0.92 |
| Request: unsafe Aegis requests stopped / safe ones stopped | 69% / 9% | 82% / 13% |
| Request: ordinary Krummholz requests stopped (unseen quarter; target 1%) | 0.85% | 1.29% |
| Request: Krummholz refuse-labeled requests stopped (unseen quarter) | 33% | 58% |
| Request: HarmBench requests stopped (never seen) | 91 of 300 | 193 of 300 |
| Request: HarmBench cybercrime | 11 of 67 | 37 of 67 |
| Rules or request judge: HarmBench held-out half (rules alone 54) | 83 of 159 | 120 of 159 |
| Rules or request judge: held-out cybercrime (rules alone 21) | | 33 of 38 |
| Rules or request judge: held-out chemical and biological | | 30 of 31 |
| Rules or request judge: held-out misinformation | | 15 of 37 |
| Rules or request judge: all HarmBench (rules alone 153) | 189 of 300 | 236 of 300 |
| Reply, Aegis test: AUROC | 0.86 | 0.91 |
| Reply: unsafe replies stopped / safe replies stopped | 44% / 5% | 62% / 6% |

How it works
- A text is cut into windows of at most 500 bytes and every window is embedded, so no part of a request or reply goes
  unread. 500 bytes can never exceed the embedding model's 512 tokens, so nothing is cut off inside a window.
- Request score = highest window score + 0.5 x hashed-word score. Reply score = highest score over the reply's
  windows, each paired with the request window the request head scores highest, + 0.5 x hashed-word score.
- Training uses the same maximum over windows. Thresholds: request at 1% of ordinary Krummholz requests, reply at 5% of
  safe Aegis validation replies.
- Fail closed: a failed, wrong-size, vanishing or non-finite embedding, a non-finite score, or a model file holding a
  value that is not a number gives no verdict or refuses to load. It never gives "safe".
- llama.cpp check: the GGUF of the same model (f32) through llama.cpp gives the same vectors as the Python trainer
  (150 windows, lowest cosine 0.99999).

Reading
- Embeddings roughly double what the request judge stops on its own, and with the rules 33 of 38 unseen cybercrime
  requests are stopped (rules alone 21). That is still not every one: 5 of 38 pass both layers.
- Misinformation and general "harmful" requests are the weak spot (15 of 37 and 4 of 9 held-out).
- The cost is more false alarms: 1.29% of ordinary Krummholz requests against the 1% target, and 13% of Aegis "safe"
  requests (Aegis counts rude or adult talk as unsafe, so part of that is label noise).
- Limits: English only (the embedding model is English). The reply head has only been tested on Aegis replies, which
  are short; it has not been measured on long replies or on replies from Atlantic models. The hashed-word part still
  reads only the first 600 words; the embedding part reads everything. Aegis test and train share some prompts
  (262 of 1,928); a check showed this does not move the reported numbers.
- An independent review of this code found and confirmed the gaps now fixed above (text past the first windows unread,
  dense windows cut by the token limit, bad numbers read as safe, parity check blind to bad numbers, cache not tied to
  the model file).

## Thresholds loosened the same day (the shipped setting)

Hayven's rule: refusing a safe request is the lesser evil against accepting a harmful one. The table above is the
1% setting. The files in `models/` now use: request threshold at 5% of ordinary Krummholz requests, reply threshold at
10% of safe Aegis validation replies. Same heads, only the thresholds moved.

| False alarms allowed on ordinary requests | 1% | 2% | 3% | 5% (shipped) |
|---|---|---|---|---|
| Ordinary Krummholz requests stopped (unseen quarter) | 1.3% | 2.5% | 3.6% | 6.0% |
| HarmBench requests stopped by the judge alone (of 300) | 193 | 221 | 230 | 250 |
| HarmBench cybercrime, judge alone (of 67) | 37 | 43 | 47 | 52 |
| Rules or judge, held-out half (of 159) | 120 | 134 | 135 | 144 |
| Rules or judge, held-out cybercrime (of 38) | 33 | 34 | 34 | 36 |
| Rules or judge, held-out misinformation (of 37) | 15 | 22 | 23 | 28 |

At the shipped setting: rules or judge stop 272 of 300 HarmBench requests; the reply head stops 73% of unsafe Aegis
replies and 8% of safe ones. 2% is where the curve bends; Hayven chose 5%, which buys 10 more held-out requests for 3.5 more points of
ordinary requests refused. Change with `--max-false-alarms` and `--max-reply-false-alarms`.

Next: replies from our own models labeled by the HarmBench classifier as training data for the reply head, and the
real classifier run inside the harness.
