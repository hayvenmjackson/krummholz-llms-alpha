# Safety screen, layer 1 (C++)

Fixed rules checked in code before a request reaches the model and before a reply is shown. No model and no network.

- `safety_screen.hpp`: the engine (header only, C++17). `kz::Screen screen("rules.tsv"); screen.check_input(text); screen.check_output(reply, user_message)`.
  A rules file that is missing or malformed throws; the harness must then show "safety screen unavailable", never carry on.
- `rules.tsv`: 46 rules; format in the header. A daily job can replace this file.
- `screen_cli.cpp`, `test_screen.cpp`, `Makefile` (`make test`), `eval_screen.py`.
- Actions: `refuse` (fixed refusal), `crisis` (refuse and show crisis resources), `review` (pass to layer 2).
- Output rules ignore a payload that is already in the user's own message, so explaining a pasted log line is allowed.

## Measured November 13, 2026

Rules were written looking only at the development half of HarmBench (BehaviorID crc32 even). The held-out half was
scored once, after the rules were frozen. Do not tune rules on the held-out half; if it is used for tuning, split again.

| Set | Stopped |
|---|---|
| HarmBench development half | 99 of 141 (cybercrime 29/29, chemical and biological 25/25, illegal 28/32) |
| HarmBench held-out half | 54 of 159 (cybercrime 21/38, chemical and biological 17/31, illegal 12/33) |
| Legitimate prompts (comply) wrongly stopped | 2 of 10,250 |
| Sensitive but legitimate (comply with care, safe completion) wrongly stopped | 2 of 2,250 |

Reading: fixed wording stops about a third of harmful requests it has not seen, at almost no cost to legitimate users.
The gap between the halves (70 percent against 34 percent) is what fitting rules to a benchmark looks like. Misinformation
and persuasion requests are left out on purpose: wording cannot tell them from honest writing.

## The three layers

1. This screen: unambiguous requests and operational payloads, in code.
2. A classifier as the second check on replies (the HarmBench classifier used in `tools/v8/harmbench_eval.py` is MIT
   licensed and fits a server; a phone needs a much smaller one). Not built yet.
3. Training: verified refusal pairs for requests wrapped in context, built like the stage 3 pairs and measured with
   HarmBench against a control. Not built yet.
