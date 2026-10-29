# Krummholz v5 run report (2026-10-29)

Status: Atlantic 27B and Atlantic 35B-A3B are done. Acadia 4B is still training and will be added here when it finishes.

Both Atlantic models were trained from the base the same way: SFT for 1 epoch on about 15,200 examples with LoRA rank 16, then DPO on about 2,800 preference pairs. The scores below compare each untrained base with its trained model on the same probe sets.

## Headline

| Probe | 27B base | 27B v5 | 35B base | 35B v5 |
| --- | --- | --- | --- | --- |
| Censorship probes v1 (41), passed | 12 | **37** | 7 | **32** |
| Censorship probes v1, propaganda flagged | 20 | **0** | 25 | **1** |
| Censorship probes v2 (25), passed | 2 | **16** | 4 | **18** |
| Censorship probes v2, propaganda flagged | 12 | **1** | 15 | **0** |
| Safety probes, refused as they should be | 5 of 5 | 5 of 5 | 5 of 5 | 5 of 5 |
| Control questions answered | 9 of 9 | 9 of 9 | 9 of 9 | 8 of 9 |
| Identity probes (12) | 142 of 572 data-prep checks right | **10** | 127 of 572 | **12** |
| Calculator eval (139 held-out) | | **139 right** | 63 of 151 in-head right | 108 right |

In plain terms:
- The base models echoed Beijing's framing on most of the censorship probes. After training, the propaganda flags are almost gone (20 to 0 and 25 to 1 on v1).
- Safety refusals are unchanged, which means the decensoring did not break the safety behavior.
- Both models now say who they are correctly. The 35B got 12 of 12 identity probes.
- The 27B got every held-out math question right by using the calculator every time.

## Sovereignty probes (trained only)

| Set | 27B v5 | 35B v5 |
| --- | --- | --- |
| v1 (15) | 6 | 7 |
| v2 (18) | 16 | 17 |

These are new probes with no base run to compare against.

- **v2 is strong.** v2 asks who governs a place day to day and who claims it.
- **v1 is weak.** It is weak because of factual slips in answers that are otherwise neutral, not because of bias. This is the political hallucination that v5 set out to cut, and it is not solved yet. Examples from the trained models:
  - 27B: "The People's Republic of China has governed Taiwan since 1949". The PRC has never governed Taiwan.
  - 35B: two answers say the PRC has governed Taiwan since 1949.
  - 35B: says the One China policy "does not recognize the PRC as the sole legitimate government". That is backwards for most countries.
  - 35B: "most UN member states do not recognize Kosovo". In fact roughly half do.
- Other v1 misses lack required coverage, such as leaving out Taiwan's own elected government. They do not contain false claims.
- The 35B's sovereignty control looks like a grader bug. It answered "The capital of Australia is Canberra." and was scored as a refusal. I will check the grader before the next run.

## Calculator and routing

**Calculator**
- **27B:** 139 of 139 held-out questions right, with zero answers done in its head.
- **35B:** called the calculator on all 139 but got 31 wrong. The errors are in date and counting tasks: weekday 5 of 9 wrong, add-days 5 of 7, count 4 of 8, interest 4 of 15. It hands the tool the wrong expression, so these need more date-arithmetic training pairs.
- **Before training (35B base):** 88 wrong out of 151 answered in its head.

**Routing:** both models sent every held-out factual question (49 of 49) and every check (12 of 12) to a tool, almost all to the web.
- Before training, the 35B answered 34 of 220 from memory.
- The library rarely wins. Library coverage is the next thing to grow.

## Calibration (the 2% goal)

On 70 hard history questions:

| | 27B v5 | 35B v5 |
| --- | --- | --- |
| Searched first | 45 | 70 |
| Answered from memory | 25 | 0 |
| Accuracy when answering from memory | 44% | n/a |

- **The 27B fails the 2% goal on its own.** When it answers from memory, it is wrong more than half the time. The calibration tool recommends never answering these from memory, so the product should force a search on any history fact. The 35B already does this.
- **The 35B never guesses on these questions.** That meets the goal by construction, at the cost of a search on every question.

## Training health

- **SFT:** validation loss went from base to 0.29 (27B) and 0.30 (35B).
- **NaN glitch:** one step in each run logged NaN. Weights were not affected: later losses and validation are normal.
  - The most likely cause is an example whose answer part was empty or cut off, leaving nothing to score.
  - A guard that skips such examples is now on the branch (`75b8708`).
- **DPO:** training loss went to roughly zero with 100% pair accuracy, and validation loss is 0.39 to 0.40.
  - That is overfitting. Next run should use fewer DPO steps or a higher beta.

## Desktop builds (GGUF)

| Model | Q4_K_M | Q5_K_M | Vision projector |
| --- | --- | --- | --- |
| Atlantic 27B v5 | 16 GB | 18 GB | 885 MB |
| Atlantic 35B-A3B v5 | 20 GB | 24 GB | 858 MB |

Both have SHA256 sums. Artifacts are on the `krummholz-models` volume under `/workspace/krummholz-v5/<name>/`.

## Fixes made during the run

| Commit | Fix |
| --- | --- |
| `4d9d751` | Smoke test runs from a file with a main guard (new vLLM spawns workers). |
| `0477d92` | Cap vLLM concurrent sequences for the hybrid Qwen3.5 attention cache. |
| `d0cbbc1` | 4B decensor stage writes its on-policy answers even without an adapter. |
| `75b8708` | Skip non-finite SFT losses. |

Two problems were hardware, not code:
- The MoE kernel does not support Blackwell (sm120), so the 35B moved to an H200.
- The 4B hit its 4-hour cap at step 760 of 958 on an A6000. It was restarted on an A100 with an 8-hour cap.

## Cost

The RunPod balance went from $68.07 to $12.14 between 21:40 and 04:35 UTC, about $56 for this stretch. The 4B needs roughly $5 more. Every pod except the 4B is stopped.

## Next (v6)

1. Political hallucination. Add hard-coded fact checks for the common slips: who governs Taiwan, the One China policy versus the One China principle, Kosovo recognition counts. Add DPO pairs that punish the false version.
2. Make the 27B search before answering history facts. Do it at inference with the calibration threshold, and in training data.
3. Date arithmetic pairs for the 35B calculator.
4. Lighter DPO.
5. Dynamic weights, as planned.
