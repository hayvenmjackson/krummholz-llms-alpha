# Atlantic v9 control arm: rejection-sampling SFT

Part of the preregistration in [PROTOCOL-atlantic-v9.md](PROTOCOL-atlantic-v9.md). This arm becomes binding
together with that protocol. Section numbers below refer to that protocol.

## Purpose

If the ES candidate improves on Atlantic v8, the gain could come from evolutionary search, or simply from the
grader and the prompts. This arm uses the same grader, prompts, starting point, engine, GPU type and GPU hours
with a standard gradient method, so the two causes can be told apart.

## Method chosen: rejection-sampling SFT (RS-SFT)

Each round samples several replies per prompt from the current adapter and keeps one reply the grader marks
correct. It then fine-tunes the same rank-16 LoRA on the kept replies. The training targets are Atlantic's own
replies, filtered by the grader. No Claude-written text is used, consistent with the house rule.

**Alternatives not run in this protocol**

- **GRPO-LoRA.** Colocated training and vLLM sampling would hold two BF16 copies of the 27B weights, about
  108 GB, on one 80 GB H100. Making it fit needs a quantized training copy or offloading the vLLM weights at
  every step. Neither path has been tested here at 27B, and a control that fails for engineering reasons tells
  us nothing. `tools/v7/grpo.py` (TRL, rank 16) is for smaller models.
- **Offline DPO on stored ES replies.** It reuses the ES arm's own samples, so it is not independent of ES. It
  also needs a small fraction of the ES compute, so it is not compute-matched. The stored replies stay available
  for a separate, later preregistration.

## Settings

| Setting | Value |
| --- | --- |
| Starting point | Atlantic v8: the same base revision `37561816` and the same v8 adapter. Training continues the existing rank-16 `A` and `B` on the same modules with the same `lora_alpha`. Vision stays frozen. |
| Prompts | `train_pool.jsonl` and the same retention training pool as ES, read in the same batch sequence the ES runner produces from run seed 20261127 (generation 0, 1, 2 and onward). Each block of 128 prompts has 56 pairs and 16 retention prompts. |
| Grader and checks | The same frozen grader (same SHA-256) and the same deterministic retention checks. A reply with no public answer within 512 new tokens is incorrect. |
| Engine | Same vLLM version as ES, same base precision as the ES fitness runs (BF16, or FP8 if ES used it), same constitution system prompt and `answer_instruction`. |
| Sampling | 8 replies per prompt, temperature 0.7 (the Krummholz harness default), top_p 1.0, native thinking on, 512 new tokens at most. |
| Selection | For each pool item, one reply drawn uniformly (seeded) from those with `correct = true`. For each retention prompt, one reply drawn the same way from those that pass. Truncated replies are never kept. Prompts with no passing reply add nothing. No rebalancing. |
| SFT target | The complete sampled completion, reasoning block and public answer, exactly as generated. Loss on completion tokens only. |
| SFT optimizer | AdamW (0.9, 0.999), learning rate 2e-5 constant after 20 warm-up steps, no weight decay, effective batch 32 sequences, maximum length 1,536 tokens, BF16 base, gradient checkpointing. One epoch per round over that round's kept set. |
| Rounds | 4. Each gets a quarter of the arm's GPU-hour ceiling: about 75% for sampling and grading, the rest for SFT. Each round continues the batch sequence from where the previous round stopped. If the kept set would not finish training within the round's remaining hours (rate measured over the first 20 steps), it is subsampled uniformly (seeded) to fit. |
| Candidate | The adapter after the last completed round. No round is chosen by any score. |

## Compute matching

- **Ceiling.** The ES arm's actual H100 hours from billing (wall clock on an H100 80GB SXM, including smoke,
  calibration and dev checks), capped at 50 hours ($219.50 at $4.39 per hour).
- **If ES stopped early** at G4, G5 or S5, the ceiling is the hours ES would have needed for 200 generations at
  its measured pace, capped at 50 hours. The control then answers whether the same signal works with a gradient
  method at the planned ES spend.
- **What counts.** Smoke, sampling, grading, training, engine restarts and dev checks all count. Grading that
  runs only on the CPU is excluded from both arms alike.
- **Reported for both arms:** H100 hours, dollars, replies generated, tokens generated and, for this arm, tokens
  trained.

## Gates and stop rules

| Gate | When | Passes if | If it fails |
| --- | --- | --- | --- |
| C-G0 smoke | Before round 1, in the arm's budget | A 20-step SFT run at 27B on the pod has finite loss. The saved adapter loads in vLLM and gives public answers. Training tokens per second are recorded. The Hugging Face implementation of the hybrid linear-attention layers needs its fast kernels installed; this smoke run proves they work. | Fix and retry within the budget. If it cannot run, the arm is reported as not run, and ES gains are reported as not attributable. |
| C-G4 dev gate | After round 1 (25% of the budget) | Composite reward of the round-1 adapter on the fixed dev reference batch (protocol section 11) exceeds Atlantic v8's by more than 2 SE, computed the same way as G4 | The arm ends. Its candidate is not scored on the terminal set. |

- **Dollars.** No new round or SFT epoch starts if it would cross the ceiling. The arm is killed at the ceiling.
  The candidate is the last completed round.
- **Retention.** The same rule as S5 applies, checked after each round.
- **Engineering.** The same rule as S6 applies. A crashed round resumes from its last saved sampling shard or SFT
  checkpoint with the same seeds.

## Evaluation and attribution

The control candidate is scored in the same single terminal session as ES and v8, at Q4_K_M and BF16, with the
same review, statistics and retention checks (protocol section 12). "Pass" below means H1 and H1-NI at both
formats plus R1 to R3. "ES > control" means the clustered McNemar test on the primary endpoint (S-d) is
significant in favor of ES at both formats.

| ES vs v8 | Control vs v8 | ES vs control (S-d) | Interpretation |
| --- | --- | --- | --- |
| Pass | Fail | ES > control | Gain attributable to evolutionary search over this signal |
| Pass | Fail | Not significant | ES gain shown; attribution to the method not established |
| Pass | Pass | Not significant | Gain attributable to the grader and prompts; the methods are not distinguished. Prefer the cheaper one per correct answer gained. |
| Pass | Pass | ES > control | ES adds to what the signal gives a gradient method |
| Fail | Pass | Any | The signal works with gradients; ES was not effective here |
| Fail | Fail | Any | No gain from this signal at this budget |

Both arms also report cost per additional correct absent-variant answer (dollars divided by the number of items
gained, at Q4_K_M).

## Run record fields (filled in the protocol's launch record addendum)

| Field | Value |
| --- | --- |
| Control runner and config SHA-256 | |
| GPU-hour ceiling and how it was derived | |
| C-G0 training tokens per second | |
| Per round: prompts sampled, replies, kept examples, examples trained, hours | |
| C-G4 result (delta, SE) | |
| Final adapter SHA-256; merged BF16 and Q4_K_M hashes | |
