# Can Atlantic Lite improve itself? Findings and plan

Krummholz AI, November 9, 2026. Branch `rsi-lite-explore`. Status: design and CPU prototype done; no GPU spent.

## The question

Can a 4B model (Atlantic Lite: Qwen3.5-4B, v7 SFT plus DPO, then RLAIF round 1) recursively self-improve? Here that
means the model writes, grades or filters its own training signal, and a measured gain holds or grows over at least 2
to 3 rounds instead of a one-time bump. A careful "no" is also worth having.

## Short answer from the literature

Yes, with an external check, on tasks that have one; mostly no without one.

- **Loops with a verifier work at small scale.** STaR (Zelikman et al., 2022; GPT-J 6B) improved over many rounds
  on arithmetic and GSM8K by keeping only rationales that reached the right answer. ReST-EM (Singh et al., 2023)
  improved MATH and HumanEval over 2 to 3 rounds, then overfit; it retrains from the base model each round to limit
  drift. Iterative RPO (Pang et al., 2024) added DPO on correct over incorrect chains and climbed for 4 rounds with
  shrinking steps. rStar-Math (Guan et al., 2025) is the strongest small-model case: a 3.8B and a 7B model improved
  over 4 self-evolution rounds with code-checked steps and a self-trained process reward model. R-Zero (2025)
  reported gains on Qwen3-4B-Base over about 3 rounds of challenger/solver self-play before saturating.
- **Self-judged loops are weaker and fragile at 4B.** Self-Rewarding LMs (Yuan et al., 2024) worked on a 70B model
  for 3 rounds; Meta-Rewarding (Wu et al., 2024) on an 8B, with responses growing longer each round (length is the
  easiest thing for a self-judge to reward). Huang et al. (2023) found models cannot fix their own reasoning without
  outside feedback. Song et al. (2024) tie self-improvement to the generation-verification gap (how much better the
  model judges than it generates), which grows with scale and is small for small models.
- **Known failure modes.** Training on unfiltered own outputs loses the tails of the distribution (model collapse,
  Shumailov et al., 2024). Gains often stop after 1 to 2 rounds (ReST-EM, our own RLAIF early-stopping after 320 to
  800 pairs). RL with verifiable rewards often raises pass@1 while pass@k at large k stays flat or drops (Yue et al.,
  2025): it sharpens what the base model could already do instead of extending it.
- **A Qwen-specific trap.** Qwen2.5-Math models gained on MATH under RL with random or even wrong rewards (Shao et
  al., 2025), likely by surfacing skills from pretraining. Our base is a Qwen model, so any gain must beat a
  random-label control before we call it self-improvement.
- **Our own data point.** v7's math GRPO (mostly GSM8K plus DeepMath, 300 steps) showed no gain. GSM8K is likely
  saturated for this model, so the loop below trains on harder problems where the model is right only some of the time.

These citations are from memory (knowledge through mid-2026) and were not re-checked online for this note.

## What a positive result would have to show

1. A gain on sealed sets (GSM8K test, MATH-500, MBPP test, HumanEval+) that never enter any training path; the
   problem builder drops any question whose hash is in `SEALED.json` or the sealed files.
2. Gains in rounds 2 and 3, each against the previous round, judged by an exact McNemar test on paired items
   (`rsi.py compare`, p < 0.05 on the pooled sealed sets). A round-1 jump alone does not count.
3. The verified arm beats the random-label control.
4. No regression on the censorship, safety, identity and sovereignty probes (train_vl.py runs them every round),
   and no rise in the share of math answers missing a boxed answer (`no_boxed`), so the gain is not format only.
5. pass@8 on MATH-500 and dev reported each round, so we can say whether the loop extends the model or only sharpens it.

## The loops (all in `tools/rsi/`)

One shared round, five label sources. The model always writes every training target; only where the label comes from
changes. Training data per round is SFT on positive answers plus DPO pairs (positive over negative, same problem),
through `tools/vl/train_vl.py` with `--init-adapter` and its held-out DPO early stopping.

**Experiment A: verified expert iteration (run first).** Problems: about 4,000 MATH (levels 3 to 5) and DeepMath
items, 500 GSM8K, all of MBPP train (MIT / CC BY 4.0, already on the volume via `fetch_public.py`); 6 percent held
out as a dev split. Each round, the current model answers each problem 8 times (temperature 0.8, thinking off); the
verifier labels answers (`verify.py`: reference answer match, or every MBPP unit test in an isolated subprocess);
problems with no correct answer get one STaR rationalization pass (reference as a hint, trained on the plain prompt).
Problems the model always solves are downsampled to 20 percent, so the signal sits where it is sometimes right. Each
round trains from the RLAIF-1 adapter on the newest model's data (ReST-EM style, `INIT=fixed`); `INIT=chain` is the
alternative.

**Experiment B: self-labeled loop (the true RSI test).** Same round, but the verifier is hidden from training. Labels
come from the model itself: majority vote over its 8 math answers (needs at least half the votes, TTRL style) AND its
own YES/NO judgment of each answer (P(YES) from its first-token logprobs); for code, the one test shown in the prompt
AND its own judgment. When the two self-signals disagree the answer is dropped. The verifier still runs, only to
report the self-labels' precision and recall each round (`label-report.json`). That number is the
generation-verification gap, measured directly on our model.

**Control: random labels.** The verified labels shuffled within each problem (same counts, no information). One
round, run only if A shows a gain.

**Experiment C (design only): self-play on code.** The model proposes small programs plus inputs; Python runs them
for ground truth; the model then predicts outputs or writes programs to match. Proposals are kept when the solver's
pass rate is between 0.2 and 0.8. This is the Absolute Zero / R-Zero pattern and the only loop that is not capped by
a fixed problem set. Not built: it is the riskiest, and A and B tell us first whether this model can climb at all.

## Data rule

The loop reads only problems with reference answers or unit tests: GSM8K, MATH (the final answer taken from the
reference solution), DeepMath final answers, and MBPP tests. It never reads the OpenMathReasoning or OpenCodeReasoning
solution sets (DeepSeek-R1 and QwQ outputs), or any other model-written solutions. Every training target comes from
the policy itself. The starting adapter is a separate question: v7 Lite's SFT data included those solution sets, so
which Lite the loop starts from (`START`) waits on Hayven's v7 decision.

## Prototype status

- `tools/rsi/verify.py`: math answer check (reuses the v7 GRPO reward's checks) and an isolated code runner.
- `tools/rsi/rsi.py`: `problems`, `sample`, `rationalize`, `label`, `build`, `eval`, `compare`, and `smoke`.
- `tools/rsi/job_rsi.sh`: resumable pod job, one arm per pod; preflights every input, runs the CPU smoke, then the
  round-0 eval and up to `ROUNDS` rounds; stops early when a round is significantly worse than the previous one.
  `SMOKE=1` runs 24 problems, k=2 and `train_vl.py --smoke` end to end. It reads `/workspace/krummholz-v7/` and writes
  only `/workspace/krummholz-rsi/<name>/<arm>/`; it refuses an output path inside `krummholz-v7`.
- `tests/test_rsi.py`: 18 tests (verifier, label modes, self modes never reading the verifier, sealed-set exclusion,
  easy-problem downsampling, McNemar). The full suite passes on CPU (278 tests, 5 skipped).
- Not yet run on a GPU: vLLM sampling, the logprob self-judge, and `train_vl.py` on these files. The pod smoke covers them.

## Cost and time (one H100, about $3 per hour; check the live price before launch)

| Step per round (Lite 4B) | Time |
|---|---|
| Merge adapter, load vLLM | 5 to 10 min |
| Sample 4,600 problems x 8 (about 16M tokens) | 15 to 25 min |
| Verify, rationalize (B: self-judge 37k answers instead) | 5 to 15 min |
| SFT about 4,000 examples, DPO with early stopping, probes | 45 to 60 min |
| Sealed eval (about 2,500 items greedy, 800 x 8 sampled) | 10 to 15 min |
| **Round total** | **about 1.5 to 2 hours, $5 to $6** |

| Run | Rounds | Estimate |
|---|---|---|
| Pod smoke (`SMOKE=1`) | 1 tiny | about $1.50 |
| A: verified | round-0 eval plus 3 | about $20 |
| B: self-labeled | round-0 eval plus 3 | about $20 to $22 |
| Control: random labels | 1 | about $6 |
| **All of it** | | **about $48, cap $55** |

Experiment A alone with its pod smoke is about $22 (cap $25). Nothing has been launched, and nothing will be without
Hayven's OK. With about $90 shared by three RLAIF runs, the sensible time is after those finish.

## What I would run first, and why

**Experiment A (verified expert iteration), after a $1.50 pod smoke.** It is the cheapest loop with an external check, so
any gain cannot be the model fooling itself, and it reuses the trainer and the data already on the volume. It also
answers the prerequisite: if Lite cannot climb for 3 rounds even with a perfect grader, a self-graded loop will not
either, and we save the $20 for B. If A gains, B is the investor-relevant follow-up, and its label precision tells us
how much of A's gain survives without the verifier. The random control then guards against the Qwen
spurious-reward effect.

Expectation, stated plainly: a round-1 gain on MATH-500 of a few points is likely; rounds 2 and 3 adding smaller but
significant gains is plausible; B matching A at 4B would be surprising. A clean result either way is useful. The
claim we could make to investors is narrow: "self-improvement on verifiable math and code over N rounds," not general
self-improvement.

## Launch (when approved)

```bash
NAME=atlantic-lite-4b MODEL=Qwen/Qwen3.5-4B ARM=verified SMOKE=1 bash tools/rsi/job_rsi.sh
NAME=atlantic-lite-4b MODEL=Qwen/Qwen3.5-4B ARM=verified ROUNDS=3 bash tools/rsi/job_rsi.sh
```

Prerequisites: the RLAIF round 1 Lite adapter at `/workspace/krummholz-v7/rlaif1/atlantic-lite-4b/train/adapter`
(the job's preflight checks it), and the branch pushed so the pod can clone it (needs Hayven's OK; the repo is public
and this branch holds no private data).
