# Atlantic v9 evolution strategies on LoRA singular values

This directory holds the engine for the Atlantic v9 post-training run: evolution strategies (ES)
over a small search space, the singular values of the cumulative Atlantic v8 LoRA adapter. The run's
pre-registered rules are in [PROTOCOL-atlantic-v9.md](PROTOCOL-atlantic-v9.md); `config/atlantic-v9.yaml`
carries the same values and `tests/test_config.py` checks the main ones.

Atlantic v8.2 ran full-parameter ES (26.9 billion text parameters, 32 generations, 16 directions,
eight record pairs per generation) and gave no measurable gain on the 64-question comparison
(54 versus 53, *p* = 1.00). Version 9 changes the search, not the objective's spirit:

* **Search space.** Each LoRA delta B @ A is factored once as U diag(S) V^T. U and V stay fixed and
  only S moves, one coordinate per singular value (rank 16 times the number of adapted modules;
  `python -m es.es_loop run` logs the exact dimension). This follows ESSA (arXiv 2507.04453), which
  evolved LoRA singular values with populations of 48 to 64 and batches of 256 prompts.
* **Signal per generation.** 24 antithetic pairs (48 perturbed candidates, 24 independent
  directions) plus the unperturbed parent, all answering the same 128 prompts: 112 present/absent
  pair items from the yardstick training pool and 16 retention prompts (12.5%).
* **Length.** 200 generations instead of 32, with full checkpoint/resume every generation.

## Method

For generation g with parent theta (FP32 master; theta = 0 is Atlantic v8 exactly):

1. Seeds `seed_j = derive_seed(run_seed, "noise", g, j)`; `eps_j ~ N(0, I)` from NumPy PCG64.
2. Candidates: parent, and `theta + sigma eps_j`, `theta - sigma eps_j` for j < N.
   With the relative parameterization a candidate's singular values are `S0 * (1 + theta)`.
3. Every candidate is written as a PEFT adapter with the source adapter's keys, rank and
   `adapter_config.json`, using `B' = U diag(sign(S) sqrt|S|)`, `A' = diag(sqrt|S|) V^T`.
4. One vLLM engine answers the batch for all candidates (per-request `LoRARequest`).
5. Fitness per candidate:
   `mean grader reward over pair items + 0.5 x (fraction of pairs correct in both variants)
   - 1.5 x (fraction of retention items failed)`. Truncated or unfinished replies get -1.
6. Tie-averaged centered ranks over the 48 perturbed fitnesses give utilities `u`.
   Coefficients `c_j = (u_j+ - u_j-) / (2 N sigma)`; gradient `g = sum_j c_j eps_j`.
   The parent's same-batch score is logged as the reference (a constant baseline cancels in
   antithetic differences, so it does not enter the update).
7. Plain SGD ascent with `lr = 1.7 sigma^2` (Adam is available for other runs), then Anchored Weight
   Decay toward the start: `theta = (1 - awd) (theta + lr g)` with `awd = 0.005`. Optional clipping
   is off for this run.
8. The journal line (seeds, coefficients, fitnesses, theta hashes before and after) is fsynced,
   then a full checkpoint (theta, Adam moments, generation, elapsed time) is written atomically.

Dev checks run at generation 0 (Atlantic v8) and every 10 generations on a fixed reference batch
(all 64 dev pairs, ranked by `sha256("20261127|" + group)`, plus 32 prompts from a separate dev
retention pool, `validation.retention_path`). The loader refuses a dev pool that shares a group, item
id, record id, entity or block with the training pool, and a dev retention pool that shares an id or a
normalized prompt with the training retention pool. Generations 0 and 40 are always checked, whatever
`validation.every` is, so G4 cannot be skipped. Dev checks feed the stop rules and the report; they
never select a checkpoint. The exported adapter is always the final theta. Stop rules (`gates.py`,
`gates` in the config):

* G4 at generation 40: the composite reward must beat generation 0 by more than 2 SE (paired cluster
  bootstrap over dev source blocks, 2,000 resamples), or ES ends.
* G5 at generation 40: at least 150 generations must be projected within the $220 ceiling.
* S1: no generation starts after $215 of ES spend or if it would end past $220; 50 hours at most. At
  $220 the run stops even inside a generation (`budget.hard_kill`): between waves the loop stops and
  exports the last committed theta, and a watchdog thread (`budget.heartbeat_seconds`) exits the process
  if a single long generate() call crosses the line. Billed time lives in `billing.json`, which a resume
  never rolls back, so time lost to a crash or a lost pod still counts. Time spent in `preflight`,
  `parity`, `coverage`, `calibrate` and `export` is added there too; `budget.overhead_hours` is only for
  pod setup and idle time.
* S5: 3 or more extra dev retention failures versus generation 0 at two consecutive checks.
* S6 pauses without committing the generation: grader exceptions above 1% or incomplete replies above
  5% of a generation (retention replies included), a non-finite theta, an adapter round-trip error above
  1e-3 at a dev check (checked before any cached dev result is reused, so a resume cannot skip it; BF16
  candidates measure about 1e-4 on the tiny test model), or grader exceptions above 1% at a dev check.
  A grader file whose SHA-256 differs from the checkpoint's refuses to resume, and so does a change to
  the engine's numeric settings (`dtype`, `fp8_base`, `kv_cache_dtype`, `lora_dtype`, `max_model_len`);
  memory and scheduling settings may change between pods. The terminal held-out set and the conflict
  set are never read by this code; the loader raises on `terminal.jsonl`, `conflict.jsonl` and on any
  item whose `split` is `terminal` or `conflict`.

Method credit: Salimans et al., *Evolution Strategies as a Scalable Alternative to Reinforcement
Learning* (arXiv 1703.03864) for antithetic sampling and centered-rank shaping; Qiu et al.,
*Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning* (ICML) for ES on
language models; ESSA (arXiv 2507.04453) for searching LoRA singular values.

## Files

| file | role |
| --- | --- |
| `essa.py` | compact SVD of each LoRA delta without forming it, candidate singular values, PEFT writing, decoding a written adapter back into theta |
| `es_loop.py` | generation loop, shaping, SGD/Adam, AWD, journal, checkpoints, resume, budget and wall-clock guards, STOP file, dev checks, pauses, preflight, parity, sigma calibration, `plan`, `replay`, `export` |
| `gates.py` | dev gate (paired cluster bootstrap), pace and retention stop rules |
| `fitness.py` | grader adapter interface (`grade(item, reply) -> {verdict, correct, reward}`), deterministic retention checks, candidate scoring, parallel grading, grader preflight |
| `data.py` | pair loading, terminal-set refusal, stateless epoch-based batch sampling, fixed dev batch, prompt construction |
| `backends/vllm_backend.py` | vLLM offline multi-LoRA: per-request LoRA, prefix caching, `max_loras` and `max_cpu_loras`, optional FP8 base and FP8 KV cache, vision disabled |
| `backends/hf_backend.py` | slow Hugging Face plus PEFT reference for CPU tests |
| `backends/mock_backend.py` | deterministic model-free backend for loop tests |
| `config/atlantic-v9.yaml` | every hyperparameter, with the budget arithmetic |
| `tests/` | CPU tests (see below) |

## Data and privacy

Evaluation items, the training pool, the retention slice, the grader and the system prompt are
private. The config references pod copies under `/workspace/yardstick` and `/workspace/pilot`; none
of these files belong in this public repository. Copy them to the pod with the private-file method
in the repository handoff, and do not copy `heldout/terminal.jsonl` to the pod for this run.

Training signal comes only from the policy's own replies, scored by the deterministic grader and
the retention checks. No Claude-generated training text is used.

## Running on the pod

From the repository checkout on the pod (`/workspace/krummholz-llms-alpha`), in the pod's shell:

    cd /workspace/krummholz-llms-alpha
    C=es/config/atlantic-v9.yaml; R=/workspace/es-runs/atlantic-v9
    python -m es.es_loop plan --config $C --replies-per-s 8,12,20
    python -m es.es_loop preflight --config $C --run-dir $R
    python -m es.es_loop parity --config $C --run-dir $R
    python -m es.es_loop coverage --config $C --run-dir $R
    python -m es.es_loop calibrate --config $C --run-dir $R
    nohup python -m es.es_loop run --config $C --run-dir $R > $R.out 2>&1 &

* `preflight` grades every pool item's code-template `target` and refuses to start below 99%
  acceptance (an earlier grader build rejected 433 of 4,800 targets; the current build accepts all).
  `run` repeats it at the start of every session.
* `parity` (gate G0 on the pod) compares the original v8 adapter with the rebuilt `theta = 0` adapter
  on 64 training-pool prompts and needs at least 63 identical public replies. Both adapters run in the
  same engine, so parity cannot see a module kind the engine skips in both.
* `coverage` (also G0) amplifies one module group at a time (layer index wildcarded, for example
  `layers.*.linear_attn.in_proj_qkv`; singular values times 4) and needs every group to change at least
  one of 8 full completions against `theta = 0`. A group that changes nothing is one vLLM is not
  applying (for example Gated DeltaNet projections inside the multimodal wrapper); searching it would
  waste the budget and the exported adapter would change those modules untested. With
  `gates.require_g0: true`, `run` refuses to start generation 0 unless `parity.json` and
  `coverage.json` passed for the same decomposition and backend numerics.
* `calibrate` scores 4 antithetic directions for each sigma in {0.02, 0.05, 0.10, 0.20, 0.30} on a
  separate batch and writes `recommended_sigma` and `recommended_lr` (the protocol's rule: smallest
  sigma with flip rate at least 0.20 and nonzero antithetic differences, one step smaller if it costs
  more than 2 extra retention failures). Set `es.sigma` before generation 1; sigma is part of the
  config hash and cannot change mid-run. `preflight`, `parity` and `calibrate` do not pin the config, so
  they can share the run directory before the first `run`; put their measured hours in
  `budget.overhead_hours`.
* Progress: `status.json` (generation, hours, dollars so far), `journal.jsonl` (per generation: seeds,
  coefficients, every candidate's fitness terms, theta hashes, cumulative hours and dollars),
  `val.jsonl` and `val/` (per-item dev rewards), `gates.jsonl`, `run.log`. Every reply is archived as
  gzip JSONL under `$R/replies/` on the pod volume. `touch $R/STOP` stops cleanly after the current
  generation and exports the adapter. A paused run (S6) writes the reason to `status.json`.
* Resume after a lost pod: rerun the same `run` command. The latest valid checkpoint is loaded,
  journal lines past it are dropped, and saved partial results of the interrupted generation are
  reused. Budget and wall-clock limits count time across sessions. `run` holds `RUN.lock` in the run
  directory, so a second `run` on the same directory is refused. `export`, `preflight`, `parity`,
  `coverage` and `calibrate` are safe beside a live run: they write candidates to their own scratch
  directory and never truncate the journal or rewrite `config.json`.
* Audit: `python -m es.es_loop replay --run-dir $R` rebuilds theta from the journal alone and checks
  every recorded hash and the checkpoint.
* Output: `$R/final-adapter` is a drop-in PEFT replacement for the v8 adapter (same keys, rank and
  config) with `ES_PROVENANCE.json`. Score it once on the terminal set with the separate evaluation
  harness.

## Memory on one H100 80GB

* BF16 base about 54 GB. KV cache uses most of the rest at `gpu_memory_utilization: 0.92`.
* `max_loras: 4` sets GPU LoRA slots (small at rank 16). `max_cpu_loras` defaults to the
  population (49 host copies). Candidate adapters are written in BF16 to `essa.adapter_tmp_dir`;
  check the v8 adapter size with `du -sh` and multiply by 49 for the scratch space needed per
  generation.
* If KV space is too small for 256 concurrent sequences of about 1,300 tokens, set
  `backend.engine.fp8_base: true` (about 27 GB base) or `kv_cache_dtype: fp8`. Scores then come from a
  different numeric base than the release path; prefer BF16 if the pilot shows it fits.
* Prefix caching keys cached blocks by LoRA id, so the constitution prefix is computed once per
  candidate and reused for its 128 prompts.
* Choose `wave_size` (0 = all candidates per call) and the backend from the pilot's measurements in
  `pilot/PILOT.md` once it exists.

## CPU tests

The tests need CPU builds of torch, transformers, peft and safetensors, plus PyYAML and NumPy.
`test_hf_tiny.py` uses Qwen/Qwen2.5-0.5B-Instruct from the local Hugging Face cache with a random
rank-4 LoRA on every language-model linear layer, and skips when the model is absent.
`test_yardstick_integration.py` runs only when `ES_YARDSTICK_DIR` points at a yardstick checkout.
On Windows, from the repository root in PowerShell:

    py -3.12 -m venv .venv-es
    .venv-es\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
    .venv-es\Scripts\pip install transformers peft safetensors pyyaml numpy pytest
    .venv-es\Scripts\huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
    .venv-es\Scripts\python -m pytest es\tests -q

What they prove:

* `test_essa.py`: SVD reconstruction exact at theta = 0, written candidates carry exactly the intended
  singular values, negative values survive, PEFT layout and dtypes match the source, modules narrower
  than the rank keep rank-r shapes, exclusions pass through unchanged, the decomposition cache
  round-trips, DoRA is rejected.
* `test_es_math.py`: centered ranks with ties, antithetic symmetry (equal pair fitness gives a zero
  gradient; swapping signs negates it), pinned seed and noise values, and the gradient, SGD,
  momentum, two Adam steps, AWD, clipping and the cosine schedule against hand calculations.
* `test_data_fitness.py`: terminal- and conflict-set refusal, train/dev and retention-pool disjointness,
  pair loading, stateless batches that cover each epoch, prompt construction, reply parsing, the
  fitness formula by hand, incomplete retention replies in the S6 rate, fail-closed grader output,
  retention checks, parallel grading equal to serial, grader preflight.
* `test_loop_mock.py`: ES climbs a hidden objective; crash at six different points (including a hard
  `os._exit` in a subprocess) and resume reproduces the uninterrupted run's journal, validation and
  final adapter bytes; journal replay matches; wall-clock, dollar, no-start and STOP guards; dev gate
  and pace gate stops; grader-error pause and recovery; the calibration rule; config changes
  outside the resume-mutable sections are refused. Review fixes: dev retention from its own pool;
  round-trip and dev grader-error pauses that survive a resume; G4 and G5 run off the dev-check
  cadence and unevaluable gate settings are refused; engine numerics cannot change on resume; billed
  time survives a crash; hard stop between waves and watchdog exit inside a slow generation; model
  load kept out of generation time; export and auxiliary commands leave a live run alone and bill
  their time; a second `run` process is refused; coverage finds module groups a backend ignores;
  `require_g0` blocks launch until parity and coverage pass.
* `test_gates.py`: block clustering, the paired composite bootstrap by hand, retention streaks, pace.
* `test_vllm_backend.py`: against a fake `vllm` module, engine arguments (LoRA, rank 16, prefix
  caching, FP8, residency, no images), per-request LoRA routing, fresh adapter ids that are never
  reused, adapter removal and cleanup of temporary adapters, and refusal of `skip_special_tokens`
  when it would delete `<think>` or `</think>`.
* `test_hf_tiny.py`: on the real tiny Qwen, a zero perturbation reproduces the parent's logits and
  greedy text, the LoRA matters, small antithetic perturbations move logits in opposite directions,
  a written candidate loads with `PeftModel.from_pretrained` and matches the in-place swap, the
  loop resumes deterministically after a crash, `parity` finds the original and rebuilt adapters
  identical, and `coverage` sees all 7 module kinds.
* `test_config.py`: the config sets every hyperparameter, the budget arithmetic, the resume-mutable
  hash, no item files or terminal set in the repository, no em dashes.
