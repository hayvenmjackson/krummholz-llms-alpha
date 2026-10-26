# Original-BF16 GPU comparison

This is a prepared evaluation harness, not a completed model evaluation or an
application release. It compares two original BF16 checkpoints against the
unchanged 36-case fictional source-QA pilot. It performs no training and has no
tool executor. Earlier models' failed answers remain part of their own evidence.

## Frozen comparison

| Model | Exact publisher revision | Mode | Primary sampling |
| --- | --- | --- | --- |
| [Ai2 OLMo 3 7B Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct/blob/6e5971d9eba42665f5bd5a0fcf047f299ce1dccc/README.md) | `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc` | Instruct checkpoint; no thinking switch or forced reasoning prompt | Temperature 0.6, top-p 0.95, top-k 50; min-p filter disabled |
| [Qwen3 8B](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/README.md) | `b968826d9c46dd6066d109eabc6255188de91218` | Publisher `enable_thinking=False` | Temperature 0.7, top-p 0.8, top-k 20, min-p 0 |

Each model runs all 36 cases with primary sampling seed 42, then primary sampling
seed 1337, then a separate secondary greedy control with seed 42: **216 scheduled
attempts**. The seed is reset before each individual request. There is no
best-of selection, retry to replace a poor answer, cross-case conversation
history, or prompt tuning against the pilot's expected answers.

OLMo's pinned card recommends temperature 0.6 and top-p 0.95; its Transformers
example uses top-k 50, also the default for this runtime. Qwen's settings are its
explicit non-thinking recommendation. Greedy decoding is a separately labelled
control, not a replacement for either publisher's primary settings.

The shared system instruction is exactly `qa_system` in the frozen
`config/smollm3-reference.json`. The user message contains only the case's query
and supplied sources. Expected answers, rubrics, rejection criteria and required
source IDs never enter the model prompt. Each tokenizer renders its own pinned
publisher chat template with tools disabled. Qwen's template adds its empty
thinking prefix; the harness does not strip generated reasoning or repair JSON.
Neither template needs a date override. Both rendered text and token IDs are
saved, with a check that direct template tokenization produces the same IDs.

The output limit is **384 new tokens**, below the pilot's 1,024-token ceiling.
Input plus this output allowance must fit 4,096 tokens. A publisher EOS is
required; complete-looking JSON at the token ceiling is not accepted without
EOS. Both publishers' pinned generation configurations declare two EOS IDs;
the harness accepts either and removes only the single terminal EOS before
strict parsing. Each request has a 120-second generation/request deadline.

## Source and runtime verification

`config/reference-comparators.json` pins every downloaded file's size and
SHA-256, the model revisions, template hashes, runtime versions, settings and
limits. The runner also pins the configuration's canonical SHA-256:

`e4dc6673e9fe036bcf62fd1fb07d3dddb994d53d0c93c8f9f8603cbd2ed33ad1`

Large weight hashes come from the official Hugging Face LFS metadata at those
revisions. Small metadata, tokenizer and license files were downloaded and
hashed during preparation. Full weight bytes are verified on the remote worker
before loading; **they have not been downloaded or tested locally**.

The source-QA suite hash remains
`6eda5672c65edf015409198eb21ed9f13ca346fde9d914f51497a38c3065a541`.
The frozen QA reference specification hash remains
`8503c0a847c05828517f4b160a0b194659020ea27cd55ca8d4cfa7034e5cf8c6`.
The runner rejects silent changes to any of these frozen inputs.

Use Python 3.12, PyTorch 2.9.1+cu128, Transformers 4.57.1, tokenizers 0.22.2,
safetensors 0.6.2, huggingface-hub 0.36.0 and Jinja2 3.1.6. The
[Transformers OLMo 3 implementation](https://huggingface.co/docs/transformers/v4.57.1/model_doc/olmo3)
supports this checkpoint; Qwen's card requires at least Transformers 4.51.
The [official PyTorch installation archive](https://pytorch.org/get-started/previous-versions/)
provides the pinned CUDA 12.8 wheel.

Loading uses the standard installed Transformers implementation,
`trust_remote_code=False`, verified local safetensors and BF16 CUDA parameters.
All source tensors must be BF16. There is no quantization, CPU offload, remote
model code or generated code execution. Inference uses SDPA attention,
deterministic PyTorch algorithms and a fresh generation cache for each request.
`use_cache=True` is explicit, including for OLMo whose model configuration
defaults it to false. Kernel/version/hardware differences can still affect
reproducibility; the run records these details rather than promising identical
answers across every GPU.

Both publishers declare Apache 2.0. Qwen includes a pinned LICENSE file; this
OLMo revision declares the license in its pinned model card and points to
[Ai2's Responsible Use Guidelines](https://allenai.org/responsible-use).
Those declarations do not independently establish rights in every training
source. The harness does not redistribute model files.

## Remote prerequisites and budget

Actual inference is restricted to Linux Runpod workers; local `--plan` and tests
do not import torch or start inference. Cloud provisioning, account access,
funding and shutdown are handled separately.

- One idle **24 or 48 GiB-class GPU** with native BF16 support, at least 21 GiB
  free VRAM, and a driver compatible with the CUDA 12.8 PyTorch wheel.
- At least **24 GiB effectively available RAM**. The check takes the smaller of
  host MemAvailable and each visible cgroup v1/v2 memory limit minus its current
  usage plus conservatively reclaimable clean inactive file cache, including
  visible ancestor groups. Active cache, dirty/writeback pages and anonymous
  memory receive no reclaim credit. All raw counters are recorded; the effective
  allowance cannot exceed the container's limit. This prevents model downloads from being
  mistaken for permanently occupied RAM. Unknown container accounting fails
  preflight. Choose a pod with at least 32 GiB allocated RAM so the operating
  system and Python have room. The check runs again after download, before loading.
- About **31.0 GB** of pinned model files in total, plus **8 GiB free-disk reserve**
  and space for the Python environment. A 60 GB or larger disk is a practical
  starting point; inspect actual free space before launch.
- A current confirmed **all-in hourly rate** and the amount already spent on
  this evaluation, including pod setup/package installation.
- A separately verified pod-stop watchdog arranged by the authorized cloud
  operator, plus confirmation that the evidence directory under `/workspace` is
  on persistent storage. A workload timeout alone is not a billing safeguard.

The evaluation allocation is $20. At least $2 stays reserved. The script sets
its workload limit to the smaller of two hours or the time affordable from the
remaining allocation. Downloading, hashing and model loading happen inside that
limit. The first model receives half the remaining work time; the second gets
the remainder. Thirty seconds are reserved for evidence finalization.

The program **does not stop or terminate the paid pod**. Stopping Python is not
a billing stop. Retain the evidence, then promptly stop the pod through the
authorized cloud operator. Stored disks may continue to incur charges.
Do not rely on historical `runpodctl --stop-after` or `--terminate-after` flags
as a shutdown guarantee. Keep evidence on the verified persistent `/workspace`
volume; container-disk files do not survive a stop, and volume persistence
depends on the selected stop/termination action and storage type.
Consult the current [Runpod pricing](https://docs.runpod.io/pods/pricing) and
[pod lifecycle documentation](https://docs.runpod.io/pods/manage-pods). The
software's cost estimate cannot override the provider's bill or account for an
incorrect rate. No prices are hard-coded as a claim about current availability.

## Prepare and run

From the checked-out repository on the authorized pod, using Python 3.12:

```bash
python3.12 -m venv .venv-gpu
.venv-gpu/bin/python -m pip install --upgrade pip
.venv-gpu/bin/python -m pip install 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cu128
.venv-gpu/bin/python -m pip install 'transformers==4.57.1' 'tokenizers==0.22.2' 'safetensors==0.6.2' 'huggingface-hub==0.36.0' 'Jinja2==3.1.6'
.venv-gpu/bin/python -m unittest tests.test_gpu_comparison tests.test_pilot_eval
.venv-gpu/bin/python tools/run_gpu_comparison.py --plan
```

Record the actual quoted all-in rate and current prior spend. Replace the two
placeholders below with those confirmed numeric values; they are deliberately
not estimated here. Use a new output directory for each run:

```bash
HOURLY_USD=CONFIRMED_ALL_IN_HOURLY_RATE
PRIOR_SPEND_USD=CONFIRMED_SPEND_BEFORE_THIS_RUN
.venv-gpu/bin/python tools/run_gpu_comparison.py \
  --run \
  --work /workspace/krummholz-model-cache \
  --output /workspace/krummholz-gpu-evidence/run-001 \
  --hourly-usd "$HOURLY_USD" \
  --already-spent-usd "$PRIOR_SPEND_USD" \
  --reserve-usd 2
```

No Hugging Face token is needed for these public files. Download URLs use exact
revisions, and cached files are reverified before reuse. Existing mismatched or
partial files cause a visible failure; the harness does not silently overwrite
them. Both models remain on disk, and the harness performs no automatic cleanup.
Only one model process occupies the GPU at a time.

An output directory cannot be reused or nested inside the model cache. A model
setup failure, CUDA failure or hard watchdog timeout leaves that model's
remaining attempts explicitly unrun. The other model can still proceed within
the remaining allocation. A GPU hang is killed at the process boundary; the
same process is never silently retried. The script exits nonzero when attempts
remain incomplete. This is useful diagnostic evidence, not permission to select
only successful cases.

## Read the evidence

`run.json` records inputs, helper/source hashes, billing assumptions, limits and
per-model execution. `python-packages.json` and `artifact-hashes.json` record the
actual dependency inventory and evidence hashes. Each model records verified
source hashes, device/runtime metadata, loading time and worker logs.

Each case/variant preserves the exact prompt, input IDs, generated IDs, raw
completion, timings and structured result. Partial token journals are marked
diagnostic and never become accepted answers. An EOS-complete answer must still
pass strict JSON shape, source-ID and literal-quote validation. These checks
cannot establish that its factual claims are supported.

There are six separate `responses.json` files, each with all 36 cases, and six
deterministic reports. `summary.json` counts all 216 scheduled outcomes without
combining seeds into a best answer. All semantic reviews begin empty; application
approval remains false. Use the frozen rubrics in
[the pilot evaluation guide](pilot-evaluation.md) to review the actual answers,
record agent semantic review explicitly, and distinguish it from independent
human or device validation. Compare both primary seeds, the secondary control,
and failure categories; do not present schema success as semantic accuracy.

This small fictional pilot is a development comparison, not a representative
production benchmark. It does not test unrestricted web access, external
actions, real client data, mobile performance or long-context reasoning.
