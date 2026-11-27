# Vision in the Atlantic v9 ES run

The owner requires that vision is trained too, not only text (2026-11-27). Evolution strategies can
only improve what the fitness rewards, and only the weights in the search vector. Both change:

1. **Image items in the fitness and the evaluation.** About 25 percent of every candidate's batch is
   image items from the vision yardstick, graded by its deterministic grader inside the one fitness.
   The dev checks carry image items too.
2. **Vision weights in the search space.** A low-rank, seeded subspace over the vision tower is
   applied in place to the served weights (no adapter support needed) and searched jointly with the
   text vector.

Files: `vision.py` (inventory, search space, in-place patcher, ES on the vision vector, export),
`backends/vision_hooks.py` (worker RPC hooks for vLLM, the Hugging Face fallback, request helpers),
`config/atlantic-v9-vision.yaml` (overlay on `atlantic-v9.yaml`), `tests/test_vision.py` (CPU tests).

Status: code and CPU tests are done. The changes to `es_loop.py`, the backends, `fitness.py`,
`data.py` and the protocol listed in section 4 are not made yet (those files belong to the text
engine work in progress). Nothing has run on the pod or a GPU.

## 1. Atlantic's vision tower

Read from Qwen/Qwen3.8-27B at revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0`: `config.json`,
`preprocessor_config.json`, the safetensors header of `model-00001-of-00018.safetensors` (all 333
vision tensors live there), transformers `modeling_qwen3_5.py` (`Qwen3_5VisionModel`) and vLLM
`qwen3_5.py`, which reuses `Qwen3_VisionTransformer` from `qwen3_vl.py`. Every tensor is BF16.

| module | tensors | shape | parameters |
| --- | --- | --- | ---: |
| `patch_embed.proj` (Conv3d, kernel = stride = 2 x 16 x 16) | weight, bias | [1152, 3, 2, 16, 16], [1152] | 1,770,624 |
| `pos_embed` (learned 48 x 48 grid, bilinearly resampled per image) | weight | [2304, 1152] | 2,654,208 |
| `blocks.N.norm1`, `blocks.N.norm2` (LayerNorm) | weight, bias | 4 x [1152] | 4,608 |
| `blocks.N.attn.qkv` (fused q, k, v; 16 heads of 72) | weight, bias | [3456, 1152], [3456] | 3,984,768 |
| `blocks.N.attn.proj` | weight, bias | [1152, 1152], [1152] | 1,328,256 |
| `blocks.N.mlp.linear_fc1` (gelu_pytorch_tanh) | weight, bias | [4304, 1152], [4304] | 4,962,512 |
| `blocks.N.mlp.linear_fc2` | weight, bias | [1152, 4304], [1152] | 4,959,360 |
| one block, N = 0..26 | | | 15,239,504 (x 27 = 411,466,608) |
| `merger.norm` (LayerNorm over 1152, before the 2 x 2 merge) | weight, bias | 2 x [1152] | 2,304 |
| `merger.linear_fc1` (GELU) | weight, bias | [4608, 4608], [4608] | 21,238,272 |
| `merger.linear_fc2` (projector into the 5120-wide language model) | weight, bias | [5120, 4608], [5120] | 23,598,080 |
| merger | | | 44,838,656 |
| **tower** (333 tensors) | | | **460,730,096** |

There are no deepstack mergers (`deepstack_visual_indexes: []`); the vision rotary embedding has no
parameters. `python -m es.vision inventory --table` prints every tensor with its GGUF name.

**Image input.** `Qwen2VLImageProcessorFast` with mean = std = 0.5 (pixels to [-1, 1]); each side is
resized to a multiple of 32 within 65,536 to 16,777,216 pixels; each patch is 3 x 2 x 16 x 16 =
1,536 values (a still image is repeated over the two temporal frames); `grid_thw = (1, H/16, W/16)`;
the language model sees H x W / 1024 image tokens (512 x 512 gives 1,024 patches and 256 tokens).
The chat template renders an image part as `<|vision_start|><|image_pad|><|vision_end|>` (ids 248053,
248056, 248054) and vLLM expands the pad token to the image's token count; the merger outputs replace
those embeddings. The overlay caps images at 262,144 pixels (512 x 512).

**Names.** Checkpoint and transformers: `model.visual.*`. vLLM: `visual.*` (with tensor parallel 1
the fused qkv keeps the checkpoint's q, k, v row order; the attach check below proves it byte for
byte). GGUF mmproj: `v.blk.N.attn_qkv`, `attn_out`, `ffn_up`, `ffn_down`, `ln1`, `ln2`; `mm.0`
(merger fc1), `mm.2` (merger fc2), `v.post_ln` (merger norm), `v.patch_embd`, `v.position_embd`.

## 2. Search space

For each chosen tensor W, viewed as a matrix rows x cols, and its slice c of the vision vector:

    W(c) = W0 + s_W * U diag(c) V^T          (core "diag", r coordinates; "full" uses an r x r core)
    b(c) = b0 + s_b * U c                    (biases and LayerNorm vectors, r coordinates)

* U (rows x r) and V (cols x r) are fixed orthonormal bases: QR of PCG64 Gaussians seeded by
  `derive_seed(basis_seed, "vision-basis", tensor name, "U" or "V")`, signs fixed. They are built once
  per run and stored in `vision-space/bases.safetensors`; `space.json` holds each tensor's offset,
  rank, scale and the sha256 of its base bytes, and the space fingerprint covers all of it. Resume
  loads them from disk, so it never depends on LAPACK reproducing a QR.
* `s_W = ||W0||_F / sqrt(min(rows, cols))` is the base tensor's RMS singular value (vectors: RMS entry,
  floored at 1e-3), so one sigma means the same relative size in every tensor. With orthonormal
  bases a perturbation gives each of the r chosen directions a singular value sigma_v x s_W x |eps_k|.
  At sigma_v = 0.05 that is about 5 percent of an average singular value per direction, and a
  Frobenius change of sigma_v x sqrt(r / min(rows, cols)): 1.0 percent of a block matrix (r = 48) and
  0.8 percent of a merger matrix (r = 128).
* A candidate's vision state is its FP32 coefficient vector. The ES master z is FP32; z = 0 is the
  base tower exactly.

Default scope (`config/atlantic-v9-vision.yaml`): the merger fully (both linear layers, both biases,
the merger LayerNorm) plus `attn.qkv`, `attn.proj`, `mlp.linear_fc1` and `mlp.linear_fc2` of the last 9
of 27 blocks. Ranks 48 per block matrix, 128 per merger matrix, 16 per vector. From
`python -m es.vision plan`:

| scope | blocks | dim | touched parameters | pinned originals (BF16) | bases (FP32) |
| --- | --- | ---: | ---: | ---: | ---: |
| 25 percent | 20..26 (7) | 1,664 | 151,412,480 | 303 MB | 34 MB |
| **33 percent (default)** | **18..26 (9)** | **2,048** | **181,862,144** | **364 MB** | **41 MB** |
| 50 percent | 13..26 (14) | 3,008 | 257,986,304 | 516 MB | 58 MB |

Change it with `vision.space.blocks: {last: n}`, `{fraction: f}` or `{indices: [...]}` (exactly one).
Why this shape: the late blocks and the merger hold the language-facing, task-specific features;
the early blocks are generic edge and texture detectors that every task shares, where a
perturbation is most likely to break general image ability. The text vector is rank 16 x the number
of LoRA modules; if every language-model linear layer carries the v8 LoRA that is 496 x 16 = 7,936
(48 linear-attention layers x 8 plus 16 full-attention layers x 7; the loop logs the real number).
Vision would then be 2,048 of 9,984 coordinates (21 percent), close to the 25 percent image share.

**ES on z** (class `VisionES`), sharing everything with the text block except its own noise, sigma
and learning rate:

* seeds `derive_seed(run_seed, "vision-noise", gen, j)`, eps_j from PCG64 like the text noise, so
  enabling vision leaves the text noise stream (and every text candidate) unchanged;
* candidate j+ uses theta + sigma_t eps_t,j **and** z + sigma_v eps_v,j; candidate j- uses both minus
  signs. Both members of a pair share the same bases, and delta(-c) = -delta(c) exactly;
* one fitness per candidate, one set of centered-rank utilities u for the 48 perturbed candidates;
* coefficients c_j = (u_j+ - u_j-) / (2 N sigma_v), gradient g = sum_j c_j eps_v,j in float64;
* plain SGD with lr_v = 1.7 sigma_v^2 (Adam available), then Anchored Weight Decay toward the base
  tower: z = (1 - 0.005) (z + lr_v g). This is ES with a block-diagonal noise covariance.

**Exactness.** Every apply writes cast(W0 + delta) computed in FP64 from the pinned original, never
from the current weights, so applies cannot accumulate rounding; restore copies the pinned original
back, so it is exact to the bit, and applying c = 0 reproduces W0 bit for bit. The export computes
the same expression on the CPU (section 7).

## 3. Runtime: applying candidates inside vLLM

**Engine.** The overlay adds `worker_extension_cls: es.backends.vision_hooks.VisionWorkerExtension`,
`limit_mm_per_prompt: {image: 1, video: 0}`, `mm_processor_kwargs` (512 x 512 cap) and keeps
`fp8_base: false`. vLLM mixes the extension class into its Worker, so
`llm.collective_rpc("vision_apply", args=(coeff_bytes, tag, fingerprint))` runs inside the worker
next to the model (`self.get_model()`, or `self.model_runner.model` on older workers). Without the
extension, the controller sends the module-level functions instead (`llm.collective_rpc(rpc_apply,
args=...)`, called as `rpc_apply(worker, ...)`; the engine core inherits the driver's `sys.path`, so
`es` imports there). `llm.apply_model(fn)` reaches the model too but has nowhere to keep the pinned
originals between calls, so the hooks keep their state on the worker object.

**Attach** (once per session, right after the engine is built):

    from es.backends.vision_hooks import make_vision_controller
    vctl = make_vision_controller(backend, run_dir / "vision-space", cfg["vision"]["runtime"])
    vctl.attach()

Each worker loads the space, finds the 42 tensors by canonical name (any prefix before `visual.` and
any LoRA `.base_layer` segment are ignored), and refuses, changing nothing, unless every one has the
expected shape, a float dtype, contiguous storage and served bytes whose sha256 equals the
snapshot's. That rejects an FP8-quantized tower, a different base revision, a tensor-parallel shard
and a re-fused layout. It then pins the originals of only the touched tensors: on the GPU when free
memory exceeds three times the 364 MB needed, otherwise in page-locked host memory.

**Per candidate:**

    with vctl.candidate(c.vision):                  # RPC 1: W0 + delta(c) into the served storage
        replies = backend.generate([ref], prompts)  # image requests carry vctl.request_fields(shas)
                                                    # RPC 2 on exit, even after an exception: restore

Only 8 KB of FP32 coefficients and a tag cross the RPC. Writes use `Tensor.copy_` into the existing
storage, so CUDA graphs and a compiled encoder read the new values (keep vLLM's
`compile_mm_encoder` off anyway). Apply is about 25 GFLOP of FP64 matrix products plus about 1 GB of
memory traffic (milliseconds on an H100, estimated); restore is a 364 MB device copy, followed by an
equality audit of the touched tensors against the originals. Transient memory during apply peaks
near 0.6 GB (the merger fc2 in FP64) on top of the 364 MB originals and 82.5 MB of FP64 bases, all
within the 6.4 GB left by `gpu_memory_utilization: 0.92`; `compute_dtype: float32` halves the
transient, `originals: cpu` moves the 364 MB to host memory.

**Caches.** vLLM keys its encoder-output cache by the image's multimodal hash and its KV blocks by
token ids plus the hashes of the images inside them; neither key knows the tower's weights.

* `reset` (default when the engine has `reset_encoder_cache`, as vLLM main does on 2026-11-27): after
  every apply and restore, `llm.llm_engine.reset_encoder_cache()` and `llm.reset_prefix_cache()`.
  Image requests carry the image sha256 as `multi_modal_uuids`, so preprocessed pixels are reused.
  Each wave holds one candidate with its own LoRA and vLLM keys cached blocks by LoRA, so nothing
  useful survives a wave boundary anyway.
* `keyed` (older vLLM): the uuid names the state, `"<sha256>@<tag>"`, plus `cache_salt:
  "vision-<tag>"`, so no cache entry can be shared across tower states without any reset.

Text-only requests carry neither field: they never run the tower.

**Scheduling.** The tower is one global state per engine, so each candidate's 128 requests (text
and image) run in one engine call under its vision state: `backend.wave_size: 1`,
`vision.schedule.mode: per_candidate` (`es.vision.plan_waves`). That keeps one LoRA per call. The
cost against the text-only engine (one call of 6,272 requests at up to 256 concurrent sequences) is
49 calls of 128 sequences, each with a drain tail: roughly 15 to 40 percent more wall time per
generation, an estimate the pilot must measure. A faster path for later, not implemented: start
vLLM with `enable_mm_embeds=True`, have the worker encode each candidate's images under its delta,
and submit every candidate's image prompts in one call as `{"image_embeds": E, "image_grid_thw":
thw}`. That removes the per-wave barrier at the cost of moving about 4 GB of embeddings per
generation through the engine's IPC.

**Hugging Face and mock backends.** `HFVisionController(model, space)` patches an in-process model
the same way (the CPU tests prove it serves the same bytes as the vLLM path). The HF backend loads
`AutoModelForCausalLM` today, which has no tower; image items there need
`AutoModelForImageTextToText` and `AutoProcessor`. `RecordingVisionController` records the current
vision vector for mock-backend loop tests.

## 4. Integration points (to make in the text engine)

1. `es_loop.load_config`: honour `extends:` (or call `es.vision.load_vision_config`), add
   `"vision": es.vision.DEFAULT_VISION` to `DEFAULT_CONFIG`, and add `("vision", "runtime")` to
   `RESUME_MUTABLE`. The rest of `vision` (scope, ranks, seeds, sigma_v, lr, awd) is hashed by
   `config_hash`, so it is pinned for the run.
2. `ESRun.prepare`: when `vision.enabled`, load `run_dir/vision-space` if present, else
   `VisionSpace.from_snapshot(cfg["backend"]["base_model"], cfg["vision"]["space"]).save(...)`. On
   resume, check the checkpoint's `vision_space_fingerprint` and `vision_noise_fingerprint` like
   `svd_fingerprint` and `noise_fingerprint`, then `VisionES.load(<checkpoint>/vision.npz, vcfg["es"],
   run_seed, expect={"gen": state.gen, "space_fingerprint": fp})`; a new run starts
   `VisionES(space.dim, vcfg["es"], run_seed)`.
3. `CheckpointStore.save` and `_ckpt_meta`: write `vision.npz` (`VisionES.save`) inside the
   checkpoint's temporary directory before its atomic rename, so the text and vision state commit
   together; add `vision_space_fingerprint`, `vision_noise_fingerprint` and `vision_z_sha256` to
   the metadata.
4. `ensure_backend`: after `make_backend`, `self.vctl = make_vision_controller(...)` and
   `self._session_note({"vision": self.vctl.attach()})`. An attach refusal stops the session with
   its reason.
5. `_prepare_data` and `data.py`: load the image pool with `load_jsonl` (the terminal refusal
   applies), check every image file's sha256 against the manifest once at start, and sample 32
   image items per generation statelessly (per-epoch permutations seeded by
   `derive_seed(run_seed, "vision-batch", epoch)`), appended to the 96 text items.
   `batch_sha256` covers all 128. `fixed_batch` adds `dev_per_check` image dev items.
6. `Candidate` gets `vision: np.ndarray`; `candidates(gen)` pairs the text candidates with
   `self.vision_es.candidates(gen, n)` by name (`es.vision.candidate_names` gives the shared order).
7. `evaluate`: with vision on, require `wave_size == 1` and wrap
   `self.backend.generate(refs, prompts)` in `with self.vctl.candidate(c.vision):`. Add
   `vision_space_fingerprint` and the parent's `vision_z_sha256` to `partial_meta`, so saved partial
   scores are never reused under another vision state.
8. `backends/base.py`: `Prompt` gains `images: list of (path, sha256)` (empty for text). The prompt
   builder uses `vision_hooks.image_messages` for image items.
9. `backends/vllm_backend.py` `generate`: image prompts become
   `image_request(ids, [load_image_verified(p, sha) ...], self.vision_fields(shas))`, where the loop
   sets `vision_fields = vctl.request_fields`; text requests are unchanged. Cache decoded images by
   sha256 for the generation (all 49 candidates reuse them). `limit_mm_per_prompt` and
   `extra_kwargs` are already passed through.
10. `fitness.py` `Scorer`: route items with `modality: image` to the vision grader
    (`vision.fitness.grader`, same `grade(item, reply)` contract, reward in [-1, 1]); the item
    mean covers all items, so image rewards weigh by their 25 percent share. Record each candidate's
    `image_mean_reward` and image item count. The grader preflight (99 percent target acceptance)
    covers the image pool too.
11. `generation`: after `shape_fitness` and the text `es_update`, `vinfo = self.vision_es.update(gen,
    u_plus, u_minus)`; a `FloatingPointError` becomes `PausedRun` with nothing committed. Journal:
    `entry["vision"] = vinfo` plus the parent's and candidates' image-only mean reward.
12. `validate`: the parent's dev image items run under `vctl.candidate(z)`.
13. `_pause_checks` (S6): pause if `vctl.mismatches()` is not empty after a generation or if a worker
    raises during apply or restore.
14. `export`: next to the PEFT adapter, `es.vision.export_vision(space, z, base_model,
    out_dir / "vision", mode=vcfg["export"]["mode"])`; then `apply(z)`, compare
    `vctl.served_sha256()` with the export manifest for the touched tensors, `restore()`, and record
    the comparison in `ES_PROVENANCE.json`.
15. `replay`: `VisionES.replay(dim, vcfg["es"], seed, entries)` next to the theta replay (it checks
    every recorded z hash).
16. `calibrate`: a vision grid (sigma_v in 0.02, 0.05, 0.10, 0.20) with the text sigma at 0, scored
    on image items: the smallest sigma_v whose image-reply flip rate is at least 0.20 with nonzero
    antithetic differences.
17. `parity`: add G0-V and the efficacy check from section 6.
18. `plan`: replies per generation stay 6,272; report 49 engine calls per generation.
19. `PROTOCOL-atlantic-v9.md` needs an amendment before a vision run: it pre-registers 112 text pair
    items and a frozen tower. The amendment covers the batch (80 + 16 + 32), the vision block,
    sigma_v and its calibration, G0-V, the efficacy check and the vision export.

## 5. Image items (contract with the vision yardstick)

The vision yardstick (jsonl, code and manifests in the private `krummholz-training-data-internal`
repository under `yardstick/heldout-vision/`; image files in the private Hugging Face dataset
`krummholz-ai/yardstick-vision`) supplies `train_pool`, `dev` and a terminal split. The overlay's
paths are placeholders until that build's `FORMAT.md` exists; set them to its names. Fields this
integration needs per item: `id`, `split`, `modality: "image"`, `images: [{sha256, file}]`, `prompt`,
`answer_instruction`, `gold`, optional `group` and `variant` for pairs, and `target` in the training
pool. Every image is opened through `load_image_verified`, which refuses bytes that do not match
the manifest. The terminal split is never copied to the pod; the separate evaluation harness scores
it once for the finished candidate. As for text, all items, questions and targets come from code over
licensed datasets and public records, never from an LLM; the search learns only from the policy's
own replies scored by the deterministic graders.

## 6. Preflight on the pod (before generation 1)

    R=/workspace/es-runs/atlantic-v9-vision
    python -m es.vision plan  --config es/config/atlantic-v9-vision.yaml
    python -m es.vision build --config es/config/atlantic-v9-vision.yaml --out $R/vision-space

1. **Attach**: succeeds on the pod's vLLM (sha256 of all 42 served tensors equal the snapshot).
2. **G0-V parity**: 32 image dev items, greedy; replies with the hooks attached and `apply(0)` equal
   the replies of an engine without hooks; at least 31 of 32 identical (as G0 for text).
3. **Efficacy**: one large delta (sigma_v = 1.0 along one direction) changes at least half of the
   image replies, which proves the delta reaches the encoder and no cache serves stale embeddings;
   after `restore()` the replies equal the base replies again and `vctl.mismatches()` is empty.
   Run it under the cache policy the run will use.
4. **Calibration** of sigma_v (integration point 16).
5. **Export round trip**: `export_vision` at z = 0 writes tensors bitwise equal to the base; for a
   random z, the manifest's sha256 equal `served_sha256()` after `apply(z)`.

## 7. Export to an F16 mmproj GGUF (llama.cpp b12407 or newer)

On the pod (Linux shell), after the run:

    R=/workspace/es-runs/atlantic-v9-vision
    python -m es.vision export --space $R/vision-space --state $R/checkpoints/<latest>/vision.npz \
        --base-snapshot /workspace/models-base/Qwen3.8-27B@37561816 \
        --out $R/exports/atlantic-27b-v9-vision --mode mmproj
    git clone https://github.com/ggml-org/llama.cpp /workspace/llama.cpp
    cd /workspace/llama.cpp && git checkout b12407        # or any later release tag
    pip install -r requirements/requirements-convert_hf_to_gguf.txt
    python convert_hf_to_gguf.py $R/exports/atlantic-27b-v9-vision --mmproj --outtype f16 \
        --outfile $R/exports/mmproj-atlantic-27b-v9-vision-f16.gguf

`export --mode mmproj` writes `config.json`, the preprocessor configs and one `model.safetensors` with
all 333 `model.visual.*` tensors (the 42 searched ones evolved), plus `VISION_ES_EXPORT.json` (sha256
of every tensor, which changed, z and the space fingerprint) and `MMPROJ_COMMAND.txt`.
`--mode full` instead rewrites the shard that holds the vision tensors and symlinks the rest, for
loading the evolved model in transformers or vLLM.

What the converter does with this architecture: `architectures: ["Qwen3_5ForConditionalGeneration"]`
maps to `conversion/qwen3vl.py`, class `Qwen3VLVisionModel` (projector type `qwen3vl`, GELU,
spatial merge 2, image size 768 from sqrt(2304) x 16, mean and std from `preprocessor_config.json`,
projection width 5120 from `text_config.hidden_size`). It keeps only `model.visual.*`, splits the
Conv3d patch kernel into two Conv2d kernels (`v.patch_embd.weight`, `v.patch_embd.weight.1`), and
maps the merger to `mm.0`, `mm.2` and `v.post_ln`. `--outtype f16` stores the matrices in F16. A BF16
value converts to F16 exactly unless its magnitude is below 2^-14 (F16 subnormals keep fewer bits)
or above 65,504; the base mmproj goes through the same conversion, so comparisons are like for like.

Checks:

    # convert the untouched base snapshot the same way to mmproj-base-f16.gguf, then:
    python - <<'EOF'
    from gguf import GGUFReader
    a = {t.name: t.data.tobytes() for t in GGUFReader("mmproj-base-f16.gguf").tensors}
    b = {t.name: t.data.tobytes() for t in GGUFReader("mmproj-atlantic-27b-v9-vision-f16.gguf").tensors}
    assert a.keys() == b.keys()
    print(sorted(n for n in a if a[n] != b[n]))
    EOF

Only `mm.0.*`, `mm.2.*`, `v.post_ln.*` and `v.blk.18` to `v.blk.26` `attn_qkv`, `attn_out`, `ffn_up` and
`ffn_down` weights may differ. Then a smoke test with the text GGUF (the text export path is the text
engine's), on Hayven's Windows machine in PowerShell:

    cd D:\llama.cpp
    git checkout b12407
    py -3.12 -m pip install -r requirements\requirements-convert_hf_to_gguf.txt
    .\build\bin\Release\llama-mtmd-cli.exe -m D:\krummholz\atlantic-27b-v9-Q4_K_M.gguf `
        --mmproj D:\krummholz\mmproj-atlantic-27b-v9-vision-f16.gguf `
        --image D:\krummholz\check.png -p "Describe this image."

## 8. CPU tests

    python -m pytest es/tests/test_vision.py -q

(PowerShell: `.venv-es\Scripts\python -m pytest es\tests\test_vision.py -q`.) What they prove:

* the inventory equals the checkpoint header (333 tensors, 460,730,096 parameters, per-tensor shapes)
  and a meta-device transformers `Qwen3_5VisionModel` built from the real config; the default scope
  has 2,048 coordinates over blocks 18..26;
* bases are orthonormal and seeded, the space round-trips through disk, and a tampered basis fails
  its fingerprint;
* on a tiny random vision-like stack (fused qkv, proj, fc1, fc2, LayerNorms, 2 x 2 merger, Conv3d
  patch, position table) in BF16 and FP32: restore is exact to the bit after repeated applies, output
  after restore equals the original output exactly, untouched parameters never change, applies never
  accumulate, c = 0 is the identity, the served value equals the export value, and in FP64 the change
  equals s U diag(c) V^T computed independently in NumPy;
* antithetic symmetry: candidates are z +/- sigma_v eps with shared bases, the delta is exactly
  antisymmetric, and weight changes mirror;
* the update by hand (coefficients, gradient, lr = 1.7 sigma_v^2, AWD), equal utilities only decay,
  Adam and clipping, the vision seed stream differs from the text stream, `derive_seed` matches the
  text engine, and replay rebuilds z with every hash;
* an ES run through the patcher climbs, and when interrupted and resumed from disk (space reloaded,
  fresh module) reproduces the uninterrupted run's fitness, hashes and final z bitwise;
* the worker hooks against a fake vLLM (pickled RPC arguments, extension and callable paths, both
  cache policies and their resets, fingerprint refusal, restore on exceptions), and the HF controller
  serving the same bytes;
* fail-closed attach (other weights, an integer tensor, a reshaped tower, bad vectors) leaves the
  model untouched, the restore audit detects outside writes, PEFT and LoRA-wrapped names resolve;
* a real tiny transformers `Qwen3_5VisionModel`: c = 0 and restore give identical outputs;
* both export modes write the served bytes, keep untouched tensors and the language model shard, and
  refuse a snapshot that differs from the space's base;
* the overlay loads over `atlantic-v9.yaml` with a 25 percent image share and no em dashes.

## 9. Risks and open items

* **Signal per coordinate.** One fitness covers all items, and the vision block cannot affect the 75
  percent that are text, so its gradient estimate is noisier than the text block's. The journal's
  image-only rewards show whether it moves; a vision-only fitness would be a different estimator and
  is not used.
* **Wall time.** Per-candidate waves cost an estimated 15 to 40 percent more per generation (not
  measured). G5 still applies; fallbacks are `es.pairs: 16`, a lower `max_tokens` for image items, or
  the embeddings path.
* **vLLM API drift.** `worker_extension_cls`, `collective_rpc`, `reset_encoder_cache`,
  `reset_prefix_cache`, `multi_modal_uuids` and `cache_salt` were checked in vLLM's main branch
  source on 2026-11-27, not in the pod's installed version. The hooks are tested against a fake
  worker only; attach and the preflight are the first real contact.
* **FP8 base.** An FP8 tower cannot take in-place BF16 deltas; `fp8_base` must stay false (attach
  refuses otherwise). Do not use vLLM sleep mode, which discards weights.
* **Memory headroom.** About 1 GB is allocated after engine start. If it does not fit, use
  `originals: cpu` and `compute_dtype: float32`.
* **Kernel rounding.** FP64 matrix products on the GPU and on the CPU can differ in the last bit, so
  a rare BF16 element may round differently between the served and the exported weights. The export
  check compares them; the export is the reference. Resume on the same GPU type.
* **Image retention.** The retention slice has no image items, so general image ability outside the
  vision yardstick is unguarded. An image retention slice is an open item for the vision yardstick.
* **Data paths and licences.** The overlay's vision yardstick paths are placeholders, and each
  dataset's README governs its use (counsel review for share-alike or GPL sources, as for text).
* **Protocol.** The run needs the protocol amendment in integration point 19 before generation 1.
