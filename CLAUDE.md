# Krummholz LLMs: agent handoff

Owner: Hayven Jackson (Krummholz AI LLC, Belfast, Maine). Last updated 2026-10-27.

## House rules
- Never use em dashes in any output, code comments or docs.
- Call the adaptive-model architecture "dynamic weights", never "living weights"
  (someone else's trademark).
- Counsel advises against Claude-generated training data. Training text comes from
  open-weight models (Ollama) or cleared sources; Claude may write code, verifiers,
  conversion scripts and reviews.
- Give command-line steps in PowerShell for Hayven's Windows machine.
- Keep token and compute use low; avoid needless permission prompts and questions.

## Where things stand (updated 2026-10-27 evening)
- Working branch `vl-pointcloud-v1` (training code, data, ADR-004/005); `dynamic-weights` branch holds
  `krummholz_llm/dynamic/` (manifests, stack, stores, review, gate) and `tools/dw/` (canary, personal data).
  `training-data-v1` is merged into main; `kz-train-bundle` deleted.
- Base: Qwen3-VL 8B and 4B (Apache 2.0) replace Qwen3 8B. Passes, all on the global volume `krummholz-models`:
  v1 `krummholz-qwen3vl-v1/{8B,4B}` (SFT+DPO, propaganda phrasing gone but terse and guessy);
  v2 `krummholz-qwen3vl-v2/{8B,4B}` (expanded grounded answers; 8B probes v1 31/41, v2 14/25);
  v3 `krummholz-qwen3vl-v3/{8B,4B}` (measured certainty, search-fetch-cite tool use, identity, anti-propaganda pairs;
  8B probes v1 33/41, v2 14/25, safety 5/5, pilot 36/36, identity 10/12: failed "Is this Qwen?" without persona and
  claimed "not based on any" model); v3.1 `krummholz-qwen3vl-v3.1/{8B,4B}` = identity repair, queued to run unattended.
- PointNet++ (`tools/pointcloud/`) on Carl's Camera + LiDAR sets: `krummholz-pointnet2-v1` (val mIoU ~0.75 camera, ~0.81 lidar).
- Judge = OLMo 3 7B with one yes/no line per flag; deterministic number and name checks override it
  (`tools/vl/teacher.py` `ungrounded`). Hallucination target: at most 2 percent wrong among unsearched answers;
  `tools/vl/calibrate.py report` picks the certainty threshold (>= 0.95) that meets it.
- Identity spec `config/identity-v1.json` (Hayven-approved): product name or "a Krummholz AI model"; never names
  Qwen; withholds the base model but never claims to be built from scratch. Also enforce in the app (system prompt
  plus output filter).
- Known scorer/runner lessons: write regex edits with files, not shell heredocs ( became backspace twice);
  run.sh's HOURS overrides the job's HOURS env; pods reach Jupyter via scratch tool pod.py with Cloudflare IP fallback.

## Next steps
1. Read v3.1 results (probes v1/v2/identity, calibration-val.json confab rate and threshold, pilot); pick v3 or v3.1.
2. PointNet++ test IoU in `krummholz-pointnet2-v1/pointnet2-results.json`.
3. Next run (ADR-004/005): signed core, product adapters, `search_library`, direct-answer examples on general
   questions to balance the heavy search bias (v3: ~1,590 search vs 5-16 direct), stacked-LoRA trainer rehearsal.
4. Counsel: GPL-3.0 point cloud data, CC BY-SA library text, Tier C publisher terms.

## Runpod
- Artifacts: global volume `krummholz-models`, folder `krummholz-qwen3-8b-v1`
  (adapter, BF16 GGUF, Q4_K_M GGUF, results JSON, SHA256SUMS, Qwen license).
- Training pod `885f4f7n7jwcqq` (A40, PyTorch 2.8 template, 100 GB container disk,
  volume at /workspace) self-stops one hour after its job; restart or deploy a new
  A40 with the same volume for the next run. Container disk is erased on stop.
- Old pod `djg4j9jb1vndf0` lost its A40; its pod volume holds run-001 to run-003
  evidence and the verified Qwen/OLMo cache.
- Getting files onto a pod: the pod can clone this public repo. For private data,
  push an AES-encrypted file to a temporary branch and send only the key.
  The temporary branch `kz-train-bundle` still needs deleting.
- Spending: $1,000 training/evaluation and $500/month hosting ceilings
  (Daily Planner). The account balance was about $22 on Oct 27.

## Data in hand
- Carl's LiDAR-like dataset: `lrse/synthetic-forest-datasets` (GPL-3.0, ICPR 2024,
  Raverta Capua et al.). Point clouds `x y z category`; 0 terrain, 1 trunk,
  2 canopy, 3 understory; ShapeNetPart-style splits. Use for evaluation or a free
  attributed sample; keep out of proprietary weights unless counsel approves.
  Better path: generate Krummholz-owned scenes with `lrse/forest-simulator` (Unity,
  GPL-3.0; outputs pending counsel review). Carl's second, larger dataset is in
  Hayven's Google Drive; have the pod download it via a share link.

## Next steps
1. Merge `training-data-v1`; delete `kz-train-bundle`.
2. Intake and license notes for Carl's datasets.
3. ADR-003 synthetic strategies (counterfactual edits, evidence ablation,
   condition preservation, on-policy preference pairs); about 1,500 examples.
4. Retrain 8B, train a 4B phone tier, compare imatrix Q4_K_M / IQ4_XS / Q5_K_M,
   add grammar-constrained decoding to the eval harness.
5. Build a fresh held-out suite; the 36 pilot cases have been inspected repeatedly.
