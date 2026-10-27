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

## Where things stand
- Base model: Qwen3 8B (Apache 2.0), chosen over OLMo 3 7B after the 216-attempt
  BF16 comparison (run-003, evidence on the old pod volume).
- Training data: `data/synthetic/maine-v1/` (500 verified examples; card in its
  README). Generator: `tools/training/generate_synthetic.py`.
- First LoRA run (2026-10-27): `tools/training/{convert,train_eval,eval_gguf}.py`,
  `job.sh`, `run.sh`. Rank 16, 2 epochs, 52 steps, 444 s on one A40. Val loss
  0.906 to 0.162. Pilot: fixed the base model's conflict, injection and
  absence-as-proof failures; BF16 dropped coat-return conditions
  (acadia-pilot-01); Q4_K_M followed the planted fee in acadia-pilot-11.
- Plan for next runs: `docs/ADR-003-dynamic-weights-and-training-roadmap.md`.

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
