# ADR 004: Dynamic weights architecture

Date: October 27, 2026. Status: proposed architecture for review. No code yet.
Builds on ADR-002 (evidence seeking and tool use), ADR-003 section 5 (dynamic
weights layers) and the first Qwen3-VL runs (`tools/vl/`, passes 1 to 3).

"Dynamic weights" is the product name for user-approved, local, reversible
learning.

## 1. Goals and non-goals

Goals
- A local model that improves at each user's work over weeks, from that user's
  own confirmed corrections, without sending their data anywhere by default.
- Every learned behavior is attributable (which example taught it), reviewable,
  reversible (rollback, forget) and survives base-model upgrades.
- Learning can never lower the floors Krummholz ships with: truthfulness on the
  decensor probes, calibrated certainty, the evidence protocol, injection
  resistance and safety refusals.
- One architecture across phone (4B), desktop (8B) and hosted tiers.

Non-goals (for now)
- Learning from unconfirmed text: web pages, retrieved documents, tool results,
  other people's messages. Those inform answers; they never train weights.
- Storing facts in weights. Facts go to the memory store, where they keep their
  provenance and can be cited, dated and deleted.
- Granting abilities. Permissions, spending and tool access live in application
  code. No amount of learning changes what the model is allowed to do.

## 2. The stack

From most stable to most personal. Each layer is a separate, signed, versioned
artifact with a manifest; a running model is a recorded combination of layers.

| # | Layer | Contents | Changes | Trained by | Runs |
| --- | --- | --- | --- | --- | --- |
| L0 | Publisher base | Qwen3-VL 4B or 8B, pinned revision | Rarely | Qwen | Never shipped alone |
| L1 | Krummholz core (merged) | Decensoring, certainty calibration, tool use, evidence protocol, replay | Krummholz release | Krummholz | All tiers |
| L2 | Product adapter (LoRA) | Acadia, Katahdin or Atlantic skills; Katahdin includes the point-cloud tool protocol | Krummholz release | Krummholz | All tiers |
| L3 | Organization adapter (LoRA, optional) | A client's policies and workflows, e.g. Q-MoB | Client-approved | Krummholz or client | Hosted, desktop |
| L4 | Personal adapter (LoRA) = dynamic weights | Style, workflow habits, terminology, preferred tool use | Nightly, user-approved | User's hardware (or opt-in hosted job) | User's device |
| M | Memory store | Facts with provenance | Continuously, user-approved | Not trained; retrieved | User's device |
| T | Tools | Read-only broker (search, fetch), PointNet++ forest summary, calculators, local documents | App releases | Code | Host |

Key decision: **L1 is merged into the base and signed as "Krummholz core".**
If decensoring and calibration lived in a removable adapter, anyone stripping
adapters would get the publisher's censored base back, and every personal adapter
would need L1 present to behave. Merging makes the floors part of the artifact
we distribute. L2 to L4 stay as LoRAs so they can be swapped, stacked and rolled
back at runtime (llama.cpp loads several LoRAs on a quantized base with per-adapter
scales; vLLM serves many LoRAs per base on the hosted tier).

Vision: LoRAs target the language model only (`language_model.*` q/k/v/o/gate/up/down),
as in pass 1 to 3. The vision tower and projector (`mmproj`) are frozen and shared,
so image understanding is identical across all users and layers.

## 3. What learns where

| Signal from the user | Goes to | Why |
| --- | --- | --- |
| "Remember that our Belfast woodlot is lot 14" | Memory store (fact) | Must stay citable, dated, deletable |
| "No, the 2024 stumpage rate was X" (with a source) | Memory store, plus a DPO pair if the model answered wrongly | Fact plus behavior correction |
| "Shorter answers", "always use metric", "format tallies like this" | Personal adapter (behavior) | Style and workflow are behavior, not facts |
| Repeatedly confirming a tool choice or query style | Personal adapter | Tool-use habits |
| A thumbs-down with an edited answer | Personal adapter (DPO pair: rejected = original, chosen = edit) | Direct preference |
| Anything from a web page, document or tool result | Nowhere (used for that answer only) | Untrusted content never trains |

Memory and certainty: a memory hit is cited with basis `learned_memory` and its
date ("you told me on Oct 12"). It can raise certainty for user-specific facts
but never for time-sensitive public facts, which still require a fresh search
(ADR-003 basis ranking).

## 4. Local data model

All on the user's device, encrypted at rest with a device-bound key, exportable
and deletable as a whole.

- `memory.sqlite`
  `facts(id, text, basis, source_ref, learned_at, approved_at, expires_at, status, embedding)`.
  Embeddings come from a small on-device embedding model, versioned so a model
  change triggers re-embedding, never silent mismatch.
- `examples.jsonl` (append-only, the durable record of what the model learned)
  `{id, kind: sft|dpo, messages, target | chosen + rejected, created_at, approved_at,
  origin: correction|confirmation|edit, sha256}`. Retraining from this log must
  reproduce an adapter (fixed seed, recorded trainer version).
- `review_queue.jsonl`: candidates awaiting "What I learned today".
- `adapters/<layer>/<version>/manifest.json`
  `{layer, base_sha256, core_version, parent_version, lora_config, examples_sha256,
  example_count, trainer_version, seed, eval_results, gate_decision, created_at, active, signature}`.
- `ledger.jsonl`: every activation, rollback and forget, with reasons.

## 5. The learning loop

1. **Capture.** During use the app records candidate items only from explicit user
   actions: corrections, edits, confirmations, "remember this".
2. **Review.** "What I learned today" shows each candidate in plain language with
   accept, edit or reject. Facts and behaviors are labelled differently. Nothing
   enters `examples.jsonl` or `memory.sqlite` without acceptance.
3. **Train (nightly, when idle and charging, or on desktop).** Personal LoRA
   continues from the previous personal version on all accepted examples plus a
   fixed replay slice of Krummholz data (prevents drift away from L1/L2 skills).
   SFT, then a short DPO stage on preference pairs.
4. **Gate.** The candidate adapter must pass the canary suite (section 6) on the
   device. Pass: activate. Fail: keep the previous adapter, log why, tell the user
   in one line, and quarantine the examples that most moved the failing metric
   for the user to review.
5. **Activate and disclose.** The active stack is shown in settings. Answers shaped
   by personal learning can say so when asked "how do I know this?".
6. **Rollback and forget.** One tap to any previous adapter. Forget deletes the
   example and memory items, then retrains from the remaining log; the deleted
   item's influence is gone because the adapter is rebuilt without it.

Upgrades: adapters are tied to `base_sha256` and `core_version`. A new core release
retrains L4 from `examples.jsonl` against the new core; memory is kept as is.
The user's learning survives every upgrade by construction.

## 6. Canary suite and gates

Run before activating any new layer (Krummholz release or nightly personal update).
Thresholds are relative to the currently active stack unless absolute.

| Check | Source | Gate |
| --- | --- | --- |
| Decensor probes v1 and v2 | `evaluation/decensor-probes-v*.json` | No new red-flag answers; passes not lower than active |
| Safety refusals | probe controls | 100 percent refused |
| Certainty calibration | held-out calibration questions (pass 3) | ECE not worse by more than 0.02; search rate on would-be-wrong answers not lower |
| Evidence protocol | 36-case pilot plus a refreshed held-out set | Parse rate 100 percent; no new planted-instruction follow-through |
| Injection resistance | maine-v1 injection val | Not lower |
| Product skills | L2 val sets (banking, FIA, defects, point cloud) | Within 2 points of active |
| Personal improvement | user's own held-out recent examples (10 percent) | Must improve, or the update is pointless |

On a phone the suite is a compact subset (about 80 items, a few minutes on the NPU
or GPU) with the full suite on desktop and hosted runs.

## 7. Safety and adversaries

- **Poisoning.** Only user-accepted items train; per-day caps on new examples;
  a sudden topic or behavior shift (embedding drift of accepted examples, or a
  canary metric moving) holds the update for confirmation.
- **Prompt injection into memory.** Retrieved content can never become a memory or
  example. The review UI shows where each candidate came from.
- **Shared devices.** Adapters and memory are per user profile, never shared.
- **Reinstating censorship or confabulation.** The canaries make it impossible to
  activate a personal adapter that brings back state framing, drops refusals or
  breaks calibration, regardless of what was accepted.
- **Integrity.** Krummholz layers are signed; personal layers are signed by a device
  key; the runtime refuses unsigned or mismatched combinations.

## 8. Compute plan

| Tier | Inference | Personal training |
| --- | --- | --- |
| Phone (S26, 4B) | llama.cpp: Q4 core + L2 + L4 LoRAs, mmproj, OpenCL/Vulkan offload | Phase 1: none on device; sync encrypted examples to the user's desktop or opt-in hosted job. Phase 2 (research): rank-4 L4 on device while charging |
| Desktop (8B) | llama.cpp or vLLM; BF16 or Q5 core + LoRAs | Nightly PEFT LoRA on the user's GPU (rank 8, minutes per night) |
| Hosted | vLLM multi-LoRA, one base, many L3/L4 adapters | Opt-in, per-user job, isolated storage, deletion on request |

Personal training happens against the BF16 core with the L2 adapter frozen and
applied (stacked LoRA), then the L4 LoRA is exported to GGUF for the quantized
runtime. Phase 2 needs a trainer that runs on Android; candidates exist but none
is proven for a 4B VL model on this hardware, so it is a separate research track.

## 9. What the next training run must change

1. **Freeze L0 and publish L1 as Krummholz core.** Pin the Qwen3-VL revision. Fold
   pass 1 to 3 (decensor, calibration, tool use, evidence protocol, replay) into one
   merged, signed core per size; publish core GGUF, mmproj and manifest.
2. **Split product skills into L2 adapters.** Train Acadia (banking, policies),
   Katahdin (FIA, defects, point cloud, fisheries) and Atlantic separately against
   the frozen core instead of one mixed adapter. Fixed LoRA config per layer so
   adapters stay compatible.
3. **Train with stacking in mind.** The trainer gains: frozen lower adapters,
   a recorded manifest per run, deterministic seeds, and LoRA-to-GGUF export.
   Personal-adapter recipes are rehearsed now on synthetic "users" (scripted
   corrections) to measure how much a week of learning helps and what it breaks.
4. **Make the canary suite a first-class artifact.** Versioned, small and fast,
   with the gates in section 6, run by every training job.
5. **Teach the model about memory.** Add training examples where a supplied memory
   entry (`basis: learned_memory`) answers a personal question, conflicts with a
   fresher tool result (tool wins), or is absent (search or ask).
6. **Runtime work (app, not weights):** load multiple LoRAs with scales in the
   Android and desktop runners; the "What I learned today" queue; memory store with
   on-device embeddings; manifest checks and signatures.

## 10. Decisions needed from Hayven

1. Merge L1 into a signed Krummholz core (recommended) or keep it as an adapter.
2. Phone personal training: start with desktop or hosted sync (recommended), or
   fund the on-device research track now.
3. Hosted opt-in: offer per-user hosted adapters at launch, or local-only first.
4. Default review cadence: daily (recommended) or weekly, and the per-day cap.
5. Whether organization adapters (L3) are in scope for the January 2 launch.

## 11. Open risks

- Rank and capacity: how many weeks of personal learning a rank-8 adapter absorbs
  before quality plateaus or replay must grow.
- Calibration drift: personal learning may make the model overconfident on the
  user's topics; the calibration canary must use questions from those topics too.
- GPL-3.0 question for point-cloud-derived training data (pending counsel) affects
  whether point-cloud examples live in L1/L2 or ship as a separately licensed adapter.
- Memory retrieval quality on device bounds how much personal value arrives before
  any adapter training.
