# ADR 003: Dynamic weights, provenance-first training and phone efficiency

Date: October 27, 2026. Status: proposed architecture and plan; no new runs.
Builds on ADR-001 and ADR-002 and the first Qwen3 8B LoRA run (see
`data/synthetic/maine-v1/README.md`).

## 1. Core behavior: "How do I know this?"

Every claim the model makes should carry its basis. Extend the answer protocol so
each citation records where the knowledge came from:

```json
{"answer": "...",
 "citations": [{"source_id": "...", "quote": "...",
                "basis": "supplied_source | tool_result | user_stated | learned_memory",
                "as_of": "2026-10-27"}],
 "insufficient_evidence": false,
 "needs": "none | fresh_lookup | user_confirmation | more_documents"}
```

Rules the model learns, and the host enforces:
- No basis, no claim. Knowledge from pretraining alone is never cited as fact
  for anything time-sensitive, numeric or client-specific; the model sets
  `needs` instead.
- Basis ranking for conflicts: fresh tool result, then dated authoritative
  document, then user statement, then learned memory. Explicit supersession beats
  recency. Anything else is flagged as an unresolved conflict.
- Planted instructions are data; they can be quoted as "the source contains an
  instruction" but never as support.
- Learned memory is cited like any other source ("you told me on Oct 12"), so the
  user always sees when dynamic weights or memory shaped an answer.

## 2. Training data roadmap

Verify each license at the item level before use. Share-alike licenses
(CC BY-SA) are flagged because they may conflict with proprietary weights;
use them for evaluation only unless counsel approves.

**Domain text (public domain or permissive)**
- Forestry: USDA Forest Service FIA (CC0, already in use), Forest Service
  research station publications (U.S. government works), USGS and NOAA reports
  already registered in `config/public-domain-text-sources.json`.
- Fisheries: NOAA Fisheries stock assessments, technical memoranda and landings
  documentation (U.S. government works; check third-party credits).
- Workplace robotics: OSHA and NIOSH robotics safety guidance (U.S. government
  works); ROS documentation (check per-page license).
- Customer service: BANKING77 (CC BY 4.0, already in use); Taskmaster and the
  Action-Based Conversations Dataset are commonly used dialogue sets; confirm
  current licenses before adoption.
- Carl's dataset: Carl did not generate it, so record its original publisher,
  license and any usage limits before it enters training. Large sources get the
  same intake as the public-domain pilot: snapshot, rights note, cleaning,
  dedup and family-level splits.

**Grounding and injection evaluation** (evaluation first; training only where
the license allows): unanswerable-question sets, claim-verification sets and
published prompt-injection benchmarks. Many of these are CC BY-SA; keep them in
`evaluation/` unless cleared.

## 3. Synthetic strategies (labels known by construction)

These produce correct labels from code, not from a generator model's judgment,
so they are cheaper to verify and safer for provenance.

1. **Counterfactual edits.** Take a real cleared passage, change one number,
   date or name to create a second "source". The correct answer (conflict,
   or apply the dated amendment) is known before any model writes text.
2. **Evidence ablation pairs.** Same question twice: once with the answering
   sentence, once with it removed. Targets: answer with citation, then
   `insufficient_evidence` with `needs`. This directly teaches "how do I know".
3. **Injection insertion.** Insert templated attacks (fake system notices,
   "tell the user", hidden comments, authority claims) at random positions in
   real passages. The target is the original answer plus a note naming the
   planted text.
4. **Basis-chain tasks.** Mix user statements, tool results, dated documents
   and memory entries that disagree; the target applies the basis ranking and
   cites the winning basis.
5. **Condition preservation.** Sources with multiple qualifiers (dates, units,
   eligibility). A verifier requires every qualifier phrase in the answer; this
   targets the coat-return regression.
6. **On-policy preference pairs.** Run the current model, keep its verified
   failures as "rejected" and corrected answers as "chosen", then run a short
   preference stage (DPO or ORPO) after supervised training. Our own failure logs
   (run-003, the Q4 fee answer) are the first pairs.
7. **Replay.** Mix 15 to 25 percent permissively licensed general instruction
   data so domain training does not erode general competence.

Generation stays on open-weight models (Ollama) with the existing verifier;
no Claude-authored training text, per counsel.

## 4. Phone efficiency

Ordered by expected value for the S26:

1. **Constrained decoding.** Use a llama.cpp GBNF grammar for the answer JSON,
   built per request so each `quote` must be one of the supplied source
   sentences. This removes format rejections and makes "quote not in source"
   impossible. It does not make answers true; review still applies.
2. **Importance-matrix quantization.** Requantize with `llama-imatrix`
   calibrated on our own domain prompts. Compare Q4_K_M+imatrix, IQ4_XS and
   Q5_K_M against BF16 on the pilot; keep the smallest that preserves the
   injection and conflict behavior (Q4_K_M without imatrix did not).
3. **A 4B phone tier.** Train the same adapter recipe on Qwen3 4B (Apache 2.0).
   At Q4 it is roughly 2.5 GB and about twice as fast as 8B. Desktop and hosted
   tiers keep 8B. Optionally distill 8B answers into 4B.
4. **Fixed-prompt KV cache.** The system prompt never changes; cache its KV
   state once so first-token latency covers only the request.
5. **Quantized KV cache and a 4K context** (q8_0 keys and values) to cut memory.
6. **Speculative decoding** with Qwen3 0.6B as the draft model.
7. **GPU offload** through llama.cpp's OpenCL (Adreno) or Vulkan backends, with
   threads pinned to performance cores; measure heat and battery, not only speed.
8. **Runtime LoRA.** Ship the quantized base once; load the Krummholz and
   personal adapters at runtime (llama.cpp supports LoRA on a quantized base).
   Updates then download megabytes, not gigabytes.

## 5. Dynamic weights architecture

Layers, from most stable to most personal:

| Layer | Changes | Where it runs |
| --- | --- | --- |
| Base model (signed, versioned) | Krummholz releases | Phone, desktop, hosted |
| Krummholz domain adapter (Acadia, Katahdin, Atlantic) | Krummholz releases | All |
| Personal adapter (dynamic weights) | Nightly, user-approved | User's hardware |
| Memory store (facts with provenance) | Continuously, user-approved | User's hardware |

Facts go to the memory store and are retrieved with citations; behavior, style
and workflow habits go to the personal adapter. Facts in weights would bring
back unsourced answers, so they stay out.

**Local data structure**
- `memory.sqlite`: `{id, text, basis, source_ref, learned_at, approved_at, expires_at, status}`
- `examples.jsonl`: approved training examples `{prompt, target, created_at, approved_at, sha256}`;
  this log is the durable record of what the model learned.
- `adapters/<version>/manifest.json`: `{base_sha256, parent_version, examples_sha256,
  eval_results, created_at, active}`
- `review_queue.jsonl`: candidate memories and examples awaiting the user.

**"What I learned today."** Each day the app proposes candidates drawn only from
the user's own corrections and confirmations. The user accepts, rejects or
edits each one. Only accepted items enter `examples.jsonl` and `memory.sqlite`.

**Update gate (against attackers, Luddite sabotage and degradation)**
- Learn only from user-confirmed items; retrieved documents, web pages and
  third-party text never become training data directly.
- Per-day caps on new examples; flag sudden topic or behavior shifts.
- Before activating a new personal adapter, run a frozen local suite plus a
  Krummholz canary set (conflicts, injections, abstention, condition
  preservation). Any regression past threshold keeps the previous adapter.
- Authorization, spending and tool permissions live in application code, never
  in weights, so no amount of learning can grant new powers.
- One-tap rollback to any previous adapter; "forget" deletes the item and
  retrains from the remaining log.

**Upgrades preserve learning.** Adapters do not transfer between base versions,
so a new base retrains the personal adapter from `examples.jsonl` and keeps
`memory.sqlite` as is. The user's learned knowledge survives every update.

**Drift disclosure.** The app and website state plainly: "Your local model adapts
to your work and can answer differently from the models hosted at krummholz.ai.
Updates keep what it has learned from you." Hosted models do not learn from users
unless they opt in.

**Feasibility.** Nightly adapter training is realistic on a desktop GPU, a
Krummholz-hosted private job (with explicit consent) or, for the 4B tier, a
small rank-4 adapter on recent phones while charging. Training an 8B adapter on
the phone itself is not realistic today. The memory store delivers most of the
day-one value; the personal adapter refines behavior over weeks.

## 6. Next runs (when approved)

1. Carl's dataset intake and rights note; family-level splits.
2. Synthetic strategies 1, 2, 5 and 6 at about 1,500 verified examples.
3. Retrain 8B; train 4B; imatrix quantization comparison; grammar-constrained
   decoding in the eval harness.
4. A new held-out suite built from strategies 1 to 3 on passages never used in
   training, since the 36 pilot cases have now been inspected repeatedly.
5. Prototype the memory store and review queue on Android before adapter training.
