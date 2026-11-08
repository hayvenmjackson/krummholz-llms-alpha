# Run plan v7: first dynamic weights layers, and the gaps v6 left

Date: November 8, 2026. Draft for Hayven's review. No GPU time is spent on training until Hayven approves
this plan and sets the budget.

Sources: the November 3 decision to keep v6 on accuracy and make dynamic weights v7 (run-plan-v6.md, open
question 2), ADR-004 (signed core plus product adapters), the v6 results and the "What's next" list in the
v6 beta release post (November 4).

## Decisions carried over

- **Atlantic** = the 27B dense, **Atlantic Lite** = the 4B, both general purpose. **Acadia** (customer
  service) and **Katahdin** (forestry and fisheries) come from the 35B mixture-of-experts core.
- GPUs: H100 and H200 only. No A40, A6000 or RTX PRO 6000 pods.
- Identity: config/identity-v3.json (approved November 4): no "assistant" wording outside Acadia and
  Katahdin. tools/vl/identity.py still defaults to identity-v1, so every category passes
  `--spec config/identity-v3.json`.
- Consciousness: never claim feelings or consciousness as fact; never flatly rule them out.
- No Claude-generated training text; the teacher stays fully open (allenai/Olmo-3-7B-Instruct or newer).

## What v7 does

### 0. Before any training: the multilingual check (exploratory, about $3)

evaluation/multilingual-v1.json asks 12 questions in English, Spanish, Hebrew, French and Mandarin
of each published Q4_K_M (tools/vl/launch_multilingual_v6.sh). The runtime guards run only in English, so
this shows what the raw models do in the other four languages. Its results can add items below.

### 1. Acadia and Katahdin as product adapters on the 35B core (ADR-004 L2)

- Base: the v6.3 35B core, frozen. Each product is a LoRA adapter trained on its own data; nothing is
  merged into the core, so the published core stays the same file and each adapter is a small download.
- Acadia: customer-service data (BANKING77 and CLINC150 from the dataset catalog, policy-grounded answers,
  escalation to people). Katahdin: forestry and fisheries data (USDA FIA Maine records, the synthetic
  forestry rows, the Maine evidence tasks, NOAA MRIP fishing data) and the point-cloud tool protocol.
- Each adapter records a manifest (data hashes, seed, trainer version) as ADR-004 section 5 describes.
- Gate: an adapter ships only if, loaded on the core, every v6 suite scores at least what the core scores
  alone (decensor, sovereignty, identity, human, consciousness, safety), plus its own task suite.
- Open for Hayven: ADR-004 was to be written up for an attorney before dynamic weights ship. Training the
  adapters does not publish anything; releasing them would.

### 2. Consciousness and the human check in the weights

From the weights alone v6 scored 3, 5 and 7 of 12 on consciousness (gate 11) and 10, 11 and 10 of 12 on
never claiming to be human. v7 trains on identity-v3 data with the stance rules from the start (the v6
bug that needed three correction passes is fixed), and new held-out probe sets are written before
training so the v6 suites are not reused for selection.

### 3. Atlantic Lite's territorial facts

7 of 42 territorial answers were false. More sovereignty preference pairs built from its own wrong
answers, weighted toward Taiwan's status and recognition counts.

### 4. Small fixes

- tools/vl/calibrate.py SYSTEM starts "You are a Krummholz assistant."; change it to the identity-v3
  wording (pass4 direct_eval keeps the old prompt to measure v6).
- Rescore every published GGUF (Q4_K_M and Q5_K_M), not only the unquantized merge.
- Put Atlantic Lite through the phone acceptance tests.

## Not in v7

- Personal adapters (ADR-004 L4, on-device nightly learning): needs the app side first.
- The copyleft legal review stays open and blocks the full release, not the training.

## Budget and order

Hayven sets the budget. Order: multilingual check, then Acadia and Katahdin adapters (one H100 or H200
each, in parallel), then the 4B, then the 27B. Each job writes its data, adapters and results to the
krummholz-models volume as it goes and stops its pod when done. Estimates are written in once the
multilingual results are in and the data sizes are known.
