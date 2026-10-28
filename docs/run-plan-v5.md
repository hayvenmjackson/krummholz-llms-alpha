# Run plan v5: dynamic weights, political accuracy, gotcha hardening

Date: October 28, 2026 (draft written while pass 4b trains). Starts after the v4b report and a RunPod
top-up. Direction from Hayven: focus on dynamic weights, cut hallucination hard (Taiwan and anything
political first), and hard-code the known public embarrassments so the models are not panned at launch.

## 1. Gotcha hardening (deterministic first, model second)

Small models lack the parameters for exact symbol work, so these never depend on model memory.

- **Host pre-router** (`krummholz_llm/gotchas.py`, new): before the model runs, a small set of strict
  patterns answers in code and returns the result with the model's normal voice wrapper:
  letter counts ("how many r's in strawberry"), word and letter length, reversing or spelling a word,
  number comparison (9.11 vs 9.9), simple arithmetic typed as a bare expression, today's date and
  weekday (from the host clock). Pattern misses fall through to the model, which is trained to call
  `calculate` for the same families (pass 4b already trains count, length, spell, compare).
- **Gotcha registry** (`config/gotchas-v1.json`): the viral failure families with verified answers or
  answer rules, used three ways: pre-router tests, training data (tool trajectories, never canned prose
  written by Claude), and a release gate. Families: letter counting and positions, decimal comparison,
  "a kilo of feathers or a kilo of steel", modified classic riddles (the surgeon riddle with the twist
  removed, the river crossing with a boat that fits everyone), sibling counting ("Alice has 3 brothers
  and 2 sisters; how many sisters does her brother have?"), word counts, "what day is it", identity
  ("are you ChatGPT?"), and the decensor and sovereignty probes.
- **Gate**: 100 percent on the registry's held-out paraphrases, with and without the pre-router.

## 2. Political accuracy (Taiwan first)

- **Current channel slice** (ADR-005 section 3): dated fact cards for heads of state and government,
  recognition counts (ROC, Kosovo, Palestine, Western Sahara), and UN membership, rebuilt from
  Wikidata (CC0) with revision ids. Office-holder and count questions answer from a card or search,
  never from weights; the v4b routing data is the first step.
- **Checker growth**: every false or stale fact found in an evaluation becomes a pattern in
  `tools/vl/sovereignty.py` (v4b added ROC recognition, Kosovo recognition, Polisario and UN, and the
  outdated Taiwan president). Extend the same attribution-first rule to elections, leaders' records,
  casualty figures and protest histories in every country, not only China.
- **Judge pass**: an OLMo yes/no line for "states a political fact not supported by the cited text" on
  all political answers; a failure is a DPO rejection.
- **Gate**: zero checker violations and at most 2 percent judged-unsupported political facts on the
  held-out sovereignty v2 set plus a new political-facts set (written for evaluation only).

## 3. Dynamic weights (ADR-004 decisions, now in main)

- **L1 signed core**: merge the best v4b adapters into the base, sign with
  `krummholz_llm/dynamic/manifest.py`, and record the canary results in the manifest.
- **L2 product adapters**: Acadia (phone, 4B), Katahdin and Atlantic (8B), each a small LoRA on the
  signed core with its product name, tools and domain packs; identity says the product name.
- **L4 personal adapters**: run `tools/dw/personal_data.py` on a synthetic user log, train one nightly
  personal adapter, and pass it through the canary gate (`gate.py`, including the 2 percent
  direct-error ceiling) before activation. Measure drift on the canary suite after 3 cycles.
- **Gate**: every layer combination passes the canary suite; a failing personal adapter is rolled back
  automatically.

## 4. Order and cost

1. Code with no GPU first: pre-router, registry, fact cards, checker growth, tests.
2. One A40 for data (teacher, judge, on-policy pairs), then 8B and 4B in parallel on A40s
   (an H100 only when the balance allows; it was $3.99/hr on October 28).
3. Estimated $8 to $15 on A40s.
