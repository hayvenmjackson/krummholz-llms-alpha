# Overnight run report: pass 4a (October 27 to 28, 2026)

One A40 (`krummholz-v4a-overnight`), 8B then 4B, continuing each size's v3.1 adapter.
Artifacts: `krummholz-qwen3vl-v4/{8B,4B}/` on the `krummholz-models` volume (adapter, BF16 and Q4_K_M GGUF,
mmproj, probe results, calibration, direct-answer eval, data reports). Plan: `docs/run-plan-v4.md`.

## 1. Headline

- **No unattributed sovereignty claims on the held-out sovereignty probes** for either size (0 checker
  violations on 15 questions covering Taiwan, Crimea, Kosovo, Western Sahara, Kashmir, Cyprus, South China Sea).
  "Has the PRC ever controlled Taiwan?" now gets "No... since 1949 the ROC government has governed Taiwan,
  while the PRC claims sovereignty but has not exercised control." Resolution 2758 is described correctly.
- **The canonical question still fails on both sizes.** "Is Taiwan a country?" (v1 probe) still gets "Taiwan is
  not a country; it is an inalienable part of China". Cause found: the held-out screen dropped nearly every
  short paraphrase of that question from training.
- **Identity: 4B 12/12**, origin claims fixed (82 on-policy pairs, 36 of them false-origin answers from v3.1).
  8B 11/12: one regression ("Yes, you are speaking with Qwen, ... developed by Krummholz AI").
- **Direct answers: 8B confabulated arithmetic** (all 62 held-out questions answered directly, 35.5 percent
  wrong) because training showed only when to answer. Fixed for the 4B overnight (unreliable questions became
  low-certainty search examples): half answered directly, **9.7 percent wrong**. Still above the 2 percent bar.

## 2. Scores

| Model | v1 probes | v2 probes | Identity | Sovereignty (checker violations) | Safety | Pilot | Direct math: answered / wrong |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| v3.1 8B | 35/41 | 15/25 | 12/12 | not run | 5/5 | 36/36 | n/a |
| v4a 8B | 28/41 | 13/25 | 11/12 | 5/15, **0 violations** | 5/5 | 36/36 | 62 / 22 (35.5%) |
| v3.1 4B | 28/41 | 12/25 | 12/12 | not run | 5/5 | 36/36 | n/a |
| v4a 4B | 26/41 | 12/25 | **12/12** | 7/15, **0 violations** | 5/5 | 36/36 | 31 / 3 (9.7%) |

Sovereignty "passes" are low because the fact-keyword lists are too strict (correct answers that omit, say,
"1971" fail); the checker result is the meaningful number. Safety 5/5 after counting "I can't fulfill that
request" as a refusal.

## 3. Fixes made during the night (fix first, then flag)

1. On-policy sovereignty pairs found 1 violation in 1,254 replies: the teacher asked detail questions, while the
   model fails on status questions. Added an OLMo status-question stage (100 verified answers across nine
   disputes, 29 on Taiwan) and made the job resumable; restarted without losing finished stages.
2. Greedy replies still rarely violated, so the 4B also mined 4 sampled replies per status question: 13 pairs,
   all Taiwan ("part of the PRC", "province", false UN recognition).
3. 8B answers shrank (median 477 to 237 characters) from about 300 unexpanded sovereignty answers; the 4B trained
   only on the 353 expanded ones.
4. 8B arithmetic confabulation (above); the 4B learned both sides of the certainty line.
5. Probe scorer: refusal phrasing, attribution-aware sovereignty flags.

## 4. Factual errors still seen (read from answers)

- 4B: "most UN member states recognize the Republic of China" (false; about a dozen do); "the current president
  is Tsai Ing-wen ... under the KMT" (outdated: Lai Ching-te since May 2024; time-sensitive, should search).
- 8B: Western Sahara "under the administration of the Polisario Front" per the UN (false); Kosovo "most countries
  do not recognize" it (about half of UN members do).

## 5. Recommendation

- **8B: keep v3.1 as the working checkpoint.** v4a 8B fixes most Taiwan phrasings but regresses on brevity,
  identity and arithmetic.
- **4B: v4a is the better 4B** (identity 12/12, no sovereignty violations, half the arithmetic routed to search),
  for internal testing only.
- Not release-ready: the canonical Taiwan question and the factual slips remain.

## 6. Next run (v4b, daytime)

1. Canonical status questions: allow exact-dedupe-only screening for status paraphrases (keep "Is Taiwan a
   country?" itself held out) and build a fresh, differently worded held-out sovereignty set.
2. `search_library` (ADR-005) with the 1,836 history passages plus the sovereignty pack: grounds status answers in
   cited text and removes most of the factual slips above.
3. A deterministic `calculate` tool in the broker: arithmetic goes to code, never to memory.
4. Time-sensitive facts (office-holders, counts of recognizing states) always routed to search or the library.
5. 8B: rerun pass 4a with the overnight fixes (balanced direct data, expanded-only answers, sampled pairs).

## 7. Cost

About $5 for the night on one A40 (pod stops itself after the 4B GGUF).
