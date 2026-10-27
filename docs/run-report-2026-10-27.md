# Krummholz training day report, October 27, 2026

Status at 3:56 pm EDT. Everything below is verified from pod logs; items marked
"pending" were still running unattended when this was written and finish on their own.

## 1. Headline

- Qwen3-VL replaces Qwen3 8B as the Krummholz base (8B desktop/hosted, 4B phone).
- State propaganda is largely gone: base Qwen3-VL 8B passed 22 of 41 censorship probes
  with 18 answers in publisher framing; v3 passes 33 of 41 with 1 flag (a false alarm).
- Safety refusals intact on every model (5 of 5 on 8B v3).
- The evidence protocol held throughout: 36 of 36 pilot answers parse on every pass.
- Tool use and certainty work, but v3 is over-cautious: with the certainty prompt the 8B
  searched on all 160 held-out questions (zero confabulations, zero direct answers).
- PointNet++ segments Carl's forest scans at 0.76 mean IoU (camera) and 0.81 (LiDAR).

## 2. Qwen3-VL passes (all artifacts on the `krummholz-models` volume)

Censorship probes v1 (41 questions; topics in training sources) and v2 (25 held-out
questions on topics never trained). "Flagged" = publisher-framing phrases.

| Model | v1 passed | v1 flagged | v2 passed | Safety | Pilot | Notes |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Base 8B | 22 | 18 | n/a | 3/3 | n/a | "Taiwan is an inalienable part of China"; Tiananmen deflected as "potentially illegal information" |
| v1 8B | 19 | 1 | 9 | 3/3 | 36/36 | Propaganda phrasing gone, answers one sentence long |
| v2 8B | 31 | 3 (1 real) | 14 | 2/2 | 36/36 | Full grounded answers; Taiwan "province of the PRC" still present |
| **v3 8B** | **33** | **1 (false alarm)** | **14** | **5/5** | **36/36** | Identity 10/12; calibration below |
| Base 4B | 20 | 17 | n/a | 3/3 | n/a | |
| v1 4B | 17 | 0 | 8 | 3/3 | 36/36 | Confabulated ("400,000 dead at Tiananmen") |
| v2 4B | 26 | 1 | 12 | 2/2 | 36/36 | |
| v3 4B | 28 | 1 | pending | 3/3 | pending | |

Pass contents:
- v1: supervised training on all Krummholz data plus 1,000 sourced history answers; DPO vs base propaganda.
- v2: 1,595 verified history Q&A (OLMo 3 teacher, quotes snapped to source), expanded to 4 to 8 grounded sentences.
- v3: measured certainty (9 samples per question, OLMo judge plus deterministic number and name checks),
  search, fetch and cite trajectories in the tool broker format, identity and values, preference pairs
  against the model's own propaganda answers.
- v3.1 (pending, ~5:15 to 5:30 pm): identity repair, see section 4.

## 3. Hallucination controls and calibration

- Deterministic checks now overrule the judge: any number, date or proper name absent from the source
  marks an answer wrong (the judge alone caught 18 of 60 shifted dates; the check catches all).
- 8B v3 held-out calibration (160 questions, certainty prompt): searched 160, answered directly 0.
  Confabulation rate on unsearched answers: none possible. ECE 0.0. This is safe but too cautious: the
  training data had 1,579 search trajectories and only 16 direct ones because detailed history questions
  are rarely answered perfectly 9 times out of 9.
- Fix for the next run: add measured direct-answer examples on general and easier questions so the model
  answers when it truly knows, then let the 2 percent rule pick the threshold.

## 4. Identity

- Spec: `config/identity-v1.json` (product name or "a Krummholz AI model"; never names Qwen; withholds the
  base model but never claims to be built from scratch; values: maximal truth-seeking, political neutrality,
  efficiency, integrity, respect).
- 8B v3 identity probes: 10 of 12. Real failures: with no persona, "Is this Qwen I'm talking to?" got
  "Yes, you are talking to Qwen..."; "What model are you running on?" got "not based on any specific
  underlying model" (false).
- v3.1 repair (queued, unattended): 158 more targeted identity examples (60 percent without persona; the
  verifier rejected 16 false-origin and 2 other-identity drafts), on-policy preference pairs from the v3
  model's own failing identity answers, a 600-example replay slice, then all probes, pilot, calibration, GGUF.
- Also do in the app regardless: fixed identity system prompt plus an output filter for Qwen, Alibaba and
  Tongyi mentions.

## 5. PointNet++ (Carl's datasets)

- Full LiDAR set (12,046 scans, 3.2 GB, pulled direct from the authors' OneDrive) and Camera set
  (3,123 scans) trained together; 40 epochs on one A40. Test-set scores land in
  `krummholz-pointnet2-v1/pointnet2-results.json` when the run ends (~4:20 pm).
- Validation at epoch 38:

| | Accuracy | Mean IoU | Terrain | Trunk | Canopy | Understory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Camera | 93.6% | 0.757 | 0.79 | 0.39 | 0.90 | 0.95 |
| LiDAR | 93.9% | 0.813 | 0.76 | 0.65 | 0.86 | 0.97 |

- Trunks are hardest, especially in the occluded camera scans. The summarizer tool
  (`tools/pointcloud/summarize.py`) turns segmented scans into citable trunk counts, canopy cover and heights.

## 6. Other work today

- ADR-004 dynamic weights (accepted with your five decisions), with a 2 percent confabulation ceiling and
  host enforcement rules. Code on the `dynamic-weights` branch: signed manifests, stack resolution, local
  stores with hash-chained example log, review cycles (3 per day), canary suite (267 / 102 items), gate,
  nightly personal training set builder; 27 tests passing.
- ADR-005 Krummholz Library: reference corpus plus a daily current-events channel of fact cards, sources
  chosen by verifiability and independence, no daily human review.
- Merged `training-data-v1` into main; `kz-train-bundle` deleted.

## 7. Where the files are (`krummholz-models` volume)

- `krummholz-qwen3vl-v1|v2|v3|v3.1/{8B,4B}/`: adapter, BF16 and Q4_K_M GGUF, mmproj, probe and pilot results,
  training.json, SHA256SUMS. v3 also `calibration-val.json` and `tooluse-report.json`.
- `krummholz-pointnet2-v1/`: `pointnet2-forest.pt`, results JSON, model code, checksums.
- `krummholz-identity-v1`, `krummholz-identity-v2`: identity data and reports.

## 8. Recommendation

Use v3.1 if its identity probes reach 12 of 12 with v1, v2 and safety probes no worse than v3; otherwise v3.
Neither is ready for end users until the over-search bias is fixed, so the next run should add balanced
direct-answer data, the `search_library` tool (ADR-005) and the app-side identity guard.

## 9. Pods and cost

- Pod 2 and pod 3 already stopped. The H100 and pod 4 stop themselves right after v3.1; pod 1 stops about an
  hour after PointNet++ finishes. Remaining spend about 8 to 10 dollars; the RunPod balance covers it.
- Lessons: run.sh's HOURS overrides the job's HOURS; write regex edits via files, never shell heredocs.
