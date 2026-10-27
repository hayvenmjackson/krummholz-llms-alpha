# Krummholz VL training set, version 1

Deterministic training records for the first Qwen3-VL run (8B desktop/hosted tier,
4B phone tier). Built 2026-10-27 by `tools/vl/build_local.py`. Answers and quotes
come from the data itself, so labels are correct by construction; no Claude-written
example text. The history set (OLMo 3 teacher) and replay set are generated on the
pod by `tools/vl/teacher.py` and are not stored here.

| Set | Train | Val | Source and license | Teaches |
| --- | ---: | ---: | --- | --- |
| maine | 404 | 46 | `data/synthetic/maine-v1` (Krummholz, open-weight generator) | Conflicts, injections, absence of evidence |
| banking | 1,350 | 150 | Krummholz BANKING77 sample v2, CC BY 4.0 (PolyAI) | Acadia intent and routing classification |
| fia | 1,350 | 150 | Krummholz FIA Maine sample v2, source CC0 (USDA Forest Service) | Record-grounded answers; condition preservation; missing fields |
| defects | 168 | 18 | `forestry_synthetic_ME_NNE` v1.0, CC0 | Flag bad rows: missing values, text in numbers, units, codes |
| pointcloud | 2,250 | 225 | Tool results from `tools/pointcloud/summarize.py` on ground-truth labels of the LRSE synthetic forest datasets (camera 1,500, LiDAR 750), GPL-3.0 | Citing a point cloud tool result; saying species are not in a structure scan |

Attribution: Raverta Capua et al., LRSE synthetic forest datasets (ICPR 2024),
https://github.com/lrse/synthetic-forest-datasets. Only derived summary statistics
appear here, not the point clouds. Whether GPL-3.0 reaches trained weights is
pending counsel review.

Notes
- Exact-duplicate and near-duplicate defects are excluded from the row-level task:
  a single row cannot show them.
- Point cloud distances are in the simulator's units.
- Held out from all training: `data/pilot_eval_cases.json`, `data/smoke_cases.json`
  and `evaluation/decensor-probes-v1.json`.

Checksums are in `manifest.json`.
