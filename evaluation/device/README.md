# Device results for Atlantic Lite 4B (dynamic weights phase P)

`report.json` files from `tools/dw/device_canary.py`; `bench.txt` from `llama-bench`. Model file
`krummholz-atlantic-lite-4b-v8.1-Q4_K_M.gguf` (SHA256 `5dcb83f6...`), llama.cpp `e574a17`, canary-v1
compact (102 items), thinking off, temperature 0.

canary-v1 predates the v8 protocol, so these are device baselines, not release results.

| Date | Device | Server settings | Notes |
| --- | --- | --- | --- |
| 2026-11-23 | Cloud container, 4 x86 vCPU | `-np 1`, default prompt cache, repack on | Reference only. Peak memory 13.5 GB with these defaults |
| 2026-11-23 | Samsung Galaxy S26, 16 GB, Android 17, foreground | `-np 1 --cache-ram 0 --no-repack` | Peak memory 3.0 GB. Battery, read by hand: 50% to 29% over the run (21 points in about 40 minutes, screen on). Hottest sensor 105.1 C (sensor name not recorded in this version); the owner reports the phone did not feel hot. An earlier background run froze Termux and was force stopped; the phone felt warm, not hot; no results kept |
