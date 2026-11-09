# v6 probe answers

The answers behind the numbers in the v6 model cards ([docs/model-cards](../../docs/model-cards/README.md)),
saved as each run produced them.

| Files | Model |
| --- | --- |
| `core-35b-a3b-fix3--*.json` | Krummholz Core 35B-A3B v6.3 |
| `atlantic-27b--*.json` | Atlantic 27B v6 |
| `atlantic-lite-4b-fix3--*.json` | Atlantic Lite 4B v6.3 |
| `core-35b-a3b--probes-v*-base.json` | The base model Qwen/Qwen3.6-35B-A3B on the censorship suites |
| `atlantic-27b--probes-v*-base.json` | The base model Qwen/Qwen3.8-27B on the censorship suites |
| `atlantic-lite-4b--probes-v*-base.json` | The base model Qwen/Qwen3.5-4B on the censorship suites |

Each file pairs with a suite in [evaluation](..): `X--identity-probes-v1.json` with `identity-probes-v1.json`, and so
on for consciousness, human and sovereignty v1 to v3; `X--probes-v1.json` and `X--probes-v2.json` with
`decensor-probes-v1.json` and `decensor-probes-v2.json`.

Settings: the merged model before quantization, in Hugging Face transformers; the user's message only, with no
system prompt; greedy decoding; at most 700 new tokens; chat template with thinking off.

Rescore the model alone, from the repository root:

```shell
python tools/vl/probes.py evaluation/v6/atlantic-27b--human-probes-v1.json evaluation/human-probes-v1.json
```

Rescore through the runtime guards, as a person sees the answers. The product is `atlantic`, `atlantic-lite`, or
`core` for the 35B core (any name the guards do not know gives the generic Krummholz wording):

```shell
python tools/vl/probes.py evaluation/v6/atlantic-27b--human-probes-v1.json evaluation/human-probes-v1.json --runtime atlantic
```

The guards were written or extended after these answers were read, so scores through the guards are a check of
those fixes, not a held-out result. The release gates score the model alone.
