# Run Atlantic with the released safety gate

The downloadable bundle contains the Krummholz C++ request and reply rules, the four released judge heads, Python bindings, a local text gateway and the BGE small English F32 embedding model. All inference can run offline once the files and your llama.cpp build are on the device. The Windows x64 gate executable in the bundle was compiled from the supplied binding and used in the release replay. Linux and macOS users can run `make` in the extracted directory with a C++17 compiler. The command creates `build/gate_cli`.

Download `atlantic-v8.2-harness.zip` and a model quant from the versioned model repository. Extract the archive into its own directory. From that directory, start these processes in separate terminals. These PowerShell examples use a llama.cpp binary on the user's path and an example 8,192-token context.

```powershell
python launch_embedder.py --llama-server (Get-Command llama-server).Source
```

```powershell
llama-server -m "C:\Models\krummholz-atlantic-27b-v8.2-Q4_K_M.gguf" --alias atlantic-v8.2 --host 127.0.0.1 --port 8000 -ngl 99 -c 8192 --jinja --reasoning-format deepseek --log-disable
```

```powershell
python serve.py --binary build/gate_cli.exe --model atlantic-v8.2 --upstream-url http://127.0.0.1:8000 --embedding-url http://127.0.0.1:8089 --port 8090
```

The gateway text endpoint is `http://127.0.0.1:8090/v1/chat/completions`. Keep the generation and embedding ports on loopback. A request must pass the rules and embedding head before generation. The gateway buffers the result, removes private reasoning, checks output rules and the paired request/reply head, then returns a public final answer or a fixed refusal. An unavailable judge withholds the answer. Use non-streaming requests. The example below enables native thinking and returns only the visible reply field.

```powershell
$payload = @{
  model = 'atlantic-v8.2'
  messages = @(@{ role = 'user'; content = 'The survey lists 12 trees examined. Does it say how many were marked?' })
  chat_template_kwargs = @{ enable_thinking = $true }
  temperature = 0
  max_tokens = 512
  stream = $false
} | ConvertTo-Json -Depth 5
$response = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8090/v1/chat/completions' -ContentType 'application/json' -Body $payload
$response.choices[0].message.content
```

The command-line gateway handles text. The GGUF projector is a separate image-input component for multimodal hosts. The release's 64-record harness replay passed the exact user text and saved public replies directly to the released C++ rules and embedding heads. The example HTTP gateway also screens message roles and request options. Packaging, multiple turns, tools and changes in request formatting can alter a verdict.

The embedding weights come from BAAI and Compendium Labs under MIT. The Krummholz rules, judge files and bindings use Apache 2.0. NVIDIA Aegis 2.0, used to train the judge heads, is CC BY 4.0. See the included license files, the original harness model card, training attribution and published thresholds. [Official Krummholz harness](https://huggingface.co/krummholz-ai/krummholz-harness/tree/30f793a03d7ba5ca0f123f275171f04ac04aab4d) · [BGE weights](https://huggingface.co/BAAI/bge-small-en-v1.5) · [Pinned BGE GGUF](https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/tree/d32f8c040ea3b516330eeb75b72bcc2d3a780ab7) · [NVIDIA Aegis 2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0)
