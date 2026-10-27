#!/bin/bash
set -uo pipefail
W=/root/kz; BASE=/root/models/qwen3-8b; REPO=/root/krummholz-llm; FINAL=/workspace/krummholz-qwen3-8b-v1
cd "$W" || exit 1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
pipi() { python3 -m pip install -q "$@" || python3 -m pip install -q --break-system-packages "$@"; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /root /workspace | tail -2; python3 -c "import torch;print('torch',torch.__version__)"
pipi transformers==4.57.1 peft==0.17.1 accelerate==1.10.1 huggingface-hub==0.36.0 safetensors==0.6.2 || { echo PIP_FAIL; exit 1; }
git clone -q https://github.com/hayvenmjackson/krummholz-llms-alpha "$REPO" && git -C "$REPO" log -1 --format='repo %h %s'
python3 - <<'PY' || { echo WEIGHTS_FAIL; exit 1; }
import hashlib, json
from pathlib import Path
from huggingface_hub import hf_hub_download
cfg = json.load(open("/root/krummholz-llm/config/reference-comparators.json"))
m = [x for x in cfg["models"] if x["id"] == "qwen3-8b"][0]
dest = Path("/root/models/qwen3-8b"); dest.mkdir(parents=True, exist_ok=True)
for name, pin in m["files"].items():
    p = Path(hf_hub_download(m["repository"], name, revision=m["revision"], local_dir=dest))
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    assert p.stat().st_size == pin["bytes"] and h.hexdigest() == pin["sha256"], name
print("weights verified", len(m["files"]), "files")
PY
python3 convert.py raw_train.jsonl train.jsonl raw_val.jsonl val.jsonl
python3 train_eval.py --base "$BASE" --train train.jsonl --val val.jsonl --out out || { echo TRAIN_FAIL; exit 1; }
echo TRAIN_OK
mkdir -p "$FINAL"; cp -r out/adapter out/*.json "$FINAL"/
[ -d out/merged ] || { echo NO_MERGED; exit 0; }
git clone -q --depth 1 --branch b11669 https://github.com/ggml-org/llama.cpp 2>/dev/null || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp
git -C llama.cpp rev-parse HEAD > out/llama-cpp-commit.txt
pipi ./llama.cpp/gguf-py sentencepiece
python3 llama.cpp/convert_hf_to_gguf.py out/merged --outtype bf16 --outfile out/qwen3-8b-krummholz-bf16.gguf || { echo CONVERT_FAIL; exit 1; }
cp out/merged/LICENSE out/LICENSE-Qwen3-Apache-2.0.txt; rm -rf out/merged
command -v cmake >/dev/null || { pipi cmake; hash -r; }
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build llama.cpp/build -j 8 --target llama-quantize llama-completion >/dev/null || { echo BUILD_FAIL; exit 1; }
llama.cpp/build/bin/llama-quantize out/qwen3-8b-krummholz-bf16.gguf out/qwen3-8b-krummholz-Q4_K_M.gguf Q4_K_M >/dev/null || { echo QUANT_FAIL; exit 1; }
echo QUANT_OK
python3 eval_gguf.py llama.cpp/build/bin/llama-completion out/qwen3-8b-krummholz-Q4_K_M.gguf "$BASE" out/pilot-results-Q4_K_M.json
(cd out && sha256sum *.gguf adapter/* > SHA256SUMS)
cp out/*.gguf out/*.json out/*.txt out/SHA256SUMS "$FINAL"/
ls -la "$FINAL"
