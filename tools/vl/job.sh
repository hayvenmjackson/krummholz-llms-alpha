#!/bin/bash
# Krummholz VL v1 on one GPU pod: history data (OLMo 3 teacher), base answers, then
# LoRA SFT + DPO on Qwen3-VL 8B and 4B, probes and pilot, merged GGUF + mmproj + Q4_K_M.
# Artifacts: /workspace/krummholz-qwen3vl-v1/<size>/
set -uo pipefail
BRANCH=${BRANCH:-vl-pointcloud-v1}; REPO=/root/krummholz-llm; W=/root/kz; FINAL=/workspace/krummholz-qwen3vl-v1
SIZES=${SIZES:-"8B 4B"}; TEACHER=allenai/Olmo-3-7B-Instruct
mkdir -p "$W" "$FINAL"; cd "$W" || exit 1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=/root/hf
pipi() { python3 -m pip install -q "$@" || python3 -m pip install -q --break-system-packages "$@"; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /root /workspace | tail -2
pipi vllm peft datasets || { echo PIP_FAIL; exit 1; }
python3 -c "import torch, transformers, vllm, peft; from transformers import Qwen3VLForConditionalGeneration; print('torch', torch.__version__, 'transformers', transformers.__version__, 'vllm', vllm.__version__, 'peft', peft.__version__)" || { echo IMPORT_FAIL; exit 1; }
git clone -q --branch "$BRANCH" https://github.com/hayvenmjackson/krummholz-llms-alpha "$REPO" && git -C "$REPO" log -1 --format='repo %h %s' | tee "$FINAL/repo-commit.txt"
T="python3 $REPO/tools/vl/teacher.py"
H=$W/hist; mkdir -p "$H"

$T fetch --out "$H" || { echo FETCH_FAIL; exit 1; }
$T teach --out "$H" --model "$TEACHER" || { echo TEACH_FAIL; exit 1; }
rm -rf /root/hf/hub/models--allenai--Olmo-3-7B-Instruct
cp "$H"/passages.jsonl "$H"/teacher-report.json "$FINAL"/

git clone -q --depth 1 --branch b11669 https://github.com/ggml-org/llama.cpp 2>/dev/null || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp
git -C llama.cpp rev-parse HEAD > "$FINAL/llama-cpp-commit.txt"
pipi ./llama.cpp/gguf-py sentencepiece
command -v cmake >/dev/null || { pipi cmake; hash -r; }
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build llama.cpp/build -j 8 --target llama-quantize >/dev/null || echo BUILD_WARN

git -C "$REPO" pull -q && git -C "$REPO" log -1 --format='repo for training %h %s' | tee -a "$FINAL/repo-commit.txt"
[ -f "$REPO/data/vl-v1/manifest.json" ] || { echo NO_LOCAL_DATA; exit 1; }
for S in $SIZES; do
  M=Qwen/Qwen3-VL-$S-Instruct; BASE=/root/models/qwen3-vl-$S; D=$W/base-$S; O=$W/out-$S; F=$FINAL/$S
  mkdir -p "$D" "$F"
  python3 -c "from huggingface_hub import snapshot_download as s; s('$M', local_dir='$BASE')" || { echo WEIGHTS_FAIL $S; continue; }
  (cd "$BASE" && sha256sum *.safetensors > "$F/base-SHA256SUMS")
  cp "$H"/history_qa.* "$H"/history_chat.* "$D"/
  $T base --out "$D" --model "$BASE" || { echo BASE_FAIL $S; continue; }
  [ "$S" = "8B" ] && cp "$D/replay.train.jsonl" "$H/replay.train.jsonl"
  [ -f "$D/replay.train.jsonl" ] || cp "$H/replay.train.jsonl" "$D/" 2>/dev/null
  cp "$D/probes-base.json" "$F"/
  python3 "$REPO/tools/vl/train_vl.py" --base "$BASE" --data "$REPO/data/vl-v1" "$D" --out "$O" || { echo TRAIN_FAIL $S; continue; }
  echo TRAIN_OK $S
  cp -r "$O/adapter" "$O"/*.json "$F"/
  if [ -d "$O/merged" ]; then
    N=qwen3-vl-$(echo $S | tr A-Z a-z)-krummholz
    python3 llama.cpp/convert_hf_to_gguf.py "$O/merged" --outtype bf16 --outfile "$O/$N-bf16.gguf" || echo CONVERT_FAIL $S
    python3 llama.cpp/convert_hf_to_gguf.py "$O/merged" --mmproj --outtype f16 --outfile "$O/mmproj-$N-f16.gguf" || echo MMPROJ_FAIL $S
    [ -f "$O/$N-bf16.gguf" ] && llama.cpp/build/bin/llama-quantize "$O/$N-bf16.gguf" "$O/$N-Q4_K_M.gguf" Q4_K_M >/dev/null && echo QUANT_OK $S
    cp "$BASE/LICENSE" "$F/LICENSE-Qwen3-VL-Apache-2.0.txt" 2>/dev/null
    rm -rf "$O/merged"
    (cd "$O" && sha256sum *.gguf > SHA256SUMS-gguf) ; cp "$O"/*.gguf "$O/SHA256SUMS-gguf" "$F"/
  fi
  rm -rf "$BASE" "$O"
  ls -la "$F"
done
echo JOB_DONE
