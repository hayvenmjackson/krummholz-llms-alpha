#!/bin/bash
# Re-run only the training and export stage of job.sh for one size, from inputs saved on the volume
# (history Q&A, DPO pairs, replay, base probes). Then optionally chain decensor pass 2.
# Usage: SIZE=8B CHAIN_V2=1 bash job_train_only.sh
set -uo pipefail
S=${SIZE:-8B}; BRANCH=${BRANCH:-vl-pointcloud-v1}; REPO=/root/krummholz-llm; W=/root/kz
IN=/workspace/krummholz-qwen3vl-v1/inputs-$S; F=/workspace/krummholz-qwen3vl-v1/$S
M=Qwen/Qwen3-VL-$S-Instruct; BASE=/root/models/qwen3-vl-$S; O=$W/out-$S
mkdir -p "$W" "$F"; cd "$W" || exit 1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=/root/hf
pipi() { python3 -m pip install -q "$@" || python3 -m pip install -q --break-system-packages "$@"; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -m pip install -q --break-system-packages --ignore-installed PyJWT blinker 2>/dev/null
pipi vllm peft datasets || { echo PIP_FAIL; exit 1; }
[ -d "$REPO/.git" ] || git clone -q --branch "$BRANCH" https://github.com/hayvenmjackson/krummholz-llms-alpha "$REPO"
git -C "$REPO" pull -q; git -C "$REPO" log -1 --format='repo %h %s' | tee "$F/repo-commit-train.txt"
python3 -c "from huggingface_hub import snapshot_download as s; s('$M', local_dir='$BASE')" || { echo WEIGHTS_FAIL; exit 1; }
(cd "$BASE" && sha256sum *.safetensors > "$F/base-SHA256SUMS")
cp "$IN/probes-base.json" "$F/" 2>/dev/null
git clone -q --depth 1 --branch b11669 https://github.com/ggml-org/llama.cpp 2>/dev/null || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp
git -C llama.cpp rev-parse HEAD > "$F/llama-cpp-commit.txt"
pipi ./llama.cpp/gguf-py sentencepiece
command -v cmake >/dev/null || { pipi cmake; hash -r; }
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build llama.cpp/build -j 8 --target llama-quantize >/dev/null || echo BUILD_WARN

python3 "$REPO/tools/vl/train_vl.py" --base "$BASE" --data "$REPO/data/vl-v1" "$IN" --out "$O" || { echo TRAIN_FAIL $S; exit 1; }
echo TRAIN_OK $S
cp -r "$O/adapter" "$O"/*.json "$F"/
if [ -d "$O/merged" ]; then
  N=qwen3-vl-$(echo $S | tr A-Z a-z)-krummholz
  python3 llama.cpp/convert_hf_to_gguf.py "$O/merged" --outtype bf16 --outfile "$O/$N-bf16.gguf" || echo CONVERT_FAIL $S
  python3 llama.cpp/convert_hf_to_gguf.py "$O/merged" --mmproj --outtype f16 --outfile "$O/mmproj-$N-f16.gguf" || echo MMPROJ_FAIL $S
  [ -f "$O/$N-bf16.gguf" ] && llama.cpp/build/bin/llama-quantize "$O/$N-bf16.gguf" "$O/$N-Q4_K_M.gguf" Q4_K_M >/dev/null && echo QUANT_OK $S
  cp "$BASE/LICENSE" "$F/LICENSE-Qwen3-VL-Apache-2.0.txt" 2>/dev/null
  rm -rf "$O/merged"
  (cd "$O" && sha256sum *.gguf > SHA256SUMS-gguf); cp "$O"/*.gguf "$O/SHA256SUMS-gguf" "$F"/
fi
rm -rf "$O"
echo V1_DONE $S
if [ "${CHAIN_V2:-0}" = "1" ]; then
  mkdir -p /workspace/krummholz-qwen3vl-v2/logs
  SIZES=$S bash "$REPO/tools/vl/job_v2.sh" > "/workspace/krummholz-qwen3vl-v2/logs/job-$S.log" 2>&1
  echo "V2_EXIT $?"
fi
echo JOB_DONE
