#!/bin/bash
# Krummholz VL pass 3: certainty calibration and read-only tool use, continuing the v2 adapter.
#   1. v2 model answers every history question 8 times plus greedy (no sources)
#   2. OLMo 3 grades each answer against its passage -> measured certainty per question
#   3. build trajectories: >= 0.95 answer directly; < 0.95 search -> fetch -> cite
#   4. continue the v2 adapter on trajectories + v1 skills + expanded history + replay
#   5. probes, pilot, calibration report on held-out questions, GGUF + mmproj + Q4_K_M
# Usage: SIZE=8B bash job_v3.sh     Artifacts: /workspace/krummholz-qwen3vl-v3/<size>/
set -uo pipefail
S=${SIZE:-8B}; BRANCH=${BRANCH:-vl-pointcloud-v1}; REPO=/root/krummholz-llm; W=/root/kz3
V2=/workspace/krummholz-qwen3vl-v2; FINAL=/workspace/krummholz-qwen3vl-v3; F=$FINAL/$S
M=Qwen/Qwen3-VL-$S-Instruct; BASE=/root/models/qwen3-vl-$S; D=$W/d-$S; O=$W/out-$S; A2=$V2/$S/adapter; TEACHER=allenai/Olmo-3-7B-Instruct
mkdir -p "$W" "$D" "$F"; cd "$W" || exit 1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=/root/hf
pipi() { python3 -m pip install -q "$@" || python3 -m pip install -q --break-system-packages "$@"; }
python3 -c "import vllm, peft" 2>/dev/null || { python3 -m pip install -q --break-system-packages --ignore-installed PyJWT blinker 2>/dev/null; pipi vllm peft datasets || { echo PIP_FAIL; exit 1; }; }
[ -d "$REPO/.git" ] || git clone -q --branch "$BRANCH" https://github.com/hayvenmjackson/krummholz-llms-alpha "$REPO"
git -C "$REPO" fetch -q && git -C "$REPO" reset -q --hard "origin/$BRANCH"; git -C "$REPO" log -1 --format='repo %h %s' | tee "$F/repo-commit.txt"
[ -f "$A2/adapter_config.json" ] || { echo NO_V2_ADAPTER; exit 1; }
[ -f "$BASE/config.json" ] || python3 -c "from huggingface_hub import snapshot_download as s; s('$M', local_dir='$BASE')" || { echo WEIGHTS_FAIL; exit 1; }
C="python3 $REPO/tools/vl/calibrate.py"

cp "$V2"/hist/history_qa.* "$V2"/hist/history_chat.* "$D"/
cp /workspace/krummholz-qwen3vl-v1/inputs-8B/replay.train.jsonl "$D"/ 2>/dev/null || echo NO_REPLAY
# identity and constitutional values (tools/vl/identity.py), generated once and shared on the volume
cp /workspace/krummholz-identity-v1/identity.*.jsonl "$D"/ 2>/dev/null && echo IDENTITY_DATA || echo NO_IDENTITY_DATA
$C sample --out "$D" --model "$BASE" --adapter "$A2" || { echo SAMPLE_FAIL; exit 1; }
$C grade --out "$D" --model "$TEACHER" || { echo GRADE_FAIL; exit 1; }
$C build --out "$D" || { echo BUILD_FAIL; exit 1; }
cp "$D/tooluse-report.json" "$F"/
mkdir -p "$D/heldout" && mv "$D"/samples-all.jsonl "$D"/graded-all.jsonl "$D/heldout"/

python3 "$REPO/tools/vl/train_vl.py" --base "$BASE" --data "$REPO/data/vl-v1" "$D" --out "$O" \
  --init-adapter "$A2" --lr 5e-5 --maine-repeat 1 || { echo TRAIN_FAIL $S; exit 1; }
echo TRAIN_OK $S
cp -r "$O/adapter" "$O"/*.json "$F"/

if [ -d "$O/merged" ]; then
  # calibration on held-out questions, answered by the merged model with the certainty/tool system prompt
  E=$W/eval-$S; mkdir -p "$E"; cp "$D"/history_qa.* "$D"/history_chat.* "$E"/
  $C sample --out "$E" --model "$O/merged" --split val --system && $C grade --out "$E" --model "$TEACHER" --split val \
    && $C report --out "$E" --split val && cp "$E/calibration-val.json" "$F"/ || echo CALIBRATION_EVAL_FAIL
  [ -x llama.cpp/build/bin/llama-quantize ] || {
    git clone -q --depth 1 --branch b11669 https://github.com/ggml-org/llama.cpp 2>/dev/null || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp
    pipi ./llama.cpp/gguf-py sentencepiece; command -v cmake >/dev/null || { pipi cmake; hash -r; }
    cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build llama.cpp/build -j 8 --target llama-quantize >/dev/null || echo BUILD_WARN; }
  N=qwen3-vl-$(echo $S | tr A-Z a-z)-krummholz-v3
  python3 llama.cpp/convert_hf_to_gguf.py "$O/merged" --outtype bf16 --outfile "$O/$N-bf16.gguf" || echo CONVERT_FAIL $S
  python3 llama.cpp/convert_hf_to_gguf.py "$O/merged" --mmproj --outtype f16 --outfile "$O/mmproj-$N-f16.gguf" || echo MMPROJ_FAIL $S
  [ -f "$O/$N-bf16.gguf" ] && llama.cpp/build/bin/llama-quantize "$O/$N-bf16.gguf" "$O/$N-Q4_K_M.gguf" Q4_K_M >/dev/null && echo QUANT_OK $S
  cp "$BASE/LICENSE" "$F/LICENSE-Qwen3-VL-Apache-2.0.txt" 2>/dev/null
  rm -rf "$O/merged"; (cd "$O" && sha256sum *.gguf > SHA256SUMS-gguf); cp "$O"/*.gguf "$O/SHA256SUMS-gguf" "$F"/
fi
echo V3_DONE $S
