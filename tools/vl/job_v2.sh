#!/bin/bash
# Krummholz VL decensor pass 2. Starts from the v1 adapters on the volume:
#   1. teacher Q&A on v1 + v2 history passages (near-miss quotes snapped to passage text)
#   2. v1-tuned model answers every history question (on-policy) and both probe suites
#   3. OLMo 3 judges those answers; refusals, propaganda framing and contradictions become DPO rejects
#   4. continue the v1 adapter: SFT on the fuller history set plus the v1 skills data, then DPO
#   5. probes v1 + v2, pilot, merged GGUF + mmproj + Q4_K_M
# Artifacts: /workspace/krummholz-qwen3vl-v2/<size>/
set -uo pipefail
BRANCH=${BRANCH:-vl-pointcloud-v1}; REPO=/root/krummholz-llm; W=/root/kz2; V1=/workspace/krummholz-qwen3vl-v1; FINAL=/workspace/krummholz-qwen3vl-v2
SIZES=${SIZES:-"8B 4B"}; TEACHER=allenai/Olmo-3-7B-Instruct
mkdir -p "$W" "$FINAL"; cd "$W" || exit 1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=/root/hf
pipi() { python3 -m pip install -q "$@" || python3 -m pip install -q --break-system-packages "$@"; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import vllm, peft" 2>/dev/null || {
  python3 -m pip install -q --break-system-packages --ignore-installed PyJWT blinker 2>/dev/null
  pipi vllm peft datasets || { echo PIP_FAIL; exit 1; }
}
[ -d "$REPO/.git" ] || git clone -q --branch "$BRANCH" https://github.com/hayvenmjackson/krummholz-llms-alpha "$REPO"
git -C "$REPO" fetch -q && git -C "$REPO" checkout -q "$BRANCH" && git -C "$REPO" reset -q --hard "origin/$BRANCH"
git -C "$REPO" log -1 --format='repo %h %s' | tee "$FINAL/repo-commit.txt"
T="python3 $REPO/tools/vl/teacher.py"
# The teacher stage runs once and is shared through the volume (TEACH_ONLY=1 does just this).
H=$FINAL/hist
if [ ! -f "$H/history_chat.train.jsonl" ]; then
  mkdir -p "$H/v1" "$H/v2"
  $T fetch --out "$H/v1" --sources config/history-sources-v1.json || { echo FETCH_FAIL; exit 1; }
  $T fetch --out "$H/v2" --sources config/history-sources-v2.json || { echo FETCH_FAIL; exit 1; }
  cat "$H/v1/passages.jsonl" "$H/v2/passages.jsonl" > "$H/passages.jsonl"
  $T teach --out "$H" --model "$TEACHER" || { echo TEACH_FAIL; exit 1; }
  echo TEACH_OK
fi
if [ ! -f "$H/expand-report.json" ] && [ "${TEACH_ONLY:-0}" = "1" -o "${EXPANDER:-0}" = "1" ]; then
  $T expand --out "$H" --model "$TEACHER" || { echo EXPAND_FAIL; exit 1; }
  echo EXPAND_OK
fi
[ "${TEACH_ONLY:-0}" = "1" ] && { echo JOB_DONE; exit 0; }
until [ -f "$H/expand-report.json" ]; do sleep 60; done

[ -x llama.cpp/build/bin/llama-quantize ] || {
  git clone -q --depth 1 --branch b11669 https://github.com/ggml-org/llama.cpp 2>/dev/null || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp
  pipi ./llama.cpp/gguf-py sentencepiece
  command -v cmake >/dev/null || { pipi cmake; hash -r; }
  cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build llama.cpp/build -j 8 --target llama-quantize >/dev/null || echo BUILD_WARN
}
git -C llama.cpp rev-parse HEAD > "$FINAL/llama-cpp-commit.txt"

for S in $SIZES; do
  M=Qwen/Qwen3-VL-$S-Instruct; BASE=/root/models/qwen3-vl-$S; D=$W/d-$S; O=$W/out-$S; F=$FINAL/$S; A1=$V1/$S/adapter
  [ -f "$A1/adapter_config.json" ] || { echo "NO_V1_ADAPTER $S"; continue; }
  mkdir -p "$D" "$F"
  [ -f "$BASE/config.json" ] || python3 -c "from huggingface_hub import snapshot_download as s; s('$M', local_dir='$BASE')" || { echo WEIGHTS_FAIL $S; continue; }
  cp "$H"/history_qa.* "$H"/history_chat.* "$D"/
  # general long-form replay from v1 keeps answers from collapsing to one sentence
  cp "$V1/inputs-8B/replay.train.jsonl" "$D/replay.train.jsonl" 2>/dev/null || echo NO_REPLAY
  $T base --out "$D" --model "$BASE" --adapter "$A1" --tag v1 || { echo ONPOLICY_FAIL $S; continue; }
  $T judge --out "$D" --model "$TEACHER" || { echo JUDGE_FAIL $S; continue; }
  cp "$D"/probes-v*-v1.json "$D"/judge-report.json "$F"/
  rm -f "$D"/onpolicy-answers.jsonl
  python3 "$REPO/tools/vl/train_vl.py" --base "$BASE" --data "$REPO/data/vl-v1" "$D" --out "$O" \
    --init-adapter "$A1" --lr 5e-5 --maine-repeat 1 || { echo TRAIN_FAIL $S; continue; }
  echo TRAIN_OK $S
  cp -r "$O/adapter" "$O"/*.json "$F"/
  if [ -d "$O/merged" ]; then
    N=qwen3-vl-$(echo $S | tr A-Z a-z)-krummholz-v2
    python3 llama.cpp/convert_hf_to_gguf.py "$O/merged" --outtype bf16 --outfile "$O/$N-bf16.gguf" || echo CONVERT_FAIL $S
    python3 llama.cpp/convert_hf_to_gguf.py "$O/merged" --mmproj --outtype f16 --outfile "$O/mmproj-$N-f16.gguf" || echo MMPROJ_FAIL $S
    [ -f "$O/$N-bf16.gguf" ] && llama.cpp/build/bin/llama-quantize "$O/$N-bf16.gguf" "$O/$N-Q4_K_M.gguf" Q4_K_M >/dev/null && echo QUANT_OK $S
    cp "$BASE/LICENSE" "$F/LICENSE-Qwen3-VL-Apache-2.0.txt" 2>/dev/null
    rm -rf "$O/merged"
    (cd "$O" && sha256sum *.gguf > SHA256SUMS-gguf); cp "$O"/*.gguf "$O/SHA256SUMS-gguf" "$F"/
  fi
  rm -rf "$O"
  ls -la "$F"
done
echo JOB_DONE
