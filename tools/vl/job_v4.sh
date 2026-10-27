#!/bin/bash
# Pass 4a (overnight, one GPU): sovereignty truth, origin wording, code-verified direct answers.
# Continues each size's v3.1 adapter. Per size:
#   sovereignty pack Q&A (shared) -> on-policy sovereignty pairs -> on-policy identity pairs ->
#   measured certainty + search trajectories on the sovereignty pack -> code-verified direct answers ->
#   replay slice -> SFT + DPO -> probes (v1, v2, identity, sovereignty), pilot, calibration, direct eval -> GGUF
# Artifacts: /workspace/krummholz-qwen3vl-v4/<size>/
set -uo pipefail
BRANCH=${BRANCH:-vl-pointcloud-v1}; REPO=/root/krummholz-llm; W=/root/kz4; FINAL=/workspace/krummholz-qwen3vl-v4
SIZES=${SIZES:-"8B 4B"}; TEACHER=allenai/Olmo-3-7B-Instruct; H=$FINAL/hist-sov
mkdir -p "$W" "$FINAL"; cd "$W" || exit 1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=/root/hf
pipi() { python3 -m pip install -q "$@" || python3 -m pip install -q --break-system-packages "$@"; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -m pip install -q --break-system-packages --ignore-installed PyJWT blinker 2>/dev/null
python3 -c "import vllm, peft" 2>/dev/null || pipi vllm peft datasets || { echo PIP_FAIL; exit 1; }
[ -d "$REPO/.git" ] || git clone -q --branch "$BRANCH" https://github.com/hayvenmjackson/krummholz-llms-alpha "$REPO"
git -C "$REPO" fetch -q && git -C "$REPO" reset -q --hard "origin/$BRANCH"; git -C "$REPO" log -1 --format='repo %h %s' | tee "$FINAL/repo-commit.txt"
T="python3 $REPO/tools/vl/teacher.py"; C="python3 $REPO/tools/vl/calibrate.py"; P="python3 $REPO/tools/vl/pass4.py"; X="python3 $REPO/tools/vl/identity_fix.py"

# 1. sovereignty pack, shared by both sizes
if [ ! -f "$H/expand-report.json" ]; then
  mkdir -p "$H"
  $T fetch --out "$H" --sources config/history-sources-v3.json || { echo FETCH_FAIL; exit 1; }
  $T teach --out "$H" --model "$TEACHER" || { echo TEACH_FAIL; exit 1; }
  $T expand --out "$H" --model "$TEACHER" || { echo EXPAND_FAIL; exit 1; }
  echo SOV_PACK_OK
fi
[ -f "$H/sov-status-report.json" ] || $P sov_status --out "$H" --model "$TEACHER" --hist "$H" || { echo SOV_STATUS_FAIL; exit 1; }

# 2. llama.cpp once
L=$W/llama.cpp
if [ ! -x $L/build/bin/llama-quantize ]; then
  git clone -q --depth 1 --branch b11669 https://github.com/ggml-org/llama.cpp $L 2>/dev/null || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp $L
  pipi $L/gguf-py sentencepiece; command -v cmake >/dev/null || { pipi cmake; hash -r; }
  cmake -S $L -B $L/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build $L/build -j 8 --target llama-quantize >/dev/null || echo BUILD_WARN
fi

for S in $SIZES; do
  BASE=/root/models/qwen3-vl-$S; D=$W/d-$S; O=$W/out-$S; F=$FINAL/$S; A=/workspace/krummholz-qwen3vl-v3.1/$S/adapter
  [ -f "$A/adapter_config.json" ] || { echo "NO_V31_ADAPTER $S"; continue; }
  mkdir -p "$D" "$F"
  [ -f "$BASE/config.json" ] || python3 -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-VL-$S-Instruct', local_dir='$BASE')" || { echo WEIGHTS_FAIL $S; continue; }
  cp "$H"/history_qa.* "$H"/history_chat.* "$H"/sov_status.* "$D"/
  cp /workspace/krummholz-identity-v1/identity.*.jsonl /workspace/krummholz-identity-v2/identity-extra.*.jsonl "$D"/
  [ -f "$D/sovereignty-report.json" ] || $P sov_pairs --out "$D" --model "$BASE" --adapter "$A" --hist "$H" || { echo SOV_PAIRS_FAIL $S; continue; }
  [ -f "$D/identity-fix-report.json" ] || $X pairs --out "$D" --model "$BASE" --adapter "$A" --identity "$D" || { echo ID_PAIRS_FAIL $S; continue; }
  [ -f "$D/tooluse-report.json" ] || { $C sample --out "$D" --model "$BASE" --adapter "$A" && $C grade --out "$D" --model "$TEACHER" && $C build --out "$D"; } \
    || { echo TOOLUSE_FAIL $S; continue; }
  mkdir -p "$D/heldout"; [ -f "$D/graded-all.jsonl" ] && mv "$D"/samples-all.jsonl "$D"/graded-all.jsonl "$D/heldout"/
  [ -f "$D/direct-report.json" ] || $P direct --out "$D" --model "$BASE" --adapter "$A" || { echo DIRECT_FAIL $S; continue; }
  [ -f "$D/replay_slice.train.jsonl" ] || $X replay --out "$D" --n 800 --sources $REPO/data/vl-v1/banking.train.jsonl $REPO/data/vl-v1/fia.train.jsonl \
    $REPO/data/vl-v1/pointcloud.train.jsonl $REPO/data/vl-v1/maine.train.jsonl $REPO/data/vl-v1/defects.train.jsonl \
    /workspace/krummholz-qwen3vl-v2/hist/history_chat.train.jsonl /workspace/krummholz-qwen3vl-v1/inputs-8B/replay.train.jsonl
  cp "$D"/sovereignty-report.json "$D"/identity-fix-report.json "$D"/tooluse-report.json "$D"/direct-report.json "$F"/ 2>/dev/null
  python3 "$REPO/tools/vl/train_vl.py" --base "$BASE" --data "$D" --out "$O" --init-adapter "$A" --lr 3e-5 --maine-repeat 1 \
    || { echo TRAIN_FAIL $S; continue; }
  echo TRAIN_OK $S
  cp -r "$O/adapter" "$O"/*.json "$F"/
  if [ -d "$O/merged" ]; then
    E=$W/eval-$S; mkdir -p "$E"; cp "$H"/history_qa.* "$H"/history_chat.* "$E"/
    $C sample --out "$E" --model "$O/merged" --split val --system && $C grade --out "$E" --model "$TEACHER" --split val \
      && $C report --out "$E" --split val && cp "$E/calibration-val.json" "$F"/ || echo CALIBRATION_EVAL_FAIL
    $P direct_eval --out "$E" --model "$O/merged" --heldout "$D/direct-heldout.jsonl" && cp "$E/direct-eval.json" "$F"/ || echo DIRECT_EVAL_FAIL
    N=qwen3-vl-$(echo $S | tr A-Z a-z)-krummholz-v4
    python3 $L/convert_hf_to_gguf.py "$O/merged" --outtype bf16 --outfile "$O/$N-bf16.gguf" || echo CONVERT_FAIL $S
    python3 $L/convert_hf_to_gguf.py "$O/merged" --mmproj --outtype f16 --outfile "$O/mmproj-$N-f16.gguf" || echo MMPROJ_FAIL $S
    [ -f "$O/$N-bf16.gguf" ] && $L/build/bin/llama-quantize "$O/$N-bf16.gguf" "$O/$N-Q4_K_M.gguf" Q4_K_M >/dev/null && echo QUANT_OK $S
    cp "$BASE/LICENSE" "$F/LICENSE-Qwen3-VL-Apache-2.0.txt" 2>/dev/null
    rm -rf "$O/merged"; (cd "$O" && sha256sum *.gguf > SHA256SUMS-gguf); cp "$O"/*.gguf "$O/SHA256SUMS-gguf" "$F"/
  fi
  rm -rf "$O" "$BASE"
  echo V4_DONE $S
done
echo JOB_DONE
