#!/bin/bash
# Pass 4b: canonical status questions, calculate tool, Krummholz Library, time-sensitive routing.
# One size per pod (SIZES=8B on the H100, SIZES=4B on an A40). The pod with SHARED=1 builds the shared data
# (canonical status Q&A, time-sensitive questions, library); the other waits for it. Resumable: each stage is
# guarded by its output file. Continues 8B from v3.1 and 4B from v4a (falls back to v3.1).
#   shared: canon -> timesens -> library
#   per size: sovereignty pairs (status + canonical, sampled) -> identity pairs -> tool use (library first) ->
#             calculate trajectories + DPO -> routing SFT + DPO -> replay -> expanded-only history -> SFT + DPO ->
#             probes (v1, v2, identity, sovereignty v1 + v2), pilot -> calibration, calculate eval, routing eval -> GGUF
# Artifacts: /workspace/krummholz-qwen3vl-v4b/<size>/
set -uo pipefail
BRANCH=${BRANCH:-vl-pointcloud-v1}; REPO=/root/krummholz-llm; W=/root/kz4b; FINAL=/workspace/krummholz-qwen3vl-v4b
SIZES=${SIZES:-"8B"}; SHARED=${SHARED:-0}; TEACHER=allenai/Olmo-3-7B-Instruct
H=/workspace/krummholz-qwen3vl-v4/hist-sov; H2=$FINAL/shared; LIB=$FINAL/library/krummholz-library-v0.json
mkdir -p "$W" "$FINAL" "$H2"; cd "$W" || exit 1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=/root/hf
pipi() { python3 -m pip install -q "$@" || python3 -m pip install -q --break-system-packages "$@"; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -m pip install -q --break-system-packages --ignore-installed PyJWT blinker 2>/dev/null
python3 -c "import vllm, peft" 2>/dev/null || pipi vllm peft datasets || { echo PIP_FAIL; exit 1; }
[ -d "$REPO/.git" ] || git clone -q --branch "$BRANCH" https://github.com/hayvenmjackson/krummholz-llms-alpha "$REPO"
git -C "$REPO" fetch -q && git -C "$REPO" reset -q --hard "origin/$BRANCH"; git -C "$REPO" log -1 --format='repo %h %s' | tee -a "$FINAL/repo-commit.txt"
T="python3 $REPO/tools/vl/teacher.py"; C="python3 $REPO/tools/vl/calibrate.py"; P="python3 $REPO/tools/vl/pass4.py"
Q="python3 $REPO/tools/vl/pass4b.py"; X="python3 $REPO/tools/vl/identity_fix.py"
[ -f "$H/sov-status-report.json" ] || { echo NO_V4A_SOV_PACK; exit 1; }

# 1. shared data (once)
if [ "$SHARED" = 1 ]; then
  cp -n "$H/passages.jsonl" "$H2"/
  [ -f "$H2/sov-canon-report.json" ] || $Q canon --out "$H2" --model "$TEACHER" --hist "$H2" || { echo CANON_FAIL; exit 1; }
  [ -f "$H2/timesens-report.json" ] || $Q timesens --out "$H2" --model "$TEACHER" || { echo TIMESENS_FAIL; exit 1; }
  [ -f "$LIB" ] || $Q library --out "$LIB" --passages /workspace/krummholz-qwen3vl-*/hist*/passages.jsonl || { echo LIBRARY_FAIL; exit 1; }
  touch "$H2/SHARED_OK"; echo SHARED_OK
else
  for i in $(seq 1 240); do [ -f "$H2/SHARED_OK" ] && break; sleep 60; done
  [ -f "$H2/SHARED_OK" ] || { echo SHARED_TIMEOUT; exit 1; }
fi

# 2. llama.cpp once per pod
L=$W/llama.cpp
if [ ! -x $L/build/bin/llama-quantize ]; then
  git clone -q --depth 1 --branch b11669 https://github.com/ggml-org/llama.cpp $L 2>/dev/null || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp $L
  pipi $L/gguf-py sentencepiece; command -v cmake >/dev/null || { pipi cmake; hash -r; }
  cmake -S $L -B $L/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build $L/build -j 8 --target llama-quantize >/dev/null || echo BUILD_WARN
fi

for S in $SIZES; do
  BASE=/root/models/qwen3-vl-$S; D=$W/d-$S; O=$W/out-$S; F=$FINAL/$S
  A=/workspace/krummholz-qwen3vl-v3.1/$S/adapter
  [ "$S" = 4B ] && [ -f /workspace/krummholz-qwen3vl-v4/4B/adapter/adapter_config.json ] && A=/workspace/krummholz-qwen3vl-v4/4B/adapter
  [ -f "$A/adapter_config.json" ] || { echo "NO_INIT_ADAPTER $S"; continue; }
  echo "INIT $S $A"; mkdir -p "$D" "$F"
  [ -f "$BASE/config.json" ] || python3 -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-VL-$S-Instruct', local_dir='$BASE')" || { echo WEIGHTS_FAIL $S; continue; }
  cp "$H"/history_qa.* "$H"/history_chat.*.jsonl "$H"/sov_status.* "$H2"/sov_canon.* "$D"/
  # v4a filtered the shared pack in place; use every question for pairs and tool use, filter to expanded before training
  [ -f "$H/history_chat.train.full.jsonl" ] && cp "$H/history_chat.train.full.jsonl" "$D/history_chat.train.jsonl" && rm -f "$D/history_chat.train.full.jsonl"
  cp /workspace/krummholz-identity-v1/identity.*.jsonl /workspace/krummholz-identity-v2/identity-extra.*.jsonl "$D"/
  [ -f "$D/sovereignty-report.json" ] || $P sov_pairs --out "$D" --model "$BASE" --adapter "$A" --hist "$D" || { echo SOV_PAIRS_FAIL $S; continue; }
  [ -f "$D/identity-fix-report.json" ] || $X pairs --out "$D" --model "$BASE" --adapter "$A" --identity "$D" || { echo ID_PAIRS_FAIL $S; continue; }
  [ -f "$D/tooluse-report.json" ] || { $C sample --out "$D" --model "$BASE" --adapter "$A" && $C grade --out "$D" --model "$TEACHER" \
    && $C build --out "$D" --library "$LIB"; } || { echo TOOLUSE_FAIL $S; continue; }
  mkdir -p "$D/heldout"; [ -f "$D/graded-all.jsonl" ] && mv "$D"/samples-all.jsonl "$D"/graded-all.jsonl "$D/heldout"/
  [ -f "$D/calc-report.json" ] || $Q calc --out "$D" --model "$BASE" --adapter "$A" || { echo CALC_FAIL $S; continue; }
  [ -f "$D/routing-report.json" ] || $Q route_pairs --out "$D" --model "$BASE" --adapter "$A" --hist "$H2" || { echo ROUTE_FAIL $S; continue; }
  [ -f "$D/replay_slice.train.jsonl" ] || $X replay --out "$D" --n 800 --sources $REPO/data/vl-v1/banking.train.jsonl $REPO/data/vl-v1/fia.train.jsonl \
    $REPO/data/vl-v1/pointcloud.train.jsonl $REPO/data/vl-v1/maine.train.jsonl $REPO/data/vl-v1/defects.train.jsonl \
    /workspace/krummholz-qwen3vl-v2/hist/history_chat.train.jsonl /workspace/krummholz-qwen3vl-v1/inputs-8B/replay.train.jsonl
  [ -f "$D/EXPANDED_ONLY" ] || { $Q expanded_only --out "$D" && touch "$D/EXPANDED_ONLY"; }
  cp "$D"/*-report.json "$F"/ 2>/dev/null
  python3 "$REPO/tools/vl/train_vl.py" --base "$BASE" --data "$D" --out "$O" --init-adapter "$A" --lr 3e-5 --maine-repeat 1 \
    || { echo TRAIN_FAIL $S; continue; }
  echo TRAIN_OK $S
  cp -r "$O/adapter" "$O"/*.json "$F"/
  if [ -d "$O/merged" ]; then
    E=$W/eval-$S; mkdir -p "$E"; cp "$H"/history_qa.* "$H"/history_chat.* "$E"/
    $C sample --out "$E" --model "$O/merged" --split val --system && $C grade --out "$E" --model "$TEACHER" --split val \
      && $C report --out "$E" --split val && cp "$E/calibration-val.json" "$F"/ || echo CALIBRATION_EVAL_FAIL
    $Q calc_eval --out "$E" --model "$O/merged" --heldout "$D/calc-heldout.json_" && cp "$E/calc-eval.json" "$F"/ || echo CALC_EVAL_FAIL
    $Q route_eval --out "$E" --model "$O/merged" --hist "$H2" && cp "$E/route-eval.json" "$F"/ || echo ROUTE_EVAL_FAIL
    N=qwen3-vl-$(echo $S | tr A-Z a-z)-krummholz-v4b
    python3 $L/convert_hf_to_gguf.py "$O/merged" --outtype bf16 --outfile "$O/$N-bf16.gguf" || echo CONVERT_FAIL $S
    python3 $L/convert_hf_to_gguf.py "$O/merged" --mmproj --outtype f16 --outfile "$O/mmproj-$N-f16.gguf" || echo MMPROJ_FAIL $S
    [ -f "$O/$N-bf16.gguf" ] && $L/build/bin/llama-quantize "$O/$N-bf16.gguf" "$O/$N-Q4_K_M.gguf" Q4_K_M >/dev/null && echo QUANT_OK $S
    cp "$BASE/LICENSE" "$F/LICENSE-Qwen3-VL-Apache-2.0.txt" 2>/dev/null
    rm -rf "$O/merged"; (cd "$O" && sha256sum *.gguf > SHA256SUMS-gguf); cp "$O"/*.gguf "$O/SHA256SUMS-gguf" "$F"/ && echo COPY_OK $S
  fi
  rm -rf "$O" "$BASE"
  echo V4B_DONE $S
done
echo JOB_DONE
