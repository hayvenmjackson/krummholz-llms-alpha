#!/bin/bash
# Pass 3.1: short identity repair on top of the pass-3 adapter, on the pod that ran pass 3.
# Waits for that pod's pass 3 to finish, takes over its self-stop, then:
#   on-policy identity pairs -> SFT (identity + targeted identity + replay slice) + DPO -> probes, pilot,
#   calibration, GGUF. Artifacts: /workspace/krummholz-qwen3vl-v3.1/<size>/
# Usage: SIZE=8B bash job_v31.sh
set -uo pipefail
S=${SIZE:-8B}; BRANCH=${BRANCH:-vl-pointcloud-v1}; REPO=/root/krummholz-llm; W=/root/kz31
V3=/workspace/krummholz-qwen3vl-v3; FINAL=/workspace/krummholz-qwen3vl-v3.1; F=$FINAL/$S; A3=$V3/$S/adapter
BASE=/root/models/qwen3-vl-$S; D=$W/d-$S; O=$W/out-$S; TEACHER=allenai/Olmo-3-7B-Instruct
ID1=/workspace/krummholz-identity-v1; ID2=/workspace/krummholz-identity-v2
mkdir -p "$W" "$D" "$F"; cd "$W" || exit 1
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=/root/hf
until grep -q "V3_DONE $S" "$V3/logs-$S/job.log" 2>/dev/null; do sleep 60; done
pkill -f run_v3.sh   # cancel pass 3's pending self-stop; this job owns the pod now
until [ -f "$ID2/identity-extra.train.jsonl" ]; do sleep 30; done
git -C "$REPO" fetch -q && git -C "$REPO" reset -q --hard "origin/$BRANCH"; git -C "$REPO" log -1 --format='repo %h %s' | tee "$F/repo-commit.txt"
[ -f "$BASE/config.json" ] || python3 -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-VL-$S-Instruct', local_dir='$BASE')"
X="python3 $REPO/tools/vl/identity_fix.py"
cp "$ID1"/identity.train.jsonl "$D"/identity.train.jsonl; cp "$ID1"/identity.val.jsonl "$D"/identity.val.jsonl
cp "$ID2"/identity-extra.train.jsonl "$D"/; cp "$ID2"/identity-extra.val.jsonl "$D"/ 2>/dev/null
$X pairs --out "$D" --model "$BASE" --adapter "$A3" --identity "$D" || { echo PAIRS_FAIL; exit 1; }
cp "$D/identity-fix-report.json" "$F"/
$X replay --out "$D" --n 600 --sources /root/kz3/d-$S/tooluse.train.jsonl /root/kz3/d-$S/history_chat.train.jsonl \
  /root/kz3/d-$S/replay.train.jsonl $REPO/data/vl-v1/banking.train.jsonl $REPO/data/vl-v1/fia.train.jsonl \
  $REPO/data/vl-v1/pointcloud.train.jsonl $REPO/data/vl-v1/maine.train.jsonl $REPO/data/vl-v1/defects.train.jsonl || { echo REPLAY_FAIL; exit 1; }
python3 "$REPO/tools/vl/train_vl.py" --base "$BASE" --data "$D" --out "$O" --init-adapter "$A3" --lr 3e-5 --maine-repeat 1 \
  || { echo TRAIN_FAIL $S; exit 1; }
echo TRAIN_OK $S
cp -r "$O/adapter" "$O"/*.json "$F"/
if [ -d "$O/merged" ]; then
  C="python3 $REPO/tools/vl/calibrate.py"; E=$W/eval-$S; mkdir -p "$E"
  cp /root/kz3/d-$S/history_qa.* /root/kz3/d-$S/history_chat.* "$E"/ 2>/dev/null || cp $V3/../krummholz-qwen3vl-v2/hist/history_qa.* $V3/../krummholz-qwen3vl-v2/hist/history_chat.* "$E"/
  $C sample --out "$E" --model "$O/merged" --split val --system && $C grade --out "$E" --model "$TEACHER" --split val \
    && $C report --out "$E" --split val && cp "$E/calibration-val.json" "$F"/ || echo CALIBRATION_EVAL_FAIL
  L=/root/kz3/llama.cpp; [ -x $L/build/bin/llama-quantize ] || L=$W/llama.cpp
  if [ ! -x $L/build/bin/llama-quantize ]; then
    git clone -q --depth 1 --branch b11669 https://github.com/ggml-org/llama.cpp $L 2>/dev/null || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp $L
    python3 -m pip install -q --break-system-packages $L/gguf-py sentencepiece cmake >/dev/null 2>&1; hash -r
    cmake -S $L -B $L/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build $L/build -j 8 --target llama-quantize >/dev/null || echo BUILD_WARN
  fi
  N=qwen3-vl-$(echo $S | tr A-Z a-z)-krummholz-v3.1
  python3 $L/convert_hf_to_gguf.py "$O/merged" --outtype bf16 --outfile "$O/$N-bf16.gguf" || echo CONVERT_FAIL $S
  python3 $L/convert_hf_to_gguf.py "$O/merged" --mmproj --outtype f16 --outfile "$O/mmproj-$N-f16.gguf" || echo MMPROJ_FAIL $S
  [ -f "$O/$N-bf16.gguf" ] && $L/build/bin/llama-quantize "$O/$N-bf16.gguf" "$O/$N-Q4_K_M.gguf" Q4_K_M >/dev/null && echo QUANT_OK $S
  cp "$BASE/LICENSE" "$F/LICENSE-Qwen3-VL-Apache-2.0.txt" 2>/dev/null
  rm -rf "$O/merged"; (cd "$O" && sha256sum *.gguf > SHA256SUMS-gguf); cp "$O"/*.gguf "$O/SHA256SUMS-gguf" "$F"/
fi
echo V31_DONE $S
