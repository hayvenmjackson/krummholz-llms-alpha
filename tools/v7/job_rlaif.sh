#!/bin/bash
# RLAIF round 1 on one v7 model (tools/v7/rlaif.py), resumable from the volume. Env: NAME (atlantic-lite-4b), MODEL
# (Qwen/Qwen3.5-4B). Steps: preflight every input, merge the v7 adapter, a tiny smoke run of sample/judge/build, the full
# sample (policy) and judge (OLMo 3.1 32B, Apache 2.0), DPO pairs, DPO continuing the v7 adapter (train_vl.py --epochs 0,
# probes run there), the raw/guarded/gated eval for v7 and RLAIF in the same vLLM setup, then Q4_K_M and Q5_K_M GGUFs.
# Artifacts: $VOL/krummholz-v7/rlaif1/$NAME/. Needs /workspace/krummholz-v7/private/constitution.md (never in the repo).
set -uo pipefail
NAME=${NAME:?}; MODEL=${MODEL:?}; BRANCH=${BRANCH:-v7-runtime-gates}; VOL=${VOL:-/workspace}
V=$VOL/krummholz-v7; F=$V/$NAME; R=$V/rlaif1/$NAME; W=/root/rl; REPO=/root/krummholz-llm
GENERAL=${GENERAL:-2400}; HISTN=${HISTN:-1200}; K=${K:-4}
mkdir -p "$R" "$W"; cd "$W" || exit 1
[ -f "$R/RLAIF_DONE" ] && { echo "RLAIF_DONE $NAME (already finished)"; exit 0; }
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=/root/hf
pipi() { python3 -m pip install -q "$@" || python3 -m pip install -q --break-system-packages "$@"; }

# 0. preflight: every input the job touches, before anything is installed or downloaded
miss=0
for p in "$F/train/adapter/adapter_config.json" "$V/gen/krummholz_prompts.jsonl" "$V/private/constitution.md" \
         "$V/data/hist-sov/history_chat.train.jsonl" "$V/data/hist/history_chat.train.jsonl" "$V/data/hist-sov/history_chat.val.jsonl"; do
  [ -s "$p" ] || { echo "PREFLIGHT_MISSING $p"; miss=1; }
done
[ $miss = 0 ] || exit 1
echo "PREFLIGHT_OK prompts $(wc -l < $V/gen/krummholz_prompts.jsonl) hist $(cat $V/data/hist-sov/history_chat.train.jsonl $V/data/hist/history_chat.train.jsonl | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$R/gpu.txt"

python3 -m pip install -q --break-system-packages --ignore-installed PyJWT blinker 2>/dev/null
pipi -U vllm peft datasets || { echo PIP_FAIL; exit 1; }
[ -d "$REPO/.git" ] || git clone -q --branch "$BRANCH" https://github.com/hayvenmjackson/krummholz-llms-alpha "$REPO"
git -C "$REPO" fetch -q && git -C "$REPO" reset -q --hard "origin/$BRANCH"; git -C "$REPO" log -1 --format='repo %h %s' | tee -a "$R/repo-commit.txt"
RL="python3 $REPO/tools/v7/rlaif.py"; BASE=/root/models/base; VM=$W/v7merged; RM=$W/rl-merged
dl() { python3 -c "from huggingface_hub import snapshot_download as s; s('$1', local_dir='$2')"; }
[ -f "$BASE/config.json" ] || dl "$MODEL" "$BASE" || { echo WEIGHTS_FAIL; exit 1; }
[ -f "$VM/config.json" ] || $RL merge --base "$BASE" --adapter "$F/train/adapter" --out "$VM" || { echo MERGE_FAIL; exit 1; }
for f in chat_template.json preprocessor_config.json video_preprocessor_config.json; do
  [ -f "$BASE/$f" ] && [ ! -f "$VM/$f" ] && cp "$BASE/$f" "$VM/"
done
HIST="$V/data/hist-sov $V/data/hist"

# 1. smoke: the whole data path on a few prompts (both models load, judge parses, pairs build)
if [ ! -f "$R/smoke/rlaif1-report.json" ]; then
  $RL sample --model "$VM" --prompts "$V/gen/krummholz_prompts.jsonl" --hist $HIST --out "$R/smoke" --general 8 --hist-n 6 --k 2 \
    && $RL judge --out "$R/smoke" --constitution "$V/private/constitution.md" \
    && $RL build --out "$R/smoke" --policy "v7-$NAME" || { echo SMOKE_FAIL; exit 1; }
fi
echo SMOKE_OK

# 2. full sample, judge, pairs
[ -s "$R/samples.jsonl" ] || $RL sample --model "$VM" --prompts "$V/gen/krummholz_prompts.jsonl" --hist $HIST --out "$R" \
  --general "$GENERAL" --hist-n "$HISTN" --k "$K" || { echo SAMPLE_FAIL; exit 1; }
echo SAMPLE_OK
[ -s "$R/judged.jsonl" ] || $RL judge --out "$R" --constitution "$V/private/constitution.md" || { echo JUDGE_FAIL; exit 1; }
echo JUDGE_OK
[ -s "$R/pairs/rlaif1_pairs.train.jsonl" ] || $RL build --out "$R" --policy "v7-$NAME" || { echo BUILD_FAIL; exit 1; }
echo "PAIRS_OK $(wc -l < $R/pairs/rlaif1_pairs.train.jsonl)"
cp "$V/data/hist-sov/history_chat.val.jsonl" "$R/pairs/"   # SFT val loss needs a few plain examples
rm -rf /root/hf/hub/models--allenai--Olmo*

# 3. DPO continuing the v7 adapter (no SFT epoch); train_vl.py runs every probe suite on the result
if [ ! -f "$R/train/adapter/adapter_config.json" ] || [ ! -f "$RM/config.json" ]; then
  python3 $REPO/tools/vl/train_vl.py --base "$BASE" --init-adapter "$F/train/adapter" --epochs 0 --data "$R/pairs" \
    --out "$R/train" --merged-dir "$RM" || { echo TRAIN_FAIL; exit 1; }
fi
echo TRAIN_OK

# 4. raw / guarded / gated eval, v7 and RLAIF side by side in the same vLLM setup
serve_eval() {  # model dir, label
  [ -f "$R/gate-eval-$2.json" ] && return 0
  python3 -m vllm.entrypoints.openai.api_server --model "$1" --port 8001 --max-model-len 8192 --gpu-memory-utilization 0.85 \
    --served-model-name m > "$W/vllm-$2.log" 2>&1 &
  local pid=$!
  for i in $(seq 120); do curl -s localhost:8001/health >/dev/null && break; sleep 5; done
  python3 $REPO/tools/v7/gate_eval.py --url http://localhost:8001 --name "$NAME-$2" --out "$R/gate-eval-$2.json" || echo "GATE_EVAL_FAIL $2"
  kill $pid; wait $pid 2>/dev/null
}
serve_eval "$VM" v7
serve_eval "$RM" rlaif1
echo EVAL_OK

# 5. GGUF of the RLAIF model
L=$W/llama.cpp
if [ ! -x $L/build/bin/llama-quantize ]; then
  git clone -q --depth 1 https://github.com/ggml-org/llama.cpp $L
  pipi $L/gguf-py sentencepiece; command -v cmake >/dev/null || { pipi cmake; hash -r; }
  cmake -S $L -B $L/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build $L/build -j 16 --target llama-quantize >/dev/null || echo BUILD_WARN
fi
N=krummholz-$NAME-v7-rlaif1
python3 $L/convert_hf_to_gguf.py "$RM" --outtype bf16 --outfile "$W/$N-bf16.gguf" || echo CONVERT_FAIL
for q in Q4_K_M Q5_K_M; do
  [ -f "$W/$N-bf16.gguf" ] && $L/build/bin/llama-quantize "$W/$N-bf16.gguf" "$W/$N-$q.gguf" $q >/dev/null && echo QUANT_OK $q \
    && (cd "$W" && sha256sum "$N-$q.gguf" >> "$R/SHA256SUMS-gguf") && cp "$W/$N-$q.gguf" "$R"/ && rm -f "$W/$N-$q.gguf"
done
touch "$R/RLAIF_DONE"; echo RLAIF_DONE $NAME
