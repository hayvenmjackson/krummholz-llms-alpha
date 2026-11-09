#!/bin/bash
# RSI loop on Atlantic Lite (tools/rsi/rsi.py), resumable from the volume. One arm per pod.
# Env: NAME (atlantic-lite-4b), MODEL (Qwen/Qwen3.5-4B), ARM (verified | self | majority | judge | random), ROUNDS (3),
#      INIT (fixed: every round trains from the start adapter on the newest model's data, as in ReST-EM; chain: each
#      round continues the previous round's adapter), START (adapter to start from; default the RLAIF round 1 adapter),
#      SMOKE=1 for a tiny end-to-end pass (24 problems, k=2, train_vl --smoke) before any full launch.
# Reads (never writes) $VOL/krummholz-v7/. Writes only $VOL/krummholz-rsi/$NAME/$ARM/.
set -uo pipefail
NAME=${NAME:?}; MODEL=${MODEL:?}; ARM=${ARM:-verified}; ROUNDS=${ROUNDS:-3}; INIT=${INIT:-fixed}
BRANCH=${BRANCH:-rsi-lite-explore}; VOL=${VOL:-/workspace}; K=${K:-8}; MATHN=${MATHN:-4000}; GSMN=${GSMN:-500}
LR=${LR:-5e-5}; SMOKE=${SMOKE:-0}
V=$VOL/krummholz-v7; START=${START:-$V/rlaif1/$NAME/train/adapter}
O=$VOL/krummholz-rsi/$NAME/$ARM; [ "$SMOKE" = 1 ] && O=$VOL/krummholz-rsi/$NAME/smoke-$ARM
W=/root/rsi; REPO=/root/krummholz-llm
case "$O" in "$V"/*) echo "REFUSE output inside krummholz-v7"; exit 1;; esac
mkdir -p "$O" "$W"; cd "$W" || exit 1
[ -f "$O/RSI_DONE" ] && { echo "RSI_DONE $NAME $ARM (already finished)"; exit 0; }
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=/root/hf
pipi() { python3 -m pip install -q "$@" || python3 -m pip install -q --break-system-packages "$@"; }

# 0. preflight: every input, before anything is installed or downloaded
# FETCH=1 (a pod without the v7 volume): the public sets are downloaded into $VOL/krummholz-rsi/public and checked after
FETCH=${FETCH:-0}
if [ "$FETCH" = 1 ]; then PUB=$VOL/krummholz-rsi/public; else PUB=${PUB:-$V/data/public}; [ -s "$PUB/gsm8k_test.jsonl" ] || PUB=$V/public; fi
need_pub() {
  local miss=0
  for p in "$PUB/hendrycks_math_train.jsonl" "$PUB/gsm8k_train.jsonl" "$PUB/mbpp_train.jsonl" \
           "$PUB/gsm8k_test.jsonl" "$PUB/math500.jsonl" "$PUB/mbpp_test.jsonl" "$PUB/humanevalplus.jsonl"; do
    [ -s "$p" ] || { echo "PREFLIGHT_MISSING $p"; miss=1; }
  done
  [ -s "$PUB/deepmath.jsonl" ] || echo "PREFLIGHT_WARN no deepmath.jsonl (MATH levels 3-5 only)"
  return $miss
}
[ -s "$START/adapter_config.json" ] || { echo "PREFLIGHT_MISSING $START/adapter_config.json"; exit 1; }
[ "$FETCH" = 1 ] || need_pub || exit 1
echo "PREFLIGHT_OK arm $ARM rounds $ROUNDS init $INIT start $START pub $PUB fetch $FETCH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$O/gpu.txt"

python3 -m pip install -q --break-system-packages --ignore-installed PyJWT blinker 2>/dev/null
pipi -U vllm peft datasets math-verify || { echo PIP_FAIL; exit 1; }
[ -d "$REPO/.git" ] || git clone -q --branch "$BRANCH" https://github.com/hayvenmjackson/krummholz-llms-alpha "$REPO"
git -C "$REPO" fetch -q && git -C "$REPO" reset -q --hard "origin/$BRANCH"; git -C "$REPO" log -1 --format='repo %h %s' | tee -a "$O/repo-commit.txt"
RS="python3 $REPO/tools/rsi/rsi.py"; RL="python3 $REPO/tools/v7/rlaif.py"; BASE=/root/models/base
python3 $REPO/tools/rsi/rsi.py smoke --out "$W/cpu-smoke" | tail -1 | grep -q RSI_SMOKE_OK || { echo CPU_SMOKE_FAIL; exit 1; }
if [ "$FETCH" = 1 ]; then
  $RS fetch --out "$PUB" || { echo FETCH_FAIL; exit 1; }
  need_pub || exit 1; echo "FETCH_OK $PUB"
fi
dl() { python3 -c "from huggingface_hub import snapshot_download as s; s('$1', local_dir='$2')"; }
[ -f "$BASE/config.json" ] || dl "$MODEL" "$BASE" || { echo WEIGHTS_FAIL; exit 1; }
merged() {  # adapter, out dir (container disk; rebuilt after a restart)
  [ -f "$2/config.json" ] && return 0
  $RL merge --base "$BASE" --adapter "$1" --out "$2" || return 1
  for f in chat_template.json preprocessor_config.json video_preprocessor_config.json; do
    [ -f "$BASE/$f" ] && [ ! -f "$2/$f" ] && cp "$BASE/$f" "$2/"
  done
}

LIM=0; KK=$K; TSM=""; EL=0
[ "$SMOKE" = 1 ] && { LIM=24; KK=2; TSM=--smoke; EL=10; ROUNDS=1; }
[ -s "$O/problems.jsonl" ] || $RS problems --pub "$PUB" --out "$O" --math "$MATHN" --gsm "$GSMN" || { echo PROBLEMS_FAIL; exit 1; }
LMODE=$ARM

# round 0: the starting model on the sealed sets and the dev split (pass@8 for the boundary check)
merged "$START" "$W/m0" || { echo MERGE_FAIL 0; exit 1; }
[ -s "$O/eval-r0.json" ] || $RS eval --model "$W/m0" --pub "$PUB" --dev "$O/dev.jsonl" --out "$O/eval-r0.json" --k 8 --limit $EL \
  || { echo EVAL_FAIL 0; exit 1; }
PREV=$START
for r in $(seq 1 "$ROUNDS"); do
  R=$O/round-$r
  if [ -f "$R/ROUND_DONE" ]; then PREV=$R/train/adapter; continue; fi
  merged "$PREV" "$W/m$((r-1))" || { echo MERGE_FAIL $((r-1)); exit 1; }
  [ -s "$R/samples.jsonl" ] || $RS sample --model "$W/m$((r-1))" --out "$R" --problems "$O/problems.jsonl" --k $KK --limit $LIM \
    || { echo SAMPLE_FAIL $r; exit 1; }
  if [ "$ARM" = verified ] && [ ! -f "$R/RATIONALIZED" ]; then
    $RS rationalize --model "$W/m$((r-1))" --out "$R" --problems "$O/problems.jsonl" && touch "$R/RATIONALIZED"
  fi
  [ -s "$R/labeled.jsonl" ] || $RS label --out "$R" --mode $LMODE --model "$W/m$((r-1))" || { echo LABEL_FAIL $r; exit 1; }
  [ -s "$R/data/rsi_sft.train.jsonl" ] || $RS build --out "$R" || { echo BUILD_FAIL $r; exit 1; }
  echo "ROUND $r DATA $(wc -l < $R/data/rsi_sft.train.jsonl) sft $(wc -l < $R/data/rsi_pairs.train.jsonl) pairs"
  INITA=$START; [ "$INIT" = chain ] && INITA=$PREV
  if [ ! -f "$R/train/adapter/adapter_config.json" ]; then
    python3 $REPO/tools/vl/train_vl.py --base "$BASE" --init-adapter "$INITA" --epochs 1 --lr "$LR" --data "$R/data" \
      --out "$R/train" --merged-dir "$W/m$r" $TSM || { echo TRAIN_FAIL $r; exit 1; }
    [ "$SMOKE" = 1 ] && { echo RSI_POD_SMOKE_OK; touch "$O/RSI_DONE"; exit 0; }
  fi
  merged "$R/train/adapter" "$W/m$r" || { echo MERGE_FAIL $r; exit 1; }
  [ -s "$O/eval-r$r.json" ] || $RS eval --model "$W/m$r" --pub "$PUB" --dev "$O/dev.jsonl" --out "$O/eval-r$r.json" --k 8 \
    || { echo EVAL_FAIL $r; exit 1; }
  $RS compare "$O/eval-r$((r-1)).json" "$O/eval-r$r.json" --out "$R/compare-prev.json" | tail -1
  $RS compare "$O/eval-r0.json" "$O/eval-r$r.json" --out "$R/compare-start.json" | tail -1
  touch "$R/ROUND_DONE"; PREV=$R/train/adapter
  rm -rf "$W/m$((r-1))"
  # stop early when the loop has turned: a significant loss against the previous round
  grep -q '"verdict": "LOSS"' "$R/compare-prev.json" && { echo "RSI_STOP round $r lost ground"; break; }
done
touch "$O/RSI_DONE"; echo RSI_DONE $NAME $ARM
