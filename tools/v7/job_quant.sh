#!/bin/bash
# More GGUF sizes of a finished RLAIF model, for workstations and consumer hardware (job_rlaif.sh makes Q4_K_M and Q5_K_M).
# Env: NAME (atlantic-27b), MODEL (Qwen/Qwen3.8-27B), QUANTS (default below, most useful first). Rebuilds the merged model
# from the RLAIF adapter (the complete adapter, applied alone to the base), converts it to a bf16 GGUF, quantizes each size
# onto the volume next to the others and appends to SHA256SUMS-gguf. Also builds the vision projector if no v7 one exists.
# A size already in SHA256SUMS-gguf is skipped; a size that would not fit on the volume is skipped and reported.
set -uo pipefail
NAME=${NAME:?}; MODEL=${MODEL:?}; BRANCH=${BRANCH:-v7-runtime-gates}; VOL=${VOL:-/workspace}
V=$VOL/krummholz-v7; R=$V/rlaif1/$NAME; W=/root/rl; REPO=/root/krummholz-llm
QUANTS=${QUANTS:-Q3_K_M IQ4_XS Q8_0 Q6_K}
N=krummholz-$NAME-v7-rlaif1
mkdir -p "$W"; cd "$W" || exit 1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=/root/hf
pipi() { python3 -m pip install -q "$@" || python3 -m pip install -q --break-system-packages "$@"; }

# 0. preflight
[ -f "$R/RLAIF_DONE" ] && [ -s "$R/train/adapter/adapter_config.json" ] || { echo "PREFLIGHT_MISSING $R/RLAIF_DONE or adapter"; exit 1; }
echo "PREFLIGHT_OK quants: $QUANTS; volume free $(df -BG --output=avail "$VOL" | tail -1 | tr -d ' ')"
todo=""
for q in $QUANTS; do grep -q "$N-$q.gguf" "$R/SHA256SUMS-gguf" 2>/dev/null && echo "QUANT_HAVE $q" || todo="$todo $q"; done
[ -n "$todo" ] || { echo QUANT_DONE nothing to do; exit 0; }

python3 -m pip install -q --break-system-packages --ignore-installed PyJWT blinker 2>/dev/null
pipi -U transformers peft accelerate || { echo PIP_FAIL; exit 1; }
[ -d "$REPO/.git" ] || git clone -q --branch "$BRANCH" https://github.com/hayvenmjackson/krummholz-llms-alpha "$REPO"
git -C "$REPO" log -1 --format='repo %h %s'
BASE=/root/models/base; RM=$W/rl-merged
[ -f "$RM/config.json" ] || {
  python3 -c "from huggingface_hub import snapshot_download as s; s('$MODEL', local_dir='$BASE')" || { echo WEIGHTS_FAIL; exit 1; }
  python3 $REPO/tools/v7/rlaif.py merge --base "$BASE" --adapter "$R/train/adapter" --out "$RM" || { echo MERGE_FAIL; exit 1; }
  for f in chat_template.json preprocessor_config.json video_preprocessor_config.json; do
    [ -f "$BASE/$f" ] && [ ! -f "$RM/$f" ] && cp "$BASE/$f" "$RM/"
  done
  rm -rf "$BASE"
}
echo MERGE_OK

L=$W/llama.cpp
if [ ! -x $L/build/bin/llama-quantize ]; then
  git clone -q --depth 1 https://github.com/ggml-org/llama.cpp $L
  pipi $L/gguf-py sentencepiece; command -v cmake >/dev/null || { pipi cmake; hash -r; }
  cmake -S $L -B $L/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build $L/build -j 16 --target llama-quantize >/dev/null || { echo BUILD_FAIL; exit 1; }
fi
if ! ls "$R"/mmproj-*.gguf "$V/$NAME"/mmproj-*v7*.gguf >/dev/null 2>&1; then
  python3 $L/convert_hf_to_gguf.py "$RM" --mmproj --outtype f16 --outfile "$R/mmproj-krummholz-$NAME-v7-f16.gguf" \
    && echo MMPROJ_OK || echo "MMPROJ_FAIL (not needed for text)"
fi
[ -f "$W/$N-bf16.gguf" ] || python3 $L/convert_hf_to_gguf.py "$RM" --outtype bf16 --outfile "$W/$N-bf16.gguf" || { echo CONVERT_FAIL; exit 1; }
rm -rf "$RM"
echo CONVERT_OK

declare -A GB=([Q3_K_M]=14 [IQ4_XS]=16 [Q4_K_S]=16 [Q6_K]=23 [Q8_0]=30)   # 27B sizes, rounded up
for q in $todo; do
  free=$(df -BG --output=avail "$VOL" | tail -1 | tr -dc 0-9)
  need=$(( ${GB[$q]:-30} + 5 ))
  [ "$free" -ge "$need" ] || { echo "QUANT_SKIP_SPACE $q needs ${need}G, volume has ${free}G"; continue; }
  $L/build/bin/llama-quantize "$W/$N-bf16.gguf" "$W/$N-$q.gguf" $q >/dev/null \
    && (cd "$W" && sha256sum "$N-$q.gguf" >> "$R/SHA256SUMS-gguf") && cp "$W/$N-$q.gguf" "$R"/ && rm -f "$W/$N-$q.gguf" \
    && echo "QUANT_OK $q $(du -h "$R/$N-$q.gguf" | cut -f1)" || echo "QUANT_FAIL $q"
done
echo QUANT_DONE $NAME
