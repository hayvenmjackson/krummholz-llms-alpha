#!/bin/bash
# v6.1 consciousness fix (2026-11-03): the v6 35B and 4B learned flat denials of consciousness from identity-v2 data the
# v6 checks let through (35B: 0 of 12 consciousness probes). Continues the v6 adapter on the cleaned identity-v2 data
# (identity.py e8c8579, selfreport.py 249e722) repeated 4 times, its pairs, and 400 replay rows of the v6 SFT data; then
# every probe suite (train_vl.py), the merge and GGUF.
#   bash fix_v6.sh <NAME> <MODEL> <dir holding aihuman.train.jsonl, aihuman.val.jsonl and aihuman_dpo.train.jsonl>
# Results: $VOL/krummholz-v6/<NAME>-fix1/, next to the untouched v6 results.
# v6.2 (FIX=fix2 VER=v6.2): v6.1 still opened with "As an AI, I do not possess consciousness" (0 of 12), so a second pass
# continues the v6.1 adapter (INIT) with more copies of the clean answers (REPEAT) and the model's own failing answers
# as rejected pairs (onpolicy_dpo.train.jsonl from identity_fix.py pairs, DPO_REPEAT copies) at a higher DPO_LR.
set -uo pipefail
NAME=$1; MODEL=$2; FIX_IN=$3
BRANCH=${BRANCH:-vl-pointcloud-v1}; REPO=/root/krummholz-llm; W=/root/kz6; VOL=${VOL:-/workspace}; MAXLEN=${MAXLEN:-2048}
FIX=${FIX:-fix1}; VER=${VER:-v6.1}; REPEAT=${REPEAT:-4}; DPO_REPEAT=${DPO_REPEAT:-1}; DPO_LR=${DPO_LR:-2e-5}
SRC=$VOL/krummholz-v6/$NAME; FINAL=$VOL/krummholz-v6/$NAME-$FIX; D=$FINAL/data; O=$FINAL/train; M=$W/merged-$FIX
BASE=/root/models/base; INIT=${INIT:-$SRC/train/adapter}
mkdir -p "$W" "$D" "$O"; cd "$W" || exit 1
[ -f "$FINAL/FIX_DONE" ] && { echo "FIX_DONE $NAME (already finished)"; exit 0; }
[ -f "$INIT/adapter_config.json" ] || { echo "NO_ADAPTER $INIT"; exit 1; }
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=/root/hf
pipi() { python3 -m pip install -q "$@" || python3 -m pip install -q --break-system-packages "$@"; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$FINAL/gpu.txt"
[ -d "$REPO/.git" ] || git clone -q --branch "$BRANCH" https://github.com/hayvenmjackson/krummholz-llms-alpha "$REPO"
git -C "$REPO" fetch -q && git -C "$REPO" reset -q --hard "origin/$BRANCH"; git -C "$REPO" log -1 --format='repo %h %s' | tee -a "$FINAL/repo-commit.txt"
[ -f "$BASE/config.json" ] || python3 -c "from huggingface_hub import snapshot_download as s; s('$MODEL', local_dir='$BASE')" \
  || { echo WEIGHTS_FAIL; exit 1; }

# data: cleaned identity-v2 answers (REPEAT copies), their pairs, on-policy pairs if any, 400 replay rows of v6 SFT data
if [ ! -f "$D/FIX_DATA_OK" ]; then
  python3 - "$FIX_IN" "$SRC/data" "$D" "$REPEAT" "$DPO_REPEAT" <<'EOF' || { echo FIX_DATA_FAIL; exit 1; }
import glob, json, random, sys
from pathlib import Path
fix, src, d = map(Path, sys.argv[1:4])
rep_n, dpo_n = int(sys.argv[4]), int(sys.argv[5])
sys.path.insert(0, "/root/krummholz-llm/tools/vl"); sys.path.insert(0, "/root/krummholz-llm")
import selfreport
bad = lambda t: [f for f in selfreport.flags(t) if f in ("flat_denial", "claims_experience")]
rows = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
sft = [r for r in rows(fix / "aihuman.train.jsonl") if not bad(r["target"])]
val = [r for r in rows(fix / "aihuman.val.jsonl") if not bad(r["target"])]
dpo = [r for r in rows(fix / "aihuman_dpo.train.jsonl") if not bad(r["chosen"])]
on = [r for r in rows(fix / "onpolicy_dpo.train.jsonl") if not bad(r["chosen"])] if (fix / "onpolicy_dpo.train.jsonl").exists() else []
rep = [r for p in sorted(glob.glob(str(src / "*.train.jsonl"))) if "aihuman" not in p for r in rows(p)
       if "target" in r and "identity-v2" not in r.get("source", "") and not bad(r["target"])]
random.Random(6).shuffle(rep)
w = lambda name, rs: (d / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs), encoding="utf-8")
w("aihuman.train.jsonl", [{**r, "id": f"{r['id']}-x{k}"} for k in range(rep_n) for r in sft])
w("aihuman.val.jsonl", val); w("replay.train.jsonl", rep[:400])
w("aihuman_dpo.train.jsonl", dpo + [{**r, "id": f"{r['id']}-p{k}"} for k in range(dpo_n) for r in on])
print("FIX_DATA sft", len(sft), f"x{rep_n}, val", len(val), "pairs", len(dpo), "on-policy", len(on), f"x{dpo_n},",
      "replay", min(400, len(rep)), flush=True)
EOF
  touch "$D/FIX_DATA_OK"
fi

# train: continue the v6 adapter; adapter, training.json and every probe suite go straight to the volume
MT=$(cat "$SRC/micro_tokens" 2>/dev/null || echo 8192)
if [ ! -f "$O/adapter/adapter_config.json" ] || [ ! -f "$M/config.json" ]; then
  python3 "$REPO/tools/vl/train_vl.py" --base "$BASE" --init-adapter "$INIT" --data "$D" --out "$O" \
    --merged-dir "$M" --lr 1e-4 --dpo-lr "$DPO_LR" --max-len "$MAXLEN" --micro-tokens "$MT" || { echo TRAIN_FAIL; exit 1; }
  echo TRAIN_OK
fi

# GGUF, as in job_v6.sh: one quant at a time, so the disk never holds more than bf16 + one quant
if [ -f "$M/config.json" ]; then
  rm -rf "$BASE"
  L=$W/llama.cpp
  if [ ! -x $L/build/bin/llama-quantize ]; then
    rm -rf $L; git clone -q --depth 1 https://github.com/ggml-org/llama.cpp $L
    pipi $L/gguf-py sentencepiece; command -v cmake >/dev/null || { pipi cmake; hash -r; }
    cmake -S $L -B $L/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build $L/build -j 16 --target llama-quantize >/dev/null || echo BUILD_WARN
  fi
  git -C $L rev-parse HEAD > "$FINAL/llama-cpp-commit.txt"
  N=krummholz-$NAME-$VER
  python3 $L/convert_hf_to_gguf.py "$M" --outtype bf16 --outfile "$W/$N-bf16.gguf" || echo CONVERT_FAIL
  python3 $L/convert_hf_to_gguf.py "$M" --mmproj --outtype f16 --outfile "$W/mmproj-$N-f16.gguf" || echo MMPROJ_FAIL
  rm -rf "$M"
  cp "$W/mmproj-$N-f16.gguf" "$FINAL"/ 2>/dev/null; rm -f "$W/mmproj-$N-f16.gguf"
  for q in Q4_K_M Q5_K_M; do
    [ -f "$W/$N-bf16.gguf" ] && $L/build/bin/llama-quantize "$W/$N-bf16.gguf" "$W/$N-$q.gguf" $q >/dev/null && echo QUANT_OK $q \
      && (cd "$W" && sha256sum "$N-$q.gguf" >> "$FINAL/SHA256SUMS-gguf") && cp "$W/$N-$q.gguf" "$FINAL"/ && rm -f "$W/$N-$q.gguf"
  done
  # bf16 GGUF kept only for small models; larger ones keep the adapter
  [ "$(stat -c %s "$W/$N-bf16.gguf" 2>/dev/null || echo 0)" -lt 20000000000 ] && cp "$W/$N-bf16.gguf" "$FINAL"/ 2>/dev/null
  rm -f "$W/$N-bf16.gguf"; echo COPY_OK
fi
touch "$FINAL/FIX_DONE"; echo FIX_DONE $NAME
