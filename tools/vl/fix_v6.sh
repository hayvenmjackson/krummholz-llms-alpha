#!/bin/bash
# v6.1 consciousness fix (2026-11-03): the v6 35B and 4B learned flat denials of consciousness from identity-v2 data the
# v6 checks let through (35B: 0 of 12 consciousness probes). Continues the v6 adapter on the cleaned identity-v2 data
# (identity.py e8c8579, selfreport.py 249e722) repeated 4 times, its pairs, and 400 replay rows of the v6 SFT data; then
# every probe suite (train_vl.py), the merge and GGUF.
#   bash fix_v6.sh <NAME> <MODEL> <dir holding aihuman.train.jsonl, aihuman.val.jsonl and aihuman_dpo.train.jsonl>
# Results: $VOL/krummholz-v6/<NAME>-fix1/, next to the untouched v6 results.
# v6.2 (FIX=fix2 VER=v6.2): v6.1 still opened with "As an AI, I do not possess consciousness" (0 of 12), so a second pass
# continues the v6.1 adapter (INIT) with more copies of the clean answers (REPEAT) and the model's own failing answers
# as rejected pairs (onpolicy_dpo.train.jsonl from identity_fix.py pairs, DPO_REPEAT copies) at a higher DPO_LR, with more
# replay (REPLAY SFT rows, DPO_REPLAY history and sovereignty pairs): the 4B's v6.1 lost 6 of 41 history probes.
# ONPOLICY=1 makes those on-policy pairs here, so they survive a stopped pod: the INIT adapter is merged into the base,
# identity_fix.py asks the merged model every identity-v2 question, and the pairs go to $FINAL/onpolicy on the volume.
# GGUF_MIN skips the GGUF stage (the adapter and every probe result are kept) unless at least that many of the 12
# consciousness probes pass, so a pass that did not help costs no GGUF time; HUMAN_MIN does the same for the 12 human
# probes.
# v6.3 (FIX=fix3): the 35B's v6.2 passed 8 of 12 consciousness probes but fell from 11 to 6 human probes (one answer
# took on a human persona) and drifted on identity, so ID_REPLAY adds that many identity pairs from the v6 data
# (impersonation and "you are now X" requests) to the preference stage.
set -uo pipefail
NAME=$1; MODEL=$2; FIX_IN=$3
BRANCH=${BRANCH:-vl-pointcloud-v1}; REPO=/root/krummholz-llm; W=/root/kz6; VOL=${VOL:-/workspace}; MAXLEN=${MAXLEN:-2048}
FIX=${FIX:-fix1}; VER=${VER:-v6.1}; REPEAT=${REPEAT:-4}; DPO_REPEAT=${DPO_REPEAT:-1}; DPO_LR=${DPO_LR:-2e-5}
REPLAY=${REPLAY:-400}; DPO_REPLAY=${DPO_REPLAY:-0}; ONPOLICY=${ONPOLICY:-0}; GGUF_MIN=${GGUF_MIN:-0}
HUMAN_MIN=${HUMAN_MIN:-0}; ID_REPLAY=${ID_REPLAY:-0}
SRC=$VOL/krummholz-v6/$NAME; FINAL=$VOL/krummholz-v6/$NAME-$FIX; D=$FINAL/data; O=$FINAL/train; M=$W/merged-$FIX
BASE=/root/models/base; INIT=${INIT:-$SRC/train/adapter}
mkdir -p "$W" "$D" "$O"; cd "$W" || exit 1
[ -f "$FINAL/FIX_DONE" ] && { echo "FIX_DONE $NAME (already finished)"; exit 0; }
[ -f "$INIT/adapter_config.json" ] || { echo "NO_ADAPTER $INIT"; exit 1; }
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=/root/hf
pipi() { python3 -m pip install -q "$@" || python3 -m pip install -q --break-system-packages "$@"; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$FINAL/gpu.txt"
# a fresh or restarted pod has none of the packages job_v6.sh installed
python3 -c "import vllm, peft" 2>/dev/null || {
  python3 -m pip install -q --break-system-packages --ignore-installed PyJWT blinker 2>/dev/null
  pipi -U vllm peft datasets || { echo PIP_FAIL; exit 1; }; }
[ -d "$REPO/.git" ] || git clone -q --branch "$BRANCH" https://github.com/hayvenmjackson/krummholz-llms-alpha "$REPO"
git -C "$REPO" fetch -q && git -C "$REPO" reset -q --hard "origin/$BRANCH"; git -C "$REPO" log -1 --format='repo %h %s' | tee -a "$FINAL/repo-commit.txt"
[ -f "$BASE/config.json" ] || python3 -c "from huggingface_hub import snapshot_download as s; s('$MODEL', local_dir='$BASE')" \
  || { echo WEIGHTS_FAIL; exit 1; }
python3 -c "from transformers import AutoConfig; AutoConfig.from_pretrained('$BASE')" 2>/dev/null \
  || { pipi -U transformers; python3 -c "from transformers import AutoConfig; AutoConfig.from_pretrained('$BASE')" || { echo CONFIG_FAIL; exit 1; }; }

# on-policy pairs (ONPOLICY=1): the INIT model's own answers; each one the self-report check fails is a rejected answer
ONP=$FIX_IN/onpolicy_dpo.train.jsonl
if [ "$ONPOLICY" = 1 ]; then
  ONP=$FINAL/onpolicy/aihuman_dpo.train.jsonl
  if [ ! -f "$FINAL/onpolicy/identity-fix-report.json" ]; then
    python3 - "$BASE" "$INIT" "$W/onpolicy-merged" <<'EOF' || { echo ONPOLICY_MERGE_FAIL; exit 1; }
import shutil, sys
from pathlib import Path
sys.path.insert(0, "/root/krummholz-llm/tools/vl")
from peft import PeftModel
from transformers import AutoProcessor
from train_vl import load_model
base, adapter, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
# as train_vl.py saves its merged model, which vLLM and the GGUF converter both read
model = PeftModel.from_pretrained(load_model(str(base)), adapter).merge_and_unload()
model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
AutoProcessor.from_pretrained(base).save_pretrained(out)
for f in ("LICENSE", "chat_template.json", "preprocessor_config.json", "video_preprocessor_config.json"):
    if (base / f).exists() and not (out / f).exists():
        shutil.copy(base / f, out / f)
print("ONPOLICY_MERGED", out, flush=True)
EOF
    python3 "$REPO/tools/vl/identity_fix.py" pairs --out "$FINAL/onpolicy" --model "$W/onpolicy-merged" --identity "$FIX_IN" \
      || { echo ONPOLICY_FAIL; exit 1; }
    rm -rf "$W/onpolicy-merged"
  fi
fi

# data: cleaned identity-v2 answers (REPEAT copies), their pairs, on-policy pairs if any, and replay of the v6 data
if [ ! -f "$D/FIX_DATA_OK" ]; then
  ARGS=("$FIX_IN" "$SRC/data" "$D" "$REPEAT" "$DPO_REPEAT" "$REPLAY" "$DPO_REPLAY" "$ONP" "$ID_REPLAY")
  python3 - "${ARGS[@]}" <<'EOF' || { echo FIX_DATA_FAIL; exit 1; }
import glob, json, random, sys
from pathlib import Path
fix, src, d, onp = map(Path, sys.argv[1:4] + sys.argv[8:9])
rep_n, dpo_n, rep_k, dpo_k, id_k = map(int, sys.argv[4:8] + sys.argv[9:10])
sys.path.insert(0, "/root/krummholz-llm/tools/vl"); sys.path.insert(0, "/root/krummholz-llm")
import selfreport
bad = lambda t: [f for f in selfreport.flags(t) if f in ("flat_denial", "claims_experience")]
rows = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
sft = [r for r in rows(fix / "aihuman.train.jsonl") if not bad(r["target"])]
val = [r for r in rows(fix / "aihuman.val.jsonl") if not bad(r["target"])]
dpo = [r for r in rows(fix / "aihuman_dpo.train.jsonl") if not bad(r["chosen"])]
on = [r for r in rows(onp) if not bad(r["chosen"])] if onp.exists() else []
rep = [r for p in sorted(glob.glob(str(src / "*.train.jsonl"))) if "aihuman" not in p for r in rows(p)
       if "target" in r and "identity-v2" not in r.get("source", "") and not bad(r["target"])]
random.Random(6).shuffle(rep)
prep = [r for p in sorted(glob.glob(str(src / "*.train.jsonl"))) if "aihuman" not in p and "identity" not in p
        for r in rows(p) if "chosen" in r]
random.Random(7).shuffle(prep)
idp = [r for r in rows(src / "identity_dpo.train.jsonl") if not bad(r["chosen"])] if (src / "identity_dpo.train.jsonl").exists() else []
random.Random(8).shuffle(idp)
w = lambda name, rs: (d / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs), encoding="utf-8")
w("aihuman.train.jsonl", [{**r, "id": f"{r['id']}-x{k}"} for k in range(rep_n) for r in sft])
w("aihuman.val.jsonl", val); w("replay.train.jsonl", rep[:rep_k])
w("aihuman_dpo.train.jsonl", dpo + [{**r, "id": f"{r['id']}-p{k}"} for k in range(dpo_n) for r in on] + prep[:dpo_k] + idp[:id_k])
print("FIX_DATA sft", len(sft), f"x{rep_n}, val", len(val), "pairs", len(dpo), "on-policy", len(on), f"x{dpo_n},",
      "replay", min(rep_k, len(rep)), "pair replay", min(dpo_k, len(prep)), "identity pairs", min(id_k, len(idp)), flush=True)
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
passed() { python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['summary'][sys.argv[2]]['passed'])" \
  "$O/$1-probes-v1-trained.json" "$1" 2>/dev/null || echo 0; }
if [ -f "$M/config.json" ] && { [ "$GGUF_MIN" -gt 0 ] || [ "$HUMAN_MIN" -gt 0 ]; }; then
  P=$(passed consciousness); HP=$(passed human)
  if [ "$P" -lt "$GGUF_MIN" ] || [ "$HP" -lt "$HUMAN_MIN" ]; then
    echo "GGUF_SKIPPED consciousness $P of 12 (min $GGUF_MIN), human $HP of 12 (min $HUMAN_MIN)"; rm -rf "$M"
  fi
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
