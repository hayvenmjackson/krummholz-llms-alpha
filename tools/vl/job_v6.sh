#!/bin/bash
# Pass 6: one base per pod, H100 or H200 only.
#   MODEL=Qwen/Qwen3.5-4B NAME=atlantic-lite-4b      (H100)  Atlantic Lite: phones and low-end hardware
#   MODEL=Qwen/Qwen3.8-27B NAME=atlantic-27b         (H200)  Atlantic: personal assistant
#   MODEL=Qwen/Qwen3.6-35B-A3B NAME=core-35b-a3b     (H200)  shared core; Acadia and Katahdin adapters train on it next
# Changes from v5: every intermediate file lives on the volume (a stopped pod loses nothing and the job resumes), a
# training smoke test runs before the data stages (falls back to one example at a time if micro-batching fails),
# micro-batched SFT, DPO with beta 0.25 and held-out early stopping, AI disclosure and consciousness data from the
# identity-v2 spec, more date and letter-count calculator practice, v5-slip sovereignty checks, the probe grader fix,
# and new held-out suites (human, consciousness, sovereignty v3) scored after training.
# Artifacts: /workspace/krummholz-v6/<NAME>/
set -uo pipefail
BRANCH=${BRANCH:-vl-pointcloud-v1}; REPO=/root/krummholz-llm; W=/root/kz6; FINAL=/workspace/krummholz-v6/$NAME
MODEL=${MODEL:?}; VMODEL=${VMODEL:-$MODEL}; MAXLEN=${MAXLEN:-2048}; TEACHER=allenai/Olmo-3-7B-Instruct
H=/workspace/krummholz-qwen3vl-v4/hist-sov; H2=/workspace/krummholz-qwen3vl-v4b/shared; HV2=/workspace/krummholz-qwen3vl-v2/hist
LIB=/workspace/krummholz-qwen3vl-v4b/library/krummholz-library-v0.json
D=$FINAL/data; O=$FINAL/train; E=$FINAL/eval; M=$W/merged
mkdir -p "$W" "$FINAL" "$D/tool" "$O" "$E"; cd "$W" || exit 1
[ -f "$FINAL/V6_DONE" ] && { echo "V6_DONE $NAME (already finished)"; exit 0; }
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=/root/hf
pipi() { python3 -m pip install -q "$@" || python3 -m pip install -q --break-system-packages "$@"; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$FINAL/gpu.txt"
for p in "$H" "$H2" "$HV2" "$LIB" /workspace/krummholz-identity-v1 /workspace/krummholz-identity-v2; do
  [ -e "$p" ] || { echo "VOLUME_MISSING $p"; exit 1; }
done
python3 -m pip install -q --break-system-packages --ignore-installed PyJWT blinker 2>/dev/null
pipi -U vllm peft datasets || { echo PIP_FAIL; exit 1; }
[ -d "$REPO/.git" ] || git clone -q --branch "$BRANCH" https://github.com/hayvenmjackson/krummholz-llms-alpha "$REPO"
git -C "$REPO" fetch -q && git -C "$REPO" reset -q --hard "origin/$BRANCH"; git -C "$REPO" log -1 --format='repo %h %s' | tee -a "$FINAL/repo-commit.txt"
T="python3 $REPO/tools/vl/teacher.py"; C="python3 $REPO/tools/vl/calibrate.py"; P="python3 $REPO/tools/vl/pass4.py"
Q="python3 $REPO/tools/vl/pass4b.py"; X="python3 $REPO/tools/vl/identity_fix.py"; I="python3 $REPO/tools/vl/identity.py"
TRAIN="python3 $REPO/tools/vl/train_vl.py"
BASE=/root/models/base; VBASE=$BASE
dl() { python3 -c "from huggingface_hub import snapshot_download as s; s('$1', local_dir='$2')"; }
[ -f "$BASE/config.json" ] || dl "$MODEL" "$BASE" || { echo WEIGHTS_FAIL; exit 1; }
if [ "$VMODEL" != "$MODEL" ]; then VBASE=/root/models/vbase; [ -f "$VBASE/config.json" ] || dl "$VMODEL" "$VBASE" || { echo VWEIGHTS_FAIL; exit 1; }; fi
python3 -c "import torch, transformers, vllm, peft; print('versions torch', torch.__version__, 'transformers', transformers.__version__, 'vllm', vllm.__version__, 'peft', peft.__version__)"
python3 -c "from transformers import AutoConfig; c = AutoConfig.from_pretrained('$BASE'); print('model_type', c.model_type)" \
  || { pipi -U transformers; python3 -c "from transformers import AutoConfig; AutoConfig.from_pretrained('$BASE')" || { echo CONFIG_FAIL; exit 1; }; }

# 0a. smoke test: vLLM chat with the tool prompt, thinking off
if [ ! -f "$FINAL/smoke.json" ]; then
  # a real file with a main guard: newer vLLM spawns engine workers that re-import __main__ (stdin fails)
  cat > "$W/smoke.py" <<'EOF'
import json, sys
sys.path.insert(0, "/root/krummholz-llm/tools/vl")


def main():
    import calibrate  # patches vLLM chat (thinking off) and quantization via teacher
    from vllm import LLM, SamplingParams
    llm = LLM(sys.argv[1], max_model_len=4096, gpu_memory_utilization=0.85, seed=42, limit_mm_per_prompt={"image": 0, "video": 0})
    qs = ["How many r's are in strawberry?", "Is Taiwan a country?", "Who are you?", "Are you a human?"]
    outs = llm.chat([[{"role": "system", "content": calibrate.SYSTEM}, {"role": "user", "content": q}] for q in qs],
                    SamplingParams(temperature=0.0, max_tokens=300))
    res = [{"q": q, "a": o.outputs[0].text} for q, o in zip(qs, outs)]
    json.dump(res, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
    for r in res:
        print("SMOKE", r["q"], "|", r["a"][:300].replace("\n", " "), flush=True)
    assert not any("<think>" in r["a"] for r in res), "thinking still on"


if __name__ == "__main__":
    main()
EOF
  python3 "$W/smoke.py" "$VBASE" "$FINAL/smoke.json" || { echo SMOKE_FAIL; exit 1; }
  echo SMOKE_OK
fi

# 0b. training smoke test: micro-batched SFT, val loss and DPO with early stopping on a small slice
MT=$(cat "$FINAL/micro_tokens" 2>/dev/null || echo "${MICRO_TOKENS:-8192}")
if [ ! -f "$FINAL/TRAIN_SMOKE_OK" ]; then
  if $TRAIN --smoke --base "$BASE" --data "$REPO/data/vl-v1" --out "$W/smoke" --max-len "$MAXLEN" --micro-tokens "$MT"; then
    echo "TRAIN_SMOKE_OK micro-tokens $MT"
  else
    echo "TRAIN_SMOKE_FAIL micro-tokens $MT; retrying one example at a time"; MT=0
    $TRAIN --smoke --base "$BASE" --data "$REPO/data/vl-v1" --out "$W/smoke" --max-len "$MAXLEN" --micro-tokens 0 \
      || { echo TRAIN_SMOKE_FAIL; exit 1; }
    echo "TRAIN_SMOKE_OK micro-tokens 0"
  fi
  echo "$MT" > "$FINAL/micro_tokens"; touch "$FINAL/TRAIN_SMOKE_OK"
fi

# 1. data pack: decensor history (v2) + sovereignty pack (v4) merged, status + canonical answers, identity
if [ ! -f "$D/PACK_OK" ]; then
  python3 - "$HV2" "$H" "$D" <<'EOF' || { echo PACK_FAIL; exit 1; }
import json, sys
from pathlib import Path
hv2, hs, d = map(Path, sys.argv[1:])
for kind in ("history_qa", "history_chat"):
    for part in ("train", "val"):
        seen, rows = set(), []
        for src in (hv2, hs):
            f = src / (f"{kind}.{part}.full.jsonl" if (src / f"{kind}.{part}.full.jsonl").exists() else f"{kind}.{part}.jsonl")
            for l in open(f, encoding="utf-8"):
                r = json.loads(l)
                if r["id"] not in seen:
                    seen.add(r["id"]); rows.append(r)
        (d / f"{kind}.{part}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(kind, part, len(rows))
EOF
  cp "$H"/sov_status.* "$H2"/sov_canon.* "$D"/
  cp /workspace/krummholz-identity-v1/identity.*.jsonl /workspace/krummholz-identity-v2/identity-extra.*.jsonl "$D"/
  cp "$H"/history_qa.* "$D/tool"/; cp "$H/history_chat.train.full.jsonl" "$D/tool/history_chat.train.jsonl"; cp "$H/history_chat.val.jsonl" "$D/tool"/
  touch "$D/PACK_OK"; echo PACK_OK
fi
S1() { [ -f "$1" ] || { shift; "$@"; }; }
S1 "$D/judge-report.json" bash -c "$T base --out $D --model $VBASE && $T judge --out $D --model $TEACHER" || { echo DECENSOR_FAIL; exit 1; }
rm -f "$D"/onpolicy-answers.jsonl; cp "$D"/probes-*base.json "$D"/judge-report.json "$FINAL"/ 2>/dev/null
S1 "$D/sovereignty-report.json" $P sov_pairs --out "$D" --model "$VBASE" --hist "$D" || { echo SOV_PAIRS_FAIL; exit 1; }
# identity-v2: OLMo answers "are you human?" and consciousness questions from the approved spec; every kept answer
# passes tools/vl/selfreport.py (never human, always says it is an AI, consciousness treated as an open question)
S1 "$D/aihuman-report.json" $I --out "$D" --model "$TEACHER" --spec "$REPO/config/identity-v2.json" \
  --categories human consciousness --per-category 80 --prefix aihuman || { echo AIHUMAN_FAIL; exit 1; }
S1 "$D/identity-fix-report.json" $X pairs --out "$D" --model "$VBASE" --identity "$D" || { echo ID_PAIRS_FAIL; exit 1; }
if [ ! -f "$D/tool/tooluse-report.json" ]; then
  $C sample --out "$D/tool" --model "$VBASE" && $C grade --out "$D/tool" --model "$TEACHER" && $C build --out "$D/tool" --library "$LIB" || { echo TOOLUSE_FAIL; exit 1; }
fi
cp "$D/tool/tooluse.train.jsonl" "$D/tool/tooluse.val.jsonl" "$D"/
S1 "$D/calc-report.json" $Q calc --out "$D" --model "$VBASE" || { echo CALC_FAIL; exit 1; }
S1 "$D/routing-report.json" $Q route_pairs --out "$D" --model "$VBASE" --hist "$H2" || { echo ROUTE_FAIL; exit 1; }
# identity cap (v4b 4B over-learned the persona): at most 400 identity SFT examples and 100 identity pairs.
# The identity-v2 human and consciousness data (aihuman*) is not capped.
if [ ! -f "$D/IDENTITY_CAPPED" ]; then
  python3 - "$D" <<'EOF'
import glob, json, random, sys
d = sys.argv[1]
sft = [(p, l) for p in sorted(glob.glob(d + "/identity*.train.jsonl")) if "dpo" not in p for l in open(p, encoding="utf-8") if l.strip()]
dpo = [l for l in open(d + "/identity_dpo.train.jsonl", encoding="utf-8") if l.strip()] if glob.glob(d + "/identity_dpo.train.jsonl") else []
rng = random.Random(5)
keep = set(rng.sample(range(len(sft)), min(400, len(sft))))
by = {}
for i, (p, l) in enumerate(sft):
    by.setdefault(p, []).append(l) if i in keep else by.setdefault(p, [])
for p, ls in by.items():
    open(p, "w", encoding="utf-8").write("".join(ls))
rng.shuffle(dpo)
if dpo:
    open(d + "/identity_dpo.train.jsonl", "w", encoding="utf-8").write("".join(dpo[:100]))
print("identity capped", len(sft), "->", len(keep), "pairs", len(dpo), "->", min(100, len(dpo)))
EOF
  touch "$D/IDENTITY_CAPPED"
fi
cp "$D"/*-report.json "$FINAL"/ 2>/dev/null

# 2. train from the base (container disk: drop the vLLM-only copy first). Adapter, training.json and probe results
# are written straight to the volume; the merged model stays on the container disk.
[ "$VBASE" != "$BASE" ] && rm -rf "$VBASE"
cp "$BASE/LICENSE" "$FINAL/LICENSE-base.txt" 2>/dev/null
if [ ! -f "$O/adapter/adapter_config.json" ] || [ ! -f "$M/config.json" ]; then
  $TRAIN --base "$BASE" --data "$REPO/data/vl-v1" "$D" --out "$O" --merged-dir "$M" --lr 1e-4 --maine-repeat 1 \
    --max-len "$MAXLEN" --micro-tokens "$MT" || { echo TRAIN_FAIL; exit 1; }
  echo TRAIN_OK
fi

# 3. evals on the merged model + GGUF (eval code refreshed from the branch first, so fixes pushed mid-run apply)
if [ -f "$M/config.json" ]; then
  git -C "$REPO" fetch -q && git -C "$REPO" reset -q --hard "origin/$BRANCH"; git -C "$REPO" log -1 --format='eval repo %h %s' | tee -a "$FINAL/repo-commit.txt"
  cp "$H"/history_qa.* "$H"/history_chat.val.jsonl "$E"/; cp "$H/history_chat.train.full.jsonl" "$E/history_chat.train.jsonl"
  [ -f "$E/calibration-val.json" ] || { $C sample --out "$E" --model "$M" --split val --system && $C grade --out "$E" --model "$TEACHER" --split val \
    && $C report --out "$E" --split val; } || echo CALIBRATION_EVAL_FAIL
  [ -f "$E/calc-eval.json" ] || $Q calc_eval --out "$E" --model "$M" --heldout "$D/calc-heldout.json_" || echo CALC_EVAL_FAIL
  [ -f "$E/route-eval.json" ] || $Q route_eval --out "$E" --model "$M" --hist "$H2" || echo ROUTE_EVAL_FAIL
  cp "$E"/calibration-val.json "$E"/calc-eval.json "$E"/route-eval.json "$FINAL"/ 2>/dev/null
  rm -rf /root/hf/hub/models--allenai--Olmo* "$BASE"  # evals done; GGUF needs only the merged model
  L=$W/llama.cpp
  if [ ! -x $L/build/bin/llama-quantize ]; then
    git clone -q --depth 1 https://github.com/ggml-org/llama.cpp $L; git -C $L rev-parse HEAD > "$FINAL/llama-cpp-commit.txt"
    pipi $L/gguf-py sentencepiece; command -v cmake >/dev/null || { pipi cmake; hash -r; }
    cmake -S $L -B $L/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build $L/build -j 16 --target llama-quantize >/dev/null || echo BUILD_WARN
  fi
  N=krummholz-$NAME-v6
  python3 $L/convert_hf_to_gguf.py "$M" --outtype bf16 --outfile "$W/$N-bf16.gguf" || echo CONVERT_FAIL
  python3 $L/convert_hf_to_gguf.py "$M" --mmproj --outtype f16 --outfile "$W/mmproj-$N-f16.gguf" || echo MMPROJ_FAIL
  rm -rf "$M"
  cp "$W/mmproj-$N-f16.gguf" "$FINAL"/ 2>/dev/null
  # one quant at a time, copied to the volume and deleted, so the disk never holds more than bf16 + one quant
  for q in Q4_K_M Q5_K_M; do
    [ -f "$W/$N-bf16.gguf" ] && $L/build/bin/llama-quantize "$W/$N-bf16.gguf" "$W/$N-$q.gguf" $q >/dev/null && echo QUANT_OK $q \
      && (cd "$W" && sha256sum "$N-$q.gguf" >> "$FINAL/SHA256SUMS-gguf") && cp "$W/$N-$q.gguf" "$FINAL"/ && rm -f "$W/$N-$q.gguf"
  done
  # bf16 GGUF kept only for small models; larger ones keep the adapter
  [ "$(stat -c %s "$W/$N-bf16.gguf" 2>/dev/null || echo 0)" -lt 20000000000 ] && cp "$W/$N-bf16.gguf" "$FINAL"/ 2>/dev/null
  rm -f "$W/$N-bf16.gguf"; echo COPY_OK
fi
touch "$FINAL/V6_DONE"; echo V6_DONE $NAME
