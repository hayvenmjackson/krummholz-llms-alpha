#!/bin/bash
# v7 data job (one H200 on the krummholz-models volume). No training. Resumable: each stage skips when its report exists.
#   0. rebuild the v6 inputs lost with the old volume (history and sovereignty packs, canonical and time-sensitive
#      questions, tool library, identity-v3 data), with OLMo 3.1 32B as teacher
#   1. public datasets: download and subset, sealed eval hashes (fetch_public.py)
#   2. pushback pairs built by code (pushback.py)
#   3. constitutional SFT and pairs on the Krummholz prompt set (teach_v7.py); waits for the prompt generator
# Needs /workspace/krummholz-v7/private/constitution.md (Part 1 of the private v7 plan), uploaded to the pod, never
# committed. Artifacts: /workspace/krummholz-v7/data/. Log: /workspace/krummholz-v7/logs/data.log. Status: STATUS file.
set -uo pipefail
BRANCH=${BRANCH:-vl-pointcloud-v1}; REPO=/root/krummholz-llm; V=/workspace/krummholz-v7; D=$V/data
TEACHER=${TEACHER:-allenai/Olmo-3.1-32B-Instruct}
H=$D/hist-sov; H2=$D/shared; HV2=$D/hist; LIB=$D/library/krummholz-library-v0.json; ID=$D/identity-v3
PROMPTS=$V/gen/krummholz_prompts.jsonl; CONST=$V/private/constitution.md
mkdir -p "$D" "$V/logs" "$H" "$H2" "$HV2/v1" "$HV2/v2" "$(dirname $LIB)" "$ID"
st() { echo "$(date -u +%FT%TZ) $*" | tee -a "$V/STATUS"; }
export PYTHONUNBUFFERED=1 HF_HOME=/root/hf
pipi() { python3 -m pip install -q "$@" || python3 -m pip install -q --break-system-packages "$@"; }
st START data job
python3 -m pip install -q --break-system-packages --ignore-installed PyJWT blinker 2>/dev/null
python3 -c "import vllm" 2>/dev/null || pipi -U vllm datasets || { st PIP_FAIL; exit 1; }
[ -d "$REPO/.git" ] || git clone -q --branch "$BRANCH" https://github.com/hayvenmjackson/krummholz-llms-alpha "$REPO"
git -C "$REPO" fetch -q && git -C "$REPO" reset -q --hard "origin/$BRANCH"; git -C "$REPO" log -1 --format='repo %h %s' >> "$V/STATUS"
cd "$REPO" || exit 1
T="python3 $REPO/tools/vl/teacher.py"; P="python3 $REPO/tools/vl/pass4.py"; Q="python3 $REPO/tools/vl/pass4b.py"
I="python3 $REPO/tools/vl/identity.py"; V7=$REPO/tools/v7
# public datasets need no GPU: run them alongside the teacher stages
( unset HF_TOKEN HUGGING_FACE_HUB_TOKEN; python3 $V7/fetch_public.py --out $D/public > $V/logs/public.log 2>&1 \
    && st PUBLIC_OK || st PUBLIC_FAIL ) &
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN; export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

# 0a. history passages (v2 set, replay source) and teacher Q&A
if [ ! -f "$HV2/teacher-report.json" ]; then
  $T fetch --out "$HV2/v1" --sources config/history-sources-v1.json && $T fetch --out "$HV2/v2" --sources config/history-sources-v2.json \
    && cat "$HV2/v1/passages.jsonl" "$HV2/v2/passages.jsonl" > "$HV2/passages.jsonl" && $T teach --out "$HV2" --model "$TEACHER" \
    && st HIST_OK || st HIST_FAIL
fi
# 0b. sovereignty pack
if [ ! -f "$H/expand-report.json" ]; then
  $T fetch --out "$H" --sources config/history-sources-v3.json && $T teach --out "$H" --model "$TEACHER" \
    && $T expand --out "$H" --model "$TEACHER" && st SOV_PACK_OK || st SOV_PACK_FAIL
fi
[ -f "$H/sov-status-report.json" ] || { $P sov_status --out "$H" --model "$TEACHER" --hist "$H" && st SOV_STATUS_OK || st SOV_STATUS_FAIL; }
# 0c. shared: canonical status questions, time-sensitive questions, library
cp -n "$H/passages.jsonl" "$H2"/ 2>/dev/null
[ -f "$H2/sov-canon-report.json" ] || { $Q canon --out "$H2" --model "$TEACHER" --hist "$H2" && st CANON_OK || st CANON_FAIL; }
[ -f "$H2/timesens-report.json" ] || { $Q timesens --out "$H2" --model "$TEACHER" && st TIMESENS_OK || st TIMESENS_FAIL; }
[ -f "$LIB" ] || { $Q library --out "$LIB" --passages "$HV2/passages.jsonl" "$H/passages.jsonl" && st LIBRARY_OK || st LIBRARY_FAIL; }
# 0d. identity and values data from the identity-v3 spec
ls "$ID"/identity.*.jsonl > /dev/null 2>&1 || { $I --out "$ID" --model "$TEACHER" --spec config/identity-v3.json && st IDENTITY_OK || st IDENTITY_FAIL; }

# 2. pushback pairs
[ -f "$D/pushback/v7-pushback-report.json" ] || { python3 $V7/pushback.py --out "$D/pushback" --model "$TEACHER" && st PUSHBACK_OK || st PUSHBACK_FAIL; }

# 3. constitutional data: wait for the prompt generator (file stops growing for 20 minutes, up to 6 hours)
if [ ! -f "$D/constitutional/v7-constitutional-report.json" ]; then
  [ -f "$CONST" ] || { st NO_CONSTITUTION; }
  last=-1; still=0
  for i in $(seq 360); do
    n=$(wc -l < "$PROMPTS" 2>/dev/null || echo 0)
    if [ "$n" = "$last" ] && [ "$n" -gt 0 ]; then still=$((still+1)); else still=0; fi
    last=$n; [ $still -ge 20 ] && break; sleep 60
  done
  st PROMPTS_READY $last
  if [ -f "$CONST" ]; then
    C="python3 $V7/teach_v7.py"; O=$D/constitutional
    { [ -f "$O/answers.jsonl" ] || $C answer --out "$O" --prompts "$PROMPTS" --constitution "$CONST" --model "$TEACHER"; } \
      && { [ -f "$O/judged.jsonl" ] || $C judge --out "$O" --model "$TEACHER"; } && $C build --out "$O" \
      && st CONSTITUTIONAL_OK || st CONSTITUTIONAL_FAIL
  fi
fi
wait
st DATA_DONE
