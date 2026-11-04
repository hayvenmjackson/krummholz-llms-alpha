#!/bin/bash
# Multilingual check of the published v6 GGUFs: bash launch_multilingual_v6.sh [nostop]. In the background, builds
# llama.cpp with CUDA, then serves each model's Q4_K_M from the volume in turn (the file people download) and asks it
# the multilingual probes with multilingual.py. Answers go to $V/multilingual/<model>.json, the log to
# $V/logs/multilingual.log, and the pod stops IDLE seconds (default 20 minutes) after the job ends.
HERE=$(cd "$(dirname "$0")" && pwd); V=/workspace/krummholz-v6; OUT=$V/multilingual
mkdir -p $V/logs $OUT; LOG=$V/logs/multilingual.log
STOP="sleep \${IDLE:-1200}; runpodctl stop pod \$RUNPOD_POD_ID >> $LOG 2>&1 || runpodctl stop pod \$RUNPOD_POD_ID"
[ "${1:-}" = nostop ] && STOP=true
cat > /root/multilingual-job.sh <<EOF
set -x
L=/root/llama.cpp
command -v cmake > /dev/null || python3 -m pip install -q cmake || python3 -m pip install -q --break-system-packages cmake
git clone -q --depth 1 https://github.com/ggml-org/llama.cpp \$L && git -C \$L rev-parse HEAD > $OUT/llama-cpp-commit.txt
cmake -S \$L -B \$L/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build \$L/build -j \$(nproc) --target llama-server > /dev/null || { echo BUILD_FAIL; exit 1; }
for m in atlantic-lite-4b-fix3:krummholz-atlantic-lite-4b-v6.3 atlantic-27b:krummholz-atlantic-27b-v6 core-35b-a3b-fix3:krummholz-core-35b-a3b-v6.3; do
  dir=\${m%%:*}; n=\${m##*:}
  \$L/build/bin/llama-server -m $V/\$dir/\$n-Q4_K_M.gguf -ngl 99 -c 8192 --jinja --reasoning-budget 0 --port 8080 > /root/server-\$n.log 2>&1 &
  pid=\$!
  for i in \$(seq 120); do curl -sf 127.0.0.1:8080/health > /dev/null && break; sleep 5; done
  python3 $HERE/multilingual.py \$n $OUT/\$n.json && echo MODEL_OK \$n || { echo MODEL_FAIL \$n; tail -n 20 /root/server-\$n.log; }
  kill \$pid; wait \$pid 2> /dev/null
done
echo MULTILINGUAL_DONE
EOF
nohup bash -c "echo START \$(date -u +%FT%TZ) >> $LOG; bash /root/multilingual-job.sh >> $LOG 2>&1; echo EXIT \$? \$(date -u +%FT%TZ) >> $LOG; $STOP" > /dev/null 2>&1 &
echo LAUNCHED multilingual
