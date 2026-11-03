#!/bin/bash
# Pass 6 on a pod without the shared volume (the global volume would not mount where H200s were free):
# bash novol_v6.sh <NAME> <MODEL> <cap seconds> <input code>. Runs in the background: receives the job inputs from the
# hub pod (the one with the volume, see hub_v6.sh), runs job_v6.sh under the cap, sends the results folder back, then
# stops this pod. runpodctl adds a random relay suffix to send codes, so the full code is published in
# /root/sendback-code.txt, where the hub reads it through this pod's Jupyter server.
NAME=$1; MODEL=$2; CAP=$3; IN=$4; HERE=$(cd "$(dirname "$0")" && pwd)
V=/workspace/krummholz-v6; LOG=$V/logs/job-$NAME.log
mkdir -p $V/logs /root/in
nohup bash -c "
  cd /root/in; for i in \$(seq 30); do runpodctl receive $IN >> /root/in/receive.log 2>&1 && [ -f v6-inputs.tgz ] && break; sleep 20; done
  tar xzf /root/in/v6-inputs.tgz -C /workspace && echo \"\$(date +%T) INPUTS_OK\" >> $LOG || echo INPUTS_FAILED >> $LOG
  NAME=$NAME MODEL=$MODEL MAXLEN=2048 timeout $CAP bash $HERE/job_v6.sh >> $LOG 2>&1; echo EXIT \$? >> $LOG
  cp $LOG $V/$NAME/ 2>/dev/null; rm -rf /root/hf /root/kz6/merged; cd $V; end=\$((SECONDS + 6 * 3600))
  while [ \$SECONDS -lt \$end ]; do
    rm -f /root/sendback-code.txt; runpodctl send --code kz6-$NAME-\$(openssl rand -hex 6 2>/dev/null)\$RANDOM $NAME > /root/sendback.log 2>&1 & pid=\$!
    for i in \$(seq 30); do c=\$(grep -o -E 'runpodctl receive [^ ]+' /root/sendback.log | awk '{print \$3}'); [ -n \"\$c\" ] && break; sleep 2; done
    echo \"\$c\" > /root/sendback-code.txt
    wait \$pid && { echo \"\$(date +%T) SENDBACK_OK\" >> $LOG; break; }
    sleep 30
  done
  sleep 600; runpodctl stop pod \$RUNPOD_POD_ID >> $LOG 2>&1" > /dev/null 2>&1 &
echo LAUNCHED $NAME
