#!/bin/bash
# Pass 6 launcher: bash launch_v6.sh <NAME> <MODEL> <cap seconds> [MAXLEN]. Runs job_v6.sh in the background under a
# time cap with its log on the volume, then stops this pod IDLE seconds (default 20 minutes) after the job ends. The stop
# runs again without the log when the log cannot be opened (a dropped volume mount left the first 27B pod billing).
NAME=$1; MODEL=$2; CAP=$3; MAXLEN=${4:-2048}; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p /workspace/krummholz-v6/logs; LOG=/workspace/krummholz-v6/logs/job-$NAME.log
nohup bash -c "NAME=$NAME MODEL=$MODEL MAXLEN=$MAXLEN timeout $CAP bash $HERE/job_v6.sh >> $LOG 2>&1; echo EXIT \$? >> $LOG; \
  sleep \${IDLE:-1200}; runpodctl stop pod \$RUNPOD_POD_ID >> $LOG 2>&1 || runpodctl stop pod \$RUNPOD_POD_ID" > /dev/null 2>&1 &
echo LAUNCHED $NAME
