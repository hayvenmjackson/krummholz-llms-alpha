#!/bin/bash
# Launch a job script under a time cap, keep its log on the volume, then stop the pod after an hour.
# Usage: bash run.sh <job.sh> <log-dir> [hours]
JOB=$1; LOGDIR=$2; HOURS=${3:-8}
mkdir -p "$LOGDIR" /root/kz
cd /root/kz
timeout $((HOURS * 3600)) bash "$JOB" > "$LOGDIR/job.log" 2>&1
echo "EXIT $?" >> "$LOGDIR/job.log"
touch "$LOGDIR/DONE"
sleep 3600
runpodctl stop pod "$RUNPOD_POD_ID" >> "$LOGDIR/job.log" 2>&1
