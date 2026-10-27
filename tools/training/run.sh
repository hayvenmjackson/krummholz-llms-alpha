#!/bin/bash
cd /root/kz
timeout 10800 bash job.sh > job.log 2>&1
echo "EXIT $?" >> job.log
cp job.log /workspace/krummholz-qwen3-8b-v1/ 2>/dev/null
touch DONE
sleep 3600
runpodctl stop pod "$RUNPOD_POD_ID" >> job.log 2>&1
