#!/bin/bash
# v6 beta publish launcher: bash launch_publish_hf.sh [nostop]. Uploads the v6 GGUFs from the volume to Hugging Face
# with publish_hf.py in the background, its log on the volume, then stops this pod IDLE seconds (default 20 minutes)
# after it ends, like launch_fix_v6.sh. HF_TOKEN comes from the pod's RunPod secret (HF_TOKEN set to
# {{ RUNPOD_SECRET_hf_token }} when the pod is deployed) and is never printed.
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p /workspace/krummholz-v6/logs; LOG=/workspace/krummholz-v6/logs/hf-publish.log
STOP="sleep \${IDLE:-1200}; runpodctl stop pod \$RUNPOD_POD_ID >> $LOG 2>&1 || runpodctl stop pod \$RUNPOD_POD_ID"
[ "${1:-}" = nostop ] && STOP=true
PIP="python3 -m pip install -q -U huggingface_hub hf_xet || python3 -m pip install -q -U --break-system-packages huggingface_hub hf_xet"
if [ -z "${HF_TOKEN:-}" ]; then echo "NO_HF_TOKEN $(date -u +%FT%TZ)" >> $LOG; bash -c "$STOP" > /dev/null 2>&1 & exit 1; fi
nohup bash -c "echo START \$(date -u +%FT%TZ) >> $LOG; { $PIP; } >> $LOG 2>&1 && python3 $HERE/publish_hf.py >> $LOG 2>&1; echo EXIT \$? \$(date -u +%FT%TZ) >> $LOG; $STOP" > /dev/null 2>&1 &
echo LAUNCHED publish
