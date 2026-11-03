#!/bin/bash
# v6.1 fix launcher: bash launch_fix_v6.sh <NAME> <MODEL> <fix data dir> [nostop]. Runs fix_v6.sh in the background with
# its log on the volume, then stops this pod 20 minutes after it ends (not with nostop, e.g. on the hub pod, where
# hub_v6.sh stops it). Like launch_v6.sh, the stop runs again without the log when the log cannot be opened. A later
# pass (FIX=fix2) logs to its own file.
NAME=$1; MODEL=$2; IN=$3; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p /workspace/krummholz-v6/logs; LOG=/workspace/krummholz-v6/logs/fix-$NAME${FIX:+-$FIX}.log
STOP="sleep 1200; runpodctl stop pod \$RUNPOD_POD_ID >> $LOG 2>&1 || runpodctl stop pod \$RUNPOD_POD_ID"
[ "${4:-}" = nostop ] && STOP=true
nohup bash -c "bash $HERE/fix_v6.sh $NAME $MODEL $IN >> $LOG 2>&1; echo EXIT \$? >> $LOG; $STOP" > /dev/null 2>&1 &
echo LAUNCHED fix $NAME
