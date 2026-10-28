#!/bin/bash
# Start pass 3 for one size as soon as this pod's pass-2 job has finished, taking over the
# pod's self-stop so it does not shut down mid-run. Usage: SIZE=8B bash v3_waiter.sh
S=${SIZE:-8B}
until [ -f "/workspace/krummholz-qwen3vl-v2/$S/adapter/adapter_config.json" ] && ! pgrep -f job_v2.sh >/dev/null; do sleep 120; done
pkill -f boot/tools/vl/run.sh   # the old wrapper's pending self-stop
mkdir -p /workspace/krummholz-qwen3vl-v3/logs
git clone -q -b vl-pointcloud-v1 https://github.com/hayvenmjackson/krummholz-llms-alpha /root/boot3 2>/dev/null || git -C /root/boot3 pull -q
cp /root/boot3/tools/vl/run.sh /root/run_v3.sh
SIZE=$S exec bash /root/run_v3.sh /root/boot3/tools/vl/job_v3.sh /workspace/krummholz-qwen3vl-v3/logs-$S 12
