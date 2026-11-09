#!/bin/bash
# Pass 6 hub, on the pod that has the shared volume: bash hub_v6.sh. Each line "<NAME> <Jupyter URL with token>" in
# /root/hub-expect names a pod running novol_v6.sh; the hub waits for that pod's send code, receives its results folder
# onto the volume (marked RECEIVED), and stops this pod once every listed folder is in and no job runs here
# (at most 24 hours). Lines can be added while it runs.
V=/workspace/krummholz-v6; L=$V/logs/hub.log
recv() {
  local base=${2%%\?*} tok=${2##*token=} c
  until [ -f "$V/$1/RECEIVED" ]; do
    c=$(curl -sf "${base%/}/files/root/sendback-code.txt?token=$tok")
    if [ -n "$c" ]; then
      rm -rf "/root/incoming-$1"; mkdir -p "/root/incoming-$1"; cd "/root/incoming-$1" || return
      if runpodctl receive "$c" >> "$L" 2>&1 && [ -d "$1" ]; then
        mkdir -p "$V/$1" && cp -r "$1"/. "$V/$1"/ && touch "$V/$1/RECEIVED" && echo "$(date +%T) RECEIVED $1" >> "$L"
        cd / && rm -rf "/root/incoming-$1"
        continue
      fi
    fi
    sleep 120
  done
}
declare -A started
end=$((SECONDS + 24 * 3600))
while [ $SECONDS -lt $end ]; do
  pending=0; [ -s /root/hub-expect ] || pending=1
  while read -r n u; do
    [ -z "$n" ] && continue
    [ -z "${started[$n]}" ] && { recv "$n" "$u" < /dev/null & started[$n]=1; echo "$(date +%T) waiting for $n" >> "$L"; }
    [ -f "$V/$n/RECEIVED" ] || pending=1
  done < /root/hub-expect
  if [ $pending = 0 ] && ! pgrep -f 'tools/vl/(job|fix)_v6.sh' > /dev/null; then echo "$(date +%T) HUB_DONE" >> "$L"; break; fi
  sleep 120
done
sleep 1200; runpodctl stop pod "$RUNPOD_POD_ID" >> "$L" 2>&1 || runpodctl stop pod "$RUNPOD_POD_ID"
