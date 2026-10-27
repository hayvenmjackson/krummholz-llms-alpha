#!/bin/bash
# PointNet++ on Carl's forest datasets. Expects the zips in /workspace/incoming
# (sent with `runpodctl send`). Results go to /workspace/krummholz-pointnet2-v1.
set -uo pipefail
BRANCH=${BRANCH:-vl-pointcloud-v1}; REPO=/root/krummholz-llm; DATA=/root/pc; FINAL=/workspace/krummholz-pointnet2-v1
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git clone -q --branch "$BRANCH" https://github.com/hayvenmjackson/krummholz-llms-alpha "$REPO" && git -C "$REPO" log -1 --format='repo %h %s'
mkdir -p "$DATA" "$FINAL"
for z in /workspace/incoming/*.zip; do
  sha256sum "$z" | tee -a "$FINAL/input-SHA256SUMS"
  (cd "$DATA" && (unzip -q -o "$z" || bsdtar -xf "$z" 2>/dev/null || python3 -c "import sys,zipfile;zipfile.ZipFile(sys.argv[1]).extractall('.')" "$z")) || echo "EXTRACT_WARN $z"
done
ARGS=()
[ -d "$DATA/Camera-like Dataset" ] && ARGS+=("camera=$DATA/Camera-like Dataset")
[ -d "$DATA/LiDAR-like Dataset" ] && ARGS+=("lidar=$DATA/LiDAR-like Dataset")
[ ${#ARGS[@]} -gt 0 ] || { echo NO_DATA; exit 1; }
cd "$REPO/tools/pointcloud"
python3 train_seg.py --data "${ARGS[@]}" --out "$FINAL" --epochs "${EPOCHS:-40}" --hours "${HOURS:-4}" || { echo TRAIN_FAIL; exit 1; }
cp pointnet2.py data.py summarize.py "$FINAL"/
(cd "$FINAL" && sha256sum *.pt *.json *.py > SHA256SUMS)
echo TRAIN_OK
ls -la "$FINAL"
