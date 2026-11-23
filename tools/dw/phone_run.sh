#!/data/data/com.termux/files/usr/bin/bash
# One-command on-device test of Atlantic Lite 4B (dynamic weights phase P).
#   bash tools/dw/phone_run.sh "Galaxy S26"
# Safe to run again: finished steps are skipped (build, download), an interrupted download resumes.
# Works in Termux on Android and in any Linux or macOS shell. Override paths with LLAMA_DIR, MODEL_DIR.
set -euo pipefail

# Background mode: bash tools/dw/phone_run.sh --background "Galaxy S26"
# Asks for storage access and takes the wake lock now (neither can prompt later), then keeps running
# after you leave Termux or close its window. Follow it with: tail -f ~/phone_run.log
if [ "${1:-}" = "--background" ]; then
  shift
  : "${1:?give the device name, for example: bash tools/dw/phone_run.sh --background \"Galaxy S26\"}"
  if [ -d /data/data/com.termux ]; then
    command -v termux-wake-lock >/dev/null && termux-wake-lock || true
    [ -d "$HOME/storage/downloads" ] || termux-setup-storage || true
  fi
  RUN_MODE=background nohup bash "$0" "$@" > "$HOME/phone_run.log" 2>&1 < /dev/null &
  echo "Running in the background (process $!)."
  echo "Watch progress:  tail -f ~/phone_run.log    (Ctrl+C stops watching, not the test)"
  echo "Stop the test:   pkill -f phone_run.sh; pkill -f llama-server"
  exit 0
fi

LABEL="${1:?give the device name, for example: bash tools/dw/phone_run.sh \"Galaxy S26\"}"
[ "${RUN_MODE:-}" = background ] && LABEL="$LABEL, background"
THREADS="${THREADS:-4}"
LIMIT="${LIMIT:-}"                      # LIMIT=5 for a quick check
REPACK="${REPACK:-0}"                   # REPACK=1 measures llama.cpp weight repacking (about 2.6 GB more memory)
[ "$REPACK" = 1 ] && LABEL="$LABEL, repack"
KIT="$(cd "$(dirname "$0")/../.." && pwd)"
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
COMMIT=e574a17
NAME=krummholz-atlantic-lite-4b-v8.1-Q4_K_M.gguf
SHA=5dcb83f693151a4bf836541c2a54f9eeb0cbd49c769400c36debb9d1b3bad0cb
URL="https://huggingface.co/krummholz-ai/atlantic-lite-4b-v8.1-GGUF/resolve/main/$NAME"
SLUG="$(echo "$LABEL" | tr -cs 'A-Za-z0-9' '-' | sed 's/-*$//')"
OUT="$HOME/results/$SLUG-q4km${LIMIT:+-limit$LIMIT}"
say() { printf '\n== %s\n' "$*"; }
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

if [ -d /data/data/com.termux ]; then
  say "Termux packages"
  pkg install -y git cmake clang python curl >/dev/null
  command -v termux-wake-lock >/dev/null && termux-wake-lock || true
fi

BIN="$LLAMA_DIR/build/bin"
if [ ! -x "$BIN/llama-server" ] || [ ! -x "$BIN/llama-bench" ]; then
  say "Building llama.cpp $COMMIT (10 to 20 minutes on a phone)"
  [ -d "$LLAMA_DIR/.git" ] || git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
  git -C "$LLAMA_DIR" checkout -q "$COMMIT"
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$LLAMA_DIR/build" -j4 --target llama-server llama-bench
fi
say "llama.cpp ready: $BIN"

MODEL="$MODEL_DIR/$NAME"
mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL" ] || [ "$(sha "$MODEL")" != "$SHA" ]; then
  say "Downloading the model (2.7 GB; run this script again to resume if it stops)"
  curl -fL -C - -o "$MODEL" "$URL"
  say "Checking the model file"
  got="$(sha "$MODEL")"
  if [ "$got" != "$SHA" ]; then
    echo "Checksum mismatch: got $got, want $SHA. Delete $MODEL and run again." >&2
    exit 1
  fi
fi
say "Model verified: $MODEL"

if [ -r /proc/meminfo ]; then
  avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  say "Free memory now: ${avail} MB (the server needs about 3,000 MB)"
  need=$([ "$REPACK" = 1 ] && echo 6500 || echo 3500)
  if [ "$avail" -lt "$need" ]; then
    echo "Too little free memory to run safely. Close other apps (or restart the phone) and run this again." >&2
    exit 1
  fi
fi

say "Speed test"
"$BIN/llama-bench" -m "$MODEL" -p 512 -n 128 -t "$THREADS" --repack "$REPACK" | tee "$HOME/bench-$SLUG.txt"

say "Starting the model server"
pkill -f "llama-server.*--port 8080" 2>/dev/null || true
# --no-repack: the default keeps a second, repacked copy of the weights (5.6 GB peak against 3.0 GB)
"$BIN/llama-server" -m "$MODEL" -c 4096 -t "$THREADS" -np 1 --cache-ram 0 $([ "$REPACK" = 1 ] || echo --no-repack) --port 8080 > "$HOME/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
for _ in $(seq 180); do
  curl -sf http://127.0.0.1:8080/health >/dev/null && break
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "The server stopped. Last log lines:" >&2; tail -20 "$HOME/server.log" >&2; exit 1; fi
  sleep 2
done
curl -sf http://127.0.0.1:8080/health >/dev/null || { echo "The server did not start in 6 minutes. Log:" >&2; tail -20 "$HOME/server.log" >&2; exit 1; }

say "Running the canary (30 to 60 minutes on a phone for the full set; keep the screen on)"
python3 "$KIT/tools/dw/device_canary.py" --label "$LABEL, Q4_K_M, $THREADS threads" --server-pid "$SERVER_PID" \
  --model-file "$MODEL" --out "$OUT" ${LIMIT:+--limit "$LIMIT"}
cp "$HOME/bench-$SLUG.txt" "$OUT/bench.txt"

SHARE="$OUT/report.json"
if [ -d /data/data/com.termux ]; then
  [ -d "$HOME/storage/downloads" ] || termux-setup-storage || true
  if [ -d "$HOME/storage/downloads" ]; then
    cp "$OUT/report.json" "$HOME/storage/downloads/$SLUG-report.json"
    cp "$OUT/bench.txt" "$HOME/storage/downloads/$SLUG-bench.txt"
    SHARE="Downloads/$SLUG-report.json and Downloads/$SLUG-bench.txt"
  fi
fi
say "Done. Send these: $SHARE"
