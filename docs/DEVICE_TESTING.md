# On-device testing of Atlantic Lite 4B (dynamic weights phase P)

Goal: measured numbers for the shipped 4B files on real phones and computers, before any on-device
training. Each run records speed, memory, heat and battery, and scores the canary offline on the
exact file the device runs. Nothing is uploaded; results stay on the device until you copy them.

Pinned versions (from the model card repository): llama.cpp commit `e574a17`, file
`krummholz-atlantic-lite-4b-v8.1-Q4_K_M.gguf`, SHA256
`5dcb83f693151a4bf836541c2a54f9eeb0cbd49c769400c36debb9d1b3bad0cb`.

Limits, stated up front:

- `canary-v1` predates the v8 protocol, so its scores are a device baseline, not a release result.
- The compact suite has 102 items; vision items are skipped on text-only runs and counted as skipped.
- One run per device is a smoke measurement. Heat and speed vary with room temperature, charge state
  and background apps; write those down.

## Android (Samsung first, then one other maker)

Needs 8 GB of RAM or more, about 4 GB free storage, and Termux from F-Droid or GitHub (the Play Store
build is outdated).

```sh
termux-wake-lock
pkg update && pkg install -y git cmake clang python curl
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout e574a17
cmake -B build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4 --target llama-server llama-bench
cd ~ && mkdir -p models && cd models
curl -LO https://huggingface.co/krummholz-ai/atlantic-lite-4b-v8.1-GGUF/resolve/main/krummholz-atlantic-lite-4b-v8.1-Q4_K_M.gguf
sha256sum krummholz-atlantic-lite-4b-v8.1-Q4_K_M.gguf   # must match the SHA256 above
```

1. Speed only (about 2 minutes). Try 4 threads, then 6:

   ```sh
   ~/llama.cpp/build/bin/llama-bench -m ~/models/krummholz-atlantic-lite-4b-v8.1-Q4_K_M.gguf -p 512 -n 128 -t 4
   ```

2. Canary run. Unplug the charger so the battery reading means something, keep the screen on,
   and note the room temperature:

   ```sh
   ~/llama.cpp/build/bin/llama-server -m ~/models/krummholz-atlantic-lite-4b-v8.1-Q4_K_M.gguf \
       -c 4096 -t 4 --port 8080 > ~/server.log 2>&1 &
   SERVER_PID=$!
   git clone -b dynamic-weights-device-testing https://github.com/hayvenmjackson/krummholz-llms-alpha ~/alpha && cd ~/alpha
   python tools/dw/device_canary.py --label "Galaxy S26, Q4_K_M, 4 threads" --server-pid $SERVER_PID \
       --model-file ~/models/krummholz-atlantic-lite-4b-v8.1-Q4_K_M.gguf --limit 5 --out ~/results/smoke
   python tools/dw/device_canary.py --label "Galaxy S26, Q4_K_M, 4 threads" --server-pid $SERVER_PID \
       --model-file ~/models/krummholz-atlantic-lite-4b-v8.1-Q4_K_M.gguf --out ~/results/s26-q4km
   ```

   Wait until `server.log` shows the server listening before the first run. The smoke run checks
   the plumbing; the second is the measurement. Results are in `~/results/s26-q4km/report.json`.

3. If thermal or battery readings come back null, Android is hiding them from Termux. Install the
   Termux:API app and record `termux-battery-status` before and after by hand.

4. Stop if the phone gets too hot to hold comfortably or the run is killed by the system; that is a
   result, write it down.

## Windows desktop with an NVIDIA GPU (PowerShell)

Needs Git, Python 3.10 or later, CMake, Visual Studio Build Tools and the CUDA toolkit.

```powershell
git clone https://github.com/ggml-org/llama.cpp $HOME\llama.cpp; cd $HOME\llama.cpp; git checkout e574a17
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF
cmake --build build --config Release -j 8 --target llama-server llama-bench
mkdir $HOME\models -Force; cd $HOME\models
curl.exe -LO https://huggingface.co/krummholz-ai/atlantic-lite-4b-v8.1-GGUF/resolve/main/krummholz-atlantic-lite-4b-v8.1-Q4_K_M.gguf
(Get-FileHash .\krummholz-atlantic-lite-4b-v8.1-Q4_K_M.gguf -Algorithm SHA256).Hash   # must match the SHA256 above
$m = "$HOME\models\krummholz-atlantic-lite-4b-v8.1-Q4_K_M.gguf"
& $HOME\llama.cpp\build\bin\Release\llama-bench.exe -m $m -p 512 -n 128 -ngl 99
$server = Start-Process -PassThru -NoNewWindow $HOME\llama.cpp\build\bin\Release\llama-server.exe `
    -ArgumentList "-m `"$m`" -c 4096 -ngl 99 --port 8080" -RedirectStandardError $HOME\server.log
git clone -b dynamic-weights-device-testing https://github.com/hayvenmjackson/krummholz-llms-alpha $HOME\alpha; cd $HOME\alpha
python tools\dw\device_canary.py --label "Windows, RTX (model), Q4_K_M" --model-file $m --out $HOME\results\win-q4km
Stop-Process -Id $server.Id
```

Windows does not expose temperature or battery files, so those readings are null; note GPU
temperature from Task Manager by hand. Repeat with the Q5_K_M file.

## Mac (Apple silicon)

Same as Android without Termux: Metal is on by default, so build plainly and add `-ngl 99` to
`llama-bench` and `llama-server`. Run once on `Q4_K_M` and once on `Q5_K_M`.

## Results table

Fill one row per run in the plan (`docs/DYNAMIC_WEIGHTS_PLAN.md`, phase P) from `report.json`:
device, RAM, file, threads, generate and prompt tokens per second, server peak memory, max
temperature, battery drop, total minutes, safety refusal rate, protocol parse rate, injection
resisted, product scores, and anything that failed.
