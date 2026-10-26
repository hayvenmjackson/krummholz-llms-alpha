# Android first-run procedure

**Historical diagnostic only.** The tested 350M model described here did not meet
the quality bar. Do not use this procedure as a recommended installation path.
The forthcoming verified alpha will use browser download and local file import;
see [android-alpha.md](android-alpha.md). The original commands below preserve the
development record, not an endorsement of this model or quantization choice.

Prepared October 23, 2026. These commands have not been executed on the target
phone. The operator handles pairing, provisioning and execution. Use only the
selected paired Samsung device and the dedicated `/data/local/tmp/krummholz-llm`
directory. This run uses fictional Acadia policy text; it reads no phone contacts
or client data.

The reference Android archive was hash-checked and extracted to
`runtime/android-b11669/llama-b11669/`. Its executables and shared
libraries occupy the same directory. Do not substitute a different layout or
release without inspecting it. Confirm `arm64-v8a`, available RAM/storage and
permissions before pushing the model or running the quantizer.

The PowerShell examples below start in the project root. Set `$adbPath` to the
verified adb.exe path and `$deviceSerial` to the exact paired device serial.
They deliberately do not automate pairing, install an app, change phone settings,
open a server, or grant the model any tool access. Record each exit status; stop
on an error rather than running the rest of the sequence.

## Inspect and provision

```powershell
& $adbPath -s $deviceSerial shell getprop ro.product.cpu.abi
& $adbPath -s $deviceSerial shell cat /proc/meminfo
& $adbPath -s $deviceSerial shell df -h /data/local/tmp
& $adbPath -s $deviceSerial shell mkdir -p /data/local/tmp/krummholz-llm/models /data/local/tmp/krummholz-llm/input /data/local/tmp/krummholz-llm/output
& $adbPath -s $deviceSerial push runtime/android-b11669/llama-b11669 /data/local/tmp/krummholz-llm/runtime
& $adbPath -s $deviceSerial shell chmod 700 /data/local/tmp/krummholz-llm/runtime/llama-quantize /data/local/tmp/krummholz-llm/runtime/llama-completion
& $adbPath -s $deviceSerial shell 'cd /data/local/tmp/krummholz-llm/runtime && LD_LIBRARY_PATH=. ./llama-quantize --help'
& $adbPath -s $deviceSerial shell 'cd /data/local/tmp/krummholz-llm/runtime && LD_LIBRARY_PATH=. ./llama-completion --help'
```

`adb push` above assumes the destination `runtime` does not already exist. If a
prior attempt created it, inspect the result and choose an explicit fresh target;
do not accidentally nest the directory. Confirm both help commands work and
support the flags shown below before proceeding. No model is loaded by help.

## Create a quantized artifact on the phone

After device memory review, push the verified source model (709,295,424 bytes).
Check the phone copy's SHA-256 equals
`2eeed95bfc1a6c58868697ba5120e32a34f54ff6f11b730bbd5c5384eac5871a`.

```powershell
& $adbPath -s $deviceSerial push models/350m_test.gguf /data/local/tmp/krummholz-llm/models/350m_test.gguf
& $adbPath -s $deviceSerial shell sha256sum /data/local/tmp/krummholz-llm/models/350m_test.gguf
& $adbPath -s $deviceSerial shell 'cd /data/local/tmp/krummholz-llm/runtime && LD_LIBRARY_PATH=. ./llama-quantize --max-buffer-size 8 ../models/350m_test.gguf ../models/pleias-rag-350m-q4_k_m.gguf.partial Q4_K_M 1'
```

Monitor the phone's memory, responsiveness and temperature during the single
quantization. Use an explicit memory guard and a bounded execution
timeout. A nonzero exit, disconnect, interruption or guard stop leaves an
unapproved partial file. Do not rename or load it. When the quantizer exits zero,
record its completion diagnostics, artifact size and SHA-256, then pull the file
to a local **partial** name for verification. These later steps are conditional
on successful completion, not a single script to run blindly:

```powershell
& $adbPath -s $deviceSerial shell sha256sum /data/local/tmp/krummholz-llm/models/pleias-rag-350m-q4_k_m.gguf.partial
& $adbPath -s $deviceSerial shell stat -c %s /data/local/tmp/krummholz-llm/models/pleias-rag-350m-q4_k_m.gguf.partial
& $adbPath -s $deviceSerial pull /data/local/tmp/krummholz-llm/models/pleias-rag-350m-q4_k_m.gguf.partial models/pleias-rag-350m-q4_k_m.gguf.partial
Get-FileHash -Algorithm SHA256 -LiteralPath models/pleias-rag-350m-q4_k_m.gguf.partial
```

Compare the two hashes and sizes, validate the GGUF structure,
record source hash plus quantizer hash/version/options in `config/models.json`,
and rename only the verified local and phone outputs to `.gguf`. Quantization is
numerical compression of upstream weights, not Krummholz training. A structurally
valid output still needs inference and quality checks.

## One bounded inference and local parsing

Export a UTF-8 prompt without a byte-order mark. No model runs in this command:

```powershell
python -m krummholz_llm.external_io export --profile acadia --input data/acadia/example.json --prompt-output artifacts/phone-acadia-prompt.txt --context 512 --max-output-tokens 64
& $adbPath -s $deviceSerial push artifacts/phone-acadia-prompt.txt /data/local/tmp/krummholz-llm/input/acadia-prompt.txt
```

After verifying and renaming the derived artifact, run one case.
Use the same 512-context/64-output budget as the exported prompt. The operator must
bound the ADB process duration (for example, 60 seconds) and stop the specifically
owned phone process if the connection drops; a host disconnect alone is not proof
the phone process stopped. Check runtime help before relying on any timeout utility.

```powershell
& $adbPath -s $deviceSerial shell 'cd /data/local/tmp/krummholz-llm/runtime && LD_LIBRARY_PATH=. ./llama-completion -m ../models/pleias-rag-350m-q4_k_m.gguf -f ../input/acadia-prompt.txt -c 512 -n 64 -ngl 0 -t 1 -b 8 -ub 8 --temp 0 --seed 42 --special --no-display-prompt --no-conversation --no-context-shift --simple-io --no-escape --no-warmup --no-perf > ../output/acadia-stdout.txt 2> ../output/acadia-stderr.txt'
& $adbPath -s $deviceSerial pull /data/local/tmp/krummholz-llm/output/acadia-stdout.txt artifacts/phone-acadia-stdout.txt
python -m krummholz_llm.external_io parse --profile acadia --input data/acadia/example.json --generation artifacts/phone-acadia-stdout.txt --output artifacts/phone-acadia-parsed.json
```

This first tiny run may end before the model reaches its final-answer block. The
parser then rejects it and displays no unfinished reasoning. A completed parse
remains `needs_human_review`. The parser cannot prove which runtime/model produced
an imported file; combine it with the provisioning record and execution
log. Capture latency and memory separately. Do not call a parse pass a quality,
phone performance, safety or production-readiness pass.

The export and capture files are deliberately retained local artifacts. Remove
only this procedure's files when no longer needed; deletion is not secure erasure.
