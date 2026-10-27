"""Run the frozen pilot against a GGUF with llama.cpp's raw completion binary."""
import json, os, subprocess, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, "/root/krummholz-llm")
from tools import run_smollm3_pilot as pilot
from transformers import AutoTokenizer

engine, gguf, tokdir, outpath = sys.argv[1:5]
suite, qa = pilot.frozen_inputs()
tok = AutoTokenizer.from_pretrained(tokdir)
threads = str(min(16, len(os.sched_getaffinity(0))))
results = []
for case in suite["cases"]:
    prompt = tok.apply_chat_template(pilot.messages_for(case["request"], qa), tokenize=False,
                                     add_generation_prompt=True, tools=None, enable_thinking=False)
    with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False) as f:
        f.write(prompt.encode())
    started = time.time()
    try:
        p = subprocess.run([engine, "-m", gguf, "--binary-file", f.name, "-c", "4096", "-n", "384",
                            "-ngl", "0", "-t", threads, "-b", "512", "-ub", "512", "--temp", "0",
                            "--seed", "42", "--special", "--no-display-prompt", "--no-conversation",
                            "--no-context-shift", "--simple-io", "--no-escape", "--no-warmup", "--no-perf"],
                           capture_output=True, text=True, timeout=240)
        raw = p.stdout
    except subprocess.TimeoutExpired:
        raw = None
    os.unlink(f.name)
    entry = {"case": case["id"], "seconds": round(time.time() - started, 1), "raw": raw}
    if raw is None:
        entry.update(status="runtime_timeout")
    else:
        text = raw.strip().removesuffix("[end of text]").strip()
        eos = text.endswith("<|im_end|>") or text.endswith("<|endoftext|>")
        text = text.removesuffix("<|im_end|>").removesuffix("<|endoftext|>")
        try:
            entry.update(pilot.parsed_result(text, case["request"]), eos_marker=eos)
        except (ValueError, TypeError) as e:
            entry.update(status="output_rejected", reason=str(e), eos_marker=eos)
    results.append(entry)
    print(case["id"], entry["status"], entry["seconds"], flush=True)
Path(outpath).write_text(json.dumps(results, indent=1, ensure_ascii=False))
print("parsed", sum(r["status"] == "completed" for r in results), "of", len(results))
