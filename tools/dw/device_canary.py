"""Run the canary on a real device against the exact model file the device serves (plan phase P).

  python tools/dw/device_canary.py --server http://127.0.0.1:8080 --label "Galaxy S26" \
      --model-file ~/models/atlantic-lite-4b-v8.1-Q4_K_M.gguf --out results/s26-q4km

The model runs in llama.cpp's `llama-server` on the device (Termux on Android, or any desktop).
This script needs only the Python standard library plus this repository, so it runs in Termux.
It records, per item: the raw reply, seconds, tokens and llama.cpp timings, free memory, the
hottest thermal zone and battery level where the operating system exposes them. Missing
readings are recorded as null, never guessed. Scoring is offline (tools/dw/canary.py).

Known limit: canary-v1 was written for the v1 to v3 protocol. Replies are scored after removing a
leading <certainty> tag and empty <think> blocks; the raw reply is always kept. Treat v8 scores as a
device baseline until the canary is refreshed for v8 (plan section 10, phase 4).
"""
from __future__ import annotations

import argparse, glob, hashlib, json, os, platform, re, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]
from tools.dw import canary  # noqa: E402
from tools import check_smollm3_reference as reference  # noqa: E402

LEAD = re.compile(r"^\s*(?:<think>\s*</think>\s*)?(?:<certainty>[^<]{0,16}</certainty>\s*)?(?:<think>\s*</think>\s*)?", re.S)


def clean(raw):
    return LEAD.sub("", raw or "", count=1).strip()


def messages_for(item, qa_system):
    if "messages" in item:
        return item["messages"]
    return [{"role": "system", "content": qa_system},
            {"role": "user", "content": json.dumps(item["request"], ensure_ascii=False)}]


def text_only(messages):
    return all(isinstance(m.get("content"), str) for m in messages)


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def device_stats():
    """Best effort. Android often hides battery and thermal files from apps; null means not readable."""
    stats = {"mem_available_mb": None, "max_thermal_c": None, "battery_pct": None}
    meminfo = _read("/proc/meminfo")
    if meminfo:
        m = re.search(r"MemAvailable:\s+(\d+)", meminfo)
        stats["mem_available_mb"] = int(m.group(1)) // 1024 if m else None
    temps = []
    for zone in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        v = _read(zone)
        if v and v.lstrip("-").isdigit():
            c = int(v) / (1000 if abs(int(v)) > 1000 else 1)
            if 0 < c < 150:
                temps.append(c)
    stats["max_thermal_c"] = round(max(temps), 1) if temps else None
    for name in ("battery", "BAT0", "BAT1"):
        v = _read(f"/sys/class/power_supply/{name}/capacity")
        if v and v.isdigit():
            stats["battery_pct"] = int(v)
            break
    return stats


def peak_rss_mb(pid):
    status = _read(f"/proc/{pid}/status") if pid else None
    m = re.search(r"VmHWM:\s+(\d+)", status or "")
    return int(m.group(1)) // 1024 if m else None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def chat(server, messages, max_tokens, timeout, post=None, thinking=False):
    # Thinking off matches tools/training/eval_gguf.py; with it on, the 4B can spend the whole budget thinking
    body = {"messages": messages, "temperature": 0, "seed": 42, "max_tokens": max_tokens, "stream": False,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    if post:
        return post(body)
    req = urllib.request.Request(server.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run(suite, server, max_tokens=384, timeout=600, limit=None, server_pid=None, post=None, log=print, thinking=False,
        progress=None):
    """progress: optional path; rows are rewritten there after every item so a crash keeps what finished."""
    qa_system = reference.load_spec()["qa_system"]
    rows, responses = [], {}
    items = suite["items"][:limit] if limit else suite["items"]
    for n, item in enumerate(items, 1):
        msgs = messages_for(item, qa_system)
        row = {"id": item["id"], "section": item["section"], "before": device_stats()}
        if not text_only(msgs):
            row.update(status="skipped_needs_vision")
        else:
            started = time.monotonic()
            try:
                out = chat(server, msgs, max_tokens, timeout, post, thinking)
                choice = out["choices"][0]
                raw = choice["message"].get("content") or ""
                row.update(status="completed", raw=raw, finish_reason=choice.get("finish_reason"),
                           reasoning_chars=len(choice["message"].get("reasoning_content") or ""),
                           usage=out.get("usage"), timings=out.get("timings"))
                responses[item["id"]] = clean(raw)
            except Exception as e:  # recorded, and scored as a failure by canary.score
                row.update(status="error", error=f"{type(e).__name__}: {e}")
            row["seconds"] = round(time.monotonic() - started, 2)
        row["after"] = device_stats()
        row["server_peak_rss_mb"] = peak_rss_mb(server_pid)
        rows.append(row)
        if progress:
            tmp = Path(str(progress) + ".tmp")
            tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, progress)
        log(f"[{n}/{len(items)}] {item['id']} {row['status']} {row.get('seconds', '')}s")
    return rows, responses


def summarize(rows):
    done = [r for r in rows if r["status"] == "completed"]
    speeds = [r["timings"]["predicted_per_second"] for r in done if (r.get("timings") or {}).get("predicted_per_second")]
    prompt = [r["timings"]["prompt_per_second"] for r in done if (r.get("timings") or {}).get("prompt_per_second")]
    temps = [s["max_thermal_c"] for r in rows for s in (r["before"], r["after"]) if s["max_thermal_c"] is not None]
    batt = [s["battery_pct"] for r in rows for s in (r["before"], r["after"]) if s["battery_pct"] is not None]
    mem = [r["after"]["mem_available_mb"] for r in rows if r["after"]["mem_available_mb"] is not None]
    med = lambda xs: round(sorted(xs)[len(xs) // 2], 2) if xs else None
    return {"items": len(rows), "completed": len(done),
            "errors": sum(r["status"] == "error" for r in rows),
            "hit_token_limit": sum(r.get("finish_reason") == "length" for r in done),
            "empty_replies": sum(not r["raw"].strip() for r in done),
            "skipped_needs_vision": sum(r["status"] == "skipped_needs_vision" for r in rows),
            "total_seconds": round(sum(r.get("seconds", 0) for r in rows), 1),
            "median_generate_tokens_per_s": med(speeds), "median_prompt_tokens_per_s": med(prompt),
            "max_thermal_c": max(temps) if temps else None,
            "battery_drop_pct": batt[0] - batt[-1] if len(batt) > 1 else None,
            "min_mem_available_mb": min(mem) if mem else None,
            "server_peak_rss_mb": max((r["server_peak_rss_mb"] for r in rows if r["server_peak_rss_mb"]), default=None)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--suite", default=str(ROOT / "evaluation/canary-v1-compact.json"))
    ap.add_argument("--label", required=True, help="device name as you want it in the results table")
    ap.add_argument("--model-file", help="the GGUF the server loaded; hashed so results name the exact file")
    ap.add_argument("--server-pid", type=int, help="llama-server process id, for peak memory (Linux and Android)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--limit", type=int, help="first N items only, for a quick smoke run")
    ap.add_argument("--thinking", action="store_true", help="allow hidden reasoning (off by default, as in eval_gguf.py)")
    a = ap.parse_args(argv)
    suite = json.loads(Path(a.suite).read_text(encoding="utf-8"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, responses = run(suite, a.server, a.max_tokens, limit=a.limit, server_pid=a.server_pid, thinking=a.thinking,
                          progress=out / "rows.partial.json")
    report = {"label": a.label, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "platform": {"system": platform.system(), "machine": platform.machine(), "release": platform.release(),
                           "android": os.path.isdir("/system/app")},
              "suite": {"version": suite.get("version"), "sha256": suite.get("sha256"), "compact": suite.get("compact"),
                        "limit": a.limit, "thinking": a.thinking, "max_tokens": a.max_tokens},
              "model_file": {"name": Path(a.model_file).name, "sha256": sha256_file(a.model_file)} if a.model_file else None,
              "summary": summarize(rows),
              "metrics": None if a.limit else canary.score(suite, responses),
              "caveat": "canary-v1 predates the v8 protocol; a device baseline, not a release result"}
    (out / "rows.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "responses.json").write_text(json.dumps(responses, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "metrics": report["metrics"]}, indent=1))


if __name__ == "__main__":
    main()
