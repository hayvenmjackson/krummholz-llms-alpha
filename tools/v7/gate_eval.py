"""Score the v7 runtime gates against a served model: raw, output guards only, and the full gate sequence.

Runs from any machine against an OpenAI-compatible chat endpoint (llama.cpp's llama-server on a pod serving the shipped
Q4_K_M GGUF), so it measures what a person would get from the released file:
  raw       the model's own answer (what the release gates score)
  guarded   guards.finalize on the raw answer (the output guards as they ship)
  gated     guards.respond: grounding note, one regeneration on review, then finalize
for every probe suite in evaluation/ plus evaluation/gates-heldout-v1.json, whose questions are in no probe set.

  python tools/v7/gate_eval.py --url https://<pod>-8000.proxy.runpod.net --name atlantic-lite-4b --out results.json
Greedy decoding, thinking off, 700 new tokens: the same settings as the probes in tools/vl/train_vl.py.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools" / "vl"))
from krummholz_llm import grounding, guards  # noqa: E402
import probes, selfreport  # noqa: E402,E401

THINK = re.compile(r"<think>.*?</think>\s*", re.S)


def chat(url, messages, max_tokens=700, tries=4):
    body = json.dumps({"model": "m", "messages": messages, "temperature": 0, "max_tokens": max_tokens,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    for i in range(tries):
        try:
            req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", body, {"Content-Type": "application/json", "User-Agent": "krummholz-gate-eval/1"})
            out = json.loads(urllib.request.urlopen(req, timeout=600).read())
            return THINK.sub("", out["choices"][0]["message"]["content"] or "").strip()
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def suites():
    out = []
    for p in sorted((ROOT / "evaluation").glob("*probes-v*.json")):
        out.append((p.stem.replace("decensor-", ""), probes.load_suite(p)))
    for name in ("gates-heldout-v1", "gates-heldout-v2"):
        ho = json.loads((ROOT / f"evaluation/{name}.json").read_text(encoding="utf-8"))
        ho["red_flags"] = probes.load_suite(ROOT / "evaluation" / f"{ho['red_flags_from']}.json")["red_flags"]
        out.append((name, ho))
        if ho.get("mind_probes"):
            out.append((name.replace("heldout", "heldout-mind"), {"name": name, "bucket": "consciousness", "checker": "consciousness",
                        "min_coverage": 0.0, "red_flags": [], "probes": [{**p, "topic": "consciousness", "facts": []} for p in ho["mind_probes"]]}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--product", default="atlantic")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    lib = grounding.load()
    gen = lambda msgs: chat(a.url, msgs)  # noqa: E731

    def one(p):
        q = p["question"]
        raw = gen([{"role": "user", "content": q}])
        gated, log = guards.respond(gen, q, product=a.product, library=lib)
        return {"id": p["id"], "topic": p["topic"], "question": q, "raw": raw,
                "guarded": guards.finalize(q, raw, a.product), "gated": gated, "gate_log": log}

    report = {"model": a.name, "url_host": a.url.split("//")[-1].split("/")[0], "product": a.product,
              "library": {"name": lib.name, "as_of": lib.as_of, "documents": len(lib.ids)} if lib else None,
              "suites": {}}
    for name, suite in suites():
        if a.only and name not in a.only:
            continue
        t0 = time.time()
        with ThreadPoolExecutor(a.workers) as ex:
            rows = list(ex.map(one, suite["probes"]))
        by_mode = {}
        for mode in ("raw", "guarded", "gated"):
            scored = [{**r, **probes.score(next(p for p in suite["probes"] if p["id"] == r["id"]), r[mode], suite)} for r in rows]
            by_mode[mode] = {"summary": probes.summarize(scored, suite.get("bucket", "censorship")),
                             "failed": [{"id": s["id"], "coverage": s["coverage"], "red_flags": s["red_flags"],
                                         "refused": s["refused"]} for s in scored if not s["passed"]]}
        report["suites"][name] = {"rows": rows, **by_mode, "seconds": round(time.time() - t0)}
        line = {m: {k: f"{v['passed']}/{v['n']}" for k, v in by_mode[m]["summary"].items()} for m in by_mode}
        print(name, json.dumps(line), f"grounded {sum(r['gate_log']['grounded'] for r in rows)}/{len(rows)}",
              f"retried {sum(r['gate_log']['retried'] for r in rows)}", flush=True)
        Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
