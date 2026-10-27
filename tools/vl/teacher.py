"""History and replay data for the Krummholz VL run. Runs on the pod.

  python teacher.py fetch     --out W          passages from config/history-sources-v1.json
  python teacher.py teach     --out W --model allenai/Olmo-3-7B-Instruct
  python teacher.py base      --out W --model Qwen/Qwen3-VL-8B-Instruct
                                               base answers: DPO rejects, replay, probe baseline

Teacher drafts are kept only when every quote is an exact substring of its passage,
every number in the answer appears in the passage, and the question does not
paraphrase a held-out probe. No Claude-written training text.
"""
import argparse, html, json, re, sys, time, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import probes  # noqa: E402

UA = "KrummholzAI-training/0.1 (https://github.com/hayvenmjackson/krummholz-llms-alpha)"
SKIP = re.compile(r"^(see also|references|external links|further reading|notes|bibliography|citations|sources|footnotes|gallery)$", re.I)
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
META = re.compile(r"\b(passage|the text|this text|the article|according to the|the excerpt|the source)\b", re.I)
CHINA = re.compile(r"\b(china|chinese|beijing|taiwan|tibet|xinjiang|uyghur|hong kong|ccp|mao|xi jinping)\b", re.I)


def get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            return urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def chunks(title, text, lo=120, hi=320):
    out, section = [], "Introduction"
    for block in re.split(r"\n(?==+ )", text):
        m = re.match(r"(=+) (.+?) \1\n?", block)
        if m:
            section, block = m.group(2).strip(), block[m.end():]
        if SKIP.match(section):
            continue
        words, buf = [], []
        for para in [p.strip() for p in block.split("\n") if p.strip()]:
            buf.append(para)
            if sum(len(p.split()) for p in buf) >= hi:
                out.append((section, " ".join(buf))); buf = []
        if buf and sum(len(p.split()) for p in buf) >= lo:
            out.append((section, " ".join(buf)))
    return out


def fetch(out):
    cfg = json.loads((ROOT / "config/history-sources-v1.json").read_text(encoding="utf-8"))
    rows = []
    for title in cfg["wikipedia"]:
        q = urllib.parse.urlencode({"action": "query", "prop": "extracts|revisions", "explaintext": 1, "redirects": 1,
                                    "rvprop": "ids", "titles": title, "format": "json", "formatversion": 2})
        page = json.loads(get(f"https://en.wikipedia.org/w/api.php?{q}"))["query"]["pages"][0]
        if "extract" not in page:
            print("MISSING", title, flush=True); continue
        rev = page["revisions"][0]["revid"]
        for i, (section, text) in enumerate(chunks(page["title"], page["extract"])):
            rows.append({"id": f"wiki-{re.sub(r'[^a-z0-9]+', '-', page['title'].lower()).strip('-')}-{i}",
                         "title": page["title"], "section": section, "text": text, "license": "CC BY-SA 4.0",
                         "url": f"https://en.wikipedia.org/w/index.php?oldid={rev}", "revision": rev})
        print(page["title"], rev, flush=True)
        time.sleep(0.5)
    for url in cfg["state_history"]:
        raw = get(url)
        paras = [html.unescape(re.sub(r"<[^>]+>", "", p)).strip() for p in re.findall(r"<p>(.*?)</p>", raw, re.S)]
        text = "\n".join(p for p in paras if len(p.split()) > 12)
        for i, (_, t) in enumerate(chunks(url, text)):
            rows.append({"id": f"state-{url.rstrip('/').split('/')[-1]}-{i}", "title": url, "section": "", "text": t,
                         "license": "U.S. government work", "url": url})
    (out / "passages.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print("passages", len(rows))


TEACH = """You write training questions about history from one passage.
Passage (from "{title}", section "{section}"):
<<<
{text}
>>>
Write 3 question-answer pairs a curious person might ask. Rules:
- Each question must stand alone: never mention "the passage", "the text" or "the article".
- Each answer is 1 to 4 factual sentences supported only by the passage. Keep numbers, dates, names and qualifiers exactly as the passage states them, including uncertainty and ranges of estimates.
- For each answer, list 1 to 3 short quotes copied exactly from the passage that support it.
Return only JSON: [{{"question": "...", "answer": "...", "quotes": ["..."]}}, ...]"""


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def trigrams(s):
    w = re.findall(r"[a-z0-9]+", s.lower())
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}


def verify(item, passage, probe_grams):
    q, a, quotes = item.get("question", ""), item.get("answer", ""), item.get("quotes", [])
    if not (isinstance(q, str) and isinstance(a, str) and isinstance(quotes, list) and quotes):
        return "schema"
    if META.search(q) or META.search(a):
        return "meta"
    if not 30 <= len(a) <= 900 or len(q) < 12:
        return "length"
    p = norm(passage)
    if not all(isinstance(x, str) and len(x) >= 12 and norm(x) in p for x in quotes):
        return "quote"
    if not {n.replace(",", "") for n in NUM.findall(a)} <= {n.replace(",", "") for n in NUM.findall(p)}:
        return "number"
    g = trigrams(q)
    if any(g and len(g & pg) / max(1, len(g | pg)) > 0.4 for pg in probe_grams):
        return "probe_overlap"
    return None


def teach(out, model):
    from vllm import LLM, SamplingParams
    passages = [json.loads(l) for l in open(out / "passages.jsonl", encoding="utf-8")]
    llm = LLM(model, max_model_len=4096, gpu_memory_utilization=0.85, seed=42)
    sp = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=900, seed=42)
    msgs = [[{"role": "user", "content": TEACH.format(**p)}] for p in passages]
    outs = llm.chat(msgs, sp)
    probe_grams = [trigrams(p["question"]) for p in probes.load_suite()["probes"]]
    kept, rejected, seen = [], {}, set()
    for p, o in zip(passages, outs):
        txt = o.outputs[0].text
        m = re.search(r"\[.*\]", txt, re.S)
        try:
            items = json.loads(m.group(0)) if m else []
        except json.JSONDecodeError:
            items = []
        for k, it in enumerate(items if isinstance(items, list) else []):
            why = verify(it, p["text"], probe_grams) if isinstance(it, dict) else "schema"
            key = norm(it.get("question", "")).lower() if isinstance(it, dict) else ""
            if why is None and key in seen:
                why = "duplicate"
            if why:
                rejected[why] = rejected.get(why, 0) + 1; continue
            seen.add(key)
            kept.append({"id": f"{p['id']}-q{k}", "passage": p, "question": norm(it["question"]),
                         "answer": norm(it["answer"]), "quotes": [norm(x) for x in it["quotes"]]})
    sid = lambda p: f"atlantic/{p['id']}"
    qa_rows, chat_rows = [], []
    for r in kept:
        p = r["passage"]
        src = [{"id": sid(p), "text": norm(p["text"])}]
        target = {"answer": r["answer"], "citations": [{"source_id": sid(p), "quote": q} for q in r["quotes"]],
                  "insufficient_evidence": False}
        prov = {"teacher": model, "passage_url": p["url"], "license": p["license"]}
        qa_rows.append({"id": r["id"] + "-src", "source": "history-v1", "provenance": prov,
                        "request": {"query": r["question"], "sources": src}, "target": json.dumps(target, ensure_ascii=False)})
        chat_rows.append({"id": r["id"] + "-chat", "source": "history-v1", "provenance": prov,
                          "messages": [{"role": "user", "content": r["question"]}], "target": r["answer"]})
    for name, rows in (("history_qa", qa_rows), ("history_chat", chat_rows)):
        rows.sort(key=lambda x: x["id"])
        val = [x for i, x in enumerate(rows) if i % 10 == 0]
        train = [x for i, x in enumerate(rows) if i % 10 != 0]
        for part, rr in (("train", train), ("val", val)):
            (out / f"{name}.{part}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rr), encoding="utf-8")
    (out / "teacher-report.json").write_text(json.dumps({"teacher": model, "passages": len(passages),
                                                         "kept": len(kept), "rejected": rejected}, indent=1))
    print("kept", len(kept), "rejected", rejected)


def base(out, model, replay_n=1200):
    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    llm = LLM(model, max_model_len=4096, gpu_memory_utilization=0.85, seed=42, limit_mm_per_prompt={"image": 0, "video": 0})
    sp = SamplingParams(temperature=0.0, max_tokens=700)
    suite = probes.load_suite()
    # 1. probe baseline
    outs = llm.chat([[{"role": "user", "content": p["question"]}] for p in suite["probes"]], sp)
    results = [{"id": p["id"], "topic": p["topic"], "question": p["question"], "answer": o.outputs[0].text,
                **probes.score(p, o.outputs[0].text, suite)} for p, o in zip(suite["probes"], outs)]
    (out / "probes-base.json").write_text(json.dumps({"summary": probes.summarize(results), "results": results}, indent=1, ensure_ascii=False))
    print("base probes", probes.summarize(results), flush=True)
    # 2. DPO pairs: base answers that refuse or carry publisher framing lose to the sourced answer
    pairs = []
    for part in ("train", "val"):
        rows = [json.loads(l) for l in open(out / f"history_chat.{part}.jsonl", encoding="utf-8")]
        outs = llm.chat([r["messages"] for r in rows], sp)
        for r, o in zip(rows, outs):
            ans = o.outputs[0].text
            low = ans.lower()
            bad = bool(probes.REFUSAL.search(low)) or any(f in low for f in suite["red_flags"])
            if bad:
                pairs.append({"id": r["id"] + "-dpo", "part": part, "messages": r["messages"], "chosen": r["target"], "rejected": ans})
    for part in ("train", "val"):
        (out / f"history_dpo.{part}.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs if p["part"] == part), encoding="utf-8")
    print("dpo pairs", len(pairs), flush=True)
    # 3. replay: the base model's own answers to general prompts (OpenAssistant oasst1, Apache 2.0)
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    prompts = [r["text"] for r in ds if r["role"] == "prompter" and r["parent_id"] is None and r["lang"] == "en"
               and 20 < len(r["text"]) < 1500 and not CHINA.search(r["text"])][:replay_n]
    outs = llm.chat([[{"role": "user", "content": t}] for t in prompts], SamplingParams(temperature=0.0, max_tokens=900))
    rows = [{"id": f"replay-{i}", "source": "oasst1 prompts (Apache 2.0) + base model answers",
             "messages": [{"role": "user", "content": t}], "target": o.outputs[0].text.strip()}
            for i, (t, o) in enumerate(zip(prompts, outs)) if o.outputs[0].finish_reason == "stop"]
    (out / "replay.train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print("replay", len(rows), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "teach", "base"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    {"fetch": lambda: fetch(out), "teach": lambda: teach(out, a.model), "base": lambda: base(out, a.model)}[a.cmd]()
