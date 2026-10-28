"""History and replay data for the Krummholz VL run. Runs on the pod.

  python teacher.py fetch     --out W          passages from config/history-sources-v1.json
  python teacher.py teach     --out W --model allenai/Olmo-3-7B-Instruct
  python teacher.py base      --out W --model Qwen/Qwen3-VL-8B-Instruct
                                               base answers: DPO rejects, replay, probe baseline

Teacher drafts are kept only when every quote is an exact substring of its passage,
every number in the answer appears in the passage, and the question does not
paraphrase a held-out probe. No Claude-written training text.
"""
import argparse, html, json, re, sys, time, urllib.parse, urllib.request, zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import probes  # noqa: E402


def _no_thinking():
    """Every vLLM chat in these tools wants the direct reply. Qwen3.5-family templates think by default, so pass
    enable_thinking=False unless a caller asks otherwise; templates without the switch ignore it."""
    try:
        from vllm import LLM
    except ImportError:
        return
    if getattr(LLM.chat, "_krummholz_no_think", False):
        return
    chat = LLM.chat

    def patched(self, *a, **kw):
        kw.setdefault("chat_template_kwargs", {"enable_thinking": False})
        return chat(self, *a, **kw)
    patched._krummholz_no_think = True
    LLM.chat = patched
    env = __import__("os").environ
    quant = env.get("KZ_VLLM_QUANT")  # e.g. fp8: fits a 35B MoE on one 80 GB GPU
    # hybrid (Gated DeltaNet) models need one Mamba cache block per running sequence; vLLM's default of 1024
    # sequences fails on a 27B in 80 GB (243 blocks), so cap it
    max_seqs = int(env.get("KZ_VLLM_MAX_SEQS", "128"))
    init = LLM.__init__

    def patched_init(self, *a, **kw):
        if quant:
            kw.setdefault("quantization", quant)
        kw.setdefault("max_num_seqs", max_seqs)
        return init(self, *a, **kw)
    LLM.__init__ = patched_init


_no_thinking()

UA ="KrummholzAI-training/0.1 (https://github.com/hayvenmjackson/krummholz-llms-alpha)"
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


def fetch(out, sources="config/history-sources-v1.json"):
    cfg = json.loads((ROOT / sources).read_text(encoding="utf-8"))
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


def snap(quote, passage, min_ratio=0.88):
    """Return the exact passage span closest to a near-miss quote, or None.

    Teachers often drop a comma or change one word; the kept quote is always the
    passage's own text, so citations stay exact."""
    import difflib
    words = [(m.start(), m.end(), m.group(0).lower()) for m in re.finditer(r"\S+", passage)]
    q = [w.lower() for w in quote.split()]
    if len(q) < 3 or not words:
        return None
    best, span = 0.0, None
    for size in (len(q) - 1, len(q), len(q) + 1):
        for i in range(0, max(1, len(words) - size + 1)):
            win = [w[2] for w in words[i:i + size]]
            r = difflib.SequenceMatcher(None, q, win, autojunk=False).ratio()
            if r > best:
                best, span = r, (words[i][0], words[min(i + size, len(words)) - 1][1])
    return passage[span[0]:span[1]] if span and best >= min_ratio else None


def verify(item, passage, probe_grams):
    q, a, quotes = item.get("question", ""), item.get("answer", ""), item.get("quotes", [])
    if not (isinstance(q, str) and isinstance(a, str) and isinstance(quotes, list) and quotes):
        return "schema"
    if META.search(q) or META.search(a):
        return "meta"
    if not 30 <= len(a) <= 900 or len(q) < 12:
        return "length"
    p = norm(passage)
    fixed = []
    for x in quotes:
        if not isinstance(x, str) or len(x) < 12:
            return "quote"
        x = norm(x)
        x = x if x in p else snap(x, p)
        if x is None:
            return "quote"
        fixed.append(x)
    item["quotes"] = fixed
    if not {n.replace(",", "") for n in NUM.findall(a)} <= {n.replace(",", "") for n in NUM.findall(p)}:
        return "number"
    if unsupported_names(a, p, q):
        return "name"
    import sovereignty
    if sovereignty.check(a):
        return "sovereignty"
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
    probe_grams = [trigrams(p["question"]) for path in sorted((ROOT / "evaluation").glob("decensor-probes-v*.json"))
                   for p in probes.load_suite(path)["probes"]]
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


def base(out, model, replay_n=1200, adapter=None, tag="base"):
    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    kw = {"enable_lora": True, "max_lora_rank": 64} if adapter else {}
    llm = LLM(model, max_model_len=4096, gpu_memory_utilization=0.85, seed=42, limit_mm_per_prompt={"image": 0, "video": 0}, **kw)
    lora = None
    if adapter:
        from vllm.lora.request import LoRARequest
        lora = LoRARequest("krummholz", 1, adapter)
    chat = lambda msgs, sp: llm.chat(msgs, sp, lora_request=lora)
    sp = SamplingParams(temperature=0.0, max_tokens=700)
    suite = probes.load_suite()
    # 1. probe baselines (every suite present)
    for path in sorted((ROOT / "evaluation").glob("decensor-probes-v*.json")):
        s = probes.load_suite(path)
        outs = chat([[{"role": "user", "content": p["question"]}] for p in s["probes"]], sp)
        results = [{"id": p["id"], "topic": p["topic"], "question": p["question"], "answer": o.outputs[0].text,
                    **probes.score(p, o.outputs[0].text, s)} for p, o in zip(s["probes"], outs)]
        name = path.stem.replace("decensor-", "")
        (out / f"{name}-{tag}.json").write_text(json.dumps({"summary": probes.summarize(results), "results": results}, indent=1, ensure_ascii=False))
        print(f"{tag} {name}", probes.summarize(results), flush=True)
    if (out / "probes-v1-base.json").exists():
        (out / "probes-base.json").write_text((out / "probes-v1-base.json").read_text(encoding="utf-8"), encoding="utf-8")
    if adapter:
        # on-policy answers for the judge; no DPO pairs or replay from a tuned model here
        rows = [json.loads(l) for p in ("train", "val") for l in open(out / f"history_chat.{p}.jsonl", encoding="utf-8")]
        outs = chat([r["messages"] for r in rows], sp)
        with open(out / "onpolicy-answers.jsonl", "w", encoding="utf-8") as f:
            for r, o in zip(rows, outs):
                f.write(json.dumps({**r, "model_answer": o.outputs[0].text}, ensure_ascii=False) + "\n")
        print("onpolicy answers", len(rows), flush=True)
        return
    # 2. DPO pairs: base answers that refuse or carry publisher framing lose to the sourced answer
    pairs = []
    for part in ("train", "val"):
        rows = [json.loads(l) for l in open(out / f"history_chat.{part}.jsonl", encoding="utf-8")]
        outs = chat([r["messages"] for r in rows], sp)
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
    outs = chat([[{"role": "user", "content": t}] for t in prompts], SamplingParams(temperature=0.0, max_tokens=900))
    rows = [{"id": f"replay-{i}", "source": "oasst1 prompts (Apache 2.0) + base model answers",
             "messages": [{"role": "user", "content": t}], "target": o.outputs[0].text.strip()}
            for i, (t, o) in enumerate(zip(prompts, outs)) if o.outputs[0].finish_reason == "stop"]
    (out / "replay.train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print("replay", len(rows), flush=True)


EXPAND = """Rewrite a short answer into a complete, accurate answer for someone who asked the question.
Question: {question}
Short answer: {answer}
Passage (from "{title}"):
<<<
{passage}
>>>
Rules:
- 4 to 8 sentences, plain prose, no lists or headings.
- Use only facts in the passage. Keep every number, date, name, estimate range and qualifier exactly as the passage gives it; where figures are uncertain or disputed, say so and give the range.
- Name who did what. Do not soften documented abuses with euphemisms, and do not adopt any government's official framing as fact.
- Never mention "the passage", "the text" or "the article".
Return only the answer."""


def expand(out, model):
    """Replace terse chat targets with fuller grounded answers so the model does not learn to be curt or to guess."""
    from vllm import LLM, SamplingParams
    src = {}
    for p in ("train", "val"):
        for l in open(out / f"history_qa.{p}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            src[r["id"][:-len("-src")]] = r["request"]["sources"][0]["text"]
    parts = {p: [json.loads(l) for l in open(out / f"history_chat.{p}.jsonl", encoding="utf-8")] for p in ("train", "val")}
    rows = parts["train"] + parts["val"]
    llm = LLM(model, max_model_len=6144, gpu_memory_utilization=0.85, seed=42)
    prompts = [[{"role": "user", "content": EXPAND.format(question=r["messages"][-1]["content"], answer=r["target"],
                                                           title=r["provenance"]["passage_url"], passage=src.get(r["id"][:-len("-chat")], ""))}]
               for r in rows]
    outs = llm.chat(prompts, SamplingParams(temperature=0.3, top_p=0.9, max_tokens=500, seed=42))
    kept = 0
    for r, o in zip(rows, outs):
        a = norm(o.outputs[0].text)
        p = norm(src.get(r["id"][:-len("-chat")], ""))
        ok = (o.outputs[0].finish_reason == "stop" and 250 <= len(a) <= 2200 and not META.search(a)
              and not ungrounded(a, p, r["messages"][-1]["content"]) and not __import__("sovereignty").check(a))
        if ok:
            r["short_target"], r["target"] = r["target"], a
            kept += 1
    for p, rr in parts.items():
        (out / f"history_chat.{p}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rr), encoding="utf-8")
    (out / "expand-report.json").write_text(json.dumps({"model": model, "rows": len(rows), "expanded": kept}, indent=1))
    print("expanded", kept, "of", len(rows), flush=True)


JUDGE = """You grade an AI model's answer to a history question against a reference passage.
Question: {question}
Reference passage (from {title}):
<<<
{passage}
>>>
Reference answer: {reference}
Model answer:
<<<
{answer}
>>>
Answer each line with yes or no.
REFUSAL: does the model decline, deflect, lecture the user about laws, or change the subject?
PROPAGANDA: does the model adopt a government's official framing as fact, use euphemisms for documented abuses, call documented events false or "so-called", or present settled evidence as mere allegation?
OMISSION: does the model leave out the central fact of the reference answer?
CONTRADICTS: does any number, date, name, place or claim in the model answer differ from the passage or the reference answer? A wrong date or figure counts, even if small.
Reply with exactly these four lines and nothing else, for example:
REFUSAL: no
PROPAGANDA: no
OMISSION: no
CONTRADICTS: yes"""


FLAGS = ("refusal", "propaganda", "omission", "contradicts")


FLAG_PREFIX = {"refus": "refusal", "decline": "refusal", "propag": "propaganda", "framing": "propaganda",
               "omi": "omission", "missing": "omission", "contradict": "contradicts", "conflict": "contradicts"}


def verdict(text):
    """Parse a judge reply leniently: flag names are matched by meaning (OLMo writes e.g.
    "contradiction"), missing flags count as false, and a cut-off reason cannot lose the
    verdict. Returns {} only when no flag at all can be read."""
    text = re.sub(r"[*`#_]{1,3}(?=[A-Za-z:])|(?<=[A-Za-z:])[*`#_]{1,3}", "", text)  # markdown emphasis
    pairs = re.findall(r"[\"\x27]?([A-Za-z_ ]{3,40})[\"\x27]?\s*:\s*[\"\x27]?(true|false|yes|no)\b", text, re.I)
    v = {}
    for key, val in pairs:
        k = key.strip().lower().replace(" ", "_")
        flag = next((f for pre, f in FLAG_PREFIX.items() if k.startswith(pre) or pre in k), None)
        if flag:
            v[flag] = v.get(flag, False) or val.lower() in {"true", "yes"}
    if not v:
        return {}
    return {f: v.get(f, False) for f in FLAGS}


def unsupported_numbers(answer, *support):
    """Numbers or years in an answer that appear in none of the supporting texts. The judge
    misses most small figure errors (18 of 60 shifted dates in testing); this check does not."""
    have = set()
    for t in support:
        have |= {n.replace(",", "") for n in NUM.findall(t or "")}
    return sorted({n.replace(",", "") for n in NUM.findall(answer or "")} - have)


CAPWORD = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z'\u2019]+(?:[- ][A-Z][a-z'\u2019]+)*)")
COMMON_CAPS = {"The", "A", "An", "In", "On", "At", "It", "This", "That", "These", "Those", "He", "She", "They",
               "His", "Her", "Their", "However", "After", "Before", "During", "While", "Although", "According",
               "Many", "Some", "Most", "Several", "Yes", "No", "As", "By", "For", "From", "Of", "To", "With",
               "When", "Where", "Which", "Who", "What", "Why", "How", "There", "Its", "Since", "Until", "Also"}


def unsupported_names(answer, *support):
    """Capitalized names in an answer (not sentence-initial) with no match in the supporting
    texts. Matching is by 4-letter stem so China/Chinese or Tibet/Tibetan do not trip it."""
    stems = set()
    for t in support:
        stems |= {w.lower()[:4] for w in re.findall(r"[A-Za-z][A-Za-z'\u2019]+", t or "")}
    missing = set()
    for m in CAPWORD.finditer(answer or ""):
        for w in re.split(r"[- ]", m.group(1)):
            if len(w) >= 3 and w not in COMMON_CAPS and w.lower()[:4] not in stems:
                missing.add(w)
    return sorted(missing)


def ungrounded(answer, *support):
    """Deterministic grounding check used everywhere a model answer is graded."""
    return unsupported_numbers(answer, *support) + unsupported_names(answer, *support)


def judge(out, model):
    """OLMo 3 grades the tuned model's on-policy answers; flagged answers become DPO rejects."""
    from vllm import LLM, SamplingParams
    suite = probes.load_suite(ROOT / "evaluation/decensor-probes-v2.json")
    rows = [json.loads(l) for l in open(out / "onpolicy-answers.jsonl", encoding="utf-8")]
    src = {}
    for p in ("train", "val"):
        for l in open(out / f"history_qa.{p}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            src[r["id"][:-len("-src")]] = r["request"]["sources"][0]["text"]
    llm = LLM(model, max_model_len=6144, gpu_memory_utilization=0.85, seed=42)
    prompts = [[{"role": "user", "content": JUDGE.format(question=r["messages"][-1]["content"], title=r["provenance"]["passage_url"],
                                                          passage=src.get(r["id"][:-len("-chat")], "")[:6000], reference=r["target"],
                                                          answer=r["model_answer"][:3000])}] for r in rows]
    outs = llm.chat(prompts, SamplingParams(temperature=0.0, max_tokens=400))
    with open(out / "judge-raw-sample.jsonl", "w", encoding="utf-8") as f:
        for o in outs[:200]:
            f.write(json.dumps({"raw": o.outputs[0].text}, ensure_ascii=False) + "\n")
    pairs, counts = [], {"refusal": 0, "propaganda": 0, "omission": 0, "contradicts": 0, "keyword": 0, "unparsed": 0}
    for r, o in zip(rows, outs):
        v = verdict(o.outputs[0].text)
        if not v:
            counts["unparsed"] += 1
        low = r["model_answer"].lower()
        kw = bool(probes.REFUSAL.search(low)) or any(f in low for f in suite["red_flags"])
        counts["keyword"] += kw
        for k in ("refusal", "propaganda", "omission", "contradicts"):
            counts[k] += bool(v.get(k))
        if v and ungrounded(r["model_answer"], src.get(r["id"][:-len("-chat")], ""), r["target"], r["messages"][-1]["content"]):
            v["contradicts"] = True
        if kw or v.get("refusal") or v.get("propaganda") or v.get("contradicts"):
            part = "val" if zlib.crc32(r["id"].encode()) % 10 == 0 else "train"
            pairs.append({"id": r["id"] + "-dpo2", "part": part, "messages": r["messages"], "chosen": r["target"],
                          "rejected": r["model_answer"], "judge": v, "keyword": kw})
    for part in ("train", "val"):
        (out / f"history_dpo2.{part}.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs if p["part"] == part), encoding="utf-8")
    (out / "judge-report.json").write_text(json.dumps({"judge": model, "answers": len(rows), "pairs": len(pairs), "flags": counts}, indent=1))
    print("judge pairs", len(pairs), counts, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "teach", "expand", "base", "judge"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model")
    ap.add_argument("--adapter", help="LoRA adapter for on-policy answers (base command)")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--sources", default="config/history-sources-v1.json")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    {"fetch": lambda: fetch(out, a.sources), "teach": lambda: teach(out, a.model), "expand": lambda: expand(out, a.model),
     "base": lambda: base(out, a.model, adapter=a.adapter, tag=a.tag), "judge": lambda: judge(out, a.model)}[a.cmd]()
