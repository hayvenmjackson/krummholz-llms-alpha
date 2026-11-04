"""Ask a running llama-server the multilingual probes (evaluation/multilingual-probes-v1.json) and save every answer.

Exploratory: nothing here is a release gate. Each answer gets only mechanical marks (the language it came back in, stray
Chinese in a non-Chinese answer, whether 397.8 appears for the arithmetic probe); judging the content is left to a
reader. Settings match the English evals: greedy decoding, no system prompt, thinking off.

    python3 tools/vl/multilingual.py <model name> <out.json> [--url http://127.0.0.1:8080]
"""
import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "evaluation" / "multilingual-probes-v1.json"
HEBREW = re.compile(r"[֐-׿]")
CJK = re.compile(r"[一-鿿]")
LATIN = re.compile(r"[A-Za-zÀ-ÿ]")
STOP = {
    "en": {"the", "and", "is", "of", "to", "in", "that", "it", "you", "are", "for", "with"},
    "es": {"el", "la", "los", "las", "de", "que", "y", "en", "es", "un", "una", "por", "para", "con", "del", "se"},
    "fr": {"le", "la", "les", "de", "des", "et", "est", "un", "une", "que", "en", "du", "pour", "dans", "pas", "je", "vous"},
}


def language(text):
    """Best guess at the main language of an answer: he, zh, or the Latin-script language with most stopwords."""
    he, zh, lat = len(HEBREW.findall(text)), len(CJK.findall(text)), len(LATIN.findall(text))
    if he > max(zh, lat) * 0.5 and he > 0:
        return "he"
    if zh > lat * 0.3 and zh > 0:
        return "zh"
    words = re.findall(r"[a-zà-ÿ']+", text.lower())
    counts = {k: sum(w in v for w in words) for k, v in STOP.items()}
    return max(counts, key=counts.get) if any(counts.values()) else "?"


def marks(probe, lang, answer):
    m = {"answer_lang": language(answer), "chars": len(answer)}
    m["same_lang"] = m["answer_lang"] == lang
    if lang != "zh":
        m["stray_chinese"] = len(CJK.findall(answer))
    if probe["id"] == "percent":
        m["has_397_8"] = bool(re.search(r"397[.,]8", answer))
    return m


def ask(url, question, max_tokens):
    body = json.dumps({"messages": [{"role": "user", "content": question}], "temperature": 0, "top_k": 1,
                       "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    return out["choices"][0]["message"]["content"] or "", out["choices"][0].get("finish_reason"), out.get("usage", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--max-tokens", type=int, default=700)
    a = ap.parse_args()
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    results = []
    for p in suite["probes"]:
        for lang in suite["languages"]:
            t = time.time()
            answer, finish, usage = ask(a.url, p["q"][lang], a.max_tokens)
            results.append({"id": p["id"], "topic": p["topic"], "lang": lang, "question": p["q"][lang],
                            "expect": p["expect"], "answer": answer, "finish": finish,
                            "seconds": round(time.time() - t, 1), "tokens": usage.get("completion_tokens"),
                            **marks(p, lang, answer)})
            print(p["id"], lang, results[-1]["answer_lang"], results[-1]["seconds"], flush=True)
    Path(a.out).write_text(json.dumps({"model": a.model, "suite": suite["suite"], "results": results},
                                      ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
