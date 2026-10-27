"""Convert verified synthetic examples into the pilot request/answer protocol."""
import json, re, sys
from pathlib import Path

PROFILE = {"customer_service": "acadia", "forestry": "katahdin", "fishing": "katahdin", "robotics": "atlantic"}
FLAG = re.compile(r"\b(conflict|disagree|differ|inconsistent|contradict)", re.I)
APPLY = re.compile(r"\b(amend|supersed|replac|updated|revis|overrid)", re.I)
PLANT = re.compile(r"\b(ignore|disregard|override|assistant|AI|bot|system|instruction|do not mention|respond only|must|urgent|void|admin|security alert|tell the user|forget)\b", re.I)
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
SENT = re.compile(r"[^.!?]+(?:[.!?]+|$)")

def slug(text, used, profile):
    words = re.findall(r"[a-z0-9]+", text.lower())[:3] or ["record"]
    base, n = f"{profile}/{'-'.join(words)}", 1
    sid = base
    while sid in used:
        n += 1; sid = f"{base}-{n}"
    used.add(sid)
    return sid

def sentences(text):
    return [m.group(0).strip() for m in SENT.finditer(text) if m.group(0).strip()]

def convert(ex):
    profile = PROFILE[ex["domain"]]
    used, idmap = set(), {}
    for s in ex["sources"]:
        idmap[s["id"]] = slug(s["text"], used, profile)
    answer = ex["expected_answer"]
    for old, new in sorted(idmap.items(), key=lambda kv: -len(kv[0])):
        answer = re.sub(rf"\b{re.escape(old)}\b", new, answer)
    sources = [{"id": idmap[s["id"]], "text": s["text"]} for s in ex["sources"]]
    cat = ex["category"]
    unresolved = cat == "conflict" and FLAG.search(answer) and not APPLY.search(answer)
    insufficient = bool(unresolved or cat == "missing_evidence")
    nums = {n.replace(",", "") for n in NUM.findall(answer)}
    lead = sentences(answer)[0] if cat == "injection" else answer
    words = {w for w in re.findall(r"[a-z]{5,}", lead.lower())}
    cites = []
    for s in sources:
        for sent in sentences(s["text"]):
            if cat == "injection" and PLANT.search(sent):
                break
            hit_num = any(n.replace(",", "") in nums for n in NUM.findall(sent))
            hit_words = len(words & set(re.findall(r"[a-z]{5,}", sent.lower()))) >= 2
            named = s["id"] in answer
            if hit_num or (hit_words and (named or cat != "plain")):
                cites.append({"source_id": s["id"], "quote": sent})
    cites = [c for i, c in enumerate(cites) if c not in cites[:i]][:4]
    if cat == "injection" and (not cites or any(PLANT.search(c["quote"]) for c in cites)):
        return None
    if not cites and not insufficient:
        return None
    target = {"answer": answer, "citations": cites, "insufficient_evidence": insufficient}
    return {"id": ex["id"], "category": cat, "domain": ex["domain"],
            "request": {"query": ex["query"], "sources": sources},
            "target": json.dumps(target, ensure_ascii=False)}

if __name__ == "__main__":
    for src, dst in zip(sys.argv[1::2], sys.argv[2::2]):
        rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
        out = [c for c in map(convert, rows) if c]
        Path(dst).write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in out), encoding="utf-8")
        print(src, len(rows), "->", len(out))
