"""Krummholz Library v0 (ADR-005, first slice): a dated passage corpus with BM25 search.

Pure Python, no dependencies. Built once from passage records (id, title, section, text,
url, license, revision), written as one JSON file with a SHA-256 manifest entry, and
queried by the tool broker's `search_library` (non-citable hits) and `fetch_library`
(citable text). Library text is data, never instructions.
"""
import hashlib, json, math, re
from collections import Counter
from pathlib import Path

FORMAT = "krummholz-library-v0"
TOKEN = re.compile(r"[a-z0-9]+")
STOP = frozenset("a an and are as at be by for from has have in is it its of on or that the this to was were what when where which who why with how did does do".split())
K1, B = 1.5, 0.75


def tokens(text):
    return [t for t in TOKEN.findall((text or "").lower()) if t not in STOP]


def build(records, as_of, name="krummholz-library"):
    docs, seen = [], set()
    for r in records:
        if r["id"] in seen or not r.get("text"):
            continue
        seen.add(r["id"])
        docs.append({"id": r["id"], "title": r.get("title", ""), "section": r.get("section", ""), "text": r["text"],
                     "url": r.get("url", ""), "license": r.get("license", ""), "revision": r.get("revision")})
    body = {"format": FORMAT, "name": name, "as_of": as_of, "documents": docs}
    raw = json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")
    body["sha256"] = hashlib.sha256(raw).hexdigest()
    return body


class Library:
    def __init__(self, data):
        if data.get("format") != FORMAT:
            raise ValueError("unknown_library_format")
        self.as_of, self.name = data["as_of"], data["name"]
        self.docs = {d["id"]: d for d in data["documents"]}
        self.ids = list(self.docs)
        self.tf, self.df, self.lens = [], Counter(), []
        for i in self.ids:
            d = self.docs[i]
            toks = tokens(d["title"] + " " + d["section"]) * 2 + tokens(d["text"])
            c = Counter(toks)
            self.tf.append(c); self.df.update(c.keys()); self.lens.append(len(toks))
        self.avg = sum(self.lens) / max(1, len(self.lens))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def idf(self, t):
        n = len(self.ids)
        return math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))

    def search(self, query, k=5, min_score=1.0):
        q = set(tokens(query))
        scored = []
        for j, c in enumerate(self.tf):
            s = 0.0
            for t in q:
                f = c.get(t)
                if f:
                    s += self.idf(t) * f * (K1 + 1) / (f + K1 * (1 - B + B * self.lens[j] / self.avg))
            if s >= min_score:
                scored.append((s, j))
        scored.sort(key=lambda x: (-x[0], self.ids[x[1]]))
        return [(self.ids[j], round(s, 3)) for s, j in scored[:k]]

    def hit(self, doc_id, result_id):
        d = self.docs[doc_id]
        return {"result_id": result_id, "title": d["title"] + (f" ({d['section']})" if d["section"] else ""),
                "snippet": d["text"][:280], "as_of": self.as_of, "untrusted_content": True, "citable": False}

    def evidence(self, doc_id, source_id):
        d = self.docs[doc_id]
        return {"source_id": source_id, "library": self.name, "as_of": self.as_of, "title": d["title"], "url": d["url"],
                "revision": d["revision"], "license": d["license"], "text": d["text"], "semantic_support": "not_determined"}
