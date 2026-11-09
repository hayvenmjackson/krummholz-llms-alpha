"""Procure freely licensed real-world text in Hebrew, Spanish, French and Mandarin for Krummholz v7 (equivalents of the
English domain data from procure_domain.py). Hebrew and Spanish get the most.

  python procure_multilingual.py --out DIR [--shards 12] [--upload REPO]

Sources (all ungated, permissive; each row records its license):
  - AmazonScience MASSIVE 1.1 intents (CC BY 4.0): he-IL, es-ES, fr-FR, zh-CN; the counterpart of CLINC/BANKING77
  - OpenAssistant oasst1 + oasst2 conversations (Apache 2.0): human-written, es/he/fr/zh
  - CohereLabs aya_dataset (Apache 2.0): human-written prompts and answers, es/fr/zh (no Hebrew in it)
  - clips/mqa FAQ pages (CC0 1.0): he/es/fr/zh question and answer pairs; kept when they match a domain
  - HPLT 2.0 cleaned web text (CC0 1.0 for the collection): he/es/fr/zh documents kept when they match a domain
Domain matching uses keyword lists (filters only; no generated text). Web and FAQ text is third-party writing; the
collection licenses cover the datasets, not each page, so treat those rows as grounding material and review before
any redistribution. Each output row: {"domain", "lang", "source", "license", "id", "text"}. Writes MANIFEST.json.
"""
import argparse, hashlib, io, json, re, tarfile, time, traceback, urllib.request
from pathlib import Path

KEYWORDS = {
    "forestry": {"he": ["יער", "יערות", "ייעור", "כריתה", "כריתת", "קק\"ל", "חורש", "נטיעה", "נטיעות", "עצי יער", "שריפת יער"],
                 "es": ["forestal", "forestales", "silvicultura", "bosque", "bosques", "madera", "incendio forestal", "reforestación"],
                 "fr": ["forestier", "forestière", "sylviculture", "forêt", "forêts", "bois d'œuvre", "reboisement"],
                 "zh": ["林业", "森林", "造林", "木材", "林场"]},
    "fishing": {"he": ["דיג", "דייג", "דייגים", "דגה", "ספינת דיג", "חכה", "רשתות דיג", "מכסות דיג"],
                "es": ["pesca", "pesquero", "pesqueros", "pesquería", "caladero", "cuota de captura", "flota pesquera"],
                "fr": ["pêche", "halieutique", "chalutier", "quota de pêche", "pêcheurs"],
                "zh": ["渔业", "捕捞", "渔船", "渔民", "水产"]},
    "customer_service": {"he": ["שירות לקוחות", "החזר כספי", "ביטול עסקה", "משלוח", "משלוחים", "חשבונית", "תלונה", "מוקד"],
                         "es": ["atención al cliente", "reembolso", "devolución", "factura", "reclamación", "envío"],
                         "fr": ["service client", "remboursement", "réclamation", "facture", "livraison"],
                         "zh": ["客服", "退款", "售后", "订单", "发票"]},
    "workplace_robotics": {"he": ["רובוט", "רובוטים", "רובוטיקה", "אוטומציה", "בטיחות בעבודה", "קו ייצור"],
                           "es": ["robot", "robots", "robótica", "automatización", "seguridad laboral", "prevención de riesgos"],
                           "fr": ["robot", "robots", "robotique", "automatisation", "sécurité au travail"],
                           "zh": ["机器人", "自动化", "安全生产", "工业机器人"]},
    "llm_training": {"he": ["למידת מכונה", "בינה מלאכותית", "מודל שפה", "מודלי שפה", "רשת נוירונים", "למידה עמוקה"],
                     "es": ["aprendizaje automático", "inteligencia artificial", "modelo de lenguaje", "red neuronal", "aprendizaje profundo"],
                     "fr": ["apprentissage automatique", "intelligence artificielle", "modèle de langage", "réseau de neurones", "apprentissage profond"],
                     "zh": ["机器学习", "人工智能", "语言模型", "神经网络", "深度学习"]},
}
# whole-word patterns: Hebrew allows up to two one-letter prefixes (ה ו ב ל מ ש כ) and no letters after; Spanish and French use
# word boundaries; Mandarin has no word spacing, so substrings
def _pat(k, lang):
    k = re.escape(k.lower())
    if lang == "he": return re.compile(r"(?<![א-ת])[הובלמשכ]{0,2}" + k + r"(?![א-ת])")
    if lang == "zh": return re.compile(k)
    return re.compile(r"(?<!\w)" + k + r"(?!\w)")
PATS = {d: {l: [_pat(k, l) for k in ks] for l, ks in kw.items()} for d, kw in KEYWORDS.items()}
HPLT = {"he": "heb_Hebr", "es": "spa_Latn", "fr": "fra_Latn", "zh": "zho_Hans_1"}
MQA = {"he": "he-faq-question", "es": "es-faq-question", "fr": "fr-faq-question", "zh": "zh-faq-question"}
MASSIVE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
MASSIVE = {"he-IL": "he", "es-ES": "es", "fr-FR": "fr", "zh-CN": "zh"}


def match(text, lang, need, distinct=1):
    """Domain with the most keyword hits, if it has at least `need` hits from at least `distinct` different keywords."""
    t = text.lower(); best, score = None, (0, 0)
    for dom, pats in PATS.items():
        counts = [len(p.findall(t)) for p in pats[lang]]
        sc = (sum(1 for c in counts if c), sum(counts))
        if sc > score: best, score = dom, sc
    return best if score[1] >= need and score[0] >= distinct else None


def fetch(url, tries=8):
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "krummholz-procure/1"}), timeout=300).read()
        except Exception:
            if i == tries - 1: raise
            time.sleep(15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--shards", type=int, default=12)
    ap.add_argument("--cap", type=int, default=20000, help="max rows per (source, lang, domain)")
    ap.add_argument("--upload"); ap.add_argument("--only", nargs="*", help="run only these steps, e.g. mqa-he hplt-es")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    manifest = []

    def write(name, license, rows):
        p = out / (name.replace("/", "__") + ".jsonl"); n = 0; by = {}
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                key = (r["lang"], r["domain"])
                if by.get(key, 0) >= a.cap or len(r["text"].strip()) < 20: continue
                by[key] = by.get(key, 0) + 1; n += 1
                f.write(json.dumps({**r, "source": name, "license": license, "text": r["text"].strip()}, ensure_ascii=False) + "\n")
        manifest.append({"file": p.name, "source": name, "license": license, "rows": n,
                         "by_lang_domain": {f"{l}/{d}": c for (l, d), c in sorted(by.items())},
                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
        print("wrote", name, n, flush=True)

    old = out / "MANIFEST.json"
    if a.only and old.exists():  # partial rerun: keep the other sources' manifest entries
        tags = {"massive": "AmazonScience/massive", "oasst": "OpenAssistant/", "aya": "CohereLabs/aya"}
        tags.update({f"mqa-{l}": f"clips/mqa ({MQA[l]}" for l in MQA}); tags.update({f"hplt-{l}": f"HPLT/HPLT2.0_cleaned ({HPLT[l]}" for l in HPLT})
        redo = [tags[o] for o in a.only if o in tags]
        manifest.extend(m for m in json.loads(old.read_text(encoding="utf-8"))["files"] if not any(m["source"].startswith(r) for r in redo))

    def attempt(name, fn):
        if a.only and name not in a.only: return
        try: fn()
        except Exception: print("FAILED", name, flush=True); traceback.print_exc()

    def massive():
        tf = tarfile.open(fileobj=io.BytesIO(fetch(MASSIVE_URL)), mode="r:gz"); rows = []
        for m in tf.getmembers():
            loc = Path(m.name).stem
            if loc in MASSIVE and m.name.endswith(".jsonl"):
                for line in tf.extractfile(m).read().decode("utf-8").splitlines():
                    r = json.loads(line)
                    if r.get("partition") == "test": continue  # keep the test split unseen
                    rows.append({"domain": "customer_service", "lang": MASSIVE[loc], "id": f"massive-{loc}-{r['id']}",
                                 "text": f"{r['utt']}\tintent: {r['intent']}\tscenario: {r['scenario']}"})
        write("AmazonScience/massive-1.1 (train+dev)", "CC BY 4.0", rows)

    def oasst():
        from datasets import load_dataset
        rows = []
        for name in ("OpenAssistant/oasst1", "OpenAssistant/oasst2"):
            ds = load_dataset(name, split="train")
            msgs = {r["message_id"]: r for r in ds if r["lang"] in ("he", "es", "fr", "zh")}
            for r in msgs.values():
                if r["role"] == "assistant" and r["parent_id"] in msgs and (r.get("rank") in (0, None)):
                    q = msgs[r["parent_id"]]
                    rows.append({"domain": match(q["text"] + " " + r["text"], r["lang"], 1) or "general", "lang": r["lang"],
                                 "id": f"{name.split('/')[1]}-{r['message_id']}", "text": f"Q: {q['text']}\nA: {r['text']}"})
        write("OpenAssistant/oasst1+oasst2", "Apache 2.0", rows)

    def aya():
        from datasets import load_dataset
        code = {"Spanish": "es", "French": "fr", "Simplified Chinese": "zh"}
        ds = load_dataset("CohereLabs/aya_dataset", split="train").filter(lambda r: r["language"] in code)
        write("CohereLabs/aya_dataset", "Apache 2.0", ({"domain": match(r["inputs"] + " " + r["targets"], code[r["language"]], 1) or "general",
               "lang": code[r["language"]], "id": f"aya-{i}", "text": f"Q: {r['inputs']}\nA: {r['targets']}"} for i, r in enumerate(ds)))

    def mqa(lang):
        from datasets import load_dataset
        # the repo's loader script no longer runs in datasets>=4; read Hugging Face's parquet conversion instead
        # large configs are only partly converted (a "partial-train" folder, the first ~5 GB); that is plenty here
        base = f"hf://datasets/clips/mqa@refs%2Fconvert%2Fparquet/{MQA[lang]}"
        from huggingface_hub import HfApi
        sub = [f.path.split("/")[-1] for f in HfApi().list_repo_tree("clips/mqa", path_in_repo=MQA[lang], repo_type="dataset",
                                                                     revision="refs/convert/parquet")]
        ds = load_dataset("parquet", data_files=f"{base}/{'train' if 'train' in sub else 'partial-train'}/*.parquet",
                          split="train", streaming=True)
        def rows():
            for i, r in enumerate(ds):
                if i > 2_000_000: break
                ans = " ".join(x.get("text", "") for x in (r.get("answers") or [])[:1])
                text = f"{r.get('name') or ''}\n{r.get('text') or ''}\n{ans}".strip()
                d = match(text, lang, 1)
                if d: yield {"domain": d, "lang": lang, "id": f"mqa-{lang}-{r.get('id', i)}", "text": f"Q: {r.get('name') or r.get('text')}\nA: {ans}"}
        write(f"clips/mqa ({MQA[lang]}, domain-matched)", "CC0 1.0", rows())

    def hplt(lang):
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        cfg = HPLT[lang]; n = a.shards if lang in ("he", "es") else max(2, a.shards // 4)
        def rows():
            for s in range(n):
                fname = f"{cfg}/train-{s:05d}-of-"
                files = [f for f in listing(cfg) if f.startswith(fname)]
                if not files: break
                p = hf_hub_download("HPLT/HPLT2.0_cleaned", files[0], repo_type="dataset")
                t = pq.read_table(p, columns=["text", "id"] if "id" in pq.read_schema(p).names else ["text"])
                for i, row in enumerate(t.to_pylist()):
                    text = row["text"]
                    if 300 <= len(text) <= 20000:
                        d = match(text, lang, 3, distinct=2)
                        if d: yield {"domain": d, "lang": lang, "id": f"hplt-{lang}-{row.get('id', f'{s}-{i}')}", "text": text}
                Path(p).unlink(missing_ok=True)  # keep the disk small
        write(f"HPLT/HPLT2.0_cleaned ({cfg}, domain-matched)", "CC0 1.0 (collection)", rows())

    _ls = {}
    def listing(cfg):
        if cfg not in _ls:
            from huggingface_hub import HfApi
            _ls[cfg] = sorted(f.path for f in HfApi().list_repo_tree("HPLT/HPLT2.0_cleaned", path_in_repo=cfg, repo_type="dataset"))
        return _ls[cfg]

    for name, fn in [("massive", massive), ("oasst", oasst), ("aya", aya)]:
        attempt(name, fn)
    for lang in ("he", "es", "fr", "zh"):
        attempt(f"mqa-{lang}", lambda lang=lang: mqa(lang))
    for lang in ("he", "es", "fr", "zh"):
        attempt(f"hplt-{lang}", lambda lang=lang: hplt(lang))

    (out / "MANIFEST.json").write_text(json.dumps({"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": manifest},
                                                  indent=1, ensure_ascii=False), encoding="utf-8")
    total = sum(m["rows"] for m in manifest)
    print("MULTILINGUAL_PROCURE_DONE", total, "rows in", len(manifest), "files", flush=True)
    if a.upload:
        from huggingface_hub import HfApi
        api = HfApi(); api.create_repo(a.upload, repo_type="dataset", private=True, exist_ok=True)
        assert api.repo_info(a.upload, repo_type="dataset").private, "repo is not private; refusing to upload"
        api.upload_folder(folder_path=str(out), repo_id=a.upload, repo_type="dataset", path_in_repo="multilingual",
                          commit_message="Multilingual real-world data for v7")
        print("UPLOADED", a.upload, flush=True)


if __name__ == "__main__":
    main()
