"""Procure freely licensed domain text for Krummholz v7 and push it to a PRIVATE Hugging Face dataset repo.
Each output row: {"domain", "source", "license", "id", "text"}. Writes MANIFEST.json (counts, licenses, sha256)."""
import hashlib, json, os, sys, time, traceback
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from huggingface_hub import HfApi, hf_hub_download

OUT = Path("/root/domain"); OUT.mkdir(exist_ok=True)
REPO = os.environ.get("DOMAIN_REPO", "krummholz-ai/krummholz-v7-domain-data")
STATUS = Path("/root/www/status.txt")
CC = "CC BY 4.0 (KL3M, ALEA Institute); underlying U.S. government text is public domain"
KL3M = {
    "forestry": ["www.fs.usda.gov", "www.blm.gov", "www.nrcs.usda.gov", "www.ars.usda.gov", "www.fws.gov", "www.usgs.gov", "www.nps.gov", "www.aphis.usda.gov", "www.nifa.usda.gov"],
    "fishing": ["www.nmfs.noaa.gov", "www.noaa.gov", "oceanservice.noaa.gov", "www.nws.noaa.gov"],
    "customer_service": ["www.consumerfinance.gov", "www.ftc.gov", "www.consumer.gov", "consumer.gov", "www.sba.gov"],
    "workplace_robotics": ["www.osha.gov", "www.nist.gov"],
}
log = lambda *a: (print(*a, flush=True), STATUS.write_text(" ".join(map(str, a)) + "\n"))
manifest = []

def write(domain, source, license, rows):
    p = OUT / domain / (source.replace("/", "__") + ".jsonl"); p.parent.mkdir(exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for i, t in rows:
            if t and len(t.strip()) > 40:
                f.write(json.dumps({"domain": domain, "source": source, "license": license, "id": i, "text": t.strip()}, ensure_ascii=False) + "\n"); n += 1
    manifest.append({"file": str(p.relative_to(OUT)), "domain": domain, "source": source, "license": license, "rows": n,
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    log("wrote", domain, source, n)

def attempt(name, fn):
    try: fn()
    except Exception: log("FAILED", name); traceback.print_exc()

tok = Tokenizer.from_file(hf_hub_download("alea-institute/kl3m-004-128k-cased", "tokenizer.json"))
for domain, sites in KL3M.items():
    for site in sites:
        def go(site=site, domain=domain):
            ds = load_dataset(f"alea-institute/kl3m-filter-data-dotgov-{site}", split="train")
            ds = ds.filter(lambda r: r.get("q_decision", "keep") == "keep")
            write(domain, f"kl3m-dotgov/{site}", CC, ((r["identifier"], tok.decode(r["tokens"])) for r in ds))
        attempt(site, go)

def banking():
    ds = load_dataset("PolyAI/banking77", split="train")
    names = ds.features["label"].names
    write("customer_service", "PolyAI/banking77", "CC BY 4.0", ((f"b77-{i}", f"{r['text']}\tintent: {names[r['label']]}") for i, r in enumerate(ds)))
def clinc():
    ds = load_dataset("clinc/clinc_oos", "plus", split="train")
    names = ds.features["intent"].names
    write("customer_service", "clinc/clinc_oos", "CC BY 3.0", ((f"clinc-{i}", f"{r['text']}\tintent: {names[r['intent']]}") for i, r in enumerate(ds)))
def taskmaster():
    for v in ("taskmaster1", "taskmaster2", "taskmaster3"):
        def go(v=v):
            ds = load_dataset(f"google-research-datasets/{v}", split="train")
            def rows():
                for i, r in enumerate(ds):
                    utt = r.get("utterances") or []
                    yield (r.get("conversation_id", f"{v}-{i}"), "\n".join(f"{u.get('speaker','')}: {u.get('text','')}" for u in utt))
            write("customer_service", f"google-research-datasets/{v}", "CC BY 4.0", rows())
        attempt(v, go)
def arxiv():
    ds = load_dataset("gfissore/arxiv-abstracts-2021", split="train")
    keep = ("cs.CL", "cs.LG", "stat.ML", "cs.AI", "cs.RO")
    ds = ds.filter(lambda r: any(c in (r["categories"] or [""])[0].split() for c in keep))
    write("llm_training", "gfissore/arxiv-abstracts-2021 (cs.CL, cs.LG, stat.ML, cs.AI, cs.RO)", "CC0 1.0 (arXiv metadata)",
          ((r["id"], f"{r['title'].strip()}\n\n{r['abstract'].strip()}") for r in ds))
for n, fn in [("banking77", banking), ("clinc", clinc), ("taskmaster", taskmaster), ("arxiv", arxiv)]:
    attempt(n, fn)

(OUT / "MANIFEST.json").write_text(json.dumps({"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": manifest}, indent=1))
(OUT / "README.md").write_text("# Krummholz v7 domain data (private)\n\nFreely licensed domain text gathered for v7. Licenses per file in MANIFEST.json. "
    "KL3M rows are CC BY 4.0 (ALEA Institute) over public-domain U.S. government pages; attribution required. Not for redistribution until reviewed.\n")
api = HfApi()
api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
assert api.repo_info(REPO, repo_type="dataset").private, "repo is not private; refusing to upload"
api.upload_folder(folder_path=str(OUT), repo_id=REPO, repo_type="dataset", commit_message="Domain data for v7")
log("DOMAIN_DONE", sum(m["rows"] for m in manifest), "rows in", len(manifest), "files")
