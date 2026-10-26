"""Bounded, review-driven CC0 text intake. Never trains or publishes a corpus.

Python 3.12+, pypdf==6.10.0, pdfplumber==0.11.9; Node 22+ for an HTTPS fallback.
Fetch/extract first, then select explicit line
ranges in the review file. Cleaning fails closed on changed source text.
Raw/evidence/extracted/clean files stay in the ignored local output directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import time
import unicodedata
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/public-domain-pilot"
MAX_PDF_BYTES = 16 * 1024 * 1024
MAX_HTML_BYTES = 3 * 1024 * 1024
ALLOWED_HOSTS = {"repository.library.noaa.gov", "journals.plos.org", "creativecommons.org"}
VERSION = "cc0-selected-prose-v1"
EXTRACTOR = "pdfplumber==0.11.9; x_tolerance=2; y_tolerance=3"
PRIMARY_PDF_ALTERNATES = {
    # Link inspected on the primary publisher article; supplements excluded.
    "plos-metabarcoding-signal-noise-2023": "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0285674&type=printable",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, text):
        self.parts.append(text)


def html_text(data: bytes) -> str:
    parser = PlainText()
    parser.feed(data.decode("utf-8", errors="replace"))
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def allowed(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.hostname in ALLOWED_HOSTS and not parsed.username


class RestrictedRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        if not allowed(newurl):
            raise ValueError("redirect outside the reviewed primary-source hosts")
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def node_fetch(url: str, path: Path, limit: int) -> dict:
    """Alternative standard HTTPS client; no cookies, credentials or TLS bypass."""
    temporary = path.with_suffix(path.suffix + ".node-download")
    if temporary.exists():
        raise ValueError("stale download must be inspected before retry")
    script = r'''
const fs = require('node:fs');
const crypto = require('node:crypto');
const [initial, destination, maximum] = process.argv.slice(1);
const limit = Number(maximum);
const allowed = new Set(['repository.library.noaa.gov','journals.plos.org','creativecommons.org']);
(async()=>{
 let url=initial, response;
 for(let redirects=0;redirects<6;redirects++) {
  const parsed=new URL(url);
  if(parsed.protocol!=='https:' || !allowed.has(parsed.hostname) || parsed.username) throw Error('unapproved redirect');
  response=await fetch(url,{redirect:'manual',signal:AbortSignal.timeout(45000)});
  if([301,302,303,307,308].includes(response.status)) {
   const location=response.headers.get('location'); if(!location) throw Error('missing redirect URL');
   await response.body.cancel(); url=new URL(location,url).href; response=undefined; continue;
  }
  break;
 }
 if(!response || !response.ok) throw Error('HTTPS fetch status '+response?.status);
 const chunks=[]; let size=0;
 for await(const chunk of response.body) { size+=chunk.length; if(size>limit) throw Error('size limit exceeded'); chunks.push(chunk); }
 const bytes=Buffer.concat(chunks);
 fs.writeFileSync(destination,bytes,{flag:'wx'});
 console.log(JSON.stringify({requested_url:initial,final_url:url,retrieved_at:new Date().toISOString(),bytes:size,
  sha256:crypto.createHash('sha256').update(bytes).digest('hex'),content_type:response.headers.get('content-type'),
  etag:response.headers.get('etag'),last_modified:response.headers.get('last-modified'),transport:'Node standard fetch; no cookies/authentication'}));
})().catch(e=>{console.error(e.message);process.exitCode=1;});
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["node", "-e", script, url, str(temporary), str(limit)],
                            capture_output=True, text=True, check=True, timeout=300)
    receipt = json.loads(result.stdout)
    data = temporary.read_bytes()
    if len(data) > limit or sha(data) != receipt["sha256"] or not allowed(receipt["final_url"]):
        raise ValueError("fallback download validation failed")
    if path.exists():
        raise ValueError("source appeared during download; refusing replacement")
    temporary.replace(path)
    write_json(path.with_suffix(path.suffix + ".receipt.json"), receipt)
    return receipt


def fetch(url: str, path: Path, limit: int) -> dict:
    if not allowed(url):
        raise ValueError("URL is outside the reviewed primary-source hosts")
    receipt = path.with_suffix(path.suffix + ".receipt.json")
    if path.exists() and receipt.exists():
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        data = path.read_bytes()
        if previous["requested_url"] != url or previous["sha256"] != sha(data) or len(data) > limit:
            raise ValueError("cached source or receipt changed; refusing silent replacement")
        return previous
    if path.exists() or receipt.exists():
        raise ValueError("incomplete cached source requires explicit inspection")
    opener = build_opener(RestrictedRedirect())
    error = None
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "Krummholz-CC0-Research-Intake/0.1"})
            with opener.open(request, timeout=45) as response:
                data = response.read(limit + 1)
                if len(data) > limit:
                    raise ValueError("source exceeds bounded intake size")
                metadata = {"requested_url": url, "final_url": response.geturl(),
                            "retrieved_at": utc(), "bytes": len(data), "sha256": sha(data),
                            "content_type": response.headers.get_content_type(),
                            "etag": response.headers.get("ETag"),
                            "last_modified": response.headers.get("Last-Modified")}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            write_json(receipt, metadata)
            return metadata
        except Exception as exc:
            error = exc
            if isinstance(exc, HTTPError) and exc.code == 403:
                return node_fetch(url, path, limit)
            if attempt < 2:
                time.sleep(attempt + 1)
    raise error


def explicit_cc0(data: bytes) -> bool:
    text = html_text(data).lower()
    return ("cc0 public domain" in text and "creativecommons.org/publicdomain/zero/1.0" in data.decode("utf-8", "replace")) or "creative commons cc0 public domain dedication" in text


def selected_sources() -> list[dict]:
    register = json.loads((ROOT / "config/public-domain-text-sources.json").read_text(encoding="utf-8"))
    sources = [s for s in register["sources"] if s["rights"]["classification"] == "CC0-1.0"]
    if len(sources) != 6 or len({s["id"] for s in sources}) != 6:
        raise ValueError("expected exactly the six reviewed CC0 candidates")
    return sources


def acquire(output: Path) -> None:
    import pypdf
    import pdfplumber
    if pypdf.__version__ != "6.10.0" or pdfplumber.__version__ != "0.11.9":
        raise ValueError("use pypdf==6.10.0 and pdfplumber==0.11.9")
    report = {"schema_version": 1, "started_at": utc(), "tool_version": VERSION,
              "extractor": EXTRACTOR, "training_started": False,
              "published": False, "sources": []}
    for source in selected_sources():
        sid = source["id"]
        item = {"id": sid, "source_group_id": source["source_group_id"],
                "partition": source["proposed_partition"], "title": source["title"],
                "publication_date": source["publication_date"], "publisher": source["publisher"],
                "rights": {"classification": "CC0-1.0", "evidence": []}, "status": "acquiring"}
        try:
            urls = [source["rights"]["evidence_url"]]
            corroboration = source["rights"].get("corroborating_evidence_url")
            if corroboration:
                urls.append(corroboration)
            for index, url in enumerate(urls):
                evidence_path = output / "evidence" / f"{sid}-{index}.html"
                try:
                    receipt = fetch(url, evidence_path, MAX_HTML_BYTES)
                    receipt["explicit_cc0_found"] = explicit_cc0(evidence_path.read_bytes())
                    receipt["path"] = str(evidence_path.relative_to(output)).replace("\\", "/")
                    item["rights"]["evidence"].append(receipt)
                except Exception as exc:
                    item["rights"]["evidence"].append({"requested_url": url, "status": "fetch_failed", "error": type(exc).__name__})
            if not any(e.get("explicit_cc0_found") is True for e in item["rights"]["evidence"]):
                raise ValueError("no saved primary item-level CC0 notice; item held")
            pdf = output / "raw" / f"{sid}.pdf"
            document_url = PRIMARY_PDF_ALTERNATES.get(sid, source["document_url"])
            item["document"] = fetch(document_url, pdf, MAX_PDF_BYTES)
            item["document"]["registered_url"] = source["document_url"]
            item["document"]["path"] = str(pdf.relative_to(output)).replace("\\", "/")
            raw = pdf.read_bytes()
            if not raw.startswith(b"%PDF-"):
                raise ValueError("download is not a PDF")
            checksum_matches = []
            for evidence in item["rights"]["evidence"]:
                if "path" not in evidence:
                    continue
                content = html_text((output / evidence["path"]).read_bytes())
                for algorithm, digest in re.findall(r"urn:sha-?(256|512):([0-9a-fA-F]+)", content):
                    expected_len = 64 if algorithm == "256" else 128
                    if len(digest) != expected_len:
                        continue
                    actual = hashlib.new("sha" + algorithm, raw).hexdigest()
                    if actual != digest.lower():
                        raise ValueError("official repository document checksum mismatch")
                    checksum_matches.append({"algorithm": "sha" + algorithm, "digest": digest.lower(), "matches": True})
            item["document"]["repository_checksum_checks"] = checksum_matches
            reader = pypdf.PdfReader(pdf)
            if reader.is_encrypted or not 1 <= len(reader.pages) <= 200:
                raise ValueError("encrypted or unexpectedly large page count")
            pages = []
            with pdfplumber.open(pdf) as parsed:
                for number, page in enumerate(parsed.pages, 1):
                    text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
                    text = text.replace("\r\n", "\n").replace("\r", "\n")
                    lines = text.splitlines()
                    pages.append({"page": number, "text_sha256": sha(text.encode("utf-8")), "lines": lines})
                    page.close()
            extracted = output / "extracted" / f"{sid}.pages.json"
            write_json(extracted, {"source_id": sid, "document_sha256": item["document"]["sha256"],
                                   "extractor": EXTRACTOR, "pages": pages})
            text_path = extracted.with_suffix(".txt")
            text_path.write_text("\n\n".join(f"=== PDF PAGE {p['page']} ===\n" + "\n".join(f"{i:03d} {line}" for i, line in enumerate(p["lines"], 1)) for p in pages), encoding="utf-8")
            item["extraction"] = {"path": str(extracted.relative_to(output)).replace("\\", "/"),
                                  "sha256": sha(extracted.read_bytes()), "pages": len(pages),
                                  "characters": sum(sum(len(l) for l in p["lines"]) for p in pages)}
            item["status"] = "acquired_awaiting_passage_review"
        except Exception as exc:
            item["status"] = "hold_acquisition_or_rights"
            item["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        report["sources"].append(item)
        write_json(output / "acquisition.json", report)
        print(json.dumps({"id": sid, "status": item["status"], "pages": item.get("extraction", {}).get("pages")}))
    report["completed_at"] = utc()
    write_json(output / "acquisition.json", report)


def normalize(lines: list[str]) -> str:
    # Preserve punctuation, signs and scientific values. No guessed OCR repair.
    value = unicodedata.normalize("NFC", " ".join(line.strip() for line in lines))
    return re.sub(r"\s+", " ", value).strip()


def risk_flags(text: str) -> list[str]:
    patterns = {
        "email": r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
        "url": r"(?:https?://|www\.)\S+",
        "phone_or_orcid": r"(?<!\d)(?:\+?1[- .]?)?\(?\d{3}\)?[- .]\d{3}[- .]\d{4}(?!\d)|\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b",
        "rights_or_credit": r"\b(?:copyright|reproduced|reprinted|permission|courtesy|all rights reserved)\b|©",
        "personal_frontmatter": r"\b(?:corresponding author|affiliation|acknowledg(?:e)?ments?|e-mail)\b",
        "replacement_character": r"\ufffd",
    }
    return [key for key, pattern in patterns.items() if re.search(pattern, text, re.I)]


def shingles(text: str) -> set[tuple]:
    words = re.findall(r"\w+", text.casefold())
    return {tuple(words[i:i + 5]) for i in range(max(0, len(words) - 4))}


def reviewed_text(page: dict, selection: dict) -> str:
    start, end = selection["start_line"], selection["end_line"]
    if page["page"] != selection["page"] or not 1 <= start <= end <= len(page["lines"]):
        raise ValueError("invalid reviewed line interval")
    original = "\n".join(page["lines"][start - 1:end])
    if sha(original.encode("utf-8")) != selection["extracted_span_sha256"]:
        raise ValueError("reviewed text changed; do not silently repair")
    text = normalize(page["lines"][start - 1:end])
    for before, after in selection.get("reviewed_line_wrap_replacements", []):
        if not before or before not in text:
            raise ValueError("reviewed line-wrap replacement no longer matches")
        text = text.replace(before, after)
    for citation in selection.get("reviewed_reference_removals", []):
        if not citation or citation not in text:
            raise ValueError("reviewed reference removal no longer matches")
        text = text.replace(citation, "")
    return re.sub(r"\s+", " ", text).strip()


def clean(output: Path, review_path: Path) -> None:
    acquisition = json.loads((output / "acquisition.json").read_text(encoding="utf-8"))
    reviews = json.loads(review_path.read_text(encoding="utf-8"))
    if reviews["schema_version"] != 1:
        raise ValueError("unknown review schema")
    decisions = {r["source_id"]: r for r in reviews["sources"]}
    sources = {s["id"]: s for s in selected_sources()}
    if len(decisions) != len(reviews["sources"]) or set(decisions) != set(sources):
        raise ValueError("review must cover each of the six source IDs exactly once")
    if {s["id"] for s in acquisition["sources"]} != set(sources) or len(acquisition["sources"]) != 6:
        raise ValueError("acquisition manifest must contain the six source IDs exactly once")
    kept, audit, rejected = [], [], []
    seen, previous_shingles = {}, []
    for item in acquisition["sources"]:
        sid = item["id"]
        if (item["partition"] != sources[sid]["proposed_partition"] or
                item["source_group_id"] != sources[sid]["source_group_id"]):
            raise ValueError("acquisition changed a source-family split")
        review = decisions.get(sid)
        result = {"source_id": sid, "source_group_id": item["source_group_id"], "partition": item["partition"],
                  "status": "hold_unreviewed", "retained_passages": 0, "words": 0}
        audit.append(result)
        if item["status"] != "acquired_awaiting_passage_review" or not review:
            continue
        result["review"] = review
        if review["document_sha256"] != item["document"]["sha256"]:
            raise ValueError(f"review source hash mismatch: {sid}")
        raw_path = (output / item["document"]["path"]).resolve()
        if not raw_path.is_relative_to(output) or sha(raw_path.read_bytes()) != item["document"]["sha256"]:
            raise ValueError("acquired PDF changed or left the local intake directory")
        for evidence in item["rights"]["evidence"]:
            if "path" not in evidence:
                continue
            evidence_path = (output / evidence["path"]).resolve()
            if not evidence_path.is_relative_to(output) or sha(evidence_path.read_bytes()) != evidence["sha256"]:
                raise ValueError("saved rights evidence changed")
            if evidence.get("explicit_cc0_found") and not explicit_cc0(evidence_path.read_bytes()):
                raise ValueError("saved primary rights notice no longer matches CC0 evidence")
        if review["status"] == "hold":
            result["status"] = "hold_review"
            continue
        if review["status"] != "reviewed_selected_prose" or not review.get("all_extracted_pages_credit_scan"):
            raise ValueError("review must identify a completed document credit check")
        extract_path = output / item["extraction"]["path"]
        if sha(extract_path.read_bytes()) != item["extraction"]["sha256"]:
            raise ValueError("extraction changed after acquisition")
        pages = json.loads(extract_path.read_text(encoding="utf-8"))["pages"]
        for index, selection in enumerate(review["passages"], 1):
            page = pages[selection["page"] - 1]
            start, end = selection["start_line"], selection["end_line"]
            text = reviewed_text(page, selection)
            flags = risk_flags(text)
            if flags:
                rejected.append({"source_id": sid, "selection": index, "reason": flags})
                continue
            digest = sha(text.casefold().encode("utf-8"))
            words = len(text.split())
            if not 20 <= words <= 1000:
                rejected.append({"source_id": sid, "selection": index, "reason": ["length_outside_20_1000_words"]})
                continue
            if digest in seen:
                rejected.append({"source_id": sid, "selection": index, "reason": ["exact_duplicate"], "duplicate_of": seen[digest]})
                continue
            grams = shingles(text)
            duplicate = next((rid for rid, prior in previous_shingles if grams and prior and len(grams & prior) / min(len(grams), len(prior)) >= 0.8), None)
            if duplicate:
                rejected.append({"source_id": sid, "selection": index, "reason": ["near_duplicate_5gram_containment"], "duplicate_of": duplicate})
                continue
            record_id = f"{sid}-p{selection['page']}-s{index}"
            seen[digest] = record_id
            previous_shingles.append((record_id, grams))
            record = {"id": record_id, "source_id": sid, "source_group_id": item["source_group_id"],
                      "partition": item["partition"], "retained_for_training_candidate": item["partition"] == "train",
                      "text": text, "text_sha256": sha(text.encode("utf-8")), "words": words,
                      "document_url": sources[sid]["document_url"], "document_sha256": item["document"]["sha256"],
                      "acquired_document_url": item["document"]["final_url"], "publication_date": item["publication_date"],
                      "source_pdf_page": selection["page"], "source_lines": [start, end],
                      "extracted_span_sha256": selection["extracted_span_sha256"],
                      "reviewed_line_wrap_replacements": selection.get("reviewed_line_wrap_replacements", []),
                      "reference_removals_count": len(selection.get("reviewed_reference_removals", [])),
                      "section": selection["section"], "license": "CC0-1.0", "rights_evidence": item["rights"]["evidence"],
                      "reviewer": review["reviewer"], "reviewed_on": review["reviewed_on"],
                      "filter_version": VERSION, "training_started": False,
                      "training_authorized_by_this_intake": False}
            kept.append(record)
            result["retained_passages"] += 1
            result["words"] += words
        result["status"] = "selected_prose_retained" if result["retained_passages"] else "hold_no_passages_passed"
    if sum(r["words"] for r in kept) > 150000:
        raise ValueError("pilot clean-word budget exceeded")
    target = output / "clean"
    target.mkdir(parents=True, exist_ok=True)
    files = {}
    for partition in ("train", "dev", "test"):
        rows = [r for r in kept if r["partition"] == partition]
        path = target / f"{partition}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        files[partition] = {"path": str(path.relative_to(output)).replace("\\", "/"), "sha256": sha(path.read_bytes()),
                            "records": len(rows), "words": sum(r["words"] for r in rows)}
    manifest = {"schema_version": 1, "completed_at": utc(), "filter_version": VERSION,
                "review_file": str(review_path.relative_to(ROOT)).replace("\\", "/"), "review_sha256": sha(review_path.read_bytes()),
                "acquisition_sha256": sha((output / "acquisition.json").read_bytes()), "files": files,
                "selected_text_secondary_review": reviews.get("selected_text_secondary_review"),
                "sources": audit, "rejected_selections": rejected, "records": len(kept), "words": sum(r["words"] for r in kept),
                "training_started": False, "published": False, "whole_documents_cleared": False,
                "limits": ["Only explicit selected prose is retained; other pages, equations, tables, figures and back matter are excluded.",
                           "Agent review plus pattern checks is not a guarantee of absence of personal data or third-party rights.",
                           "Deduplication covers retained pilot passages only; wider-corpus overlap and inherited base exposure remain unverified."]}
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"status": "cleaned_without_training_or_publication", "records": len(kept), "words": manifest["words"], "rejected": len(rejected)}))


def self_test() -> None:
    assert explicit_cc0(b'<a href="https://creativecommons.org/publicdomain/zero/1.0/">License</a><p>CC0 Public Domain</p>')
    assert not explicit_cc0(b"Open access. CC-BY 4.0. All rights reserved.")
    assert normalize(["measurement", " uncertainty"]) == "measurement uncertainty"
    assert risk_flags("Reach person@example.invalid for details") == ["email"]
    assert "rights_or_credit" in risk_flags("Reprinted with permission.")
    assert "phone_or_orcid" in risk_flags("0000-0002-1234-567X")
    assert not allowed("https://repository.library.noaa.gov.evil.invalid/paper.pdf")
    assert not allowed("http://repository.library.noaa.gov/paper.pdf")
    assert not allowed("https://user@repository.library.noaa.gov/paper.pdf")
    assert len(selected_sources()) == 6
    page = {"page": 2, "lines": ["A fictional measure-", "ment uses metres (Example, 2020)."]}
    selection = {"page": 2, "start_line": 1, "end_line": 2,
                 "extracted_span_sha256": sha("\n".join(page["lines"]).encode("utf-8")),
                 "reviewed_line_wrap_replacements": [["measure- ment", "measurement"]],
                 "reviewed_reference_removals": [" (Example, 2020)"]}
    assert reviewed_text(page, selection) == "A fictional measurement uses metres."
    for bad in (dict(selection, page=3), dict(selection, start_line=0),
                dict(selection, extracted_span_sha256="0" * 64),
                dict(selection, reviewed_reference_removals=["unreviewed text"])):
        try:
            reviewed_text(page, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("changed or unreviewed source span accepted")
    print("intake rights, contact filters, URL boundary, reviewed transforms and stale-span rejection checks passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review", type=Path, default=ROOT / "config/public-domain-pilot-review.json")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    output = args.output.resolve()
    if output != DEFAULT_OUTPUT.resolve():
        parser.error("this bounded pilot writes only to data/public-domain-pilot")
    if args.fetch:
        acquire(output)
    if args.clean:
        clean(output, args.review.resolve())
    if not (args.fetch or args.clean or args.self_test):
        parser.error("select --fetch, --clean or --self-test")


if __name__ == "__main__":
    main()
