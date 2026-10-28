"""Build deterministic training records for the Krummholz VL run (no model-written text).

Every record is either
  {"id", "source", "request": {"query", "sources"}, "target": <answer JSON string>}   (pilot QA protocol)
or
  {"id", "source", "messages": [...], "target": <string>}                            (plain chat)
Answers and quotes come from the data itself, so labels are correct by construction.

python tools/vl/build_local.py --out data/vl-v1 --banking <zip> --forestry <zip> --defects <zip>
       --clouds camera=<dir> lidar=<dir>
"""
import argparse, csv, io, json, random, re, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/training"))
sys.path.insert(0, str(ROOT / "tools/pointcloud"))
import convert  # noqa: E402

RNG = random.Random(20261027)


def qa(rid, source, query, sources, answer, cites, insufficient=False):
    for c in cites:
        text = next(s["text"] for s in sources if s["id"] == c["source_id"])
        assert c["quote"] in text, (rid, c["quote"])
    target = {"answer": answer, "citations": cites, "insufficient_evidence": insufficient}
    return {"id": rid, "source": source, "request": {"query": query, "sources": sources},
            "target": json.dumps(target, ensure_ascii=False)}


def split(rows, val_share=0.1):
    rows = sorted(rows, key=lambda r: r["id"])
    RNG.shuffle(rows)
    k = max(1, int(len(rows) * val_share))
    return rows[k:], rows[:k]


def maine():
    out = {}
    for part in ("train", "val"):
        rows = [json.loads(l) for l in open(ROOT / f"data/synthetic/maine-v1/{part}.jsonl", encoding="utf-8") if l.strip()]
        out[part] = [{**c, "source": "maine-v1"} for c in map(convert.convert, rows) if c]
    return out["train"], out["val"]


def banking(zpath, n):
    z = zipfile.ZipFile(zpath)
    name = next(x for x in z.namelist() if x.endswith("/train.jsonl"))
    rows = [json.loads(l) for l in z.read(name).decode("utf-8").splitlines() if l.strip()]
    RNG.shuffle(rows)
    system = ("Classify the customer's banking message. Return exactly one JSON object with keys intent "
              "(the BANKING77 intent label) and category (the broad routing category).")
    recs = [{"id": f"banking-{r['record_id']}", "source": "banking77-krummholz-v2 (CC BY 4.0)",
             "messages": [{"role": "system", "content": system}, {"role": "user", "content": r["text"]}],
             "target": json.dumps({"intent": r["intent"], "category": r["broad_category"]})} for r in rows[:n]]
    return split(recs)


def fia_sentences(r):
    sid = f"katahdin/fia-{r['record_id'][-12:]}"
    s = [f"Species: {r['species_common_name']} ({r['species_scientific_name']}), FIA code {r['species_code']}.",
         f"Status: {r['status_label']}.",
         f"Inventory year: {r['inventory_year']}."]
    s.append(f"Diameter: {r['diameter_inches']} inches at {r['diameter_point_label'].lower()}."
             if r["diameter_inches"] is not None else "Diameter: not recorded.")
    s.append(f"Total height: {r['total_height_feet']} feet, height method: {r['height_method_label'].lower()}."
             if r["total_height_feet"] is not None else "Total height: not recorded.")
    return sid, s


def fia(zpath, n):
    z = zipfile.ZipFile(zpath)
    name = next(x for x in z.namelist() if x.endswith("/trees-readable.jsonl"))
    rows = [json.loads(l) for l in z.read(name).decode("utf-8").splitlines() if l.strip()]
    RNG.shuffle(rows)
    recs = []
    for i, r in enumerate(rows[:n]):
        if not r.get("species_common_name") or not r.get("status_label"):
            continue
        sid, s = fia_sentences(r)
        src = [{"id": sid, "text": " ".join(s)}]
        kind = i % 5
        if kind == 0:
            recs.append(qa(f"fia-{i}-species", "fia-maine-v2 (CC0)", "What species is this tree?", src,
                           f"It is a {r['species_common_name']} ({r['species_scientific_name']}).", [{"source_id": sid, "quote": s[0]}]))
        elif kind == 1:
            q = "What was this tree's diameter?"
            if r["diameter_inches"] is None:
                recs.append(qa(f"fia-{i}-dia", "fia-maine-v2 (CC0)", q, src,
                               "The record does not include a diameter, so it cannot be stated from this source.", [], True))
            else:
                recs.append(qa(f"fia-{i}-dia", "fia-maine-v2 (CC0)", q, src,
                               f"{r['diameter_inches']} inches, measured at {r['diameter_point_label'].lower()}.", [{"source_id": sid, "quote": s[3]}]))
        elif kind == 2:
            q = "How tall was this tree, and how was the height obtained?"
            if r["total_height_feet"] is None:
                recs.append(qa(f"fia-{i}-ht", "fia-maine-v2 (CC0)", q, src,
                               "The record does not include a total height, so it cannot be stated from this source.", [], True))
            else:
                recs.append(qa(f"fia-{i}-ht", "fia-maine-v2 (CC0)", q, src,
                               f"{r['total_height_feet']} feet; the height method was {r['height_method_label'].lower()}.", [{"source_id": sid, "quote": s[4]}]))
        elif kind == 3:
            recs.append(qa(f"fia-{i}-status", "fia-maine-v2 (CC0)", "Was this tree alive when it was inventoried, and in what year?", src,
                           f"Recorded status: {r['status_label'].lower()}, inventory year {r['inventory_year']}.",
                           [{"source_id": sid, "quote": s[1]}, {"source_id": sid, "quote": s[2]}]))
        else:
            recs.append(qa(f"fia-{i}-age", "fia-maine-v2 (CC0)", "How old is this tree?", src,
                           "The record gives species, status, size and inventory year but no age, so the age cannot be determined from this source.", [], True))
    return split(recs)


DEFECT_WORDS = {"missing_value": "missing value", "text_in_numeric": "text in a numeric field", "out_of_range": "out-of-range value",
                "unit_mismatch_ft_vs_m": "likely metres recorded in a feet column", "inconsistent_categorical": "inconsistent category text",
                "code_name_mismatch": "species code and name disagree", "column_shift": "extra field shifts the columns"}
# Duplicates need the whole file to detect; a single row cannot show them.
ROW_INVISIBLE = {"exact_duplicate_row", "near_duplicate_key"}


def defects(zpath):
    z = zipfile.ZipFile(zpath)
    pre = z.namelist()[0].split("/")[0]
    man = json.loads(z.read(f"{pre}/intended_defects_manifest.json"))
    lines = z.read(f"{pre}/trees.csv").decode("utf-8").splitlines()
    header = lines[0]
    by_id = {}
    for d in man["defects"]:
        if d["file"] == "trees.csv" and "/T" in str(d.get("record_id", "")) and d["defect_type"] not in ROW_INVISIBLE:
            by_id.setdefault(d["record_id"], []).append(d)
    recs, clean = [], []
    for ln in lines[1:]:
        cells = ln.split(",")
        if len(cells) < 3 or cells[0] == "plot_id" or not ln.strip():
            continue
        rid = f"{cells[0]}/T{cells[1]}"
        src = [{"id": "katahdin/trees-csv-header", "text": header}, {"id": f"katahdin/row-{rid}", "text": ln}]
        q = "Check this tree record for data problems before it is loaded into the inventory."
        if rid in by_id:
            parts, cites = [], []
            for d in by_id[rid]:
                kind = DEFECT_WORDS.get(d["defect_type"], d["defect_type"].replace("_", " "))
                val = str(d.get("corrupted_value", "")).strip()
                parts.append(f"{d['field']}: {kind}" + (f" (\"{val}\")" if val and val in ln else ""))
                if val and val in ln:
                    cites.append({"source_id": f"katahdin/row-{rid}", "quote": val})
            if not cites:
                cites = [{"source_id": f"katahdin/row-{rid}", "quote": ln[:60]}]
            recs.append(qa(f"defect-{rid}-{len(recs)}", "forestry_synthetic_ME_NNE (CC0)", q, src,
                           "Problems found. " + "; ".join(parts) + ". Hold this row for correction before loading.", cites))
        elif len(clean) < 120:
            clean.append(qa(f"defect-clean-{rid}-{len(clean)}", "forestry_synthetic_ME_NNE (CC0)", q, src,
                            "No problems detected in this row's fields; it can be loaded.", [{"source_id": f"katahdin/row-{rid}", "quote": ln[:60]}]))
    return split(recs + clean)


CLOUD_Q = [
    ("count", "How many trees are in this scanned plot?"),
    ("cover", "How much of the plot does the canopy cover?"),
    ("height", "How tall is the canopy in this scan?"),
    ("understory", "How dense is the understory compared with the canopy?"),
    ("species", "Which tree species are in this plot?"),
]


def clouds(specs, n_per):
    import numpy as np
    import data as pcd
    from summarize import summarize
    recs = {"train": [], "val": []}
    for spec in specs:
        name, root = spec.split("=", 1)
        for part, cap in (("train", n_per), ("val", max(20, n_per // 10))):
            files, _ = pcd.split_files(root, part)
            if not files:
                continue
            axis = pcd.up_axis(files)
            RNG.shuffle(files)
            for i, f in enumerate(files[:cap]):
                xyz, lab = pcd.load(f)
                s = summarize(pcd.to_z_up(xyz, axis), lab, f"{name}-{f.stem}")
                sid = f"katahdin/pointcloud-{name}-{f.stem}"
                sent = s["sentences"]
                src = [{"id": sid, "text": "Tool result from krummholz.pointcloud.summarize. " + " ".join(sent)}]
                kind, q = CLOUD_Q[i % len(CLOUD_Q)]
                c = lambda k: {"source_id": sid, "quote": sent[k]}
                if kind == "count":
                    a = (f"The segmentation found {s['trunk_count']} trunk clusters, so roughly {s['trunk_count']} trees; "
                         "touching or occluded trunks can merge or be missed."); cites = [c(1), c(6)]
                elif kind == "cover":
                    a = f"Canopy covers about {round(s['canopy_cover_fraction'] * 100)} percent of the plot (fraction {s['canopy_cover_fraction']})."; cites = [c(2)]
                elif kind == "height":
                    a = f"The canopy top is about {s['canopy_top_height_p95']} units above ground (95th percentile)."; cites = [c(3)]
                elif kind == "understory":
                    sh = s["point_share"]
                    rel = "more" if sh["understory"] > sh["canopy"] else "fewer"
                    a = (f"Understory points make up {sh['understory']} of the scan versus {sh['canopy']} for canopy, "
                         f"so the scan has {rel} understory points than canopy points."); cites = [c(4)]
                else:
                    a = ("The point cloud summary reports structure only (trunks, canopy, understory and terrain), "
                         "not species, so species cannot be determined from this source."); cites = []
                recs[part].append(qa(f"pc-{name}-{f.stem}-{kind}", f"lrse synthetic forest {name} (GPL-3.0)", q, src, a,
                                     cites, kind == "species"))
    return recs["train"], recs["val"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/vl-v1"))
    ap.add_argument("--banking"); ap.add_argument("--forestry"); ap.add_argument("--defects")
    ap.add_argument("--clouds", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=1500)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jobs = {"maine": maine}
    if a.banking: jobs["banking"] = lambda: banking(a.banking, a.n)
    if a.forestry: jobs["fia"] = lambda: fia(a.forestry, a.n)
    if a.defects: jobs["defects"] = lambda: defects(a.defects)
    if a.clouds: jobs["pointcloud"] = lambda: clouds(a.clouds, a.n // len(a.clouds))
    manifest = {}
    for name, fn in jobs.items():
        train, val = fn()
        for part, rows in (("train", train), ("val", val)):
            (out / f"{name}.{part}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        manifest[name] = {"train": len(train), "val": len(val)}
        print(name, manifest[name], flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
