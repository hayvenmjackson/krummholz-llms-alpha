"""Pack v7's new data into a training directory for train_vl.py, with the sealed-eval check.

  python pack_v7.py --data /workspace/krummholz-v7/data --out D [--math 8000] [--code 4000]

Copies the constitutional and pushback SFT and pair files, turns subsets of the open-weight math and code solutions
(OpenMathReasoning, OpenCodeReasoning; CC BY 4.0) into SFT rows, and drops any row whose user message matches a sealed
eval question (SEALED.json from fetch_public.py). Writes pack-v7-report.json. HelpSteer3 pairs wait for counsel.
"""
import argparse, hashlib, json, random, re, shutil, zlib
from pathlib import Path


def qhash(text):
    return hashlib.sha256(re.sub(r"\W+", " ", text.lower()).strip().encode()).hexdigest()


def rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if Path(p).exists() else []


def write(p, rr):
    Path(p).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rr), encoding="utf-8")


def user_text(r):
    return " ".join(m["content"] for m in r.get("messages", []) if m["role"] == "user")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--math", type=int, default=8000); ap.add_argument("--code", type=int, default=4000)
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sealed = set(h for v in json.loads((data / "public/SEALED.json").read_text()).values() for h in v) \
        if (data / "public/SEALED.json").exists() else set()
    report = {"sealed_hashes": len(sealed), "dropped_sealed": 0, "files": {}}

    def emit(name, rr):
        keep = []
        for r in rr:
            if qhash(user_text(r)) in sealed:
                report["dropped_sealed"] += 1
            else:
                keep.append(r)
        held = lambda r: zlib.crc32(r["id"].encode()) % 20 == 0
        write(out / f"{name}.train.jsonl", [r for r in keep if not held(r)])
        write(out / f"{name}.val.jsonl", [r for r in keep if held(r)])
        report["files"][name] = len(keep)

    for sub, names in (("constitutional", ["v7_constitutional", "v7_constitutional_pairs"]), ("pushback", ["v7_pushback", "v7_pushback_pairs"])):
        for n in names:
            emit(n, rows(data / sub / f"{n}.train.jsonl") + rows(data / sub / f"{n}.val.jsonl"))
    rng = random.Random(3)
    math = [r for r in rows(data / "public/openmathreasoning_50k.jsonl") if r.get("solution") and len(r["solution"]) < 5000]
    rng.shuffle(math)
    emit("v7_math", [{"id": f"omr-{i}", "source": "nvidia/OpenMathReasoning", "provenance": {"license": "CC BY 4.0", "generator": r.get("generator")},
                      "messages": [{"role": "user", "content": r["question"]}], "target": r["solution"].strip()} for i, r in enumerate(math[:a.math])])
    code = [r for r in rows(data / "public/opencodereasoning_30k.jsonl") if r.get("solution") and len(r["solution"]) < 4000]
    rng.shuffle(code)
    emit("v7_code", [{"id": f"ocr-{i}", "source": "nvidia/OpenCodeReasoning", "provenance": {"license": "CC BY 4.0"},
                      "messages": [{"role": "user", "content": r["question"]}],
                      "target": "```python\n" + r["solution"].strip().strip("`").removeprefix("python\n") + "\n```"} for i, r in enumerate(code[:a.code])])
    (out / "pack-v7-report.json").write_text(json.dumps(report, indent=1))
    print("PACK_V7", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
