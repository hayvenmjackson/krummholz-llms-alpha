"""Build the nightly personal (L4) training set from a user's approved example log.

The output uses the same record formats as tools/vl/train_vl.py, so one trainer serves
Krummholz releases and dynamic weights. A fixed, seeded replay slice of Krummholz data is
mixed in so personal learning cannot quietly erode core and product behavior; 10 percent
of the user's recent examples are held out for the gate's personal-improvement check.

  python tools/dw/personal_data.py PROFILE_DIR OUT_DIR --product katahdin [--replay-ratio 0.5]
"""
from __future__ import annotations

import argparse, hashlib, json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from krummholz_llm.dynamic.store import LocalStore  # noqa: E402

REPLAY_SOURCES = {
    "acadia": ["maine", "banking"],
    "katahdin": ["maine", "fia", "defects", "pointcloud"],
    "atlantic": ["maine"],
}
MAX_EXAMPLES_PER_RUN = 2000


def records_from_log(examples):
    out = []
    for e in examples:
        base = {"id": e["id"], "source": f"personal:{e['origin']}", "messages": e["messages"]}
        if e["kind"] == "sft":
            out.append({**base, "target": e["target"]})
        else:
            out.append({**base, "chosen": e["chosen"], "rejected": e["rejected"]})
    return out


def holdout_split(records, share=0.1):
    """Most recent examples are the best test of whether learning helped; hold out the newest share."""
    k = max(1, int(len(records) * share)) if len(records) >= 10 else 0
    return (records[:-k], records[-k:]) if k else (records, [])


def replay_slice(product, n, seed):
    pool = []
    for name in REPLAY_SOURCES[product]:
        path = ROOT / f"data/vl-v1/{name}.train.jsonl"
        if path.exists():
            pool += [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    pool.sort(key=lambda r: r["id"])
    random.Random(seed).shuffle(pool)
    return pool[:n]


def build(profile_dir, out_dir, product, replay_ratio=0.5, seed=None):
    store = LocalStore(profile_dir)
    try:
        if not store.verify_chain():
            raise SystemExit("example log failed its hash chain; refusing to train on it")
        records = records_from_log(store.examples())[-MAX_EXAMPLES_PER_RUN:]
        log_sha = store.examples_sha256()
    finally:
        store.close()
    train, held = holdout_split(records)
    seed = seed if seed is not None else int(log_sha[:8], 16)  # same log -> same training set
    replay = replay_slice(product, int(len(train) * replay_ratio), seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sft = [r for r in train if "target" in r] + [r for r in replay if "chosen" not in r]
    dpo = [r for r in train if "chosen" in r]
    files = {"personal.train.jsonl": sft, "personal_dpo.train.jsonl": dpo, "personal.heldout.jsonl": held}
    for name, rows in files.items():
        (out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    manifest = {"examples_sha256": log_sha, "seed": seed, "product": product, "replay_ratio": replay_ratio,
                "counts": {k: len(v) for k, v in files.items()}, "replay": len(replay),
                "files_sha256": {k: hashlib.sha256((out / k).read_bytes()).hexdigest() for k in files}}
    (out / "build.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("profile_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--product", required=True, choices=sorted(REPLAY_SOURCES))
    ap.add_argument("--replay-ratio", type=float, default=0.5)
    a = ap.parse_args()
    print(json.dumps(build(a.profile_dir, a.out_dir, a.product, a.replay_ratio), indent=1))
